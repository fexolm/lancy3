//! Lower one Rust operation on a borrowed runtime object. Scalar values
//! already computed in the surrounding IR cross the boundary as words.
use super::{Binding, Elem, Lower, Scalar, Stage, Value};
use proc_macro2::{Ident, TokenStream};
use std::cell::RefCell;
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};
use syn::{Expr, ExprPath, Path, Result, UnOp};

pub(in crate::scalar) struct Context {
    argument: Ident,
    /// The Rust type of the object, with its lifetimes elided.
    object: syn::Type,
    free_function: bool,
    hosts: Vec<(Ident, Scalar)>,
    operations: RefCell<Vec<Operation>>,
    children: RefCell<Vec<Child>>,
}

struct Child {
    key: crate::kast::NodeId,
    target: ExprPath,
    method: bool,
    witness: TokenStream,
}

struct Operation {
    key: crate::kast::NodeId,
    expr: Expr,
    operands: Vec<Operand>,
    borrowed: Vec<SliceOperand>,
    widths: Vec<Path>,
    imports: Vec<Vec<syn::ItemUse>>,
}

/// A captured slice or scalar: a path of the body and what it is bound to
/// where the operation reads it.
struct SliceOperand {
    path: ExprPath,
    binding: Option<Binding>,
    element: Elem,
    writable: bool,
}

struct Operand {
    path: ExprPath,
    binding: Option<Binding>,
    kind: Scalar,
}

impl Operand {
    /// The Rust type of the word: a runtime scalar of unknown type is read
    /// as an `i32`, and staging checks that it is one.
    fn ty(&self) -> Ident {
        self.kind
            .rust_type()
            .unwrap_or_else(|| Ident::new("i32", proc_macro2::Span::call_site()))
    }
}

impl Operation {
    /// The operation on `__splice_object` and `__splice_arguments`.
    fn body(&self, root: &Path, types: &[&Ident]) -> TokenStream {
        let word = emit!(#root::frontend::scalar::ImportWord);
        let operands = self.operands.iter().enumerate().map(|(index, operand)| {
            let name = operand.path.path.get_ident().expect("a captured binding");
            let ty = operand.ty();
            emit!(let #name: #ty = #word::decode(__splice_arguments.word(#index));)
        });
        let views = (!self.borrowed.is_empty())
            .then(|| emit!(let mut __splice_arguments = __splice_arguments;));
        let borrowed = self.borrowed.iter().map(|slice| {
            let name = slice
                .path
                .path
                .get_ident()
                .expect("a captured slice binding");
            let element = slice.element.ty(root, types);
            if slice.writable {
                emit!(let #name = __splice_arguments.write::<#element>();)
            } else {
                emit!(let #name = __splice_arguments.read::<#element>();)
            }
        });
        let expr = &self.expr;
        let width_slots = (!self.widths.is_empty()).then(|| {
            let first = self.operands.len();
            let slots = self.widths.iter().map(|owner| {
                let element = super::simd::width_element(root, owner);
                emit!({ let slot = next; next += usize::from((#element).is_some()); slot })
            });
            emit!(let mut next = #first; let __splice_width_slots = [#(#slots),*];)
        });
        let mut code = emit!({ #(#operands)* #views #(#borrowed)* #width_slots #expr });
        for imports in self.imports.iter().rev().filter(|items| !items.is_empty()) {
            code = emit!({ #(#[allow(unused_imports)] #imports)* #code });
        }
        code
    }
}

impl Context {
    pub(in crate::scalar) fn new(argument: Ident, object: &syn::Type) -> Self {
        struct Elide;
        impl VisitMut for Elide {
            fn visit_lifetime_mut(&mut self, lifetime: &mut syn::Lifetime) {
                lifetime.ident = Ident::new("_", lifetime.ident.span());
            }
        }
        let mut object = object.clone();
        Elide.visit_type_mut(&mut object);
        Self {
            argument,
            object,
            free_function: false,
            hosts: Vec::new(),
            operations: RefCell::new(Vec::new()),
            children: RefCell::new(Vec::new()),
        }
    }

    pub(in crate::scalar) fn for_function(argument: Ident, object: &syn::Type) -> Self {
        Self {
            free_function: true,
            ..Self::new(argument, object)
        }
    }

    pub(in crate::scalar) fn register_host(&mut self, name: Ident, kind: Scalar) {
        self.hosts.push((name, kind));
    }

    /// The scalar kind of the comptime parameter `path` names. A local
    /// that shadows the parameter can carry another Rust type.
    fn host_kind(&self, path: &ExprPath, lower: &Lower<'_>) -> Option<Scalar> {
        let parameter = lower.body.parameter(path)?;
        self.hosts
            .iter()
            .find_map(|(name, kind)| (name == parameter).then_some(*kind))
    }

    pub(in crate::scalar) fn signature(&self, root: &Path) -> TokenStream {
        let operations = self.operations.borrow();
        let arities = operations.iter().flat_map(|operation| {
            (0..=operation.widths.len()).map(|extra| (operation.operands.len() + extra) as u32)
        });
        let borrows = operations
            .iter()
            .flat_map(|op| std::iter::repeat_n(op.borrowed.len() as u32, op.widths.len() + 1));
        let children = self.children.borrow();
        let signatures = children.iter().map(|child| {
            let target = &child.target;
            if !child.method {
                let (item, _) = function_path(target);
                return super::calls::dispatch_free(
                    root, target,
                    emit!(&#item::__SPLICE_CONTEXT_SIGNATURE),
                    emit!(&#root::__private::ContextSignature::new(&[])),
                );
            }
            let symbol = method_metadata(target, "context_signature");
            let name = &symbol.path.segments.last().unwrap().ident;
            emit!({
                trait Missing { #[allow(non_upper_case_globals)] const #name: &'static #root::__private::ContextSignature =
                    &#root::__private::ContextSignature::new(&[]); }
                impl<T: ?Sized> Missing for T {}
                #symbol
            })
        });
        emit!(#root::__private::ContextSignature::with_borrows(&[#(#arities),*], &[#(#borrows),*], &[#(#signatures),*]))
    }

    pub(in crate::scalar) fn dispatcher(
        &self,
        root: &Path,
        signature: TokenStream,
        types: &[&Ident],
    ) -> TokenStream {
        let local = self.arms(root, types);
        let children = self.children.borrow();
        if children.is_empty() {
            return emit!(match __splice_operation { #local _ => ::core::unreachable!("verified Rust operation") });
        }
        let routes = children.iter().enumerate().map(|(index, child)| {
            let index = index as u32;
            let target = &child.target;
            let witness = &child.witness;
            if !child.method {
                let (item, generics) = function_path(target);
                let dispatch = super::calls::dispatch_free(
                    root, target,
                    emit!(#item::__splice_dispatch_context #generics (__splice_object, operation, __splice_arguments, #witness)),
                    emit!(::core::unreachable!("an ordinary Rust function has no child context")),
                );
                return emit!(#index => #dispatch,);
            }
            let dispatch = crate::scalar::methods::rename_path(target, "dispatch_context");
            let name = &dispatch.path.segments.last().unwrap().ident;
            emit!(#index => {
                trait Missing { fn #name<Elements>(&mut self, _:u32, _:#root::__private::ContextArguments<'_>, _: ::core::marker::PhantomData<Elements>) -> u64 {
                    ::core::unreachable!("unlinked method has no context operations")
                } }
                impl<T: ?Sized> Missing for T {}
                #dispatch(__splice_object, operation, __splice_arguments, #witness)
            },)
        });
        emit!(match (#signature).operation_at(__splice_operation) {
            ::core::option::Option::Some(#root::__private::ContextOperation::Local(__splice_operation)) =>
                match __splice_operation { #local _ => ::core::unreachable!("verified local operation") },
            ::core::option::Option::Some(#root::__private::ContextOperation::Child{ index, operation }) =>
                match index { #(#routes)* _ => ::core::unreachable!("verified child context") },
            ::core::option::Option::None => ::core::unreachable!("verified Rust operation"),
        })
    }

    fn arms(&self, root: &Path, types: &[&Ident]) -> TokenStream {
        let word = emit!(#root::frontend::scalar::ImportWord);
        self.operations
            .borrow()
            .iter()
            .scan(0u32, |next, operation| {
                let first = *next;
                let last = first + operation.widths.len() as u32;
                *next = last + 1;
                let body = operation.body(root, types);
                Some(emit!(#first..=#last => #word::encode(#body),))
            })
            .collect()
    }

    pub(super) fn lower(
        &self,
        lower: &Lower<'_>,
        expr: &Expr,
        hint: Option<Scalar>,
    ) -> Result<Option<Value>> {
        let owns = |expr: &Expr| self.owns(lower, expr);
        let mut rust_callee = None;
        let unit = match expr {
            Expr::Assign(assign) if owns(&assign.left) => true,
            Expr::Binary(binary)
                if crate::scalar::compound_op(&binary.op).is_some() && owns(&binary.left) =>
            {
                true
            }
            Expr::Field(field) if owns(&field.base) => false,
            Expr::Index(index) if owns(&index.expr) => false,
            Expr::Unary(unary) if matches!(unary.op, UnOp::Deref(_)) && owns(&unary.expr) => false,
            Expr::MethodCall(call) if owns(&call.receiver) => false,
            Expr::Call(call) => {
                let Expr::Path(target) = &*call.func else {
                    return Ok(None);
                };
                if crate::scalar::methods::method_path(target)
                    && call
                        .args
                        .first()
                        .is_some_and(|arg| direct_object(arg, lower, &self.argument))
                {
                    return self
                        .forward_method(lower, expr, target, call, hint)
                        .map(Some);
                }
                if crate::scalar::methods::method_path(target)
                    || lower.binding(target).is_some()
                    || !call.args.iter().any(owns)
                {
                    return Ok(None);
                }
                let mut forwarded = call
                    .args
                    .iter()
                    .enumerate()
                    .filter(|(_, arg)| direct_object(arg, lower, &self.argument));
                if let Some((index, _)) = forwarded.next()
                    && forwarded.next().is_none()
                {
                    return self
                        .forward_function(lower, expr, target, call, index, hint)
                        .map(Some);
                }
                rust_callee = Some(target);
                false
            }
            _ => return Ok(None),
        };
        let output = hint.unwrap_or(Scalar::Inferred);
        let root = lower.root;
        let mut code = self.opaque_operation(lower, expr)?;
        if let Some(target) = rust_callee {
            let unsupported = emit!({
                ::core::result::Result::Err::<#root::hir::Reg, #root::BuildError>(#root::BuildError::staging("a nested kernel cannot take this runtime object"))?
            });
            code = lower.dispatch_call(target, unsupported, code);
        }
        Ok(Some(if unit {
            Value::unit(emit!({ #code; }))
        } else {
            Value::new(Stage::Reg(output), code)
        }))
    }

    fn opaque_operation(&self, lower: &Lower<'_>, expr: &Expr) -> Result<TokenStream> {
        let root = lower.root;
        let mut captures = Captures {
            lower,
            context: self,
            operands: Vec::new(),
            borrowed: Vec::new(),
            rejected: false,
        };
        captures.visit_expr(expr);
        if captures.rejected {
            return Ok(
                emit!(::core::result::Result::Err::<#root::hir::Reg,#root::BuildError>(
                #root::BuildError::staging("a Rust object operation requires materializable operands and cannot transfer control"))?),
            );
        }
        let operands = captures
            .operands
            .iter()
            .map(|operand| {
                let value = lower.bound_path(&operand.path, operand.binding);
                let reg = lower.reg(value, operand.kind, &operand.path)?;
                let ty = lower
                    .scalar_type(operand.kind)
                    .unwrap_or_else(|| emit!(#root::hir::ScalarType::I32));
                Ok(emit!({
                    let __splice_operand = #reg;
                    #root::frontend::scalar::context_operand(__splice_operand, #ty, __splice_builder)?
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        let slices = captures
            .borrowed
            .iter()
            .map(|slice| {
                let buffer = lower.bound_path(&slice.path, slice.binding).code;
                let writable = slice.writable;
                emit!(#root::hir::ContextSlice { slice: #buffer, writable: #writable })
            })
            .collect::<Vec<_>>();
        // A width is pure and target-dependent. It can cross the object ABI as
        // an IR word, even inside a short circuit or between effectful calls.
        let mut nested = NestedIntrinsics {
            lower,
            widths: Vec::new(),
            guarded: Vec::new(),
            paths: Vec::new(),
        };
        nested.visit_expr(expr);
        let mut operations = self.operations.borrow_mut();
        let key = lower.body.expr(expr);
        let index = if let Some(index) = operations.iter().position(|op| op.key == key) {
            index
        } else {
            let mut objects = FindObjects {
                lower,
                name: &self.argument,
                found: Vec::new(),
            };
            objects.visit_expr(expr);
            let mut expr = expr.clone();
            ReplaceObject(objects.found.into_iter()).visit_expr_mut(&mut expr);
            if !nested.widths.is_empty() {
                ReplaceWidths {
                    paths: nested.paths.iter().copied(),
                    slot: 0,
                    root,
                }
                .visit_expr_mut(&mut expr);
            }
            operations.push(Operation {
                key,
                expr,
                operands: captures.operands,
                borrowed: captures.borrowed,
                widths: nested
                    .widths
                    .iter()
                    .map(|(_, owner)| owner.clone())
                    .collect(),
                imports: lower.imports.clone(),
            });
            operations.len() - 1
        };
        // The signature is static even when an alias resolves to an ordinary
        // Rust type. One layout per possible word count shares a callback.
        let base: u32 = operations[..index]
            .iter()
            .map(|op| op.widths.len() as u32 + 1)
            .sum();
        let packet = if nested.widths.is_empty() {
            emit!(let arguments = [#(#operands),*]; let ordinal = #base;)
        } else {
            let widths = nested.widths.iter().map(|(name, owner)| {
                let probe = super::simd::width_element(root, owner);
                let reg = super::simd::width(root, name, emit!(element));
                emit!(if let Some(element) = #probe { arguments.push(#reg); ordinal += 1; })
            });
            emit!(let mut arguments = ::std::vec![#(#operands),*]; let mut ordinal = #base; #(#widths)*)
        };
        let body = operations[index].body(root, lower.types);
        let object = &self.object;
        let buffer = &self.argument;
        let guarded = nested.guarded;
        Ok(emit!({
            #(if #guarded { return Err(#root::BuildError::staging("nested intrinsic in an opaque Rust object expression is unsupported; bind its result before the object operation")); })*
            #packet
            // The closure only names the operation's Rust result type.
            #[allow(unreachable_code, unused_variables, unused_mut, clippy::diverging_sub_expression)]
            let __splice_operation = || {
                let __splice_object: #object = ::core::unreachable!();
                let __splice_arguments: #root::__private::ContextArguments<'_> = ::core::unreachable!();
                #body
            };
            let __splice_result = #root::frontend::scalar::result_type(__splice_operation);
            __splice_builder.rust_call(#root::hir::RustTarget::Context { context: #buffer, operation: __splice_context_frame.operation(ordinal)? }, &arguments, &[#(#slices),*], __splice_result)?
        }))
    }

    fn forward_method(
        &self,
        lower: &Lower<'_>,
        expr: &Expr,
        target: &ExprPath,
        call: &syn::ExprCall,
        hint: Option<Scalar>,
    ) -> Result<Value> {
        let root = lower.root;
        let buffer = &self.argument;
        let mut arguments = Vec::new();
        let mut elements = Vec::new();
        for expr in call.args.iter().skip(1) {
            arguments.push(context_argument_code(lower, expr, &mut elements)?);
        }
        let witness = emit!(::core::marker::PhantomData::<(#(#elements,)*)>);
        let recursive = target
            .path
            .segments
            .first()
            .is_some_and(|p| p.ident == "Self")
            && target
                .path
                .segments
                .last()
                .is_some_and(|p| &p.ident == lower.name);
        let frame = if recursive {
            emit!(__splice_context_frame)
        } else {
            let child = self.child(lower, expr, target, true, &witness);
            emit!(__splice_context_frame.child(#child)?)
        };
        let fallback = if recursive {
            emit!(::core::unreachable!("the current method is published"))
        } else {
            self.opaque_operation(lower, expr)?
        };
        let published = method_metadata(target, "has_context");
        let marker = &published.path.segments.last().unwrap().ident;
        let stage = crate::scalar::methods::rename_path(target, "stage_context");
        let name = &stage.path.segments.last().unwrap().ident;
        let params = (0..arguments.len())
            .map(|i| quote::format_ident!("Arg{i}"))
            .collect::<Vec<_>>();
        let code = emit!({
            trait Missing {
                fn #marker() -> bool {false}
                fn #name<Elements, #(#params),*>(_: #root::hir::Slice, #(_: #params,)*
                    _: ::core::marker::PhantomData<Elements>, _: #root::__private::ContextFrame, _: &mut dyn #root::frontend::KernelBuilder) -> #root::Result<#root::hir::Reg> {
                    ::core::result::Result::Err(#root::BuildError::staging("Rust method cannot be linked into the scalar kernel"))
                }
            }
            impl<T: ?Sized> Missing for T {}
            if #published() { #stage(#buffer, #(#arguments,)* #witness, #frame, __splice_builder)? } else { #fallback }
        });
        Ok(Value::new(
            Stage::Reg(hint.unwrap_or(Scalar::Inferred)),
            code,
        ))
    }

    fn forward_function(
        &self,
        lower: &Lower<'_>,
        expr: &Expr,
        target: &ExprPath,
        call: &syn::ExprCall,
        object_index: usize,
        hint: Option<Scalar>,
    ) -> Result<Value> {
        let buffer = &self.argument;
        let root = lower.root;
        let recursive = self.free_function && target.path.is_ident(lower.name);
        let (item, generics) = function_path(target);
        let mut elements = Vec::new();
        let arguments = call
            .args
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                if index == object_index {
                    return Ok(emit!(#buffer));
                }
                context_argument_code(lower, expr, &mut elements)
            })
            .collect::<Result<Vec<_>>>();
        let staged = match arguments {
            Ok(arguments) => {
                let witness = emit!(::core::marker::PhantomData::<(#(#elements,)*)>);
                let frame = if recursive {
                    emit!(__splice_context_frame)
                } else {
                    let child = self.child(lower, expr, target, false, &witness);
                    emit!(__splice_context_frame.child(#child)?)
                };
                emit!(#item::__splice_stage_context #generics (#(#arguments,)* #witness, #frame, __splice_builder)?)
            }
            Err(error) => {
                let error = error.to_string();
                emit!(::core::result::Result::Err::<#root::hir::Reg, #root::BuildError>(#root::BuildError::staging(#error))?)
            }
        };
        let code = if recursive {
            staged
        } else {
            let rust = self.opaque_operation(lower, expr)?;
            lower.dispatch_call(target, staged, rust)
        };
        Ok(Value::new(
            Stage::Reg(hint.unwrap_or(Scalar::Inferred)),
            code,
        ))
    }

    fn child(
        &self,
        lower: &Lower<'_>,
        expr: &Expr,
        target: &ExprPath,
        method: bool,
        witness: &TokenStream,
    ) -> u32 {
        let mut children = self.children.borrow_mut();
        let key = lower.body.expr(expr);
        children
            .iter()
            .position(|child| child.key == key)
            .unwrap_or_else(|| {
                children.push(Child {
                    key,
                    target: target.clone(),
                    method,
                    witness: witness.clone(),
                });
                children.len() - 1
            }) as u32
    }
    fn owns(&self, lower: &Lower<'_>, expr: &Expr) -> bool {
        object_place(expr, lower, &self.argument)
    }
}

fn context_argument_code(
    lower: &Lower<'_>,
    expr: &Expr,
    elements: &mut Vec<TokenStream>,
) -> Result<TokenStream> {
    let value = lower.expr(expr)?;
    let root = lower.root;
    if let Stage::Buffer(kind) = value.stage {
        // The register ABI erases slice types. Carry their element types
        // separately so Rust can infer the child's generic arguments.
        elements.push(kind.elem.ty(root, lower.types));
    }
    Ok(if value.stage.is_host() {
        emit!(#root::frontend::scalar::Host(#value))
    } else {
        value.code
    })
}

fn function_path(target: &ExprPath) -> (Path, syn::PathArguments) {
    let mut item = target.path.clone();
    let generics = std::mem::take(&mut item.segments.last_mut().unwrap().arguments);
    (item, generics)
}

fn method_metadata(target: &ExprPath, kind: &str) -> ExprPath {
    let mut target = crate::scalar::methods::rename_path(target, kind);
    target.path.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
    target
}

fn direct_object(expr: &Expr, lower: &Lower<'_>, object: &Ident) -> bool {
    context_argument(expr).is_some_and(|path| is_object(path, lower, object))
}

fn context_argument(expr: &Expr) -> Option<&ExprPath> {
    match ungroup(expr) {
        Expr::Path(path) => Some(path),
        _ => reborrow_argument(expr, true),
    }
}

fn reborrow_argument(expr: &Expr, mutable: bool) -> Option<&ExprPath> {
    let Expr::Reference(reference) = ungroup(expr) else {
        return None;
    };
    if reference.mutability.is_some() != mutable {
        return None;
    }
    let Expr::Unary(deref) = ungroup(&reference.expr) else {
        return None;
    };
    match ungroup(&deref.expr) {
        Expr::Path(path) if matches!(deref.op, UnOp::Deref(_)) => Some(path),
        _ => None,
    }
}

fn ungroup(mut expr: &Expr) -> &Expr {
    loop {
        expr = match expr {
            Expr::Paren(value) => &value.expr,
            Expr::Group(value) => &value.expr,
            _ => return expr,
        };
    }
}

fn object_place(mut expr: &Expr, lower: &Lower<'_>, object: &Ident) -> bool {
    loop {
        match expr {
            Expr::Path(path) => return is_object(path, lower, object),
            Expr::Field(field) => expr = &field.base,
            Expr::Index(index) => expr = &index.expr,
            Expr::Unary(unary) if matches!(unary.op, UnOp::Deref(_)) => expr = &unary.expr,
            Expr::Reference(reference) => expr = &reference.expr,
            Expr::Paren(value) => expr = &value.expr,
            Expr::Group(value) => expr = &value.expr,
            _ => return false,
        }
    }
}

fn writable_object_place(mut expr: &Expr, lower: &Lower<'_>, object: &Ident) -> bool {
    // Replacing the original reference would change which object later IR
    // operations address. A callback can mutate only through that reference.
    loop {
        match expr {
            Expr::Path(_) => return false,
            Expr::Paren(value) => expr = &value.expr,
            Expr::Group(value) => expr = &value.expr,
            _ => return object_place(expr, lower, object),
        }
    }
}

fn is_object(path: &ExprPath, lower: &Lower<'_>, object: &Ident) -> bool {
    path.path.is_ident(object) && matches!(lower.binding(path), Some(Binding::Context))
}

/// Name resolution is shared with scalar lowering. The Rust callback sees a
/// decoded word in place of each width, and cannot execute an intrinsic body.
struct NestedIntrinsics<'a, 'ast> {
    lower: &'a Lower<'ast>,
    widths: Vec<(String, Path)>,
    guarded: Vec<TokenStream>,
    paths: Vec<bool>,
}

impl NestedIntrinsics<'_, '_> {
    fn path(&mut self, path: &ExprPath, zero_arg_call: bool) -> Option<(String, Path)> {
        if self.lower.local_owner(path) {
            return None;
        }
        let root = self.lower.root;
        let mut width = None;
        let member = &path.path.segments.last().unwrap().ident;
        let name = member.to_string();
        if !crate::scalar::methods::method_path(path)
            && path.path.segments.len() > 1
            && super::simd::registered(&name)
        {
            // A lowercase prefix may denote a module or a type alias.
            // Only a module's free function can be imported. Require
            // that proof before allowing the expression into Rust.
            // An associated call on a lowercase alias uses <alias>::m.
            let mut function = path.path.clone();
            function.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
            self.guarded.push(emit!({
                #[allow(unused_imports)] use #function as __splice_opaque_free_function;
                false
            }));
        }
        if super::simd::simd_path(path) {
            let owner = super::simd::owner_path(path).unwrap();
            let nominal = super::simd::type_probe(root, &owner);
            if (member == "element_width" || member == "byte_width") && zero_arg_call {
                width = Some((member.to_string(), owner));
            } else {
                self.guarded.push(super::calls::select_builtin(
                    root,
                    &owner,
                    "simd",
                    emit!(true),
                    emit!(#nominal.is_some()),
                ));
            }
        } else if super::memory::math_op(&name).is_some()
            && super::memory::integer_owner(path).is_some()
        {
            self.guarded.push(emit!(true));
        }
        width
    }
}
impl Visit<'_> for NestedIntrinsics<'_, '_> {
    fn visit_expr_path(&mut self, path: &ExprPath) {
        // A function item passed to a callback must not expose the Rust stub.
        self.path(path, false);
    }

    fn visit_expr_call(&mut self, call: &syn::ExprCall) {
        let width = match &*call.func {
            Expr::Path(path) => self.path(path, call.args.is_empty()),
            _ => None,
        };
        self.paths.push(width.is_some());
        if let Some(width) = width {
            self.widths.push(width);
        } else {
            if !matches!(&*call.func, Expr::Path(_)) {
                self.visit_expr(&call.func);
            }
            for arg in &call.args {
                self.visit_expr(arg);
            }
        }
    }
    fn visit_expr_method_call(&mut self, call: &syn::ExprMethodCall) {
        if super::memory::math_op(&call.method.to_string()).is_some()
            && (matches!(&*call.receiver, Expr::Path(p) if matches!(self.lower.binding(p), Some(Binding::Reg(t, _) | Binding::Word(t)) if t.is_number()))
                || matches!(&*call.receiver, Expr::Lit(_)))
        {
            self.guarded.push(emit!(true));
        }
        visit::visit_expr_method_call(self, call);
    }
}

struct ReplaceWidths<'a> {
    paths: std::iter::Copied<std::slice::Iter<'a, bool>>,
    slot: usize,
    root: &'a Path,
}
impl VisitMut for ReplaceWidths<'_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Call(call) = expr
            && self
                .paths
                .next()
                .expect("a call of the original expression")
        {
            let Expr::Path(path) = &*call.func else {
                unreachable!("a width has a type owner")
            };
            let owner = super::simd::owner_path(path).expect("a registered width");
            let slot = self.slot;
            let root = self.root;
            self.slot += 1;
            *expr = syn::parse2(emit!({
                use #root::__private::ContextWidth as _;
                (&::core::marker::PhantomData::<#owner>).context_width(
                    || <usize as #root::frontend::scalar::ImportWord>::decode(__splice_arguments.word(__splice_width_slots[#slot])),
                    || #call,
                )
            })).expect("context width expression");
        } else {
            visit_mut::visit_expr_mut(self, expr);
        }
    }
}

struct Captures<'a, 'ast> {
    lower: &'a Lower<'ast>,
    context: &'a Context,
    operands: Vec<Operand>,
    borrowed: Vec<SliceOperand>,
    rejected: bool,
}

impl Captures<'_, '_> {
    fn borrow(&mut self, path: &ExprPath, element: Elem, writable: bool) {
        if let Some(slice) = self
            .borrowed
            .iter_mut()
            .find(|slice| slice.path.path.get_ident() == path.path.get_ident())
        {
            slice.writable |= writable;
        } else {
            self.borrowed.push(SliceOperand {
                path: path.clone(),
                binding: self.lower.binding(path),
                element,
                writable,
            });
        }
    }
}

impl Visit<'_> for Captures<'_, '_> {
    fn visit_expr_path(&mut self, path: &ExprPath) {
        let Some(name) = path.path.get_ident() else {
            return;
        };
        if is_object(path, self.lower, &self.context.argument)
            || self
                .operands
                .iter()
                .any(|operand| operand.path.path.is_ident(name))
        {
            return;
        }
        match self.lower.binding(path) {
            Some(Binding::Reg(kind, _) | Binding::Word(kind))
                if kind.is_scalar() || kind == Scalar::Inferred =>
            {
                self.operands.push(Operand {
                    path: path.clone(),
                    binding: self.lower.binding(path),
                    kind,
                });
            }
            Some(Binding::Buffer(buffer)) => {
                self.borrow(path, buffer.elem, buffer.write);
            }
            Some(Binding::Host(_)) => {
                if let Some(kind) = self.context.host_kind(path, self.lower) {
                    self.operands.push(Operand {
                        path: path.clone(),
                        binding: self.lower.binding(path),
                        kind,
                    });
                } else {
                    self.rejected = true;
                }
            }
            Some(_) => self.rejected = true,
            None => (),
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        if let Some(path) = reborrow_argument(expr, false)
            && let Some(Binding::Buffer(buffer)) = self.lower.binding(path)
        {
            self.borrow(path, buffer.elem, false);
            return;
        }
        let object = &self.context.argument;
        self.rejected |= match expr {
            Expr::Assign(assign) => !writable_object_place(&assign.left, self.lower, object),
            Expr::Binary(binary) if crate::scalar::compound_op(&binary.op).is_some() => {
                !writable_object_place(&binary.left, self.lower, object)
            }
            _ => false,
        };
        self.rejected |= matches!(
            expr,
            Expr::Async(_)
                | Expr::Block(_)
                | Expr::Break(_)
                | Expr::Closure(_)
                | Expr::Continue(_)
                | Expr::ForLoop(_)
                | Expr::If(_)
                | Expr::Let(_)
                | Expr::Loop(_)
                | Expr::Macro(_)
                | Expr::Match(_)
                | Expr::Return(_)
                | Expr::Try(_)
                | Expr::While(_)
        );
        visit::visit_expr(self, expr);
    }
}

/// Whether each path of an expression, in walk order, names the context
/// object.
struct FindObjects<'a, 'ast> {
    lower: &'a Lower<'ast>,
    name: &'a Ident,
    found: Vec<bool>,
}

impl Visit<'_> for FindObjects<'_, '_> {
    fn visit_expr_path(&mut self, path: &ExprPath) {
        let object = is_object(path, self.lower, self.name);
        self.found.push(object);
        if !object {
            visit::visit_expr_path(self, path);
        }
    }
}

/// Replace the paths [`FindObjects`] found in a copy of the expression.
struct ReplaceObject(std::vec::IntoIter<bool>);

impl VisitMut for ReplaceObject {
    fn visit_expr_path_mut(&mut self, path: &mut ExprPath) {
        if self.0.next().expect("a path of the original expression") {
            *path = syn::parse2(emit!(__splice_object)).expect("a generated object binding");
        } else {
            visit_mut::visit_expr_path_mut(self, path);
        }
    }
}
