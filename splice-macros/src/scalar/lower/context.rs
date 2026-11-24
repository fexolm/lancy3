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
    free_function: bool,
    hosts: Vec<(Ident, Scalar)>,
    operations: RefCell<Vec<Operation>>,
    children: RefCell<Vec<Child>>,
}

struct Child {
    key: usize,
    target: ExprPath,
    method: bool,
    witness: TokenStream,
}

struct Operation {
    key: usize,
    expr: Expr,
    operands: Vec<Operand>,
    borrowed: Vec<SliceOperand>,
}

struct SliceOperand {
    path: ExprPath,
    element: Elem,
    writable: bool,
}

struct Operand {
    path: ExprPath,
    kind: Scalar,
}

impl Context {
    pub(in crate::scalar) fn new(argument: Ident) -> Self {
        Self {
            argument,
            free_function: false,
            hosts: Vec::new(),
            operations: RefCell::new(Vec::new()),
            children: RefCell::new(Vec::new()),
        }
    }

    pub(in crate::scalar) fn for_function(argument: Ident) -> Self {
        Self {
            free_function: true,
            ..Self::new(argument)
        }
    }

    pub(in crate::scalar) fn register_host(&mut self, name: Ident, kind: Scalar) {
        self.hosts.push((name, kind));
    }

    fn host_kind(&self, name: &Ident, lower: &Lower<'_>) -> Option<Scalar> {
        let current = lower.debug_names.get(&name.to_string())?;
        self.hosts.iter().find_map(|(parameter, kind)| {
            if parameter != current {
                return None;
            }
            // A local can shadow the parameter name and carry a different
            // Rust type. Use the binding declaration, not the use-site span.
            if proc_macro::is_available() {
                let parameter = parameter.span().unwrap();
                let current = current.span().unwrap();
                if (parameter.file(), parameter.line(), parameter.column())
                    != (current.file(), current.line(), current.column())
                {
                    return None;
                }
            }
            Some(*kind)
        })
    }

    pub(in crate::scalar) fn signature(&self, root: &Path) -> TokenStream {
        let operations = self.operations.borrow();
        let arities = operations
            .iter()
            .map(|operation| operation.operands.len() as u32);
        let borrows = operations
            .iter()
            .map(|operation| operation.borrowed.len() as u32);
        let children = self.children.borrow();
        let signatures = children.iter().map(|child| {
            let target = &child.target;
            if !child.method {
                let (item, _) = function_path(target);
                return super::calls::dispatch_free(
                    root, target,
                    emit!(&#item::__SPLICE_CONTEXT_SIGNATURE),
                    emit!(&#root::ContextSignature::new(&[])),
                );
            }
            let symbol = method_metadata(target, "context_signature");
            let name = &symbol.path.segments.last().unwrap().ident;
            emit!({
                trait Missing { #[allow(non_upper_case_globals)] const #name: &'static #root::ContextSignature =
                    &#root::ContextSignature::new(&[]); }
                impl<T: ?Sized> Missing for T {}
                #symbol
            })
        });
        emit!(#root::ContextSignature::with_borrows(&[#(#arities),*], &[#(#borrows),*], &[#(#signatures),*]))
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
                trait Missing { fn #name<Elements>(&mut self, _:u32, _:#root::ContextArguments<'_>, _: ::core::marker::PhantomData<Elements>) -> i32 {
                    ::core::unreachable!("unlinked method has no context operations")
                } }
                impl<T: ?Sized> Missing for T {}
                #dispatch(__splice_object, operation, __splice_arguments, #witness)
            },)
        });
        emit!(match (#signature).operation_at(__splice_operation) {
            ::core::option::Option::Some(#root::ContextOperation::Local(__splice_operation)) =>
                match __splice_operation { #local _ => ::core::unreachable!("verified local operation") },
            ::core::option::Option::Some(#root::ContextOperation::Child{ index, operation }) =>
                match index { #(#routes)* _ => ::core::unreachable!("verified child context") },
            ::core::option::Option::None => ::core::unreachable!("verified Rust operation"),
        })
    }

    fn arms(&self, root: &Path, types: &[&Ident]) -> TokenStream {
        let word = emit!(#root::frontend::scalar::ImportWord);
        self.operations
            .borrow()
            .iter()
            .enumerate()
            .map(|(ordinal, operation)| {
                let ordinal = ordinal as u32;
                let operands = operation
                    .operands
                    .iter()
                    .enumerate()
                    .map(|(index, operand)| {
                        let slot = emit!(__splice_arguments.word(#index));
                        let name = operand.path.path.get_ident().expect("a captured binding");
                        // A native slice length is nonnegative and fits one word.
                        match operand.kind {
                            Scalar::Bool => emit!(let #name: bool = #word::decode(#slot);),
                            Scalar::Length => emit!(let #name: usize = #slot as usize;),
                            _ => emit!(let #name: i32 = #slot;),
                        }
                    });
                let views = (!operation.borrowed.is_empty())
                    .then(|| emit!(let mut __splice_arguments = __splice_arguments;));
                let borrowed = operation.borrowed.iter().map(|slice| {
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
                let expr = &operation.expr;
                emit!(#ordinal => { #(#operands)* #views #(#borrowed)* #word::encode(#expr) },)
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
                let value = lower.path(&operand.path);
                lower.reg(value, operand.kind, &operand.path)
            })
            .collect::<Result<Vec<_>>>()?;
        let slices = captures
            .borrowed
            .iter()
            .map(|slice| {
                let buffer = lower.path(&slice.path).code;
                let writable = slice.writable;
                emit!(#root::hir::ContextSlice { buffer: #buffer, writable: #writable })
            })
            .collect::<Vec<_>>();
        let mut operations = self.operations.borrow_mut();
        let key = crate::scalar::flow::key(expr);
        let ordinal =
            if let Some(ordinal) = operations.iter().position(|operation| operation.key == key) {
                ordinal as u32
            } else {
                let ordinal = operations.len() as u32;
                let mut expr = expr.clone();
                ReplaceObject {
                    lower,
                    name: &self.argument,
                }
                .visit_expr_mut(&mut expr);
                operations.push(Operation {
                    key,
                    expr,
                    operands: captures.operands,
                    borrowed: captures.borrowed,
                });
                ordinal
            };
        let buffer = &self.argument;
        if slices.is_empty() {
            return Ok(emit!({
                let arguments = [#(#operands),*];
                __splice_builder.rust_context_call(#buffer, __splice_context_frame.operation(#ordinal)?, &arguments, &[])?
            }));
        }
        Ok(emit!({
            let arguments = [#(#operands),*];
            __splice_builder.rust_context_call(#buffer, __splice_context_frame.operation(#ordinal)?, &arguments, &[#(#slices),*])?
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
            let child = self.child(call, target, true, &witness);
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
                fn #name<Elements, #(#params),*>(_: #root::hir::Buffer, #(_: #params,)*
                    _: ::core::marker::PhantomData<Elements>, _: #root::ContextFrame, _: &mut dyn #root::frontend::KernelBuilder) -> #root::Result<#root::hir::Reg> {
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
                    let child = self.child(call, target, false, &witness);
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
        call: &syn::ExprCall,
        target: &ExprPath,
        method: bool,
        witness: &TokenStream,
    ) -> u32 {
        let mut children = self.children.borrow_mut();
        let key = crate::scalar::flow::key(call);
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
                if matches!(
                    kind,
                    Scalar::Bool | Scalar::Int | Scalar::Inferred | Scalar::Length
                ) =>
            {
                self.operands.push(Operand {
                    path: path.clone(),
                    kind,
                });
            }
            Some(Binding::Buffer(buffer)) => {
                self.borrow(path, buffer.elem, buffer.write);
            }
            Some(Binding::Host(_)) => {
                if let Some(kind) = self.context.host_kind(name, self.lower) {
                    self.operands.push(Operand {
                        path: path.clone(),
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

struct ReplaceObject<'a, 'ast> {
    lower: &'a Lower<'ast>,
    name: &'a Ident,
}

impl VisitMut for ReplaceObject<'_, '_> {
    fn visit_expr_path_mut(&mut self, path: &mut ExprPath) {
        if is_object(path, self.lower, self.name) {
            *path = syn::parse2(emit!(__splice_object)).expect("a generated object binding");
        } else {
            visit_mut::visit_expr_path_mut(self, path);
        }
    }
}
