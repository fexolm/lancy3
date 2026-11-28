use super::{Lower, Scalar, Stage, Value, err};
use crate::kast::types::Primitive;
use proc_macro2::{Ident, Span, TokenStream};
use quote::ToTokens;
mod table;
use syn::{Expr, GenericArgument, Path, PathArguments, Result, Type, visit::Visit};
use table::{Output, Owner, Param, Signature, lookup};

/// The element of a slice, vector or mask: a named primitive, a generic
/// parameter of the kernel, or the lane index type of a generic parameter.
/// rustc checked that a named element is a `BufferElement` (or a
/// `LaneElement` for a vector); the IR type is `<T as KernelScalar>::TYPE`.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Elem {
    Named(Primitive),
    Param(usize),
    Index(usize),
    /// Kernel data other than a scalar the macro can name: a struct that
    /// derives `KernelType` (or its `transparent` scalar). Only the builder
    /// knows its layout.
    Record,
}
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct BufferType {
    pub elem: Elem,
    pub write: bool,
    /// A `&str`: read-only `u8`s that hold UTF-8.
    pub utf8: bool,
    /// A shared slice whose elements are shared slices of `elem` (`&str`s
    /// when [`SliceOf::Strs`]): `&[&[T]]` or `&[&str]`.
    pub slice_of: Option<SliceOf>,
}
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum SliceOf {
    Slices,
    Strs,
}
impl BufferType {
    /// The type of one element of a slice of slices.
    pub(crate) fn inner(self) -> Option<Self> {
        let slice_of = self.slice_of?;
        Some(Self {
            elem: self.elem,
            write: false,
            utf8: slice_of == SliceOf::Strs,
            slice_of: None,
        })
    }
}
impl Elem {
    pub(crate) const I32: Self = Self::Named(Primitive::I32);
    fn of(primitive: Primitive) -> Option<Self> {
        (primitive.is_kernel_scalar() && primitive != Primitive::Bool)
            .then_some(Self::Named(primitive))
    }
    /// The primitive type of a named element.
    pub(in crate::scalar) fn primitive(self) -> Option<Primitive> {
        match self {
            Self::Named(p) => Some(p),
            Self::Param(_) | Self::Index(_) | Self::Record => None,
        }
    }
    /// The register an element of a slice loads into: a generic element's
    /// type is the builder's.
    pub(super) fn scalar(self) -> Option<Scalar> {
        match self {
            Self::Named(p) => Scalar::of(p),
            Self::Param(_) => Some(Scalar::Inferred),
            Self::Index(_) | Self::Record => None,
        }
    }
    pub(crate) fn parse(t: &Type, types: &[&Ident]) -> Option<Self> {
        if let Some(primitive) = Primitive::of(t) {
            return Self::of(primitive);
        }
        let Type::Path(p) = t else { return None };
        let name = p.path.get_ident()?;
        types.iter().position(|ty| *ty == name).map(Self::Param)
    }
    pub(crate) fn tokens(self, root: &Path, types: &[&Ident]) -> TokenStream {
        let ty = self.ty(root, types);
        emit!(<#ty as #root::KernelScalar>::TYPE)
    }
    pub(in crate::scalar) fn ty(self, root: &Path, types: &[&Ident]) -> TokenStream {
        match self {
            Self::Param(i) => types[i].to_token_stream(),
            Self::Index(i) => {
                let ty = types[i];
                emit!(<#ty as #root::std::intrinsics::simd::LaneElement>::Index)
            }
            Self::Named(p) => Ident::new(p.name(), Span::call_site()).to_token_stream(),
            Self::Record => emit!(::core::compile_error!(
                "SIMD operations and Rust calls do not support slices of structs yet"
            )),
        }
    }
    fn index(self) -> Self {
        match self {
            Self::Named(Primitive::F32) => Self::I32,
            Self::Named(Primitive::F64) => Self::Named(Primitive::I64),
            Self::Param(i) => Self::Index(i),
            x => x,
        }
    }
}
fn generic(seg: &syn::PathSegment, types: &[&Ident]) -> Option<Elem> {
    let PathArguments::AngleBracketed(a) = &seg.arguments else {
        return None;
    };
    element_argument(a, types)
}
pub(super) fn element_argument(
    a: &syn::AngleBracketedGenericArguments,
    types: &[&Ident],
) -> Option<Elem> {
    if a.args.len() != 1 {
        return None;
    }
    let GenericArgument::Type(t) = a.args.first()? else {
        return None;
    };
    Elem::parse(t, types)
}
fn splat_element(expr: &Expr) -> Option<Elem> {
    match expr {
        Expr::Paren(p) => splat_element(&p.expr),
        Expr::Group(g) => splat_element(&g.expr),
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => splat_element(&u.expr),
        Expr::Cast(c) => Elem::parse(&c.ty, &[]),
        Expr::Lit(l) => match &l.lit {
            syn::Lit::Int(v) => Primitive::named(v.suffix()).and_then(Elem::of),
            syn::Lit::Float(v) => Primitive::named(v.suffix()).and_then(Elem::of),
            _ => None,
        },
        _ => None,
    }
}
pub(super) fn vector_type(t: &Type, types: &[&Ident]) -> Option<Scalar> {
    let Type::Path(p) = t else { return None };
    let s = p.path.segments.last()?;
    if s.ident == "Lane" {
        Some(Scalar::Vector(generic(s, types)?))
    } else if s.ident == "Mask" {
        Some(Scalar::Mask(generic(s, types)?))
    } else {
        None
    }
}
/// Resolve Type::method and <Type>::method. Rust checks the nominal owner
/// of each recognized spelling in the generated code.
fn qualified(path: &syn::ExprPath) -> Option<(Owner, &syn::PathSegment, Signature)> {
    let member = path.path.segments.last()?;
    let ty = match &path.qself {
        Some(q) if q.position == 0 || standard_operator(path) => {
            let Type::Path(p) = q.ty.as_ref() else {
                return None;
            };
            p.path.segments.last()?
        }
        Some(_) => return None,
        None => path.path.segments.iter().rev().nth(1)?,
    };
    let owner = match ty.ident.to_string().as_str() {
        "Lane" => Owner::Lane,
        "Mask" => Owner::Mask,
        other
            if other != "Self"
                && (other.starts_with(char::is_uppercase)
                    || !matches!(ty.arguments, PathArguments::None)
                    || path.qself.is_some()) =>
        {
            if lookup(Owner::Lane, &member.ident.to_string()).is_some() {
                Owner::Lane
            } else {
                Owner::Mask
            }
        }
        _ => return None,
    };
    Some((owner, ty, lookup(owner, &member.ident.to_string())?))
}

/// Full and imported standard operator traits have the same SIMD rule.
fn standard_operator(path: &syn::ExprPath) -> bool {
    let Some(q) = &path.qself else { return false };
    let mut segments = path.path.segments.iter();
    if q.position == 3 {
        return segments
            .next()
            .is_some_and(|s| s.ident == "std" || s.ident == "core")
            && segments.next().is_some_and(|s| s.ident == "ops");
    }
    q.position == 1
        && segments.next().is_some_and(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "Add"
                    | "Sub"
                    | "Mul"
                    | "Div"
                    | "BitAnd"
                    | "BitOr"
                    | "BitXor"
                    | "Not"
                    | "Shr"
                    | "AddAssign"
                    | "SubAssign"
                    | "MulAssign"
                    | "DivAssign"
                    | "BitAndAssign"
                    | "BitOrAssign"
                    | "BitXorAssign"
            )
        })
}

/// Rust resolves a short trait path. Check its identity before interpreting
/// a user-written UFCS call as an operator, even if a local trait shares its name.
fn operator_witness(path: &syn::ExprPath, owner: &Path) -> TokenStream {
    if path.qself.as_ref().is_none_or(|q| q.position != 1) || !standard_operator(path) {
        return TokenStream::new();
    }
    let name = &path.path.segments.first().unwrap().ident;
    let args = if name == "Not" {
        emit!(<Output = #owner>)
    } else if name == "Shr" {
        emit!(<u32, Output = #owner>)
    } else if name.to_string().ends_with("Assign") {
        emit!(<#owner>)
    } else {
        emit!(<#owner, Output = #owner>)
    };
    emit!(let _: ::core::marker::PhantomData<dyn ::core::ops::#name #args> =
        ::core::marker::PhantomData::<dyn #name #args>;)
}

pub(super) fn owner_path(p: &syn::ExprPath) -> Option<Path> {
    if let Some(q) = &p.qself {
        let Type::Path(path) = &*q.ty else {
            return None;
        };
        Some(path.path.clone())
    } else {
        let mut path = p.path.clone();
        path.segments.pop();
        path.segments.pop_punct();
        (!path.segments.is_empty()).then_some(path)
    }
}

pub(super) fn type_probe(root: &Path, owner: &Path) -> TokenStream {
    emit!({
        use #root::__private::SimdType as _;
        (&::core::marker::PhantomData::<#owner>).simd_type()
    })
}

pub(super) fn width_element(root: &Path, owner: &Path) -> TokenStream {
    let probe = type_probe(root, owner);
    emit!(match #probe {
        ::core::option::Option::Some(#root::hir::ValueType::Vector(e)) => ::core::option::Option::Some(e),
        _ => ::core::option::Option::None,
    })
}

pub(crate) fn simd_path(p: &syn::ExprPath) -> bool {
    qualified(p).is_some()
}

pub(super) fn has_method(ty: Scalar, name: &str) -> bool {
    lookup(Owner::of(ty).0, name).is_some()
}

pub(super) fn registered(name: &str) -> bool {
    lookup(Owner::Lane, name).is_some() || lookup(Owner::Mask, name).is_some()
}

struct Method<'a> {
    receiver: &'a Expr,
    name: &'a Ident,
    generics: Option<&'a syn::AngleBracketedGenericArguments>,
    args: Vec<&'a Expr>,
}

pub(super) fn uses_lane(e: &Expr) -> bool {
    struct Find(bool);
    impl<'a> Visit<'a> for Find {
        fn visit_expr_path(&mut self, p: &'a syn::ExprPath) {
            self.0 |= simd_path(p);
        }
    }
    let mut f = Find(false);
    f.visit_expr(e);
    f.0
}
impl Scalar {
    /// The syntactic signature supplies the IR shape. Rust checks that the
    /// declared type actually denotes the facade type, including re-exports.
    pub(in crate::scalar) fn vector_witness(
        self,
        ty: &Type,
        root: &Path,
        types: &[&Ident],
    ) -> Option<TokenStream> {
        let (name, element) = match self {
            Self::Vector(e) => (emit!(Lane), e),
            Self::Mask(e) => (emit!(Mask), e),
            _ => return None,
        };
        let element = element.ty(root, types);
        Some(
            emit!(let _: ::core::marker::PhantomData<#root::std::intrinsics::#name<#element>> = ::core::marker::PhantomData::<#ty>;),
        )
    }

    pub(in crate::scalar) fn value_type(self, root: &Path, types: &[&Ident]) -> TokenStream {
        match self {
            Self::Vector(e) => {
                let e = e.tokens(root, types);
                emit!(#root::hir::ValueType::Vector(#e))
            }
            Self::Mask(e) => {
                let e = e.tokens(root, types);
                emit!(#root::hir::ValueType::Mask(#e))
            }
            scalar => {
                let ty = super::scalar_type(root, scalar)
                    .unwrap_or_else(|| emit!(#root::hir::ScalarType::I32));
                emit!(#root::hir::ValueType::Scalar(#ty))
            }
        }
    }
}

/// Lower a width whose element was resolved by Rust (possibly through a type alias).
pub(super) fn width(root: &Path, name: &str, element: TokenStream) -> TokenStream {
    let sig = lookup(Owner::Lane, name).expect("lane count intrinsic");
    let op = (sig.lower)(&[]);
    emit!({
        use #root::hir::{Intrinsic as __SpliceIntrinsic, Simd as __SpliceSimd, SimdOp as __SpliceOp};
        let __splice_element = #element;
        #op.ok_or_else(|| #root::BuildError::staging("missing lane count"))?
    })
}

impl Lower<'_> {
    /// Bind operands in Rust evaluation order before the rule arranges them
    /// in IR order (notably stores and masked gathers).
    fn simd_call(
        &self,
        sig: Signature,
        own: Elem,
        output: Option<Elem>,
        operands: Vec<TokenStream>,
    ) -> Value {
        let root = self.root;
        let result = match sig.result {
            Output::Vector => Some(Scalar::Vector(own)),
            Output::Converted => Some(Scalar::Vector(output.expect("type-checked output element"))),
            Output::Mask => Some(Scalar::Mask(own)),
            Output::Bool => Some(Scalar::Bool),
            Output::Int => Some(Scalar::I32),
            Output::Element => Some(own.scalar().unwrap_or(Scalar::Inferred)),
            Output::Length => Some(Scalar::Length),
            Output::Unit => None,
        };
        let element = own.tokens(root, self.types);
        let output = output.map(|e| e.tokens(root, self.types));
        let hir = emit!(#root::hir);
        let call = (sig.lower)(&operands);
        let call = emit!({
            #[allow(unused_imports)]
            use #hir::{Intrinsic as __SpliceIntrinsic, Simd as __SpliceSimd, SimdOp as __SpliceOp};
            #[allow(unused_imports)]
            use #root::std::intrinsics::simd::Shuffle as __SpliceShuffle;
            let __splice_element = #element;
            #call
        });
        let call = if let Some(to) = output {
            emit!({ let __splice_to = #to; #call })
        } else {
            call
        };
        match result {
            Some(ty) => Value::new(
                Stage::Reg(ty),
                emit!(#call.ok_or_else(|| #root::BuildError::staging("missing SIMD result"))?),
            ),
            None => Value::unit(emit!({#call;})),
        }
    }

    fn simd_operand(&self, arg: &Expr, param: Param, own: Elem) -> Result<TokenStream> {
        let root = self.root;
        match param {
            Param::Buffer => {
                let value = self.expr(arg)?;
                let Stage::Buffer(buffer) = value.stage else {
                    return err(arg, "expected a runtime buffer");
                };
                if buffer.elem != own {
                    return err(arg, "SIMD buffer element type mismatch");
                }
                Ok(value.code)
            }
            Param::Splat => {
                let value = self.expr(arg)?;
                // A runtime scalar splats to lanes of its type.
                if matches!(value.stage, Stage::Reg(_) | Stage::Word(_)) {
                    let lane = own.scalar().unwrap_or(Scalar::Inferred);
                    let reg = self.reg(value, lane, arg)?;
                    Ok(emit!((__SpliceOp::SplatScalar, ::core::option::Option::Some(#reg))))
                } else {
                    let host = self.host(value, arg)?;
                    let ty = own.ty(root, self.types);
                    Ok(
                        emit!((__SpliceOp::Splat(#root::std::intrinsics::simd::LaneElement::bits({let value: #ty = #host; value})), ::core::option::Option::None)),
                    )
                }
            }
            Param::Count => {
                let value = self.host(self.expr(arg)?, arg)?;
                Ok(emit!({let count: u32 = #value; count & 31}))
            }
            Param::Mask => {
                // Rust and the IR verifier accept masks with the same index element.
                let value = self.expr_with_hint(arg, Some(Scalar::Mask(own)))?;
                if !matches!(value.stage, Stage::Reg(Scalar::Mask(_))) {
                    return err(arg, "expected a SIMD mask");
                }
                Ok(value.code)
            }
            Param::Vector => self.operand(arg, Scalar::Vector(own)),
            Param::Indices => self.operand(arg, Scalar::Vector(own.index())),
            Param::Offset => self.operand(arg, Scalar::Index),
            Param::Int => self.operand(arg, Scalar::I32),
            Param::Element => {
                let value = self.expr(arg)?;
                if matches!(value.stage, Stage::Reg(_) | Stage::Word(_)) {
                    let lane = own.scalar().unwrap_or(Scalar::Inferred);
                    self.reg(value, lane, arg)
                } else {
                    // A known element is staged as its bit pattern in the lane type.
                    let host = self.host(value, arg)?;
                    let ty = own.ty(root, self.types);
                    let hir = emit!(#root::hir);
                    Ok(emit!(__splice_builder.constant(#hir::Constant::new(
                        __splice_element,
                        #root::std::intrinsics::simd::LaneElement::bits({let value: #ty = #host; value}),
                    ))?))
                }
            }
            Param::Data => unreachable!("select types its data operands together"),
        }
    }

    pub(super) fn local_owner(&self, path: &syn::ExprPath) -> bool {
        if self.local_types.is_empty() {
            return false;
        }
        let Some(owner) = owner_path(path) else {
            return false;
        };
        let mut parts = owner.segments.iter();
        let Some(first) = parts.next() else {
            return false;
        };
        let qualified = first.ident == "self";
        let name = if qualified {
            let Some(next) = parts.next() else {
                return false;
            };
            &next.ident
        } else {
            &first.ident
        };
        if parts.next().is_some() || !self.local_types.contains(name) {
            return false;
        }
        fn imports(tree: &syn::UseTree, name: &Ident) -> bool {
            match tree {
                syn::UseTree::Name(t) => t.ident == *name,
                syn::UseTree::Rename(t) => t.rename == *name,
                syn::UseTree::Path(t) => imports(&t.tree, name),
                syn::UseTree::Group(g) => g.items.iter().any(|t| imports(t, name)),
                syn::UseTree::Glob(_) => true,
            }
        }
        qualified
            || !self
                .imports
                .iter()
                .flatten()
                .any(|u| imports(&u.tree, name))
    }

    pub(super) fn qualified_simd(&self, e: &Expr, hint: Option<Scalar>) -> Result<Option<Value>> {
        let Expr::Call(c) = e else { return Ok(None) };
        let Expr::Path(p) = c.func.as_ref() else {
            return Ok(None);
        };
        if self.local_owner(p) {
            return Ok(None);
        }
        let Some((owner, lane, sig)) = qualified(p) else {
            return Ok(None);
        };
        // A valid Rust call with different arity belongs to another declaration.
        if c.args.len() != sig.params.len() + usize::from(sig.receiver) {
            return Ok(None);
        }
        let owner_path = owner_path(p).expect("qualified SIMD owner");
        let canonical = lane.ident == "Lane" || lane.ident == "Mask";
        let root = self.root;
        let probe = type_probe(root, &owner_path);
        let name = p.path.segments.last().unwrap().ident.to_string();
        if matches!(name.as_str(), "element_width" | "byte_width") && c.args.is_empty() {
            let ordinary = self.call(c, hint)?;
            let probe = width_element(root, &owner_path);
            let resolved = width(root, &name, emit!(element));
            let alias = emit!(match #probe {
                ::core::option::Option::Some(element) => #resolved,
                ::core::option::Option::None => #ordinary,
            });
            let code = if let Some(element) = generic(lane, self.types) {
                let builtin = width(root, &name, element.tokens(root, self.types));
                super::calls::select_builtin(root, &owner_path, "lane", builtin, alias)
            } else {
                alias
            };
            let ty = if canonical || !matches!(lane.arguments, PathArguments::None) {
                Scalar::Length
            } else {
                Scalar::Inferred
            };
            return Ok(Some(Value::new(Stage::Reg(ty), code)));
        }
        let builtin = || -> Result<Value> {
            let explicit = generic(lane, self.types);
            if !matches!(lane.arguments, PathArguments::None) && explicit.is_none() {
                return err(
                    lane,
                    "expected Lane<T> with one of i8, u8, i16, u16, i32, u32, i64, f32, f64",
                );
            }
            let value = if sig.receiver {
                let mut args = c.args.iter();
                let receiver = args.next().expect("checked receiver");
                let member = p.path.segments.last().unwrap();
                let name = &member.ident;
                let receiver = match receiver {
                    Expr::Reference(r) if name.to_string().ends_with("_assign") => &*r.expr,
                    receiver => receiver,
                };
                let call = Method {
                    receiver,
                    name,
                    args: args.collect(),
                    generics: match &member.arguments {
                        PathArguments::AngleBracketed(a) => Some(a),
                        _ => None,
                    },
                };
                let expected = explicit
                    .map(|e| match owner {
                        Owner::Lane => Scalar::Vector(e),
                        Owner::Mask => Scalar::Mask(e),
                    })
                    .or(self.store_hint(name, call.args.first().copied())?);
                let value = self.expr_with_hint(receiver, expected)?;
                let Stage::Reg(t @ (Scalar::Vector(_) | Scalar::Mask(_))) = value.stage else {
                    return err(receiver, "expected a SIMD receiver");
                };
                let (actual, elem) = Owner::of(t);
                if (canonical && owner != actual) || explicit.is_some_and(|e| e != elem) {
                    return err(receiver, "SIMD receiver type mismatch");
                }
                let sig = lookup(actual, &name.to_string())
                    .ok_or_else(|| syn::Error::new_spanned(name, "unsupported SIMD receiver"))?;
                self.lower_simd_method(sig, elem, value, &call, hint)?
            } else {
                let inferred = match (sig.params.first(), c.args.first()) {
                    (Some(Param::Buffer), Some(arg)) => match self.expr(arg)?.stage {
                        Stage::Buffer(b) => Some(b.elem),
                        _ => return err(arg, "expected a runtime buffer"),
                    },
                    (Some(Param::Splat), Some(arg)) => {
                        // A runtime scalar splats to lanes of its type.
                        splat_element(arg).or(match self.expr(arg)?.stage {
                            Stage::Reg(Scalar::Inferred) => Some(Elem::I32),
                            Stage::Reg(Scalar::Number(p)) | Stage::Word(Scalar::Number(p)) => {
                                Elem::of(p)
                            }
                            _ => None,
                        })
                    }
                    _ => None,
                };
                let hinted = match hint {
                    Some(Scalar::Vector(e) | Scalar::Mask(e)) => Some(e),
                    _ => None,
                };
                let elem = explicit.or(inferred).or(hinted).ok_or_else(|| {
                    syn::Error::new_spanned(c, "specify Lane::<T> or annotate the vector binding")
                })?;
                let operands = c
                    .args
                    .iter()
                    .zip(sig.params)
                    .map(|(arg, &param)| self.simd_operand(arg, param, elem))
                    .collect::<Result<_>>()?;
                self.simd_call(sig, elem, None, operands)
            };
            let witness = operator_witness(p, &owner_path);
            let code = value.code;
            Ok(Value::new(value.stage, emit!({ #witness #code })))
        };
        let ordinary = self.call(c, hint);
        let (stage, builtin, by_type) = match builtin() {
            Ok(value) => {
                let code = value.code;
                (value.stage, code.clone(), code)
            }
            Err(error) => {
                let diagnostic = error.to_string();
                let stage = ordinary
                    .as_ref()
                    .map_or(Stage::Reg(Scalar::Inferred), |v| v.stage);
                (
                    stage,
                    error.into_compile_error(),
                    emit!(return ::core::result::Result::Err(#root::BuildError::staging(#diagnostic))),
                )
            }
        };
        let ordinary = ordinary
            .map(|v| v.code)
            .unwrap_or_else(|e| e.into_compile_error());
        let ordinary = if stage == Stage::Unit {
            emit!({let _ = #ordinary;})
        } else {
            ordinary
        };
        let fallback = emit!(if #probe.is_some() { #by_type } else { #ordinary });
        let code = super::calls::select_builtin(root, &owner_path, "simd", builtin, fallback);
        let stage = hint
            .filter(|t| {
                t.is_scalar() && matches!(stage, Stage::Reg(Scalar::Vector(_) | Scalar::Mask(_)))
            })
            .map_or(stage, Stage::Reg);
        Ok(Some(Value::new(stage, code)))
    }

    pub(super) fn simd_operator(
        &self,
        t: Scalar,
        method: &str,
        args: Vec<TokenStream>,
        at: impl ToTokens,
    ) -> Result<Value> {
        let (owner, elem) = Owner::of(t);
        let Some(sig) = lookup(owner, method) else {
            return err(
                at,
                &format!("{owner:?} has no `{method}` operator in a kernel"),
            );
        };
        Ok(self.simd_call(sig, elem, None, args))
    }
    pub(super) fn simd_binary(
        &self,
        t: Scalar,
        op: &syn::BinOp,
        l: Value,
        r: Value,
        at: &syn::ExprBinary,
    ) -> Result<Value> {
        let method = match op {
            syn::BinOp::Add(_) => "add",
            syn::BinOp::Sub(_) => "sub",
            syn::BinOp::Mul(_) => "mul",
            syn::BinOp::Div(_) => "div",
            syn::BinOp::BitAnd(_) => "bitand",
            syn::BinOp::BitOr(_) => "bitor",
            syn::BinOp::BitXor(_) => "bitxor",
            syn::BinOp::Shr(_) => "shr",
            syn::BinOp::Eq(_)
            | syn::BinOp::Ne(_)
            | syn::BinOp::Lt(_)
            | syn::BinOp::Le(_)
            | syn::BinOp::Gt(_)
            | syn::BinOp::Ge(_) => {
                return err(
                    op,
                    "use simd_eq/lt/le/gt/ge/ne for element-wise comparisons",
                );
            }
            _ => {
                return err(
                    op,
                    "a kernel supports +, -, *, /, &, |, ^ and >> on SIMD vectors",
                );
            }
        };
        if method == "shr" {
            // The shift count is known when the kernel is built.
            let shift = self.host(r, &at.right)?;
            let receiver = self.reg(l, t, at)?;
            return self.simd_operator(
                t,
                method,
                vec![receiver, emit!({let count: u32 = #shift; count & 31})],
                at,
            );
        }
        self.simd_operator(
            t,
            method,
            vec![self.reg(l, t, at)?, self.reg(r, t, at)?],
            at,
        )
    }
    pub(super) fn simd_unary(
        &self,
        t: Scalar,
        op: &syn::UnOp,
        v: Value,
        at: &syn::ExprUnary,
    ) -> Result<Value> {
        if !matches!(op, syn::UnOp::Not(_)) {
            return err(op, "unsupported SIMD unary operator");
        }
        self.simd_operator(t, "not", vec![v.code], at)
    }

    pub(super) fn store_hint(&self, name: &Ident, first: Option<&Expr>) -> Result<Option<Scalar>> {
        if lookup(Owner::Lane, &name.to_string())
            .is_some_and(|s| s.receiver && s.params.first() == Some(&Param::Buffer))
            && let Some(dst) = first
            && let Stage::Buffer(buffer) = self.expr(dst)?.stage
        {
            return Ok(Some(Scalar::Vector(buffer.elem)));
        }
        Ok(None)
    }

    pub(super) fn simd_method(
        &self,
        t: Scalar,
        receiver: Value,
        m: &syn::ExprMethodCall,
        hint: Option<Scalar>,
    ) -> Result<Value> {
        let (owner, elem) = Owner::of(t);
        let method = m.method.to_string();
        let Some(sig) = lookup(owner, &method).filter(|s| s.receiver) else {
            return err(
                &m.method,
                &format!("unsupported kernel SIMD method '{method}'"),
            );
        };
        self.lower_simd_method(sig, elem, receiver, &Method::from(m), hint)
    }

    fn lower_simd_method(
        &self,
        sig: Signature,
        elem: Elem,
        receiver: Value,
        m: &Method<'_>,
        hint: Option<Scalar>,
    ) -> Result<Value> {
        if m.args.len() != sig.params.len() {
            return err(m.name, "wrong number of SIMD arguments");
        }
        let mut output = m
            .generics
            .and_then(|a| element_argument(a, self.types))
            .or(match hint {
                Some(Scalar::Vector(e)) => Some(e),
                _ => None,
            });
        let mut operands = vec![receiver.code];
        let mut element = elem;
        if sig.params == [Param::Data, Param::Data] {
            let (yes, no) = self.paired_operands(
                m.args[0],
                m.args[1],
                output.map(Scalar::Vector),
                Some(Scalar::Vector(elem)),
            )?;
            let Stage::Reg(ty @ Scalar::Vector(data)) = yes.stage else {
                return err(m.name, "select requires two vectors");
            };
            operands.extend([yes.code, self.reg(no, ty, m.args[1])?]);
            element = data;
            output = Some(data);
        } else {
            for (arg, &param) in m.args.iter().zip(sig.params) {
                operands.push(self.simd_operand(arg, param, elem)?);
            }
        }
        if sig.result == Output::Converted && output.is_none() {
            return err(m.name, "specify the output lane type");
        }
        if m.name.to_string().ends_with("_assign") {
            let Expr::Path(place) = m.receiver else {
                return err(m.name, "assign to a mutable runtime variable");
            };
            if !matches!(self.binding(place), Some(super::Binding::Reg(_, true))) {
                return err(m.name, "assignment requires a mutable runtime variable");
            }
            let value = self.simd_call(sig, element, None, operands);
            return Ok(Value::unit(
                emit!({let result = #value; __splice_builder.assign(#place, result)?;}),
            ));
        }
        Ok(self.simd_call(
            sig,
            element,
            output.filter(|_| sig.result == Output::Converted),
            operands,
        ))
    }
}

impl<'a> From<&'a syn::ExprMethodCall> for Method<'a> {
    fn from(m: &'a syn::ExprMethodCall) -> Self {
        Self {
            receiver: &m.receiver,
            name: &m.method,
            generics: m.turbofish.as_ref(),
            args: m.args.iter().collect(),
        }
    }
}
