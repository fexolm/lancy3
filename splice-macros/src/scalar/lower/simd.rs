use super::{Lower, Scalar, Stage, Value, err};
use crate::kast::types::Primitive;
use proc_macro2::{Ident, Span, TokenStream};
use quote::ToTokens;
use syn::{Expr, GenericArgument, Path, PathArguments, Result, Type, visit::Visit};

/// The element of a slice, vector or mask: a named primitive, a generic
/// parameter of the kernel, or the lane index type of a generic parameter.
/// rustc checked that a named element is a `BufferElement` (or a
/// `LaneElement` for a vector); the IR type is `<T as KernelScalar>::TYPE`.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Elem {
    Named(Primitive),
    Param(usize),
    Index(usize),
}
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct BufferType {
    pub elem: Elem,
    pub write: bool,
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
            Self::Param(_) | Self::Index(_) => None,
        }
    }
    /// The register an element of a slice loads into: a generic element's
    /// type is the builder's.
    pub(super) fn scalar(self) -> Option<Scalar> {
        match self {
            Self::Named(p) => Scalar::of(p),
            Self::Param(_) => Some(Scalar::Inferred),
            Self::Index(_) => None,
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
                emit!(<#ty as #root::simd::LaneElement>::Index)
            }
            Self::Named(p) => Ident::new(p.name(), Span::call_site()).to_token_stream(),
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
/// The declarations of `splice::simd` (`splice_core::simd` expands them).
const API: &str = include_str!("../../simd_api.rs");

/// The items of `simd_api.rs`.
pub(crate) fn simd_api_tokens() -> TokenStream {
    API.parse().expect("simd_api.rs is Rust")
}

/// The type that declares a SIMD intrinsic.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Owner {
    Lane,
    Mask,
}

/// An element in a declared type.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Declared {
    /// The element of `Self`.
    Own,
    /// `T::Index`.
    Index,
    /// A type parameter of the method (`U` of `cast::<U>`).
    Method,
}

/// A parameter or result type of a SIMD intrinsic.
#[derive(Copy, Clone, PartialEq, Eq)]
enum DeclaredType {
    /// `Self`: the vector or mask of the call.
    Own,
    Lane(Declared),
    Mask(Declared),
    /// `&[T]` or `&mut [T]`.
    Slice,
    /// `T`, the element.
    Element,
    Scalar(Scalar),
    Unit,
}

/// The signature of a SIMD intrinsic: `receiver` is false for an
/// associated function (`Lane::load`).
pub(crate) struct Signature {
    receiver: bool,
    params: Vec<DeclaredType>,
    result: DeclaredType,
}

/// The SIMD intrinsics `simd_api.rs` declares, by owner and method. A
/// method declared in several impls (`rotate_left` of `Lane<i32>` and
/// `Lane<u32>`) has one signature in terms of `Self`.
fn declared() -> &'static std::collections::HashMap<(Owner, String), Signature> {
    static DECLARED: std::sync::OnceLock<std::collections::HashMap<(Owner, String), Signature>> =
        std::sync::OnceLock::new();
    DECLARED.get_or_init(|| {
        let file = syn::parse_file(API).expect("simd_api.rs is Rust");
        let mut declared = std::collections::HashMap::new();
        for item in file.items {
            let syn::Item::Impl(implementation) = item else {
                continue;
            };
            let Type::Path(owner) = &*implementation.self_ty else {
                panic!("a SIMD impl is for Lane or Mask");
            };
            let owner = owner.path.segments.last().unwrap();
            let own = match &owner.arguments {
                PathArguments::AngleBracketed(a) => {
                    a.args.first().map(|a| a.to_token_stream().to_string())
                }
                _ => None,
            };
            let owner = match owner.ident.to_string().as_str() {
                "Lane" => Owner::Lane,
                "Mask" => Owner::Mask,
                other => panic!("a SIMD impl for `{other}`"),
            };
            for item in implementation.items {
                let syn::ImplItem::Verbatim(tokens) = item else {
                    continue;
                };
                let method: syn::TraitItemFn = syn::parse::Parser::parse2(
                    |input: syn::parse::ParseStream| {
                        let _ = input.call(syn::Attribute::parse_outer)?;
                        let _: syn::Visibility = input.parse()?;
                        input.parse()
                    },
                    tokens,
                )
                .expect("a body-less SIMD intrinsic");
                let sig = &method.sig;
                let generic: Vec<_> = sig
                    .generics
                    .type_params()
                    .map(|t| t.ident.to_string())
                    .collect();
                let ty = |t: &Type| declared_type(t, own.as_deref(), &generic);
                let receiver = matches!(sig.inputs.first(), Some(syn::FnArg::Receiver(_)));
                let params = sig
                    .inputs
                    .iter()
                    .filter_map(|input| match input {
                        syn::FnArg::Typed(t) => Some(ty(&t.ty)),
                        syn::FnArg::Receiver(_) => None,
                    })
                    .collect();
                let result = match &sig.output {
                    syn::ReturnType::Default => DeclaredType::Unit,
                    syn::ReturnType::Type(_, t) => match ty(t) {
                        DeclaredType::Scalar(Scalar::Index) => DeclaredType::Scalar(Scalar::Length),
                        t => t,
                    },
                };
                declared
                    .entry((owner, sig.ident.to_string()))
                    .or_insert(Signature {
                        receiver,
                        params,
                        result,
                    });
            }
        }
        declared
    })
}

/// A type of a declaration in an impl whose `Self` element is `own` (`T`
/// or a primitive) and whose method has the type parameters `generic`.
fn declared_type(t: &Type, own: Option<&str>, generic: &[String]) -> DeclaredType {
    let element = |t: &Type| {
        let name = t.to_token_stream().to_string().replace(' ', "");
        if Some(name.as_str()) == own {
            Declared::Own
        } else if name == "T::Index" {
            Declared::Index
        } else if generic.contains(&name) {
            Declared::Method
        } else {
            panic!("SIMD declaration element `{name}`")
        }
    };
    match t {
        Type::Reference(r) if matches!(&*r.elem, Type::Slice(_)) => DeclaredType::Slice,
        Type::Path(p) if p.path.is_ident("Self") => DeclaredType::Own,
        Type::Path(p) if own == Some("T") && p.path.is_ident("T") => DeclaredType::Element,
        Type::Path(p) if matches!(p.path.segments.last(), Some(s) if s.ident == "Lane" || s.ident == "Mask") =>
        {
            let s = p.path.segments.last().unwrap();
            let PathArguments::AngleBracketed(a) = &s.arguments else {
                panic!("SIMD declaration `{}` without an element", s.ident)
            };
            let Some(GenericArgument::Type(e)) = a.args.first() else {
                panic!("SIMD declaration `{}` without an element", s.ident)
            };
            if s.ident == "Lane" {
                DeclaredType::Lane(element(e))
            } else {
                DeclaredType::Mask(element(e))
            }
        }
        t => DeclaredType::Scalar(
            Scalar::from_type(t, &[])
                .unwrap_or_else(|| panic!("SIMD declaration type `{}`", t.to_token_stream())),
        ),
    }
}

impl Signature {
    /// The kernel type of `t` in a call on `owner` of element `own`, whose
    /// method type parameter is `generic`; `None` for a unit result, an
    /// error for an unknown type parameter.
    fn resolve(
        t: DeclaredType,
        owner: Owner,
        own: Elem,
        generic: Option<Elem>,
    ) -> std::result::Result<Option<Scalar>, ()> {
        let element = |d| match d {
            Declared::Own => Ok(own),
            Declared::Index => Ok(own.index()),
            Declared::Method => generic.ok_or(()),
        };
        Ok(Some(match t {
            DeclaredType::Own if owner == Owner::Mask => Scalar::Mask(own),
            DeclaredType::Own => Scalar::Vector(own),
            DeclaredType::Lane(d) => Scalar::Vector(element(d)?),
            DeclaredType::Mask(d) => Scalar::Mask(element(d)?),
            DeclaredType::Element => own.scalar().ok_or(())?,
            DeclaredType::Scalar(s) => s,
            DeclaredType::Slice | DeclaredType::Unit => return Ok(None),
        }))
    }
}

/// The signature of `Lane::<T>::name` that `p` names, an associated
/// function of `Lane`.
fn constructor(p: &Path) -> Option<&'static Signature> {
    let mut segments = p.segments.iter().rev();
    let member = &segments.next()?.ident;
    if segments.next()?.ident != "Lane" {
        return None;
    }
    declared()
        .get(&(Owner::Lane, member.to_string()))
        .filter(|s| !s.receiver)
}
pub(crate) fn constructor_arity(p: &Path) -> Option<usize> {
    constructor(p).map(|s| s.params.len())
}
pub(super) fn uses_lane(e: &Expr) -> bool {
    struct Find(bool);
    impl<'a> Visit<'a> for Find {
        fn visit_expr_path(&mut self, p: &'a syn::ExprPath) {
            self.0 |= constructor(&p.path).is_some();
        }
    }
    let mut f = Find(false);
    f.visit_expr(e);
    f.0
}
impl Scalar {
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
impl Lower<'_> {
    /// Stage the SIMD intrinsic `method` of `this` (`SimdSelf` tokens) on
    /// `operands` (`Operand` tokens); `result` types the value.
    fn simd_call(
        &self,
        method: &str,
        this: TokenStream,
        operands: Vec<TokenStream>,
        output: Option<Elem>,
        result: Option<Scalar>,
    ) -> Value {
        let root = self.root;
        let output = match output {
            Some(to) => {
                let to = to.tokens(root, self.types);
                emit!(::core::option::Option::Some(#to))
            }
            None => emit!(::core::option::Option::None),
        };
        let call = emit!({
            let __splice_this = #this;
            let __splice_operands = [#(#operands),*];
            #root::frontend::methods::simd(__splice_builder, #method, __splice_this, &__splice_operands, #output)?
        });
        match result {
            Some(ty) => Value::new(
                Stage::Reg(ty),
                emit!(#call.ok_or_else(|| #root::BuildError::staging("missing SIMD result"))?),
            ),
            None => Value::unit(emit!({ #call; })),
        }
    }
    /// The operand of the declared parameter `param` of a call on `owner`
    /// of element `own`.
    fn declared_operand(
        &self,
        arg: &Expr,
        param: DeclaredType,
        owner: Owner,
        own: Elem,
        generic: Option<Elem>,
    ) -> Result<TokenStream> {
        let root = self.root;
        let operand = emit!(#root::frontend::methods::Operand);
        Ok(match param {
            DeclaredType::Slice => {
                let v = self.expr(arg)?;
                let Stage::Buffer(b) = v.stage else {
                    return err(arg, "expected a runtime buffer");
                };
                if b.elem != own {
                    return err(arg, "SIMD buffer element type mismatch");
                }
                emit!(#operand::Buffer(#v))
            }
            DeclaredType::Element => {
                let v = self.expr(arg)?;
                if own == Elem::I32 && matches!(v.stage, Stage::Reg(_)) {
                    let reg = self.reg(v, Scalar::I32, arg)?;
                    emit!(#operand::Reg(#reg))
                } else {
                    // A known element is staged as its bit pattern.
                    let value = self.host(v, arg)?;
                    let ty = own.ty(root, self.types);
                    emit!(#operand::Known(#root::hir::Constant::from(#root::simd::LaneElement::bits({let scalar: #ty = #value; scalar}))))
                }
            }
            DeclaredType::Own | DeclaredType::Mask(_) | DeclaredType::Lane(_) => {
                let ty = Signature::resolve(param, owner, own, generic.or(Some(own)))
                    .ok()
                    .flatten()
                    .expect("a vector or mask parameter");
                self.simd_argument(arg, ty)?.1
            }
            DeclaredType::Scalar(ty) => self.simd_argument(arg, ty)?.1,
            DeclaredType::Unit => unreachable!("a unit parameter"),
        })
    }
    pub(super) fn simd_constructor(&self, e: &Expr, hint: Option<Scalar>) -> Result<Option<Value>> {
        let Expr::Call(c) = e else { return Ok(None) };
        let Expr::Path(p) = c.func.as_ref() else {
            return Ok(None);
        };
        let Some(sig) = constructor(&p.path) else {
            return Ok(None);
        };
        let lane = p.path.segments.iter().rev().nth(1).unwrap();
        let explicit = generic(lane, self.types);
        if !matches!(lane.arguments, PathArguments::None) && explicit.is_none() {
            return err(
                lane,
                "expected Lane<T> with one of u16, u32, i32, i64, f32, f64",
            );
        }
        let name = p.path.segments.last().unwrap().ident.to_string();
        if c.args.len() != sig.params.len() {
            return err(c, "wrong number of SIMD arguments");
        }
        // The element: explicit, else a slice argument's, else a typed
        // splatted value's, else the type the use expects.
        let mut buffer = None;
        let mut splat = None;
        for (arg, param) in c.args.iter().zip(&sig.params) {
            match param {
                DeclaredType::Slice if buffer.is_none() => {
                    let Stage::Buffer(b) = self.expr(arg)?.stage else {
                        return err(c, "expected a runtime buffer");
                    };
                    buffer = Some(b.elem);
                }
                DeclaredType::Element => {
                    let v = self.expr(arg)?;
                    splat = splat_element(arg).or(matches!(
                        v.stage,
                        Stage::Reg(Scalar::I32 | Scalar::Inferred)
                    )
                    .then_some(Elem::I32));
                }
                _ => {}
            }
        }
        let hinted = match hint {
            Some(Scalar::Vector(e) | Scalar::Mask(e)) => Some(e),
            _ => None,
        };
        let elem = explicit.or(buffer).or(splat).or(hinted).ok_or_else(|| {
            syn::Error::new_spanned(c, "specify Lane::<T> or annotate the vector binding")
        })?;
        let root = self.root;
        let ty = elem.ty(root, self.types);
        let mut lane_path = p.path.clone();
        lane_path.segments.pop();
        lane_path.segments.pop_punct();
        lane_path.segments.last_mut().unwrap().arguments = PathArguments::None;
        let witness = emit!(let _: ::core::marker::PhantomData<#root::simd::Lane<#ty>>=::core::marker::PhantomData::<#lane_path<#ty>>;);
        let operands = c
            .args
            .iter()
            .zip(&sig.params)
            .map(|(arg, &param)| match param {
                // An offset or count stages before the other operands.
                DeclaredType::Scalar(Scalar::Index) => {
                    let reg = self.operand(arg, Scalar::Index)?;
                    Ok(emit!(#root::frontend::methods::Operand::Reg(#reg)))
                }
                param => self.declared_operand(arg, param, Owner::Lane, elem, None),
            })
            .collect::<Result<Vec<_>>>()?;
        let Ok(result) = Signature::resolve(sig.result, Owner::Lane, elem, None) else {
            return err(c, "unsupported SIMD element");
        };
        let element = elem.tokens(root, self.types);
        let mut value = self.simd_call(
            &name,
            emit!(#root::frontend::methods::SimdSelf::Lane(#element)),
            operands,
            None,
            result,
        );
        let code = value.code;
        value.code = emit!({#witness #code});
        Ok(Some(value))
    }
    /// The operator trait method `method` (`add`, `not`) of the vector or
    /// mask `t`, on the registers `args`, the receiver first.
    pub(super) fn simd_operator(
        &self,
        t: Scalar,
        method: &str,
        args: Vec<TokenStream>,
        at: impl ToTokens,
    ) -> Result<Value> {
        let owner = match t {
            Scalar::Vector(_) => Owner::Lane,
            Scalar::Mask(_) => Owner::Mask,
            _ => unreachable!(),
        };
        if !declared().contains_key(&(owner, method.to_string())) {
            return err(
                at,
                &format!("{owner:?} has no `{method}` operator in a kernel"),
            );
        }
        let root = self.root;
        let mut args = args.into_iter();
        let receiver = args.next().expect("a receiver");
        let operands = args
            .map(|a| emit!(#root::frontend::methods::Operand::Reg(#a)))
            .collect();
        Ok(self.simd_call(
            method,
            emit!(#root::frontend::methods::SimdSelf::Value(#receiver)),
            operands,
            None,
            Some(t),
        ))
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
            let root = self.root;
            let shift = self.host(r, &at.right)?;
            let receiver = self.reg(l, t, at)?;
            return Ok(self.simd_call(
                method,
                emit!(#root::frontend::methods::SimdSelf::Value(#receiver)),
                vec![emit!(#root::frontend::methods::Operand::Known(#root::hir::Constant::from({let shift: u32 = #shift; shift})))],
                None,
                Some(t),
            ));
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
    /// The receiver type that a SIMD method writing the slice `dst` (its
    /// first parameter) gives its receiver: a vector of the slice's element.
    pub(super) fn simd_store_receiver(&self, m: &syn::ExprMethodCall) -> Result<Option<Scalar>> {
        let writes = declared()
            .get(&(Owner::Lane, m.method.to_string()))
            .is_some_and(|s| s.receiver && s.params.first() == Some(&DeclaredType::Slice));
        if writes
            && let Some(dst) = m.args.first()
            && let Stage::Buffer(buffer) = self.expr(dst)?.stage
        {
            return Ok(Some(Scalar::Vector(buffer.elem)));
        }
        Ok(None)
    }
    /// A method of the vector or mask `receiver`, typed by its declaration;
    /// `frontend::methods::simd` stages it on the operands.
    pub(super) fn simd_method(
        &self,
        t: Scalar,
        receiver: Value,
        m: &syn::ExprMethodCall,
        hint: Option<Scalar>,
    ) -> Result<Value> {
        let (owner, e) = match t {
            Scalar::Vector(e) => (Owner::Lane, e),
            Scalar::Mask(e) => (Owner::Mask, e),
            _ => unreachable!(),
        };
        let root = self.root;
        let method = m.method.to_string();
        let Some(sig) = declared()
            .get(&(owner, method.clone()))
            .filter(|s| s.receiver)
        else {
            return err(
                &m.method,
                &format!("`{method}` is not a kernel SIMD operation"),
            );
        };
        if m.args.len() != sig.params.len() {
            return err(m, "wrong number of SIMD arguments");
        }
        // The method's element parameter: explicit, else the element of
        // the result the use expects.
        let mut generic = m
            .turbofish
            .as_ref()
            .and_then(|args| element_argument(args, self.types))
            .or(match hint {
                Some(Scalar::Vector(e)) => Some(e),
                _ => None,
            });
        let method_data = DeclaredType::Lane(Declared::Method);
        let mut operands = Vec::new();
        if sig.params == [method_data, method_data] {
            // `mask.select(yes, no)`: either data operand can type an
            // unqualified Lane constructor, and they give the element.
            let (yes, no) = self.paired_operands(
                &m.args[0],
                &m.args[1],
                generic.map(Scalar::Vector),
                Some(Scalar::Vector(e)),
            )?;
            let Stage::Reg(ty @ Scalar::Vector(data)) = yes.stage else {
                return err(m, &format!("{method} requires two vectors"));
            };
            let no = self.reg(no, ty, &m.args[1])?;
            let operand = emit!(#root::frontend::methods::Operand::Reg);
            generic = Some(data);
            operands.extend([emit!(#operand(#yes)), emit!(#operand(#no))]);
        } else {
            for (arg, &param) in m.args.iter().zip(&sig.params) {
                operands.push(self.declared_operand(arg, param, owner, e, generic)?);
            }
        }
        let Ok(result) = Signature::resolve(sig.result, owner, e, generic) else {
            return err(m, "specify the output lane type");
        };
        let output = generic.filter(|_| matches!(sig.result, DeclaredType::Lane(Declared::Method)));
        Ok(self.simd_call(
            &method,
            emit!(#root::frontend::methods::SimdSelf::Value(#receiver)),
            operands,
            output,
            result,
        ))
    }

    /// An argument of a SIMD method: its stage and its operand.
    pub(super) fn simd_argument(&self, arg: &Expr, hint: Scalar) -> Result<(Stage, TokenStream)> {
        let root = self.root;
        let operand = emit!(#root::frontend::methods::Operand);
        if self.body.known(arg) {
            let value = self.known_operand(arg, None)?;
            return Ok((
                Stage::Host,
                emit!(#operand::Known(#root::frontend::scalar::ScalarConstant::constant(#value))),
            ));
        }
        let value = self.expr_with_hint(arg, Some(hint))?;
        Ok(match value.stage {
            Stage::Buffer(_) => (value.stage, emit!(#operand::Buffer(#value))),
            Stage::Reg(_) => (value.stage, emit!(#operand::Reg(#value))),
            Stage::Word(t) => {
                let reg = self.reg(value, t, arg)?;
                (Stage::Reg(t), emit!(#operand::Reg(#reg)))
            }
            _ => return err(arg, "expected a SIMD operand"),
        })
    }
}
