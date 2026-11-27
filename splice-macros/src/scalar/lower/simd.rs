use super::{Lower, Scalar, Stage, Value, err};
use crate::kast::types::Primitive;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, format_ident};
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
    pub(crate) const U32: Self = Self::Named(Primitive::U32);
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
    pub(in crate::scalar) fn is_float(self) -> bool {
        self.primitive().is_some_and(Primitive::is_float)
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
pub(crate) fn constructor_arity(p: &Path) -> Option<usize> {
    let mut segments = p.segments.iter().rev();
    let member = &segments.next()?.ident;
    if segments.next()?.ident != "Lane" {
        return None;
    }
    match member.to_string().as_str() {
        "lanes" | "iota" => Some(0),
        "splat" | "first_n" => Some(1),
        "alignment_prefix" | "load" | "gather" => Some(2),
        "load_select" | "gather_select" => Some(4),
        _ => None,
    }
}
pub(super) fn uses_lane(e: &Expr) -> bool {
    struct Find(bool);
    impl<'a> Visit<'a> for Find {
        fn visit_expr_path(&mut self, p: &'a syn::ExprPath) {
            self.0 |= constructor_arity(&p.path).is_some();
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
    fn vector_op(
        &self,
        e: Elem,
        op: TokenStream,
        inputs: Vec<TokenStream>,
        output: Option<Scalar>,
    ) -> Value {
        let root = self.root;
        let elem = e.tokens(root, self.types);
        let res = emit!(__splice_builder.simd(#root::hir::Simd{element:#elem,op:#root::hir::SimdOp::#op},&__splice_operands)?);
        let args = emit!(let __splice_operands=[#(#inputs),*];);
        match output {
            Some(ty) => Value::new(
                Stage::Reg(ty),
                emit!({#args #res.ok_or_else(|| #root::BuildError::staging("missing SIMD result"))?}),
            ),
            None => Value::unit(emit!({#args #res;})),
        }
    }
    fn simd_operand(&self, expr: &Expr, ty: Scalar) -> Result<TokenStream> {
        if let Scalar::Mask(want) = ty {
            let actual = self.expr_with_hint(expr, Some(ty))?;
            if let Stage::Reg(Scalar::Mask(have)) = actual.stage
                && have.index() == want.index()
            {
                return Ok(actual.code);
            }
            return err(expr, "expected a mask with one predicate per data lane");
        }
        self.operand(expr, ty)
    }
    pub(super) fn simd_constructor(&self, e: &Expr, hint: Option<Scalar>) -> Result<Option<Value>> {
        let Expr::Call(c) = e else { return Ok(None) };
        let Expr::Path(p) = c.func.as_ref() else {
            return Ok(None);
        };
        let Some(n) = constructor_arity(&p.path) else {
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
        if c.args.len() != n {
            return err(c, "wrong number of SIMD arguments");
        }
        let buffer = if matches!(
            name.as_str(),
            "alignment_prefix" | "load" | "gather" | "load_select" | "gather_select"
        ) {
            let v = self.expr(&c.args[0])?;
            let Stage::Buffer(b) = v.stage else {
                return err(c, "expected a runtime buffer");
            };
            Some((b.elem, v.code))
        } else {
            None
        };
        let hinted = match hint {
            Some(Scalar::Vector(e) | Scalar::Mask(e)) => Some(e),
            _ => None,
        };
        let runtime_splat = if name == "splat" {
            Some(self.expr(&c.args[0])?)
        } else {
            None
        };
        let runtime_i32 = runtime_splat
            .as_ref()
            .filter(|v| matches!(v.stage, Stage::Reg(Scalar::I32 | Scalar::Inferred)))
            .map(|_| Elem::I32);
        let elem = explicit
            .or_else(|| buffer.as_ref().map(|b| b.0))
            .or_else(|| {
                runtime_splat
                    .as_ref()
                    .and_then(|_| splat_element(&c.args[0]))
            })
            .or(runtime_i32)
            .or(hinted)
            .ok_or_else(|| {
                syn::Error::new_spanned(c, "specify Lane::<T> or annotate the vector binding")
            })?;
        let v = Scalar::Vector(elem);
        let mask = Scalar::Mask(elem);
        let idx = Scalar::Vector(elem.index());
        let root = self.root;
        let ty = elem.ty(root, self.types);
        let mut lane_path = p.path.clone();
        lane_path.segments.pop();
        lane_path.segments.pop_punct();
        lane_path.segments.last_mut().unwrap().arguments = PathArguments::None;
        let witness = emit!(let _: ::core::marker::PhantomData<#root::simd::Lane<#ty>>=::core::marker::PhantomData::<#lane_path<#ty>>;);
        let get = |j: usize, t| self.simd_operand(&c.args[j], t);
        let mut pre = TokenStream::new();
        let mut result = match name.as_str() {
            "first_n" => self.vector_op(
                elem,
                emit!(FirstN),
                vec![self.operand(&c.args[0], Scalar::Index)?],
                Some(mask),
            ),
            "lanes" => self.vector_op(elem, emit!(Lanes), vec![], Some(Scalar::Length)),
            "iota" => self.vector_op(elem, emit!(Iota), vec![], Some(v)),
            "splat" => {
                let arg = runtime_splat.unwrap();
                if elem == Elem::I32 && matches!(arg.stage, Stage::Reg(_)) {
                    let reg = self.reg(arg, Scalar::I32, &c.args[0])?;
                    self.vector_op(elem, emit!(SplatScalar), vec![reg], Some(v))
                } else {
                    let val = self.host(arg, &c.args[0])?;
                    pre.extend(emit!(let scalar:#ty=#val;));
                    self.vector_op(
                        elem,
                        emit!(Splat(#root::simd::LaneElement::bits(scalar))),
                        vec![],
                        Some(v),
                    )
                }
            }
            _ => {
                let (b, buf) = buffer.unwrap();
                if b != elem {
                    return err(c, "SIMD buffer element type mismatch");
                }
                pre.extend(emit!(let buffer=#buf;));
                match name.as_str() {
                    "alignment_prefix" if elem == Elem::I32 => {
                        let index = self.expr(&c.args[1])?;
                        let offset = self.reg(index, Scalar::Index, &c.args[1])?;
                        self.vector_op(
                            elem,
                            emit!(AlignmentPrefix(buffer)),
                            vec![offset],
                            Some(Scalar::Length),
                        )
                    }
                    "alignment_prefix" => return err(c, "alignment_prefix expects i32 lanes"),
                    "load" | "load_select" => {
                        let mut a = vec![self.operand(&c.args[1], Scalar::Index)?];
                        let op = if n == 4 {
                            a.extend([get(2, mask)?, get(3, v)?]);
                            emit!(LoadSelect(buffer))
                        } else {
                            emit!(Load(buffer))
                        };
                        self.vector_op(elem, op, a, Some(v))
                    }
                    "gather" => {
                        self.vector_op(elem, emit!(Gather(buffer)), vec![get(1, idx)?], Some(v))
                    }
                    _ => {
                        let m = get(1, mask)?;
                        pre.extend(emit!(let mask=#m;));
                        self.vector_op(
                            elem,
                            emit!(GatherSelect(buffer)),
                            vec![get(2, idx)?, emit!(mask), get(3, v)?],
                            Some(v),
                        )
                    }
                }
            }
        };
        let code = result.code;
        result.code = emit!({#witness #pre #code});
        Ok(Some(result))
    }
    pub(super) fn simd_operator(
        &self,
        t: Scalar,
        op: &str,
        args: Vec<TokenStream>,
        at: impl ToTokens,
    ) -> Result<Value> {
        let (e, name, out) = match t {
            Scalar::Mask(e) => match op {
                "And" => (e, "MaskAnd", t),
                "Or" => (e, "MaskOr", t),
                "Xor" => (e, "MaskXor", t),
                "Not" => (e, "MaskNot", t),
                _ => return err(at, "unsupported SIMD mask operator"),
            },
            Scalar::Vector(e) => match op {
                "Add" | "Sub" | "Mul" => (e, op, t),
                "Div" if e.is_float() => (e, op, t),
                "And" | "Or" | "Xor" | "Not" if !e.is_float() => (e, op, t),
                "Eq" | "Ne" | "Lt" | "Le" | "Gt" | "Ge" => (e, op, Scalar::Mask(e)),
                _ => return err(at, "unsupported SIMD vector operator"),
            },
            _ => unreachable!(),
        };
        let name = format_ident!("{name}");
        Ok(self.vector_op(e, emit!(#name), args, Some(out)))
    }
    pub(super) fn simd_binary(
        &self,
        t: Scalar,
        op: &syn::BinOp,
        l: Value,
        r: Value,
        at: &syn::ExprBinary,
    ) -> Result<Value> {
        if let Scalar::Vector(elem @ (Elem::I32 | Elem::U32)) = t
            && matches!(op, syn::BinOp::Shr(_))
        {
            let shift = self.host(r, &at.right)?;
            return Ok(self.vector_op(
                elem,
                emit!(Shr({let shift:u32=#shift;shift & 31})),
                vec![self.reg(l, t, at)?],
                Some(t),
            ));
        }
        let name = match op {
            syn::BinOp::Add(_) => "Add",
            syn::BinOp::Sub(_) => "Sub",
            syn::BinOp::Mul(_) => "Mul",
            syn::BinOp::Div(_) => "Div",
            syn::BinOp::BitAnd(_) => "And",
            syn::BinOp::BitOr(_) => "Or",
            syn::BinOp::BitXor(_) => "Xor",
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
                    "a kernel supports +, -, *, /, &, | and ^ on SIMD vectors",
                );
            }
        };
        self.simd_operator(t, name, vec![self.reg(l, t, at)?, self.reg(r, t, at)?], at)
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
        self.simd_operator(t, "Not", vec![v.code], at)
    }
    /// A method of the vector or mask `receiver`: the registry
    /// (`frontend::methods::simd_method`) stages it on the operands,
    /// typed by the builder. The macro gives the result its stage.
    pub(super) fn simd_method(
        &self,
        t: Scalar,
        receiver: Value,
        m: &syn::ExprMethodCall,
        hint: Option<Scalar>,
    ) -> Result<Value> {
        let (Scalar::Mask(e) | Scalar::Vector(e)) = t else {
            unreachable!()
        };
        let root = self.root;
        let method = m.method.to_string();
        // The element of a conversion's result or of the selected data.
        let data = m
            .turbofish
            .as_ref()
            .and_then(|args| element_argument(args, self.types))
            .map(Scalar::Vector)
            .or(hint.filter(|t| matches!(t, Scalar::Vector(_))));
        let mut stages = Vec::new();
        let mut operands = Vec::new();
        if method == "select" && m.args.len() == 2 {
            // Either data operand can type an unqualified Lane constructor.
            let (yes, no) =
                self.paired_operands(&m.args[0], &m.args[1], data, Some(Scalar::Vector(e)))?;
            let Stage::Reg(ty @ Scalar::Vector(_)) = yes.stage else {
                return err(m, "mask.select requires two vectors");
            };
            let no = self.reg(no, ty, &m.args[1])?;
            let operand = emit!(#root::frontend::methods::Operand::Reg);
            stages.push(yes.stage);
            operands.extend([emit!(#operand(#yes)), emit!(#operand(#no))]);
        } else {
            let hint = data.unwrap_or(Scalar::Vector(e));
            for arg in &m.args {
                let (stage, operand) = self.simd_argument(arg, hint)?;
                stages.push(stage);
                operands.push(operand);
            }
        }
        let result = match method.as_str() {
            "simd_eq" | "simd_ne" | "simd_lt" | "simd_le" | "simd_gt" | "simd_ge" => {
                Some(Scalar::Mask(e))
            }
            "any" | "all" => Some(Scalar::Bool),
            "count" => Some(Scalar::I32),
            // A scalar of the lane type.
            "reduce_sum" | "extract_wrapping" | "last" => e.scalar(),
            "cast" | "widen_low" | "widen_high" | "narrow" => match data {
                Some(to) => Some(to),
                None => return err(m, "specify the output lane type"),
            },
            "select" => match stages.first() {
                Some(Stage::Reg(ty @ Scalar::Vector(_))) => Some(*ty),
                _ => return err(m, "mask.select requires two vectors"),
            },
            "store" | "store_select" | "scatter" | "scatter_select" | "compress_store" => None,
            _ => Some(Scalar::Vector(e)),
        };
        let output = match data {
            Some(Scalar::Vector(to)) => {
                let to = to.tokens(root, self.types);
                emit!(::core::option::Option::Some(#to))
            }
            _ => emit!(::core::option::Option::None),
        };
        // rustc checks that `Lane`/`Mask` has the method; the registry
        // stages it or reports a method it does not have.
        let call = emit!({
            let receiver = #receiver;
            let operands = [#(#operands),*];
            #root::frontend::methods::simd_method(__splice_builder, #method, receiver, &operands, #output)?
        });
        Ok(match result {
            Some(ty) => Value::new(
                Stage::Reg(ty),
                emit!(#call.ok_or_else(|| #root::BuildError::staging("missing SIMD result"))?),
            ),
            None => Value::unit(emit!({ #call; })),
        })
    }

    /// An argument of a SIMD method: its stage and its registry operand.
    fn simd_argument(&self, arg: &Expr, hint: Scalar) -> Result<(Stage, TokenStream)> {
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
