use super::{Lower, Scalar, Stage, Value, err};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, format_ident};
use syn::{Expr, GenericArgument, Path, PathArguments, Result, Type, visit::Visit};

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Elem {
    U16,
    I32,
    I64,
    F32,
    F64,
}
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct BufferType {
    pub elem: Elem,
    pub write: bool,
}
impl Elem {
    fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::U16 => ("u16", "U16"),
            Self::I32 => ("i32", "I32"),
            Self::I64 => ("i64", "I64"),
            Self::F32 => ("f32", "F32"),
            Self::F64 => ("f64", "F64"),
        }
    }
    fn named(name: &str) -> Option<Self> {
        [Self::U16, Self::I32, Self::I64, Self::F32, Self::F64]
            .into_iter()
            .find(|v| v.names().0 == name)
    }
    pub(crate) fn parse(t: &Type) -> Option<Self> {
        let Type::Path(p) = t else { return None };
        Self::named(&p.path.get_ident()?.to_string())
    }
    pub(crate) fn tokens(self, root: &Path) -> TokenStream {
        let id = format_ident!("{}", self.names().1);
        emit!(#root::simd::Element::#id)
    }
    fn ty(self) -> Ident {
        Ident::new(self.names().0, Span::call_site())
    }
    fn index(self) -> Self {
        match self {
            Self::F32 => Self::I32,
            Self::F64 => Self::I64,
            x => x,
        }
    }
}
fn generic(seg: &syn::PathSegment) -> Option<Elem> {
    let PathArguments::AngleBracketed(a) = &seg.arguments else {
        return None;
    };
    element_argument(a)
}
fn element_argument(a: &syn::AngleBracketedGenericArguments) -> Option<Elem> {
    if a.args.len() != 1 {
        return None;
    }
    let GenericArgument::Type(t) = a.args.first()? else {
        return None;
    };
    Elem::parse(t)
}
fn splat_element(expr: &Expr) -> Option<Elem> {
    match expr {
        Expr::Paren(p) => splat_element(&p.expr),
        Expr::Group(g) => splat_element(&g.expr),
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => splat_element(&u.expr),
        Expr::Cast(c) => Elem::parse(&c.ty),
        Expr::Lit(l) => match &l.lit {
            syn::Lit::Int(v) => Elem::named(v.suffix()),
            syn::Lit::Float(v) => Elem::named(v.suffix()),
            _ => None,
        },
        _ => None,
    }
}
pub(super) fn vector_type(t: &Type) -> Option<Scalar> {
    let Type::Path(p) = t else { return None };
    let s = p.path.segments.last()?;
    if s.ident == "Lane" {
        Some(Scalar::Vector(generic(s)?))
    } else if s.ident == "Mask" {
        Some(Scalar::Mask(generic(s)?))
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
    pub(in crate::scalar) fn value_type(self, root: &Path) -> TokenStream {
        match self {
            Self::Vector(e) => {
                let e = e.tokens(root);
                emit!(#root::hir::ValueType::Vector(#e))
            }
            Self::Mask(e) => {
                let e = e.tokens(root);
                emit!(#root::hir::ValueType::Mask(#e))
            }
            _ => emit!(#root::hir::ValueType::Scalar),
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
        let elem = e.tokens(root);
        let res = emit!(__splice_builder.simd(#root::hir::Simd{element:#elem,op:#root::hir::SimdOp::#op},&__splice_operands)?);
        let args = emit!(let __splice_operands=[#(#inputs),*];);
        match output {
            Some(ty) => Value::new(
                Stage::Reg(ty),
                emit!({#args #res.ok_or("missing SIMD result")?}),
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
    fn offset(&self, v: Value, at: &Expr) -> Result<TokenStream> {
        if v.stage.is_host() {
            Ok(
                emit!({let index:usize=#v; __splice_builder.constant(i32::try_from(index).unwrap_or(-1))}),
            )
        } else {
            self.reg(v, Scalar::Index, at)
        }
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
        let explicit = generic(lane);
        if !matches!(lane.arguments, PathArguments::None) && explicit.is_none() {
            return err(lane, "expected Lane<T> with one of u16, i32, i64, f32, f64");
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
            .filter(|v| matches!(v.stage, Stage::Reg(Scalar::Int | Scalar::Inferred)))
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
        let ty = elem.ty();
        let mut lane_path = p.path.clone();
        lane_path.segments.pop();
        lane_path.segments.pop_punct();
        lane_path.segments.last_mut().unwrap().arguments =
            PathArguments::AngleBracketed(syn::parse_quote!(::<#ty>));
        let witness = emit!(let _: ::core::marker::PhantomData<#root::simd::Lane<#ty>>=::core::marker::PhantomData::<#lane_path>;);
        let get = |j: usize, t| self.simd_operand(&c.args[j], t);
        let mut pre = TokenStream::new();
        let mut result = match name.as_str() {
            "first_n" => self.vector_op(
                elem,
                emit!(FirstN),
                vec![self.offset(self.expr(&c.args[0])?, &c.args[0])?],
                Some(mask),
            ),
            "lanes" => self.vector_op(elem, emit!(Lanes), vec![], Some(Scalar::Length)),
            "iota" => self.vector_op(elem, emit!(Iota), vec![], Some(v)),
            "splat" => {
                let arg = runtime_splat.unwrap();
                if elem == Elem::I32 && matches!(arg.stage, Stage::Reg(_)) {
                    let reg = self.reg(arg, Scalar::Int, &c.args[0])?;
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
                        // Alignment depends only on low address bits, including for
                        // host offsets above the range representable by a JIT index.
                        let offset = if index.stage.is_host() {
                            emit!({let index:usize=#index; __splice_builder.constant(index as i32)})
                        } else {
                            self.reg(index, Scalar::Index, &c.args[1])?
                        };
                        self.vector_op(
                            elem,
                            emit!(AlignmentPrefix(buffer)),
                            vec![offset],
                            Some(Scalar::Length),
                        )
                    }
                    "alignment_prefix" => return err(c, "alignment_prefix expects i32 lanes"),
                    "load" | "load_select" => {
                        let mut a = vec![self.offset(self.expr(&c.args[1])?, &c.args[1])?];
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
                "Div" if matches!(e, Elem::F32 | Elem::F64) => (e, op, t),
                "And" | "Or" | "Xor" | "Not" if !matches!(e, Elem::F32 | Elem::F64) => (e, op, t),
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
        if t == Scalar::Vector(Elem::I32) && matches!(op, syn::BinOp::Shr(_)) {
            let shift = self.host(r, &at.right)?;
            return Ok(self.vector_op(
                Elem::I32,
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
            _ => {
                return err(
                    op,
                    "use simd_eq/lt/le/gt/ge/ne for element-wise comparisons",
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
        let is_mask = matches!(t, Scalar::Mask(_));
        let v = Scalar::Vector(e);
        let mask = Scalar::Mask(e);
        let idx = Scalar::Vector(e.index());
        let method = m.method.to_string();
        let args = &m.args;
        let op = |j: usize, t| self.simd_operand(&args[j], t);
        let recv = emit!(__splice_receiver);
        let mut pre = emit!(let __splice_receiver=#receiver;);
        let mut value = match (method.as_str(), args.len(), is_mask) {
            (
                "interleave_low" | "interleave_high" | "deinterleave_even" | "deinterleave_odd",
                1,
                false,
            ) if e == Elem::I32 => {
                let order = format_ident!(
                    "{}",
                    match method.as_str() {
                        "interleave_low" => "InterleaveLow",
                        "interleave_high" => "InterleaveHigh",
                        "deinterleave_even" => "DeinterleaveEven",
                        _ => "DeinterleaveOdd",
                    }
                );
                let root = self.root;
                self.vector_op(
                    e,
                    emit!(Shuffle(#root::simd::Shuffle::#order)),
                    vec![recv, op(0, v)?],
                    Some(v),
                )
            }
            ("compress", 1, false) if e == Elem::I32 => {
                self.vector_op(e, emit!(Compress), vec![recv, op(0, mask)?], Some(v))
            }
            ("expand", 2, false) if e == Elem::I32 => self.vector_op(
                e,
                emit!(Expand),
                vec![recv, op(0, mask)?, op(1, v)?],
                Some(v),
            ),
            ("swizzle", 1, false) if e == Elem::I32 => {
                self.vector_op(e, emit!(Swizzle), vec![recv, op(0, v)?], Some(v))
            }
            ("prefix_sum", 0, false) if e == Elem::I32 => {
                self.vector_op(e, emit!(PrefixSum), vec![recv], Some(v))
            }
            ("count", 0, true) if e == Elem::I32 => {
                self.vector_op(e, emit!(Count), vec![recv], Some(Scalar::Int))
            }
            ("extract_wrapping", 1, false) if e == Elem::I32 => self.vector_op(
                e,
                emit!(ExtractWrapping),
                vec![recv, op(0, Scalar::Int)?],
                Some(Scalar::Int),
            ),
            ("replace_wrapping", 2, false) if e == Elem::I32 => self.vector_op(
                e,
                emit!(ReplaceWrapping),
                vec![recv, op(0, Scalar::Int)?, op(1, Scalar::Int)?],
                Some(v),
            ),
            ("last", 0, false) if e == Elem::I32 => {
                self.vector_op(e, emit!(Last), vec![recv], Some(Scalar::Int))
            }
            ("any" | "all", 0, true) => {
                let f = if method == "any" {
                    emit!(Any)
                } else {
                    emit!(All)
                };
                self.vector_op(e, f, vec![recv], Some(Scalar::Bool))
            }
            ("select", 2, true) => {
                let data_type = m
                    .turbofish
                    .as_ref()
                    .and_then(element_argument)
                    .map(Scalar::Vector);
                let (first, second) =
                    self.paired_operands(&args[0], &args[1], data_type.or(hint), Some(v))?;
                let Stage::Reg(ty @ Scalar::Vector(element)) = first.stage else {
                    return err(m, "mask.select requires two vectors");
                };
                if element.index() != e.index() {
                    return err(m, "mask and data must have the same lane count");
                }
                self.vector_op(
                    element,
                    emit!(Select),
                    vec![recv, first.code, self.reg(second, ty, &args[1])?],
                    Some(ty),
                )
            }
            ("simd_eq" | "simd_ne" | "simd_lt" | "simd_le" | "simd_gt" | "simd_ge", 1, false) => {
                let name = match method.as_str() {
                    "simd_eq" => "Eq",
                    "simd_ne" => "Ne",
                    "simd_lt" => "Lt",
                    "simd_le" => "Le",
                    "simd_gt" => "Gt",
                    _ => "Ge",
                };
                self.simd_operator(t, name, vec![recv, op(0, v)?], m)?
            }
            ("add_masked", 3, false) => {
                let (a, b, f) = (op(0, mask)?, op(1, v)?, op(2, v)?);
                pre.extend(emit!(let (mask,rhs,fallback)=(#a,#b,#f);));
                let sum = self.vector_op(e, emit!(Add), vec![recv, emit!(rhs)], Some(v));
                self.vector_op(
                    e,
                    emit!(Select),
                    vec![emit!(mask), sum.code, emit!(fallback)],
                    Some(v),
                )
            }
            ("store" | "scatter", 2, false)
            | ("store_select" | "scatter_select" | "compress_store", 3, false) => {
                let buf = self.expr(&args[0])?;
                let Stage::Buffer(b) = buf.stage else {
                    return err(m, "expected a mutable runtime slice");
                };
                if !b.write || b.elem != e {
                    return err(
                        m,
                        "SIMD store requires a writable slice of the vector element type",
                    );
                }
                pre.extend(emit!(let buffer=#buf;));
                match method.as_str() {
                    "store" => self.vector_op(
                        e,
                        emit!(Store(buffer)),
                        vec![self.offset(self.expr(&args[1])?, &args[1])?, recv],
                        None,
                    ),
                    "compress_store" if e != Elem::I32 => {
                        return err(m, "compress_store requires i32 lanes");
                    }
                    "store_select" | "compress_store" => self.vector_op(
                        e,
                        if method == "compress_store" {
                            emit!(CompressStore(buffer))
                        } else {
                            emit!(StoreSelect(buffer))
                        },
                        vec![
                            self.offset(self.expr(&args[1])?, &args[1])?,
                            op(2, mask)?,
                            recv,
                        ],
                        None,
                    ),
                    "scatter" => {
                        self.vector_op(e, emit!(Scatter(buffer)), vec![op(1, idx)?, recv], None)
                    }
                    _ => {
                        let m = op(1, mask)?;
                        pre.extend(emit!(let mask=#m;));
                        self.vector_op(
                            e,
                            emit!(ScatterSelect(buffer)),
                            vec![op(2, idx)?, emit!(mask), recv],
                            None,
                        )
                    }
                }
            }
            _ => return err(m, "unknown SIMD method or wrong operand count"),
        };
        let code = value.code;
        value.code = emit!({#pre #code});
        Ok(value)
    }
}
