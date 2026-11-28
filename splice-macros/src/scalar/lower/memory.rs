//! Runtime slices carry an immutable buffer identity through specialization.
//! Only lengths, element accesses, and scalar computations become instructions.
use super::{Lower, Scalar, Stage, Value, err};
use crate::kast::types::Primitive;
use proc_macro2::TokenStream;
use splice_core::hir::MathOp;
use syn::{BinOp, Expr, ExprCast, ExprIndex, ExprPath, ExprReference, Result, UnOp};

pub(super) fn math_op(name: &str) -> Option<MathOp> {
    Some(match name {
        "count_ones" => MathOp::CountOnes,
        "leading_zeros" => MathOp::LeadingZeros,
        "trailing_zeros" => MathOp::TrailingZeros,
        "rotate_left" => MathOp::RotateLeft,
        _ => return None,
    })
}

pub(super) fn integer_owner(path: &ExprPath) -> Option<Scalar> {
    use crate::kast::types::Primitive;
    let ty = match &path.qself {
        Some(q) if q.as_token.is_none() => (*q.ty).clone(),
        Some(_) => return None,
        None => {
            let owner = super::simd::owner_path(path)?;
            syn::parse_quote!(#owner)
        }
    };
    let primitive = Primitive::of(&ty)?;
    match primitive {
        Primitive::I8
        | Primitive::U8
        | Primitive::I16
        | Primitive::U16
        | Primitive::I32
        | Primitive::U32
        | Primitive::I64
        | Primitive::U64 => Scalar::of(primitive),
        _ => None,
    }
}

pub(super) fn primitive_method(name: &str) -> bool {
    math_op(name).is_some()
        || matches!(
            name,
            "wrapping_add"
                | "wrapping_sub"
                | "wrapping_mul"
                | "wrapping_neg"
                | "checked_add"
                | "checked_sub"
                | "checked_mul"
                | "checked_div"
                | "checked_rem"
                | "saturating_add"
                | "saturating_sub"
                | "saturating_mul"
                | "min"
                | "max"
        )
}

/// The core methods of runtime integers (`frontend::methods`): each stages
/// a portable operation.
const CORE_METHODS: [&str; 9] = [
    "wrapping_add",
    "wrapping_sub",
    "wrapping_mul",
    "wrapping_neg",
    "saturating_add",
    "saturating_sub",
    "saturating_mul",
    "min",
    "max",
];

impl Lower<'_> {
    /// The method call `e`.
    pub(super) fn method(&self, e: &Expr, hint: Option<Scalar>) -> Result<Value> {
        let Expr::MethodCall(m) = e else {
            unreachable!("a method call")
        };
        if let Expr::Index(index) = &*m.receiver
            && let Expr::Path(table) = &*index.expr
            && matches!(self.binding(table), Some(super::Binding::Columns))
        {
            if m.method != "as_slice" || !m.args.is_empty() {
                return err(m, "use as_slice::<T>() to bind a typed column");
            }
            let Some(elem) = m
                .turbofish
                .as_ref()
                .and_then(|args| super::simd::element_argument(args, self.types))
            else {
                return err(
                    m,
                    "as_slice::<T>() requires a slice element: a kernel scalar other than bool",
                );
            };
            let slot = self.host(self.expr(&index.index)?, &index.index)?;
            let root = self.root;
            let element = elem.tokens(root, self.types);
            return Ok(Value::new(
                Stage::Buffer(super::BufferType {
                    elem,
                    write: false,
                    utf8: false,
                    slice_of: None,
                }),
                emit!(#root::frontend::scalar::column(#table, #slot, #element)?),
            ));
        }
        if let Expr::MethodCall(chunks) = &*m.receiver
            && (m.method == "remainder" || m.method == "into_remainder")
            && m.args.is_empty()
            && chunks.args.len() == 1
        {
            let write = chunks.method == "chunks_exact_mut";
            if ((write && m.method == "into_remainder")
                || (chunks.method == "chunks_exact" && m.method == "remainder"))
                && let Some((slice, actual, size)) = self.chunks_exact(chunks, write)?
            {
                return Ok(Value::new(
                    Stage::Buffer(super::BufferType { write, ..actual }),
                    emit!({
                        let __splice_slice = #slice;
                        let __splice_size = #size;
                        __splice_builder.chunks_exact_remainder(__splice_slice, __splice_size)?
                    }),
                ));
            }
        }
        let hint_for_receiver = self.store_hint(&m.method, m.args.first())?;
        let mut receiver = self.expr_with_hint(&m.receiver, hint_for_receiver)?;
        if receiver.stage == Stage::Reg(Scalar::Inferred)
            && let Some(t @ Scalar::Vector(_)) = hint_for_receiver
        {
            receiver.stage = Stage::Reg(t);
        }

        if let Stage::Reg(t @ (Scalar::Vector(_) | Scalar::Mask(_))) = receiver.stage {
            let name = m.method.to_string();
            if !primitive_method(&name) || super::simd::has_method(t, &name) {
                return self.simd_method(t, receiver, m, hint);
            }
            // A constructor with the same spelling can return an ordinary
            // scalar. Rust checked this scalar method on the original result.
            receiver.stage = Stage::Reg(Scalar::Inferred);
        }
        if (m.method == "len" || m.method == "is_empty")
            && m.args.is_empty()
            && matches!(receiver.stage, Stage::Buffer(_))
        {
            let length = emit!({let buffer=#receiver; buffer.len});
            return Ok(if m.method == "len" {
                Value::new(Stage::Reg(Scalar::Length), length)
            } else {
                Value::new(
                    Stage::Reg(Scalar::Bool),
                    self.bin("Eq", length, self.constant(emit!(0usize))),
                )
            });
        }
        if let Stage::Buffer(text @ super::BufferType { utf8: true, .. }) = receiver.stage {
            let name = m.method.to_string();
            match (name.as_str(), m.args.len()) {
                ("as_bytes", 0) => {
                    return Ok(Value::new(
                        Stage::Buffer(super::BufferType {
                            utf8: false,
                            ..text
                        }),
                        receiver.code,
                    ));
                }
                ("is_char_boundary", 1) => {
                    let index = self.expr(&m.args[0])?;
                    let index = self.reg(index, Scalar::Index, &m.args[0])?;
                    return Ok(Value::new(
                        Stage::Reg(Scalar::Bool),
                        emit!({
                            let __splice_text = #receiver;
                            let __splice_index = #index;
                            __splice_builder.is_char_boundary(__splice_text, __splice_index)?
                        }),
                    ));
                }
                ("len" | "is_empty", 0) => {}
                ("chars" | "char_indices", 0) => {
                    return err(
                        &m.method,
                        &format!(
                            "`str::{name}` is not supported in kernels: decoding a char needs Option"
                        ),
                    );
                }
                ("bytes", 0) => {
                    return err(
                        m,
                        "iterate a str's bytes in a loop: `for b in text.bytes()`",
                    );
                }
                _ => {
                    return err(
                        &m.method,
                        &format!("`str::{name}` is not supported in kernels; use `as_bytes()`"),
                    );
                }
            }
        }
        if matches!(receiver.stage, Stage::Buffer(_)) {
            let name = m.method.to_string();
            if matches!(
                name.as_str(),
                "first"
                    | "last"
                    | "get"
                    | "first_mut"
                    | "last_mut"
                    | "get_mut"
                    | "split_first"
                    | "split_last"
                    | "split_first_mut"
                    | "split_last_mut"
            ) {
                return err(
                    &m.method,
                    &format!("`{name}` returns an Option, which kernels do not support yet"),
                );
            }
            if name.starts_with("chunks_exact") {
                return err(
                    m,
                    &format!(
                        "iterate `for chunk in xs.{name}(n)` or take `xs.{name}(n).{}()`; a chunk iterator cannot be bound",
                        if name == "chunks_exact" {
                            "remainder"
                        } else {
                            "into_remainder"
                        }
                    ),
                );
            }
            if name == "split_at" || name == "split_at_mut" {
                return err(
                    m,
                    &format!(
                        "bind the halves of `{name}` with `let (head, tail) = slice.{name}(mid);`"
                    ),
                );
            }
        }
        let kind = match receiver.stage {
            Stage::Reg(t) | Stage::Word(t)
                if t.is_number()
                    || t == Scalar::Inferred
                    || t == Scalar::Number(crate::kast::types::Primitive::Char) =>
            {
                t.base()
            }
            // A host receiver with runtime arguments, such as `k.min(x)`,
            // takes the arguments' type.
            stage if stage.is_host() => {
                if m.args.is_empty() {
                    return err(m, "a host method cannot consume runtime values");
                }
                Scalar::Inferred
            }
            _ => {
                return err(
                    m,
                    "expected slice.len/is_empty or an integer method such as min, max, wrapping_add or saturating_add",
                );
            }
        };
        let root = self.root;
        let name = m.method.to_string();
        if let Some(op) = math_op(&name) {
            return self.math_call(op, receiver, kind, &m.receiver, m.args.iter());
        }
        if !CORE_METHODS.contains(&name.as_str()) {
            return err(
                &m.method,
                &format!("unsupported kernel scalar method '{name}'"),
            );
        }
        let receiver = self.reg(receiver, kind, &m.receiver)?;
        let args = m
            .args
            .iter()
            .map(|arg| {
                // A literal argument of a receiver whose type only the
                // builder knows (a struct field) takes the receiver's type.
                if kind == Scalar::Inferred && matches!(arg, Expr::Lit(_)) {
                    let value = self.expr(arg)?;
                    let like = match value.stage {
                        Stage::Literal(t) if t.is_float() => Some(emit!(float_like)),
                        Stage::Literal(t) if t.is_number() => Some(emit!(int_like)),
                        _ => None,
                    };
                    if let Some(like) = like {
                        return Ok(emit!(#root::frontend::scalar::#like(__splice_builder, __splice_receiver, #value)?));
                    }
                }
                self.operand(arg, kind)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Value::new(
            Stage::Reg(kind),
            emit!({
                let __splice_receiver = #receiver;
                let operands = [__splice_receiver, #(#args),*];
                #root::frontend::methods::method(__splice_builder, #name, &operands)?
            }),
        ))
    }

    pub(super) fn qualified_math(
        &self,
        c: &syn::ExprCall,
        path: &syn::ExprPath,
    ) -> Result<Option<Value>> {
        if self.local_owner(path) {
            return Ok(None);
        }
        let Some(member) = path.path.segments.last() else {
            return Ok(None);
        };
        let Some(op) = math_op(&member.ident.to_string()) else {
            return Ok(None);
        };
        let Some(kind) = integer_owner(path) else {
            return Ok(None);
        };
        let mut args = c.args.iter();
        let first = args
            .next()
            .ok_or_else(|| syn::Error::new_spanned(member, "missing integer intrinsic receiver"))?;
        let receiver = self.expr_with_hint(first, Some(kind))?;
        self.math_call(op, receiver, kind, first, args).map(Some)
    }

    fn math_call<'a>(
        &self,
        op: MathOp,
        receiver: Value,
        kind: Scalar,
        at: &Expr,
        args: impl ExactSizeIterator<Item = &'a Expr>,
    ) -> Result<Value> {
        let root = self.root;
        let (name, result) = match op {
            MathOp::CountOnes => ("CountOnes", Scalar::Number(Primitive::U32)),
            MathOp::LeadingZeros => ("LeadingZeros", Scalar::Number(Primitive::U32)),
            MathOp::TrailingZeros => ("TrailingZeros", Scalar::Number(Primitive::U32)),
            MathOp::RotateLeft => ("RotateLeft", kind),
        };
        if !matches!(
            kind,
            Scalar::Inferred
                | Scalar::Number(
                    Primitive::I8
                        | Primitive::U8
                        | Primitive::I16
                        | Primitive::U16
                        | Primitive::I32
                        | Primitive::U32
                        | Primitive::I64
                        | Primitive::U64
                )
        ) {
            return err(
                at,
                "unsupported intrinsic receiver: expected i8, u8, i16, u16, i32, u32, i64 or u64",
            );
        }
        if args.len() != usize::from(op == MathOp::RotateLeft) {
            return err(at, "wrong number of integer intrinsic arguments");
        }
        let receiver = self.reg(receiver, kind, at)?;
        let count = args
            .map(|arg| self.operand(arg, Scalar::Number(Primitive::U32)))
            .collect::<Result<Vec<_>>>()?;
        let op = quote::format_ident!("{name}");
        Ok(Value::new(
            Stage::Reg(result),
            emit!({
                let args = [#receiver, #(#count),*];
                let ::core::option::Option::Some(#root::hir::ValueType::Scalar(ty)) = __splice_builder.reg_type(args[0]) else {
                    return ::core::result::Result::Err(#root::BuildError::staging("expected integer intrinsic receiver"));
                };
                __splice_builder.call_intrinsic(#root::hir::Intrinsic::Math { op: #root::hir::MathOp::#op, ty }, &args)?
                    .ok_or_else(|| #root::BuildError::staging("missing integer intrinsic result"))?
            }),
        ))
    }

    pub(super) fn cast(&self, cast: &ExprCast) -> Result<Value> {
        let value = self.expr(&cast.expr)?;
        // rustc checked the cast: to a number, or `u8` to `char`.
        let Some(kind) = Scalar::from_type(&cast.ty, self.types)
            .filter(|t| t.is_number() || *t == Scalar::Number(crate::kast::types::Primitive::Char))
        else {
            return err(&cast.ty, "a runtime cast converts to a number type or char");
        };
        let (Stage::Reg(scalar) | Stage::Word(scalar)) = value.stage else {
            return err(cast, "expected a runtime scalar to cast");
        };
        if !scalar.is_scalar() && scalar != Scalar::Inferred {
            return err(cast, "expected a runtime scalar to cast");
        }
        let value = self.reg(value, scalar, cast)?;
        let ty = self.scalar_type(kind);
        Ok(Value::new(
            Stage::Reg(kind),
            emit!({let __splice_value = #value; __splice_builder.cast(__splice_value, #ty)?}),
        ))
    }

    pub(super) fn reference(&self, reference: &ExprReference) -> Result<Value> {
        // Explicit reborrows (&*input and &mut *output) keep the same descriptor.
        let target = match &*reference.expr {
            Expr::Unary(x) if matches!(x.op, UnOp::Deref(_)) => &*x.expr,
            expr => expr,
        };
        let writable = reference.mutability.is_some();
        if let Expr::Index(index) = target
            && let Expr::Range(range) = &*index.index
        {
            return self.subslice(reference, index, range, writable);
        }
        let value = self.expr(target)?;
        match value.stage {
            Stage::Buffer(actual) if !writable || actual.write => Ok(Value::new(
                Stage::Buffer(super::BufferType {
                    write: writable,
                    ..actual
                }),
                value.code,
            )),
            _ => err(reference, "only a runtime slice can be reborrowed here"),
        }
    }

    /// `&xs[range]` or `&mut xs[range]`: a subslice, which traps as Rust's
    /// slice index panics unless the range lies in the slice.
    fn subslice(
        &self,
        reference: &ExprReference,
        index: &ExprIndex,
        range: &syn::ExprRange,
        writable: bool,
    ) -> Result<Value> {
        let slice = self.expr(&index.expr)?;
        let Stage::Buffer(actual) = slice.stage else {
            return err(&index.expr, "only a runtime slice can be sliced here");
        };
        if writable && !actual.write {
            return err(reference, "a shared slice cannot be reborrowed as mutable");
        }
        let ty = Stage::Buffer(super::BufferType {
            write: writable,
            ..actual
        });
        let take = if actual.utf8 {
            emit!(str_subslice)
        } else {
            emit!(subslice)
        };
        let bound = |e: &Expr| -> Result<TokenStream> {
            let value = self.expr(e)?;
            self.reg(value, Scalar::Index, e)
        };
        let start = range.start.as_deref().map(bound).transpose()?;
        let end = range.end.as_deref().map(bound).transpose()?;
        if start.is_none() && end.is_none() {
            return Ok(Value::new(ty, slice.code));
        }
        let start = start.unwrap_or_else(|| self.constant(emit!(0usize)));
        let inclusive = matches!(range.limits, syn::RangeLimits::Closed(_));
        let end = end.unwrap_or_else(|| emit!(__splice_slice.len));
        Ok(Value::new(
            ty,
            emit!({
                let __splice_slice = #slice;
                let __splice_start = #start;
                let __splice_end = #end;
                __splice_builder.#take(__splice_slice, __splice_start, __splice_end, #inclusive)?
            }),
        ))
    }

    /// A runtime slice element: whether the slice is writable, the slice,
    /// the offset and the element's register type.
    fn index_operands(
        &self,
        index: &ExprIndex,
    ) -> Result<(bool, TokenStream, TokenStream, Scalar)> {
        let slice = self.expr(&index.expr)?;
        let Stage::Buffer(writable) = slice.stage else {
            return err(&index.expr, "index a runtime slice parameter");
        };
        if writable.utf8 {
            return err(index, "a str has no integer index; index its `as_bytes()`");
        }
        let Some(element) = writable.elem.scalar() else {
            return err(
                index,
                "a u16 element has no scalar register; use Lane::load or Lane::store",
            );
        };
        let value = self.expr(&index.index)?;
        // An out-of-bounds host index faults on the executed path, even
        // for a literal.
        let offset = self.reg(value, Scalar::Index, &index.index)?;
        Ok((writable.write, slice.code, offset, element))
    }

    pub(super) fn index(&self, index: &ExprIndex) -> Result<Value> {
        // A known element is a known value; a known slice with an index
        // the generator does not know has no runtime element.
        if self.expr(&index.expr)?.stage.is_host() {
            let root = self.root;
            return Ok(Value::new(
                Stage::Reg(Scalar::I32),
                emit!(
                    ::core::result::Result::Err::<#root::hir::Reg, #root::BuildError>(#root::BuildError::staging("runtime index requires a runtime slice"))?
                ),
            ));
        }
        let slice = self.expr(&index.expr)?;
        if let Stage::Buffer(outer) = slice.stage
            && let Some(inner) = outer.inner()
        {
            // `xs[i]` of a `&[&[T]]` is a `&[T]` (a `&str` of a `&[&str]`):
            // both halves of the element, read together.
            let value = self.expr(&index.index)?;
            let offset = self.reg(value, Scalar::Index, &index.index)?;
            let slice = slice.code;
            return Ok(Value::new(
                Stage::Buffer(inner),
                emit!({
                    let (buffer, __splice_index) = (#slice, #offset);
                    __splice_builder.load_slice(buffer, __splice_index)?
                }),
            ));
        }
        let (_, slice, offset, element) = self.index_operands(index)?;

        Ok(Value::new(
            Stage::Reg(element),
            emit!({
                let (buffer, __splice_index) = (#slice, #offset);
                __splice_builder.load(buffer, __splice_index)?
            }),
        ))
    }

    pub(super) fn bounds_failure(&self) -> TokenStream {
        if let Some(label) = &self.source_fault {
            let root = self.root;
            emit! {
                __splice_source.transfer = ::core::option::Option::Some(#root::frontend::source::Transfer::Return);
                break #label r#"panic!("kernel slice index out of bounds")"#.to_owned();
            }
        } else {
            let region = self.region();
            emit! { __splice_builder.trap_bounds()?; break #region; }
        }
    }

    /// The element of a known slice, copied while the receiver (possibly a
    /// temporary, such as an evaluator input) lives.
    pub(super) fn known_index(&self, receiver: TokenStream, offset: TokenStream) -> TokenStream {
        self.read_known_index(receiver, offset, emit!(?.copied()))
    }

    pub(super) fn known_index_reference(
        &self,
        receiver: TokenStream,
        offset: TokenStream,
    ) -> TokenStream {
        self.read_known_index(receiver, offset, TokenStream::new())
    }

    fn read_known_index(
        &self,
        receiver: TokenStream,
        offset: TokenStream,
        then: TokenStream,
    ) -> TokenStream {
        let root = self.root;
        emit!({
            use #root::frontend::evaluation::{Operand, ReadIndex as _};
            let receiver = &(#receiver);
            let index = #offset;
            (&Operand::of(receiver)).read_index(receiver, index, |values, index| &values[index]) #then
        })
    }

    pub(super) fn store(
        &self,
        index: &ExprIndex,
        rhs: &Expr,
        op: Option<(&str, &BinOp)>,
    ) -> Result<Value> {
        let (writable, slice, offset, element) = self.index_operands(index)?;
        if !writable {
            return err(
                index,
                "writing a kernel slice requires a &mut slice parameter",
            );
        }
        let value = self.operand(rhs, element)?;

        let combine = op.map(|(op, _)| {
            let combined = self.operator_bin(
                op,
                emit!(__splice_builder.load(buffer, __splice_index)?),
                emit!(__splice_value),
            );
            emit!(let __splice_value=#combined;)
        });
        // Rust evaluates the RHS before the place for assignment and primitive
        // compound assignment. The index expression may itself mutate memory.
        Ok(Value::unit(emit!({
            let __splice_value = #value;
            let (buffer, __splice_index) = (#slice, #offset);
            #combine
            __splice_builder.store(buffer, __splice_index, __splice_value)?;
        })))
    }
}
