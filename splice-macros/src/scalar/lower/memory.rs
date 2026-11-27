//! Runtime slices carry an immutable buffer identity through specialization.
//! Only lengths, element accesses, and scalar computations become instructions.
use super::{Lower, Scalar, Stage, Value, err};
use proc_macro2::TokenStream;
use syn::{BinOp, Expr, ExprCast, ExprIndex, ExprReference, Result, UnOp};

pub(super) fn primitive_method(name: &str) -> bool {
    matches!(
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
/// a portable operation. Any other method of a runtime scalar is an
/// intrinsic.
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
                    "as_slice::<T>() requires u16, u32, i32, i64, f32, or f64",
                );
            };
            let slot = self.host(self.expr(&index.index)?, &index.index)?;
            let root = self.root;
            let element = elem.tokens(root, self.types);
            return Ok(Value::new(
                Stage::Buffer(super::BufferType { elem, write: false }),
                emit!(#root::frontend::scalar::column(#table, #slot, #element)?),
            ));
        }
        let data_type = self.simd_store_receiver(m)?;
        let receiver = self.expr_with_hint(&m.receiver, data_type)?;

        if let Stage::Reg(t @ (Scalar::Vector(_) | Scalar::Mask(_))) = receiver.stage {
            return self.simd_method(t, receiver, m, hint);
        }
        if (m.method == "len" || m.method == "is_empty")
            && m.args.is_empty()
            && matches!(receiver.stage, Stage::Buffer(_))
        {
            let length = emit!({let buffer=#receiver; __splice_builder.buffer_len(buffer)?});
            return Ok(if m.method == "len" {
                Value::new(Stage::Reg(Scalar::Length), length)
            } else {
                Value::new(
                    Stage::Reg(Scalar::Bool),
                    self.bin("Eq", length, self.constant(emit!(0usize))),
                )
            });
        }
        let kind = match receiver.stage {
            Stage::Reg(t) | Stage::Word(t) if t.is_number() || t == Scalar::Inferred => t.base(),
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
        if !CORE_METHODS.contains(&name.as_str())
            && let Some(ty) = kind.rust_type()
        {
            return self.intrinsic_method(ty, receiver, kind, m, hint);
        }
        // The registry stages the method on the receiver's type; the name
        // must be registered for some type.
        let receiver = self.reg(receiver, kind, &m.receiver)?;
        let args = m
            .args
            .iter()
            .map(|arg| self.operand(arg, kind))
            .collect::<Result<Vec<_>>>()?;
        let message = format!(
            "`{name}` is not a kernel operation on runtime integers; an intrinsic method needs a receiver of a known type"
        );
        Ok(Value::new(
            Stage::Reg(kind),
            emit!({
                const { ::core::assert!(#root::frontend::methods::is_method(#name), #message) };
                let operands = [#receiver, #(#args),*];
                #root::frontend::methods::method(__splice_builder, #name, &operands)?
            }),
        ))
    }

    /// `receiver.method(args)` on a runtime scalar, where `method` is not a
    /// core method: an intrinsic of an `#[intrinsic]` trait in scope.
    /// Rust's method resolution selects the trait's stage entry for the
    /// receiver's type; the type the use expects, when it is known, types the result.
    fn intrinsic_method(
        &self,
        ty: proc_macro2::Ident,
        receiver: Value,
        kind: Scalar,
        m: &syn::ExprMethodCall,
        hint: Option<Scalar>,
    ) -> Result<Value> {
        let root = self.root;
        let operand = emit!(#root::frontend::methods::Operand);
        let receiver = self.reg(receiver, kind, &m.receiver)?;
        let args = m
            .args
            .iter()
            .map(|arg| Ok(self.simd_argument(arg, Scalar::Inferred)?.1))
            .collect::<Result<Vec<_>>>()?;
        let stage = quote::format_ident!(
            "__splice_stage_{}",
            m.method.to_string().trim_start_matches("r#"),
            span = m.method.span()
        );
        // rustc checks the result against the type the use expects, so a
        // concrete expected type is the intrinsic's result type.
        let result = hint
            .filter(|t| t.is_scalar() && t.rust_type().is_some())
            .unwrap_or(Scalar::Inferred);
        Ok(Value::new(
            Stage::Reg(result),
            emit!({
                let __splice_receiver = #receiver;
                let __splice_operands = [#operand::Reg(__splice_receiver), #(#args),*];
                let __splice_probe: #ty = ::core::default::Default::default();
                __splice_probe.#stage(__splice_builder, &__splice_operands)?
            }),
        ))
    }

    pub(super) fn cast(&self, cast: &ExprCast) -> Result<Value> {
        let value = self.expr(&cast.expr)?;
        let Some(kind) = Scalar::from_type(&cast.ty, self.types).filter(|t| t.is_number()) else {
            return err(&cast.ty, "a runtime cast converts to a number type");
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
        let value = self.expr(target)?;
        let writable = reference.mutability.is_some();
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
