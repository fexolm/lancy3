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
        let data_type = if matches!(
            m.method.to_string().as_str(),
            "store" | "store_select" | "scatter" | "scatter_select"
        ) && let Some(dst) = m.args.first()
            && let Stage::Buffer(buffer) = self.expr(dst)?.stage
        {
            Some(Scalar::Vector(buffer.elem))
        } else {
            None
        };
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
        if receiver.stage.is_host() {
            let args = m
                .args
                .iter()
                .map(|arg| self.expr(arg))
                .collect::<Result<Vec<_>>>()?;
            if args.iter().all(|arg| arg.stage.is_host()) {
                let root = self.root;
                let name = &m.method;
                let raw_name = name.to_string();
                if matches!(raw_name.as_str(), "len" | "is_empty") && args.is_empty() {
                    let empty = (name == "is_empty").then(|| emit!(== 0));
                    return Ok(Value::new(
                        Stage::Host,
                        emit!({
                            use #root::frontend::evaluation::{ReadLength as _, Operand};
                            let receiver = &(#receiver);
                            (&Operand::of(receiver)).length(receiver)? #empty
                        }),
                    ));
                }
                if primitive_method(&raw_name) {
                    return Ok(Value::new(Stage::Host, {
                        let (value, arg) =
                            (self.primitive(emit!(&value)), self.primitive(emit!(&arg)));
                        emit!({
                            let value = #receiver;
                            #value
                            value.#name(#({let arg = #args; #arg arg}),*)
                        })
                    }));
                }
                return Ok(Value::new(Stage::Host, self.required(e, hint)?));
            }
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
        // The registry stages the method on the receiver's type; the name
        // must be registered for some type.
        let root = self.root;
        let name = m.method.to_string();
        let receiver = self.reg(receiver, kind, &m.receiver)?;
        let args = m
            .args
            .iter()
            .map(|arg| self.operand(arg, kind))
            .collect::<Result<Vec<_>>>()?;
        let message = format!("`{name}` is not a kernel operation on runtime integers");
        Ok(Value::new(
            Stage::Reg(kind),
            emit!({
                const { ::core::assert!(#root::frontend::intrinsics::is_method(#name), #message) };
                let operands = [#receiver, #(#args),*];
                #root::frontend::intrinsics::method(__splice_builder, #name, &operands)?
            }),
        ))
    }

    pub(super) fn cast(&self, cast: &ExprCast) -> Result<Value> {
        if Scalar::from_type(&cast.ty, self.types) == Some(Scalar::I32) {
            let mut operand = cast.expr.as_ref();
            loop {
                operand = match operand {
                    Expr::Paren(expr) => &expr.expr,
                    Expr::Group(expr) => &expr.expr,
                    _ => break,
                };
            }
            if let Expr::Call(call) = operand
                && matches!(call.func.as_ref(), Expr::Path(path) if crate::scalar::methods::trait_path(path).is_some())
            {
                return self.call(call, Some(Scalar::I32), Some(&cast.ty));
            }
        }
        let value = self.expr(&cast.expr).or_else(|error| {
            if super::simd::uses_lane(&cast.expr) {
                return Err(error);
            }
            self.host_raw(&cast.expr)
                .map(|code| Value::new(Stage::Host, code))
                .map_err(|_| error)
        })?;
        if value.stage.is_host() {
            let ty = &cast.ty;
            return Ok(Value::new(Stage::Host, emit!((#value) as #ty)));
        }
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
        let receiver = self.expr(&index.expr)?;
        if receiver.stage.is_host() {
            let offset = self.expr(&index.index)?;
            if !offset.stage.is_host() {
                let root = self.root;
                return Ok(Value::new(
                    Stage::Reg(Scalar::I32),
                    emit!(
                        ::core::result::Result::Err::<#root::hir::Reg, #root::BuildError>(#root::BuildError::staging("runtime index requires a runtime slice"))?
                    ),
                ));
            }
            let read = self.known_index(receiver.code, offset.code);
            let fail = self.bounds_failure();
            return Ok(Value::new(
                Stage::Host,
                emit!({
                    let ::core::option::Option::Some(value) = (#read) else { #fail };
                    value
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
                element,
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
