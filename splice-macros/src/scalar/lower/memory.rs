//! Runtime slices carry an immutable buffer identity through specialization.
//! Only lengths, element accesses, and scalar computations become instructions.
use super::{Lower, Scalar, Stage, Value, err};
use proc_macro2::TokenStream;
use syn::{BinOp, Expr, ExprCast, ExprIndex, ExprMethodCall, ExprReference, Result, UnOp};

impl Lower<'_> {
    pub(super) fn method(&self, m: &ExprMethodCall, hint: Option<Scalar>) -> Result<Value> {
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
            let length = emit!({let buffer=#receiver; __splice_builder.buffer_len(buffer)});
            return Ok(if m.method == "len" {
                Value::new(Stage::Reg(Scalar::Length), length)
            } else {
                Value::new(
                    Stage::Reg(Scalar::Bool),
                    self.bin("Eq", length, emit!(__splice_builder.constant(0))),
                )
            });
        }
        let binary = match m.method.to_string().as_str() {
            "min" => Some("Min"),
            "max" => Some("Max"),
            "wrapping_add" => Some("Add"),
            "wrapping_sub" => Some("Sub"),
            "wrapping_mul" => Some("Mul"),
            _ => None,
        };
        if let Some(op) = binary
            && m.args.len() == 1
        {
            let rhs = &m.args[0];
            let left = self.reg(receiver, Scalar::Int, &m.receiver)?;
            let right = self.operand(rhs, Scalar::Int)?;
            return Ok(Value::new(
                Stage::Reg(Scalar::Int),
                self.bin(op, left, right),
            ));
        }
        if receiver.stage.is_host() {
            return err(m, "a host method cannot consume runtime values");
        }
        err(
            m,
            "expected slice.len/is_empty, i32.min/max, or i32.wrapping_add/sub/mul",
        )
    }

    pub(super) fn cast(&self, cast: &ExprCast) -> Result<Value> {
        let value = self.expr(&cast.expr)?;
        let Some(kind @ (Scalar::Int | Scalar::Index)) = Scalar::from_type(&cast.ty, self.types)
        else {
            return err(
                &cast.ty,
                "runtime casts support i32 and usize slice indices",
            );
        };
        let code = match value.stage {
            Stage::Reg(Scalar::Index | Scalar::Length | Scalar::Int) => value.code,
            Stage::Reg(Scalar::Bool) => {
                self.bin("And", value.code, emit!(__splice_builder.constant(1)))
            }
            _ => return err(cast, "expected an integer or boolean to cast"),
        };
        Ok(Value::new(Stage::Reg(kind), code))
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

    fn index_operands(&self, index: &ExprIndex) -> Result<(bool, TokenStream, TokenStream)> {
        let slice = self.expr(&index.expr)?;
        let Stage::Buffer(writable) = slice.stage else {
            return err(
                &index.expr,
                "index a runtime &[i32] or &mut [i32] parameter",
            );
        };
        if writable.elem != super::Elem::I32 {
            return err(index, "use Lane::load or Lane::store for typed SIMD slices");
        }
        let value = self.expr(&index.index)?;
        // Every host index above i32::MAX is out of bounds for a runtime
        // slice. Keep the fault on the executed path, even for a literal.
        let offset = if matches!(value.stage, Stage::Host | Stage::Literal(Scalar::Int)) {
            emit!({let i: usize = #value; __splice_builder.constant(i32::try_from(i).unwrap_or(-1))})
        } else {
            self.reg(value, Scalar::Index, &index.index)?
        };
        Ok((writable.write, slice.code, offset))
    }

    pub(super) fn index(&self, index: &ExprIndex) -> Result<Value> {
        let (_, slice, offset) = self.index_operands(index)?;

        Ok(Value::new(
            Stage::Reg(Scalar::Int),
            emit!({
                let (buffer, __splice_index) = (#slice, #offset);
                __splice_builder.load(buffer, __splice_index)
            }),
        ))
    }

    pub(super) fn store(
        &self,
        index: &ExprIndex,
        rhs: &Expr,
        op: Option<(&str, &BinOp)>,
    ) -> Result<Value> {
        let (writable, slice, offset) = self.index_operands(index)?;
        if !writable {
            return err(index, "writing a kernel slice requires &mut [i32]");
        }
        let value = self.operand(rhs, Scalar::Int)?;

        let combine = op.map(|(op, _)| {
            let combined = self.bin(
                op,
                emit!(__splice_builder.load(buffer, __splice_index)),
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
            __splice_builder.store(buffer, __splice_index, __splice_value);
        })))
    }
}
