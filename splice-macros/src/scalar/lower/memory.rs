//! Runtime slices carry an immutable buffer identity through specialization.
//! Only lengths, element accesses, and scalar computations become instructions.
use super::{Lower, Scalar, Stage, Value, err};
use crate::scalar::private;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{BinOp, Expr, ExprCast, ExprIndex, ExprMethodCall, ExprReference, Result, UnOp};

impl Lower<'_> {
    pub(super) fn index_comparison(
        &self,
        expr: &syn::ExprBinary,
        lhs: Value,
        rhs: Value,
    ) -> Result<Value> {
        let order = match expr.op {
            BinOp::Eq(_) => "Eq",
            BinOp::Ne(_) => "Ne",
            BinOp::Lt(_) => "Lt",
            BinOp::Le(_) => "Le",
            BinOp::Gt(_) => "Gt",
            BinOp::Ge(_) => "Ge",
            _ => return err(expr, "cast the runtime index to i32 before arithmetic"),
        };
        let mut lhs = self.reg(lhs, Scalar::Index, &expr.left)?;
        let mut rhs = self.reg(rhs, Scalar::Index, &expr.right)?;
        if !matches!(expr.op, BinOp::Eq(_) | BinOp::Ne(_)) {
            // Each supported runtime usize is a sign-extended i32 cast to usize.
            // Its unsigned ordering is preserved by the low 32 bits.
            let b = self.builder;
            lhs = self.bin("Xor", lhs, quote!(#b.constant(i32::MIN)));
            rhs = self.bin("Xor", rhs, quote!(#b.constant(i32::MIN)));
        }
        Ok(Value::new(
            Stage::Reg(Scalar::Bool),
            self.bin(order, lhs, rhs),
        ))
    }

    pub(super) fn method(&self, m: &ExprMethodCall) -> Result<Value> {
        let original = Expr::MethodCall(m.clone());
        if let Ok(code) = self.host_raw(&original) {
            return Ok(Value::new(Stage::Host, code));
        }
        let receiver = self.expr(&m.receiver)?;
        let b = self.builder;
        if (m.method == "len" || m.method == "is_empty")
            && m.args.is_empty()
            && matches!(receiver.stage, Stage::Buffer(_))
        {
            let buffer = receiver.code;
            let length = quote!({let buffer=#buffer; #b.buffer_len(buffer)});
            return Ok(if m.method == "len" {
                Value::new(Stage::Reg(Scalar::Index), length)
            } else {
                Value::new(
                    Stage::Reg(Scalar::Bool),
                    self.bin("Eq", length, quote!(#b.constant(0))),
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
            let right = self.reg(self.expr(rhs)?, Scalar::Int, rhs)?;
            return Ok(Value::new(
                Stage::Reg(Scalar::Int),
                self.bin(op, left, right),
            ));
        }
        err(
            m,
            "expected slice.len/is_empty, i32.min/max, or i32.wrapping_add/sub/mul",
        )
    }

    pub(super) fn cast(&self, cast: &ExprCast) -> Result<Value> {
        let original = Expr::Cast(cast.clone());
        if let Ok(code) = self.host_raw(&original) {
            return Ok(Value::new(Stage::Host, code));
        }
        let value = self.expr(&cast.expr)?;
        let kind = if crate::scalar_type(&cast.ty, "i32") {
            Scalar::Int
        } else if crate::scalar_type(&cast.ty, "usize") {
            Scalar::Index
        } else {
            return err(
                &cast.ty,
                "runtime casts support i32 and usize slice indices",
            );
        };
        let code = match value.stage {
            Stage::Reg(Scalar::Index | Scalar::Int) => value.code,
            Stage::Reg(Scalar::Bool) => {
                let b = self.builder;
                self.bin("And", value.code, quote!(#b.constant(1)))
            }
            _ => return err(cast, "expected an integer or boolean to cast"),
        };
        Ok(Value::new(Stage::Reg(kind), code))
    }

    pub(super) fn reference(&self, reference: &ExprReference) -> Result<Value> {
        let original = Expr::Reference(reference.clone());
        if let Ok(code) = self.host_raw(&original) {
            return Ok(Value::new(Stage::Host, code));
        }
        // Explicit reborrows (&*input and &mut *output) keep the same descriptor.
        let target = match &*reference.expr {
            Expr::Unary(x) if matches!(x.op, UnOp::Deref(_)) => &*x.expr,
            expr => expr,
        };
        let value = self.expr(target)?;
        let writable = reference.mutability.is_some();
        match value.stage {
            Stage::Buffer(actual) if !writable || actual => {
                Ok(Value::new(Stage::Buffer(writable), value.code))
            }
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
        let value = self.expr(&index.index)?;
        // Every host index above i32::MAX is out of bounds for a runtime
        // slice. Keep the fault on the executed path, even for a literal.
        let offset = if matches!(value.stage, Stage::Host | Stage::Literal(Scalar::Int)) {
            let code = value.code;
            let b = self.builder;
            quote!({let i: usize = #code; #b.constant(i32::try_from(i).unwrap_or(-1))})
        } else {
            self.reg(value, Scalar::Index, &index.index)?
        };
        Ok((writable, slice.code, offset))
    }

    pub(super) fn index(&self, index: &ExprIndex) -> Result<Value> {
        if let Ok(code) = self.host_raw(&Expr::Index(index.clone())) {
            return Ok(Value::new(Stage::Host, code));
        }
        let (_, slice, offset) = self.index_operands(index)?;
        let b = self.builder;
        Ok(Value::new(
            Stage::Reg(Scalar::Int),
            quote!({
                let (buffer, index)=(#slice, #offset); #b.load(buffer,index)
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
        let value = self.reg(self.expr(rhs)?, Scalar::Int, rhs)?;
        let b = self.builder;
        let value_reg = private("__splice_store_value");
        let combine = op.map(|(op, _)| {
            let combined = self.bin(op, quote!(#b.load(buffer,index)), quote!(#value_reg));
            quote!(let #value_reg=#combined;)
        });
        // Rust evaluates the RHS before the place for assignment and primitive
        // compound assignment. The index expression may itself mutate memory.
        Ok(Value::unit(quote!({
            let #value_reg=#value; let (buffer,index)=(#slice,#offset);
            #combine #b.store(buffer,index,#value_reg);
        })))
    }
}
