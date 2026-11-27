//! Scalar call results whose stage depends on the resolved Rust declaration.
use super::{Lower, Scalar, Stage, Value, checked_operator, err};
use proc_macro2::{Ident, Span, TokenStream};
use quote::ToTokens;
use syn::{BinOp, ExprBinary, Result};

impl Lower<'_> {
    pub(super) fn word(&self, v: Value, at: impl ToTokens) -> Result<TokenStream> {
        let root = self.root;
        let word = emit!(#root::frontend::scalar::Word);
        Ok(match v.stage {
            Stage::Word(_) => v.code,
            Stage::Reg(_) => emit!(#word::Runtime(#v)),
            Stage::Host | Stage::Literal(_) => {
                emit!(#word::Known(#root::frontend::scalar::ScalarConstant::constant(#v)))
            }
            _ => return err(at, "expected a scalar word"),
        })
    }

    pub(super) fn word_binary(
        &self,
        expr: &ExprBinary,
        left: Value,
        right: Value,
    ) -> Result<Value> {
        use BinOp::*;
        let (op, predicate) = match expr.op {
            Add(_) => ("Add", false),
            Sub(_) => ("Sub", false),
            Mul(_) => ("Mul", false),
            Div(_) => ("Div", false),
            Rem(_) => ("Rem", false),
            BitAnd(_) => ("And", false),
            BitOr(_) => ("Or", false),
            BitXor(_) => ("Xor", false),
            Shl(_) => ("Shl", false),
            Shr(_) => ("Shr", false),
            Eq(_) => ("Eq", true),
            Ne(_) => ("Ne", true),
            Lt(_) => ("Lt", true),
            Le(_) => ("Le", true),
            Gt(_) => ("Gt", true),
            Ge(_) => ("Ge", true),
            And(_) => {
                return self.choice(
                    left,
                    right,
                    Value::new(Stage::Literal(Scalar::Bool), emit!(false)),
                    expr,
                );
            }
            Or(_) => {
                return self.choice(
                    left,
                    Value::new(Stage::Literal(Scalar::Bool), emit!(true)),
                    right,
                    expr,
                );
            }
            _ => return err(expr.op, "expected a scalar binary operator"),
        };
        let root = self.root;
        let variant = Ident::new(op, Span::call_site());
        let plain = emit!(#root::hir::Binary::#variant);
        let op = if let Some(checked) = checked_operator(op) {
            let checked = Ident::new(checked, Span::call_site());
            let enabled = super::host::kernel_overflow_checks();
            emit!(if #enabled { #root::hir::Binary::#checked }else{#plain})
        } else {
            plain
        };
        let operand = |value: &Value| match value.stage {
            Stage::Word(t) | Stage::Reg(t) | Stage::Literal(t) if t != Scalar::Inferred => Some(t),
            _ => None,
        };
        let kind = if predicate {
            Scalar::Bool
        } else {
            operand(&left)
                .or(operand(&right))
                .unwrap_or(Scalar::Inferred)
        };
        let left = self.word(left, &expr.left)?;
        let right = self.word(right, &expr.right)?;
        Ok(Value::new(
            Stage::Word(kind),
            emit!({
                let (left,right)=(#left,#right);
                left.binary(#op,right,__splice_builder)?
            }),
        ))
    }
}
