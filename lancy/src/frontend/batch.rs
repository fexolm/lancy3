//! Expansion ABI for the kernel attribute. Inputs and literals are materialized
//! on demand, so untaken host branches cannot leave instructions in the HIR.
use crate::hir::{BuildError, BuildValue, Builder, Program};
use std::borrow::Borrow;

#[derive(Clone, Copy)]
pub enum Operand {
    Input(u16),
    Literal(i32),
    Emitted(BuildValue),
}

impl Operand {
    pub fn integer(value: impl Borrow<i32>) -> Self {
        Self::Literal(*value.borrow())
    }

    pub fn boolean(value: impl Borrow<bool>) -> Self {
        Self::Literal(-i32::from(*value.borrow()))
    }
}

pub struct Context {
    builder: Builder,
    inputs: Vec<Option<BuildValue>>,
}

impl Context {
    pub fn new(inputs: usize) -> Self {
        Self {
            builder: Builder::new(inputs),
            inputs: vec![None; inputs],
        }
    }

    fn resolve(&mut self, value: Operand) -> BuildValue {
        match value {
            Operand::Input(i) => {
                *self.inputs[usize::from(i)].get_or_insert_with(|| self.builder.input(i))
            }
            Operand::Literal(n) => self.builder.constant(n),
            Operand::Emitted(value) => value,
        }
    }

    #[inline]
    pub fn binary(
        &mut self,
        op: impl FnOnce(&mut Builder, BuildValue, BuildValue) -> BuildValue,
        lhs: Operand,
        rhs: Operand,
    ) -> Operand {
        let lhs = self.resolve(lhs);
        let rhs = self.resolve(rhs);
        Operand::Emitted(op(&mut self.builder, lhs, rhs))
    }

    pub fn select(&mut self, mask: Operand, yes: Operand, no: Operand) -> Operand {
        let mask = self.resolve(mask);
        let yes = self.resolve(yes);
        let no = self.resolve(no);
        Operand::Emitted(self.builder.select(mask, yes, no))
    }

    pub fn finish(mut self, result: Operand) -> Result<Program, BuildError> {
        let result = self.resolve(result);
        self.builder.finish(result)
    }
}
