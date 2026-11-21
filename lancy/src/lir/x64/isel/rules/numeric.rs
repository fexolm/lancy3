use super::Rules;
use crate::{
    CompileError,
    hir::Op,
    hir::operation::operation,
    lir::x64::inst::{Inst, simd::Native},
};
use regalloc2::RegClass;

impl Rules<'_> {
    pub(super) fn numeric(&mut self, op: Op) -> Result<Inst, CompileError> {
        if let Some(inst) = self
            .immediate_rearrangement(op)
            .or_else(|| self.immediate_vector_shift(op))
            .or_else(|| self.masked_numeric(op))
        {
            return Ok(inst);
        }
        if cfg!(feature = "experimental-simd")
            && let Some(inst) = self.experimental_numeric(op)?
        {
            return Ok(inst);
        }
        let (action, input, output, args) =
            operation(op, |v| self.ty(v)).expect("numeric instruction");

        let dst = self.dst();
        let Some(native) = Native::select(action, input, output, self.cpu) else {
            let args = args.map(|v| v.map(|v| (self.ty(v), self.input(v))));
            return Ok(Inst::Lanes {
                action,
                input,
                output,
                dst,
                args,
            });
        };
        let args = args.map(|v| v.map(|v| self.input(v)));
        let mut temps = [None; 6];
        for temp in &mut temps[..native.temporaries()] {
            *temp = Some(self.body.temp(RegClass::Vector)?);
        }
        Ok(Inst::Native(native, dst, args, temps))
    }
}
