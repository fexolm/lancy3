use super::Rules;
use crate::lir::x64::{Cpu, inst::class};
use crate::support::numeric::Action;
use crate::{
    CompileError,
    hir::operation::operation,
    hir::{Lane, Op, Simd, Ternary, Type, Value},
    lir::x64::inst::{Expansion, Inst, NativeInst, simd::Native},
};
use regalloc2::{RegClass, VReg};

impl Rules<'_> {
    pub(super) fn numeric(&mut self, op: Op) -> Result<Inst, CompileError> {
        if let Some(inst) = self.general_scalar(op)? {
            return Ok(inst);
        }
        if let Op::Simd(kind @ (Simd::Compress(value, mask) | Simd::Expand(value, mask))) = op
            && self.ty(value).lane().unwrap().bytes() == 4
            && !crate::lir::x64::inst::simd::evex(self.cpu)
            && self.cpu.bytes <= 32
        {
            return Ok(Inst::CompactLookup(
                matches!(kind, Simd::Expand(..)),
                self.dst(),
                self.input(value),
                self.input(mask),
                self.body.temp(RegClass::Int)?,
            ));
        }
        if let Some(inst) = self
            .immediate_rearrangement(op)
            .or_else(|| self.paired_shuffle(op))
            .or_else(|| self.immediate_vector_shift(op))
            .or_else(|| self.masked_numeric(op))
        {
            return Ok(inst);
        }
        if let Some(inst) = self.dynamic_lane(op)? {
            return Ok(inst);
        }
        if cfg!(feature = "experimental-simd")
            && let Some(inst) = self.experimental_numeric(op)?
        {
            return Ok(inst);
        }
        let (action, input, output, args) =
            operation(op, |v| self.ty(v)).expect("numeric instruction");

        if let Some(native) = Native::select(action, input, output, self.cpu) {
            let dst = self.dst();
            return self.native(native, dst, args);
        }
        if let Some(native) = self.scalar_in_vector(action, input, output) {
            // The packed form computes lane 0; a mask result moves to a
            // general register.
            if class(output) == RegClass::Vector {
                let dst = self.dst();
                return self.native(native, dst, args);
            }
            let mask = self.body.temp(RegClass::Vector)?;
            let inst = self.native(native, mask, args)?;
            self.out.push(inst);
            return Ok(Inst::BitMove(Type::F32, Type::I32, self.dst(), mask));
        }
        let dst = self.dst();
        let args = args.map(|v| v.map(|v| (self.ty(v), self.input(v))));
        let expansion = Expansion {
            action,
            input,
            output,
            dst,
            args,
        };
        Ok(self.body.details_mut().fallback(expansion))
    }

    /// A packed 16-byte form for an element-wise operation on an `f32` or
    /// `f64` scalar, which lives in lane 0 of a vector register. The other
    /// lanes hold whatever the register held; floating-point operations on
    /// them neither trap nor affect lane 0.
    fn scalar_in_vector(&self, action: Action, input: Type, output: Type) -> Option<Native> {
        let lane = input.lane()?;
        if matches!(input, Type::Vector(_))
            || !matches!(lane, Lane::F32 | Lane::F64)
            || !matches!(
                action,
                Action::Unary(_)
                    | Action::Binary(_)
                    | Action::Ternary(Ternary::MulAdd | Ternary::BitSelect)
            )
        {
            return None;
        }
        let vector = Type::Vector(lane);
        let result = if class(output) == RegClass::Vector {
            vector
        } else {
            vector.mask_type()
        };
        let cpu = Cpu {
            bytes: 16,
            ..self.cpu
        };
        Native::select(action, vector, result, cpu)
    }

    fn native(
        &mut self,
        native: Native,
        dst: VReg,
        args: [Option<Value>; 3],
    ) -> Result<Inst, CompileError> {
        let args = args.map(|v| v.map(|v| self.input(v)));
        let mut temps = [None; 6];
        for temp in &mut temps[..native.temporaries()] {
            *temp = Some(self.body.temp(RegClass::Vector)?);
        }
        Ok(self.body.details_mut().native(NativeInst {
            native,
            dst,
            args,
            temps,
        }))
    }
}
