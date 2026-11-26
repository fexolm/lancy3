use super::{
    Function, TargetInst,
    schedule::{self, Schedulable},
};
use crate::CompileError;
use regalloc2::{Algorithm, Ctx, Output, RegallocOptions};

pub(super) fn allocate<I: TargetInst + Schedulable>(
    f: &mut Function<I>,
    ctx: &mut Ctx,
) -> Result<Output, CompileError> {
    {
        let _span =
            tracing::debug_span!(target: "lancy::compile", "schedule_instructions").entered();
        schedule::run(f);
    }
    let _span = tracing::debug_span!(target: "lancy::compile", "allocate_registers").entered();
    let options = RegallocOptions {
        algorithm: Algorithm::Ion,
        validate_ssa: cfg!(any(test, debug_assertions)),
        ..Default::default()
    };
    regalloc2::run_with_ctx(f, I::machine_env(f.machine()), &options, ctx)
        .map_err(CompileError::internal)?;
    #[cfg(test)]
    {
        let mut checker = regalloc2::checker::Checker::new(f, I::machine_env(f.machine()));
        checker.prepare(&ctx.output);
        checker.run().expect("invalid register allocation");
    }
    Ok(std::mem::take(&mut ctx.output))
}

/// The allocations of one instruction, found by the virtual register an
/// operand names. Emitters look operands up by the instruction field that
/// holds them, so the order of [`TargetInst::operands`] binds nothing.
#[derive(Clone, Copy)]
pub(super) struct Allocations<'a> {
    operands: &'a [regalloc2::Operand],
    allocs: &'a [regalloc2::Allocation],
}

impl<'a> Allocations<'a> {
    pub fn of<I: TargetInst>(
        f: &'a Function<I>,
        output: &'a Output,
        inst: regalloc2::Inst,
    ) -> Self {
        use regalloc2::Function as _;
        Self {
            operands: f.inst_operands(inst),
            allocs: output.inst_allocs(inst),
        }
    }

    /// Where `v` is. Two uses of one value read the same bits, so a
    /// register is preferred when the value has several operands.
    pub fn get(self, v: regalloc2::VReg) -> regalloc2::Allocation {
        let mut found = None;
        for (operand, &alloc) in self.operands.iter().zip(self.allocs) {
            if operand.vreg() == v {
                if alloc.is_reg() {
                    return alloc;
                }
                found.get_or_insert(alloc);
            }
        }
        found.unwrap_or_else(|| panic!("{v:?} is not an operand of this instruction"))
    }

    /// The register encoding of `v`, which has a register constraint.
    pub fn reg(self, v: regalloc2::VReg) -> usize {
        self.get(v)
            .as_reg()
            .expect("register-constrained operand")
            .hw_enc()
    }
}
