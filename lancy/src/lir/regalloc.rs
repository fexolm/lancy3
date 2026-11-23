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
