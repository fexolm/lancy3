use super::{Function, TargetInst};
use crate::CompileError;
use regalloc2::{Algorithm, Ctx, Output, RegallocOptions};

pub(super) fn allocate<I: TargetInst>(
    f: &Function<I>,
    ctx: &mut Ctx,
) -> Result<Output, CompileError> {
    let options = RegallocOptions {
        algorithm: Algorithm::Ion,
        validate_ssa: cfg!(any(test, debug_assertions)),
        ..Default::default()
    };
    regalloc2::run_with_ctx(f, I::machine_env(), &options, ctx)?;
    #[cfg(test)]
    {
        let mut checker = regalloc2::checker::Checker::new(f, I::machine_env());
        checker.prepare(&ctx.output);
        checker.run().expect("invalid register allocation");
    }
    Ok(std::mem::take(&mut ctx.output))
}
