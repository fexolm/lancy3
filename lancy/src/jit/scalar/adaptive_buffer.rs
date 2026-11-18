//! Tier whole invocations of scalar functions that access caller-owned slices.
use super::ScalarKernel;
use crate::hir::{BufferArg, Function};
use crate::jit::{Tier, worker};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, OnceLock};

struct Inner {
    function: Function,
    baseline: ScalarKernel,
    optimized: OnceLock<Result<ScalarKernel, String>>,
    submission: worker::ScalarSubmission,
}

/// A baseline slice kernel with a background-compiled generic replacement.
/// Clones share machine code. Each call supplies its own arguments and slices.
#[derive(Clone)]
pub struct AdaptiveBufferKernel(Arc<Inner>);

impl AdaptiveBufferKernel {
    pub fn compile(function: Function) -> Result<Self, String> {
        let baseline = ScalarKernel::baseline(&function)?;
        let kernel = Self(Arc::new(Inner {
            function,
            baseline,
            optimized: OnceLock::new(),
            submission: worker::ScalarSubmission::default(),
        }));
        kernel.schedule(false);
        Ok(kernel)
    }

    /// Borrow the best installed version. A caller created from this reference
    /// can reuse its workspace across a batch and stays on the selected version.
    pub fn current(&self) -> &ScalarKernel {
        match self.0.optimized.get() {
            Some(Ok(optimized)) => optimized,
            Some(Err(_)) => &self.0.baseline,
            None => {
                self.schedule(false);
                &self.0.baseline
            }
        }
    }

    /// Choose one version on entry. A runtime error returns to the caller with
    /// all stores preceding the failing instruction visible.
    pub fn call_with_buffers(
        &self,
        args: &[i32],
        buffers: &mut [BufferArg<'_>],
    ) -> Result<i32, String> {
        self.current().call_with_buffers(args, buffers)
    }

    /// Wait for background installation, retrying a previously full queue.
    pub fn wait_for_optimization(&self) -> Result<(), String> {
        self.schedule(true);
        self.0
            .optimized
            .wait()
            .as_ref()
            .map(|_| ())
            .map_err(Clone::clone)
    }

    pub fn tier(&self) -> Tier {
        match self.0.optimized.get() {
            Some(Ok(_)) => Tier::Generic,
            Some(Err(_)) => Tier::Failed,
            None if self.0.submission.is_submitted() => Tier::Compiling,
            None => Tier::Baseline,
        }
    }

    fn schedule(&self, blocking: bool) {
        self.0.submission.schedule(blocking, &self.0.optimized, || {
            let pending = Arc::downgrade(&self.0);
            worker::submit_scalar(
                move || {
                    if let Some(inner) = pending.upgrade() {
                        let result = catch_unwind(AssertUnwindSafe(|| {
                            ScalarKernel::compile(&inner.function)
                        }))
                        .unwrap_or_else(|_| {
                            Err("buffer kernel background compiler panicked".into())
                        });
                        let _ = inner.optimized.set(result);
                    }
                },
                blocking,
            )
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{Binary, Buffer, BufferKind, FunctionBuilder};

    fn increment_then_load() -> Function {
        let mut b = FunctionBuilder::with_buffers(1, vec![BufferKind::Write]);
        let zero = b.constant(0);
        let one = b.constant(1);
        let value = b.load(Buffer(0), zero);
        let incremented = b.binary(Binary::Add, value, one);
        b.store(Buffer(0), zero, incremented);
        let result = b.load(Buffer(0), b.arg(0));
        b.ret(result);
        b.finish().unwrap()
    }

    fn check_writes(kernel: &AdaptiveBufferKernel) {
        let mut data = [41, 99];
        let error = kernel
            .call_with_buffers(&[2], &mut [BufferArg::Write(&mut data)])
            .unwrap_err();
        assert!(error.contains("out of bounds"), "{error}");
        assert_eq!(data, [42, 99]);
        assert_eq!(
            kernel
                .call_with_buffers(&[1], &mut [BufferArg::Write(&mut data)])
                .unwrap(),
            99
        );
        assert_eq!(data, [43, 99]);
        assert!(
            kernel
                .call_with_buffers(&[], &mut [BufferArg::Write(&mut data)])
                .is_err()
        );
        assert_eq!(data, [43, 99]);
    }

    #[test]
    fn baseline_and_optimized_never_replay_partial_writes() {
        // The channel keeps optimization behind this job until the baseline
        // assertions finish. Dropping the sender on unwind also releases it.
        let (hold, resume) = std::sync::mpsc::channel::<()>();
        worker::submit_scalar(
            move || {
                let _ = resume.recv();
            },
            true,
        )
        .unwrap();
        let kernel = AdaptiveBufferKernel::compile(increment_then_load()).unwrap();
        assert!(matches!(kernel.tier(), Tier::Baseline | Tier::Compiling));
        let baseline = kernel.current();
        let mut caller = baseline.caller();
        check_writes(&kernel);
        drop(hold);
        kernel.wait_for_optimization().unwrap();
        assert_eq!(kernel.tier(), Tier::Generic);
        assert!(!std::ptr::eq(baseline, kernel.current()));
        let mut data = [10];
        assert_eq!(
            caller
                .call_with_buffers(&[0], &mut [BufferArg::Write(&mut data)])
                .unwrap(),
            11
        );
        check_writes(&kernel);
    }

    #[test]
    fn concurrent_invocations_keep_independent_buffers_and_errors() {
        let kernel = AdaptiveBufferKernel::compile(increment_then_load()).unwrap();
        std::thread::scope(|scope| {
            for initial in 0..8 {
                let kernel = kernel.clone();
                scope.spawn(move || {
                    let mut data = [initial];
                    for i in 1..=64 {
                        let index = i % 2;
                        let result =
                            kernel.call_with_buffers(&[index], &mut [BufferArg::Write(&mut data)]);
                        if index == 0 {
                            assert_eq!(result.unwrap(), initial + i);
                        } else {
                            assert!(result.unwrap_err().contains("out of bounds"));
                        }
                        assert_eq!(data, [initial + i]);
                    }
                });
            }
            kernel.wait_for_optimization().unwrap();
        });
    }
}
