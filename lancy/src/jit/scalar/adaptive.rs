//! Tiered scalar compilation. Entry guards run before any kernel instruction.
use super::ScalarFn;
use crate::input::{Function, Instr, Reg};
use crate::jit::{Tier, worker};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::profile::{Guards, Profile};

struct Guarded<const N: usize> {
    function: ScalarFn<N>,
    assumptions: Guards,
}

struct Optimized<const N: usize> {
    generic: ScalarFn<N>,
    specialized: Option<Guarded<N>>,
}

struct Inner<const N: usize> {
    function: Function,
    baseline: ScalarFn<N>,
    profile: Option<Mutex<Profile>>,
    assumptions: OnceLock<Guards>,
    invalidated: AtomicBool,
    submission: worker::ScalarSubmission,
    ready: OnceLock<Result<Optimized<N>, String>>,
}

/// Immediately usable scalar code with a background-compiled replacement.
/// Entry assumptions may come from the application's observations of its data.
#[derive(Clone)]
pub struct AdaptiveScalar<const N: usize>(Arc<Inner<N>>);

impl<const N: usize> AdaptiveScalar<N> {
    pub fn compile(function: Function) -> Result<Self, String> {
        Self::with_assumptions(function, &[])
    }

    /// Sample the first 32 calls on the scalar baseline, then compile with guards
    /// for up to four arguments that stayed constant. A zero-argument function
    /// requests generic optimization immediately.
    pub fn profiled(function: Function) -> Result<Self, String> {
        if N == 0 {
            return Self::compile(function);
        }
        Self::new(function, None)
    }

    /// Each (runtime argument index, expected value) pair is guarded. On the
    /// first mismatch, subsequent invocations use the optimized generic code.
    pub fn with_assumptions(
        function: Function,
        assumptions: &[(usize, i32)],
    ) -> Result<Self, String> {
        let mut assumptions = assumptions.to_vec();
        assumptions.sort_unstable();
        assumptions.dedup();
        if assumptions.iter().any(|(i, _)| *i >= N) {
            return Err("scalar assumption refers to an unknown runtime argument".into());
        }
        if assumptions.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err("contradictory assumptions on a scalar argument".into());
        }
        Self::new(function, Some(assumptions.into()))
    }

    fn new(function: Function, fixed: Option<Guards>) -> Result<Self, String> {
        let baseline = ScalarFn::baseline(&function)?;
        let profiling = fixed.is_none();
        let kernel = Self(Arc::new(Inner {
            function,
            baseline,
            profile: profiling.then(Mutex::default),
            assumptions: fixed.map(OnceLock::from).unwrap_or_default(),
            invalidated: AtomicBool::new(false),
            submission: worker::ScalarSubmission::default(),
            ready: OnceLock::new(),
        }));
        if !profiling {
            kernel.schedule(false);
        }
        Ok(kernel)
    }

    // The short profiling lock orders concurrent samples and early explicit
    // waits. It is always released before accessing the compiler queue.
    fn finish_profile(&self, args: Option<&[i32; N]>) -> bool {
        if self.0.assumptions.get().is_some() {
            return true;
        }
        let mut profile = self.0.profile.as_ref().unwrap().lock().unwrap();
        if self.0.assumptions.get().is_some() {
            return true;
        }
        let assumptions = match args {
            Some(args) => {
                let Some(guards) = profile.observe(args) else {
                    return false;
                };
                guards
            }
            None => {
                // Early wait explicitly requests code without speculative data.
                *profile = Profile::default();
                Guards::default()
            }
        };
        self.0.assumptions.set(assumptions).unwrap();
        true
    }

    fn schedule(&self, blocking: bool) {
        self.0.submission.schedule(blocking, &self.0.ready, || {
            let pending = Arc::downgrade(&self.0);
            worker::submit_scalar(
                move || {
                    if let Some(inner) = pending.upgrade() {
                        let result = catch_unwind(AssertUnwindSafe(|| inner.optimize()))
                            .unwrap_or_else(|_| Err("scalar background compiler panicked".into()));
                        let _ = inner.ready.set(result);
                    }
                },
                blocking,
            )
        });
    }

    /// Wait for the background version. An early wait on a profiled function
    /// closes an incomplete sample and requests unconditional generic code.
    pub fn wait_for_optimization(&self) -> Result<(), String> {
        self.finish_profile(None);
        self.schedule(true);
        self.0
            .ready
            .wait()
            .as_ref()
            .map(|_| ())
            .map_err(Clone::clone)
    }

    pub fn tier(&self) -> Tier {
        match self.0.ready.get() {
            Some(Ok(code))
                if code.specialized.is_some() && !self.0.invalidated.load(Ordering::Relaxed) =>
            {
                Tier::Specialized
            }
            Some(Ok(_)) => Tier::Generic,
            Some(Err(_)) => Tier::Failed,
            None if self.0.submission.is_submitted() => Tier::Compiling,
            None => Tier::Baseline,
        }
    }

    pub fn invoke(&self, args: [i32; N]) -> i32 {
        match self.0.ready.get() {
            Some(Ok(code)) => {
                if let Some(spec) = &code.specialized
                    && !self.0.invalidated.load(Ordering::Relaxed)
                {
                    if spec.assumptions.iter().all(|&(i, n)| args[i] == n) {
                        return spec.function.invoke(args);
                    }
                    self.0.invalidated.store(true, Ordering::Relaxed);
                }
                code.generic.invoke(args)
            }
            Some(Err(_)) => self.0.baseline.invoke(args),
            None => {
                if self.finish_profile(Some(&args)) {
                    self.schedule(false);
                }
                self.0.baseline.invoke(args)
            }
        }
    }
}

impl<const N: usize> Inner<N> {
    fn optimize(&self) -> Result<Optimized<N>, String> {
        let generic = ScalarFn::compile(&self.function)?;
        let assumptions = self.assumptions.get().expect("compiler request has a plan");
        let specialized = if assumptions.is_empty() {
            None
        } else {
            let mut code = Vec::with_capacity(assumptions.len() + self.function.code.len());
            for &(i, n) in assumptions.iter() {
                code.push(Instr::Const(Reg(i as u32), n));
            }
            code.extend_from_slice(&self.function.code);
            Some(Guarded {
                function: ScalarFn::compile(&self.function.rebuild(code)?)?,
                assumptions: Arc::clone(assumptions),
            })
        };
        Ok(Optimized {
            generic,
            specialized,
        })
    }
}

#[cfg(test)]
mod tests;
