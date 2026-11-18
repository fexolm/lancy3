//! Batch-level tiering for pure, column-oriented kernels.
use super::NativeKernel;
use super::profile::{live_inputs, profile};
use crate::hir::{Assumption, Program};
use crate::jit::{Tier, worker};
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};

const BASELINE_CHUNK: usize = 16_384;
const PROBE_INTERVAL: u16 = 16;
const MAX_PROBE_INTERVAL: u16 = 1024;

/// Cumulative counters. Rows count completed output elements across all callers.
/// A concurrent call can advance the counters while a snapshot is taken.
#[derive(Clone, Debug)]
pub struct KernelStats {
    pub tier: Tier,
    /// A profile, queued request or compilation can coexist with the generic tier.
    pub optimization_pending: bool,
    pub assumptions: Vec<Assumption>,
    pub baseline_rows: u64,
    pub optimized_rows: u64,
    pub deoptimizations: u64,
    /// Successful installations of optimized versions.
    pub compilations: u64,
}

/// An immutable program with a baseline and a shared, replaceable native version.
/// Cloning the handle shares both code and profiling state.
#[derive(Clone)]
pub struct AdaptiveKernel {
    inner: Arc<Inner>,
}

struct Inner {
    program: Program,
    profile_columns: OnceLock<Vec<u16>>,
    baseline: NativeKernel,
    state: Mutex<State>,
    changed: Condvar,
    baseline_rows: AtomicU64,
    optimized_rows: AtomicU64,
    deoptimizations: AtomicU64,
    compilations: AtomicU64,
}

struct Version {
    code: NativeKernel,
    guards: Vec<Assumption>,
}

struct State {
    work: Work,
    specialized: Option<Arc<Version>>,
    generic: Option<Arc<Version>>,
    probe_interval: u16,
    generic_batches: u16,
    has_invalidated: bool,
}

enum Work {
    Cold,
    Idle,
    // One caller is profiling/submitting, or the worker owns the request.
    Pending,
    // Backpressure. A later run can retry without delaying the foreground.
    Deferred(Vec<Assumption>),
    Failed(String),
}

enum Request {
    Profile,
    Retry(Vec<Assumption>),
}

impl AdaptiveKernel {
    /// Build a scalar kernel synchronously. The first nonempty batch supplies the profile.
    pub fn compile(program: Program) -> Result<Self, String> {
        let baseline = NativeKernel::baseline(&program)?;
        Ok(Self {
            inner: Arc::new(Inner {
                program,
                profile_columns: OnceLock::new(),
                baseline,
                state: Mutex::new(State {
                    work: Work::Cold,
                    specialized: None,
                    generic: None,
                    probe_interval: PROBE_INTERVAL,
                    generic_batches: 0,
                    has_invalidated: false,
                }),
                changed: Condvar::new(),
                baseline_rows: AtomicU64::new(0),
                optimized_rows: AtomicU64::new(0),
                deoptimizations: AtomicU64::new(0),
                compilations: AtomicU64::new(0),
            }),
        })
    }

    /// Execute one batch. All input lengths must match the output length.
    /// Concurrent calls can use this handle with independent output buffers.
    pub fn run(&self, columns: &[&[i32]], out: &mut [i32]) -> Result<(), String> {
        if columns.len() != self.inner.program.input_count() {
            return Err(format!(
                "expected {} input columns, got {}",
                self.inner.program.input_count(),
                columns.len()
            ));
        }
        if columns.iter().any(|col| col.len() != out.len()) {
            return Err("input and output lengths must match".into());
        }
        let mut start = 0;
        while start < out.len() {
            let (version, request) = self.dispatch(start == 0);
            match request {
                Some(Request::Profile) => {
                    let used = self
                        .inner
                        .profile_columns
                        .get_or_init(|| live_inputs(&self.inner.program));
                    let guards = profile(used, columns);
                    if guards.is_empty() && version.is_some() {
                        self.finish_work(Work::Idle);
                    } else {
                        self.submit(guards, false);
                    }
                }
                Some(Request::Retry(guards)) => self.submit(guards, false),
                None => {}
            }
            if let Some(version) = version {
                start += self.run_optimized(&version, columns, start, &mut out[start..])?;
                continue;
            }
            // A single long query can switch tiers while its first batch is in flight.
            let end = start + BASELINE_CHUNK.min(out.len() - start);
            self.run_baseline(columns, start, &mut out[start..end])?;
            start = end;
        }
        Ok(())
    }

    fn dispatch(&self, whole_batch: bool) -> (Option<Arc<Version>>, Option<Request>) {
        let mut state = self.inner.state.lock().unwrap();
        let version = state
            .specialized
            .as_ref()
            .or(state.generic.as_ref())
            .cloned();
        let generic_batch = whole_batch && state.specialized.is_none() && state.generic.is_some();
        let request = match &mut state.work {
            Work::Cold => Some(Request::Profile),
            Work::Deferred(guards) => Some(Request::Retry(mem::take(guards))),
            Work::Idle if generic_batch => {
                state.generic_batches += 1;
                if state.generic_batches == state.probe_interval {
                    state.generic_batches = 0;
                    Some(Request::Profile)
                } else {
                    None
                }
            }
            _ => None,
        };
        if request.is_some() {
            state.work = Work::Pending;
        }
        (version, request)
    }

    fn run_baseline(
        &self,
        columns: &[&[i32]],
        start: usize,
        out: &mut [i32],
    ) -> Result<(), String> {
        let written = self.inner.baseline.run_at(columns, start, out)?;
        if written != out.len() {
            return Err("baseline kernel returned an incomplete batch".into());
        }
        self.inner.baseline_rows.fetch_add(written as u64, Relaxed);
        Ok(())
    }

    fn run_optimized(
        &self,
        version: &Arc<Version>,
        columns: &[&[i32]],
        start: usize,
        out: &mut [i32],
    ) -> Result<usize, String> {
        // This Arc pins the mapping until the native call and all its guards have returned.
        let written = version.code.run_at(columns, start, out)?;
        if written > out.len() {
            return Err("optimized kernel returned an invalid prefix length".into());
        }
        self.inner.optimized_rows.fetch_add(written as u64, Relaxed);
        if written == out.len() {
            return Ok(written);
        }
        if version.guards.is_empty() {
            return Err("unguarded optimized kernel returned an incomplete batch".into());
        }

        self.inner.deoptimizations.fetch_add(1, Relaxed);
        let (compile_generic, cached_generic) = {
            let mut state = self.inner.state.lock().unwrap();
            // Late exits from an older version must never invalidate its replacement.
            let invalidate = state
                .specialized
                .as_ref()
                .is_some_and(|v| Arc::ptr_eq(v, version));
            if invalidate {
                state.specialized = None;
                if state.has_invalidated {
                    state.probe_interval = (2 * state.probe_interval).min(MAX_PROBE_INTERVAL);
                }
                state.has_invalidated = true;
                state.generic_batches = 0;
                if state.generic.is_none() {
                    state.work = Work::Pending;
                }
            }
            (invalidate && state.generic.is_none(), state.generic.clone())
        };
        if compile_generic {
            self.submit(Vec::new(), false);
        }
        if let Some(generic) = cached_generic {
            let suffix = &mut out[written..];
            let recovered = generic.code.run_at(columns, start + written, suffix)?;
            if recovered != suffix.len() {
                return Err("unguarded optimized kernel returned an incomplete batch".into());
            }
            self.inner
                .optimized_rows
                .fetch_add(recovered as u64, Relaxed);
            return Ok(written + recovered);
        }
        let end = written + BASELINE_CHUNK.min(out.len() - written);
        self.run_baseline(columns, start + written, &mut out[written..end])?;
        Ok(end)
    }

    /// Wait for the current optimization request. Before the first nonempty batch
    /// this returns immediately. Compilation errors are observable here while
    /// `run` continues to serve batches with the best available code.
    pub fn wait_for_optimization(&self) -> Result<(), String> {
        let mut state = self.inner.state.lock().unwrap();
        loop {
            match &mut state.work {
                Work::Cold | Work::Idle => return Ok(()),
                Work::Failed(reason) => return Err(reason.clone()),
                Work::Pending => state = self.inner.changed.wait(state).unwrap(),
                Work::Deferred(guards) => {
                    let guards = mem::take(guards);
                    state.work = Work::Pending;
                    drop(state);
                    self.submit(guards, true);
                    state = self.inner.state.lock().unwrap();
                }
            }
        }
    }

    /// Compile-time state and counters for inspecting or benchmarking tier changes.
    pub fn stats(&self) -> KernelStats {
        let state = self.inner.state.lock().unwrap();
        let optimization_pending = matches!(state.work, Work::Pending | Work::Deferred(_));
        let (tier, assumptions) = if let Some(v) = &state.specialized {
            (Tier::Specialized, v.guards.clone())
        } else if state.generic.is_some() {
            (Tier::Generic, Vec::new())
        } else if optimization_pending {
            (Tier::Compiling, Vec::new())
        } else if matches!(state.work, Work::Failed(_)) {
            (Tier::Failed, Vec::new())
        } else {
            (Tier::Baseline, Vec::new())
        };
        KernelStats {
            tier,
            optimization_pending,
            assumptions,
            baseline_rows: self.inner.baseline_rows.load(Relaxed),
            optimized_rows: self.inner.optimized_rows.load(Relaxed),
            deoptimizations: self.inner.deoptimizations.load(Relaxed),
            compilations: self.inner.compilations.load(Relaxed),
        }
    }

    fn submit(&self, guards: Vec<Assumption>, blocking: bool) {
        let job = BatchJob {
            kernel: Arc::downgrade(&self.inner),
            guards,
        };
        let next = match worker::submit_batch(job, blocking) {
            Ok(None) => return,
            Ok(Some(job)) => Work::Deferred(job.guards),
            Err(reason) => Work::Failed(reason),
        };
        self.finish_work(next);
    }

    fn finish_work(&self, next: Work) {
        self.inner.state.lock().unwrap().work = next;
        self.inner.changed.notify_all();
    }
}

pub(in crate::jit) struct BatchJob {
    kernel: Weak<Inner>,
    guards: Vec<Assumption>,
}

impl BatchJob {
    pub(in crate::jit) fn run(self) {
        let Some(inner) = self.kernel.upgrade() else {
            return;
        };
        let compiled = panic::catch_unwind(AssertUnwindSafe(|| {
            NativeKernel::specialize(&inner.program, &self.guards)
        }));
        let mut state = inner.state.lock().unwrap();
        state.work = match compiled {
            Ok(Ok(code)) => {
                inner.compilations.fetch_add(1, Relaxed);
                let version = Arc::new(Version {
                    code,
                    guards: self.guards,
                });
                if version.guards.is_empty() {
                    state.generic = Some(version);
                } else {
                    state.specialized = Some(version);
                }
                Work::Idle
            }
            Ok(Err(reason)) => Work::Failed(reason),
            Err(_) => Work::Failed("background compiler panicked".into()),
        };
        inner.changed.notify_all();
    }
}

#[cfg(test)]
mod tests;
