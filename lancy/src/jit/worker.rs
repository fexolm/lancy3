//! A bounded process-wide compiler queue for background kernel specialization.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Mutex, OnceLock};
use std::thread;

const QUEUE_CAPACITY: usize = 128;

type Job = Box<dyn FnOnce() + Send>;

/// Submit one scalar compilation, retaining the right to retry a full queue.
/// Construct the job inside `attempt` so pending calls do not allocate or clone its owner.
#[derive(Default)]
pub(super) struct ScalarSubmission {
    submitted: AtomicBool,
    lock: Mutex<()>,
}

impl ScalarSubmission {
    pub(super) fn is_submitted(&self) -> bool {
        self.submitted.load(Ordering::Acquire)
    }

    pub(super) fn schedule<T>(
        &self,
        blocking: bool,
        result: &OnceLock<Result<T, String>>,
        attempt: impl FnOnce() -> Result<bool, String>,
    ) {
        if self.is_submitted() {
            return;
        }
        let _guard = if blocking {
            self.lock.lock().unwrap()
        } else {
            let Ok(guard) = self.lock.try_lock() else {
                return;
            };
            guard
        };
        if self.submitted.load(Ordering::Relaxed) {
            return;
        }
        match attempt() {
            Ok(false) => return,
            Err(reason) => {
                let _ = result.set(Err(reason));
            }
            Ok(true) => {}
        }
        self.submitted.store(true, Ordering::Release);
    }
}

pub(super) fn submit_scalar(
    job: impl FnOnce() + Send + 'static,
    blocking: bool,
) -> Result<bool, String> {
    Ok(submit(Box::new(job), blocking)?.is_none())
}

fn submit(job: Job, blocking: bool) -> Result<Option<Job>, String> {
    let queue = compiler_queue()?;
    if blocking {
        return queue
            .send(job)
            .map(|()| None)
            .map_err(|_| "background compiler stopped".into());
    }
    match queue.try_send(job) {
        Ok(()) => Ok(None),
        Err(TrySendError::Full(job)) => Ok(Some(job)),
        Err(TrySendError::Disconnected(_)) => Err("background compiler stopped".into()),
    }
}

fn compiler_queue() -> Result<&'static SyncSender<Job>, String> {
    static QUEUE: OnceLock<Result<SyncSender<Job>, String>> = OnceLock::new();
    QUEUE
        .get_or_init(|| {
            let (tx, rx) = mpsc::sync_channel::<Job>(QUEUE_CAPACITY);
            thread::Builder::new()
                .name("lancy-jit-compiler".into())
                .spawn(move || {
                    for job in rx {
                        job();
                    }
                })
                .map_err(|e| format!("could not start background compiler: {e}"))?;
            Ok(tx)
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_queue_can_retry_and_terminal_results_skip_job_construction() {
        for terminal in [Ok(true), Err("worker stopped".to_owned())] {
            let submission = ScalarSubmission::default();
            let result = OnceLock::<Result<(), String>>::new();
            submission.schedule(false, &result, || Ok(false));
            assert!(!submission.is_submitted());
            assert!(result.get().is_none());
            submission.schedule(true, &result, || terminal.clone());
            assert!(submission.is_submitted());
            assert_eq!(result.get(), terminal.err().map(Err).as_ref());
            submission.schedule(false, &result, || panic!("duplicate job construction"));
        }
    }
}
