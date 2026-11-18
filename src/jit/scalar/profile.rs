//! Finds entry arguments that stay constant across the first calls.
use std::sync::Arc;

pub(super) const SAMPLE_CALLS: u8 = 32;
const MAX_PROFILE_GUARDS: usize = 4;
pub(super) type Guards = Arc<[(usize, i32)]>;

#[derive(Default)]
pub(super) struct Profile {
    calls: u8,
    stable: Vec<(usize, i32)>,
}

impl Profile {
    pub(super) fn observe(&mut self, args: &[i32]) -> Option<Guards> {
        if self.calls == 0 {
            self.stable.extend(args.iter().copied().enumerate());
        } else {
            self.stable.retain(|&(i, value)| args[i] == value);
        }
        self.calls += 1;
        if self.calls < SAMPLE_CALLS {
            return None;
        }
        self.stable.truncate(MAX_PROFILE_GUARDS);
        Some(std::mem::take(&mut self.stable).into())
    }
}
