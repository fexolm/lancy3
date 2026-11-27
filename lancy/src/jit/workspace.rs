//! The per-call workspace: frames, spills and the status pointer of a
//! running kernel (LNC-04, X64-17).
//!
//! Each thread keeps one buffer and lends it to the outermost call, so a
//! warm call allocates nothing. A kernel entered again from a Rust callback
//! finds the buffer lent and uses a fresh one. Blocks of 64 bytes align the
//! workspace to a cache line.
use std::cell::Cell;
use std::mem::MaybeUninit;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Block([MaybeUninit<u64>; 8]);

thread_local! {
    static SPARE: Cell<Vec<Block>> = const { Cell::new(Vec::new()) };
}

/// Run `f` on a cache-line-aligned workspace of at least `words` words.
pub(super) fn with<R>(words: usize, f: impl FnOnce(&mut [MaybeUninit<u64>]) -> R) -> R {
    let mut blocks = SPARE.take();
    let needed = words.div_ceil(8);
    if blocks.len() < needed {
        blocks.resize(needed, Block([MaybeUninit::uninit(); 8]));
    }
    // SAFETY: `Block` is eight `MaybeUninit<u64>` without padding.
    let words_view = unsafe {
        std::slice::from_raw_parts_mut(blocks.as_mut_ptr().cast::<MaybeUninit<u64>>(), words)
    };
    let result = f(words_view);
    SPARE.set(blocks);
    result
}

#[cfg(test)]
mod tests {
    use super::with;

    #[test]
    fn a_warm_call_reuses_one_cache_line_aligned_buffer() {
        let first = with(5, |w| w.as_ptr() as usize);
        assert_eq!(first % 64, 0);
        assert_eq!(with(3, |w| w.as_ptr() as usize), first);
        assert_eq!(with(5, |w| w.len()), 5);
    }

    #[test]
    fn a_reentrant_call_gets_its_own_buffer() {
        with(4, |outer| {
            let inner = with(4, |inner| inner.as_ptr() as usize);
            assert_ne!(inner, outer.as_ptr() as usize);
            assert_eq!(inner % 64, 0);
        });
    }
}
