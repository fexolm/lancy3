//! The per-call workspace: frames, spills and the status pointer of a
//! running kernel (LNC-04, X64-17).
//!
//! Small frames live on the stack. Each thread keeps one buffer for larger
//! frames and lends it to the outermost call. A nested call uses its own
//! stack frame or finds the buffer lent and uses a fresh one. Blocks of 64
//! bytes align the workspace to a cache line.
use std::cell::Cell;
use std::mem::MaybeUninit;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Block([MaybeUninit<u64>; 8]);

const INLINE_BLOCKS: usize = 4;

thread_local! {
    static SPARE: Cell<Vec<Block>> = const { Cell::new(Vec::new()) };
}

/// Run `f` on a cache-line-aligned workspace of at least `words` words.
pub(super) fn with<R>(words: usize, f: impl FnOnce(&mut [MaybeUninit<u64>]) -> R) -> R {
    let needed = words.div_ceil(8);
    let empty = Block([MaybeUninit::uninit(); 8]);
    if needed <= INLINE_BLOCKS {
        let mut inline = [empty; INLINE_BLOCKS];
        return f(word_view(&mut inline[..needed], words));
    }
    SPARE.with(|spare| {
        let mut blocks = spare.take();
        if blocks.len() < needed {
            blocks.resize(needed, empty);
        }
        let result = f(word_view(&mut blocks[..needed], words));
        spare.set(blocks);
        result
    })
}

fn word_view(blocks: &mut [Block], words: usize) -> &mut [MaybeUninit<u64>] {
    // SAFETY: the storage is exclusively borrowed and Block is exactly
    // eight consecutive MaybeUninit words.
    let full =
        unsafe { std::slice::from_raw_parts_mut(blocks.as_mut_ptr().cast(), blocks.len() * 8) };
    &mut full[..words]
}

#[cfg(test)]
mod tests {
    use super::{INLINE_BLOCKS, with};

    #[test]
    fn a_warm_call_reuses_one_cache_line_aligned_buffer() {
        let limit = INLINE_BLOCKS * 8;
        let first = with(limit + 8, |w| w.as_ptr() as usize);
        assert_eq!(first % 64, 0);
        assert_eq!(with(limit + 1, |w| w.as_ptr() as usize), first);
        assert_eq!(with(limit + 8, |w| w.len()), limit + 8);
        for size in [0, 1, limit, limit + 1] {
            assert_eq!(with(size, |w| w.len()), size);
        }
    }

    #[test]
    fn a_reentrant_call_gets_its_own_buffer() {
        let limit = INLINE_BLOCKS * 8;
        for (outer_len, inner_len) in [(limit, limit), (limit, limit + 1), (limit + 1, limit + 1)] {
            with(outer_len, |outer| {
                let last = outer_len - 1;
                outer[last].write(41);
                let inner = with(inner_len, |inner| {
                    assert_eq!(inner.len(), inner_len);
                    inner[last].write(53);
                    inner.as_ptr() as usize
                });
                assert_ne!(inner, outer.as_ptr() as usize);
                assert_eq!(inner % 64, 0);
                // SAFETY: the outer callback initialized the last slot.
                assert_eq!(unsafe { outer[last].assume_init() }, 41);
            });
        }
    }
}
