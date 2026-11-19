//! Parquet DELTA_BINARY_PACKED payload after parsing the page and block headers.
use splice::kernel;

pub(super) const MINIBLOCK: i32 = 32;
pub(super) const WIDTHS: [(i32, i32); 4] = [(0, 7), (7, 11), (18, 17), (35, 23)];
pub(super) const BLOCK_WORDS: i32 = 58;
pub(super) const BLOCK_VALUES: i32 = MINIBLOCK * WIDTHS.len() as i32;
const SLOTS: [i32; MINIBLOCK as usize] = slots();

const fn slots() -> [i32; MINIBLOCK as usize] {
    let mut result = [0; MINIBLOCK as usize];
    let mut i = 0;
    while i < result.len() {
        result[i] = i as i32;
        i += 1;
    }
    result
}

/// Decode complete blocks with the known miniblock widths. The first value and
/// each block's minimum delta come from its header. Packed words are little-endian.
#[kernel]
pub(super) fn decode(first: i32, minima: &[i32], packed: &[i32], out: &mut [i32]) -> i32 {
    let blocks = minima.len() as i32;
    let capacity = out.len() as i32;
    if capacity == 0 || blocks > (capacity - 1) / BLOCK_VALUES {
        return -1;
    }
    if blocks > packed.len() as i32 / BLOCK_WORDS {
        return -1;
    }

    let mut value = first;
    out[0] = value;
    for block in 0..blocks {
        let base = block * BLOCK_WORDS;
        let minimum = minima[block as usize];
        let mut dst = 1 + block * BLOCK_VALUES;
        for &(word_start, width) in WIDTHS.iter() {
            for &slot in SLOTS.iter() {
                let bit = comptime!(slot * width);
                let word = comptime!(word_start + bit / 32);
                let shift = comptime!(bit % 32);
                let mask = comptime!((1 << width) - 1);
                let low = packed[(base + word) as usize];
                let offset = if shift + width <= 32 {
                    (low >> shift) & mask
                } else {
                    let high = packed[(base + word + 1) as usize];
                    // Splice's i32 right shift sign-extends. Keep just the low fragment.
                    let fragment_mask = comptime!((1 << (32 - shift)) - 1);
                    ((low >> shift) & fragment_mask) | ((high << (32 - shift)) & mask)
                };
                value = value.wrapping_add(minimum).wrapping_add(offset);
                out[dst as usize] = value;
                dst += 1;
            }
        }
    }
    1 + blocks * BLOCK_VALUES
}
