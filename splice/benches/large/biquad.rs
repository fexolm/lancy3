//! Fixed-point audio low-pass filter implemented as a cascade of biquads.
use splice::kernel;

pub(super) const Q14: i32 = 16_384;
pub(super) type Section = (i32, i32, i32, i32); // b0, b1, a1, a2 (b2 = b0)

/// Consume signed PCM samples with zero initial state. Coefficients and sample
/// amplitudes must keep each section's Q14 products and sums in the i32 range.
#[kernel]
pub(super) fn low_pass(
    #[comptime] sections: &[Section],
    pcm: &[i32],
    out: &mut [i32],
    state: &mut [i32],
) -> i32 {
    let len = pcm.len() as i32;
    if (out.len() as i32) < len || state.len() < sections.len() * 2 {
        return -1;
    }
    for s in 0..(sections.len() * 2) as i32 {
        state[s as usize] = 0;
    }
    for i in 0..len {
        let mut x = pcm[i as usize];
        for (s, &(b0, b1, a1, a2)) in sections.iter().enumerate() {
            // Transposed direct form II keeps two delay cells per section.
            let b0x = b0 * x;
            let y = (b0x + state[2 * s]) / Q14;
            state[2 * s] = b1 * x - a1 * y + state[2 * s + 1];
            state[2 * s + 1] = b0x - a2 * y;
            x = y;
        }
        out[i as usize] = x;
    }
    len
}
