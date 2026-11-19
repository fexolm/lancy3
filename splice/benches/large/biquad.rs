//! Fixed-point audio low-pass filter implemented as a cascade of biquads.
use splice::kernel;

pub(super) const Q14: i32 = 16_384;
pub(super) type Section = (i32, i32, i32, i32); // b0, b1, a1, a2 (b2 = b0)

/// Consume signed PCM samples with zero initial state. Coefficients and sample
/// amplitudes must keep each section's Q14 products and sums in the i32 range.
#[kernel]
pub(super) fn low_pass(#[comptime] sections: &[Section], pcm: &[i32], out: &mut [i32]) -> i32 {
    let len = pcm.len() as i32;
    if (out.len() as i32) < len {
        return -1;
    }
    let mut z1 = vec![0i32; sections.len()];
    let mut z2 = vec![0i32; sections.len()];
    for i in 0..len {
        let mut x = pcm[i as usize];
        for (s, &(b0, b1, a1, a2)) in sections.iter().enumerate() {
            // Transposed direct form II keeps two delay cells per section.
            let b0x = b0 * x;
            let y = (b0x + z1[s]) / Q14;
            z1[s] = b1 * x - a1 * y + z2[s];
            z2[s] = b0x - a2 * y;
            x = y;
        }
        out[i as usize] = x;
    }
    len
}
