//! Per-channel fixed-point requantization of an int32 activation and residual.
use splice::kernel;

pub(super) type Channel = (i32, i32, i32); // Q12 activation scale, Q12 residual scale, output zero point
pub(super) const SCALE: i32 = 4096;

#[kernel]
fn round_q12(value: i32) -> i32 {
    // Round to nearest, with ties away from zero. We use signed division
    // because negative halfway cases must follow the same rule as positive ones.
    let half = SCALE / 2 - ((value >> 31) & SCALE);
    value.wrapping_add(half) / SCALE
}

/// Process channels-last tensors. Each input must be representable in Q12
/// after its per-channel multiplication and rounding bias. Output contains
/// signed int8 values in i32 words, the buffer type shared by Splice backends.
#[kernel]
#[allow(clippy::manual_clamp, reason = "Splice lowers scalar min and max")]
pub(super) fn requantize(
    #[comptime] channels: &[Channel],
    accum: &[i32],
    residual: &[i32],
    out: &mut [i32],
) -> i32 {
    let stride = comptime!(channels.len() as i32);
    let len = accum.len() as i32;
    if stride == 0 || len % stride != 0 || len > residual.len() as i32 || len > out.len() as i32 {
        return -1;
    }
    let mut base = 0;
    while base < len {
        for (c, &(main_scale, residual_scale, zero_point)) in channels.iter().enumerate() {
            let index = (base + c as i32) as usize;
            let main = round_q12(accum[index].wrapping_mul(main_scale));
            let skip = round_q12(residual[index].wrapping_mul(residual_scale));
            out[index] = (main + skip + zero_point).max(-128).min(127);
        }
        base += stride;
    }
    len
}
