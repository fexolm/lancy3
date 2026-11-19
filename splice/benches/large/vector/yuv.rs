//! Convert packed Y'CbCr 4:4:4 pixels into opaque RGBA8.
use splice::{kernel, simd::Lane};

#[kernel]
fn clamp_byte(value: Lane<i32>) -> Lane<i32> {
    let zero = Lane::<i32>::splat(0);
    let limit = Lane::<i32>::splat(255);
    let positive = value.simd_lt(zero).select(zero, value);
    positive.simd_gt(limit).select(limit, positive)
}

#[kernel]
fn convert(packed: Lane<i32>) -> Lane<i32> {
    let byte = Lane::<i32>::splat(255);
    let center = Lane::<i32>::splat(128);
    let y = (packed & byte) - Lane::<i32>::splat(16);
    let u = ((packed >> 8) & byte) - center;
    let v = ((packed >> 16) & byte) - center;
    // Integer BT.601 limited-range matrix with rounding at eight fractional bits.
    let luma = y * Lane::<i32>::splat(298) + center;
    let r: Lane<i32> = clamp_byte((luma + v * Lane::<i32>::splat(409)) >> 8);
    let g: Lane<i32> =
        clamp_byte((luma - u * Lane::<i32>::splat(100) - v * Lane::<i32>::splat(208)) >> 8);
    let b: Lane<i32> = clamp_byte((luma + u * Lane::<i32>::splat(516)) >> 8);
    r | (g * Lane::<i32>::splat(256))
        | (b * Lane::<i32>::splat(65536))
        | Lane::<i32>::splat(-16_777_216)
}

/// The input bytes of each word are Y, Cb, Cr and an unused byte, low to high.
/// Returns the number of converted pixels, or -1 before a write if output is short.
#[kernel]
pub(crate) fn yuv_to_rgba(input: &[i32], out: &mut [i32]) -> i32 {
    if out.len() < input.len() {
        return -1;
    }
    let size = input.len() as i32;
    let width = Lane::<i32>::lanes() as i32;
    let mut offset = 0;
    while size - offset >= width {
        let yuv = Lane::load(input, offset as usize);
        let rgba: Lane<i32> = convert(yuv);
        rgba.store(out, offset as usize);
        offset += width;
    }
    if offset < size {
        let valid = Lane::<i32>::first_n((size - offset) as usize);
        let yuv = Lane::load_select(input, offset as usize, valid, Lane::<i32>::splat(0));
        let rgba: Lane<i32> = convert(yuv);
        rgba.store_select(out, offset as usize, valid);
    }
    size
}
