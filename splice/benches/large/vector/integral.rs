//! Summed-area tables for local mean and variance on a grayscale image.
use splice::{kernel, std::intrinsics::simd::Lane};

/// Write two row-major planes, one for intensity and one for squared intensity.
/// The rectangle for an output pixel includes that pixel. Sums wrap modulo 2^32.
#[kernel]
pub(crate) fn integral_moments(width: i32, height: i32, pixels: &[i32], out: &mut [i32]) -> i32 {
    if width <= 0 || height < 0 {
        return -1;
    }
    if height > pixels.len() as i32 / width || height > (out.len() as i32 / 2) / width {
        return -1;
    }
    let plane = width * height;
    let lanes = Lane::<i32>::element_width() as i32;
    let zero = Lane::<i32>::splat(0);
    for y in 0..height {
        let row = y * width;
        let mut offset = 0;
        let mut carry = 0;
        let mut square_carry = 0;
        while width - offset >= lanes {
            let at = row + offset;
            let values = Lane::load(pixels, at as usize);
            let mut sum = values.prefix_sum() + Lane::<i32>::splat(carry);
            let mut squares = (values * values).prefix_sum() + Lane::<i32>::splat(square_carry);
            carry = sum.last();
            square_carry = squares.last();
            if y > 0 {
                sum += Lane::load(out, (at - width) as usize);
                squares += Lane::load(out, (plane + at - width) as usize);
            }
            sum.store(out, at as usize);
            squares.store(out, (plane + at) as usize);
            offset += lanes;
        }
        if offset < width {
            let at = row + offset;
            let valid = Lane::<i32>::first_n((width - offset) as usize);
            let values = Lane::load_select(pixels, at as usize, valid, zero);
            let mut sum = values.prefix_sum() + Lane::<i32>::splat(carry);
            let mut squares = (values * values).prefix_sum() + Lane::<i32>::splat(square_carry);
            if y > 0 {
                sum += Lane::load_select(out, (at - width) as usize, valid, zero);
                squares += Lane::load_select(out, (plane + at - width) as usize, valid, zero);
            }
            sum.store_select(out, at as usize, valid);
            squares.store_select(out, (plane + at) as usize, valid);
        }
    }
    2 * plane
}
