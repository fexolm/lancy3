mod hash;
mod integral;
mod remap;
mod yuv;

use crate::Workload;
use criterion::Criterion;
use splice::{BufferArg, Executable};

pub(super) fn compare(c: &mut Criterion) {
    yuv_to_rgba(c);
    integral_moments(c);
    hash_keys(c);
    remap_rgb16(c);
}

fn yuv_to_rgba(c: &mut Criterion) {
    use yuv::yuv_to_rgba;
    const PIXELS: usize = 512 * 512 + 3;
    let pixels: Vec<_> = crate::common::xorshift(41019)
        .take(PIXELS)
        .map(|x| x as i32)
        .collect();
    let expected: Vec<_> = pixels
        .iter()
        .map(|&pixel| {
            let [y, cb, cr, _] = pixel.to_le_bytes();
            let rgb: [u8; 3] = [[298, 0, 409], [298, -100, -208], [298, 516, 0]].map(|matrix| {
                let dot: i32 = [i32::from(y) - 16, i32::from(cb) - 128, i32::from(cr) - 128]
                    .into_iter()
                    .zip(matrix)
                    .map(|(sample, weight)| sample * weight)
                    .sum();
                ((dot + 128).div_euclid(256)).clamp(0, 255) as u8
            });
            i32::from_le_bytes([rgb[0], rgb[1], rgb[2], 255])
        })
        .collect();
    Workload {
        name: "large_simd_yuv_to_rgba",
        elements: PIXELS,
        workspace_words: expected.len(),
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            let count = kernel.call(&[], &mut [BufferArg::read(&pixels), BufferArg::write(out)])?;
            usize::try_from(count).map_err(|_| "invalid RGBA output buffer".into())
        },
    }
    .compare(
        c,
        yuv_to_rgba::ir,
        builders!(B => yuv_to_rgba::build::<B>(yuv_to_rgba::BuildArgs {})),
    );
}

fn integral_moments(c: &mut Criterion) {
    use integral::integral_moments;
    const WIDTH: usize = 515;
    const HEIGHT: usize = 257;
    const PIXELS: usize = WIDTH * HEIGHT;
    let pixels: Vec<_> = crate::common::xorshift(40829)
        .take(PIXELS)
        .map(|x| (x & 255) as i32)
        .collect();
    let mut expected = Vec::with_capacity(PIXELS * 2);
    for power in [1, 2] {
        // Scalar inclusion-exclusion on a plane with an all-zero top/left border.
        let stride = WIDTH + 1;
        let mut sums = vec![0i64; stride * (HEIGHT + 1)];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let at = (y + 1) * stride + x + 1;
                sums[at] =
                    i64::from(pixels[y * WIDTH + x]).pow(power) + sums[at - 1] + sums[at - stride]
                        - sums[at - stride - 1];
                expected.push(sums[at] as i32);
            }
        }
    }
    Workload {
        name: "large_simd_integral_moments",
        elements: PIXELS,
        workspace_words: expected.len(),
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            let count = kernel.call(
                &[WIDTH as i32, HEIGHT as i32],
                &mut [BufferArg::read(&pixels), BufferArg::write(out)],
            )?;
            usize::try_from(count).map_err(|_| "invalid integral image dimensions".into())
        },
    }
    .compare(
        c,
        integral_moments::ir,
        builders!(B => integral_moments::build::<B>(integral_moments::BuildArgs {})),
    );
}

fn hash_keys(c: &mut Criterion) {
    use hash::{COLUMNS, hash_keys};
    const ROWS: usize = 65_539;
    const SEED: i32 = 19;
    let keys: Vec<_> = crate::common::xorshift(20893)
        .take(COLUMNS * ROWS)
        .map(|x| x as i32)
        .collect();
    let expected: Vec<_> = (0..ROWS)
        .map(|row| {
            let mut h = SEED as u32;
            for column in keys.chunks_exact(ROWS) {
                let word = column[row] as u32;
                h ^= word
                    .wrapping_mul(0xcc9e_2d51)
                    .rotate_left(15)
                    .wrapping_mul(0x1b87_3593);
                h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
            }
            h ^= (COLUMNS * 4) as u32;
            h = (h ^ (h >> 16)).wrapping_mul(0x85eb_ca6b);
            h = (h ^ (h >> 13)).wrapping_mul(0xc2b2_ae35);
            (h ^ (h >> 16)) as i32
        })
        .collect();
    Workload {
        name: "large_simd_murmur3",
        elements: ROWS,
        workspace_words: expected.len(),
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            let count = kernel.call(
                &[SEED],
                &mut [BufferArg::read(&keys), BufferArg::write(out)],
            )?;
            usize::try_from(count).map_err(|_| "invalid hash buffers".into())
        },
    }
    .compare(
        c,
        hash_keys::ir,
        builders!(B => hash_keys::build::<B>(hash_keys::BuildArgs {})),
    );
}

fn remap_rgb16(c: &mut Criterion) {
    use remap::remap_rgb16;
    const EDGE: usize = 145;
    const PLANE: usize = EDGE * EDGE;
    let source: Vec<_> = crate::common::xorshift(57793)
        .take(3 * PLANE)
        .map(|x| x as u16)
        .collect();
    let mut map = Vec::with_capacity(source.len());
    let mut expected = Vec::with_capacity(source.len());
    let center = (EDGE - 1) as f64 / 2.0;
    for (channel, strength) in [-0.12, 0.04, 0.16].into_iter().enumerate() {
        let plane = &source[channel * PLANE..(channel + 1) * PLANE];
        for y in 0..EDGE {
            for x in 0..EDGE {
                let dx = (x as f64 - center) / center;
                let dy = (y as f64 - center) / center;
                let scale = center * (1.0 + strength * (dx * dx + dy * dy));
                let sx = (center + dx * scale).round().clamp(0.0, 2.0 * center) as usize;
                let sy = (center + dy * scale).round().clamp(0.0, 2.0 * center) as usize;
                let local = sy * EDGE + sx;
                map.push(u16::try_from(channel * PLANE + local).unwrap());
                expected.push(plane[local]);
            }
        }
    }
    Workload {
        name: "large_simd_remap_rgb16",
        elements: PLANE,
        workspace_words: expected.len(),
        expected: &|| &expected,
        equal: <[u16]>::eq,
        run: |kernel: &dyn Executable, out: &mut [u16]| {
            let count = kernel.call(
                &[],
                &mut [
                    BufferArg::read::<u16>(&source),
                    BufferArg::read::<u16>(&map),
                    BufferArg::write::<u16>(out),
                ],
            )?;
            usize::try_from(count).map_err(|_| "invalid RGB16 remap buffers".into())
        },
    }
    .compare(
        c,
        remap_rgb16::ir,
        builders!(B => remap_rgb16::build::<B>(remap_rgb16::BuildArgs {})),
    );
}
