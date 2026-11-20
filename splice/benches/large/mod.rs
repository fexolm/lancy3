mod biquad;
mod csv;
mod delta;
mod requantize;
mod vector;

use super::{Builders, Workload};
use criterion::Criterion;
use splice::{BufferArg, Executable};
use splice_cranelift::Cranelift;
use splice_lancy::Lancy;
use splice_llvm::LlvmManualSimd;
use std::hint::black_box;

pub(super) fn compare(c: &mut Criterion) {
    parquet_delta(c);
    decimal_csv(c);
    per_channel_requantize(c);
    audio_low_pass(c);
    vector::compare(c);
}

fn samples(mut seed: u32) -> impl Iterator<Item = u32> {
    std::iter::repeat_with(move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    })
}

fn parquet_delta(c: &mut Criterion) {
    use delta::{BLOCK_VALUES, BLOCK_WORDS, MINIBLOCK, WIDTHS, decode};
    const BLOCKS: usize = 256;
    const FIRST: i32 = -8_000_000;
    let mut expected = vec![FIRST];
    let mut minima = Vec::with_capacity(BLOCKS);
    let mut packed = Vec::with_capacity(BLOCKS * BLOCK_WORDS as usize);
    let mut random = samples(9051);
    let mut value = FIRST;
    for _ in 0..BLOCKS {
        let minimum = -1_120_000 + (random.next().unwrap() % 110_001) as i32;
        minima.push(minimum);
        for (miniblock, &(_, width)) in WIDTHS.iter().enumerate() {
            let mut reservoir = 0u64;
            let mut bits = 0;
            for slot in 0..MINIBLOCK {
                let offset = match (miniblock, slot) {
                    (0, 0) => 0,                // Every block must contain its stated minimum delta.
                    (_, 1) => 1 << (width - 1), // Ensure the declared width is needed.
                    _ => random.next().unwrap() & ((1 << width) - 1),
                };
                value = value.wrapping_add(minimum + offset as i32);
                expected.push(value);
                reservoir |= u64::from(offset) << bits;
                bits += width;
                if bits >= 32 {
                    packed.push(reservoir as i32);
                    reservoir >>= 32;
                    bits -= 32;
                }
            }
            assert_eq!(bits, 0);
        }
    }
    assert_eq!(expected.len(), 1 + BLOCKS * BLOCK_VALUES as usize);
    assert_eq!(packed.len(), BLOCKS * BLOCK_WORDS as usize);
    Workload {
        name: "large_parquet_delta",
        elements: expected.len(),
        workspace_words: expected.len(),
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            let count = kernel.call(
                &[FIRST],
                &mut [
                    BufferArg::Read(&minima),
                    BufferArg::Read(&packed),
                    BufferArg::Write(out),
                ],
            )?;
            usize::try_from(count).map_err(|_| "invalid delta page buffers".into())
        },
    }
    .compare(
        c,
        decode::ir,
        Builders {
            lancy: &decode::build::<Lancy>,
            cranelift: &decode::build::<Cranelift>,
            llvm: &decode::build::<LlvmManualSimd>,
        },
    );
}

fn decimal_csv(c: &mut Criterion) {
    use csv::{FIELDS, LIMIT, export};
    const ROWS: usize = 8192;
    let mut records: Vec<_> = samples(4193)
        .take(ROWS * FIELDS)
        .map(|sample| (sample % (2 * LIMIT + 1) as u32) as i32 - LIMIT)
        .collect();
    records[..FIELDS].copy_from_slice(&[0, -1, -LIMIT, LIMIT]);
    let mut expected = Vec::new();
    for (i, &value) in records.iter().enumerate() {
        let magnitude = value.unsigned_abs();
        let sign = if value < 0 { '-' } else { '+' };
        let sep = if i % FIELDS == FIELDS - 1 { '\n' } else { ',' };
        let field = format!("{sign}{:06}.{:03}{sep}", magnitude / 1000, magnitude % 1000);
        expected.extend(
            field
                .as_bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&bytes| i32::from_le_bytes(bytes)),
        );
    }
    Workload {
        name: "large_decimal_csv",
        elements: ROWS,
        workspace_words: expected.len(),
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            let count =
                kernel.call(&[], &mut [BufferArg::Read(&records), BufferArg::Write(out)])?;
            usize::try_from(count).map_err(|_| "invalid CSV export buffers or value".into())
        },
    }
    .compare(
        c,
        export::ir,
        Builders {
            lancy: &export::build::<Lancy>,
            cranelift: &export::build::<Cranelift>,
            llvm: &export::build::<LlvmManualSimd>,
        },
    );
}

fn per_channel_requantize(c: &mut Criterion) {
    use self::requantize::{Channel, SCALE, requantize};
    const SITES: usize = 4096;
    let channels: [Channel; 16] =
        std::array::from_fn(|c| (61 + 11 * c as i32, 329 + 41 * c as i32, c as i32 - 8));
    let len = SITES * channels.len();
    let mut accum: Vec<_> = samples(36011)
        .take(len)
        .map(|x| (x % 16_385) as i32 - 8192)
        .collect();
    let residual: Vec<_> = samples(21013)
        .take(len)
        .map(|x| (x % 256) as i32 - 128)
        .collect();
    // Channel zero's odd scale produces an exact Q12 tie at ±2048.
    accum[0] = 2048;
    accum[channels.len()] = -2048;
    let expected: Vec<_> = accum
        .iter()
        .zip(&residual)
        .zip(channels.iter().cycle())
        .map(|((&a, &r), &(s, t, z))| {
            let main = (f64::from(a) * f64::from(s) / f64::from(SCALE)).round() as i32;
            let skip = (f64::from(r) * f64::from(t) / f64::from(SCALE)).round() as i32;
            (main + skip + z).clamp(-128, 127)
        })
        .collect();
    Workload {
        name: "large_per_channel_requantize",
        elements: len,
        workspace_words: expected.len(),
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            let count = kernel.call(
                &[],
                &mut [
                    BufferArg::Read(&accum),
                    BufferArg::Read(&residual),
                    BufferArg::Write(out),
                ],
            )?;
            usize::try_from(count).map_err(|_| "invalid requantization buffers".into())
        },
    }
    .compare(
        c,
        || requantize::ir(black_box(&channels)),
        Builders {
            lancy: &|| requantize::build::<Lancy>(black_box(&channels)),
            cranelift: &|| requantize::build::<Cranelift>(black_box(&channels)),
            llvm: &|| requantize::build::<LlvmManualSimd>(black_box(&channels)),
        },
    );
}

fn audio_low_pass(c: &mut Criterion) {
    use biquad::{Q14, Section, low_pass};
    use std::f64::consts::{PI, TAU};

    const SAMPLE_RATE: usize = 48_000;
    // Bilinear-transform design of a 16th-order Butterworth low-pass at 4 kHz.
    // Start with the most damped section to limit intermediate signal peaks.
    let omega = TAU * 4000.0 / SAMPLE_RATE as f64;
    let sections: [Section; 8] = std::array::from_fn(|s| {
        let alpha = omega.sin() * ((2 * s + 1) as f64 * PI / 32.0).cos();
        let b0 = (1.0 - omega.cos()) / 2.0;
        let quantize = |x: f64| (f64::from(Q14) * x / (1.0 + alpha)).round() as i32;
        (
            quantize(b0),
            quantize(2.0 * b0),
            quantize(-2.0 * omega.cos()),
            quantize(1.0 - alpha),
        )
    });
    let pcm: Vec<_> = samples(7021)
        .take(SAMPLE_RATE)
        .enumerate()
        .map(|(i, noise)| {
            let t = i as f64 / SAMPLE_RATE as f64;
            let signal = 10_000.0 * (TAU * 440.0 * t).sin() + 3500.0 * (TAU * 6500.0 * t).sin();
            signal.round() as i32 + (noise % 513) as i32 - 256
        })
        .collect();
    // Direct form I with wider arithmetic is an independent state representation.
    let mut delays = [(0i64, 0i64, 0i64, 0i64); 8];
    let expected: Vec<_> = pcm
        .iter()
        .map(|&sample| {
            let mut x = i64::from(sample);
            for (&(b0, b1, a1, a2), state) in sections.iter().zip(&mut delays) {
                let (x1, x2, y1, y2) = *state;
                let y = (i64::from(b0) * (x + x2) + i64::from(b1) * x1
                    - i64::from(a1) * y1
                    - i64::from(a2) * y2)
                    / i64::from(Q14);
                *state = (x, x1, y, y1);
                x = y;
            }
            x as i32
        })
        .collect();
    Workload {
        name: "large_audio_low_pass",
        elements: pcm.len(),
        workspace_words: expected.len() + 2 * sections.len(),
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            let (out, state) = out.split_at_mut(pcm.len());
            let count = kernel.call(
                &[],
                &mut [
                    BufferArg::Read(&pcm),
                    BufferArg::Write(out),
                    BufferArg::Write(state),
                ],
            )?;
            usize::try_from(count).map_err(|_| "invalid audio filter buffers".into())
        },
    }
    .compare(
        c,
        || low_pass::ir(black_box(&sections)),
        Builders {
            lancy: &|| low_pass::build::<Lancy>(black_box(&sections)),
            cranelift: &|| low_pass::build::<Cranelift>(black_box(&sections)),
            llvm: &|| low_pass::build::<LlvmManualSimd>(black_box(&sections)),
        },
    );
}
