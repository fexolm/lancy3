use super::{Builders, Workload};
use criterion::Criterion;
use splice::{BufferArg, Executable, kernel};
use std::hint::black_box;

mod loops;

// A small correction curve applied as three whole-buffer passes, or one fused pass.
#[kernel]
fn tone_curve(#[comptime] fused: bool, pixels: &[i32], out: &mut [i32]) -> i32 {
    let n = pixels.len() as i32;
    if n > out.len() as i32 {
        return -1;
    }
    if fused {
        for i in 0..n {
            let x = (pixels[i as usize] - 512).max(0);
            out[i as usize] = ((x * x + 2048) / 4096).min(1023);
        }
    } else {
        for i in 0..n {
            out[i as usize] = (pixels[i as usize] - 512).max(0);
        }
        for i in 0..n {
            let x = out[i as usize];
            out[i as usize] = (x * x + 2048) / 4096;
        }
        for i in 0..n {
            out[i as usize] = out[i as usize].min(1023);
        }
    }
    n
}

// Two statistics per sample over a periodic 31-sample neighborhood.
#[kernel]
fn local_energy(src: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if n < 31 || n > out.len() as i32 {
        return -1;
    }
    for i in 0..n {
        let mut sum = 0;
        let mut squares = 0;
        for j in 0..31 {
            let mut index = i + j - 15;
            if index < 0 {
                index += n;
            }
            if index >= n {
                index -= n;
            }
            let x = src[index as usize];
            sum += x;
            squares += x * x;
        }
        out[i as usize] = (31 * squares - sum * sum) / 961;
    }
    n
}

#[kernel]
fn score(x: i32, direction: i32, weight: i32, center: i32, width: i32) -> i32 {
    let diff = x - center;
    let response = if direction == 0 {
        diff * diff
    } else if direction < 0 {
        (0 - diff).max(0)
    } else {
        diff.max(0)
    };
    (weight * response) / width
}

// The generic scoring primitive receives ordinary scalar parameters. Its
// callers supply literals, so inlining exposes fixed modes and divisors.
#[kernel]
fn feature_score(src: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if n > out.len() as i32 {
        return -1;
    }
    for i in 0..n {
        let x = src[i as usize];
        out[i as usize] =
            score(x, -1, 17, -233, 123) + score(x, 1, 29, 71, 97) + score(x, 0, 3, 185, 721);
    }
    n
}

// Integrate sampled linear ramps with per-span lengths and slopes.
#[kernel]
fn ramp_integral(spans: &[i32], out: &mut [i32]) -> i32 {
    let n = spans.len() as i32 / 3;
    if n > out.len() as i32 {
        return -1;
    }
    for i in 0..n {
        let count = spans[(3 * i) as usize];
        let mut value = spans[(3 * i + 1) as usize];
        let step = spans[(3 * i + 2) as usize];
        let mut sum = 0i32;
        for _sample in 0..count {
            sum = sum.wrapping_add(value);
            value = value.wrapping_add(step);
        }
        out[i as usize] = sum;
    }
    n
}

// Calibration coefficients are runtime inputs. Reads in the reload version
// straddle a store to a distinct slice.
#[kernel]
fn calibrate(#[comptime] hoist: bool, src: &[i32], coefficients: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if coefficients.len() < 4 {
        return -1;
    }
    if n > out.len() as i32 / 2 {
        return -1;
    }
    if hoist {
        let scale = coefficients[0];
        let offset = coefficients[1];
        let gain = coefficients[2];
        let bias = coefficients[3];
        for i in 0..n {
            let x = src[i as usize];
            out[(2 * i) as usize] = (x * scale + offset) / 1000;
            out[(2 * i + 1) as usize] = (x * gain + bias + scale) / 3600;
        }
    } else {
        for i in 0..n {
            let x = src[i as usize];
            out[(2 * i) as usize] = (x * coefficients[0] + coefficients[1]) / 1000;
            out[(2 * i + 1) as usize] =
                (x * coefficients[2] + coefficients[3] + coefficients[0]) / 3600;
        }
    }
    2 * n
}

// Fixed-point AGC with mutable, loop-carried state and conditional updates.
// The frontend represents these locals as registers.
#[kernel]
fn envelope(src: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if n > out.len() as i32 {
        return -1;
    }
    let mut fast = 0;
    let mut slow = 0;
    let mut peak = 0;
    let mut hold = 0;
    for i in 0..n {
        let x = src[i as usize];
        let level = x.max(0 - x);
        fast += (level - fast) / 4;
        slow += (level - slow) / 64;
        if level >= peak {
            peak = level;
            hold = 16;
        } else if hold > 0 {
            hold -= 1;
        } else {
            peak -= (peak - level) / 32;
        }
        out[i as usize] = x * (4096 - fast - slow - peak) / 4096;
    }
    n
}

fn run_unary<S>(
    name: &'static str,
    input: &[i32],
    expected: &[i32],
    c: &mut Criterion,
    stage: impl Fn() -> Result<splice::TypedIr<S>, splice::BuildError>,
    builds: Builders<'_, S>,
) {
    Workload {
        name,
        elements: expected.len(),
        workspace_words: expected.len(),
        expected: &|| expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            let count = kernel.call(&[], &mut [BufferArg::read(input), BufferArg::write(out)])?;
            usize::try_from(count).map_err(|_| "invalid benchmark buffers".into())
        },
    }
    .compare(c, stage, builds);
}

pub(super) fn compare(c: &mut Criterion) {
    const N: usize = 65_539;
    let pixels: Vec<i32> = crate::common::xorshift(0x4ac2_497b)
        .take(N)
        .map(|x| (x & 4095) as i32)
        .collect();
    let samples: Vec<i32> = pixels.iter().map(|&x| x / 2 - 1024).collect();

    let curve: Vec<_> = pixels
        .iter()
        .map(|&x| {
            let light = f64::from(x.saturating_sub(512).max(0)) / 64.0;
            (light.powi(2).round() as i32).min(1023)
        })
        .collect();
    for (fused, name) in [(false, "gap_tone_passes"), (true, "gap_tone_fused")] {
        run_unary(
            name,
            &pixels,
            &curve,
            c,
            || tone_curve::ir(black_box(fused)),
            builders!(B => tone_curve::build::<B>(tone_curve::BuildArgs {
                fused: black_box(fused),
            })),
        );
    }

    let energy: Vec<_> = (0..N)
        .map(|i| {
            let window = (0..31).map(|j| i64::from(samples[(i + N + j - 15) % N]));
            let sum = window.clone().sum::<i64>();
            let sq = window.map(|x| x.pow(2)).sum::<i64>();
            ((31 * sq - sum.pow(2)) / 961) as i32
        })
        .collect();
    run_unary(
        "gap_local_energy",
        &samples,
        &energy,
        c,
        local_energy::ir,
        builders!(B => local_energy::build::<B>(local_energy::BuildArgs {})),
    );

    let scores: Vec<_> = samples
        .iter()
        .map(|&sample| {
            let x = i64::from(sample);
            (17 * (-233 - x).max(0) / 123 + 29 * (x - 71).max(0) / 97 + 3 * (x - 185).pow(2) / 721)
                as i32
        })
        .collect();
    run_unary(
        "gap_feature_score",
        &samples,
        &scores,
        c,
        feature_score::ir,
        builders!(B => feature_score::build::<B>(feature_score::BuildArgs {})),
    );

    let coefficients = [733, -179, -1261, 583];
    let calibrated: Vec<_> = samples
        .iter()
        .flat_map(|&x| [(733 * x - 179) / 1000, (-1261 * x + 1316) / 3600])
        .collect();
    for (hoist, name) in [(false, "gap_alias_reload"), (true, "gap_alias_hoisted")] {
        Workload {
            name,
            elements: samples.len(),
            workspace_words: calibrated.len(),
            expected: &|| &calibrated,
            equal: <[i32]>::eq,
            run: |kernel: &dyn Executable, out: &mut [i32]| {
                let count = kernel.call(
                    &[],
                    &mut [
                        BufferArg::read(&samples),
                        BufferArg::read(&coefficients),
                        BufferArg::write(out),
                    ],
                )?;
                usize::try_from(count).map_err(|_| "invalid calibration buffers".into())
            },
        }
        .compare(
            c,
            || calibrate::ir(black_box(hoist)),
            builders!(B => calibrate::build::<B>(calibrate::BuildArgs {
                hoist: black_box(hoist),
            })),
        );
    }

    let spans: Vec<_> = samples
        .chunks_exact(3)
        .flat_map(|row| [row[0].abs() / 2, row[1], row[2]])
        .collect();
    let integrals: Vec<_> = spans
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&row| {
            let [n, base, step] = row.map(i64::from);
            (n * base + n * (n - 1) / 2 * step) as i32
        })
        .collect();
    run_unary(
        "gap_ramp_integral",
        &spans,
        &integrals,
        c,
        ramp_integral::ir,
        builders!(B => ramp_integral::build::<B>(ramp_integral::BuildArgs {})),
    );

    let mut state = (0i64, 0i64, 0i64, 0u32);
    let adjusted: Vec<_> = samples
        .iter()
        .map(|&sample| {
            let x = i64::from(sample);
            let (fast, slow, peak, hold) = &mut state;
            let level = x.abs();
            *fast += (level - *fast) / 4;
            *slow += (level - *slow) / 64;
            (*peak, *hold) = match (level >= *peak, *hold) {
                (true, _) => (level, 16),
                (false, 0) => (*peak - (*peak - level) / 32, 0),
                (false, remaining) => (*peak, remaining - 1),
            };
            (x * (4096 - *fast - *slow - *peak) / 4096) as i32
        })
        .collect();
    run_unary(
        "gap_envelope_state",
        &samples,
        &adjusted,
        c,
        envelope::ir,
        builders!(B => envelope::build::<B>(envelope::BuildArgs {})),
    );

    loops::compare(c);
}
