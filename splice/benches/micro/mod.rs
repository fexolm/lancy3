//! Small kernels that isolate one code generation pattern each, mostly
//! explicit SIMD. Every result is checked against a scalar Rust reference.
use super::Workload;
use criterion::Criterion;
use splice::simd::{Lane, Mask};
use splice::{BufferArg, Executable, KernelScalar, kernel};

/// Elements per input: small enough for L1, so the loops are compute bound,
/// and not a multiple of any vector width, so every SIMD kernel also runs
/// its masked remainder.
const N: usize = 4096 + 13;

/// `y = 1.25 * x + y` with a fused multiply-add and a masked remainder.
#[kernel]
fn saxpy(x: &[f32], y: &[f32], out: &mut [f32]) -> i32 {
    let n = x.len() as i32;
    if y.len() as i32 != n || (out.len() as i32) < n {
        return -1;
    }
    let width = Lane::<f32>::lanes() as i32;
    let a = Lane::<f32>::splat(1.25);
    let mut i = 0;
    while n - i >= width {
        let v = Lane::load(x, i as usize).mul_add(a, Lane::load(y, i as usize));
        v.store(out, i as usize);
        i += width;
    }
    if i < n {
        let tail = Lane::<f32>::first_n((n - i) as usize);
        let zero = Lane::<f32>::splat(0.0);
        let xs = Lane::load_select(x, i as usize, tail, zero);
        let ys = Lane::load_select(y, i as usize, tail, zero);
        xs.mul_add(a, ys).store_select(out, i as usize, tail);
    }
    n
}

/// Wrapping sum, minimum and maximum: three vector accumulators, a masked
/// remainder and horizontal reductions by halving shuffles.
#[kernel]
fn reduce3(src: &[i32], out: &mut [i32]) -> i32 {
    if out.len() < 3 {
        return -1;
    }
    let n = src.len() as i32;
    let width = Lane::<i32>::lanes() as i32;
    let mut sum = Lane::<i32>::splat(0);
    let mut lo = Lane::<i32>::splat(2147483647);
    let mut hi = Lane::<i32>::splat(-2147483647 - 1);
    let mut i = 0;
    while n - i >= width {
        let v = Lane::load(src, i as usize);
        sum += v;
        lo = lo.simd_min(v);
        hi = hi.simd_max(v);
        i += width;
    }
    if i < n {
        let tail = Lane::<i32>::first_n((n - i) as usize);
        let v = Lane::load_select(src, i as usize, tail, Lane::<i32>::splat(0));
        sum += v;
        lo = tail.select(lo.simd_min(v), lo);
        hi = tail.select(hi.simd_max(v), hi);
    }
    let mut step = width;
    while step > 1 {
        lo = lo.deinterleave_even(lo).simd_min(lo.deinterleave_odd(lo));
        hi = hi.deinterleave_even(hi).simd_max(hi.deinterleave_odd(hi));
        step /= 2;
    }
    out[0] = sum.reduce_sum();
    out[1] = lo.extract_wrapping(0);
    out[2] = hi.extract_wrapping(0);
    3
}

/// Compare, select between two arithmetic results and count active lanes.
#[kernel]
fn threshold(src: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if (out.len() as i32) <= n {
        return -1;
    }
    let width = Lane::<i32>::lanes() as i32;
    let limit = Lane::<i32>::splat(1000);
    let mut count = 0;
    let mut i = 0;
    while n - i >= width {
        let v = Lane::load(src, i as usize);
        let big: Mask<i32> = v.simd_gt(limit);
        big.select(v * Lane::<i32>::splat(3), v - Lane::<i32>::splat(7))
            .store(out, i as usize);
        count += big.count();
        i += width;
    }
    if i < n {
        let tail = Lane::<i32>::first_n((n - i) as usize);
        let v = Lane::load_select(src, i as usize, tail, Lane::<i32>::splat(0));
        let big: Mask<i32> = v.simd_gt(limit);
        big.select(v * Lane::<i32>::splat(3), v - Lane::<i32>::splat(7))
            .store_select(out, i as usize, tail);
        count += (big & tail).count();
    }
    out[n as usize] = count;
    n + 1
}

/// i32 → f32, fused scale and offset, f32 → i32.
#[kernel]
fn requantize(src: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if (out.len() as i32) < n {
        return -1;
    }
    let width = Lane::<i32>::lanes() as i32;
    let scale = Lane::<f32>::splat(0.37);
    let offset = Lane::<f32>::splat(-12.5);
    let mut i = 0;
    while n - i >= width {
        let x = Lane::load(src, i as usize).cast::<f32>();
        x.mul_add(scale, offset)
            .cast::<i32>()
            .store(out, i as usize);
        i += width;
    }
    if i < n {
        let tail = Lane::<i32>::first_n((n - i) as usize);
        let x = Lane::load_select(src, i as usize, tail, Lane::<i32>::splat(0)).cast::<f32>();
        x.mul_add(scale, offset)
            .cast::<i32>()
            .store_select(out, i as usize, tail);
    }
    n
}

/// Interleave two channels into pairs (full vectors only).
#[kernel]
fn interleave(left: &[i32], right: &[i32], out: &mut [i32]) -> i32 {
    let width = Lane::<i32>::lanes() as i32;
    let n = left.len() as i32 / width * width;
    if (right.len() as i32) < n || (out.len() as i32) < 2 * n {
        return -1;
    }
    let mut i = 0;
    while i < n {
        let a = Lane::load(left, i as usize);
        let b = Lane::load(right, i as usize);
        a.interleave_low(b).store(out, (2 * i) as usize);
        a.interleave_high(b).store(out, (2 * i + width) as usize);
        i += width;
    }
    2 * n
}

#[kernel]
fn horner(x: Lane<f32>) -> Lane<f32> {
    let c3 = Lane::<f32>::splat(0.125);
    let c2 = Lane::<f32>::splat(-0.5);
    let c1 = Lane::<f32>::splat(1.0);
    let c0 = Lane::<f32>::splat(0.25);
    x.mul_add(c3, c2).mul_add(x, c1).mul_add(x, c0)
}

#[kernel]
fn clamp_unit(x: Lane<f32>) -> Lane<f32> {
    let lo = Lane::<f32>::splat(-1.0);
    let hi = Lane::<f32>::splat(1.0);
    let below: Mask<f32> = x.simd_lt(lo);
    let above: Mask<f32> = x.simd_gt(hi);
    above.select(hi, below.select(lo, x))
}

/// Nested kernel functions on vectors, called inside the loop.
#[kernel]
fn polynomial(x: &[f32], out: &mut [f32]) -> i32 {
    let n = x.len() as i32;
    if (out.len() as i32) < n {
        return -1;
    }
    let width = Lane::<f32>::lanes() as i32;
    let mut i = 0;
    while n - i >= width {
        let p: Lane<f32> = horner(Lane::load(x, i as usize));
        let c: Lane<f32> = clamp_unit(p);
        c.store(out, i as usize);
        i += width;
    }
    if i < n {
        let tail = Lane::<f32>::first_n((n - i) as usize);
        let p: Lane<f32> = horner(Lane::load_select(x, i as usize, tail, Lane::splat(0.0)));
        let c: Lane<f32> = clamp_unit(p);
        c.store_select(out, i as usize, tail);
    }
    n
}

/// Three input slices and one output: scalar indexing or explicit lanes.
#[kernel]
fn madd3(#[comptime] simd: bool, a: &[i32], b: &[i32], c: &[i32], out: &mut [i32]) -> i32 {
    let n = a.len() as i32;
    if (b.len() as i32) < n || (c.len() as i32) < n || (out.len() as i32) < n {
        return -1;
    }
    if simd {
        let width = Lane::<i32>::lanes() as i32;
        let mut i = 0;
        while n - i >= width {
            let x =
                Lane::load(a, i as usize) * Lane::load(b, i as usize) + Lane::load(c, i as usize);
            x.store(out, i as usize);
            i += width;
        }
        while i < n {
            out[i as usize] = a[i as usize]
                .wrapping_mul(b[i as usize])
                .wrapping_add(c[i as usize]);
            i += 1;
        }
    } else {
        for i in 0..n {
            out[i as usize] = a[i as usize]
                .wrapping_mul(b[i as usize])
                .wrapping_add(c[i as usize]);
        }
    }
    n
}

/// Neighbouring reads through the same slice: each index needs a bounds
/// check unless the loop range proves it.
#[kernel]
fn stencil(src: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if n < 2 || (out.len() as i32) < n {
        return -1;
    }
    out[0] = src[0];
    for i in 1..n - 1 {
        out[i as usize] = src[(i - 1) as usize]
            .wrapping_add(src[i as usize].wrapping_mul(2))
            .wrapping_add(src[(i + 1) as usize]);
    }
    out[(n - 1) as usize] = src[(n - 1) as usize];
    n
}

/// Data-dependent branches per element.
#[kernel]
fn classify(src: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if (out.len() as i32) < n {
        return -1;
    }
    for i in 0..n {
        let x = src[i as usize];
        let y = if x < 0 {
            0 - x
        } else if x & 1 == 0 {
            x / 2
        } else if x > 3000 {
            x - 3000
        } else {
            3 * x + 1
        };
        out[i as usize] = y;
    }
    n
}

fn floats(words: &mut [i32]) -> &mut [f32] {
    // SAFETY: f32 and i32 have the same size and alignment, every bit pattern
    // is a valid f32, and the result borrows `words` exclusively.
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<f32>(), words.len()) }
}

fn length(count: u64) -> Result<usize, Box<dyn std::error::Error>> {
    usize::try_from(i32::from_word(count)).map_err(|_| "invalid benchmark buffers".into())
}

fn unary<S>(
    name: &'static str,
    input: &[i32],
    expected: &[i32],
    c: &mut Criterion,
    stage: impl Fn() -> Result<splice::TypedIr<S>, splice::BuildError>,
    builds: super::Builders<'_, S>,
) {
    Workload {
        name,
        elements: input.len(),
        workspace_words: expected.len(),
        expected: &|| expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            length(kernel.call(&[], &mut [BufferArg::read(input), BufferArg::write(out)])?)
        },
    }
    .compare(c, stage, builds);
}

pub(super) fn compare(c: &mut Criterion) {
    let ints: Vec<i32> = crate::common::xorshift(0x5eed_1234)
        .take(N)
        .map(|x| (x % 8192) as i32 - 2048)
        .collect();
    let xs: Vec<f32> = ints.iter().map(|&v| v as f32 / 2048.0).collect();
    let ys: Vec<f32> = ints.iter().rev().map(|&v| v as f32 / 512.0).collect();
    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|v| v.to_bits() as i32)
            .collect::<Vec<_>>()
    };

    let axpy: Vec<_> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| x.mul_add(1.25, y))
        .collect();
    let axpy = bits(&axpy);
    Workload {
        name: "micro_simd_saxpy",
        elements: N,
        workspace_words: N,
        expected: &|| &axpy,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            length(kernel.call(
                &[],
                &mut [
                    BufferArg::read(&xs),
                    BufferArg::read(&ys),
                    BufferArg::write(floats(out)),
                ],
            )?)
        },
    }
    .compare(
        c,
        saxpy::ir,
        builders!(B => saxpy::build::<B>(saxpy::BuildArgs {})),
    );

    let reduced = [
        ints.iter().fold(0i32, |s, &x| s.wrapping_add(x)),
        *ints.iter().min().unwrap(),
        *ints.iter().max().unwrap(),
    ];
    unary(
        "micro_simd_reduce3",
        &ints,
        &reduced,
        c,
        reduce3::ir,
        builders!(B => reduce3::build::<B>(reduce3::BuildArgs {})),
    );

    let mut selected: Vec<_> = ints
        .iter()
        .map(|&x| if x > 1000 { x * 3 } else { x - 7 })
        .collect();
    selected.push(ints.iter().filter(|&&x| x > 1000).count() as i32);
    unary(
        "micro_simd_threshold",
        &ints,
        &selected,
        c,
        threshold::ir,
        builders!(B => threshold::build::<B>(threshold::BuildArgs {})),
    );

    let quantized: Vec<_> = ints
        .iter()
        .map(|&x| (x as f32).mul_add(0.37, -12.5) as i32)
        .collect();
    unary(
        "micro_simd_i32_f32",
        &ints,
        &quantized,
        c,
        requantize::ir,
        builders!(B => requantize::build::<B>(requantize::BuildArgs {})),
    );

    let right: Vec<i32> = ints.iter().map(|&x| x ^ 0x55).collect();
    let full = N / 64 * 64; // Whole vectors at every supported width.
    let pairs: Vec<_> = (0..full).flat_map(|i| [ints[i], right[i]]).collect();
    Workload {
        name: "micro_simd_interleave",
        elements: full,
        workspace_words: pairs.len(),
        expected: &|| &pairs,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            length(kernel.call(
                &[],
                &mut [
                    BufferArg::read(&ints[..full]),
                    BufferArg::read(&right[..full]),
                    BufferArg::write(out),
                ],
            )?)
        },
    }
    .compare(
        c,
        interleave::ir,
        builders!(B => interleave::build::<B>(interleave::BuildArgs {})),
    );

    let poly: Vec<_> = xs
        .iter()
        .map(|&x| {
            let p = x.mul_add(0.125, -0.5).mul_add(x, 1.0).mul_add(x, 0.25);
            p.clamp(-1.0, 1.0)
        })
        .collect();
    let poly = bits(&poly);
    Workload {
        name: "micro_simd_nested_calls",
        elements: N,
        workspace_words: N,
        expected: &|| &poly,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            length(kernel.call(
                &[],
                &mut [BufferArg::read(&xs), BufferArg::write(floats(out))],
            )?)
        },
    }
    .compare(
        c,
        polynomial::ir,
        builders!(B => polynomial::build::<B>(polynomial::BuildArgs {})),
    );

    let addend: Vec<i32> = ints.iter().map(|&x| x.rotate_left(7)).collect();
    let fused: Vec<_> = (0..N)
        .map(|i| ints[i].wrapping_mul(right[i]).wrapping_add(addend[i]))
        .collect();
    for (simd, name) in [(false, "micro_scalar_madd3"), (true, "micro_simd_madd3")] {
        Workload {
            name,
            elements: N,
            workspace_words: N,
            expected: &|| &fused,
            equal: <[i32]>::eq,
            run: |kernel: &dyn Executable, out: &mut [i32]| {
                length(kernel.call(
                    &[],
                    &mut [
                        BufferArg::read(&ints),
                        BufferArg::read(&right),
                        BufferArg::read(&addend),
                        BufferArg::write(out),
                    ],
                )?)
            },
        }
        .compare(
            c,
            || madd3::ir(std::hint::black_box(simd)),
            builders!(B => madd3::build::<B>(madd3::BuildArgs {
                simd: std::hint::black_box(simd),
            })),
        );
    }

    let mut smoothed = ints.clone();
    for i in 1..N - 1 {
        smoothed[i] = ints[i - 1] + 2 * ints[i] + ints[i + 1];
    }
    unary(
        "micro_scalar_stencil",
        &ints,
        &smoothed,
        c,
        stencil::ir,
        builders!(B => stencil::build::<B>(stencil::BuildArgs {})),
    );

    let classes: Vec<_> = ints
        .iter()
        .map(|&x| match x {
            x if x < 0 => -x,
            x if x % 2 == 0 => x / 2,
            x if x > 3000 => x - 3000,
            x => 3 * x + 1,
        })
        .collect();
    unary(
        "micro_scalar_classify",
        &ints,
        &classes,
        c,
        classify::ir,
        builders!(B => classify::build::<B>(classify::BuildArgs {})),
    );
}
