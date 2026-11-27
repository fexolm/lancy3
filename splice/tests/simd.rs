use splice::hir::{Buffer, Constant, ScalarType, Simd, SimdOp};
use splice::{
    BufferArg, BufferKind, BuildError, CallError, Codegen, Executable, Trap, TypedKernel, kernel,
    simd::{Lane, Mask},
};

/// This machine's `SPLICE_SIMD` target in the mode of this test build.
fn options() -> splice::BuildOptions {
    splice::BuildOptions::host(splice::Mode::for_profile(cfg!(debug_assertions))).unwrap()
}

#[kernel]
fn add_block(lhs: &[u16], rhs: &[u16], dst: &mut [u16]) -> i32 {
    use splice::simd::Lane;
    let a: Lane<u16> = Lane::load(lhs, 1);
    let b = splice::simd::Lane::<u16>::load(rhs, 1);
    (a + b).store(dst, 1);
    Lane::<u16>::lanes() as i32
}

#[kernel]
fn indexed(src: &[u16], idx: &[u16], flags: &[u16], dst: &mut [u16]) -> i32 {
    let ids = Lane::load(idx, 0);
    let enabled = Lane::load(flags, 0).simd_ne(Lane::<u16>::splat(0));
    let fallback = Lane::<u16>::iota();
    let gathered = Lane::gather_select(src, enabled, ids, fallback);
    let ones = Lane::<u16>::splat(1);
    let result = gathered.add_masked(enabled, ones, fallback);
    result.scatter_select(dst, enabled, ids);
    if (!enabled).all() { 0 } else { 1 }
}

#[kernel]
fn scatter(indices: &[u16], flags: &[u16], values: &[u16], dst: &mut [u16]) -> i32 {
    let mask = Lane::load(flags, 0).simd_ne(Lane::<u16>::splat(0));
    Lane::load(values, 0).scatter_select(dst, mask, Lane::load(indices, 0));
    mask.any() as i32
}

#[kernel]
fn scatter_all(indices: &[i32], dst: &mut [i32]) -> i32 {
    Lane::<i32>::iota().scatter(dst, Lane::load(indices, 0));
    0
}

#[kernel]
fn copy_selected(src: &[i32], flags: &[i32], dst: &mut [i32], offset: i32) -> i32 {
    let active = Lane::load(flags, 0).simd_ne(Lane::<i32>::splat(0));
    let values = Lane::load_select(src, offset as usize, active, Lane::<i32>::iota());
    values.store_select(dst, offset as usize, active);
    values.last()
}

#[kernel]
fn tail(src: &[u16], flags: &[u16], dst: &mut [u16]) -> i32 {
    let mask = Lane::load(flags, 0).simd_ne(Lane::<u16>::splat(0));
    let x = Lane::load_select(src, 1, mask, Lane::<u16>::splat(77));
    x.store(dst, 0);
    x.store_select(dst, 1, mask);
    0
}

#[kernel]
fn float_width(a: &[f32], b: &mut [f32], c: &[f64], d: &mut [f64], flag: i32) -> i32 {
    let x = Lane::load(a, 0);
    let mut y = if flag != 0 {
        x + x
    } else {
        Lane::<f32>::splat(0.)
    };
    y += Lane::<f32>::splat(1.);
    y.store(b, 0);
    let z = Lane::load(c, 0);
    (z * z).store(d, 0);
    Lane::<f32>::lanes() as i32 - Lane::<f64>::lanes() as i32
}

fn kernels<S, E: Executable + 'static>(
    out: &mut Vec<Box<dyn Executable>>,
    direct: Result<TypedKernel<S, E>, BuildError>,
    portable: Result<TypedKernel<S, E>, BuildError>,
) {
    out.push(Box::new(direct.unwrap().into_inner()));
    out.push(Box::new(portable.unwrap().into_inner()));
}
macro_rules! compile_all {
    ($name:ident) => {{
        let ir = $name::ir().unwrap();
        let mut v = Vec::new();
        splice_testkit::each_backend!(B => kernels(
            &mut v,
            $name::build::<B>($name::BuildArgs {}),
            B::compile(&ir),
        ));
        v
    }};
}

#[kernel]
fn typed_float_lanes(
    input: &[f32],
    fused: &mut [f32],
    widened: &mut [f64],
    round_trip: &mut [f32],
    integers: &mut [u32],
) -> i32 {
    let x = Lane::<f32>::load(input, 0);
    x.mul_add(Lane::<f32>::splat(0.999_999_9), Lane::<f32>::splat(-1.0))
        .store(fused, 0);
    let lo = x.widen_low::<f64>();
    let hi = x.widen_high::<f64>();
    lo.store(widened, 0);
    hi.store(widened, Lane::<f64>::lanes());
    lo.narrow::<f32>(hi).store(round_trip, 0);
    x.sqrt().cast::<u32>().store(integers, 0);
    x.cast::<i32>().reduce_sum()
}

#[test]
fn typed_simd_conversions_and_fused_arithmetic_match_rust() {
    let n = Lane::<f32>::lanes();
    let input: Vec<_> = [
        1.000_000_1f32,
        -0.0,
        9.0,
        -4.0,
        f32::INFINITY,
        f32::NAN,
        123.75,
        -123.75,
    ]
    .into_iter()
    .cycle()
    .take(n)
    .collect();
    for kernel in compile_all!(typed_float_lanes) {
        let mut fused = vec![0.0f32; n];
        let mut widened = vec![0.0f64; n];
        let mut round_trip = vec![0.0f32; n];
        let mut integers = vec![0u32; n];
        let expected = input
            .iter()
            .fold(0i32, |sum, &x| sum.wrapping_add(x as i32));
        let result = splice_testkit::call_i32(
            &kernel,
            &[],
            &mut [
                BufferArg::from(input.as_slice()),
                BufferArg::from(fused.as_mut_slice()),
                BufferArg::from(widened.as_mut_slice()),
                BufferArg::from(round_trip.as_mut_slice()),
                BufferArg::from(integers.as_mut_slice()),
            ],
        );
        assert_eq!(result, Ok(expected));
        for i in 0..n {
            let x = input[i];
            if x.is_nan() {
                assert!(round_trip[i].is_nan() && widened[i].is_nan() && fused[i].is_nan());
            } else {
                assert_eq!(round_trip[i].to_bits(), x.to_bits());
                assert_eq!(widened[i].to_bits(), f64::from(x).to_bits());
                assert_eq!(fused[i].to_bits(), x.mul_add(0.999_999_9, -1.0).to_bits());
            }
            assert_eq!(integers[i], x.sqrt() as u32);
        }
        assert_ne!(fused[0], 0.0);
    }
}

#[kernel]
fn unsigned_words(input: &[u32], out: &mut [u32], narrow: &mut [u16]) -> i32 {
    let words = Lane::<u32>::load(input, 0);
    let high = words.simd_ge(Lane::<u32>::splat(0x80000000));
    high.select(words >> 1, words).store(out, 0);
    words.narrow::<u16>(words).store(narrow, 0);
    0
}

#[test]
fn unsigned_lanes_compare_shift_and_narrow_across_backends() {
    let n = Lane::<u32>::lanes();
    let input: Vec<_> = [0, 0x7fffffff, 0x80000000, 0xd13e_beef, u32::MAX]
        .into_iter()
        .cycle()
        .take(n)
        .collect();
    for kernel in compile_all!(unsigned_words) {
        let mut out = vec![21u32; n + 1];
        let mut narrow = vec![43u16; n * 2 + 1];
        assert_eq!(
            splice_testkit::call_i32(
                &kernel,
                &[],
                &mut [
                    BufferArg::from(input.as_slice()),
                    BufferArg::from(out.as_mut_slice()),
                    BufferArg::from(narrow.as_mut_slice()),
                ]
            ),
            Ok(0)
        );
        for (i, &x) in input.iter().enumerate() {
            assert_eq!(out[i], if x >= 0x80000000 { x >> 1 } else { x });
            assert_eq!(narrow[i], x as u16);
            assert_eq!(narrow[n + i], x as u16);
        }
        assert_eq!(out[n], 21);
        assert_eq!(narrow[2 * n], 43);
    }
}

#[kernel]
fn unsigned_indexed_load(input: &[u32], index: &[u32], out: &mut [u32], count: i32) -> i32 {
    let active = Lane::<u32>::first_n(count as usize);
    Lane::gather_select(input, active, Lane::load(index, 0), Lane::<u32>::splat(9)).store(out, 0);
    0
}

#[test]
fn unsigned_indices_above_i32_max_fault_only_when_active() {
    let n = Lane::<u32>::lanes();
    let mut ids = vec![u32::MAX; n];
    ids[0] = 0;
    for kernel in compile_all!(unsigned_indexed_load) {
        let mut out = vec![21u32; n];
        for count in [1, 2] {
            out.fill(21);
            let actual = splice_testkit::call_i32(
                &kernel,
                &[count],
                &mut [
                    BufferArg::from(&[17u32][..]),
                    BufferArg::from(ids.as_slice()),
                    BufferArg::from(out.as_mut_slice()),
                ],
            );
            if count == 1 {
                assert_eq!(actual, Ok(0));
                assert_eq!(out[0], 17);
                assert!(out[1..].iter().all(|&x| x == 9));
            } else {
                assert_eq!(actual, Err(CallError::Trap(Trap::Bounds)));
                assert!(out.iter().all(|&x| x == 21));
            }
        }
    }
}

#[kernel]
fn signed_bounds(lhs: &[i32], rhs: &[i32], mins: &mut [i32], maxs: &mut [i32]) -> i32 {
    let a = Lane::load(lhs, 0);
    let b = Lane::load(rhs, 0);
    a.simd_min(b).store(mins, 1);
    a.simd_max(b).store(maxs, 1);
    Lane::<i32>::lanes() as i32
}

#[test]
fn signed_vector_min_max_stay_explicit_and_preserve_lane_order() {
    use splice::hir::{Instr, SimdOp};

    let ir = signed_bounds::ir().unwrap();
    let ops: Vec<_> = ir
        .instructions()
        .iter()
        .filter_map(|instr| match instr {
            Instr::Simd { inst, .. } => Some(inst.op),
            _ => None,
        })
        .collect();
    assert_eq!(
        ops,
        [
            SimdOp::Load(splice::hir::Buffer(0)),
            SimdOp::Load(splice::hir::Buffer(1)),
            SimdOp::Min,
            SimdOp::Store(splice::hir::Buffer(2)),
            SimdOp::Max,
            SimdOp::Store(splice::hir::Buffer(3)),
            SimdOp::Lanes,
        ]
    );
    let lanes = Lane::<i32>::lanes();
    let pairs = [
        (i32::MIN, i32::MAX),
        (i32::MAX, i32::MIN),
        (-9, 4),
        (0, -1),
        (12, 12),
    ];
    let (lhs, rhs): (Vec<_>, Vec<_>) = pairs.into_iter().cycle().take(lanes).unzip();
    let expected_min: Vec<_> = lhs.iter().zip(&rhs).map(|(&a, &b)| a.min(b)).collect();
    let expected_max: Vec<_> = lhs.iter().zip(&rhs).map(|(&a, &b)| a.max(b)).collect();
    for kernel in compile_all!(signed_bounds) {
        let mut mins = vec![23; lanes + 2];
        let mut maxs = vec![23; lanes + 2];
        let result = splice_testkit::call_i32(
            &kernel,
            &[],
            &mut [
                BufferArg::read(&lhs),
                BufferArg::read(&rhs),
                BufferArg::write(&mut mins),
                BufferArg::write(&mut maxs),
            ],
        );
        assert_eq!(result, Ok(lanes as i32));
        assert_eq!(mins[1..=lanes], expected_min);
        assert_eq!(maxs[1..=lanes], expected_max);
        assert_eq!(
            [mins[0], mins[lanes + 1], maxs[0], maxs[lanes + 1]],
            [23; 4]
        );
    }
}

#[kernel]
fn rotate_words(#[comptime] count: u32, src: &[i32], out: &mut [i32]) -> i32 {
    Lane::load(src, 0).rotate_left(count).store(out, 0);
    Lane::<i32>::lanes() as i32
}

#[test]
fn vector_rotate_wraps_the_count_and_rotates_signed_lane_bits() {
    use splice::hir::{Instr, SimdOp};

    let lanes = Lane::<i32>::lanes();
    let source: Vec<_> = [
        i32::MIN,
        i32::MAX,
        -1,
        0,
        1,
        0x8000_0001u32 as i32,
        0x1234_5678,
    ]
    .into_iter()
    .cycle()
    .take(lanes)
    .collect();
    for count in [0, 1, 13, 15, 31, 32, 33, u32::MAX] {
        let ir = rotate_words::ir(count).unwrap();
        assert!(ir.instructions().iter().any(|instr| {
            matches!(instr, Instr::Simd { inst, .. } if inst.op == SimdOp::RotateLeft(count & 31))
        }));
        let mut backends = Vec::new();
        splice_testkit::each_backend!(B => kernels(
            &mut backends,
            rotate_words::build::<B>(rotate_words::BuildArgs { count }),
            B::compile(&ir),
        ));
        let expected: Vec<_> = source.iter().map(|v| v.rotate_left(count)).collect();
        for kernel in backends {
            let mut out = vec![23; lanes + 1];
            let result = splice_testkit::call_i32(
                &kernel,
                &[],
                &mut [BufferArg::read(&source), BufferArg::write(&mut out)],
            );
            assert_eq!(result, Ok(lanes as i32));
            assert_eq!(out[..lanes], expected, "rotation count {count}");
            assert_eq!(out[lanes], 23);
        }
    }
}

#[kernel]
fn xor_complement(a: &[i32], b: &[i32], c: &[i32], out: &mut [i32]) -> i32 {
    let x = Lane::load(a, 0);
    let y = Lane::load(b, 0);
    let z = Lane::load(c, 0);
    (!(x ^ y ^ z)).store(out, 0);
    Lane::<i32>::lanes() as i32
}

#[test]
fn triple_xor_and_complement_preserve_every_lane_and_the_tail() {
    let lanes = Lane::<i32>::lanes();
    let a: Vec<_> = (0..64).collect();
    let b: Vec<_> = a.iter().map(|&x| 0x7654_3210 ^ (x << 16)).collect();
    let c: Vec<_> = a.iter().map(|&x| !(x << 8)).collect();
    let mut expected = vec![77; a.len()];
    for i in 0..lanes {
        expected[i] = !(a[i] ^ b[i] ^ c[i]);
    }
    for kernel in compile_all!(xor_complement) {
        let mut out = vec![77; a.len()];
        let result = splice_testkit::call_i32(
            &kernel,
            &[],
            &mut [
                a.as_slice().into(),
                b.as_slice().into(),
                c.as_slice().into(),
                out.as_mut_slice().into(),
            ],
        );
        assert_eq!(result, Ok(lanes as i32));
        assert_eq!(out, expected);
    }
}

#[kernel]
fn floating_iotas(singles: &mut [f32], doubles: &mut [f64]) -> i32 {
    Lane::<f32>::iota().store(singles, 0);
    Lane::<f64>::iota().store(doubles, 0);
    Lane::<f32>::lanes() as i32
}

#[test]
fn floating_iota_uses_numeric_lane_indices() {
    let (n, m) = (Lane::<f32>::lanes(), Lane::<f64>::lanes());
    let mut expected_singles = [-1.0_f32; 64];
    let mut expected_doubles = [-1.0_f64; 64];
    for (i, value) in expected_singles[..n].iter_mut().enumerate() {
        *value = i as f32;
    }
    for (i, value) in expected_doubles[..m].iter_mut().enumerate() {
        *value = i as f64;
    }
    for kernel in compile_all!(floating_iotas) {
        let mut singles = [-1.0_f32; 64];
        let mut doubles = [-1.0_f64; 64];
        assert_eq!(
            splice_testkit::call_i32(
                &kernel,
                &[],
                &mut [singles.as_mut_slice().into(), doubles.as_mut_slice().into()]
            ),
            Ok(n as i32)
        );
        assert_eq!(singles, expected_singles);
        assert_eq!(doubles, expected_doubles);
    }
}

#[kernel]
fn greater_lanes(a: Lane<i32>, b: Lane<i32>) -> Mask<i32> {
    a.simd_ge(b)
}

#[kernel]
fn choose_lanes(a: Lane<i32>, b: Lane<i32>, keep_a: Mask<i32>) -> Lane<i32> {
    keep_a.select(a, b)
}

#[kernel]
fn column_batches(left: &[i32], right: &[i32], out: &mut [i32]) -> i32 {
    let a = Lane::load(left, 0);
    let b = Lane::load(right, 0);
    let width = Lane::<i32>::lanes() as i32;
    a.interleave_low(b).store(out, 0);
    a.interleave_high(b).store(out, width as usize);
    a.deinterleave_even(b).store(out, (2 * width) as usize);
    a.deinterleave_odd(b).store(out, (3 * width) as usize);
    let mask: Mask<i32> = greater_lanes(a, b);
    let selected: Lane<i32> = choose_lanes(a, b, mask);
    let sums = selected.prefix_sum();
    sums.store(out, (4 * width) as usize);
    sums.last()
}

#[test]
fn column_permutations_and_wrapping_prefixes_cross_the_full_vector() {
    let width = Lane::<i32>::lanes();
    let mut left: Vec<i32> = (0..width as i32).map(|i| i * 7).collect();
    left[0] = i32::MAX - 2;
    let right: Vec<_> = (0..width as i32)
        .map(|i| if i % 2 == 0 { -100 - i } else { 100 + i })
        .collect();
    let mut want: Vec<_> = left
        .iter()
        .zip(&right)
        .flat_map(|(&a, &b)| [a, b])
        .collect();
    // Unzip the two original inputs independently of the zip operation.
    for parity in [0, 1] {
        want.extend(left.iter().chain(&right).skip(parity).step_by(2).copied());
    }
    let mut total = 0i32;
    for (&a, &b) in left.iter().zip(&right) {
        total = total.wrapping_add(a.max(b));
        want.push(total);
    }
    let mut out = vec![0; want.len()];
    assert_eq!(column_batches(&left, &right, &mut out), total);
    assert_eq!(out, want);
    for code in compile_all!(column_batches) {
        out.fill(-1);
        assert_eq!(
            splice_testkit::call_i32(
                &code,
                &[],
                &mut [
                    (&left[..]).into(),
                    (&right[..]).into(),
                    (&mut out[..]).into()
                ]
            )
            .unwrap(),
            total
        );
        assert_eq!(out, want);
    }
}

#[kernel]
fn compact_high_words(src: &[i32], flags: &[i32], out: &mut [i32], offset: i32) -> i32 {
    let words = Lane::load(src, 0);
    let keep = Lane::load(flags, 0).simd_ne(Lane::<i32>::splat(0));
    (words >> 16).compress_store(out, offset as usize, keep);
    keep.count()
}

#[kernel]
fn rearrange_columns(src: &[i32], flags: &[i32], indices: &[i32], out: &mut [i32]) -> i32 {
    let values = Lane::load(src, 0);
    let keep = Lane::load(flags, 0).simd_ne(Lane::<i32>::splat(0));
    let compact = values.compress(keep);
    compact.store(out, 0);
    let idx = Lane::load(indices, 0);
    values.swizzle(idx).store(out, Lane::<i32>::lanes());
    compact.swizzle(idx).last()
}

#[kernel]
fn batch_alignment(src: &[i32], offset: i32) -> i32 {
    Lane::<i32>::alignment_prefix(src, offset as usize) as i32
}

#[test]
fn alignment_prefix_uses_the_actual_subslice_and_offset() {
    let width = Lane::<i32>::lanes();
    let align = width * size_of::<i32>();
    let backing = vec![37; 3 * width + 1];
    let kernels = compile_all!(batch_alignment);
    for start in 0..width {
        for len in [0, 1, width - 1, 2 * width + 1] {
            let slice = &backing[start..start + len];
            for offset in [0, len / 2, len] {
                let expected = slice.as_ptr().wrapping_add(offset).align_offset(align) as i32;
                assert_eq!(batch_alignment(slice, offset as i32), expected);
                for kernel in &kernels {
                    let actual =
                        splice_testkit::call_i32(&kernel, &[offset as i32], &mut [slice.into()])
                            .unwrap();
                    assert_eq!(actual, expected, "start={start} len={len} offset={offset}");
                }
            }
        }
    }
}

#[test]
fn compress_and_swizzle_return_vectors_with_zero_in_unused_lanes() {
    use splice::hir::{Instr, SimdOp};

    let ir = rearrange_columns::ir().unwrap();
    for wanted in [SimdOp::Compress, SimdOp::Swizzle] {
        assert!(ir.instructions().iter().any(
            |instr| matches!(instr, Instr::Simd { dst: Some(_), inst, .. } if inst.op == wanted)
        ));
    }
    let width = Lane::<i32>::lanes();
    let n = width as i32;
    let values: Vec<_> = (0..n).map(|i| 0x1734_2695 + i * 0x0103_0709).collect();
    // On a 512-bit target, select both ends of each 128-bit part. Out-of-range
    // indices with the same low bits must still zero the destination lane.
    let lookups = [
        0,
        4,
        8,
        12,
        3,
        7,
        11,
        15,
        n / 2,
        n / 2,
        n,
        32,
        35,
        128,
        143,
        -1,
        i32::MIN,
        i32::MAX,
    ];
    let kernels = compile_all!(rearrange_columns);
    for phase in 0..lookups.len() {
        let mask: Vec<_> = (0..width)
            .map(|i| {
                i32::from(match phase % 4 {
                    0 => false,
                    1 => true,
                    2 => i == width - 1,
                    _ => (i + phase) % 3 != 1,
                })
            })
            .collect();
        let indices: Vec<_> = (0..width)
            .map(|i| lookups[(i + phase) % lookups.len()])
            .collect();
        let mut compact: Vec<_> = values
            .iter()
            .zip(&mask)
            .filter(|&(_, &flag)| flag != 0)
            .map(|(&v, _)| v)
            .collect();
        compact.resize(width, 0);
        let lookup = |data: &[i32], index: i32| data.get(index as usize).copied().unwrap_or(0);
        let last = lookup(&compact, indices[width - 1]);
        let mut want = compact;
        want.extend(indices.iter().map(|&i| lookup(&values, i)));
        let mut out = vec![12345; 2 * width];
        assert_eq!(rearrange_columns(&values, &mask, &indices, &mut out), last);
        assert_eq!(out, want);
        for kernel in &kernels {
            out.fill(12345);
            let result = splice_testkit::call_i32(
                &kernel,
                &[],
                &mut [
                    (&values[..]).into(),
                    (&mask[..]).into(),
                    (&indices[..]).into(),
                    (&mut out[..]).into(),
                ],
            )
            .unwrap();
            assert_eq!(result, last);
            assert_eq!(out, want);
        }
    }
}

#[test]
fn compress_store_counts_lanes_and_writes_the_fitting_prefix_in_order() {
    let width = Lane::<i32>::lanes();
    let src: Vec<_> = (0..width as i32)
        .map(|i| match i % 4 {
            0 => i32::MIN,
            1 => ((-9 - i) << 16) | i,
            2 => i32::MAX,
            _ => ((20 + i) << 16) | i,
        })
        .collect();
    let empty = vec![0; width];
    let full = vec![1; width];
    let mixed: Vec<_> = (0..width)
        .map(|i| if i % 3 == 1 { 0 } else { -1 })
        .collect();
    let kernels = compile_all!(compact_high_words);
    let count = mixed.iter().filter(|&&v| v != 0).count();
    for (flags, offset, len) in [
        (&empty, -1i32, 0),
        (&mixed, -1, width),
        (&full, 0, width),
        (&mixed, 2, 2 + count),
        (&mixed, 2, 2 + count - 1),
        (&mixed, 1, 1),
    ] {
        let selected: Vec<_> = src
            .iter()
            .zip(flags)
            .filter(|&(_, &f)| f != 0)
            .map(|(&v, _)| v >> 16)
            .collect();
        let start = offset as usize;
        let fits = len.saturating_sub(start).min(selected.len());
        // Keep guard words on both sides of the writable window.
        let mut expected = vec![0x1234_5678; len + 2];
        if fits > 0 {
            expected[1 + start..1 + start + fits].copy_from_slice(&selected[..fits]);
        }
        let expected_rows = (fits == selected.len()).then_some(selected.len() as i32);
        let mut out = vec![0x1234_5678; len + 2];
        let rust = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compact_high_words(&src, flags, &mut out[1..1 + len], offset)
        }));
        assert_eq!(rust.ok(), expected_rows);
        assert_eq!(out, expected);
        for k in &kernels {
            out.fill(0x1234_5678);
            let result = splice_testkit::call_i32(
                &k,
                &[offset],
                &mut [
                    (&src[..]).into(),
                    (&flags[..]).into(),
                    (&mut out[1..1 + len]).into(),
                ],
            );
            assert_eq!(result.ok(), expected_rows);
            assert_eq!(out, expected);
        }
    }
}

#[test]
fn scalable_add_wraps_at_u16_and_preserves_prefix_on_short_store() {
    let w = Lane::<u16>::lanes();
    let a: Vec<u16> = (0..w + 2)
        .map(|i| if i == w { u16::MAX } else { i as u16 * 3 })
        .collect();
    let b = vec![1u16; w + 2];
    let mut expected = vec![901; w + 2];
    for j in 1..=w {
        expected[j] = a[j].wrapping_add(b[j]);
    }
    let mut native = vec![901; w + 2];
    assert_eq!(add_block(&a, &b, &mut native), w as i32);
    assert_eq!(native, expected);
    let ir = add_block::ir().unwrap();
    assert_eq!(
        ir.buffer_kinds(),
        [
            BufferKind::new(ScalarType::U16, false),
            BufferKind::new(ScalarType::U16, false),
            BufferKind::new(ScalarType::U16, true)
        ]
    );
    let ops = splice_testkit::simd_ops(&ir);
    for op in [
        SimdOp::Load(Buffer(0)),
        SimdOp::Add,
        SimdOp::Store(Buffer(2)),
        SimdOp::Lanes,
    ] {
        assert!(
            ops.contains(&Simd {
                element: ScalarType::U16,
                op
            }),
            "{ops:?}"
        );
    }
    for k in compile_all!(add_block) {
        for len in [w + 2, w + 1, w, 1, 0] {
            let mut got = vec![901u16; w + 2];
            let mut bufs = [
                BufferArg::from(&a[..]),
                BufferArg::from(&b[..]),
                BufferArg::from(&mut got[..len]),
            ];
            let r = splice_testkit::call_i32(&k, &[], &mut bufs);
            if len > w {
                assert_eq!(r.unwrap(), w as i32);
            } else {
                assert!(r.is_err());
            }
            let mut want = vec![901; w + 2];
            want[..len].copy_from_slice(&expected[..len]);
            assert_eq!(got, want);
        }
        let mut wrong = vec![0i32; w + 1];
        assert!(
            splice_testkit::call_i32(
                &k,
                &[],
                &mut [
                    (&a[..]).into(),
                    (&b[..]).into(),
                    BufferArg::write(&mut wrong)
                ]
            )
            .is_err()
        );
    }
}
#[test]
fn masked_indices_skip_inactive_accesses_and_write_in_order() {
    let w = Lane::<u16>::lanes();
    let mut ids = vec![u16::MAX; w];
    ids[..5].copy_from_slice(&[1, u16::MAX, 1, 3, 0]);
    let mut flags = vec![0u16; w];
    flags[..5].copy_from_slice(&[1, 0, 1, 1, 1]);
    let mut vals = vec![0u16; w];
    vals[..5].copy_from_slice(&[11, 99, 22, 33, 44]);
    let ks = compile_all!(scatter);
    for active in [true, false] {
        flags[3] = u16::from(active);
        let mut want = [90, 91, 92];
        for i in (0..w).filter(|&i| flags[i] != 0) {
            let Some(d) = want.get_mut(usize::from(ids[i])) else {
                break;
            };
            *d = vals[i];
        }
        let mut rust = [90, 91, 92];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scatter(&ids, &flags, &vals, &mut rust)
        }));
        assert_eq!(r.is_err(), active);
        assert_eq!(rust, want);
        for k in &ks {
            let mut got = [90u16, 91, 92];
            let r = splice_testkit::call_i32(
                &k,
                &[],
                &mut [
                    (&ids[..]).into(),
                    (&flags[..]).into(),
                    (&vals[..]).into(),
                    (&mut got[..]).into(),
                ],
            );
            assert_eq!(r.is_err(), active);
            assert_eq!(got, want);
        }
    }
    let ks = compile_all!(indexed);
    for n in [0, 2] {
        flags.fill(0);
        if n != 0 {
            flags[0] = 1;
            flags[w - 1] = 1;
            ids[w - 1] = 0;
        }
        let src = &[21u16, 41][..n];
        let want = if n == 0 { vec![] } else { vec![22, 42] };
        let mut rust = vec![99; n];
        assert_eq!(indexed(src, &ids, &flags, &mut rust), i32::from(n != 0));
        assert_eq!(rust, want);
        for k in &ks {
            let mut got = vec![99u16; n];
            assert_eq!(
                splice_testkit::call_i32(
                    &k,
                    &[],
                    &mut [
                        src.into(),
                        (&ids[..]).into(),
                        (&flags[..]).into(),
                        (&mut got[..]).into()
                    ]
                )
                .unwrap(),
                i32::from(n != 0)
            );
            assert_eq!(got, want);
        }
    }
}

#[test]
fn unmasked_scatter_writes_aliases_in_lane_order_up_to_the_first_fault() {
    let width = Lane::<i32>::lanes();
    let kernels = compile_all!(scatter_all);
    for fault in [None, Some((2, -1)), Some((width - 1, 3))] {
        let mut ids: Vec<_> = (0..width).map(|i| (i / 2 % 3) as i32).collect();
        if let Some((lane, bad)) = fault {
            ids[lane] = bad;
        }
        let mut want = [71; 5];
        for (lane, &index) in ids.iter().enumerate() {
            let Some(cell) = want[1..4].get_mut(index as usize) else {
                break;
            };
            *cell = lane as i32;
        }
        let mut out = [71; 5];
        let native = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scatter_all(&ids, &mut out[1..4])
        }));
        assert_eq!(native.is_err(), fault.is_some());
        assert_eq!(out, want);
        for kernel in &kernels {
            out.fill(71);
            let result = splice_testkit::call_i32(
                &kernel,
                &[],
                &mut [ids.as_slice().into(), (&mut out[1..4]).into()],
            );
            assert_eq!(result.is_err(), fault.is_some());
            assert_eq!(out, want);
        }
    }
}

#[test]
fn masked_contiguous_access_skips_empty_masks_and_preserves_the_valid_store_prefix() {
    let w = Lane::<i32>::lanes();
    let src: Vec<_> = (0..=w).map(|i| 100 + i as i32).collect();
    let last = 1_u32 << (w - 1);
    let ends = 1 | last;
    let fallback = Some(w as i32 - 1);
    let kernels = compile_all!(copy_selected);
    for (offset, src_len, dst_len, mask, written, want_result) in [
        (-1, 0, 0, 0, 0, fallback),
        (i32::MAX, 0, 0, 0, 0, fallback),
        (-1, w, w, last, 0, None),
        (i32::MAX, w, w, ends, 0, None),
        (1, w + 1, w + 1, ends, ends, Some(src[w])),
        (1, w, w, 1, 1, fallback),
        (1, w, w + 1, ends, 0, None), // load fails before the store
        (1, w + 1, w, ends, 1, None), // store commits its first active lane
    ] {
        let flags: Vec<i32> = (0..w).map(|i| ((mask >> i) & 1) as i32).collect();
        let src = &src[..src_len];
        let mut want = vec![-91; dst_len + 2];
        for i in 0..w {
            if written & (1 << i) != 0 {
                let index = offset as usize + i;
                want[1 + index] = src[index];
            }
        }
        let mut out = vec![-91; dst_len + 2];
        let native = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            copy_selected(src, &flags, &mut out[1..1 + dst_len], offset)
        }));
        assert_eq!(native.ok(), want_result);
        assert_eq!(out, want);
        for kernel in &kernels {
            out.fill(-91);
            let result = splice_testkit::call_i32(
                &kernel,
                &[offset],
                &mut [
                    src.into(),
                    flags.as_slice().into(),
                    (&mut out[1..1 + dst_len]).into(),
                ],
            );
            assert_eq!(result.ok(), want_result);
            assert_eq!(out, want);
        }
    }
}

#[test]
fn masked_tail_uses_each_fallback_and_skips_writes() {
    let w = Lane::<u16>::lanes();
    let mut flags = vec![0u16; w];
    flags[0] = 1;
    let mut want = vec![77u16; w + 2];
    want[0] = 12;
    want[1] = 12;
    want[w..].fill(100);
    for k in compile_all!(tail) {
        let mut got = vec![100u16; w + 2];
        assert_eq!(
            splice_testkit::call_i32(
                &k,
                &[],
                &mut [
                    (&[7u16, 12][..]).into(),
                    (&flags[..]).into(),
                    (&mut got[..]).into()
                ]
            )
            .unwrap(),
            0
        );
        assert_eq!(got, want);
    }
}

#[test]
fn element_size_controls_lane_count_and_typed_joins() {
    let (n, m) = (Lane::<f32>::lanes(), Lane::<f64>::lanes());
    assert_eq!(n, 2 * m);
    let a: Vec<f32> = (0..n).map(|i| i as f32 - 4.5).collect();
    let c: Vec<f64> = (0..m).map(|i| i as f64 + 0.25).collect();
    let ks = compile_all!(float_width);
    for flag in [0, 1] {
        let by: Vec<_> = a
            .iter()
            .map(|x| if flag == 0 { 1.0 } else { x + x + 1.0 })
            .collect();
        let dy: Vec<_> = c.iter().map(|x| x * x).collect();
        let mut rust_b = vec![0.; n];
        let mut rust_d = vec![0.; m];
        assert_eq!(
            float_width(&a, &mut rust_b, &c, &mut rust_d, flag),
            m as i32
        );
        assert_eq!(rust_b, by);
        assert_eq!(rust_d, dy);
        for k in &ks {
            let mut b = vec![0f32; n];
            let mut d = vec![0f64; m];
            assert_eq!(
                splice_testkit::call_i32(
                    &k,
                    &[flag],
                    &mut [
                        (&a[..]).into(),
                        (&mut b[..]).into(),
                        (&c[..]).into(),
                        (&mut d[..]).into()
                    ]
                )
                .unwrap(),
                m as i32
            );
            assert_eq!(b, by);
            assert_eq!(d, dy);
        }
    }
}

#[kernel]
fn gather_fallback(src: &[i32], idx: &[i32], flags: &[i32], out: &mut [i32], bias: i32) -> i32 {
    let bits = Lane::load(flags, 0);
    let off = bits.simd_eq(Lane::splat(0));
    let yes = !off & (off | !off);
    let ids = Lane::load(idx, 0);
    let fallback = Lane::<i32>::iota() + Lane::splat(bias);
    let value = Lane::gather_select(src, yes, ids, fallback);
    (value ^ Lane::<i32>::splat(2)).store(out, 0);
    (yes ^ !off).any() as i32
}

#[kernel]
fn gather_u16_mask(src: &[u16], ids: &[u16], flags: &[u16], dst: &mut [u16]) -> i32 {
    let active = Lane::load(flags, 0).simd_ne(Lane::<u16>::splat(0));
    let fallback = Lane::<u16>::iota() + Lane::splat(421);
    Lane::gather_select(src, active, Lane::load(ids, 0), fallback).store(dst, 0);
    0
}

#[test]
fn masked_u16_gather_visits_every_set_bit_and_keeps_each_fallback() {
    let width = Lane::<u16>::lanes();
    let values: Vec<_> = (0..width as u16).map(|i| 1000 + 7 * i).collect();
    let kernels = compile_all!(gather_u16_mask);
    for bits in [0, 1 << (width - 1), 0x8001_8001, 0xaaaa_5555, u32::MAX] {
        let src = if bits == 0 { &[] } else { values.as_slice() };
        let flags: Vec<u16> = (0..width).map(|i| ((bits >> i) & 1) as u16).collect();
        let mut ids = vec![u16::MAX; width];
        let mut want = vec![99; width + 2];
        for (i, &flag) in flags.iter().enumerate() {
            want[1 + i] = if flag != 0 {
                ids[i] = (width - i - 1) as u16;
                src[ids[i] as usize]
            } else {
                421 + i as u16
            };
        }
        let mut out = vec![99; width + 2];
        assert_eq!(gather_u16_mask(src, &ids, &flags, &mut out[1..=width]), 0);
        assert_eq!(out, want);
        for kernel in &kernels {
            out.fill(99);
            assert_eq!(
                splice_testkit::call_i32(
                    &kernel,
                    &[],
                    &mut [
                        src.into(),
                        ids.as_slice().into(),
                        flags.as_slice().into(),
                        (&mut out[1..=width]).into()
                    ]
                ),
                Ok(0)
            );
            assert_eq!(out, want, "mask={bits:#010x}");
        }
    }
}

#[test]
fn inactive_gather_retains_each_fallback_lane() {
    let w = Lane::<i32>::lanes();
    let mut ids = vec![i32::MAX; w];
    let mut mask = vec![0; w];
    ids[1] = 0;
    mask[1] = 7;
    let want: Vec<_> = (0..w)
        .map(|i| if i == 1 { 11 ^ 2 } else { (i as i32 + 17) ^ 2 })
        .collect();
    let mut native = vec![0; w];
    assert_eq!(gather_fallback(&[11], &ids, &mask, &mut native, 17), 0);
    assert_eq!(native, want);
    for k in compile_all!(gather_fallback) {
        let mut out = vec![0; w];
        assert_eq!(
            splice_testkit::call_i32(
                &k,
                &[17],
                &mut [
                    BufferArg::read::<i32>(&[11]),
                    (&ids[..]).into(),
                    (&mask[..]).into(),
                    (&mut out[..]).into()
                ]
            )
            .unwrap(),
            0
        );
        assert_eq!(out, want);
        for bad in [-1, 1] {
            let mut invalid = ids.clone();
            invalid[1] = bad;
            // The first active invalid lane traps before the subsequent store.
            out.fill(99);
            let result = splice_testkit::call_i32(
                &k,
                &[17],
                &mut [
                    (&[0, 11, 0][1..2]).into(),
                    (&invalid[..]).into(),
                    (&mask[..]).into(),
                    (&mut out[..]).into(),
                ],
            );
            assert!(result.is_err());
            assert!(out.iter().all(|&x| x == 99));
        }
    }
}

#[kernel]
fn scatter_i64_prefix(ids: &[i64], dst: &mut [i64], count: i32) -> i32 {
    let indices = Lane::load(ids, 0);
    let active = Lane::<i64>::first_n(count as usize);
    Lane::<i64>::iota().scatter_select(dst, active, indices);
    0
}

macro_rules! gather_bounds_test {
    ($kernel:ident, $test:ident, $ty:ident, $len:expr $(, $bad:expr)*) => {
        #[kernel]
        fn $kernel(src: &[$ty], indices: &[$ty], out: &mut [$ty], mask_last: i32) -> i32 {
            let ids = Lane::load(indices, 0);
            Lane::<$ty>::iota().store(out, 0);
            let w = Lane::<$ty>::lanes();
            let result = if mask_last != 0 {
                let mask = Lane::<$ty>::first_n((w as i32 - 1) as usize);
                Lane::gather_select(src, mask, ids, Lane::<$ty>::splat(43))
            } else {
                Lane::gather(src, ids)
            };
            result.store(out, w);
            0
        }

        #[test]
        fn $test() {
            let w = Lane::<$ty>::lanes();
            let src: Vec<_> = (0..$len).map(|i| (i as $ty).wrapping_mul(7)).collect();
            let end = ($len - 1) as $ty;
            let kernels = compile_all!($kernel);
            for (len, last) in [($len, end), ($len - 1, end) $(, ($len, $bad))*] {
                let src = &src[..len];
                let mut ids: Vec<_> = (0..w).map(|i| (len - 1 - i % len) as $ty).collect();
                ids[w - 1] = last;
                for masked in [false, true] {
                    let mut want = vec![129 as $ty; 2 * w + 2];
                    for i in 0..w {
                        want[1 + i] = i as $ty;
                    }
                    let ok = masked || src.get(last as usize).is_some();
                    if ok {
                        for (i, &id) in ids.iter().enumerate() {
                            want[1 + w + i] = if masked && i == w - 1 { 43 } else { src[id as usize] };
                        }
                    }
                    let mut out = vec![129 as $ty; 2 * w + 2];
                    let native = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        $kernel(src, &ids, &mut out[1..=2 * w], i32::from(masked))
                    }));
                    assert_eq!(native.is_ok(), ok);
                    assert_eq!(out, want);
                    for kernel in &kernels {
                        out.fill(129);
                        let result = splice_testkit::call_i32(&kernel,
                            &[i32::from(masked)],
                            &mut [src.into(), ids.as_slice().into(), (&mut out[1..=2 * w]).into()],
                        );
                        assert_eq!(result.is_ok(), ok, "index={last}, len={len}, masked={masked}");
                        assert_eq!(out, want);
                    }
                }
            }
        }
    };
}

gather_bounds_test!(
    gather_u16,
    u16_gather_zero_extends_high_indices,
    u16,
    65_536
);
gather_bounds_test!(
    gather_i32,
    i32_gather_checks_indices_as_unsigned,
    i32,
    7,
    -1,
    i32::MIN,
    i32::MAX
);
gather_bounds_test!(
    gather_i64,
    i64_gather_preserves_high_index_bits,
    i64,
    7,
    -1,
    (1_i64 << 32) + 1,
    i64::MIN
);

#[test]
fn signed_i64_scatter_checks_full_indices_and_commits_only_the_valid_prefix() {
    let n = Lane::<i64>::lanes();
    for k in compile_all!(scatter_i64_prefix) {
        for bad in [-1, 2, 1_i64 << 61] {
            let mut ids = vec![0; n];
            ids[0] = 1;
            ids[1] = bad;
            for active in [1, n] {
                let mut out = [91_i64, 42, 42, 92];
                let result = splice_testkit::call_i32(
                    &k,
                    &[active as i32],
                    &mut [(&ids[..]).into(), (&mut out[1..3]).into()],
                );
                assert_eq!(result.is_err(), active == n, "index={bad}, active={active}");
                assert_eq!(out, [91, 42, 0, 92]);
            }
        }
    }
}

#[kernel]
fn ieee_division(a: &[f32], b: &[f32], out: &mut [f32]) -> i32 {
    let x = Lane::load(a, 0) / Lane::load(b, 0);
    x.store(out, 0);
    x.simd_ne(x).any() as i32
}
#[test]
fn floating_division_and_unordered_comparison() {
    let n = Lane::<f32>::lanes();
    let a: Vec<_> = (0..n).map(|i| [0., 1., -1., 6.][i % 4]).collect();
    let b: Vec<_> = (0..n).map(|i| [0., -0., 0., 4.][i % 4]).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x / y).collect();
    let ks = compile_all!(ieee_division);
    for k in &ks {
        let mut out = vec![0f32; n];
        let r = splice_testkit::call_i32(
            &k,
            &[],
            &mut [(&a[..]).into(), (&b[..]).into(), (&mut out[..]).into()],
        )
        .unwrap();
        assert_eq!(r, 1);
        for (x, y) in out.into_iter().zip(&expected) {
            if y.is_nan() {
                assert!(x.is_nan());
            } else {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }
}

#[test]
fn verifier_checks_vector_mask_and_buffer_element_types() {
    use splice::{
        BufferKind,
        frontend::{KernelBuilder, Signature, materialize},
        hir::ScalarType,
        hir::{Buffer, Reg, Simd, SimdOp},
    };
    const KINDS: [BufferKind; 1] = [BufferKind::new(ScalarType::F32, false)];
    /// An operation built from the offset and the loaded vector.
    type Operation = fn(Reg, Reg) -> (ScalarType, SimdOp, Vec<Reg>);
    let stage = |bad: Option<Operation>| {
        materialize(
            Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &KINDS),
            Box::new(move |b: &mut dyn KernelBuilder| {
                let offset = b.constant(Constant::from(0usize))?;
                let f = b
                    .simd(
                        Simd {
                            element: ScalarType::F32,
                            op: SimdOp::Load(Buffer(0)),
                        },
                        &[offset],
                    )?
                    .unwrap();
                if let Some(bad) = bad {
                    let (element, op, args) = bad(offset, f);
                    b.simd(Simd { element, op }, &args)?;
                }
                let zero = b.constant(Constant::from(0))?;
                b.ret(zero)
            }),
            options(),
        )
    };
    assert!(stage(None).is_ok());
    let cases: [Operation; 4] = [
        |offset, _| (ScalarType::F64, SimdOp::Load(Buffer(0)), vec![offset]),
        |_, f| (ScalarType::F32, SimdOp::Any, vec![f]),
        |offset, f| (ScalarType::F32, SimdOp::Store(Buffer(0)), vec![offset, f]),
        |_, f| (ScalarType::F32, SimdOp::Store(Buffer(0)), vec![f, f]),
    ];
    for bad in cases {
        assert!(matches!(
            stage(Some(bad)),
            Err(splice::BuildError::Verify { .. })
        ));
    }
}

#[kernel]
fn blocks(src: &[f64], dst: &mut [f64]) -> i32 {
    let mut i = 0;
    let n = src.len() as i32;
    while i < n {
        let step = (n - i).min(Lane::<f64>::lanes() as i32);
        let tail = Lane::<f64>::first_n(step as usize);
        let a = Lane::load_select(src, i as usize, tail, Lane::<f64>::splat(0.));
        (a + a).store_select(dst, i as usize, tail);
        i += step;
    }
    i
}

#[kernel]
fn prefix(dst: &mut [u16], count: i32) -> i32 {
    let m: splice::simd::Mask<u16> = Lane::first_n(count as usize);
    Lane::<u16>::splat(11).store_select(dst, 0, m);
    m.all() as i32
}

#[test]
fn first_n_caps_before_narrowing_and_drives_partial_blocks() {
    let w = Lane::<u16>::lanes();
    for kernel in compile_all!(prefix) {
        for n in [0, 1, w as i32 - 1, w as i32, 65536, i32::MIN, -1] {
            let mut out = vec![100u16; w];
            let expected = (n as usize).min(w);
            assert_eq!(
                splice_testkit::call_i32(&kernel, &[n], &mut [(&mut out[..]).into()]).unwrap(),
                i32::from(expected == w)
            );
            assert_eq!(out.iter().filter(|&&v| v == 11).count(), expected);
            assert!(out[..expected].iter().all(|&v| v == 11));
            assert!(out[expected..].iter().all(|&v| v == 100));
        }
    }
    let w = Lane::<f64>::lanes();
    let kernels = compile_all!(blocks);
    for n in [0, 1, w - 1, w, w + 1, 2 * w + 3] {
        let src: Vec<_> = (0..n).map(|i| i as f64 - 0.75).collect();
        let mut want: Vec<_> = src.iter().map(|x| x + x).collect();
        want.push(-999.);
        let mut native = vec![-999.; n + 1];
        assert_eq!(blocks(&src, &mut native[..n]), n as i32);
        assert_eq!(native, want);
        for k in &kernels {
            let mut dst = vec![-999f64; n + 1];
            assert_eq!(
                splice_testkit::call_i32(&k, &[], &mut [(&src[..]).into(), (&mut dst[..n]).into()])
                    .unwrap(),
                n as i32
            );
            assert_eq!(dst, want);
        }
    }
}

mod locally_imported {
    #[splice::kernel]
    pub fn copy(a: &[f32], b: &mut [f32]) -> i32 {
        {
            let x: Lane<f32> = Lane::load(a, 0);
            x.store(b, 0);
            return Lane::<f32>::lanes() as i32;
            use splice::simd::Lane;
        }
    }
}

#[kernel]
fn forwarded(a: &[f32], b: &mut [f32]) -> i32 {
    locally_imported::copy(a, b)
}

#[test]
fn typed_slice_forwarding_and_block_local_import() {
    let n = Lane::<f32>::lanes();
    let input: Vec<_> = (0..n).map(|i| i as f32 + 0.5).collect();
    let mut native = vec![0.; n];
    assert_eq!(forwarded(&input, &mut native), n as i32);
    assert_eq!(native, input);
    for k in compile_all!(forwarded) {
        let mut out = vec![0f32; n];
        assert_eq!(
            splice_testkit::call_i32(&k, &[], &mut [(&input[..]).into(), (&mut out[..]).into()])
                .unwrap(),
            n as i32
        );
        assert_eq!(out, input);
    }
}

#[kernel]
fn gather_positive_indices(src: &[f32], indices: &[i32], dst: &mut [f32]) -> i32 {
    let i = Lane::load(indices, 0);
    let keep = i.simd_ge(Lane::splat(0));
    let fallback = Lane::<f32>::splat(-0.);
    let a = Lane::gather_select(src, keep, i, fallback);
    keep.select(a + a, fallback).store(dst, 0);
    keep.any() as i32
}

#[test]
fn integer_mask_selects_float_lanes() {
    let n = Lane::<f32>::lanes();
    let mut ids = vec![-1; n];
    ids[n - 1] = 0;
    let mut expected = vec![(-0f32).to_bits(); n];
    expected[n - 1] = 7f32.to_bits();
    let mut native = vec![0.; n];
    assert_eq!(gather_positive_indices(&[3.5], &ids, &mut native), 1);
    assert_eq!(
        native.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected
    );
    for k in compile_all!(gather_positive_indices) {
        let mut out = vec![0f32; n];
        assert_eq!(
            splice_testkit::call_i32(
                &k,
                &[],
                &mut [
                    (&[3.5f32][..]).into(),
                    (&ids[..]).into(),
                    (&mut out[..]).into()
                ]
            )
            .unwrap(),
            1
        );
        assert_eq!(
            out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected
        );
    }
}

#[kernel]
fn inferred_vectors(src: &[f32], dst: &mut [f32]) -> i32 {
    let mask = Lane::<i32>::first_n(1);
    let selected = mask.select(Lane::splat(1.25), Lane::load(src, 0));
    mask.select(Lane::splat(1.0), Lane::splat(2.0))
        .store(dst, 0);
    let value = Lane::splat(0.5) + selected + Lane::load(dst, 0);
    let second = Lane::<i32>::iota().simd_eq(Lane::splat(1));
    let negative = Lane::splat(-3.5f32);
    second.select(negative, value).store(dst, 0);
    let flag = mask.select::<f32>(Lane::splat(1.25), Lane::splat(2.5));
    flag.simd_eq(Lane::splat(1.25)).any() as i32
}

#[test]
fn data_operands_determine_the_element_type_independently_of_the_mask() {
    let n = Lane::<f32>::lanes();
    let src: Vec<_> = (0..n).map(|i| 0.5 + i as f32).collect();
    let mut expected: Vec<_> = src.iter().map(|x| 2.5 + x).collect();
    expected[0] = 2.75;
    expected[1] = -3.5;
    let mut out = vec![0.; n];
    assert_eq!(inferred_vectors(&src, &mut out), 1);
    assert_eq!(out, expected);
    for kernel in compile_all!(inferred_vectors) {
        out.fill(0.);
        assert_eq!(
            splice_testkit::call_i32(
                &kernel,
                &[],
                &mut [src.as_slice().into(), out.as_mut_slice().into()]
            ),
            Ok(1)
        );
        assert_eq!(out, expected);
    }
}

#[kernel]
fn copy_at(src: &[i32], dst: &mut [i32], offset: i32) -> i32 {
    let width: usize = Lane::<i32>::lanes();
    for i in 0..width {
        dst[0] += i as i32 + 1;
    }
    Lane::load(src, offset as usize).store(dst, 1);
    0
}

#[test]
fn vector_load_checks_the_whole_block_after_prior_writes() {
    let n = Lane::<i32>::lanes();
    let src: Vec<_> = (0..=n as i32).collect();
    let kernels = compile_all!(copy_at);
    for offset in [1, 2, -1, i32::MIN, i32::MAX] {
        let mut expected = vec![-1; n + 1];
        expected[0] += (1..=n as i32).sum::<i32>();
        if offset == 1 {
            expected[1..].copy_from_slice(&src[1..]);
        }
        let mut out = vec![-1; n + 1];
        let native = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            copy_at(&src, &mut out, offset)
        }));
        assert_eq!(native.is_ok(), offset == 1);
        assert_eq!(out, expected);
        for kernel in &kernels {
            out.fill(-1);
            let result = splice_testkit::call_i32(
                &kernel,
                &[offset],
                &mut [src.as_slice().into(), out.as_mut_slice().into()],
            );
            assert_eq!(result.is_ok(), offset == 1);
            assert_eq!(out, expected);
        }
    }
}

#[kernel]
fn refill_lanes(src: &[i32], flags: &[i32], dst: &mut [i32], idx: i32) -> i32 {
    let lanes = Lane::<i32>::load(src, 0);
    let mask = Lane::<i32>::load(flags, 0).simd_ne(Lane::splat(0));
    let expanded = lanes.expand(mask, Lane::<i32>::iota());
    let selected = expanded.extract_wrapping(idx);
    expanded.replace_wrapping(idx, -99).store(dst, 0);
    selected
}

#[test]
fn register_refill_and_wrapping_lane_index_match_across_backends() {
    let width = Lane::<i32>::lanes();
    let src: Vec<_> = (0..width as i32)
        .map(|i| 0x1234_5678 + i * 0x0103_0709)
        .collect();
    let kernels = compile_all!(refill_lanes);
    for bits in [0, 1, u32::MAX, 0x5a5a, 0xf069, 1 << (width - 1)] {
        let flags: Vec<_> = (0..width).map(|i| ((bits >> i) & 1) as i32).collect();
        for idx in [-1, 0, width as i32, i32::MIN, i32::MAX] {
            let mut source = src.iter();
            let mut want: Vec<_> = flags
                .iter()
                .enumerate()
                .map(|(i, &flag)| {
                    if flag == 0 {
                        i as i32
                    } else {
                        *source.next().unwrap()
                    }
                })
                .collect();
            let position = idx as usize % width;
            let selected = std::mem::replace(&mut want[position], -99);
            want.extend([976; 2]);
            let mut dst = vec![976; width + 2];
            assert_eq!(refill_lanes(&src, &flags, &mut dst, idx), selected);
            assert_eq!(dst, want);
            for kernel in &kernels {
                dst.fill(976);
                assert_eq!(
                    splice_testkit::call_i32(
                        &kernel,
                        &[idx],
                        &mut [
                            src.as_slice().into(),
                            flags.as_slice().into(),
                            dst.as_mut_slice().into()
                        ]
                    ),
                    Ok(selected)
                );
                assert_eq!(dst, want);
            }
        }
    }
    let ops = splice_testkit::simd_ops(&refill_lanes::ir().unwrap());
    for op in [
        SimdOp::Expand,
        SimdOp::ExtractWrapping,
        SimdOp::ReplaceWrapping,
    ] {
        assert!(ops.iter().any(|simd| simd.op == op), "{ops:?}");
    }
}

#[kernel]
fn scratch_slots(#[comptime] slot: usize, input: i32, words: &mut [i32]) -> i32 {
    words[0] = input;
    words[1] = input;
    let before = words[0];
    words[slot] += 7;
    before + words[0] + words[slot]
}

#[test]
fn host_scratch_slots_retain_independent_runtime_values() {
    fn check<B: Codegen<Output: Executable>>(slot: usize, want: i32, expected: [i32; 2]) {
        let ir = scratch_slots::ir(slot).unwrap();
        for kernel in [
            B::compile(&ir).unwrap(),
            scratch_slots::build::<B>(scratch_slots::BuildArgs { slot }).unwrap(),
        ] {
            let mut words = [-1; 2];
            assert_eq!(
                kernel.call(scratch_slots::CallArgs {
                    input: 5,
                    words: &mut words
                }),
                Ok(want)
            );
            assert_eq!(words, expected);
        }
    }

    for slot in [0, 1] {
        let want = if slot == 0 { 29 } else { 22 };
        let mut expected = [0; 2];
        assert_eq!(scratch_slots(slot, 5, &mut expected), want);
        splice_testkit::each_backend!(B => check::<B>(slot, want, expected));
    }
}
#[kernel]
fn mul_i64_block(lhs: &[i64], rhs: &[i64], dst: &mut [i64]) -> i32 {
    (Lane::load(lhs, 0) * Lane::load(rhs, 0)).store(dst, 0);
    Lane::<i64>::lanes() as i32
}

#[test]
fn i64_multiply_keeps_the_low_word_of_every_product() {
    let n = Lane::<i64>::lanes();
    let inputs = [
        (-1_i64, 2_i64),
        ((1 << 32) + 1, (1 << 32) + 1),
        (1 << 32, 3),
        (0xffffffff, (1 << 32) + 1),
        (i64::MIN, -3),
    ];
    for k in compile_all!(mul_i64_block) {
        for offset in [0, 2] {
            let (lhs, rhs): (Vec<_>, Vec<_>) =
                inputs.into_iter().cycle().skip(offset).take(n).unzip();
            let mut dst = vec![33; n + 1];
            let result = splice_testkit::call_i32(
                &k,
                &[],
                &mut [(&lhs[..]).into(), (&rhs[..]).into(), (&mut dst[..]).into()],
            );
            assert_eq!(result.unwrap(), n as i32);
            let expected = lhs
                .iter()
                .zip(rhs)
                .map(|(&a, b)| a.wrapping_mul(b))
                .collect::<Vec<_>>();
            assert_eq!(&dst[..n], &expected);
            assert_eq!(dst[n], 33);
        }
    }
}

/// A masked access with a prefix mask, as a vector loop's tail writes it.
#[kernel]
fn prefix_tail(src: &[i32], out: &mut [i32], at: i32, count: i32) -> i32 {
    let active = Lane::<i32>::first_n(count as usize);
    let v = Lane::load_select(src, at as usize, active, Lane::<i32>::splat(-1));
    v.store_select(out, at as usize, active);
    0
}

#[test]
fn prefix_masks_trap_exactly_when_an_active_lane_is_outside() {
    // Lancy checks a prefix mask against the room as lane counts; every
    // backend agrees with these element-by-element rules.
    let w = Lane::<i32>::lanes();
    let kernels = compile_all!(prefix_tail);
    let src: Vec<i32> = (0..2 * w as i32 + 3).map(|x| 10 + x).collect();
    for src_len in [w - 1, w, w + 2] {
        for out_len in [w - 2, w + 1] {
            for at in [0, 1, 2, w - 1, w, w + 1] {
                for count in [-1, 0, 1, 2, w as i32 - 1, w as i32, w as i32 + 1] {
                    let active = if count < 0 {
                        w
                    } else {
                        (count as usize).min(w)
                    };
                    let src_room = src_len.saturating_sub(at);
                    let out_room = out_len.saturating_sub(at);
                    let mut want = vec![7; out_len];
                    let expected = if active > src_room {
                        Err(CallError::Trap(Trap::Bounds))
                    } else {
                        let written = active.min(out_room);
                        if written > 0 {
                            let lanes = at..at + written;
                            want[lanes.clone()].copy_from_slice(&src[lanes]);
                        }
                        if active > out_room {
                            Err(CallError::Trap(Trap::Bounds))
                        } else {
                            Ok(0)
                        }
                    };
                    for kernel in &kernels {
                        let mut out = vec![7; out_len];
                        let actual = splice_testkit::call_i32(
                            kernel,
                            &[at as i32, count],
                            &mut [
                                BufferArg::from(&src[..src_len]),
                                BufferArg::from(out.as_mut_slice()),
                            ],
                        );
                        let case = (src_len, out_len, at, count);
                        assert_eq!(actual, expected, "{case:?}");
                        assert_eq!(out, want, "{case:?}");
                    }
                }
            }
        }
    }
}

#[kernel]
fn saturating_float_casts(
    singles: &[f32],
    doubles: &[f64],
    signed: &mut [i32],
    unsigned: &mut [u32],
    wide: &mut [i64],
) -> i32 {
    let w = Lane::<f32>::lanes();
    let mut i = 0;
    while i + w <= singles.len() {
        let x = Lane::<f32>::load(singles, i);
        x.cast::<i32>().store(signed, i);
        x.cast::<u32>().store(unsigned, i);
        i += w;
    }
    let w = Lane::<f64>::lanes();
    let mut i = 0;
    while i + w <= doubles.len() {
        Lane::<f64>::load(doubles, i).cast::<i64>().store(wide, i);
        i += w;
    }
    0
}

#[test]
fn float_to_integer_lanes_saturate_and_send_nan_to_zero() {
    // Rust's `as` is the reference: it truncates, saturates and maps NaN
    // to zero. The values sit at and next to each bound.
    let edges = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        0.5,
        -0.5,
        -1.0,
        123.9,
        -123.9,
        2f64.powi(31),
        2f64.powi(31) - 128.0,
        -(2f64.powi(31)),
        -(2f64.powi(31)) - 256.0,
        2f64.powi(32),
        2f64.powi(32) - 256.0,
        2f64.powi(63),
        2f64.powi(63) - 1024.0,
        -(2f64.powi(63)),
        -(2f64.powi(63)) - 4096.0,
        1e20,
        -1e20,
    ];
    let lanes = Lane::<f32>::lanes();
    let n = edges.len().div_ceil(lanes) * lanes;
    let singles: Vec<f32> = (0..n).map(|i| edges[i % edges.len()] as f32).collect();
    let doubles: Vec<f64> = (0..n).map(|i| edges[i % edges.len()]).collect();
    for kernel in compile_all!(saturating_float_casts) {
        let (mut signed, mut unsigned, mut wide) = (vec![7; n], vec![7; n], vec![7; n]);
        let result = splice_testkit::call_i32(
            &kernel,
            &[],
            &mut [
                BufferArg::from(singles.as_slice()),
                BufferArg::from(doubles.as_slice()),
                BufferArg::from(signed.as_mut_slice()),
                BufferArg::from(unsigned.as_mut_slice()),
                BufferArg::from(wide.as_mut_slice()),
            ],
        );
        assert_eq!(result, Ok(0));
        for i in 0..n {
            assert_eq!(signed[i], singles[i] as i32, "{}", singles[i]);
            assert_eq!(unsigned[i], singles[i] as u32, "{}", singles[i]);
            assert_eq!(wide[i], doubles[i] as i64, "{}", doubles[i]);
        }
    }
}
