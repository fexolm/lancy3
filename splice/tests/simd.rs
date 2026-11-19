use splice::{
    BufferArg, Codegen, Executable, Function, kernel,
    simd::{Lane, Mask},
};
use splice_cranelift::Cranelift;
use splice_lancy::Lancy;
use splice_llvm::Llvm;
use splice_text::Text;

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

fn kernels<E: Executable + 'static>(
    out: &mut Vec<Box<dyn Executable>>,
    direct: splice::Result<E>,
    portable: splice::Result<E>,
) {
    out.push(Box::new(direct.unwrap()));
    out.push(Box::new(portable.unwrap()));
}
macro_rules! compile_all {
    ($name:ident) => {{
        let ir: Function = $name::ir().unwrap();
        let mut v = Vec::new();
        kernels(&mut v, $name::build::<Lancy>(), Lancy::compile(&ir));
        kernels(&mut v, $name::build::<Llvm>(), Llvm::compile(&ir));
        kernels(&mut v, $name::build::<Cranelift>(), Cranelift::compile(&ir));
        v
    }};
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
            code.call(
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
                    let actual = kernel.call(&[offset as i32], &mut [slice.into()]).unwrap();
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
            let result = kernel
                .call(
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
            let result = k.call(
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
    let text = Text::compile(&ir).unwrap();
    assert_eq!(text, add_block::build::<Text>().unwrap());
    for part in [
        "lhs: &[u16]",
        "dst: &mut [u16]",
        "simd.load.u16",
        "simd.add.u16",
        "simd.store.u16",
        "simd.lanes.u16",
    ] {
        assert!(text.contains(part), "{text}");
    }
    for k in compile_all!(add_block) {
        for len in [w + 2, w + 1, w, 1, 0] {
            let mut got = vec![901u16; w + 2];
            let mut bufs = [
                BufferArg::from(&a[..]),
                BufferArg::from(&b[..]),
                BufferArg::from(&mut got[..len]),
            ];
            let r = k.call(&[], &mut bufs);
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
            k.call(
                &[],
                &mut [
                    (&a[..]).into(),
                    (&b[..]).into(),
                    BufferArg::Write(&mut wrong)
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
            let r = k.call(
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
                k.call(
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
            let result = kernel.call(&[], &mut [ids.as_slice().into(), (&mut out[1..4]).into()]);
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
            let result = kernel.call(
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
            k.call(
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
                k.call(
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
                kernel.call(
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
            k.call(
                &[17],
                &mut [
                    BufferArg::Read(&[11]),
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
            let result = k.call(
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
                        let result = kernel.call(
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
                let result = k.call(
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
        let r = k
            .call(
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
        hir::{Buffer, FunctionBuilder, Simd, SimdOp},
        simd::Element,
    };
    let mut b = FunctionBuilder::with_buffers(0, vec![BufferKind::new(Element::F32, false)]);
    let offset = b.constant(0);
    let f = b
        .simd(
            Simd {
                element: Element::F32,
                op: SimdOp::Load(Buffer(0)),
            },
            &[offset],
        )
        .unwrap()
        .unwrap();
    for (element, op, args) in [
        (Element::F64, SimdOp::Load(Buffer(0)), vec![offset]),
        (Element::F32, SimdOp::Any, vec![f]),
        (Element::F32, SimdOp::Store(Buffer(0)), vec![offset, f]),
    ] {
        assert!(b.simd(Simd { element, op }, &args).is_err());
    }
    b.ret(f);
    assert!(b.finish().is_err());
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
                kernel.call(&[n], &mut [(&mut out[..]).into()]).unwrap(),
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
                k.call(&[], &mut [(&src[..]).into(), (&mut dst[..n]).into()])
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
            k.call(&[], &mut [(&input[..]).into(), (&mut out[..]).into()])
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
            k.call(
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
            kernel.call(&[], &mut [src.as_slice().into(), out.as_mut_slice().into()]),
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
            let result = kernel.call(
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
    for bits in [0, u32::MAX, 0x5a5a, 0xf069, 1 << (width - 1)] {
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
            let mut dst = vec![0; width];
            assert_eq!(refill_lanes(&src, &flags, &mut dst, idx), selected);
            assert_eq!(dst, want);
            for kernel in &kernels {
                dst.fill(-1);
                assert_eq!(
                    kernel.call(
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
    let listing = refill_lanes::build::<Text>().unwrap();
    for op in ["expand", "extract_wrapping", "replace_wrapping"] {
        assert!(listing.contains(op));
    }
}

#[kernel]
fn staged_locals(#[comptime] size: usize, #[comptime] slot: usize, input: i32) -> i32 {
    let mut words = vec![input; size];
    let before = words[0];
    words[slot] += 7;
    before + words[0] + words[slot]
}

#[test]
fn specialization_sized_locals_have_independent_registers() {
    for slot in [0, 1] {
        let ir = staged_locals::ir(2, slot).unwrap();
        let want = if slot == 0 { 29 } else { 22 };
        assert_eq!(staged_locals(2, slot, 5), want);
        for result in [
            Lancy::compile(&ir).unwrap().call(&[5], &mut []),
            staged_locals::build::<Lancy>(2, slot)
                .unwrap()
                .call(&[5], &mut []),
            Llvm::compile(&ir).unwrap().call(&[5], &mut []),
            staged_locals::build::<Llvm>(2, slot)
                .unwrap()
                .call(&[5], &mut []),
        ] {
            assert_eq!(result, Ok(want));
        }
    }
    assert!(
        staged_locals::ir(1, 1)
            .unwrap_err()
            .contains("local index out of bounds")
    );
}
