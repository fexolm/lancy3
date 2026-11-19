use splice::{BufferArg, Codegen, Executable, Function, kernel, simd::Lane};
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
    ids[..4].copy_from_slice(&[1, 1, 3, 0]);
    let mut flags = vec![0u16; w];
    flags[..4].fill(1);
    let mut vals = vec![0u16; w];
    vals[..4].copy_from_slice(&[11, 22, 33, 44]);
    let ks = compile_all!(scatter);
    for active in [true, false] {
        flags[2] = u16::from(active);
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
