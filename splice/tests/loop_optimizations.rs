use splice::simd::Lane;
use splice::{Codegen, Executable, kernel};

#[kernel]
fn apply_coefficients(src: &[i32], coefficients: &[i32], dst: &mut [i32]) -> i32 {
    if coefficients.len() < 2 {
        return -1;
    }
    for i in 0..src.len() as i32 {
        dst[i as usize] = src[i as usize]
            .wrapping_mul(coefficients[0])
            .wrapping_add(coefficients[1]);
    }
    src.len() as i32
}

#[kernel]
fn checked_passes(src: &[i32], first: &mut [i32], second: &mut [i32]) -> i32 {
    for i in 0..src.len() as i32 {
        first[i as usize] = src[i as usize].wrapping_add(1);
    }
    for i in 0..src.len() as i32 {
        second[i as usize] = first[i as usize].wrapping_mul(2);
    }
    src.len() as i32
}

fn check<B: Codegen<Output: Executable>>() {
    let src = [i32::MAX, 3, -2];
    for kernel in [
        apply_coefficients::build::<B>(apply_coefficients::BuildArgs {}).unwrap(),
        B::compile(&apply_coefficients::ir().unwrap()).unwrap(),
    ] {
        let mut dst = [0; 3];
        assert_eq!(
            kernel.call(apply_coefficients::CallArgs {
                src: &src,
                coefficients: &[2, 7],
                dst: &mut dst,
            }),
            Ok(3)
        );
        assert_eq!(dst, [5, 13, 3]);

        let mut short = [99];
        assert!(
            kernel
                .call(apply_coefficients::CallArgs {
                    src: &src,
                    coefficients: &[2, 7],
                    dst: &mut short,
                })
                .is_err()
        );
        assert_eq!(short, [5]);

        assert_eq!(
            kernel.call(apply_coefficients::CallArgs {
                src: &[],
                coefficients: &[],
                dst: &mut [],
            }),
            Ok(-1)
        );
    }

    for kernel in [
        checked_passes::build::<B>(checked_passes::BuildArgs {}).unwrap(),
        B::compile(&checked_passes::ir().unwrap()).unwrap(),
    ] {
        let mut first = [99; 3];
        let mut second = [99];
        assert!(
            kernel
                .call(checked_passes::CallArgs {
                    src: &src,
                    first: &mut first,
                    second: &mut second
                })
                .is_err()
        );
        assert_eq!(first, [i32::MIN, 4, -1]);
        assert_eq!(second, [0]);
    }
}

#[test]
fn loop_memory_optimizations_preserve_partial_writes() {
    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn independent_stores(i: i32, j: i32, x: i32, y: i32, a: &mut [i32], b: &mut [i32]) -> i32 {
    a[i as usize] = x;
    b[j as usize] = y;
    a[i as usize]
}

#[test]
fn forwarding_between_independent_slices_preserves_each_write() {
    fn check<B: Codegen<Output: Executable>>() {
        let ir = independent_stores::ir().unwrap();
        for code in [
            independent_stores::build::<B>(independent_stores::BuildArgs {}).unwrap(),
            B::compile(&ir).unwrap(),
        ] {
            for (i, j, result, expected_a, expected_b) in [
                (1, 0, Some(41), [7, 41], [53, 10]),
                (2, 0, None, [7, 8], [9, 10]),
                (1, 2, None, [7, 41], [9, 10]),
            ] {
                let mut a = [7, 8];
                let mut b = [9, 10];
                assert_eq!(
                    code.call(independent_stores::CallArgs {
                        i,
                        j,
                        x: 41,
                        y: 53,
                        a: &mut a,
                        b: &mut b
                    })
                    .ok(),
                    result
                );
                assert_eq!((a, b), (expected_a, expected_b));
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn vector_store_reload(src: &[u16], dst: &mut [u16]) -> i32 {
    let stored = Lane::<u16>::load(src, 0);
    stored.store(dst, 0);
    let loaded = Lane::<u16>::load(dst, 0);
    if loaded.simd_eq(stored).all() { 1 } else { 0 }
}

#[test]
fn vector_store_reload_preserves_the_short_store_prefix() {
    fn check<B: Codegen<Output: Executable>>() {
        let lanes = Lane::<u16>::lanes();
        let src: Vec<_> = (0..lanes).map(|i| 65534 - i as u16).collect();
        for code in [
            vector_store_reload::build::<B>(vector_store_reload::BuildArgs {}).unwrap(),
            B::compile(&vector_store_reload::ir().unwrap()).unwrap(),
        ] {
            for len in [0, lanes - 1, lanes, lanes + 1] {
                let mut out = vec![17; lanes + 1];
                let result = code.call(vector_store_reload::CallArgs {
                    src: &src,
                    dst: &mut out[..len],
                });
                if len >= lanes {
                    assert_eq!(result, Ok(1));
                } else {
                    assert!(result.is_err());
                }
                let mut expected = vec![17; lanes + 1];
                let written = len.min(lanes);
                expected[..written].copy_from_slice(&src[..written]);
                assert_eq!(out, expected, "len={len}");
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn packet_passes(
    #[comptime] validate_lengths: bool,
    n: i32,
    adjust: i32,
    src: &[u16],
    dst: &mut [u16],
) -> i32 {
    if validate_lengths {
        if n > src.len() as i32 {
            return -1;
        }
        if n > dst.len() as i32 {
            return -1;
        }
    }
    let lanes = Lane::<u16>::lanes() as i32;
    let mut i = 0;
    while n - i >= lanes {
        let x = Lane::<u16>::load(src, i as usize);
        let y = x + Lane::<u16>::splat(1);
        let selected = if adjust != 0 { y } else { x };
        selected.store(dst, i as usize);
        i += lanes;
    }
    let mut j = 0;
    while n - j >= lanes {
        let x = Lane::<u16>::load(dst, j as usize);
        (x + Lane::<u16>::splat(2)).store(dst, j as usize);
        j += lanes;
    }
    n
}

#[test]
fn packet_loops_preserve_tail_and_partial_store_semantics() {
    fn check<B: Codegen<Output: Executable>>() {
        let lanes = Lane::<u16>::lanes();
        let n = 2 * lanes + 3;
        let src: Vec<_> = (0..n).map(|i| u16::MAX - i as u16).collect();
        for guarded in [true, false] {
            let ir = packet_passes::ir(guarded).unwrap();
            for code in [
                packet_passes::build::<B>(packet_passes::BuildArgs {
                    validate_lengths: guarded,
                })
                .unwrap(),
                B::compile(&ir).unwrap(),
            ] {
                for (len, adjust) in [(n, 0), (n, 2), (n, i32::MIN), (lanes + 2, 1)] {
                    let mut out = vec![257; len];
                    let result = code.call(packet_passes::CallArgs {
                        n: n as i32,
                        adjust,
                        src: &src,
                        dst: &mut out,
                    });
                    let mut expected = vec![257; len];
                    let first = u16::from(adjust != 0);
                    if len == n {
                        assert_eq!(result, Ok(n as i32));
                        for (dst, &x) in expected[..2 * lanes].iter_mut().zip(&src) {
                            *dst = x.wrapping_add(first + 2);
                        }
                    } else if guarded {
                        assert_eq!(result, Ok(-1));
                    } else {
                        assert!(result.is_err());
                        for (dst, &x) in expected.iter_mut().zip(&src) {
                            *dst = x.wrapping_add(first);
                        }
                    }
                    assert_eq!(
                        out, expected,
                        "guarded={guarded}, len={len}, adjust={adjust}"
                    );
                }
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn invariant_loop_branch(n: i32, mode: i32, out: &mut [i32]) -> i32 {
    let mut i = 0;
    let mut sum = 0;
    while i < n {
        if mode != 0 {
            sum += i + 2;
        } else {
            sum -= i + 1;
        }
        out[i as usize] = sum;
        i += 1;
    }
    sum
}

#[test]
fn invariant_loop_branch_matches_rust_on_both_build_paths() {
    const SENTINEL: i32 = -123_456_789;
    fn check<E: Executable>(
        name: &str,
        kernel: splice::TypedKernel<invariant_loop_branch::Signature, E>,
    ) {
        for (n, mode) in [(0, 0), (0, 1), (4, 0), (4, -1)] {
            let mut expected = [SENTINEL; 5];
            let sum = invariant_loop_branch(n, mode, &mut expected);
            let mut out = [SENTINEL; 5];
            let actual = kernel
                .call(invariant_loop_branch::CallArgs {
                    n,
                    mode,
                    out: &mut out,
                })
                .unwrap();
            assert_eq!((actual, out), (sum, expected), "{name}, n={n}, mode={mode}");
        }
        for mode in [0, -1] {
            let mut expected = [SENTINEL; 5];
            invariant_loop_branch(5, mode, &mut expected);
            let mut out = [SENTINEL; 5];
            assert!(
                kernel
                    .call(invariant_loop_branch::CallArgs {
                        n: 6,
                        mode,
                        out: &mut out
                    })
                    .is_err(),
                "{name}, mode={mode}"
            );
            assert_eq!(out, expected, "{name}, mode={mode}");
        }
    }
    let ir = invariant_loop_branch::ir().unwrap();
    splice_testkit::each_backend!(B => {
        let name = std::any::type_name::<B>();
        check(name, B::compile(&ir).unwrap());
        check(
            name,
            invariant_loop_branch::build::<B>(invariant_loop_branch::BuildArgs {}).unwrap(),
        );
    });
}
