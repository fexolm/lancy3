//! Operation × element × boundary data on every backend, build route and
//! build mode, checked against each other and against Rust.
use splice::frontend::KernelBuilder;
use splice::hir::{Binary, Buffer, Reg, Simd, SimdOp};
use splice::simd::Element;
use splice::{BufferArg, BufferKind};
use splice_testkit::{Stage, agree};
use std::rc::Rc;

const EDGE: [i32; 12] = [0, 1, -1, 2, 7, -7, 31, 32, 33, -32, i32::MIN, i32::MAX];

fn simd(b: &mut dyn KernelBuilder, element: Element, op: SimdOp, args: &[Reg]) -> Reg {
    b.simd(Simd { element, op }, args).unwrap().unwrap()
}

fn store(b: &mut dyn KernelBuilder, element: Element, buffer: u32, at: Reg, value: Reg) {
    b.simd(
        Simd {
            element,
            op: SimdOp::Store(Buffer(buffer)),
        },
        &[at, value],
    )
    .unwrap();
}

fn load(b: &mut dyn KernelBuilder, element: Element, buffer: u32, at: Reg) -> Reg {
    simd(b, element, SimdOp::Load(Buffer(buffer)), &[at])
}

#[test]
fn every_scalar_binary_matches_checked_eval() {
    use Binary::*;
    for op in [
        Add, Sub, Mul, Div, Rem, And, Or, Xor, Eq, Ne, Lt, Le, Gt, Ge, Shl, Shr, CheckedAdd,
        CheckedSub, CheckedMul, CheckedShl, CheckedShr, Min, Max,
    ] {
        let stage: Stage = Rc::new(move |b| {
            let r = b.binary(op, b.arg(0), b.arg(1));
            b.ret(r);
            Ok(())
        });
        let seen = agree(&format!("{op:?}"), 2, &[], &stage, |call| {
            let mut seen = Vec::new();
            for a in EDGE {
                for c in EDGE {
                    seen.push(format!("{:?}", call(&[a, c], &mut []).ok()));
                }
            }
            seen
        });
        let mut expected = Vec::new();
        for a in EDGE {
            for c in EDGE {
                expected.push(format!("{:?}", op.checked_eval(a, c)));
            }
        }
        assert_eq!(seen, expected, "{op:?}");
    }
}

/// Load `extra + 1` vectors of `from` (at 0 and at each argument), apply
/// `op`, store the result as `to`.
fn unary(from: Element, to: Element, op: SimdOp, extra: usize) -> Stage {
    Rc::new(move |b| {
        let zero = b.constant(0);
        let mut args = vec![load(b, from, 0, zero)];
        for i in 0..extra {
            let at = b.arg(i);
            args.push(load(b, from, 0, at));
        }
        let r = simd(b, from, op, &args);
        store(b, to, 1, zero, r);
        b.ret(zero);
        Ok(())
    })
}

fn kinds(from: Element, to: Element) -> [BufferKind; 2] {
    [BufferKind::new(from, false), BufferKind::new(to, true)]
}

#[test]
fn float_to_integer_casts_saturate_like_rust() {
    let n = Element::F32.lanes();
    let src: Vec<f32> = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        3e9,
        -3e9,
        -1.5,
        2.5,
        -0.0,
        4294967040.0,
        1e-40,
        2147483520.0,
        2147483648.0,
        -2147483904.0,
        0.5,
        65536.0,
        -1.0,
    ]
    .into_iter()
    .cycle()
    .take(n)
    .collect();
    let stage = unary(Element::F32, Element::I32, SimdOp::Cast(Element::I32), 0);
    let seen = agree(
        "f32 -> i32",
        0,
        &kinds(Element::F32, Element::I32),
        &stage,
        |call| {
            let mut out = vec![0i32; n];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {out:?}")]
        },
    );
    let expected: Vec<i32> = src.iter().map(|&x| x as i32).collect();
    assert_eq!(seen, [format!("Ok(0) {expected:?}")]);

    let stage = unary(Element::F32, Element::U32, SimdOp::Cast(Element::U32), 0);
    let seen = agree(
        "f32 -> u32",
        0,
        &kinds(Element::F32, Element::U32),
        &stage,
        |call| {
            let mut out = vec![0u32; n];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {out:?}")]
        },
    );
    let expected: Vec<u32> = src.iter().map(|&x| x as u32).collect();
    assert_eq!(seen, [format!("Ok(0) {expected:?}")]);

    let m = Element::F64.lanes();
    let src: Vec<f64> = [
        f64::NAN,
        1e19,
        -1e19,
        -1.5,
        9.3e18,
        2.5,
        f64::INFINITY,
        -0.0,
    ]
    .into_iter()
    .cycle()
    .take(m)
    .collect();
    let stage = unary(Element::F64, Element::I64, SimdOp::Cast(Element::I64), 0);
    let seen = agree(
        "f64 -> i64",
        0,
        &kinds(Element::F64, Element::I64),
        &stage,
        |call| {
            let mut out = vec![0i64; m];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {out:?}")]
        },
    );
    let expected: Vec<i64> = src.iter().map(|&x| x as i64).collect();
    assert_eq!(seen, [format!("Ok(0) {expected:?}")]);
}

#[test]
fn integer_to_float_casts_round_like_rust() {
    let m = Element::I64.lanes();
    let src: Vec<i64> = [
        i64::MAX,
        i64::MIN,
        -1,
        1 << 53 | 1,
        3,
        -(1 << 60) - 1,
        0,
        12345,
    ]
    .into_iter()
    .cycle()
    .take(m)
    .collect();
    let stage = unary(Element::I64, Element::F64, SimdOp::Cast(Element::F64), 0);
    let seen = agree(
        "i64 -> f64",
        0,
        &kinds(Element::I64, Element::F64),
        &stage,
        |call| {
            let mut out = vec![0f64; m];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {:?}", out.map_bits())]
        },
    );
    let expected: Vec<f64> = src.iter().map(|&x| x as f64).collect();
    assert_eq!(seen, [format!("Ok(0) {:?}", expected.map_bits())]);

    let n = Element::U32.lanes();
    let src: Vec<u32> = [
        u32::MAX,
        0x8000_0001,
        16777217,
        0,
        1,
        0xFFFF_FF7F,
        3_000_000_000,
        7,
    ]
    .into_iter()
    .cycle()
    .take(n)
    .collect();
    let stage = unary(Element::U32, Element::F32, SimdOp::Cast(Element::F32), 0);
    let seen = agree(
        "u32 -> f32",
        0,
        &kinds(Element::U32, Element::F32),
        &stage,
        |call| {
            let mut out = vec![0f32; n];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {:?}", out.map_bits())]
        },
    );
    let expected: Vec<f32> = src.iter().map(|&x| x as f32).collect();
    assert_eq!(seen, [format!("Ok(0) {:?}", expected.map_bits())]);
}

trait Bits {
    fn map_bits(&self) -> Vec<u64>;
}
impl Bits for Vec<f32> {
    fn map_bits(&self) -> Vec<u64> {
        self.iter().map(|x| x.to_bits().into()).collect()
    }
}
impl Bits for Vec<f64> {
    fn map_bits(&self) -> Vec<u64> {
        self.iter().map(|x| x.to_bits()).collect()
    }
}

#[test]
fn float_arithmetic_is_correctly_rounded_on_special_values() {
    let n = Element::F32.lanes();
    let src: Vec<f32> = [0.1f32, -4.0, 1e30, -0.0, 3.0, 16777217.0, 1.0000001, 7.0]
        .into_iter()
        .cycle()
        .take(n + 2)
        .collect();
    let reference = |op, i: usize| -> f32 {
        let (x, y, z) = (src[i], src[i + 1], src[i + 2]);
        match op {
            SimdOp::Sqrt => x.sqrt(),
            SimdOp::Div => x / y,
            _ => x.mul_add(y, z),
        }
    };
    for (op, extra) in [(SimdOp::Sqrt, 0), (SimdOp::Div, 1), (SimdOp::MulAdd, 2)] {
        let stage = unary(Element::F32, Element::F32, op, extra);
        let seen = agree(
            &format!("{op:?}"),
            extra,
            &kinds(Element::F32, Element::F32),
            &stage,
            |call| {
                let mut out = vec![0f32; n];
                let args: Vec<i32> = (1..=extra as i32).collect();
                let r = call(
                    &args,
                    &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
                );
                vec![format!("{r:?} {:?}", out.map_bits())]
            },
        );
        let expected: Vec<f32> = (0..n).map(|i| reference(op, i)).collect();
        assert_eq!(seen, [format!("Ok(0) {:?}", expected.map_bits())], "{op:?}");
    }
}

#[test]
fn float_comparisons_follow_ieee_unordered_rules() {
    let n = Element::F32.lanes();
    let src: Vec<f32> = [f32::NAN, 1.0, f32::NAN, 2.0, 1.0, -0.0, 0.0, f32::INFINITY]
        .into_iter()
        .cycle()
        .take(n + 1)
        .collect();
    use SimdOp::*;
    for (op, rust) in [
        (Eq, f32::eq as fn(&f32, &f32) -> bool),
        (Ne, f32::ne),
        (Lt, f32::lt),
        (Le, f32::le),
        (Gt, f32::gt),
        (Ge, f32::ge),
    ] {
        let stage: Stage = Rc::new(move |b| {
            let (zero, one) = (b.constant(0), b.constant(1));
            let x = load(b, Element::F32, 0, zero);
            let y = load(b, Element::F32, 0, one);
            let m = simd(b, Element::F32, op, &[x, y]);
            let yes = simd(b, Element::F32, Splat(1f32.to_bits().into()), &[]);
            let no = simd(b, Element::F32, Splat(0), &[]);
            let s = simd(b, Element::F32, Select, &[m, yes, no]);
            store(b, Element::F32, 1, zero, s);
            b.ret(zero);
            Ok(())
        });
        let seen = agree(
            &format!("{op:?}"),
            0,
            &kinds(Element::F32, Element::F32),
            &stage,
            |call| {
                let mut out = vec![9f32; n];
                let r = call(
                    &[],
                    &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
                );
                vec![format!("{r:?} {out:?}")]
            },
        );
        let expected: Vec<f32> = (0..n)
            .map(|i| f32::from(u8::from(rust(&src[i], &src[i + 1]))))
            .collect();
        assert_eq!(seen, [format!("Ok(0) {expected:?}")], "{op:?}");
    }
}

#[test]
fn integer_lane_operations_agree_on_boundary_values() {
    let n = Element::I32.lanes();
    let src: Vec<i32> = (0..2 * n as i32)
        .map(|i| {
            [
                i32::MIN,
                -1,
                0,
                1,
                31,
                32,
                -33,
                i32::MAX,
                100,
                -100,
                7,
                15,
                16,
                17,
                64,
                -64,
            ][i as usize % 16]
                .wrapping_add(i / 16)
        })
        .collect();
    use SimdOp::*;
    for element in [Element::I32, Element::U32] {
        for (op, extra) in [
            (Shr(5), 0),
            (Shr(37), 0),
            (RotateLeft(7), 0),
            (RotateLeft(33), 0),
            (Min, 1),
            (Max, 1),
            (Swizzle, 1),
            (PrefixSum, 0),
            (Mul, 1),
            (Not, 0),
        ] {
            if matches!(op, Swizzle | PrefixSum) && element != Element::I32 {
                continue;
            }
            let stage = unary(element, element, op, extra);
            agree(
                &format!("{element:?} {op:?}"),
                extra,
                &kinds(element, element),
                &stage,
                |call| {
                    [1, 3, n as i32 - 1]
                        .into_iter()
                        .map(|offset| {
                            let args = if extra == 1 { vec![offset] } else { vec![] };
                            let mut out = vec![0i32; n];
                            let r = if element == Element::I32 {
                                call(
                                    &args,
                                    &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
                                )
                            } else {
                                let src: Vec<u32> = src.iter().map(|&x| x as u32).collect();
                                let mut wide = vec![0u32; n];
                                let r = call(
                                    &args,
                                    &mut [
                                        BufferArg::from(&src[..]),
                                        BufferArg::from(&mut wide[..]),
                                    ],
                                );
                                out = wide.iter().map(|&x| x as i32).collect();
                                r
                            };
                            format!("{r:?} {out:?}")
                        })
                        .collect()
                },
            );
        }
    }
}

#[test]
fn first_n_masks_saturate_at_both_ends() {
    use SimdOp::*;
    let stage: Stage = Rc::new(|b| {
        let k = b.arg(0);
        let m = simd(b, Element::U16, FirstN, &[k]);
        let ones = simd(b, Element::U16, Splat(1), &[]);
        let zeros = simd(b, Element::U16, Splat(0), &[]);
        let s = simd(b, Element::U16, Select, &[m, ones, zeros]);
        let zero = b.constant(0);
        store(b, Element::U16, 0, zero, s);
        let m = simd(b, Element::I64, FirstN, &[k]);
        let any = simd(b, Element::I64, Any, &[m]);
        let all = simd(b, Element::I64, All, &[m]);
        let r = b.binary(Binary::Add, any, all);
        let m = simd(b, Element::I32, FirstN, &[k]);
        let count = simd(b, Element::I32, Count, &[m]);
        let ten = b.constant(10);
        let count = b.binary(Binary::Mul, count, ten);
        let r = b.binary(Binary::Add, r, count);
        b.ret(r);
        Ok(())
    });
    let n = Element::U16.lanes();
    let seen = agree(
        "first_n",
        1,
        &[BufferKind::new(Element::U16, true)],
        &stage,
        |call| {
            [
                0,
                1,
                3,
                8,
                31,
                32,
                33,
                65535,
                65536,
                65537,
                -1,
                i32::MIN,
                i32::MAX,
            ]
            .into_iter()
            .map(|k| {
                let mut out = vec![7u16; n];
                let r = call(&[k], &mut [BufferArg::from(&mut out[..])]);
                format!("{r:?} {out:?}")
            })
            .collect()
        },
    );
    let expected: Vec<_> = [
        0,
        1,
        3,
        8,
        31,
        32,
        33,
        65535,
        65536,
        65537,
        -1,
        i32::MIN,
        i32::MAX,
    ]
    .into_iter()
    .map(|k: i32| {
        // The count is unsigned; IR booleans are 0 or -1.
        let active = |lanes: usize| (k as u32 as usize).min(lanes);
        let i64s = Element::I64.lanes();
        let any = -i32::from(active(i64s) > 0);
        let all = -i32::from(active(i64s) == i64s);
        let count = active(Element::I32.lanes()) as i32;
        let out: Vec<u16> = (0..n).map(|i| u16::from(i < active(n))).collect();
        format!("Ok({}) {out:?}", any + all + 10 * count)
    })
    .collect();
    assert_eq!(seen, expected);
}

#[test]
fn compress_expand_and_compress_store_agree_at_buffer_edges() {
    let n = Element::I32.lanes();
    for op in [SimdOp::Compress, SimdOp::Expand] {
        let stage: Stage = Rc::new(move |b| {
            let (zero, one) = (b.constant(0), b.constant(1));
            let e = Element::I32;
            let x = load(b, e, 0, zero);
            let flags = load(b, e, 1, zero);
            let z = simd(b, e, SimdOp::Splat(0), &[]);
            let m = simd(b, e, SimdOp::Ne, &[flags, z]);
            let fallback = load(b, e, 0, one);
            let r = if op == SimdOp::Compress {
                simd(b, e, op, &[x, m])
            } else {
                simd(b, e, op, &[x, m, fallback])
            };
            store(b, e, 2, zero, r);
            let at = b.arg(0);
            b.simd(
                Simd {
                    element: e,
                    op: SimdOp::CompressStore(Buffer(3)),
                },
                &[at, m, x],
            )?;
            b.ret(zero);
            Ok(())
        });
        let x: Vec<i32> = (100..100 + n as i32 + 1).collect();
        let kinds = [
            BufferKind::Read,
            BufferKind::Read,
            BufferKind::Write,
            BufferKind::Write,
        ];
        agree(&format!("{op:?}"), 1, &kinds, &stage, |call| {
            [
                (0b1011u32, 0, 8),
                (0xffff, 2, 5),
                (0, 100, 0),
                (0x8001, 6, 7),
                (0b111, -1, 4),
            ]
            .into_iter()
            .map(|(bits, at, len)| {
                let flags: Vec<i32> = (0..n).map(|i| ((bits >> (i % 32)) & 1) as i32).collect();
                let mut out = vec![-9; n];
                let mut dst = vec![-5; len];
                let r = call(
                    &[at],
                    &mut [
                        BufferArg::from(&x[..]),
                        BufferArg::from(&flags[..]),
                        BufferArg::from(&mut out[..]),
                        BufferArg::from(&mut dst[..]),
                    ],
                );
                format!("{r:?} {out:?} {dst:?}")
            })
            .collect()
        });
    }
}

#[test]
fn u16_gather_and_scatter_check_indices_at_16_bit_boundaries() {
    let stage: Stage = Rc::new(|b| {
        let zero = b.constant(0);
        let e = Element::U16;
        let index = load(b, e, 0, zero);
        let g = simd(b, e, SimdOp::Gather(Buffer(1)), &[index]);
        store(b, e, 2, zero, g);
        b.simd(
            Simd {
                element: e,
                op: SimdOp::Scatter(Buffer(2)),
            },
            &[index, g],
        )?;
        b.ret(zero);
        Ok(())
    });
    let n = Element::U16.lanes();
    let kinds = [
        BufferKind::new(Element::U16, false),
        BufferKind::new(Element::U16, false),
        BufferKind::new(Element::U16, true),
    ];
    let cases = [
        (70000usize, 65535u16),
        (65535, 65534),
        (65535, 65535),
        (40, 39),
        (40, 40),
        (32768, 32767),
        (32769, 32768),
    ];
    let seen = agree("u16 gather", 0, &kinds, &stage, |call| {
        cases
            .into_iter()
            .map(|(len, special)| {
                let src: Vec<u16> = (0..len).map(|i| (i * 7) as u16).collect();
                let mut index: Vec<u16> = (0..n).map(|i| i as u16 % 30).collect();
                index[3] = special;
                let mut out = vec![1u16; n + 40];
                let r = call(
                    &[],
                    &mut [
                        BufferArg::from(&index[..]),
                        BufferArg::from(&src[..]),
                        BufferArg::from(&mut out[..]),
                    ],
                );
                format!("{r:?} {:?}", &out[..n])
            })
            .collect()
    });
    for ((len, special), seen) in cases.into_iter().zip(&seen) {
        // The gather reads `src[special]`; the scatter then writes `out[special]`.
        let gather_ok = (special as usize) < len;
        let scatter_ok = (special as usize) < n + 40;
        assert_eq!(
            seen.starts_with("Ok"),
            gather_ok && scatter_ok,
            "{len} {special}: {seen}"
        );
    }
}

#[test]
fn partial_vector_accesses_agree_at_negative_and_far_offsets() {
    let stage: Stage = Rc::new(|b| {
        let e = Element::I64;
        let zero = b.constant(0);
        let x = simd(b, e, SimdOp::Iota, &[]);
        let k = b.arg(1);
        let m = simd(b, e, SimdOp::FirstN, &[k]);
        let at = b.arg(0);
        let l = simd(b, e, SimdOp::LoadSelect(Buffer(1)), &[at, m, x]);
        store(b, e, 0, zero, l);
        store(b, e, 0, at, x);
        b.ret(zero);
        Ok(())
    });
    let n = Element::I64.lanes();
    let kinds = [
        BufferKind::new(Element::I64, true),
        BufferKind::new(Element::I64, false),
    ];
    agree("i64 partial accesses", 2, &kinds, &stage, |call| {
        let src: Vec<i64> = (0..n as i64 + 3).map(|i| i * 1000).collect();
        let mut seen = Vec::new();
        for at in [0, 1, 3, -1, i32::MIN, i32::MAX, n as i32, 2] {
            for k in [0, 1, n as i32, -1] {
                let mut out = vec![-1i64; n + 2];
                let r = call(
                    &[at, k],
                    &mut [BufferArg::from(&mut out[..]), BufferArg::from(&src[..])],
                );
                seen.push(format!("at={at} k={k} {r:?} {out:?}"));
            }
        }
        seen
    });
}

#[test]
fn narrowing_and_widening_agree_on_extreme_lanes() {
    let stage: Stage = Rc::new(|b| {
        let (zero, one) = (b.constant(0), b.constant(1));
        let x = load(b, Element::I64, 0, zero);
        let y = load(b, Element::I64, 0, one);
        let narrow = simd(b, Element::I64, SimdOp::Narrow(Element::I32), &[x, y]);
        store(b, Element::I32, 1, zero, narrow);
        let w = load(b, Element::U16, 2, zero);
        let high = SimdOp::Widen {
            to: Element::U32,
            high: true,
        };
        let hi = simd(b, Element::U16, high, &[w]);
        store(b, Element::U32, 3, zero, hi);
        let high = SimdOp::Widen {
            to: Element::I64,
            high: true,
        };
        let hi = simd(b, Element::I32, high, &[narrow]);
        store(b, Element::I64, 4, zero, hi);
        b.ret(zero);
        Ok(())
    });
    let (n, m, h) = (
        Element::I32.lanes(),
        Element::I64.lanes(),
        Element::U16.lanes(),
    );
    let kinds = [
        BufferKind::new(Element::I64, false),
        BufferKind::new(Element::I32, true),
        BufferKind::new(Element::U16, false),
        BufferKind::new(Element::U32, true),
        BufferKind::new(Element::I64, true),
    ];
    agree("narrow/widen", 0, &kinds, &stage, |call| {
        let x: Vec<i64> = (0..m + 1)
            .map(|i| [i64::MAX, i64::MIN, -1, 0x1_0000_0005, -0x1_0000_0005, 3][i % 6])
            .collect();
        let w: Vec<u16> = (0..h).map(|i| (i as u16).wrapping_mul(40000)).collect();
        let (mut a, mut c, mut d) = (vec![0i32; n], vec![0u32; n], vec![0i64; m]);
        let r = call(
            &[],
            &mut [
                BufferArg::from(&x[..]),
                BufferArg::from(&mut a[..]),
                BufferArg::from(&w[..]),
                BufferArg::from(&mut c[..]),
                BufferArg::from(&mut d[..]),
            ],
        );
        vec![format!("{r:?} {a:?} {c:?} {d:?}")]
    });
}
