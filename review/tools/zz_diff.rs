// Scratch differential test across backends (review only; not part of the repo).
use splice::frontend::{self, KernelBuilder};
use splice::hir::{Binary, Buffer, Simd, SimdOp};
use splice::simd::Element;
use splice::{BufferArg, BufferKind, Codegen, Executable};
use splice_cranelift::Cranelift;
use splice_lancy::Lancy;
use splice_llvm::{Llvm, LlvmManualSimd};

type Stage = std::rc::Rc<dyn Fn(&mut dyn KernelBuilder) -> splice::Result<()>>;

fn run_one<B: Codegen>(
    direct: bool,
    arity: usize,
    kinds: &[BufferKind],
    stage: &Stage,
    calls: &mut dyn FnMut(&dyn Fn(&[i32], &mut [BufferArg<'_>]) -> Result<i32, String>) -> Vec<String>,
) -> Vec<String>
where
    B::Output: Executable,
{
    eprintln!("RUN {} direct={direct}", std::any::type_name::<B>());
    let s = stage.clone();
    let built = if direct {
        B::build(arity, kinds, Box::new(move |b| s(b)))
    } else {
        let s2 = stage.clone();
        match frontend::materialize(arity, kinds, Box::new(move |b| s2(b))) {
            Ok(f) => B::compile(&f),
            Err(e) => Err(e.into()),
        }
    };
    match built {
        Ok(k) => calls(&|a, bufs| k.call(a, bufs).map_err(|e| e.to_string())),
        Err(e) => vec![format!("BUILD ERROR: {e:?}")],
    }
}

fn compare(name: &str, arity: usize, kinds: &[BufferKind], stage: Stage,
           mut calls: impl FnMut(&dyn Fn(&[i32], &mut [BufferArg<'_>]) -> Result<i32, String>) -> Vec<String>) -> usize {
    eprintln!("CASE {name}");
    let mut outs: Vec<(String, Vec<String>)> = Vec::new();
    for direct in [true, false] {
        outs.push((format!("lancy/{direct}"), run_one::<Lancy>(direct, arity, kinds, &stage, &mut calls)));
        outs.push((format!("cranelift/{direct}"), run_one::<Cranelift>(direct, arity, kinds, &stage, &mut calls)));
        outs.push((format!("llvm/{direct}"), run_one::<Llvm>(direct, arity, kinds, &stage, &mut calls)));
        outs.push((format!("llvm_manual/{direct}"), run_one::<LlvmManualSimd>(direct, arity, kinds, &stage, &mut calls)));
    }
    let base = &outs[0].1;
    let mut diffs = 0;
    for (n, o) in &outs[1..] {
        if o != base {
            diffs += 1;
            println!("DIFF {name}: {} vs {n}", outs[0].0);
            for (i, (x, y)) in base.iter().zip(o).enumerate() {
                if x != y {
                    println!("   case {i}: {x} | {y}");
                }
            }
            if base.len() != o.len() {
                println!("   len {} vs {}: {:?}", base.len(), o.len(), o.first());
            }
        }
    }
    diffs
}

const EDGE: [i32; 12] = [0, 1, -1, 2, 7, -7, 31, 32, 33, -32, i32::MIN, i32::MAX];

#[test]
fn diff_scalar_binary() {
    use Binary::*;
    let mut total = 0;
    for op in [Add, Sub, Mul, Div, Rem, And, Or, Xor, Eq, Ne, Lt, Le, Gt, Ge, Shl, Shr,
               CheckedAdd, CheckedSub, CheckedMul, CheckedShl, CheckedShr, Min, Max] {
        let stage: Stage = std::rc::Rc::new(move |b| {
            let r = b.binary(op, b.arg(0), b.arg(1));
            b.ret(r);
            Ok(())
        });
        total += compare(&format!("{op:?}"), 2, &[], stage, |call| {
            let mut v = Vec::new();
            for a in EDGE { for c in EDGE {
                let want = op.checked_eval(a, c).map(|x| x.to_string()).unwrap_or("ERR".into());
                let got = call(&[a, c], &mut []);
                let ok = match &got { Ok(x) => x.to_string() == want, Err(_) => want == "ERR" };
                if !ok { println!("REFMISMATCH {op:?}({a},{c}) got={got:?} ref={want}"); }
                v.push(format!("{op:?}({a},{c})={got:?} ref={want}"));
            }}
            v
        });
    }
    println!("scalar diffs: {total}");
}

fn lanes(e: Element) -> usize { e.lanes() }

// Load vector from buffer 0 (element e), apply unary op, store to buffer 1 (element `to`).
fn unary_vec(e: Element, to: Element, op: SimdOp, extra: usize) -> Stage {
    std::rc::Rc::new(move |b| {
        let zero = b.constant(0);
        let x = b.simd(Simd { element: e, op: SimdOp::Load(Buffer(0)) }, &[zero])?.unwrap();
        let mut args = vec![x];
        for i in 0..extra {
            let y = b.simd(Simd { element: e, op: SimdOp::Load(Buffer(0)) }, &[b.arg(i)])?.unwrap();
            args.push(y);
        }
        let r = b.simd(Simd { element: e, op }, &args)?.unwrap();
        b.simd(Simd { element: to, op: SimdOp::Store(Buffer(1)) }, &[zero, r])?;
        b.ret(zero);
        Ok(())
    })
}

#[test]
fn diff_simd_casts_and_float() {
    let mut total = 0;
    let n = lanes(Element::F32);
    let f32s: Vec<f32> = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3e9, -3e9, -1.5, 2.5, -0.0, 4294967040.0, 1e-40, 2147483520.0, 2147483648.0, -2147483904.0, 0.5, 65536.0, -1.0]
        .iter().cycle().take(n).copied().collect();
    for to in [Element::I32, Element::U32] {
        let st = unary_vec(Element::F32, to, SimdOp::Cast(to), 0);
        let src = f32s.clone();
        total += compare(&format!("cast f32->{to:?}"), 0, &[BufferKind::new(Element::F32, false), BufferKind::new(to, true)], st, |call| {
            let mut out = vec![0u32; n];
            let r = if to == Element::I32 {
                let mut o = vec![0i32; n];
                let r = call(&[], &mut [BufferArg::from(&src[..]), BufferArg::from(&mut o[..])]);
                out = o.iter().map(|&x| x as u32).collect(); r
            } else { call(&[], &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])]) };
            let reference: Vec<u32> = src.iter().map(|&x| if to == Element::I32 { x as i32 as u32 } else { x as u32 }).collect();
            vec![format!("{r:?} {out:?} ref_eq={}", out == reference)]
        });
    }
    // i64 <-> f64
    let m = lanes(Element::F64);
    let f64s: Vec<f64> = [f64::NAN, 1e19, -1e19, -1.5, 9.3e18, 2.5, f64::INFINITY, -0.0].iter().cycle().take(m).copied().collect();
    let st = unary_vec(Element::F64, Element::I64, SimdOp::Cast(Element::I64), 0);
    total += compare("cast f64->i64", 0, &[BufferKind::new(Element::F64, false), BufferKind::new(Element::I64, true)], st, |call| {
        let mut o = vec![0i64; m];
        let r = call(&[], &mut [BufferArg::from(&f64s[..]), BufferArg::from(&mut o[..])]);
        let reference: Vec<i64> = f64s.iter().map(|&x| x as i64).collect();
        vec![format!("{r:?} {o:?} ref_eq={}", o == reference)]
    });
    let i64s: Vec<i64> = [i64::MAX, i64::MIN, -1, 1 << 53 | 1, 3, -(1 << 60) - 1, 0, 12345].iter().cycle().take(m).copied().collect();
    let st = unary_vec(Element::I64, Element::F64, SimdOp::Cast(Element::F64), 0);
    total += compare("cast i64->f64", 0, &[BufferKind::new(Element::I64, false), BufferKind::new(Element::F64, true)], st, |call| {
        let mut o = vec![0f64; m];
        let r = call(&[], &mut [BufferArg::from(&i64s[..]), BufferArg::from(&mut o[..])]);
        let reference: Vec<f64> = i64s.iter().map(|&x| x as f64).collect();
        vec![format!("{r:?} {:?} ref_eq={}", o.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), o.iter().zip(&reference).all(|(a,b)| a.to_bits()==b.to_bits()))]
    });
    let u32s: Vec<u32> = [u32::MAX, 0x8000_0001, 16777217, 0, 1, 0xFFFF_FF7F, 3_000_000_000, 7].iter().cycle().take(n).copied().collect();
    let st = unary_vec(Element::U32, Element::F32, SimdOp::Cast(Element::F32), 0);
    total += compare("cast u32->f32", 0, &[BufferKind::new(Element::U32, false), BufferKind::new(Element::F32, true)], st, |call| {
        let mut o = vec![0f32; n];
        let r = call(&[], &mut [BufferArg::from(&u32s[..]), BufferArg::from(&mut o[..])]);
        let reference: Vec<f32> = u32s.iter().map(|&x| x as f32).collect();
        vec![format!("{r:?} {:?} ref_eq={}", o.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), o.iter().zip(&reference).all(|(a,b)| a.to_bits()==b.to_bits()))]
    });
    // f32 sqrt / mul_add / div on specials
    for op in [SimdOp::Sqrt, SimdOp::MulAdd, SimdOp::Div] {
        let extra = match op { SimdOp::MulAdd => 2, SimdOp::Div => 1, _ => 0 };
        let st = unary_vec(Element::F32, Element::F32, op, extra);
        let src: Vec<f32> = [0.1f32, -4.0, f32::NAN, 1e30, -0.0, 3.0, 16777217.0, 1.0000001].iter().cycle().take(n + 2).copied().collect();
        total += compare(&format!("{op:?} f32"), extra, &[BufferKind::new(Element::F32, false), BufferKind::new(Element::F32, true)], st, |call| {
            let mut o = vec![0f32; n];
            let args: Vec<i32> = (1..=extra as i32).collect();
            let r = call(&args, &mut [BufferArg::from(&src[..]), BufferArg::from(&mut o[..])]);
            vec![format!("{r:?} {:?}", o.iter().map(|x| x.to_bits()).collect::<Vec<_>>())]
        });
    }
    // Float compares with NaN -> select into i32 via Select
    for op in [SimdOp::Eq, SimdOp::Ne, SimdOp::Lt, SimdOp::Le, SimdOp::Gt, SimdOp::Ge] {
        let st: Stage = std::rc::Rc::new(move |b| {
            let zero = b.constant(0);
            let one = b.constant(1);
            let x = b.simd(Simd { element: Element::F32, op: SimdOp::Load(Buffer(0)) }, &[zero])?.unwrap();
            let y = b.simd(Simd { element: Element::F32, op: SimdOp::Load(Buffer(0)) }, &[one])?.unwrap();
            let m = b.simd(Simd { element: Element::F32, op }, &[x, y])?.unwrap();
            let c = b.simd(Simd { element: Element::I32, op: SimdOp::Count }, &[m]);
            let r = match c { Ok(Some(r)) => r, _ => {
                let a = b.simd(Simd { element: Element::F32, op: SimdOp::Splat(1f32.to_bits() as u64) }, &[])?.unwrap();
                let z = b.simd(Simd { element: Element::F32, op: SimdOp::Splat(0) }, &[])?.unwrap();
                let s = b.simd(Simd { element: Element::F32, op: SimdOp::Select }, &[m, a, z])?.unwrap();
                b.simd(Simd { element: Element::F32, op: SimdOp::Store(Buffer(1)) }, &[zero, s])?;
                zero
            }};
            b.ret(r);
            Ok(())
        });
        let src: Vec<f32> = [f32::NAN, 1.0, f32::NAN, 2.0, 1.0, -0.0, 0.0, f32::INFINITY].iter().cycle().take(n + 1).copied().collect();
        total += compare(&format!("fcmp {op:?}"), 0, &[BufferKind::new(Element::F32, false), BufferKind::new(Element::F32, true)], st, |call| {
            let mut o = vec![9f32; n];
            let r = call(&[], &mut [BufferArg::from(&src[..]), BufferArg::from(&mut o[..])]);
            vec![format!("{r:?} {o:?}")]
        });
    }
    println!("simd cast diffs: {total}");
}

#[test]
fn diff_simd_int() {
    let mut total = 0;
    let n = lanes(Element::I32);
    let src: Vec<i32> = (0..2 * n as i32).map(|i| [i32::MIN, -1, 0, 1, 31, 32, -33, i32::MAX, 100, -100, 7, 15, 16, 17, 64, -64][i as usize % 16].wrapping_add(i / 16)).collect();
    let ops = [SimdOp::Shr(5), SimdOp::Shr(37), SimdOp::RotateLeft(7), SimdOp::RotateLeft(33), SimdOp::Min, SimdOp::Max,
               SimdOp::Swizzle, SimdOp::PrefixSum, SimdOp::Mul, SimdOp::Not];
    for e in [Element::I32, Element::U32] {
        for op in ops {
            let extra = match op { SimdOp::Min | SimdOp::Max | SimdOp::Swizzle | SimdOp::Mul => 1, _ => 0 };
            let st = unary_vec(e, e, op, extra);
            let s = src.clone();
            total += compare(&format!("{e:?} {op:?}"), extra, &[BufferKind::new(e, false), BufferKind::new(e, true)], st, |call| {
                let mut v = Vec::new();
                for off in [1i32, 3, n as i32 - 1] {
                    let args: Vec<i32> = if extra == 1 { vec![off] } else { vec![] };
                    if e == Element::I32 {
                        let mut o = vec![0i32; n];
                        let r = call(&args, &mut [BufferArg::from(&s[..]), BufferArg::from(&mut o[..])]);
                        v.push(format!("{r:?} {o:?}"));
                    } else {
                        let su: Vec<u32> = s.iter().map(|&x| x as u32).collect();
                        let mut o = vec![0u32; n];
                        let r = call(&args, &mut [BufferArg::from(&su[..]), BufferArg::from(&mut o[..])]);
                        v.push(format!("{r:?} {o:?}"));
                    }
                }
                v
            });
        }
    }
    // FirstN with edge counts, Count result
    let st: Stage = std::rc::Rc::new(|b| {
        let m = b.simd(Simd { element: Element::U16, op: SimdOp::FirstN }, &[b.arg(0)])?.unwrap();
        let ones = b.simd(Simd { element: Element::U16, op: SimdOp::Splat(1) }, &[])?.unwrap();
        let zeros = b.simd(Simd { element: Element::U16, op: SimdOp::Splat(0) }, &[])?.unwrap();
        let s = b.simd(Simd { element: Element::U16, op: SimdOp::Select }, &[m, ones, zeros])?.unwrap();
        let zero = b.constant(0);
        b.simd(Simd { element: Element::U16, op: SimdOp::Store(Buffer(0)) }, &[zero, s])?;
        let m2 = b.simd(Simd { element: Element::I64, op: SimdOp::FirstN }, &[b.arg(0)])?.unwrap();
        let a = b.simd(Simd { element: Element::I64, op: SimdOp::Any }, &[m2])?.unwrap();
        let al = b.simd(Simd { element: Element::I64, op: SimdOp::All }, &[m2])?.unwrap();
        let r = b.binary(Binary::Add, a, al);
        let m3 = b.simd(Simd { element: Element::I32, op: SimdOp::FirstN }, &[b.arg(0)])?.unwrap();
        let c = b.simd(Simd { element: Element::I32, op: SimdOp::Count }, &[m3])?.unwrap();
        let ten = b.constant(10);
        let c = b.binary(Binary::Mul, c, ten);
        let r = b.binary(Binary::Add, r, c);
        b.ret(r);
        Ok(())
    });
    let nu = lanes(Element::U16);
    total += compare("first_n", 1, &[BufferKind::new(Element::U16, true)], st, |call| {
        let mut v = Vec::new();
        for k in [0, 1, 3, 8, 31, 32, 33, 65535, 65536, 65537, -1, i32::MIN, i32::MAX] {
            let mut o = vec![7u16; nu];
            let r = call(&[k], &mut [BufferArg::from(&mut o[..])]);
            v.push(format!("k={k} {r:?} {o:?}"));
        }
        v
    });
    // Compress/Expand
    for op in [SimdOp::Compress, SimdOp::Expand] {
        let st: Stage = std::rc::Rc::new(move |b| {
            let zero = b.constant(0);
            let one = b.constant(1);
            let e = Element::I32;
            let x = b.simd(Simd { element: e, op: SimdOp::Load(Buffer(0)) }, &[zero])?.unwrap();
            let f = b.simd(Simd { element: e, op: SimdOp::Load(Buffer(1)) }, &[zero])?.unwrap();
            let z = b.simd(Simd { element: e, op: SimdOp::Splat(0) }, &[])?.unwrap();
            let m = b.simd(Simd { element: e, op: SimdOp::Ne }, &[f, z])?.unwrap();
            let fb = b.simd(Simd { element: e, op: SimdOp::Load(Buffer(0)) }, &[one])?.unwrap();
            let r = if op == SimdOp::Compress {
                b.simd(Simd { element: e, op }, &[x, m])?.unwrap()
            } else {
                b.simd(Simd { element: e, op }, &[x, m, fb])?.unwrap()
            };
            b.simd(Simd { element: e, op: SimdOp::Store(Buffer(2)) }, &[zero, r])?;
            // compress store at arg offset
            b.simd(Simd { element: e, op: SimdOp::CompressStore(Buffer(3)) }, &[b.arg(0), m, x])?;
            b.ret(zero);
            Ok(())
        });
        let x: Vec<i32> = (100..100 + n as i32 + 1).collect();
        total += compare(&format!("{op:?}"), 1, &[BufferKind::Read, BufferKind::Read, BufferKind::Write, BufferKind::Write], st, |call| {
            let mut v = Vec::new();
            for (bits, off, dlen) in [(0b1011u32, 0, 8), (0xffff, 2, 5), (0, 100, 0), (0b1000_0000_0000_0001, 6, 7), (0b111, -1, 4)] {
                let flags: Vec<i32> = (0..n).map(|i| ((bits >> (i % 32)) & 1) as i32).collect();
                let mut o = vec![-9; n];
                let mut d = vec![-5; dlen];
                let r = call(&[off], &mut [BufferArg::from(&x[..]), BufferArg::from(&flags[..]), BufferArg::from(&mut o[..]), BufferArg::from(&mut d[..])]);
                v.push(format!("{r:?} {o:?} {d:?}"));
            }
            v
        });
    }
    // Gather/scatter with edge indices, u16 gather
    let st: Stage = std::rc::Rc::new(|b| {
        let zero = b.constant(0);
        let e = Element::U16;
        let idx = b.simd(Simd { element: e, op: SimdOp::Load(Buffer(0)) }, &[zero])?.unwrap();
        let g = b.simd(Simd { element: e, op: SimdOp::Gather(Buffer(1)) }, &[idx])?.unwrap();
        b.simd(Simd { element: e, op: SimdOp::Store(Buffer(2)) }, &[zero, g])?;
        b.simd(Simd { element: e, op: SimdOp::Scatter(Buffer(2)) }, &[idx, g])?;
        b.ret(zero);
        Ok(())
    });
    total += compare("u16 gather", 0, &[BufferKind::new(Element::U16, false), BufferKind::new(Element::U16, false), BufferKind::new(Element::U16, true)], st, |call| {
        let mut v = Vec::new();
        for (big, special) in [(70000usize, 65535u16), (65535, 65534), (65535, 65535), (40, 39), (40, 40), (32768, 32767), (32769, 32768)] {
            let src: Vec<u16> = (0..big).map(|i| (i * 7) as u16).collect();
            let mut idx: Vec<u16> = (0..nu).map(|i| (i as u16) % 30).collect();
            idx[3] = special;
            let mut o = vec![1u16; nu.max(big.min(nu+40))];
            let r = call(&[], &mut [BufferArg::from(&idx[..]), BufferArg::from(&src[..]), BufferArg::from(&mut o[..])]);
            v.push(format!("{big} {special} {r:?} {:?}", &o[..nu]));
        }
        v
    });
    // Contiguous store partial at negative/edge offsets; masked load select
    let st: Stage = std::rc::Rc::new(|b| {
        let e = Element::I64;
        let zero = b.constant(0);
        let x = b.simd(Simd { element: e, op: SimdOp::Iota }, &[])?.unwrap();
        let fl = b.simd(Simd { element: e, op: SimdOp::FirstN }, &[b.arg(1)])?.unwrap();
        let l = b.simd(Simd { element: e, op: SimdOp::LoadSelect(Buffer(1)) }, &[b.arg(0), fl, x])?.unwrap();
        b.simd(Simd { element: e, op: SimdOp::Store(Buffer(0)) }, &[zero, l])?;
        b.simd(Simd { element: e, op: SimdOp::Store(Buffer(0)) }, &[b.arg(0), x])?;
        b.ret(zero);
        Ok(())
    });
    let ni = lanes(Element::I64);
    total += compare("i64 store edges", 2, &[BufferKind::new(Element::I64, true), BufferKind::new(Element::I64, false)], st, |call| {
        let mut v = Vec::new();
        let src: Vec<i64> = (0..ni as i64 + 3).map(|i| i * 1000).collect();
        for off in [0, 1, 3, -1, i32::MIN, i32::MAX, ni as i32, 2] {
            for k in [0, 1, ni as i32, -1] {
                let mut o = vec![-1i64; ni + 2];
                let r = call(&[off, k], &mut [BufferArg::from(&mut o[..]), BufferArg::from(&src[..])]);
                v.push(format!("off={off} k={k} {r:?} {o:?}"));
            }
        }
        v
    });
    // Narrow/widen
    let st: Stage = std::rc::Rc::new(|b| {
        let zero = b.constant(0);
        let one = b.constant(1);
        let x = b.simd(Simd { element: Element::I64, op: SimdOp::Load(Buffer(0)) }, &[zero])?.unwrap();
        let y = b.simd(Simd { element: Element::I64, op: SimdOp::Load(Buffer(0)) }, &[one])?.unwrap();
        let nrw = b.simd(Simd { element: Element::I64, op: SimdOp::Narrow(Element::I32) }, &[x, y])?.unwrap();
        b.simd(Simd { element: Element::I32, op: SimdOp::Store(Buffer(1)) }, &[zero, nrw])?;
        let w = b.simd(Simd { element: Element::U16, op: SimdOp::Load(Buffer(2)) }, &[zero])?.unwrap();
        let hi = b.simd(Simd { element: Element::U16, op: SimdOp::Widen { to: Element::U32, high: true } }, &[w])?.unwrap();
        b.simd(Simd { element: Element::U32, op: SimdOp::Store(Buffer(3)) }, &[zero, hi])?;
        let hi2 = b.simd(Simd { element: Element::I32, op: SimdOp::Widen { to: Element::I64, high: true } }, &[nrw])?.unwrap();
        b.simd(Simd { element: Element::I64, op: SimdOp::Store(Buffer(4)) }, &[zero, hi2])?;
        b.ret(zero);
        Ok(())
    });
    total += compare("narrow/widen", 0, &[BufferKind::new(Element::I64, false), BufferKind::new(Element::I32, true), BufferKind::new(Element::U16, false), BufferKind::new(Element::U32, true), BufferKind::new(Element::I64, true)], st, |call| {
        let x: Vec<i64> = (0..ni as i64 + 1).map(|i| [i64::MAX, i64::MIN, -1, 0x1_0000_0005, -0x1_0000_0005, 3][i as usize % 6]).collect();
        let w: Vec<u16> = (0..nu).map(|i| (i as u16).wrapping_mul(40000)).collect();
        let mut a = vec![0i32; n]; let mut c = vec![0u32; n]; let mut d = vec![0i64; ni];
        let r = call(&[], &mut [BufferArg::from(&x[..]), BufferArg::from(&mut a[..]), BufferArg::from(&w[..]), BufferArg::from(&mut c[..]), BufferArg::from(&mut d[..])]);
        vec![format!("{r:?} {a:?} {c:?} {d:?}")]
    });
    println!("simd int diffs: {total}");
}

#[test]
fn iso_fcmp_count() {
    let kinds = [BufferKind::new(Element::F32, false)];
    fn stage(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let zero = b.constant(0);
        let one = b.constant(1);
        let x = b.simd(Simd { element: Element::F32, op: SimdOp::Load(Buffer(0)) }, &[zero])?.unwrap();
        let y = b.simd(Simd { element: Element::F32, op: SimdOp::Load(Buffer(0)) }, &[one])?.unwrap();
        let m = b.simd(Simd { element: Element::F32, op: SimdOp::Eq }, &[x, y])?.unwrap();
        let c = b.simd(Simd { element: Element::I32, op: SimdOp::Count }, &[m])?.unwrap();
        b.ret(c);
        Ok(())
    }
    let f = frontend::materialize(0, &kinds, Box::new(stage)).unwrap();
    eprintln!("IR ok");
    let k = Cranelift::compile(&f).unwrap();
    eprintln!("built");
    let n = Element::F32.lanes();
    let src: Vec<f32> = (0..n + 1).map(|i| (i / 2) as f32).collect();
    let r = k.call(&[], &mut [BufferArg::from(&src[..])]);
    eprintln!("called {r:?}");
}

#[test]
fn iso_fcmp_count_aligned() {
    let kinds = [BufferKind::new(Element::F32, false)];
    fn stage(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let zero = b.constant(0);
        let one = b.constant(1);
        let x = b.simd(Simd { element: Element::F32, op: SimdOp::Load(Buffer(0)) }, &[zero])?.unwrap();
        let y = b.simd(Simd { element: Element::F32, op: SimdOp::Load(Buffer(0)) }, &[zero])?.unwrap();
        let _ = one;
        let m = b.simd(Simd { element: Element::F32, op: SimdOp::Eq }, &[x, y])?.unwrap();
        let c = b.simd(Simd { element: Element::I32, op: SimdOp::Count }, &[m])?.unwrap();
        b.ret(c);
        Ok(())
    }
    let f = frontend::materialize(0, &kinds, Box::new(stage)).unwrap();
    eprintln!("IR ok");
    let k = Cranelift::compile(&f).unwrap();
    eprintln!("built");
    let n = Element::F32.lanes();
    let src: Vec<f32> = (0..n + 1).map(|i| (i / 2) as f32).collect();
    let r = k.call(&[], &mut [BufferArg::from(&src[..])]);
    eprintln!("called {r:?}");
}

#[test]
fn iso_fcmp_count_add() {
    let kinds = [BufferKind::new(Element::F32, false)];
    fn stage(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let zero = b.constant(0);
        let one = b.constant(1);
        let x = b.simd(Simd { element: Element::F32, op: SimdOp::Load(Buffer(0)) }, &[zero])?.unwrap();
        let y = b.simd(Simd { element: Element::F32, op: SimdOp::Load(Buffer(0)) }, &[one])?.unwrap();
        let m = b.simd(Simd { element: Element::F32, op: SimdOp::Add }, &[x, y])?.unwrap();
        let _ = m;
        let c = zero;
        b.ret(c);
        Ok(())
    }
    let f = frontend::materialize(0, &kinds, Box::new(stage)).unwrap();
    eprintln!("IR ok");
    let k = Cranelift::compile(&f).unwrap();
    eprintln!("built");
    let n = Element::F32.lanes();
    let src: Vec<f32> = (0..n + 1).map(|i| (i / 2) as f32).collect();
    let r = k.call(&[], &mut [BufferArg::from(&src[..])]);
    eprintln!("called {r:?}");
}
