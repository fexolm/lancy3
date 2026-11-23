//! BE-01: without AVX, Cranelift 0.132 folds an unaligned vector load into a
//! legacy-SSE `cmpps`/`cmppd` memory operand, which faults unless the address
//! is 16-byte aligned.
use splice::frontend::{self, KernelBuilder};
use splice::hir::{Buffer, Simd, SimdOp};
use splice::simd::Element;
use splice::{BufferArg, BufferKind, Codegen, Executable};
use splice_cranelift::Cranelift;

/// `count(buf[0..] == buf[1..])`: the second load starts one element past the
/// slice's base, so it is misaligned for any 16-byte-aligned allocation.
fn shifted_compare(element: Element) -> impl Fn(&mut dyn KernelBuilder) -> splice::Result<()> {
    move |b| {
        let zero = b.constant(0);
        let one = b.constant(1);
        let load = SimdOp::Load(Buffer(0));
        let x = b.simd(Simd { element, op: load }, &[zero])?.unwrap();
        let y = b.simd(Simd { element, op: load }, &[one])?.unwrap();
        let mask = b
            .simd(
                Simd {
                    element,
                    op: SimdOp::Eq,
                },
                &[x, y],
            )?
            .unwrap();
        // Count takes 32-bit lanes; wider masks report whether any lane matched.
        let result = if element.lanes() == Element::I32.lanes() {
            let count = Simd {
                element: Element::I32,
                op: SimdOp::Count,
            };
            b.simd(count, &[mask])?
        } else {
            b.simd(
                Simd {
                    element,
                    op: SimdOp::Any,
                },
                &[mask],
            )?
        };
        b.ret(result.unwrap());
        Ok(())
    }
}

fn check_shifted_compares() {
    for element in [Element::F32, Element::F64] {
        let kinds = [BufferKind::new(element, false)];
        let lanes = element.lanes();
        // Pairs of equal values: lane i equals lane i + 1 exactly when i is even.
        let expected = if element == Element::F32 {
            lanes.div_ceil(2) as i32
        } else {
            -1 // `Any` returns a true scalar mask.
        };
        let ir = frontend::materialize(0, &kinds, Box::new(shifted_compare(element))).unwrap();
        for code in [
            Cranelift::build(0, &kinds, Box::new(shifted_compare(element))).unwrap(),
            Cranelift::compile(&ir).unwrap(),
        ] {
            let result = match element {
                Element::F32 => {
                    let data: Vec<f32> = (0..=lanes).map(|i| (i / 2) as f32).collect();
                    code.call(&[], &mut [BufferArg::from(&data[..])])
                }
                _ => {
                    let data: Vec<f64> = (0..=lanes).map(|i| (i / 2) as f64).collect();
                    code.call(&[], &mut [BufferArg::from(&data[..])])
                }
            };
            assert_eq!(result, Ok(expected), "{element:?}");
        }
    }
}

#[test]
fn unaligned_float_compare_operands() {
    check_shifted_compares();
}

/// The SIMD profile is fixed per process, so run the same check in a child
/// that selects the SSE profile, where the faulting encoding is chosen.
#[cfg(target_arch = "x86_64")]
#[test]
fn unaligned_float_compare_operands_under_sse() {
    const CHILD: &str = "SPLICE_CRANELIFT_SSE_CHILD";
    if std::env::var_os(CHILD).is_some() {
        return check_shifted_compares();
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "unaligned_float_compare_operands_under_sse"])
        .env("SPLICE_SIMD", "sse")
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert!(status.success(), "SSE child failed: {status}");
}
