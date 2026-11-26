//! Encoding choices of emitted code: VEX forms and `vzeroupper` with AVX
//! (X64-02).
use crate::hir::{Function, Op, Terminator, Type, Value};
use iced_x86::{Decoder, DecoderOptions, EncodingKind};
use splice_target::{Profile, Target};

unsafe extern "C" fn double(_: *mut u8, value: u32, _: *const u8, _: i32) -> i64 {
    i64::from(value.wrapping_mul(2))
}

static DOUBLE: crate::hir::RustFunction = crate::hir::RustFunction {
    name: "double",
    entry: double,
};

/// The listing of `text` compiled for `profile`.
fn listing(text: &str, profile: Profile) -> String {
    let f = crate::hir::text::parse(text, &[&DOUBLE]).unwrap();
    let target = Target::from_profile(profile).unwrap();
    let code =
        crate::lir::x64::emit::emit(&target, f.verified_module().unwrap(), false, &mut |_| {})
            .unwrap_or_else(|e| panic!("{e:?}"));
    super::disassembly::disassemble(&code.bytes[..code.text_len]).unwrap()
}

#[test]
fn avx_code_clears_upper_vector_state_before_host_calls_and_return() {
    use Profile;
    // A Rust call and the numeric fallback (`sin`) both enter host code.
    let text = "fn main(x:f64 v:i32 fail:i32) -> f64
  p = call_scratch 0 8
  d = rust_call double p v p fail
  s = sin x
  f = cast f64 d
  r = add s f
  ret r
";
    for profile in [Profile::Avx2, Profile::Avx512] {
        let code = listing(text, profile);
        let lines: Vec<&str> = code.lines().collect();
        let calls: Vec<usize> = (0..lines.len())
            .filter(|&i| lines[i].contains(" call "))
            .collect();
        assert_eq!(calls.len(), 2, "{code}");
        for i in calls {
            assert!(lines[i - 1].ends_with("vzeroupper"), "{code}");
        }
        let last = lines
            .iter()
            .rposition(|l| l.ends_with("vzeroupper"))
            .unwrap();
        assert!(lines[last..].iter().any(|l| l.ends_with(" ret")), "{code}");
        assert!(
            !lines[last..]
                .iter()
                .any(|l| l.contains("ymm") || l.contains("zmm")),
            "{code}"
        );
    }
    assert!(!listing(text, Profile::Sse).contains("vzeroupper"));
}

/// Legacy SSE instructions in the code of `f` compiled for `profile`.
fn legacy(f: &Function, profile: Profile) -> Vec<String> {
    let target = Target::from_profile(profile).unwrap();
    let code =
        crate::lir::x64::emit::emit(&target, f.verified_module().unwrap(), false, &mut |_| {})
            .unwrap_or_else(|e| panic!("{e:?}"));
    let text = &code.bytes[..code.text_len];
    Decoder::new(64, text, DecoderOptions::NONE)
        .into_iter()
        .filter(|i| {
            i.code().op_code().encoding() == EncodingKind::Legacy
                && (0..i.op_count()).any(|n| i.op_register(n).is_xmm())
        })
        .map(|i| i.to_string())
        .collect()
}

/// With AVX every scalar and vector operation, its spills and the root's
/// saves use VEX or EVEX forms.
#[test]
fn avx_code_has_no_legacy_sse_instructions() {
    use crate::hir::Ternary;
    use crate::jit::tests::numeric_coverage::{BINARY, LANES, UNARY};
    let mut found = Vec::new();
    let mut check = |params: &[Type], make: &dyn Fn(&[Value]) -> Op| {
        let probe = Function::new(params, Type::I32);
        let op = make(probe.params(probe.entry()));
        let Some((_, _, output, _)) = crate::hir::operation::operation(op, |v| probe.value_type(v))
        else {
            return;
        };
        let mut f = Function::new(params, output);
        let b = f.entry();
        let Ok(v) = f.append(b, op) else { return };
        f.set_terminator(b, Terminator::Return(v));
        if f.verified_module().is_err() {
            return;
        }
        for profile in [Profile::Avx2, Profile::Avx512] {
            for i in legacy(&f, profile) {
                found.push(format!("{profile:?} {op:?} {params:?}: {i}"));
            }
        }
    };
    for lane in LANES {
        for ty in [lane.scalar(), Type::Vector(lane)] {
            for op in UNARY.into_iter().filter(|op| op.supports(ty)) {
                check(&[ty], &|p| Op::Unary(op, p[0]));
            }
            for op in BINARY.into_iter().filter(|op| op.supports(ty)) {
                check(&[ty, ty], &|p| Op::Bin(op, p[0], p[1]));
            }
            for op in [Ternary::Select, Ternary::BitSelect, Ternary::MulAdd] {
                let first = if op == Ternary::Select { Type::I32 } else { ty };
                check(&[first, ty, ty], &|p| Op::Ternary(op, p[0], p[1], p[2]));
            }
            for to in LANES {
                let to = if ty == lane.scalar() {
                    to.scalar()
                } else {
                    Type::Vector(to)
                };
                check(&[ty], &|p| Op::Cast(to, p[0]));
            }
        }
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}
