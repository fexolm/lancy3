use crate::ScalarKernel;
use crate::hir::{Binary, Function, Lane, Op, Scale, Simd, Terminator, Type};
use crate::lir::x64::{
    compile::compile,
    inst::{Address, Inst, Rhs},
};

fn selected(f: &Function) -> Vec<Inst> {
    let m = f.verified_module().unwrap();
    let m = compile(crate::lir::x64::Machine::host(), m, &mut |_| {}).unwrap();
    m.functions[m.call_graph.root()].body.insts().to_vec()
}

#[test]
fn branch_covers_a_single_use_comparison_and_retains_a_shared_mask() {
    for shared in [false, true] {
        let mut f = Function::scalar(1);
        let b = f.entry();
        let x = f.params(b)[0];
        let one = f.append(b, Op::Const(1)).unwrap();
        let cond = f.append(b, Op::Bin(Binary::Lt, x, one)).unwrap();
        let yes = f.create_block();
        let no = f.create_block();
        f.set_terminator(yes, Terminator::Return(if shared { cond } else { x }));
        f.set_terminator(no, Terminator::Return(one));
        let yes = f.add_edge(yes, &[]);
        let no = f.add_edge(no, &[]);
        f.set_terminator(b, Terminator::Branch { cond, yes, no });
        let insts = selected(&f);
        assert_eq!(
            insts
                .iter()
                .any(|i| matches!(i, Inst::BrCmp(_, _, _, Rhs::Imm(1)))),
            !shared
        );
        assert_eq!(insts.iter().any(|i| matches!(i, Inst::Compare(..))), shared);
        let code = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        assert_eq!(code.call(&[-9]), Ok(if shared { -1 } else { -9 }));
        assert_eq!(code.call(&[2]), Ok(1));
    }
}

#[test]
fn all_true_branch_only_covers_an_unshared_result() {
    let cpu = crate::lir::x64::Cpu::host();
    for shared in [false, true] {
        let mut f = Function::scalar(1);
        let b = f.entry();
        let x = f.params(b)[0];
        let mask = f.append(b, Op::Simd(Simd::Splat(x))).unwrap();
        let cond = f.append(b, Op::Simd(Simd::AllTrue(mask))).unwrap();
        let yes = f.create_block();
        let no = f.create_block();
        f.set_terminator(yes, Terminator::Return(if shared { cond } else { x }));
        f.set_terminator(no, Terminator::Return(x));
        let yes = f.add_edge(yes, &[]);
        let no = f.add_edge(no, &[]);
        f.set_terminator(b, Terminator::Branch { cond, yes, no });
        assert_eq!(
            selected(&f)
                .iter()
                .any(|i| matches!(i, Inst::BranchAll(Lane::I32, _))),
            cfg!(feature = "experimental-simd") && !shared && (cpu.bytes != 32 || cpu.isa.avx2)
        );
        let code = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        assert_eq!(code.call(&[0]), Ok(0));
        assert_eq!(code.call(&[1]), Ok(if shared { -1 } else { 1 }));
    }
}

#[test]
fn load_covers_scaled_address_and_shift_covers_its_constant() {
    let mut f = Function::new(&[Type::Ptr, Type::I32], Type::I32);
    let b = f.entry();
    let [base, i] = f.params(b).try_into().unwrap();
    let at = f.append(b, Op::PtrOffset(base, i, Scale::Four)).unwrap();
    let v = f.append(b, Op::Load(Type::I32, at)).unwrap();
    let n = f.append(b, Op::Const(48)).unwrap();
    let shifted = f.append(b, Op::Bin(Binary::Shr, v, n)).unwrap();
    f.set_terminator(b, Terminator::Return(shifted));
    let insts = selected(&f);
    assert!(insts.iter().any(|i| matches!(
        i,
        Inst::Load(
            _,
            _,
            Address {
                index: Some((_, Scale::Four)),
                ..
            }
        )
    )));
    assert!(insts.iter().any(|i| matches!(
        i,
        Inst::Shift {
            count: Rhs::Imm(16),
            ..
        }
    )));
    assert!(
        !insts
            .iter()
            .any(|i| matches!(i, Inst::Lea(..) | Inst::Imm(..)))
    );
    let words = [-0x123456i32, 0];
    let args = [words[1..].as_ptr() as u64, u32::MAX as u64];
    let code = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
    // SAFETY: index -1 from the second element addresses the first live i32.
    assert_eq!(
        unsafe { code.call_raw(&args) }.unwrap() as i32,
        words[0] >> 16
    );
}

#[test]
fn all_scalar_casts_up_to_eight_bytes_emit_call_free_machine_code() {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};
    // 128-bit conversions are cold: they call the shared numeric fallback.
    let lanes = || {
        crate::jit::tests::numeric_coverage::LANES
            .into_iter()
            .filter(|lane| lane.bytes() <= 8)
    };
    for from in lanes() {
        for to in lanes() {
            let f = crate::jit::tests::numeric_coverage::conversions::scalar_cast(from, to);
            let module = f.verified_module().unwrap();
            let code = crate::lir::x64::emit::emit(
                &crate::test_target::target(),
                module,
                false,
                &mut |_| {},
            )
            .unwrap();
            for inst in Decoder::new(64, &code.bytes[..code.text_len], DecoderOptions::NONE) {
                assert!(!inst.is_invalid(), "{from:?} -> {to:?} at {}", inst.ip());
                assert_ne!(
                    inst.mnemonic(),
                    Mnemonic::Call,
                    "{from:?} -> {to:?} at {}",
                    inst.ip()
                );
            }
        }
    }
}

/// Scalars of at most eight bytes select machine instructions (X64-01).
/// Only cold operations call the numeric fallback: library math, the float
/// neighbours and `signum`, arithmetic of 16-bit floats (x86 has it
/// only for AVX512-FP16 vectors), and what SSE2 lacks: rounding and FMA.
#[test]
fn scalar_operations_up_to_eight_bytes_select_machine_instructions() {
    use crate::hir::{Ternary, Unary as U, Value};
    use crate::jit::tests::numeric_coverage::{BINARY, LANES, UNARY};
    use crate::support::numeric::Action;
    use Binary as B;
    let isa = crate::lir::x64::Cpu::host().isa;
    let cold = |action: Action, lane: Lane| {
        let half = matches!(lane, Lane::F16 | Lane::BF16);
        match action {
            Action::Unary(
                U::Exp
                | U::Exp2
                | U::Ln
                | U::Log2
                | U::Log10
                | U::Sin
                | U::Cos
                | U::Tan
                | U::Asin
                | U::Acos
                | U::Atan
                | U::Sinh
                | U::Cosh
                | U::Tanh
                | U::Asinh
                | U::Acosh
                | U::Atanh
                | U::ExpM1
                | U::Ln1p
                | U::Cbrt
                | U::NextUp
                | U::NextDown
                | U::Signum,
            )
            | Action::Binary(B::Pow | B::Atan2 | B::Hypot) => true,
            Action::Binary(B::Rem) => lane.float(),
            Action::Unary(U::Floor | U::Ceil | U::Trunc | U::Nearest | U::Round | U::Fract) => {
                half || !isa.sse41
            }
            Action::Ternary(Ternary::MulAdd) => half || !isa.fma,
            Action::Unary(
                U::Not
                | U::Neg
                | U::Abs
                | U::SignBit
                | U::IsNan
                | U::IsInfinite
                | U::IsFinite
                | U::IsNormal
                | U::IsSubnormal,
            )
            | Action::Binary(B::And | B::Or | B::Xor | B::CopySign)
            | Action::Ternary(Ternary::Select | Ternary::BitSelect)
            | Action::Cast => false,
            _ => half,
        }
    };
    let check = |params: &[Type], make: &dyn Fn(&[Value]) -> Op| {
        // The parameters' values are the same in any function of this signature.
        let probe = Function::new(params, Type::I32);
        let op = make(probe.params(probe.entry()));
        let (action, input, output, _) =
            crate::hir::operation::operation(op, |v| probe.value_type(v)).unwrap();
        let mut f = Function::new(params, output);
        let b = f.entry();
        let v = f.append(b, op).unwrap();
        f.set_terminator(b, Terminator::Return(v));
        let lane = input.lane().unwrap();
        let falls = selected(&f).iter().any(|i| matches!(i, Inst::Fallback(_)));
        assert_eq!(falls, cold(action, lane), "{op:?} on {lane:?}");
    };
    for lane in LANES.into_iter().filter(|lane| lane.bytes() <= 8) {
        let ty = lane.scalar();
        for op in UNARY.into_iter().filter(|op| op.supports(ty)) {
            check(&[ty], &|p| Op::Unary(op, p[0]));
        }
        for op in BINARY.into_iter().filter(|op| op.supports(ty)) {
            check(&[ty, ty], &|p| Op::Bin(op, p[0], p[1]));
        }
        for op in [Ternary::Select, Ternary::BitSelect, Ternary::MulAdd] {
            if op == Ternary::MulAdd && !lane.float() {
                continue;
            }
            let first = if op == Ternary::Select { Type::I32 } else { ty };
            check(&[first, ty, ty], &|p| Op::Ternary(op, p[0], p[1], p[2]));
        }
        for to in LANES.into_iter().filter(|to| to.bytes() <= 8) {
            check(&[ty], &|p| Op::Cast(to.scalar(), p[0]));
        }
    }
}
