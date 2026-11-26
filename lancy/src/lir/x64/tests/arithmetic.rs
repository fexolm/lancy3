use crate::hir::ssa::InterpretError;
use crate::hir::{Binary, Function, Op as H, Terminator};
use crate::{
    CallError, ScalarKernel, Trap,
    lir::x64::{
        compile::compile,
        inst::{Alu, Inst, Rhs},
    },
};

#[test]
fn arithmetic_and_spills_match_the_reference() {
    use Binary::*;
    let ops = [
        Add, Sub, Mul, And, Or, Xor, Eq, Ne, Lt, Le, Gt, Ge, Shl, Shr, Min, Max, Div, Rem,
    ];
    for op in ops {
        let mut f = Function::scalar(40);
        let b = f.entry();
        let args = f.params(b).to_vec();
        let mut result = f.append(b, H::Bin(op, args[38], args[39])).unwrap();
        for a in &args[..38] {
            result = f.append(b, H::Bin(Add, result, *a)).unwrap();
        }
        f.set_terminator(b, Terminator::Return(result));
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        assert!(k.workspace_words() >= 36);
        let mut input = [1; 40];
        for left in [i32::MIN, -42, 0, 1, i32::MAX] {
            for right in [i32::MIN, -1, 0, 7, 32, 63, 0x5566_7788] {
                input[38] = left;
                input[39] = right;
                let expected = f.interpret(&input);
                let native = k.call(&input);
                match &expected {
                    Ok(value) => assert_eq!(native, Ok(*value), "{op:?}({left}, {right})"),
                    Err(_) => {
                        let trap = if right == 0 {
                            CallError::Trap(Trap::DivisionByZero)
                        } else {
                            CallError::Trap(Trap::DivisionOverflow)
                        };
                        assert_eq!(expected, Err(InterpretError::Trap(trap.clone())));
                        assert_eq!(native, Err(trap), "{op:?}({left}, {right})");
                    }
                }
            }
        }
    }
}

#[test]
fn selection_uses_an_immediate_and_drops_an_unused_multiply() {
    let mut f = Function::scalar(0);
    let b = f.entry();
    let two = f.append(b, H::Const(2)).unwrap();
    let four = f.append(b, H::Bin(Binary::Add, two, two)).unwrap();
    f.append_inst(b, H::Bin(Binary::Mul, four, two)).unwrap();
    f.set_terminator(b, Terminator::Return(four));
    let source = f.verified_module().unwrap();
    let module = compile(crate::lir::x64::Machine::host(), source, &mut |_| {}).unwrap();
    assert!(matches!(
        module.functions[module.call_graph.root()].body.insts(),
        [
            Inst::Imm(_, 2),
            Inst::Alu(Alu::Add, _, _, Rhs::Imm(2)),
            Inst::Ret(_)
        ]
    ));
    assert_eq!(
        ScalarKernel::compile(&f, &crate::test_target::options())
            .unwrap()
            .call(&[]),
        Ok(4)
    );
}

#[test]
fn aliased_fixed_operands_keep_the_live_input() {
    for op in [Binary::Shl, Binary::Shr, Binary::Div, Binary::Rem] {
        let mut f = Function::scalar(1);
        let b = f.entry();
        let x = f.params(b)[0];
        let y = f.append(b, H::Bin(op, x, x)).unwrap();
        let result = f.append(b, H::Bin(Binary::Xor, y, x)).unwrap();
        f.set_terminator(b, Terminator::Return(result));
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        for x in [i32::MIN, -33, -1, 0, 1, 31, 32, i32::MAX] {
            let value = match op {
                Binary::Shl => Some(x.wrapping_shl(x as u32)),
                Binary::Shr => Some(x.wrapping_shr(x as u32)),
                Binary::Div => x.checked_div(x),
                Binary::Rem => x.checked_rem(x),
                _ => unreachable!(),
            };
            assert_eq!(
                k.call(&[x]),
                value
                    .map(|y| y ^ x)
                    .ok_or(CallError::Trap(Trap::DivisionByZero))
            );
        }
    }
}

#[test]
fn disjoint_pressure_peaks_reuse_workspace() {
    fn compile(phases: usize) -> ScalarKernel {
        let mut f = Function::scalar(1);
        let b = f.entry();
        let x = f.params(b)[0];
        let mut sum = x;
        for _ in 0..phases {
            let values: Vec<_> = (1..=32)
                .map(|i| {
                    let offset = f.append(b, H::Const(i)).unwrap();
                    f.append(b, H::Bin(Binary::Add, x, offset)).unwrap()
                })
                .collect();
            for v in values {
                sum = f.append(b, H::Bin(Binary::Add, sum, v)).unwrap();
            }
        }
        f.set_terminator(b, Terminator::Return(sum));
        ScalarKernel::compile(&f, &crate::test_target::options()).unwrap()
    }
    let single = compile(1);
    let many = compile(8);
    assert_eq!(single.call(&[19]), Ok(19 + 32 * 19 + 528));
    assert_eq!(many.call(&[19]), Ok(19 + 8 * (32 * 19 + 528)));
    assert!(single.workspace_words() >= 32);
    assert!(many.workspace_words() < 2 * single.workspace_words());
}
