use crate::hir::{Binary, Function, Op as H, Terminator};
use crate::{
    CallError, ScalarKernel,
    lir::{instruction::Op, x64::graph::Graph},
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
        let k = ScalarKernel::compile(&f).unwrap();
        assert!(k.workspace_words() >= 36);
        let mut input = [1; 40];
        for left in [i32::MIN, -42, 0, 1, i32::MAX] {
            for right in [i32::MIN, -1, 0, 7, 32, 63, 0x5566_7788] {
                input[38] = left;
                input[39] = right;
                let expected = f.interpret(&input);
                let native = k.call(&input);
                match expected {
                    Ok(value) => assert_eq!(native, Ok(value), "{op:?}({left}, {right})"),
                    Err(_) => {
                        let trap = if right == 0 {
                            CallError::DivisionByZero
                        } else {
                            CallError::DivisionOverflow
                        };
                        assert_eq!(native, Err(trap), "{op:?}({left}, {right})");
                    }
                }
            }
        }
    }
}

#[test]
fn explicit_instructions_survive_lowering_and_allocation() {
    let mut f = Function::scalar(0);
    let b = f.entry();
    let two = f.append(b, H::Const(2)).unwrap();
    let four = f.append(b, H::Bin(Binary::Add, two, two)).unwrap();
    f.append(b, H::Bin(Binary::Mul, four, two));
    f.set_terminator(b, Terminator::Return(four));
    let graph = Graph::lower(&f).unwrap();
    assert!(matches!(
        &graph.functions[graph.root].ops[..],
        [
            Op::Imm(_, 2),
            Op::Alu(_, Binary::Add, ..),
            Op::Alu(_, Binary::Mul, ..)
        ]
    ));
    assert_eq!(ScalarKernel::compile(&f).unwrap().call(&[]), Ok(4));
}
