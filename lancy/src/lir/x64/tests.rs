use super::*;
use crate::ScalarKernel;
use crate::hir::{Buffer, BufferArg, BufferKind, Op as H, Terminator};

#[test]
fn arithmetic_and_spills_match_the_reference() {
    use Binary::*;
    let ops = [
        Add, Sub, Mul, And, Or, Xor, Eq, Ne, Lt, Le, Gt, Ge, Shl, Shr, Min, Max, Div, Rem,
    ];
    for op in ops {
        let mut f = Function::new(40, vec![]);
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
                    Err(reason) => assert!(native.unwrap_err().contains(&reason)),
                }
            }
        }
    }
}

#[test]
fn checked_nested_access_preserves_prior_stores_and_forwards_buffer_indices() {
    let mut leaf = Function::new(2, vec![BufferKind::Read, BufferKind::Write]);
    let b = leaf.entry();
    let input = leaf.params(b).to_vec();
    let read = leaf.append(b, H::Load(Buffer(0), input[0])).unwrap();
    let len = leaf.append(b, H::BufferLen(Buffer(0))).unwrap();
    let sum = leaf.append(b, H::Bin(Binary::Add, read, len)).unwrap();
    leaf.append(b, H::Store(Buffer(1), input[1], sum));
    leaf.set_terminator(b, Terminator::Return(sum));

    let mut root = Function::new(2, vec![BufferKind::Write, BufferKind::Read]);
    let b = root.entry();
    let args = root.params(b).to_vec();
    let zero = root.append(b, H::Const(0)).unwrap();
    let nine = root.append(b, H::Const(9)).unwrap();
    root.append(b, H::Store(Buffer(0), zero, nine));
    let call = root.add_call(Arc::new(leaf), &args, &[Buffer(1), Buffer(0)]);
    let result = root.append(b, H::Call(call)).unwrap();
    let two = root.append(b, H::Const(2)).unwrap();
    root.append(b, H::Store(Buffer(0), two, result));
    root.set_terminator(b, Terminator::Return(result));
    let k = ScalarKernel::compile(&root).unwrap();
    for (i, j) in [(-1, 1), (1, -1), (3, 0), (0, 3), (1, 1)] {
        let data = [2, 42, 1];
        let mut out = [-77; 3];
        let result = k.call_with_buffers(
            &[i, j],
            &mut [BufferArg::Write(&mut out), BufferArg::Read(&data)],
        );
        if (i, j) == (1, 1) {
            assert_eq!(result, Ok(45));
            assert_eq!(out, [9, 45, 45]);
        } else {
            assert!(result.unwrap_err().contains("bounds"));
            assert_eq!(out, [9, -77, -77]);
        }
    }
}

#[test]
fn shared_calls_preserve_live_registers_and_spilled_arguments() {
    let mut leaf = Function::new(2, vec![]);
    let b = leaf.entry();
    let [x, y] = leaf.params(b).try_into().unwrap();
    let v = leaf.append(b, H::Bin(Binary::Sub, x, y)).unwrap();
    leaf.set_terminator(b, Terminator::Return(v));
    let shared = Arc::new(leaf);
    let mut f = Function::new(6, vec![]);
    let b = f.entry();
    let a = f.params(b).to_vec();
    let call = f.add_call(shared.clone(), &[a[4], a[5]], &[]);
    let first = f.append(b, H::Call(call)).unwrap();
    let call = f.add_call(shared, &[first, a[3]], &[]);
    let mut v = f.append(b, H::Call(call)).unwrap();
    for arg in a {
        v = f.append(b, H::Bin(Binary::Add, v, arg)).unwrap();
    }
    f.set_terminator(b, Terminator::Return(v));
    assert_eq!(Graph::lower(&f).unwrap().functions.len(), 2);
    let k = ScalarKernel::compile(&f).unwrap();
    for input in [[1, 2, 3, 4, 5, 6], [-1, -2, 5, i32::MIN, i32::MAX, 0]] {
        assert_eq!(k.call(&input), f.interpret(&input));
    }
}

#[test]
fn explicit_instructions_survive_lowering_and_allocation() {
    let mut f = Function::new(0, vec![]);
    let b = f.entry();
    let two = f.append(b, H::Const(2)).unwrap();
    let four = f.append(b, H::Bin(Binary::Add, two, two)).unwrap();
    f.append(b, H::Bin(Binary::Mul, four, two));
    f.set_terminator(b, Terminator::Return(four));
    let graph = Graph::lower(&f).unwrap();
    assert!(matches!(
        &graph.functions[0].ops[..],
        [
            Op::Imm(_, 2),
            Op::Alu(_, Binary::Add, ..),
            Op::Alu(_, Binary::Mul, ..)
        ]
    ));
    assert_eq!(ScalarKernel::compile(&f).unwrap().call(&[]), Ok(4));
}

#[test]
fn conditional_backedge_rotates_block_arguments_in_parallel() {
    use crate::hir::Op;
    let mut f = Function::new(3, vec![]);
    let entry = f.entry();
    let args = f.params(entry).to_vec();
    let header = f.create_block();
    let result = f.create_block();
    let first = f.append_param(header);
    let second = f.append_param(header);
    let count = f.append_param(header);
    let x = f.append_param(result);
    let y = f.append_param(result);
    let start = f.add_edge(header, &args);
    f.set_terminator(entry, Terminator::Jump(start));
    let one = f.append(header, Op::Const(1)).unwrap();
    let next = f.append(header, Op::Bin(Binary::Sub, count, one)).unwrap();
    let again = f.add_edge(header, &[second, first, next]);
    let done = f.add_edge(result, &[first, second]);
    f.set_terminator(
        header,
        Terminator::Branch {
            cond: count,
            yes: again,
            no: done,
        },
    );
    let diff = f.append(result, Op::Bin(Binary::Sub, x, y)).unwrap();
    f.set_terminator(result, Terminator::Return(diff));

    let kernel = ScalarKernel::compile(&f).unwrap();
    for n in 0..20 {
        assert_eq!(
            kernel.call(&[-3, 8, n]),
            Ok(if n % 2 == 0 { -11 } else { 11 })
        );
    }
}
