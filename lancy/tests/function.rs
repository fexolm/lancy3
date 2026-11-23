use lancy::hir::{Binary, Block, Function, IrError, Op, Terminator};
use lancy::{CallError, CompileError, ScalarKernel};
use std::sync::Arc;

#[test]
fn verify_rejects_bad_edges_dominance_call_signatures_and_unterminated_blocks() {
    let mut f = Function::scalar(1);
    let missing = f.add_edge(Block(99), &[]);
    f.set_terminator(f.entry(), Terminator::Jump(missing));
    assert_eq!(
        f.verify(),
        Err(IrError::UnknownTarget {
            from: f.entry(),
            edge: missing,
            target: Block(99)
        })
    );

    let mut f = Function::scalar(1);
    let entry = f.entry();
    let cond = f.params(entry)[0];
    let arm = f.create_block();
    let join = f.create_block();
    let yes = f.add_edge(arm, &[]);
    let no = f.add_edge(join, &[]);
    f.set_terminator(entry, Terminator::Branch { cond, yes, no });
    let only_in_arm = f.append(arm, Op::Const(1)).unwrap();
    let merge = f.add_edge(join, &[]);
    f.set_terminator(arm, Terminator::Jump(merge));
    f.set_terminator(join, Terminator::Return(only_in_arm));
    assert_eq!(
        f.verify(),
        Err(IrError::DoesNotDominate {
            value: only_in_arm,
            definition: arm,
            block: join
        })
    );

    let mut identity = Function::scalar(1);
    let entry = identity.entry();
    identity.set_terminator(entry, Terminator::Return(identity.params(entry)[0]));
    let mut f = Function::scalar(0);
    let call = f.add_call(Arc::new(identity), &[]);
    let output = f.append(f.entry(), Op::Call(call)).unwrap();
    f.set_terminator(f.entry(), Terminator::Return(output));
    assert_eq!(
        f.verify(),
        Err(IrError::CallArity {
            call,
            expected: 1,
            actual: 0
        })
    );

    let f = Function::scalar(0);
    let Err(CompileError::InvalidIr(error)) = ScalarKernel::compile(&f) else {
        panic!("expected an SSA error for an empty function");
    };
    assert_eq!(error, IrError::UnterminatedBlock { block: f.entry() });
}

#[test]
fn compile_preserves_the_verifier_error_from_a_callee() {
    use std::error::Error;

    let child = Arc::new(Function::scalar(0));
    let mut parent = Function::scalar(0);
    let site = parent.add_call(child.clone(), &[]);
    let returned = parent.append(parent.entry(), Op::Call(site)).unwrap();
    parent.set_terminator(parent.entry(), Terminator::Return(returned));
    parent.verify().unwrap();

    let Err(err) = ScalarKernel::compile(&parent) else {
        panic!("compiled an unterminated callee");
    };
    assert!(matches!(
        err.source().and_then(|source| source.downcast_ref::<IrError>()),
        Some(IrError::UnterminatedBlock { block }) if *block == child.entry()
    ));
}

#[test]
fn a_compiled_division_reports_each_arithmetic_trap() {
    let mut f = Function::scalar(2);
    let entry = f.entry();
    let [lhs, rhs] = f.params(entry).try_into().unwrap();
    let result = f.append(entry, Op::Bin(Binary::Div, lhs, rhs)).unwrap();
    f.set_terminator(entry, Terminator::Return(result));

    let kernel = ScalarKernel::compile(&f).unwrap();
    assert_eq!(kernel.call(&[7, 0]), Err(CallError::DivisionByZero));
    assert_eq!(
        kernel.call(&[i32::MIN, -1]),
        Err(CallError::DivisionOverflow)
    );
    assert_eq!(kernel.call(&[15, 3]), Ok(5));
}

#[test]
fn calls_and_joins_preserve_noncanonical_branch_masks() {
    let mut child = Function::scalar(2);
    let entry = child.entry();
    let (mask, x) = (child.params(entry)[0], child.params(entry)[1]);
    let increment = child.create_block();
    let identity = child.create_block();
    let yes = child.add_edge(increment, &[]);
    let no = child.add_edge(identity, &[]);
    child.set_terminator(
        entry,
        Terminator::Branch {
            cond: mask,
            yes,
            no,
        },
    );
    let one = child.append(increment, Op::Const(1)).unwrap();
    let next = child
        .append(increment, Op::Bin(Binary::Add, x, one))
        .unwrap();
    child.set_terminator(increment, Terminator::Return(next));
    child.set_terminator(identity, Terminator::Return(mask));

    let mut f = Function::scalar(2);
    let entry = f.entry();
    let (mask, x) = (f.params(entry)[0], f.params(entry)[1]);
    let call = f.add_call(Arc::new(child), &[mask, x]);
    let result = f.append(entry, Op::Call(call)).unwrap();
    let zero = f.append(entry, Op::Const(0)).unwrap();
    let cond = f.append(entry, Op::Bin(Binary::Lt, result, zero)).unwrap();
    let join = f.create_block();
    let output = f.append_param(join);
    let yes = f.add_edge(join, &[x]);
    let no = f.add_edge(join, &[result]);
    f.set_terminator(entry, Terminator::Branch { cond, yes, no });
    f.set_terminator(join, Terminator::Return(output));

    let kernel = ScalarKernel::compile(&f).unwrap();
    for (mask, x) in [
        (1, i32::MIN),
        (-2, i32::MAX),
        (0, 10),
        (-1, -20),
        (0x01010101, 99),
        (1, 0),
        (2, 3),
    ] {
        let result = if mask != 0 { x.wrapping_add(1) } else { 0 };
        let expected = if result < 0 { x } else { result };

        assert_eq!(kernel.call(&[mask, x]).unwrap(), expected);
    }
    assert_eq!(
        kernel.call(&[1]),
        Err(CallError::Arity {
            expected: 2,
            actual: 1
        })
    );
}

#[test]
fn loop_carried_parameters_handle_zero_trip_inputs() {
    let mut f = Function::scalar(1);
    let entry = f.entry();
    let arg = f.params(entry)[0];
    let head = f.create_block();
    let body = f.create_block();
    let exit = f.create_block();
    let n = f.append_param(head);
    let total = f.append_param(head);
    let zero = f.append(entry, Op::Const(0)).unwrap();
    let one = f.append(entry, Op::Const(1)).unwrap();
    let start = f.add_edge(head, &[arg, zero]);
    f.set_terminator(entry, Terminator::Jump(start));

    let cond = f.append(head, Op::Bin(Binary::Gt, n, zero)).unwrap();
    let yes = f.add_edge(body, &[]);
    let no = f.add_edge(exit, &[]);
    f.set_terminator(head, Terminator::Branch { cond, yes, no });
    let sum = f.append(body, Op::Bin(Binary::Add, total, n)).unwrap();
    let decreased = f.append(body, Op::Bin(Binary::Sub, n, one)).unwrap();
    let back = f.add_edge(head, &[decreased, sum]);
    f.set_terminator(body, Terminator::Jump(back));
    f.set_terminator(exit, Terminator::Return(total));

    let kernel = ScalarKernel::compile(&f).unwrap();
    for (n, sum) in [(0, 0), (4, 10), (1, 1), (10, 55), (-1, 0)] {
        assert_eq!(kernel.call(&[n]).unwrap(), sum);
    }
}

#[test]
fn jit_agrees_with_rust_on_forward_edges_and_shared_returns() {
    #[derive(Clone, Copy)]
    enum Update {
        Constant(i32),
        AddBit,
        UseFallback,
    }
    let mut seed = 734_851u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed >> 8
    };
    for _ in 0..64 {
        let mut f = Function::scalar(3);
        let entry = f.entry();
        let (flags, initial, fallback) =
            (f.params(entry)[0], f.params(entry)[1], f.params(entry)[2]);
        let blocks: Vec<_> = (0..9)
            .map(|_| {
                let block = f.create_block();
                let value = f.append_param(block);
                (block, value)
            })
            .collect();
        let start = f.add_edge(blocks[0].0, &[initial]);
        f.set_terminator(entry, Terminator::Jump(start));
        let mut plan = Vec::new();
        for (i, &(block, value)) in blocks.iter().enumerate() {
            let successors = &blocks[i + 1..];
            if successors.is_empty() {
                f.set_terminator(block, Terminator::Return(value));
                break;
            }
            let other = i + 1 + (next() as usize % successors.len());
            let bit_value = 1 << (next() & 7);
            let bit = f.append(block, Op::Const(bit_value)).unwrap();
            let cond = f.append(block, Op::Bin(Binary::And, flags, bit)).unwrap();
            let update = match next() % 3 {
                0 => Update::Constant(next() as i32),
                1 => Update::AddBit,
                _ => Update::UseFallback,
            };
            plan.push((bit_value, other, update));
            let result = match update {
                Update::Constant(n) => f.append(block, Op::Const(n)).unwrap(),
                Update::AddBit => f.append(block, Op::Bin(Binary::Add, value, bit)).unwrap(),
                Update::UseFallback => fallback,
            };
            let yes = f.add_edge(successors[0].0, &[result]);
            let no = f.add_edge(blocks[other].0, &[value]);
            f.set_terminator(block, Terminator::Branch { cond, yes, no });
        }
        let kernel = ScalarKernel::compile(&f).unwrap();
        for mask in [0, 1, 7, 0x55, 0xaa, -1] {
            let args = [mask, -207, 42];
            let mut value: i32 = args[1];
            let mut pc = 0;
            while let Some(&(bit, other, update)) = plan.get(pc) {
                if mask & bit != 0 {
                    value = match update {
                        Update::Constant(n) => n,
                        Update::AddBit => value.wrapping_add(bit),
                        Update::UseFallback => args[2],
                    };
                    pc += 1;
                } else {
                    pc = other;
                }
            }
            assert_eq!(kernel.call(&args).unwrap(), value);
        }
    }
}

#[test]
fn release_code_drops_debug_points() {
    let build = |points: u32| {
        let mut f = Function::scalar(1);
        let entry = f.entry();
        let mut value = f.params(entry)[0];
        for id in 0..points {
            f.append(entry, Op::DebugPoint(id));
        }
        let three = f.append(entry, Op::Const(3)).unwrap();
        value = f.append(entry, Op::Bin(Binary::Mul, value, three)).unwrap();
        for id in 0..points {
            f.append(entry, Op::DebugPoint(points + id));
        }
        f.set_terminator(entry, Terminator::Return(value));
        ScalarKernel::compile(&f).unwrap()
    };
    let (plain, marked) = (build(0), build(32));
    assert_eq!(marked.call(&[7]), Ok(21));
    assert_eq!(marked.code_size(), plain.code_size());
}
