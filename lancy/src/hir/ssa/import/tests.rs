use super::*;
use crate::input::{Binary, FunctionBuilder};

#[test]
fn diamond_passes_only_the_needed_mutable_register() {
    let mut b = FunctionBuilder::new(1);
    let (left, right, join) = (b.label(), b.label(), b.label());
    let (x, dead) = (b.local(), b.local());
    b.branch(b.arg(0), left, right);
    b.mark(left);
    let one = b.constant(1);
    b.assign(x, one);
    b.assign(dead, one);
    b.jump(join);
    b.mark(right);
    let two = b.constant(2);
    b.assign(x, two);
    b.assign(dead, two);
    // Implicit fallthrough is an incoming edge too.
    b.mark(join);
    b.ret(x);
    let graph = from_input(&b.finish().unwrap()).unwrap();
    let blocks: Vec<_> = graph.blocks().collect();
    assert_eq!(blocks.len(), 4);
    let &Terminator::Return(result) = graph.terminator(blocks[3]) else {
        panic!()
    };
    assert_eq!(graph.params(blocks[3]), &[result]);
    for (arm, n) in [(blocks[1], 1), (blocks[2], 2)] {
        let &Terminator::Jump(edge) = graph.terminator(arm) else {
            panic!()
        };
        let inst = graph.instructions(arm).next().unwrap();
        assert_eq!(graph.op(inst), &Op::Const(n));
        assert_eq!(
            graph.edge(edge),
            (blocks[3], &[graph.result(inst).unwrap()][..])
        );
    }
}

#[test]
fn backedge_to_first_instruction_preserves_parallel_swap() {
    let mut b = FunctionBuilder::new(3);
    let (a, c, n) = (b.arg(0), b.arg(1), b.arg(2));
    let (head, body, exit) = (b.label(), b.label(), b.label());
    b.mark(head);
    b.branch(n, body, exit);
    b.mark(body);
    let previous = b.local();
    b.assign(previous, a);
    b.assign(a, c);
    b.assign(c, previous);
    let one = b.constant(1);
    let next = b.binary(Binary::Sub, n, one);
    b.assign(n, next);
    b.jump(head);
    b.mark(exit);
    b.ret(a);
    let graph = from_input(&b.finish().unwrap()).unwrap();
    let &Terminator::Jump(to_head) = graph.terminator(graph.entry()) else {
        panic!()
    };
    let (header, first) = graph.edge(to_head);
    assert_ne!(header, graph.entry());
    assert_eq!(first, graph.params(graph.entry()));
    let params = graph.params(header);
    assert_eq!(params.len(), 3);
    let &Terminator::Branch { cond, yes, no } = graph.terminator(header) else {
        panic!()
    };
    assert_eq!(cond, params[2]);
    let (body, _) = graph.edge(yes);
    let &Terminator::Jump(back) = graph.terminator(body) else {
        panic!()
    };
    let (target, args) = graph.edge(back);
    assert_eq!(target, header);
    assert_eq!(args[..2], [params[1], params[0]]);
    assert_ne!(args[2], params[2]);
    assert_eq!(
        graph.terminator(graph.edge(no).0),
        &Terminator::Return(params[0])
    );
    assert_eq!(graph.interpret(&[14, 82, 0]), Ok(14));
    assert_eq!(graph.interpret(&[14, 82, 1]), Ok(82));
    assert_eq!(graph.interpret(&[14, 82, 2]), Ok(14));
}

#[test]
fn partially_overlapping_loops_get_independent_entry_parameters() {
    let mut b = FunctionBuilder::new(3);
    let (a, c, exit) = (b.label(), b.label(), b.label());
    let (choose_entry, remaining, x) = (b.arg(0), b.arg(1), b.arg(2));
    let one = b.constant(1);
    b.branch(choose_entry, a, c);
    b.mark(a);
    let inc = b.binary(Binary::Add, x, one);
    b.assign(x, inc);
    let n = b.binary(Binary::Sub, remaining, one);
    b.assign(remaining, n);
    b.branch(n, c, exit);
    b.mark(c);
    let dec = b.binary(Binary::Sub, x, one);
    b.assign(x, dec);
    let n = b.binary(Binary::Sub, remaining, one);
    b.assign(remaining, n);
    b.branch(n, a, exit);
    b.mark(exit);
    b.ret(x);
    let input = b.finish().unwrap();
    let graph = from_input(&input).unwrap();
    let counts: Vec<_> = graph.blocks().map(|b| graph.params(b).len()).collect();
    assert_eq!(counts, [3, 2, 2, 1]);
    for entry in [0, 1] {
        for trips in 1..10 {
            let args = [entry, trips, i32::MAX];
            assert_eq!(graph.interpret(&args), input.interpret(&args));
        }
    }
}

#[test]
fn call_dag_remains_shared_and_unreachable_register_reads_are_ignored() {
    let mut helper = FunctionBuilder::new(1);
    helper.ret(helper.arg(0));
    let helper = Arc::new(helper.finish().unwrap());
    let mut b = FunctionBuilder::new(1);
    let first = b.call_shared(Arc::clone(&helper), &[b.arg(0)]);
    let second = b.call_shared(helper, &[first]);
    b.ret(second);
    let uninitialized = b.local();
    b.ret(uninitialized);
    let graph = from_input(&b.finish().unwrap()).unwrap();
    assert_eq!(graph.blocks().count(), 1);
    let calls: Vec<_> = graph
        .instructions(graph.entry())
        .map(|i| {
            let &Op::Call(call) = graph.op(i) else {
                panic!()
            };
            graph.call(call).0
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert!(Arc::ptr_eq(calls[0], calls[1]));
    assert_eq!(graph.interpret(&[-42]), Ok(-42));
}

#[test]
fn reference_execution_forwards_buffers_and_preserves_stores_before_a_trap() {
    use crate::input::{Buffer, BufferArg, BufferKind};
    let mut child = FunctionBuilder::with_buffers(1, vec![BufferKind::Write]);
    let zero = child.constant(0);
    child.store(Buffer(0), zero, child.arg(0));
    let read = child.load(Buffer(0), child.arg(0));
    child.ret(read);

    let mut parent = FunctionBuilder::with_buffers(1, vec![BufferKind::Read, BufferKind::Write]);
    let result = parent
        .call_with_buffers(child.finish().unwrap(), &[parent.arg(0)], &[Buffer(1)])
        .unwrap();
    parent.ret(result);
    let input = parent.finish().unwrap();
    let graph = from_input(&input).unwrap();
    for index in [-1, 0, 1, 2, 3] {
        let (mut expected, mut actual) = ([10, 20, 30], [10, 20, 30]);
        let legacy = input.interpret_with_buffers(
            &[index],
            &mut [BufferArg::Read(&[42]), BufferArg::Write(&mut expected)],
        );
        let result = graph.interpret_with_buffers(
            &[index],
            &mut [BufferArg::Read(&[42]), BufferArg::Write(&mut actual)],
        );
        assert_eq!(result, legacy);
        assert_eq!(actual, expected);
        assert_eq!(actual[0], index);
    }
}

#[test]
fn reference_execution_reports_both_division_errors() {
    for op in [Binary::Div, Binary::Rem] {
        let mut b = FunctionBuilder::new(2);
        let result = b.binary(op, b.arg(0), b.arg(1));
        b.ret(result);
        let input = b.finish().unwrap();
        let graph = from_input(&input).unwrap();
        for args in [[17, 0], [i32::MIN, -1], [-36, 7]] {
            assert_eq!(graph.interpret(&args), input.interpret(&args));
        }
    }
}
