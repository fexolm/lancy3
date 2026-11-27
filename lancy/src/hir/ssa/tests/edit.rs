use crate::hir::IrError;
use crate::hir::ssa::{Binary, Function, Op, Terminator};
use std::sync::Arc;

#[test]
fn replacing_a_call_changes_the_next_modules_reachable_functions() {
    let mut leaf = Function::scalar(1);
    let b = leaf.entry();
    leaf.set_terminator(b, Terminator::Return(leaf.params(b)[0]));
    let mut root = Function::scalar(1);
    let b = root.entry();
    let x = root.params(b)[0];
    let site = root.add_call(Arc::new(leaf), &[x]);
    let inst = root.append_inst(b, Op::Call(site)).unwrap();
    root.set_terminator(b, Terminator::Return(root.result(inst).unwrap()));

    {
        let before = root.verified_module().unwrap();
        assert_eq!(before.functions.len(), 2);
        assert_eq!(before.call_graph.callees(before.call_graph.root()).len(), 1);
    }
    root.replace_op(inst, Op::Copy(x)).unwrap();
    let after = root.verified_module().unwrap();
    assert_eq!(after.functions.len(), 1);
    assert!(after.call_graph.callees(after.call_graph.root()).is_empty());
    assert_eq!(
        crate::Kernel::compile(&root, &crate::test_target::options())
            .unwrap()
            .call(&[17]),
        Ok(17)
    );
}

#[test]
fn pruning_a_staging_block_retires_only_its_instructions() {
    let mut graph = Function::scalar(1);
    let entry = graph.entry();
    let arg = graph.params(entry)[0];
    graph.set_terminator(entry, Terminator::Return(arg));
    let dead = graph.create_block();
    let operation = graph
        .append_inst(dead, Op::Bin(Binary::Div, arg, arg))
        .unwrap();
    assert_eq!(graph.prune_unreachable(), Ok(1));
    assert_eq!(
        graph.replace_op(operation, Op::Const(7)),
        Err(IrError::RetiredInstruction { inst: operation })
    );
    assert_eq!(graph.interpret(&[91]), Ok(91));
    let new_block = graph.create_block();
    let new_inst = graph.append_inst(new_block, Op::Const(3)).unwrap();
    assert_ne!(new_block, dead);
    assert_ne!(new_inst, operation);
}

#[test]
fn batch_replacement_keeps_ids_and_rolls_back_failed_edits() {
    let mut f = Function::scalar(0);
    let block = f.entry();
    let one = f.append_inst(block, Op::Const(1)).unwrap();
    let value = f.result(one).unwrap();
    let two = f
        .append_inst(block, Op::Bin(Binary::Add, value, value))
        .unwrap();
    let result = f.result(two).unwrap();
    f.set_terminator(block, Terminator::Return(result));
    f.replace_op(two, Op::Bin(Binary::Div, value, value))
        .unwrap();
    assert!(
        f.replace_ops(&[(two, Op::Const(6)), (one, Op::Copy(result))])
            .is_err()
    );
    assert_eq!(f.result(two), Some(result));
    assert_eq!(*f.op(two), Op::Bin(Binary::Div, value, value));
    f.replace_ops(&[(two, Op::Const(6)), (two, Op::Const(7))])
        .unwrap();
    assert_eq!(*f.op(two), Op::Const(7));
    assert_eq!(f.result(one), Some(value));
    assert_eq!(
        crate::Kernel::compile(&f, &crate::test_target::options())
            .unwrap()
            .call(&[]),
        Ok(7)
    );
}

#[test]
fn merging_rewrites_dominated_call_and_loop_arguments() {
    let mut helper = Function::scalar(1);
    helper.set_terminator(
        helper.entry(),
        Terminator::Return(helper.params(helper.entry())[0]),
    );
    let mut f = Function::scalar(2);
    let entry = f.entry();
    let (a, b) = (f.params(entry)[0], f.params(entry)[1]);
    let head = f.create_block();
    let x = f.append_param(head);
    let y = f.append_param(head);
    let latch = f.create_block();
    let exit = f.create_block();
    let done = f.append_param(exit);
    f.jump(entry, head, &[b, a]);
    f.jump(head, latch, &[]);
    let call = f.add_call(Arc::new(helper), &[y]);
    let call_inst = f.append_inst(latch, Op::Call(call)).unwrap();
    let called = f.result(call_inst).unwrap();
    let yes = f.add_edge(latch, &[]);
    let no = f.add_edge(exit, &[x]);
    f.set_terminator(
        latch,
        Terminator::Branch {
            cond: called,
            yes,
            no,
        },
    );
    f.set_terminator(exit, Terminator::Return(done));
    f.merge_blocks(entry, head).unwrap();
    f.verify().unwrap();
    assert_eq!(f.call(call).1, &[a]);
    assert_eq!(f.edge(no), (exit, &[b][..]));
    assert_eq!(f.blocks().collect::<Vec<_>>(), vec![entry, latch, exit]);
    assert_eq!(f.instructions(latch).collect::<Vec<_>>(), vec![call_inst]);
    let kernel = crate::Kernel::compile(&f, &crate::test_target::options()).unwrap();
    assert_eq!(kernel.call(&[0, 17]), Ok(17));
}

#[test]
fn merging_splices_nonempty_instruction_lists_without_reindexing() {
    let mut f = Function::scalar(0);
    let entry = f.entry();
    let succ = f.create_block();
    let def = f.append_inst(entry, Op::Const(8)).unwrap();
    let val = f.result(def).unwrap();
    let param = f.append_param(succ);
    f.jump(entry, succ, &[val]);
    let add = f
        .append_inst(succ, Op::Bin(Binary::Add, param, val))
        .unwrap();
    f.set_terminator(succ, Terminator::Return(f.result(add).unwrap()));
    f.merge_blocks(entry, succ).unwrap();
    assert_eq!(f.blocks().count(), 1);
    assert_eq!(f.instructions(entry).collect::<Vec<_>>(), vec![def, add]);
    assert_eq!(*f.op(add), Op::Bin(Binary::Add, val, val));
    f.verify().unwrap();
}

/// Retargeting a jump leaves its old edge record behind; compaction drops
/// the record's arguments and keeps every ID valid.
#[test]
fn compaction_drops_the_arguments_of_retired_edges() {
    let mut f = Function::scalar(1);
    let entry = f.entry();
    let x = f.params(entry)[0];
    let join = f.create_block();
    let y = f.append_param(join);
    f.set_terminator(join, Terminator::Return(y));
    for _ in 0..4 {
        f.jump(entry, join, &[x]);
    }
    assert_eq!(f.operand_arena(), (4, 1));
    f.compact();
    assert_eq!(f.operand_arena(), (1, 1));
    f.verify().unwrap();
    assert_eq!(f.interpret(&[5]), Ok(5));
}

#[test]
fn slice_params_must_be_distinct_pointer_and_length_parameters() {
    use crate::hir::{SliceParam, Type};
    let mut f = Function::new(&[Type::Ptr, Type::I32, Type::Ptr, Type::I32], Type::I32);
    let [p, n, q, m] = f.params(f.entry()).try_into().unwrap();
    let slice = |ptr, len, element| SliceParam {
        ptr,
        len,
        element,
        writable: false,
    };
    f.set_slice_params(vec![slice(p, n, 4), slice(q, m, 8)])
        .unwrap();
    let zero = f.append(f.entry(), Op::Const(0)).unwrap();
    for bad in [
        slice(n, p, 4),
        slice(p, n, 1),
        slice(p, zero, 4),
        slice(q, n, 4),
    ] {
        let declared = vec![slice(p, n, 4), bad];
        assert_eq!(
            f.set_slice_params(declared),
            Err(IrError::InvalidSliceParam { slice: bad })
        );
    }
    assert_eq!(f.slice_param(q), Some(slice(q, m, 8)));
    assert_eq!(f.slice_param(n), None);
}
