use super::*;

fn jump(f: &mut Function, from: Block, to: Block, args: &[Value]) {
    let edge = f.add_edge(to, args);
    f.set_terminator(from, Terminator::Jump(edge));
}

#[test]
fn rejects_sibling_uses_and_bad_edge_arity() {
    let mut f = Function::new(1, vec![]);
    let first = f.entry();
    let cond = f.params(first)[0];
    let then = f.create_block();
    let other = f.create_block();
    let join = f.create_block();
    let result = f.append_param(join);
    let yes = f.add_edge(then, &[]);
    let no = f.add_edge(other, &[]);
    f.set_terminator(first, Terminator::Branch { cond, yes, no });
    let then_value = f.append(then, Op::Const(3)).unwrap();
    jump(&mut f, then, join, &[then_value]);
    jump(&mut f, other, join, &[then_value]);
    f.set_terminator(join, Terminator::Return(result));
    assert!(f.verify().unwrap_err().contains("dominate"));
    jump(&mut f, other, join, &[]);
    assert!(f.verify().unwrap_err().contains("arity"));
    jump(&mut f, other, join, &[cond]);
    f.verify().unwrap();
    assert!(f.merge_blocks(then, join).is_err());
    f.verify().unwrap();
}

#[test]
fn batch_replacement_keeps_ids_and_rolls_back_failed_edits() {
    let mut f = Function::new(0, vec![]);
    let block = f.entry();
    let one = f.append_inst(block, Op::Const(1));
    let value = f.result(one).unwrap();
    let two = f.append_inst(block, Op::Bin(Binary::Add, value, value));
    let result = f.result(two).unwrap();
    f.set_terminator(block, Terminator::Return(result));
    f.replace_op(two, Op::Bin(Binary::Div, value, value))
        .unwrap();
    assert!(f.needs_status());
    assert!(
        f.replace_ops(&[(two, Op::Const(6)), (one, Op::Copy(result))])
            .is_err()
    );
    assert_eq!(f.result(two), Some(result));
    assert_eq!(*f.op(two), Op::Bin(Binary::Div, value, value));
    assert!(f.needs_status());
    f.replace_ops(&[(two, Op::Const(6)), (two, Op::Const(7))])
        .unwrap();
    assert!(!f.needs_status());
    assert_eq!(*f.op(two), Op::Const(7));
    assert_eq!(f.result(one), Some(value));
    assert_eq!(
        crate::ScalarKernel::baseline_hir(&f).unwrap().call(&[]),
        Ok(7)
    );
}

#[test]
fn merging_rewrites_dominated_call_and_loop_arguments() {
    let mut helper = Function::new(1, vec![]);
    helper.set_terminator(
        helper.entry(),
        Terminator::Return(helper.params(helper.entry())[0]),
    );
    let mut f = Function::new(2, vec![]);
    let entry = f.entry();
    let (a, b) = (f.params(entry)[0], f.params(entry)[1]);
    let head = f.create_block();
    let x = f.append_param(head);
    let y = f.append_param(head);
    let latch = f.create_block();
    let exit = f.create_block();
    let done = f.append_param(exit);
    jump(&mut f, entry, head, &[b, a]);
    jump(&mut f, head, latch, &[]);
    let call = f.add_call(Arc::new(helper), &[y], &[]);
    let call_inst = f.append_inst(latch, Op::Call(call));
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
    let kernel = crate::ScalarKernel::compile_hir(&f).unwrap();
    assert_eq!(kernel.call(&[0, 17]), Ok(17));
}

#[test]
fn merging_splices_nonempty_instruction_lists_without_reindexing() {
    let mut f = Function::new(0, vec![]);
    let entry = f.entry();
    let succ = f.create_block();
    let def = f.append_inst(entry, Op::Const(8));
    let val = f.result(def).unwrap();
    let param = f.append_param(succ);
    jump(&mut f, entry, succ, &[val]);
    let add = f.append_inst(succ, Op::Bin(Binary::Add, param, val));
    f.set_terminator(succ, Terminator::Return(f.result(add).unwrap()));
    f.merge_blocks(entry, succ).unwrap();
    assert_eq!(f.blocks().count(), 1);
    assert_eq!(f.instructions(entry).collect::<Vec<_>>(), vec![def, add]);
    assert_eq!(*f.op(add), Op::Bin(Binary::Add, val, val));
    f.verify().unwrap();
}

#[test]
fn linear_chain_merge_resolves_transitive_parameter_aliases() {
    let mut f = Function::new(1, vec![]);
    let entry = f.entry();
    let arg = f.params(entry)[0];
    let tail = f.create_block();
    let head = f.create_block();
    let middle = f.create_block();
    let a = f.append_param(head);
    let b = f.append_param(middle);
    let c = f.append_param(tail);
    jump(&mut f, entry, head, &[arg]);
    jump(&mut f, head, middle, &[a]);
    jump(&mut f, middle, tail, &[b]);
    f.set_terminator(tail, Terminator::Return(c));
    assert_eq!(f.merge_linear_blocks().unwrap(), 3);
    f.verify().unwrap();
    assert_eq!(f.blocks().collect::<Vec<_>>(), [entry]);
    assert_eq!(*f.terminator(entry), Terminator::Return(arg));
}

#[test]
fn validates_call_permissions_and_rejects_entry_backedges() {
    let mut callee = Function::new(1, vec![BufferKind::Write]);
    let arg = callee.params(callee.entry())[0];
    callee.set_terminator(callee.entry(), Terminator::Return(arg));
    let mut f = Function::new(1, vec![BufferKind::Read]);
    let entry = f.entry();
    let arg = f.params(entry)[0];
    let call = f.add_call(Arc::new(callee), &[arg], &[Buffer(0)]);
    let res = f.append(entry, Op::Call(call)).unwrap();
    f.set_terminator(entry, Terminator::Return(res));
    assert!(f.verify().unwrap_err().contains("read-only"));
    jump(&mut f, entry, entry, &[arg]);
    assert!(f.verify().unwrap_err().contains("entry"));
}
