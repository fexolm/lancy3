use super::{ControlFlowGraph, DomTree};
use crate::hir::ssa::{Block, Function, Terminator};

#[test]
fn loop_with_two_arms_has_one_header_dominating_the_exit() {
    let mut f = Function::scalar(1);
    let entry = f.entry();
    let cond = f.params(entry)[0];
    let retired = f.create_block();
    let header = f.create_block();
    let left = f.create_block();
    let right = f.create_block();
    let join = f.create_block();
    let exit = f.create_block();

    for (from, to) in [(entry, header), (left, join), (right, join)] {
        let edge = f.add_edge(to, &[]);
        f.set_terminator(from, Terminator::Jump(edge));
    }
    for (from, a, b) in [(header, left, right), (join, header, exit)] {
        let yes = f.add_edge(a, &[]);
        let no = f.add_edge(b, &[]);
        f.set_terminator(from, Terminator::Branch { cond, yes, no });
    }
    f.set_terminator(exit, Terminator::Return(cond));
    assert_eq!(f.prune_unreachable(), Ok(1));

    let cfg = ControlFlowGraph::compute(&f).unwrap();
    assert!(cfg.successors(header).eq([left, right]));
    assert_eq!(cfg.predecessors(header), &[entry, join]);
    assert_eq!(cfg.predecessors(join), &[left, right]);
    assert_eq!(cfg.reverse_postorder().first(), Some(&entry));
    assert_eq!(cfg.reverse_postorder().len(), 6);
    let tree = DomTree::compute(&cfg);
    assert!(tree.dominates(header, header));
    assert!(tree.dominates(header, exit));
    assert!(tree.dominates(join, exit));
    assert!(!tree.dominates(left, join));
    assert!(!tree.dominates(join, header));
    assert!(!tree.dominates(entry, retired));
    assert!(!tree.dominates(retired, entry));
    assert!(!tree.dominates(entry, Block(u32::MAX)));
}
