//! JIT tests that hold on every supported architecture: they exercise
//! `ScalarKernel` against the interpreter, the evaluator or Rust.
mod control_flow;
mod generated;
mod integer_runtests;
pub(crate) mod numeric_coverage;
mod pointers;
mod traps;

use super::{Inlining, ScalarKernel, passes};
use crate::CompileError;
use crate::hir::{Binary, Function, Op, Terminator};
use std::sync::Arc;

#[test]
fn an_unrepresentable_expansion_keeps_the_shared_graph_executable() {
    let mut f = Function::scalar(1);
    let x = f.params(f.entry())[0];
    f.set_terminator(f.entry(), Terminator::Return(x));
    for _ in 0..32 {
        let child = Arc::new(f);
        f = Function::scalar(1);
        let b = f.entry();
        let x = f.params(b)[0];
        let twice = f.create_block();
        let skip = f.create_block();
        let yes = f.add_edge(twice, &[]);
        let no = f.add_edge(skip, &[]);
        f.set_terminator(b, Terminator::Branch { cond: x, yes, no });
        let one = f.append(b, Op::Const(1)).unwrap();
        let previous = f.append(twice, Op::Bin(Binary::Sub, x, one)).unwrap();
        let site = f.add_call(child, &[previous]);
        let a = f.append(twice, Op::Call(site)).unwrap();
        let c = f.append(twice, Op::Call(site)).unwrap();
        let sum = f.append(twice, Op::Bin(Binary::Add, a, c)).unwrap();
        f.set_terminator(twice, Terminator::Return(sum));
        f.set_terminator(skip, Terminator::Return(one));
    }
    let (m, ids) = f.verified_module().unwrap();
    assert!(matches!(
        passes::inline_all(&m, &ids),
        Err(crate::CompileError::FunctionTooLarge)
    ));
    let code = ScalarKernel::compile(&f).unwrap();
    assert_eq!(code.call(&[0]), Ok(1));
    assert_eq!(code.call(&[1]), Ok(2));
}

#[test]
fn every_size_limit_of_the_expanded_body_falls_back_to_calls() {
    let mut callee = Function::scalar(1);
    let x = callee.params(callee.entry())[0];
    callee.set_terminator(callee.entry(), Terminator::Return(x));
    let mut root = Function::scalar(1);
    let entry = root.entry();
    let x = root.params(entry)[0];
    let site = root.add_call(Arc::new(callee), &[x]);
    let result = root.append(entry, Op::Call(site)).unwrap();
    root.set_terminator(entry, Terminator::Return(result));
    let (module, ids) = root.verified_module().unwrap();
    let limits = [
        CompileError::FunctionTooLarge,
        CompileError::BranchOutOfRange {
            offset: 0,
            target: 0,
        },
        CompileError::WorkspaceTooLarge { slots: 0 },
        CompileError::FrameOverflow,
    ];
    for limit in limits {
        let mut limit = Some(limit);
        let lowered = super::optimize_and_lower(&module, &ids, Inlining::Always, |m, _| {
            // The first attempt is the inlined single-function body.
            match limit.take() {
                Some(error) => {
                    assert_eq!(m.functions.len(), 1);
                    Err(error)
                }
                None => Ok(m.functions.len()),
            }
        });
        assert_eq!(lowered.unwrap(), 2);
    }
}
