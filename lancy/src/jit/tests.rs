use super::{ScalarKernel, passes};
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
