use crate::ScalarKernel;
use crate::hir::{Binary, Function, Terminator};

#[test]
fn conditional_backedge_rotates_block_arguments_in_parallel() {
    use crate::hir::Op;
    let mut f = Function::scalar(3);
    let entry = f.entry();
    let args = f.params(entry).to_vec();
    // Leave a hole in the HIR block and value tables before the loop blocks.
    let unused = f.create_block();
    let dead = f.append(unused, Op::Const(42)).unwrap();
    f.set_terminator(unused, Terminator::Return(dead));
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
    assert_eq!(f.prune_unreachable().unwrap(), 1);

    let kernel = ScalarKernel::compile(&f).unwrap();
    for n in 0..20 {
        assert_eq!(
            kernel.call(&[-3, 8, n]),
            Ok(if n % 2 == 0 { -11 } else { 11 })
        );
    }
}
