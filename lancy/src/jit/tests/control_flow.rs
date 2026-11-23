use crate::ScalarKernel;
use crate::hir::{Binary, Function, Op, Terminator};

#[test]
fn conditional_backedge_rotates_block_arguments_in_parallel() {
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

#[test]
fn fused_signed_branches_choose_the_correct_edge_in_either_layout() {
    use Binary::*;
    for op in [Eq, Ne, Lt, Le, Gt, Ge] {
        for direct_on_true in [false, true] {
            for immediate in [false, true] {
                let mut f = Function::scalar(2);
                let entry = f.entry();
                let [x, y] = f.params(entry).try_into().unwrap();
                let rhs = if immediate {
                    f.append(entry, Op::Const(-129)).unwrap()
                } else {
                    y
                };
                let cond = f.append(entry, Op::Bin(op, x, rhs)).unwrap();
                let detour = f.create_block();
                let join = f.create_block();
                let result = f.append_param(join);
                let direct_value = f.append(entry, Op::Const(19)).unwrap();
                let other_value = f.append(detour, Op::Const(73)).unwrap();
                let direct = f.add_edge(join, &[direct_value]);
                let other = f.add_edge(detour, &[]);
                let (yes, no) = if direct_on_true {
                    (direct, other)
                } else {
                    (other, direct)
                };
                f.set_terminator(entry, Terminator::Branch { cond, yes, no });
                f.jump(detour, join, &[other_value]);
                f.set_terminator(join, Terminator::Return(result));
                let kernel = ScalarKernel::compile(&f).unwrap();
                for x in [i32::MIN, -130, -129, -128, 0, i32::MAX] {
                    for input_y in [i32::MIN, -129, 0, i32::MAX] {
                        let y = if immediate { -129 } else { input_y };
                        let taken = match op {
                            Eq => x == y,
                            Ne => x != y,
                            Lt => x < y,
                            Le => x <= y,
                            Gt => x > y,
                            Ge => x >= y,
                            _ => unreachable!(),
                        };
                        assert_eq!(
                            kernel.call(&[x, input_y]),
                            Ok(if taken == direct_on_true { 19 } else { 73 }),
                            "{x} {op:?} {y}, direct_on_true={direct_on_true}, immediate={immediate}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn critical_backedges_permute_spilled_block_parameters() {
    const LANES: usize = 24;
    let mut f = Function::scalar(LANES + 2);
    let entry = f.entry();
    let args = f.params(entry).to_vec();
    let selector = args[0];
    let head = f.create_block();
    let shuffle = f.create_block();
    let exit = f.create_block();
    let count = f.append_param(head);
    let lanes: Vec<_> = (0..LANES).map(|_| f.append_param(head)).collect();
    f.jump(entry, head, &args[1..]);
    let yes = f.add_edge(shuffle, &[]);
    let no = f.add_edge(exit, &[]);
    f.set_terminator(
        head,
        Terminator::Branch {
            cond: count,
            yes,
            no,
        },
    );

    let one = f.append(shuffle, Op::Const(1)).unwrap();
    let next = f.append(shuffle, Op::Bin(Binary::Sub, count, one)).unwrap();
    let cond = f
        .append(shuffle, Op::Bin(Binary::And, count, selector))
        .unwrap();
    let mut rotated = lanes.clone();
    rotated.rotate_left(1);
    let yes = f.add_edge(head, &[&[next][..], &rotated].concat());
    let mut reversed = lanes.clone();
    reversed.reverse();
    let no = f.add_edge(head, &[&[next][..], &reversed].concat());
    f.set_terminator(shuffle, Terminator::Branch { cond, yes, no });

    let multiplier = f.append(exit, Op::Const(31)).unwrap();
    let mut result = f.append(exit, Op::Const(0)).unwrap();
    for lane in lanes {
        let scaled = f
            .append(exit, Op::Bin(Binary::Mul, result, multiplier))
            .unwrap();
        result = f.append(exit, Op::Bin(Binary::Add, scaled, lane)).unwrap();
    }
    f.set_terminator(exit, Terminator::Return(result));
    let kernel = ScalarKernel::compile(&f).unwrap();
    for selector in [0, 1, 6, -1] {
        for count in [0, 1, 2, 5, 24, 25] {
            let mut lanes: [i32; LANES] = std::array::from_fn(|i| 17 * i as i32 - 83);
            let args = [&[selector, count][..], &lanes].concat();
            for n in (1..=count).rev() {
                if n & selector != 0 {
                    lanes.rotate_left(1)
                } else {
                    lanes.reverse()
                }
            }
            let expected = lanes
                .into_iter()
                .fold(0i32, |acc, x| acc.wrapping_mul(31).wrapping_add(x));
            assert_eq!(
                kernel.call(&args),
                Ok(expected),
                "selector={selector}, count={count}"
            );
        }
    }
}
