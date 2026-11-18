use super::*;

#[test]
fn inverse_blends_and_repeated_xors_drop_redundant_masks() {
    let mut b = Builder::new(3);
    let mask = b.input(0);
    let x = b.input(1);
    let y = b.input(2);
    let xy = b.xor(x, y);
    let restored = b.xor(y, xy);
    let all = b.constant(-1);
    let inverse = b.xor(all, mask);
    let val = b.select(inverse, restored, y);
    let p = b.finish(val).unwrap();
    let opt = optimize(&p, &[]);
    assert_eq!(opt.ops().len(), 4);
    for mask in [0, -1, 0x5555_5555] {
        let columns: &[&[i32]] = &[&[mask], &[123], &[-456]];
        assert_eq!(p.eval_row(columns, 0), opt.eval_row(columns, 0));
    }
}

#[test]
fn select_with_one_constant_arm_simplifies_for_arbitrary_bitmasks() {
    for yes_is_all_ones in [false, true] {
        let mut b = Builder::new(2);
        let mask = b.input(0);
        let x = b.input(1);
        let result = if yes_is_all_ones {
            let ones = b.constant(-1);
            b.select(mask, ones, x)
        } else {
            let zero = b.constant(0);
            b.select(mask, x, zero)
        };
        let program = b.finish(result).unwrap();
        let simple = optimize(&program, &[]);
        assert!(matches!(
            simple.ops()[simple.result().index()],
            Op::And(..) | Op::Or(..)
        ));
        let cols: &[&[i32]] = &[&[-1, 0, 0x5555_5555], &[0x1234_5678; 3]];
        for row in 0..3 {
            assert_eq!(program.eval_row(cols, row), simple.eval_row(cols, row));
        }
    }
}

#[test]
fn packed_indices_and_builder_provenance() {
    assert_eq!(std::mem::size_of::<Value>(), 4);
    assert_eq!(std::mem::size_of::<Op>(), 16);
    let mut first = Builder::new(1);
    let foreign = first.input(0);
    let mut second = Builder::new(1);
    let local = second.input(0);
    second.add(local, foreign);
    assert_eq!(second.finish(local).unwrap_err(), BuildError::ForeignValue);
    let mut bad = Builder::new(1);
    bad.input(1);
    let ok = bad.constant(1);
    assert_eq!(bad.finish(ok).unwrap_err(), BuildError::InputOutOfRange(1));
    let empty = Builder::new(0);
    assert_eq!(empty.finish(foreign).unwrap_err(), BuildError::ForeignValue);
}

#[test]
fn folding_respects_wraparound_and_bit_masks() {
    let mut b = Builder::new(0);
    let max = b.constant(i32::MAX);
    let one = b.constant(1);
    let min = b.add(max, one);
    let yes = b.lt(min, max);
    let all_ones = b.eq(min, min);
    let mask = b.constant(0x5555_5555);
    let picked = b.select(mask, all_ones, min);
    let root = b.and(picked, yes);
    let program = b.finish(root).unwrap();
    let expected = 0xD555_5555u32 as i32;
    assert_eq!(program.eval_row(&[], 0), expected);
    assert_eq!(optimize(&program, &[]).ops(), &[Op::Const(expected)]);
}

#[test]
fn specialization_prunes_dead_subtrees_and_numbers_commutative_ops() {
    let mut b = Builder::new(3);
    let x = b.input(0);
    let y = b.input(1);
    let flag = b.input(2);
    let sum1 = b.add(x, y);
    let sum2 = b.add(y, x);
    let doubled = b.add(sum1, sum2);
    let dead = b.mul(doubled, y);
    let root = b.select(flag, doubled, dead);
    let base = b.finish(root).unwrap();
    let spec = optimize(
        &base,
        &[Assumption {
            column: 2,
            value: -1,
        }],
    );
    assert_eq!(spec.input_count(), 3);
    assert_eq!(spec.ops().len(), 4);
    assert_eq!(spec.ops()[3], Op::Add(Value(2), Value(2)));
    let columns: &[&[i32]] = &[&[4, -3], &[8, 20], &[-1, -1]];
    for row in 0..2 {
        assert_eq!(base.eval_row(columns, row), spec.eval_row(columns, row));
    }
}
