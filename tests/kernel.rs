use lancy_jit::{AdaptiveKernel, NativeKernel, Op, kernel};
use std::cell::Cell;

enum Plan {
    Column(bool),
    Literal(i32),
    Sum(Box<Self>, Box<Self>),
    Product(Box<Self>, Box<Self>),
}

#[kernel(batch)]
fn evaluate(x: i32, #[comptime] plan: &Plan, y: i32) -> i32 {
    match plan {
        Plan::Column(first) if *first => x,
        Plan::Column(_) => y,
        Plan::Literal(n) => *n,
        Plan::Sum(a, b) => evaluate(x, a, y) + evaluate(x, b, y),
        Plan::Product(a, b) => evaluate(x, a, y) * evaluate(x, b, y),
    }
}

#[test]
fn enum_expansion_emits_only_selected_operations_before_optimization() {
    use Plan::*;
    let p = evaluate(&Literal(42)).unwrap();
    assert_eq!(p.input_count(), 2);
    assert_eq!(p.ops(), &[Op::Const(42)]);
    let p = evaluate(&Sum(Box::new(Column(false)), Box::new(Literal(8)))).unwrap();
    assert!(matches!(p.ops(), [Op::Input(1), Op::Const(8), Op::Add(..)]));
    let p = evaluate(&Product(Box::new(Column(true)), Box::new(Column(true)))).unwrap();
    assert!(matches!(p.ops(), [Op::Input(0), Op::Mul(a,b)] if a == b));
}

#[test]
fn recursive_sql_tree_executes_with_both_native_tiers() {
    use Plan::*;
    let sum = Sum(Box::new(Column(true)), Box::new(Literal(12)));
    let p = evaluate(&Product(Box::new(sum), Box::new(Column(false)))).unwrap();
    let x = [i32::MAX, -10, 0, 1, 9];
    let y = [5, -2, 9, 2, i32::MAX];
    let expected = std::array::from_fn::<_, 5, _>(|i| x[i].wrapping_add(12).wrapping_mul(y[i]));
    for code in [
        NativeKernel::baseline(&p).unwrap(),
        NativeKernel::specialize(&p, &[]).unwrap(),
    ] {
        let mut out = [0; 5];
        assert_eq!(code.run(&[&x, &y], &mut out).unwrap(), out.len());
        assert_eq!(out, expected);
    }
}

enum Choice {
    Adjust { x: i32 },
    Clip,
    Identity,
}

fn count(calls: &Cell<usize>, n: i32) -> i32 {
    calls.set(calls.get() + 1);
    n
}

#[kernel(batch)]
fn choose(#[comptime] mode: Choice, x: i32, #[comptime] calls: &Cell<usize>, y: i32) -> i32 {
    let original = x;
    match mode {
        // The pattern shadows the runtime column with an ordinary host i32.
        Choice::Adjust { x } if x > 0 => original + count(calls, x * 2 + 1),
        Choice::Adjust { .. } | Choice::Identity => original,
        Choice::Clip => {
            let positive: bool = (original >= 0) && (y != 0);
            if positive { original * y } else { -y }
        }
    }
}

#[test]
fn host_guards_and_shadowing_stay_out_of_hir() {
    let calls = Cell::new(0);
    let p = choose(Choice::Adjust { x: 3 }, &calls).unwrap();
    assert!(matches!(p.ops(), [Op::Input(0), Op::Const(7), Op::Add(..)]));
    assert_eq!(calls.get(), 1);
    let jit = AdaptiveKernel::compile(p).unwrap();
    let mut out = [0; 5];
    for _ in 0..2 {
        jit.run(&[&[1, 2, 3, 4, 5], &[0; 5]], &mut out).unwrap();
        jit.wait_for_optimization().unwrap();
    }
    assert_eq!(out, [8, 9, 10, 11, 12]);
    assert_eq!(calls.get(), 1);
    for mode in [Choice::Identity, Choice::Adjust { x: -3 }] {
        assert_eq!(choose(mode, &calls).unwrap().ops(), &[Op::Input(0)]);
    }
    assert_eq!(calls.get(), 1);
}

#[test]
fn row_if_emits_select_and_observes_each_row() {
    let p = choose(Choice::Clip, &Cell::new(0)).unwrap();
    assert!(matches!(p.ops().last(), Some(Op::Select(..))));
    let jit = NativeKernel::specialize(&p, &[]).unwrap();
    let mut out = [0; 5];
    jit.run(&[&[-2, 0, 2, 3, 5], &[3, 7, -3, 0, 9]], &mut out)
        .unwrap();
    assert_eq!(out, [-3, 0, -6, 0, 45]);
}

#[kernel(batch)]
fn host_if(#[comptime] flag: bool, #[comptime] n: i32, x: i32) -> i32 {
    let k = if flag { n * 2 } else { n + 3 };
    if k > 10 { x - k } else { k }
}

#[test]
fn host_if_folds_before_emitting_hir() {
    assert_eq!(host_if(false, 2).unwrap().ops(), &[Op::Const(5)]);
    assert!(matches!(
        host_if(true, 8).unwrap().ops(),
        [Op::Input(0), Op::Const(16), Op::Sub(..)]
    ));
}
