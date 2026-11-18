use crate::relational::*;
use lancy::BufferArg;
use splice::Codegen;
use splice::Executable;
use splice::hir::{Binary, BufferKind, Function, Instr};
use splice_lancy::{Lancy, LancyBaseline};
use splice_llvm::Llvm;

type Row = (i32, i32);

fn eval(expr: &Expr, (key, value): Row) -> i32 {
    let arg = |e| eval(e, (key, value));
    match expr {
        Expr::Key => key,
        Expr::Value => value,
        Expr::Literal(n) => *n,
        Expr::Add(a, b) => arg(a).wrapping_add(arg(b)),
        Expr::Mul(a, b) => arg(a).wrapping_mul(arg(b)),
        Expr::Min(a, b) => arg(a).min(arg(b)),
    }
}

fn keep(pred: &Predicate, row: Row) -> bool {
    match pred {
        Predicate::AtLeast(a, b) => eval(a, row) >= eval(b, row),
        Predicate::Equal(a, b) => eval(a, row) == eval(b, row),
        Predicate::And(a, b) => keep(a, row) && keep(b, row),
    }
}

// Deliberately use nested-loop joins with fresh vectors as the oracle.
fn reference(plan: &Plan, tables: &[&[Row]]) -> Vec<Row> {
    match plan {
        Plan::Scan(id) => tables[*id].to_vec(),
        Plan::Filter(child, pred) => reference(child, tables)
            .into_iter()
            .filter(|&r| keep(pred, r))
            .collect(),
        Plan::Project(child, k, v) => reference(child, tables)
            .into_iter()
            .map(|r| (eval(k, r), eval(v, r)))
            .collect(),
        Plan::Join(left, right, combine) => {
            let rhs = reference(right, tables);
            let mut result = Vec::new();
            for (k, v) in reference(left, tables) {
                for &(rk, rv) in &rhs {
                    if k == rk {
                        result.push((
                            k,
                            match combine {
                                Combine::Add => v.wrapping_add(rv),
                                Combine::Multiply => v.wrapping_mul(rv),
                            },
                        ));
                    }
                }
            }
            result
        }
    }
}

fn compare(plan: &Plan, tables: &[&[Row]], f: &Function, kernels: &[&dyn Executable]) {
    let cat = Catalog::new(tables).unwrap();
    let expected: Vec<i32> = reference(plan, tables)
        .into_iter()
        .flat_map(|(k, v)| [k, v])
        .collect();
    let mut scratch = vec![0x7fff; expected.len() * 8 + 512];
    for &code in kernels {
        assert_eq!(cat.run(code, &mut scratch).unwrap(), expected);
    }
    scratch.fill(-3);
    let mut interp = [
        BufferArg::Read(cat.offsets()),
        BufferArg::Read(cat.data()),
        BufferArg::Write(&mut scratch),
    ];
    assert_eq!(
        splice_lancy::lower(f)
            .unwrap()
            .interpret_with_buffers(&[], &mut interp)
            .unwrap() as usize
            * 2,
        expected.len()
    );
    assert_eq!(scratch[..expected.len()], expected);
    scratch.fill(-8);
    assert_eq!(
        query(plan, cat.offsets(), cat.data(), &mut scratch) as usize * 2,
        expected.len()
    );
    assert_eq!(scratch[..expected.len()], expected);
}

#[test]
fn branching_join_tree_runs_against_changing_tables() {
    let plan = pricing_plan();
    let f = query::ir(&plan).unwrap();
    let baseline = LancyBaseline::compile(&f).unwrap();
    let optimized = Lancy::compile(&f).unwrap();
    let llvm = Llvm::compile(&f).unwrap();
    let tables: &[&[Row]] = &[
        &[(-7, 5), (1, 3), (9, 6), (1, 5)],
        &[(1, 9), (-7, 3), (9, 4)],
        &[(9, 1), (1, 1), (-7, 1)],
        &[(1, 3), (9, 8)],
        &[(9, 4), (1, 3), (1, 8)],
    ];
    compare(&plan, tables, &f, &[&baseline, &optimized, &llvm]);
    for empty in 0..tables.len() {
        let mut changed = tables.to_vec();
        changed[empty] = &[];
        compare(&plan, &changed, &f, &[&baseline, &optimized, &llvm]);
    }
}

#[test]
fn self_join_after_rekey_handles_collisions_and_overlapping_compaction() {
    // Both sides scan the same table. Re-key the right before joining.
    let plan = Plan::Join(
        Box::new(Plan::Scan(0)),
        Box::new(Plan::Project(
            Box::new(Plan::Scan(0)),
            Expr::Value,
            Expr::Key,
        )),
        Combine::Multiply,
    );
    let f = query::ir(&plan).unwrap();
    let baseline = LancyBaseline::compile(&f).unwrap();
    let optimized = Lancy::compile(&f).unwrap();
    let llvm = Llvm::compile(&f).unwrap();
    let duplicate = [(3, 3); 13];
    compare(&plan, &[&duplicate], &f, &[&baseline, &optimized, &llvm]);
    let mixed = [
        (1, 9),
        (9, 1),
        (1, 1),
        (-2, -2),
        (17, 8),
        (i32::MIN, i32::MIN),
        (1, 9),
    ];
    compare(&plan, &[&mixed], &f, &[&baseline, &optimized, &llvm]);
}

#[test]
fn different_plan_shapes_match_a_nested_loop_oracle() {
    fn next(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed >> 16
    }
    fn tree(seed: &mut u32, depth: u32) -> Plan {
        if depth == 0 {
            return Plan::Scan((next(seed) % 3) as usize);
        }
        match next(seed) % 5 {
            0 => Plan::Scan((next(seed) % 3) as usize),
            1 => Plan::Filter(
                Box::new(tree(seed, depth - 1)),
                Predicate::AtLeast(Expr::Value, Expr::Literal((next(seed) % 5) as i32)),
            ),
            2 => Plan::Project(
                Box::new(tree(seed, depth - 1)),
                Expr::Key,
                Expr::Mul(Box::new(Expr::Value), Box::new(Expr::Literal(2))),
            ),
            _ => Plan::Join(
                Box::new(tree(seed, depth - 1)),
                Box::new(tree(seed, depth - 1)),
                if next(seed) & 1 == 0 {
                    Combine::Add
                } else {
                    Combine::Multiply
                },
            ),
        }
    }
    let mut seed = 3561;
    for _ in 0..24 {
        let plan = tree(&mut seed, 4);
        let f = query::ir(&plan).unwrap();
        let baseline = LancyBaseline::compile(&f).unwrap();
        let optimized = Lancy::compile(&f).unwrap();
        let llvm = Llvm::compile(&f).unwrap();
        compare(
            &plan,
            &[
                &[(0, 3), (-1, 6), (7, 2)],
                &[(7, 10), (-1, 9)],
                &[(-1, 1), (0, -1), (7, 3)],
            ],
            &f,
            &[&baseline, &optimized, &llvm],
        );
    }
}

#[test]
fn balanced_tree_with_32_scans_runs_through_one_entry_point() {
    fn subtree(depth: u32) -> Plan {
        if depth == 0 {
            Plan::Scan(0)
        } else {
            let left = subtree(depth - 1);
            Plan::Join(Box::new(left.clone()), Box::new(left), Combine::Add)
        }
    }
    let ir = query::ir(&subtree(5)).unwrap();
    let lancy = Lancy::compile(&ir).unwrap();
    let llvm = Llvm::compile(&ir).unwrap();
    let cat = Catalog::new(&[&[(12, 2), (7, 3)]]).unwrap();
    for code in [&lancy as &dyn Executable, &llvm] {
        assert_eq!(cat.run(code, &mut [0; 256]).unwrap(), [12, 64, 7, 96]);
    }
}

fn contains_op(f: &Function, op: Binary) -> bool {
    f.instructions().iter().any(|inst| match inst {
        Instr::Bin(_, b, ..) => *b == op,
        Instr::Call { function, .. } => contains_op(function, op),
        _ => false,
    })
}

#[test]
fn plan_shape_selects_the_reachable_operator_kernels_at_build_time() {
    let scan = query::ir(&Plan::Scan(0)).unwrap();
    let joined = query::ir(&pricing_plan()).unwrap();
    for f in [&scan, &joined] {
        assert_eq!(f.arity(), 0);
        assert_eq!(
            f.buffer_kinds(),
            [BufferKind::Read, BufferKind::Read, BufferKind::Write]
        );
    }
    // XOR comes from hash partitioning, Min only from the staged projection.
    for op in [Binary::Xor, Binary::Min] {
        assert!(!contains_op(&scan, op));
        assert!(contains_op(&joined, op));
    }
    // The executable outlives the temporary plan passed to build().
    let code = Lancy::compile(&scan).unwrap();
    assert_eq!(
        Catalog::new(&[&[(4, 5)]])
            .unwrap()
            .run(&code, &mut [0; 2])
            .unwrap(),
        [4, 5]
    );
}

#[test]
fn workspace_exhaustion_has_an_explicit_result_and_caller_can_retry() {
    let f = query::ir(&Plan::Join(
        Box::new(Plan::Scan(0)),
        Box::new(Plan::Scan(1)),
        Combine::Add,
    ))
    .unwrap();
    let code = Lancy::compile(&f).unwrap();
    let cat = Catalog::new(&[&[(0, 7)], &[(0, 5)]]).unwrap();
    // Cover scan, hash-table allocation, and emission of the matched row.
    for n in 0..8 {
        let mut mem = [0x1234; 9];
        assert_eq!(
            cat.run(&code, &mut mem[..n]).unwrap_err(),
            "query workspace is too small"
        );
        assert!(mem[n..].iter().all(|&x| x == 0x1234));
    }
    assert_eq!(cat.run(&code, &mut [0; 8]).unwrap(), [0, 12]);
    for id in [2, usize::MAX] {
        let unknown_table = Lancy::compile(&query::ir(&Plan::Scan(id)).unwrap()).unwrap();
        assert!(
            cat.run(&unknown_table, &mut [0; 8])
                .unwrap_err()
                .contains("bounds")
        );
    }
}
