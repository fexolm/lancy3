use crate::relational::*;
use splice::Codegen;
use splice::Executable;
use splice::cranelift::Cranelift;
use splice::hir::{BufferKind, Function, Instr, SimdOp};
use splice::lancy::Lancy;
#[cfg(feature = "llvm")]
use splice::llvm::Llvm;

type Row = (i32, i32);

// key ^ (key >> 16) has zeroes in the low 16 bits for these small salts.
fn colliding_key(salt: i32) -> i32 {
    (salt << 16) | salt
}

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

fn contains_join(plan: &Plan) -> bool {
    match plan {
        Plan::Join(..) => true,
        Plan::Filter(child, _) | Plan::Project(child, ..) => contains_join(child),
        Plan::Scan(_) => false,
    }
}

fn assert_rows(plan: &Plan, words: &[i32], expected: &[Row]) {
    assert_eq!(words.len(), expected.len() * 2);
    let mut rows: Vec<Row> = words.chunks_exact(2).map(|v| (v[0], v[1])).collect();
    if contains_join(plan) {
        let mut expected = expected.to_vec();
        expected.sort_unstable();
        rows.sort_unstable();
        assert_eq!(rows, expected);
    } else {
        assert_eq!(rows, expected);
    }
}

fn compare(plan: &Plan, tables: &[&[Row]], kernels: &[&dyn Executable]) {
    let cat = Catalog::new(tables).unwrap();
    let expected = reference(plan, tables);
    let input_rows = tables.iter().map(|t| t.len()).sum::<usize>();
    let mut state = vec![-1; plan.state_words()];
    let mut scratch = vec![0x7fff; (expected.len() + input_rows) * 8 + 512];
    for &code in kernels {
        scratch.fill(i32::MIN + 37);
        state.fill(-1);
        let short = state.len() - 1;
        assert!(cat.run(code, &mut scratch, &mut state[..short]).is_err());
        assert!(scratch.iter().all(|&v| v == i32::MIN + 37));
        assert!(state.iter().all(|&v| v == -1));
        assert_rows(
            plan,
            cat.run(code, &mut scratch, &mut state).unwrap(),
            &expected,
        );
    }
}

fn check_workspace_boundary(plan: &Plan, tables: &[&[Row]], kernels: &[&dyn Executable]) {
    let catalog = Catalog::new(tables).unwrap();
    let want = reference(plan, tables);
    let sentinel = -1_234_567;
    let mut enough = 8 * (tables.iter().map(|t| t.len()).sum::<usize>() + 2 * want.len()) + 256;
    let mut storage = vec![sentinel; enough + 2];
    let mut state = vec![-1; plan.state_words()];
    // The first kernel finds the boundary; every kernel must agree on it.
    let mut fits = |words: &mut [i32]| match catalog.run(kernels[0], words, &mut state) {
        Ok(rows) => {
            assert_rows(plan, rows, &want);
            true
        }
        Err(RunError::WorkspaceTooSmall) => false,
        Err(error) => panic!("{error:?}"),
    };
    assert!(fits(&mut storage[1..=enough]));
    let mut short = 0;
    while enough - short > 1 {
        let middle = short + (enough - short) / 2;
        if !fits(&mut storage[1..=middle]) {
            short = middle;
        } else {
            enough = middle;
        }
    }
    // Reuse the same address so partial writes have the same alignment.
    storage.fill(sentinel);
    assert!(!fits(&mut storage[1..=short]));
    let failed = storage.clone();
    assert_eq!(failed[0], sentinel);
    assert!(failed[short + 1..].iter().all(|&v| v == sentinel));
    for &kernel in kernels {
        storage.fill(sentinel);
        assert_eq!(
            catalog
                .run(kernel, &mut storage[1..=short], &mut state)
                .unwrap_err(),
            RunError::WorkspaceTooSmall
        );
        assert_eq!(storage, failed);
        assert_rows(
            plan,
            catalog
                .run(kernel, &mut storage[1..=enough], &mut state)
                .unwrap(),
            &want,
        );
        assert_eq!(storage[0], sentinel);
        assert!(storage[enough + 1..].iter().all(|&v| v == sentinel));
    }
}

fn with_kernels(plan: &Plan, test: impl FnOnce(&[&dyn Executable])) {
    let f = query::ir(&plan.prepare()).unwrap();
    let lancy = Lancy::compile(&f).unwrap().into_inner();
    #[cfg(feature = "llvm")]
    let llvm = Llvm::compile(&f).unwrap().into_inner();
    let direct_lancy = query::build::<Lancy>(query::BuildArgs {
        program: &plan.prepare(),
    })
    .unwrap()
    .into_inner();
    #[cfg(feature = "llvm")]
    let direct_llvm = query::build::<Llvm>(query::BuildArgs {
        program: &plan.prepare(),
    })
    .unwrap()
    .into_inner();
    let cranelift = Cranelift::compile(&f).unwrap().into_inner();
    let direct_cranelift = query::build::<Cranelift>(query::BuildArgs {
        program: &plan.prepare(),
    })
    .unwrap()
    .into_inner();
    test(&[
        &lancy,
        #[cfg(feature = "llvm")]
        &llvm,
        &cranelift,
        &direct_lancy,
        #[cfg(feature = "llvm")]
        &direct_llvm,
        &direct_cranelift,
    ]);
}

#[test]
fn branching_join_tree_runs_against_changing_tables() {
    let plan = pricing_plan();
    let tables: &[&[Row]] = &[
        &[(-7, 5), (1, 3), (9, 6), (1, 5)],
        &[(1, 9), (-7, 3), (9, 4)],
        &[(9, 1), (1, 1), (-7, 1)],
        &[(1, 3), (9, 8)],
        &[(9, 4), (1, 3), (1, 8)],
    ];
    with_kernels(&plan, |kernels| {
        compare(&plan, tables, kernels);
        for empty in 0..tables.len() {
            let mut changed = tables.to_vec();
            changed[empty] = &[];
            compare(&plan, &changed, kernels);
        }
    });
}

#[test]
fn filter_compacts_batches_and_project_reads_both_original_columns() {
    let width = splice::Target::host().unwrap().vector_bytes() / size_of::<i32>();
    let plan = Plan::Project(
        Box::new(Plan::Filter(
            Box::new(Plan::Scan(0)),
            Predicate::AtLeast(Expr::Value, Expr::Literal(0)),
        )),
        Expr::Value,
        Expr::Add(
            Box::new(Expr::Key),
            Box::new(Expr::Mul(
                Box::new(Expr::Value),
                Box::new(Expr::Literal(i32::MAX)),
            )),
        ),
    );
    let table: Vec<_> = (0..3 * width + 1)
        .map(|i| {
            // The first packet is empty, the second keeps its endpoints,
            // the third is full and the last contains just one valid row.
            let keep = i >= 2 * width || i == width || i == 2 * width - 1;
            (
                i32::MIN + i as i32,
                if keep { i32::MAX - i as i32 } else { -1 },
            )
        })
        .collect();
    with_kernels(&plan, |kernels| {
        for n in [0, width - 1, width, width + 1, 2 * width, table.len()] {
            compare(&plan, &[&table[..n]], kernels);
        }
    });
}

#[test]
fn short_scan_filter_project_batches_fit_shifted_workspaces() {
    let width = splice::Target::host().unwrap().vector_bytes() / size_of::<i32>();
    let plan = Plan::Project(
        Box::new(Plan::Filter(
            Box::new(Plan::Scan(1)),
            Predicate::AtLeast(Expr::Value, Expr::Literal(0)),
        )),
        Expr::Value,
        Expr::Add(Box::new(Expr::Key), Box::new(Expr::Value)),
    );
    let table: Vec<_> = (0..2 * width + 3)
        .map(|i| (7 - i as i32, if i % 3 == 0 { -1 } else { i as i32 }))
        .collect();
    let sentinel = 0x1234_5678;
    with_kernels(&plan, |kernels| {
        let mut state = vec![-1; plan.state_words()];
        for rows in [1, width - 1, width + 1, table.len()] {
            // Table one follows a different table in the catalog. The input
            // and output addresses need not have the same vector alignment.
            let tables: &[&[Row]] = &[&[(42, 13)], &table[..rows]];
            let catalog = Catalog::new(tables).unwrap();
            let want: Vec<_> = reference(&plan, tables)
                .into_iter()
                .flat_map(|(k, v)| [k, v])
                .collect();
            let words = 2 * rows;
            let mut storage = vec![sentinel; width + words + 1];
            for start in 1..=width {
                for &kernel in kernels {
                    storage.fill(sentinel);
                    assert_eq!(
                        catalog
                            .run(kernel, &mut storage[start..start + words], &mut state)
                            .unwrap(),
                        want
                    );
                    assert!(
                        storage[..start]
                            .iter()
                            .chain(&storage[start + words..])
                            .all(|&v| v == sentinel)
                    );
                }
            }
        }
    });
}

#[test]
fn fused_scan_filter_project_needs_only_two_words_for_one_survivor() {
    let width = splice::Target::host().unwrap().vector_bytes() / size_of::<i32>();
    let plan = Plan::Project(
        Box::new(Plan::Filter(
            Box::new(Plan::Scan(0)),
            Predicate::Equal(Expr::Value, Expr::Literal(7)),
        )),
        Expr::Value,
        Expr::Add(Box::new(Expr::Value), Box::new(Expr::Key)),
    );
    let n = 128 * width + 3;
    with_kernels(&plan, |kernels| {
        let mut state = vec![-1; plan.state_words()];
        for survivor in [width + 1, n - 1] {
            let mut rows: Vec<_> = (0..n).map(|i| (i as i32, -1)).collect();
            rows[survivor].1 = 7;
            let catalog = Catalog::new(&[&rows]).unwrap();
            let expected = [7, survivor as i32 + 7];
            let mut guarded = [-9; 4];
            for &kernel in kernels {
                guarded.fill(-9);
                assert_eq!(
                    catalog.run(kernel, &mut guarded[1..3], &mut state).unwrap(),
                    expected
                );
                assert_eq!((guarded[0], guarded[3]), (-9, -9));
            }
        }
    });
}

#[test]
fn fused_join_streams_all_hits_and_stays_successful_as_workspace_grows() {
    let rows = 4099;
    let right: Vec<Row> = (0..7).map(|k| (k, 5 * k - 13)).collect();
    let left: Vec<Row> = (0..rows)
        .map(|i| {
            let k = i % 7;
            (i - (5 * k - 13), k)
        })
        .collect();
    let join = Plan::Join(
        Box::new(Plan::Project(
            Box::new(Plan::Scan(0)),
            Expr::Value,
            Expr::Key,
        )),
        Box::new(Plan::Scan(1)),
        Combine::Add,
    );
    let plan = Plan::Project(
        Box::new(Plan::Filter(
            Box::new(join),
            Predicate::AtLeast(Expr::Value, Expr::Literal(rows - 6)),
        )),
        Expr::Value,
        Expr::Key,
    );
    let tables: &[&[Row]] = &[&left, &right];
    let expected = reference(&plan, tables);
    assert_eq!(expected.len(), 6);
    let catalog = Catalog::new(tables).unwrap();
    // The index and six surviving rows fit well inside this bound. Every
    // probe hits, so materializing either the left input or the join cannot fit.
    let mut storage = [-971; 98];
    let mut fits = false;
    with_kernels(&plan, |kernels| {
        let mut state = vec![-1; plan.state_words()];
        for capacity in 0..=96 {
            storage.fill(-971);
            let n = match catalog.run(kernels[0], &mut storage[1..1 + capacity], &mut state) {
                Ok(rows) => (rows.len() / 2) as i32,
                Err(RunError::WorkspaceTooSmall) => -1,
                Err(error) => panic!("{error:?}"),
            };
            if n < 0 {
                assert_eq!(n, -1);
                assert!(!fits, "adding workspace caused failure at {capacity} words");
            } else {
                fits = true;
                assert_rows(&plan, &storage[1..1 + 2 * n as usize], &expected);
            }
            for &kernel in kernels {
                storage.fill(-971);
                let result = catalog.run(kernel, &mut storage[1..1 + capacity], &mut state);
                if fits {
                    assert_rows(&plan, result.unwrap(), &expected);
                } else {
                    assert_eq!(result.unwrap_err(), RunError::WorkspaceTooSmall);
                }
                assert_eq!(storage[0], -971);
                assert!(storage[1 + capacity..].iter().all(|&x| x == -971));
            }
        }
        assert!(fits);
    });
}

#[test]
fn batched_probe_returns_every_build_side_match() {
    let width = splice::Target::host().unwrap().vector_bytes() / size_of::<i32>();
    let plan = Plan::Join(
        Box::new(Plan::Scan(0)),
        Box::new(Plan::Scan(1)),
        Combine::Add,
    );
    let a = colliding_key(1);
    let b = colliding_key(2);
    let keys = [0, a, 1, b, -1, a, i32::MIN, colliding_key(3)];
    let left: Vec<_> = (0..2 * width + 1)
        .map(|i| (keys[i % keys.len()], i32::MAX - i as i32))
        .collect();
    // Distinct payloads expose any lost or duplicated match.
    let right = [
        (a, 10),
        (0, 20),
        (b, 30),
        (a, 40),
        (0, 50),
        (a, 60),
        (i32::MIN, 70),
        (b, 80),
        (-1, 90),
    ];
    let tables: &[&[Row]] = &[&left, &right];
    with_kernels(&plan, |kernels| compare(&plan, tables, kernels));
}

#[test]
fn probe_verifies_keys_in_singleton_and_colliding_buckets() {
    let plan = Plan::Join(
        Box::new(Plan::Scan(0)),
        Box::new(Plan::Scan(1)),
        Combine::Multiply,
    );
    let a = colliding_key(1);
    let b = colliding_key(2);
    let c = colliding_key(3);
    let keys = [0, a, b, c, colliding_key(4), 65536, i32::MIN, 3, 7];
    let left: Vec<_> = (0..2 * (splice::Target::host().unwrap().vector_bytes() / size_of::<i32>())
        + 3)
        .map(|i| (keys[i % keys.len()], 17 + i as i32))
        .collect();
    // 65536 hashes to slot one. The other salts start from occupied slot zero.
    let unique = [0, 65536, 2, 3, 4, 5, 6, 7];
    // The fourth salt is absent even when the first three are present.
    let collisions = [0, a, b, c, 65536, 2, 3, 4];
    with_kernels(&plan, |kernels| {
        for build in [unique, collisions] {
            let right: Vec<_> = build
                .into_iter()
                .enumerate()
                .map(|(i, k)| (k, 200 + i as i32))
                .collect();
            compare(&plan, &[&left, &right], kernels);
        }
    });
}

#[test]
fn colliding_keys_with_duplicate_values_and_a_late_neighbor_both_match() {
    let width = splice::Target::host().unwrap().vector_bytes() / size_of::<i32>();
    let plan = Plan::Join(
        Box::new(Plan::Scan(0)),
        Box::new(Plan::Scan(1)),
        Combine::Add,
    );
    // All sixteen keys start probing from slot zero.
    let left: Vec<_> = (0..3 * width)
        .map(|i| {
            let key = match i % width {
                0 => colliding_key(12),
                1 => colliding_key(2),
                2 => colliding_key(15),
                _ => colliding_key(16),
            };
            (key, i as i32 * 1000)
        })
        .collect();
    let mut right: Vec<_> = (0..16).map(|i| (colliding_key(i), i + 1)).collect();
    with_kernels(&plan, |kernels| {
        compare(&plan, &[&left, &right], kernels);
        right[14].0 = colliding_key(12);
        compare(&plan, &[&left, &right], kernels);
    });
}

#[test]
fn refilled_probe_lanes_preserve_late_hits_and_drain_the_tail() {
    let width = splice::Target::host().unwrap().vector_bytes() / size_of::<i32>();
    let chain = 4 * width + 5;
    let mut right: Vec<_> = (0..chain)
        .map(|i| (colliding_key(i as i32), 100 + i as i32))
        .collect();
    let late = right[chain - 1].0;
    let neighbor = right[chain - 3].0;
    let fast = chain as i32 + 2;
    right.extend([(fast, 23), (fast + 1, 37)]);
    // The first two probes collide with many distinct keys. Later probes
    // hit or miss just beyond the cluster while those old lanes stay live.
    let rows = 5 * width + 3;
    let left: Vec<_> = (0..rows)
        .map(|i| {
            let k = match i {
                0 => late,
                1 => neighbor,
                i if i == rows - 1 => late,
                _ => [fast, fast + 1, colliding_key(1) ^ fast, fast + 2][i % 4],
            };
            (k, i as i32 * 1000)
        })
        .collect();
    let plan = Plan::Join(
        Box::new(Plan::Scan(0)),
        Box::new(Plan::Scan(1)),
        Combine::Add,
    );
    with_kernels(&plan, |kernels| {
        for n in [width - 1, width, width + 1, rows] {
            compare(&plan, &[&left[..n], &right], kernels);
        }
        // A displaced key now has two values; the final incoming probe for
        // that same key stays active after input has been exhausted.
        right[chain - 2].0 = late;
        // At EOF one lane retries, one hits, and one lands on an empty slot.
        compare(
            &plan,
            &[&[(late, 11), (fast, 23), (fast + 2, 45)], &right],
            kernels,
        );
        let tables: &[&[Row]] = &[&left, &right];
        compare(&plan, tables, kernels);

        check_workspace_boundary(&plan, tables, kernels);
    });
}

#[test]
fn duplicate_values_span_several_batches_of_many_to_many_matches() {
    let width = splice::Target::host().unwrap().vector_bytes() / size_of::<i32>();
    let count = 6 * width + 3;
    let other = colliding_key(1);
    // Interleave two colliding keys with unrelated rows and give each
    // right value a distinct payload.
    let right: Vec<_> = (0..count)
        .map(|i| {
            let key = if i % 13 == 4 {
                2
            } else if i % 11 == 7 {
                other
            } else {
                0
            };
            (key, i as i32 + 1)
        })
        .collect();
    let probe = [0, colliding_key(2), 2, 0, other, 17];
    let left: Vec<_> = (0..width + 3)
        .map(|i| (probe[i % probe.len()], 1000 * (i as i32 + 1)))
        .collect();
    let plan = Plan::Join(
        Box::new(Plan::Scan(0)),
        Box::new(Plan::Scan(1)),
        Combine::Add,
    );
    with_kernels(&plan, |kernels| compare(&plan, &[&left, &right], kernels));
}

#[test]
fn self_join_after_rekey_handles_collisions_and_duplicate_rows() {
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
    let f = query::ir(&plan.prepare()).unwrap();
    let lancy = Lancy::compile(&f).unwrap().into_inner();
    #[cfg(feature = "llvm")]
    let llvm = Llvm::compile(&f).unwrap().into_inner();
    let duplicate = [(3, 3); 13];
    compare(
        &plan,
        &[&duplicate],
        &[
            &lancy,
            #[cfg(feature = "llvm")]
            &llvm,
        ],
    );
    let mixed = [
        (1, 9),
        (9, 1),
        (1, 1),
        (-2, -2),
        (17, 8),
        (i32::MIN, i32::MIN),
        (1, 9),
    ];
    compare(
        &plan,
        &[&mixed],
        &[
            &lancy,
            #[cfg(feature = "llvm")]
            &llvm,
        ],
    );
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
        let f = query::ir(&plan.prepare()).unwrap();
        let direct = query::build::<Lancy>(query::BuildArgs {
            program: &plan.prepare(),
        })
        .unwrap()
        .into_inner();
        let from_ir = Lancy::compile(&f).unwrap().into_inner();
        #[cfg(feature = "llvm")]
        let llvm = Llvm::compile(&f).unwrap().into_inner();
        compare(
            &plan,
            &[
                &[(0, 3), (-1, 6), (7, 2)],
                &[(7, 10), (-1, 9)],
                &[(-1, 1), (0, -1), (7, 3)],
            ],
            &[
                &direct,
                &from_ir,
                #[cfg(feature = "llvm")]
                &llvm,
            ],
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
    let cat = Catalog::new(&[&[(12, 2), (7, 3)]]).unwrap();
    let check = |depth, code: &dyn Executable| {
        let plan = subtree(depth);
        let scans = 1 << depth;
        assert_rows(
            &plan,
            cat.run(code, &mut [0; 1024], &mut vec![0; plan.state_words()])
                .unwrap(),
            &[(12, 2 * scans), (7, 3 * scans)],
        );
    };
    let ir = |depth| query::ir(&subtree(depth).prepare()).unwrap();
    check(5, &Lancy::compile(&ir(5)).unwrap().into_inner());
    // LLVM's time grows faster than the IR (6.4k/13.5k/27.6k instructions
    // take 0.7/1.8/5.7 s at depths 3/4/5), so it stops one level earlier.
    #[cfg(feature = "llvm")]
    check(4, &Llvm::compile(&ir(4)).unwrap().into_inner());
}

fn simd_ops(f: &Function) -> Vec<SimdOp> {
    let mut ops = Vec::new();
    for instruction in f.instructions() {
        match instruction {
            Instr::CallIntrinsic {
                intrinsic: splice::hir::Intrinsic::Simd(inst),
                ..
            } => ops.push(inst.op),
            Instr::Call { function, .. } => ops.extend(simd_ops(function)),
            _ => {}
        }
    }
    ops
}

#[test]
fn plan_shape_selects_the_reachable_operator_kernels_at_build_time() {
    let scan = query::ir(&Plan::Scan(0).prepare()).unwrap();
    let filter = query::ir(
        &Plan::Filter(
            Box::new(Plan::Scan(0)),
            Predicate::And(
                Box::new(Predicate::AtLeast(Expr::Value, Expr::Literal(2))),
                Box::new(Predicate::Equal(Expr::Key, Expr::Literal(7))),
            ),
        )
        .prepare(),
    )
    .unwrap();
    let project = query::ir(
        &Plan::Project(
            Box::new(Plan::Scan(0)),
            Expr::Add(Box::new(Expr::Key), Box::new(Expr::Value)),
            Expr::Mul(Box::new(Expr::Key), Box::new(Expr::Value)),
        )
        .prepare(),
    )
    .unwrap();
    let join = query::ir(
        &Plan::Join(
            Box::new(Plan::Scan(0)),
            Box::new(Plan::Scan(1)),
            Combine::Add,
        )
        .prepare(),
    )
    .unwrap();
    for f in [&scan, &filter, &project, &join] {
        assert_eq!(f.arity(), 0);
        assert_eq!(
            f.buffer_kinds(),
            [
                BufferKind::READ_I32,
                BufferKind::READ_I32,
                BufferKind::WRITE_I32,
                BufferKind::WRITE_I32
            ]
        );
    }
    let scan_ops = simd_ops(&scan);
    assert!(
        scan_ops
            .iter()
            .any(|op| matches!(op, SimdOp::AlignmentPrefix(_)))
    );
    assert!(
        scan_ops
            .iter()
            .any(|op| matches!(op, SimdOp::Load(_) | SimdOp::LoadSelect(_)))
    );
    assert!(
        scan_ops
            .iter()
            .any(|op| matches!(op, SimdOp::Store(_) | SimdOp::StoreSelect(_)))
    );
    let filter_ops = simd_ops(&filter);
    for op in [SimdOp::Ge, SimdOp::Eq, SimdOp::MaskAnd] {
        assert!(filter_ops.contains(&op), "missing filter operation {op:?}");
    }
    let project_ops = simd_ops(&project);
    assert!(project_ops.contains(&SimdOp::Add) && project_ops.contains(&SimdOp::Mul));
    let join_ops = simd_ops(&join);
    assert!(join_ops.iter().any(|op| matches!(op, SimdOp::Gather(_))));
    assert!(join_ops.contains(&SimdOp::Eq));
    assert!(join_ops.contains(&SimdOp::Compress));
    let full_plan_ops = simd_ops(&query::ir(&pricing_plan().prepare()).unwrap());
    assert!(full_plan_ops.contains(&SimdOp::Compress));
    assert!(!full_plan_ops.iter().any(|op| matches!(
        op,
        SimdOp::GatherSelect(_) | SimdOp::Scatter(_) | SimdOp::ScatterSelect(_)
    )));
    // The executable outlives the temporary plan passed to build().
    let code = Lancy::compile(&scan).unwrap().into_inner();
    assert_eq!(
        Catalog::new(&[&[(4, 5)]])
            .unwrap()
            .run(
                &code,
                &mut [0; 2],
                &mut vec![0; Plan::Scan(0).state_words()]
            )
            .unwrap(),
        [4, 5]
    );
}

#[test]
fn workspace_exhaustion_has_an_explicit_result_and_caller_can_retry() {
    let plan = Plan::Join(
        Box::new(Plan::Scan(0)),
        Box::new(Plan::Scan(1)),
        Combine::Add,
    );
    let tables: &[&[Row]] = &[&[(0, 7)], &[(0, 5)]];
    with_kernels(&plan, |kernels| {
        check_workspace_boundary(&plan, tables, kernels)
    });
    let cat = Catalog::new(tables).unwrap();
    for id in [2, usize::MAX] {
        let unknown_table =
            Lancy::compile(&query::ir(&Plan::Scan(id).prepare()).unwrap().into_inner()).unwrap();
        assert_eq!(
            cat.run(
                &unknown_table,
                &mut [0; 8],
                &mut vec![0; Plan::Scan(id).state_words()]
            ),
            Err(RunError::Call(splice::CallError::Trap(
                splice::Trap::Bounds
            )))
        );
    }
}
