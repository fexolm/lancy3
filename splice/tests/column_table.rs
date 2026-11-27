use splice::Function;
use splice::hir::{Constant, Instr, SimdOp};
use splice::lancy::Lancy;
use splice::{
    BufferArg, BufferKind, Codegen, Executable, TypedKernel, kernel,
    simd::{Element, Lane},
};

/// Whether staging left a call to the kernel function `name`.
fn calls(function: &Function, name: &str) -> bool {
    splice_testkit::instructions(function).any(|instruction| {
        matches!(instruction, Instr::Call { function, .. }
            if function.source().is_some_and(|source| source.name == name))
    })
}

enum Expr {
    Ge { col: usize, val: i32 },
    GeFloat { col: usize, val: f32 },
}

enum Plan {
    Filter { expr: Expr },
    Empty,
}

struct Schema(Vec<BufferKind>);

#[kernel]
fn filter_ge(col: &[i32], #[comptime] val: i32, out: &mut [i32]) -> i32 {
    let mut n = 0;
    for row in 0..col.len() as i32 {
        let value = col[row as usize];
        if value >= val {
            out[n as usize] = value;
            n += 1;
        }
    }
    n
}

#[kernel(print)]
#[allow(unused_variables, reason = "schema describes the staged column layout")]
fn query(
    #[comptime] schema: &Schema,
    #[comptime] plan: &Plan,
    out: &mut [i32],
    #[columns(&schema.0)] columns: &[BufferArg<'_>],
) -> i32 {
    match plan {
        Plan::Filter {
            expr: Expr::Ge { col, val },
        } => {
            let selected = columns[*col].as_slice::<i32>();
            filter_ge(selected, *val, out)
        }
        Plan::Filter {
            expr: Expr::GeFloat { col, val },
        } => any_ge_float(columns[*col].as_slice::<f32>(), *val),
        Plan::Empty => 0,
    }
}

#[kernel(print)]
fn any_ge_float(col: &[f32], #[comptime] val: f32) -> i32 {
    let width = Lane::<f32>::lanes() as i32;
    let mut row = 0;
    let end = col.len() as i32;
    while row < end {
        let active = Lane::<f32>::first_n((end - row) as usize);
        let values = Lane::<f32>::load_select(col, row as usize, active, Lane::<f32>::splat(0.0));
        if (values.simd_ge(Lane::<f32>::splat(val)) & active).any() {
            return 1;
        }
        row += width;
    }
    0
}

#[test]
fn float_column_matches_the_rust_filter_including_nan_and_tail()
-> Result<(), Box<dyn std::error::Error>> {
    fn check<E: Executable>(
        code: TypedKernel<query::Signature, E>,
        schema: &Schema,
        plan: &Plan,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for values in [&[][..], &[f32::NAN, 0.0, -1.0], &[2.5, f32::NAN]] {
            let expected = i32::from(values.iter().any(|&x| x >= 2.5));
            let columns = [
                BufferArg::read::<i32>(&[99]),
                BufferArg::read::<f32>(values),
            ];
            assert_eq!(query(schema, plan, &mut [], &columns), expected);
            assert_eq!(
                code.call(query::CallArgs {
                    out: &mut [],
                    columns: &columns
                })?,
                expected
            );
        }
        Ok(())
    }
    let schema = Schema(vec![
        BufferKind::READ_I32,
        BufferKind::new(Element::F32, false),
    ]);
    let plan = Plan::Filter {
        expr: Expr::GeFloat { col: 1, val: 2.5 },
    };
    let ir = query::ir(&schema, &plan)?;
    assert!(
        splice_testkit::simd_ops(&ir)
            .iter()
            .any(|simd| simd.op == SimdOp::Ge && simd.element == Element::F32)
    );
    assert!(!calls(&ir, "filter_ge"));
    let rust = query::print(&schema, &plan)?;
    assert!(rust.contains("fn query(out: &mut [i32], columns:"));
    assert!(rust.contains("2.5"));
    assert!(rust.contains("load_select"));
    assert!(rust.contains("fn any_ge_float_1(col: &[f32])"));
    assert!(!rust.contains("filter_ge"));
    assert!(!rust.contains("comptime"));
    assert!(!rust.contains("store out of bounds"));
    splice_testkit::each_backend!(B => {
        check(
            query::build::<B>(query::BuildArgs {
                schema: &schema,
                plan: &plan,
            })?,
            &schema,
            &plan,
        )?;
        check(B::compile(&ir)?, &schema, &plan)?;
    });
    Ok(())
}

fn check<E: Executable>(
    code: TypedKernel<query::Signature, E>,
    schema: &Schema,
    plan: &Plan,
    slot: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let aux = [9.0_f64];
    for values in [[-4, 7, 8, 3], [7, -1, 10, 22]] {
        let mut expected = [0; 4];
        let pair = if slot == 0 {
            [BufferArg::read(&values), BufferArg::read::<f64>(&aux)]
        } else {
            [BufferArg::read::<f64>(&aux), BufferArg::read(&values)]
        };
        let count = query(schema, plan, &mut expected, &pair);
        let mut actual = [0; 4];
        assert_eq!(
            code.call(query::CallArgs {
                out: &mut actual,
                columns: &pair
            })?,
            count
        );
        assert_eq!(actual, expected);
    }
    let mut out = [0];
    assert!(
        code.call(query::CallArgs {
            out: &mut out,
            columns: &[BufferArg::read::<f64>(&aux), BufferArg::read::<f64>(&aux)],
        })
        .is_err()
    );
    assert_eq!(out, [0]);
    let values = [100, 101];
    let columns = if slot == 0 {
        [BufferArg::read(&values), BufferArg::read::<f64>(&aux)]
    } else {
        [BufferArg::read::<f64>(&aux), BufferArg::read(&values)]
    };
    assert!(
        code.call(query::CallArgs {
            out: &mut out,
            columns: &columns
        })
        .is_err()
    );
    assert_eq!(out, [100]);
    Ok(())
}

#[test]
fn sql_specialization_reads_the_selected_column_from_each_batch()
-> Result<(), Box<dyn std::error::Error>> {
    let int = BufferKind::READ_I32;
    let float = BufferKind::new(Element::F64, false);
    for (slot, val, layout) in [(0, 7, vec![int, float]), (1, 9, vec![float, int])] {
        let schema = Schema(layout);
        let plan = Plan::Filter {
            expr: Expr::Ge { col: slot, val },
        };
        let ir = query::ir(&schema, &plan)?;
        assert!(
            splice_testkit::instructions(&ir).any(|instruction| matches!(
                instruction, Instr::Const(_, constant) if *constant == Constant::from(val)
            )),
            "{ir:?}"
        );
        assert!(calls(&ir, "filter_ge"));
        assert!(!calls(&*query::ir(&schema, &Plan::Empty)?, "filter_ge"));
        splice_testkit::each_backend!(B => {
            check(
                query::build::<B>(query::BuildArgs {
                    schema: &schema,
                    plan: &plan,
                })?,
                &schema,
                &plan,
                slot,
            )?;
            check(B::compile(&ir)?, &schema, &plan, slot)?;
        });
    }
    Ok(())
}

#[kernel(print)]
fn unchecked_query(
    #[comptime] col: usize,
    #[columns([BufferKind::new(Element::F64, false)])] columns: &[BufferArg<'_>],
) -> i32 {
    columns[col].as_slice::<i32>()[0]
}

#[test]
fn invalid_column_casts_return_build_errors() {
    let staging = |error: splice::BuildError, expected: &str| {
        assert!(
            matches!(&error, splice::BuildError::Staging(message) if message == expected),
            "{error:?}"
        );
    };
    staging(
        unchecked_query::ir(0).err().unwrap(),
        "column 0 expects i32 elements",
    );
    staging(
        unchecked_query::build::<Lancy>(unchecked_query::BuildArgs { col: 1 })
            .err()
            .unwrap(),
        "unknown column 1",
    );
    staging(
        unchecked_query::print(0).unwrap_err(),
        "column 0 expects i32 elements",
    );
    staging(unchecked_query::print(1).unwrap_err(), "unknown column 1");
}
