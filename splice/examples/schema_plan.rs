use splice::{
    BufferArg, BufferKind, Executable, kernel,
    simd::{Element, Lane, LaneElement},
};
use splice_lancy::Lancy;

#[derive(Clone, Copy)]
enum Compare {
    Ge,
    Lt,
}

#[derive(Clone, Copy)]
enum Literal {
    U16(u16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

struct Expr {
    col: usize,
    op: Compare,
    val: Literal,
}

struct Projection {
    expr: Expr,
    bit: i32,
}

#[derive(splice::KernelType)]
struct Schema {
    layout: [BufferKind; 6],
    elements: [Element; 6],
}

impl Schema {
    fn from_batch(columns: &[BufferArg<'_>; 6]) -> Self {
        let layout = std::array::from_fn(|i| columns[i].kind());
        Self {
            elements: layout.map(BufferKind::element),
            layout,
        }
    }
}

#[derive(splice::KernelType)]
struct Plan {
    predicates: Vec<Expr>,
    left_key: usize,
    right_key: usize,
    projections: Vec<Projection>,
}

// One active lane keeps the operators short and works for every lane width.
#[kernel]
fn filter<T: LaneElement>(
    col: &[T],
    #[comptime] op: Compare,
    #[comptime] val: T,
    keep: &mut [i32],
) -> i32 {
    let active = Lane::<T>::first_n(1);
    let zero = Lane::<T>::splat(T::ZERO);
    let threshold = Lane::<T>::splat(val);
    for row in 0..keep.len() as i32 {
        let value = Lane::<T>::load_select(col, row as usize, active, zero);
        let pass = match op {
            Compare::Ge => value.simd_ge(threshold),
            Compare::Lt => value.simd_lt(threshold),
        };
        keep[row as usize] &= (pass & active).any() as i32;
    }
    0
}

#[kernel]
fn join<T: LaneElement>(
    left_key: &[T],
    right_key: &[T],
    keep: &[i32],
    right_rows: &mut [i32],
) -> i32 {
    let active = Lane::<T>::first_n(1);
    let zero = Lane::<T>::splat(T::ZERO);
    let mut count = 0;
    for left in 0..keep.len() as i32 {
        if keep[left as usize] != 0 {
            let key = Lane::<T>::load_select(left_key, left as usize, active, zero);
            for right in 0..right_key.len() as i32 {
                let other = Lane::<T>::load_select(right_key, right as usize, active, zero);
                if (key.simd_eq(other) & active).any() {
                    right_rows[count as usize] = right;
                    count += 1;
                }
            }
        }
    }
    count
}

#[kernel]
fn project<T: LaneElement>(
    col: &[T],
    right_rows: &[i32],
    count: i32,
    #[comptime] op: Compare,
    #[comptime] val: T,
    #[comptime] bit: i32,
    out: &mut [i32],
) -> i32 {
    let active = Lane::<T>::first_n(1);
    let zero = Lane::<T>::splat(T::ZERO);
    let threshold = Lane::<T>::splat(val);
    for i in 0..count {
        let row = right_rows[i as usize];
        let value = Lane::<T>::load_select(col, row as usize, active, zero);
        let pass = match op {
            Compare::Ge => value.simd_ge(threshold),
            Compare::Lt => value.simd_lt(threshold),
        };
        if (pass & active).any() {
            out[i as usize] |= bit;
        }
    }
    0
}

#[kernel]
fn query(
    #[comptime] schema: &Schema,
    #[comptime] plan: &Plan,
    keep: &mut [i32],
    right_rows: &mut [i32],
    out: &mut [i32],
    #[columns(&schema.layout)] columns: &[BufferArg<'_>],
) -> i32 {
    for row in 0..keep.len() as i32 {
        keep[row as usize] = 1;
    }
    for expr in &plan.predicates {
        let Expr { col, op, val } = *expr;
        if col >= schema.elements.len() {
            return -1;
        }
        match (schema.elements[col], val) {
            (Element::U16, Literal::U16(val)) => {
                filter::<u16>(columns[col].as_slice::<u16>(), op, val, keep)
            }
            (Element::I32, Literal::I32(val)) => {
                filter::<i32>(columns[col].as_slice::<i32>(), op, val, keep)
            }
            (Element::I64, Literal::I64(val)) => {
                filter::<i64>(columns[col].as_slice::<i64>(), op, val, keep)
            }
            (Element::F32, Literal::F32(val)) => {
                filter::<f32>(columns[col].as_slice::<f32>(), op, val, keep)
            }
            (Element::F64, Literal::F64(val)) => {
                filter::<f64>(columns[col].as_slice::<f64>(), op, val, keep)
            }
            _ => return -1,
        };
    }
    let left = plan.left_key;
    let right = plan.right_key;
    if left >= schema.elements.len() || right >= schema.elements.len() {
        return -1;
    }
    let count = match (schema.elements[left], schema.elements[right]) {
        (Element::U16, Element::U16) => join::<u16>(
            columns[left].as_slice::<u16>(),
            columns[right].as_slice::<u16>(),
            keep,
            right_rows,
        ),
        (Element::I32, Element::I32) => join::<i32>(
            columns[left].as_slice::<i32>(),
            columns[right].as_slice::<i32>(),
            keep,
            right_rows,
        ),
        (Element::I64, Element::I64) => join::<i64>(
            columns[left].as_slice::<i64>(),
            columns[right].as_slice::<i64>(),
            keep,
            right_rows,
        ),
        (Element::F32, Element::F32) => join::<f32>(
            columns[left].as_slice::<f32>(),
            columns[right].as_slice::<f32>(),
            keep,
            right_rows,
        ),
        (Element::F64, Element::F64) => join::<f64>(
            columns[left].as_slice::<f64>(),
            columns[right].as_slice::<f64>(),
            keep,
            right_rows,
        ),
        _ => return -1,
    };
    for row in 0..count {
        out[row as usize] = 0;
    }
    for projection in &plan.projections {
        let Projection {
            expr: Expr { col, op, val },
            bit,
        } = *projection;
        if col >= schema.elements.len() {
            return -1;
        }
        match (schema.elements[col], val) {
            (Element::U16, Literal::U16(val)) => project::<u16>(
                columns[col].as_slice::<u16>(),
                right_rows,
                count,
                op,
                val,
                bit,
                out,
            ),
            (Element::I32, Literal::I32(val)) => project::<i32>(
                columns[col].as_slice::<i32>(),
                right_rows,
                count,
                op,
                val,
                bit,
                out,
            ),
            (Element::I64, Literal::I64(val)) => project::<i64>(
                columns[col].as_slice::<i64>(),
                right_rows,
                count,
                op,
                val,
                bit,
                out,
            ),
            (Element::F32, Literal::F32(val)) => project::<f32>(
                columns[col].as_slice::<f32>(),
                right_rows,
                count,
                op,
                val,
                bit,
                out,
            ),
            (Element::F64, Literal::F64(val)) => project::<f64>(
                columns[col].as_slice::<f64>(),
                right_rows,
                count,
                op,
                val,
                bit,
                out,
            ),
            _ => return -1,
        };
    }
    count
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let columns = demo_columns();
    let schema = Schema::from_batch(&columns);
    let plan = demo_plan();
    match std::env::args().nth(1).as_deref() {
        Some("rust") => print!("{}", query::print(&schema, &plan)?),
        None | Some("run") => println!(
            "{:?}",
            run(
                query::build::<Lancy>(query::BuildArgs {
                    schema: &schema,
                    plan: &plan
                })?,
                columns
            )?
        ),
        Some(mode) => return Err(format!("unknown mode {mode}; use run or rust").into()),
    }
    Ok(())
}

fn demo_plan() -> Plan {
    // Project two boolean expressions into bits 0 and 1 of each output row:
    // SELECT p.price_cents < 120, p.rating >= 4.5
    // FROM orders o JOIN products p ON o.product_id = p.id
    // WHERE o.quantity >= 2 AND o.discount < 0.25 AND o.product_id >= 10;
    Plan {
        predicates: vec![
            Expr {
                col: 1,
                op: Compare::Ge,
                val: Literal::I32(2),
            },
            Expr {
                col: 2,
                op: Compare::Lt,
                val: Literal::F32(0.25),
            },
            Expr {
                col: 0,
                op: Compare::Ge,
                val: Literal::I64(10),
            },
        ],
        left_key: 0,
        right_key: 3,
        projections: vec![
            Projection {
                expr: Expr {
                    col: 4,
                    op: Compare::Lt,
                    val: Literal::U16(120),
                },
                bit: 1,
            },
            Projection {
                expr: Expr {
                    col: 5,
                    op: Compare::Ge,
                    val: Literal::F64(4.5),
                },
                bit: 2,
            },
        ],
    }
}

fn demo_columns() -> [BufferArg<'static>; 6] {
    [
        BufferArg::ReadI64(&[10, 10, 20, 30, 20]),
        BufferArg::Read(&[3, 1, 4, 2, 5]),
        BufferArg::ReadF32(&[0.1, 0.0, 0.3, 0.0, 0.2]),
        BufferArg::ReadI64(&[10, 10, 20]),
        BufferArg::ReadU16(&[100, 120, 250]),
        BufferArg::ReadF64(&[4.9, 4.0, 4.7]),
    ]
}

fn run(
    code: splice::TypedKernel<query::Signature, impl Executable>,
    columns: [BufferArg<'_>; 6],
) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let capacity = columns[0].len() * columns[3].len();
    let mut keep = vec![0; columns[0].len()];
    let mut right = vec![0; capacity];
    let mut out = vec![0; capacity];
    let count = code.call(query::CallArgs {
        keep: &mut keep,
        right_rows: &mut right,
        out: &mut out,
        columns: &columns,
    })?;
    let count = usize::try_from(count).map_err(|_| "query rejected the schema")?;
    out.truncate(count);
    Ok(out)
}

#[test]
fn mixed_schema_specializes_the_same_operators_for_different_elements()
-> Result<(), Box<dyn std::error::Error>> {
    use splice::Codegen;
    use splice_cranelift::Cranelift;
    #[cfg(feature = "llvm")]
    use splice_llvm::Llvm;

    let schema = Schema::from_batch(&demo_columns());
    let mut plan = demo_plan();
    let ir = query::ir(&schema, &plan)?;
    let rust = query::print(&schema, &plan)?;
    assert!(!rust.contains("match block"));
    for (name, ty) in [
        ("filter_1", "i32"),
        ("filter_2", "f32"),
        ("filter_3", "i64"),
        ("join_4", "i64"),
        ("project_5", "u16"),
        ("project_6", "f64"),
    ] {
        assert!(
            rust.lines()
                .any(|line| line.starts_with(&format!("fn {name}("))
                    && line.contains(&format!("&[{ty}]"))),
            "{name} with {ty}:\n{rust}"
        );
    }
    let expected = [3, 0, 2];
    assert_eq!(
        run(
            query::build::<Lancy>(query::BuildArgs {
                schema: &schema,
                plan: &plan
            })?,
            demo_columns()
        )?,
        expected
    );
    assert_eq!(run(Lancy::compile(&ir)?, demo_columns())?, expected);
    #[cfg(feature = "llvm")]
    assert_eq!(
        run(
            query::build::<Llvm>(query::BuildArgs {
                schema: &schema,
                plan: &plan
            })?,
            demo_columns()
        )?,
        expected
    );
    #[cfg(feature = "llvm")]
    assert_eq!(run(Llvm::compile(&ir)?, demo_columns())?, expected);
    assert_eq!(
        run(
            query::build::<Cranelift>(query::BuildArgs {
                schema: &schema,
                plan: &plan
            })?,
            demo_columns()
        )?,
        expected
    );

    let mut columns = demo_columns();
    columns[0] = BufferArg::ReadF64(&[10.0, 10.0, 20.0, f64::NAN, 20.0]);
    columns[3] = BufferArg::ReadF64(&[10.0, 10.0, 20.0]);
    plan.predicates[2].val = Literal::F64(10.0);
    let schema = Schema::from_batch(&columns);
    let rust = query::print(&schema, &plan)?;
    assert!(rust.contains("fn join_4(left_key: &[f64], right_key: &[f64]"));
    assert_eq!(
        run(
            query::build::<Lancy>(query::BuildArgs {
                schema: &schema,
                plan: &plan
            })?,
            columns
        )?,
        expected
    );
    Ok(())
}
