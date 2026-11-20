use splice::{
    BufferArg, BufferKind, Codegen, Executable, kernel,
    simd::{Element, Lane},
};
use splice_cranelift::Cranelift;
use splice_lancy::Lancy;
use splice_llvm::Llvm;
use splice_text::Text;

enum Expr {
    Ge { col: usize, val: i32 },
    GeFloat { col: usize, val: f32 },
}

enum Plan {
    Filter { expr: Expr },
    Empty,
}

struct Schema(Vec<BufferKind>);

impl Schema {
    fn column(&self, index: usize) -> Option<Element> {
        self.0.get(index).map(|kind| kind.element())
    }
}

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

#[kernel]
fn query(
    #[comptime] schema: &Schema,
    #[comptime] plan: &Plan,
    out: &mut [i32],
    #[columns(&schema.0)] columns: &[BufferArg<'_>],
) -> i32 {
    match plan {
        Plan::Filter {
            expr: Expr::Ge { col, val },
        } => match schema.column(*col) {
            Some(Element::I32) => {
                let selected = columns[*col].as_slice::<i32>();
                filter_ge(selected, *val, out)
            }
            _ => -1,
        },
        Plan::Filter {
            expr: Expr::GeFloat { col, val },
        } => match schema.column(*col) {
            Some(Element::F32) => any_ge_float(columns[*col].as_slice::<f32>(), *val),
            _ => -1,
        },
        Plan::Empty => 0,
    }
}

#[kernel]
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
fn float_column_matches_the_rust_filter_including_nan_and_tail() -> splice::Result<()> {
    fn check(code: impl Executable, schema: &Schema, plan: &Plan) -> splice::Result<()> {
        for values in [&[][..], &[f32::NAN, 0.0, -1.0], &[2.5, f32::NAN]] {
            let expected = i32::from(values.iter().any(|&x| x >= 2.5));
            let columns = [BufferArg::Read(&[99]), BufferArg::ReadF32(values)];
            assert_eq!(query(schema, plan, &mut [], &columns), expected);
            let [first, second] = columns;
            assert_eq!(
                code.call(&[], &mut [BufferArg::Write(&mut []), first, second])?,
                expected
            );
        }
        Ok(())
    }
    let schema = Schema(vec![BufferKind::Read, BufferKind::new(Element::F32, false)]);
    let plan = Plan::Filter {
        expr: Expr::GeFloat { col: 1, val: 2.5 },
    };
    let ir = query::ir(&schema, &plan)?;
    let listing = query::build::<Text>(&schema, &plan)?;
    assert!(listing.contains("simd.ge.f32"));
    assert!(!listing.contains("filter_ge"));
    let rust = query::print(&schema, &plan)?;
    assert!(rust.contains("fn query(out: &mut [i32], columns:"));
    assert!(rust.contains("2.5"));
    assert!(rust.contains("load_select"));
    assert!(rust.contains("fn any_ge_float_1(col: &[f32])"));
    assert!(!rust.contains("filter_ge"));
    assert!(!rust.contains("comptime"));
    assert!(!rust.contains("store out of bounds"));
    check(query::build::<Lancy>(&schema, &plan)?, &schema, &plan)?;
    check(Lancy::compile(&ir)?, &schema, &plan)?;
    check(query::build::<Llvm>(&schema, &plan)?, &schema, &plan)?;
    check(Llvm::compile(&ir)?, &schema, &plan)?;
    check(query::build::<Cranelift>(&schema, &plan)?, &schema, &plan)
}

fn check(code: impl Executable, schema: &Schema, plan: &Plan, slot: usize) -> splice::Result<()> {
    let aux = [9.0_f64];
    for values in [[-4, 7, 8, 3], [7, -1, 10, 22]] {
        let mut expected = [0; 4];
        let pair = if slot == 0 {
            [BufferArg::Read(&values), BufferArg::ReadF64(&aux)]
        } else {
            [BufferArg::ReadF64(&aux), BufferArg::Read(&values)]
        };
        let count = query(schema, plan, &mut expected, &pair);
        let mut actual = [0; 4];
        let [first, second] = pair;
        assert_eq!(
            code.call(&[], &mut [BufferArg::Write(&mut actual), first, second])?,
            count
        );
        assert_eq!(actual, expected);
    }
    let mut out = [0];
    assert!(
        code.call(
            &[],
            &mut [
                BufferArg::Write(&mut out),
                BufferArg::ReadF64(&aux),
                BufferArg::ReadF64(&aux)
            ]
        )
        .is_err()
    );
    assert_eq!(out, [0]);
    let values = [100, 101];
    let [first, second] = if slot == 0 {
        [BufferArg::Read(&values), BufferArg::ReadF64(&aux)]
    } else {
        [BufferArg::ReadF64(&aux), BufferArg::Read(&values)]
    };
    assert!(
        code.call(&[], &mut [BufferArg::Write(&mut out), first, second])
            .is_err()
    );
    assert_eq!(out, [100]);
    Ok(())
}

#[test]
fn sql_specialization_reads_the_selected_column_from_each_batch() -> splice::Result<()> {
    let int = BufferKind::Read;
    let float = BufferKind::new(Element::F64, false);
    for (slot, val, layout) in [(0, 7, vec![int, float]), (1, 9, vec![float, int])] {
        let schema = Schema(layout);
        let plan = Plan::Filter {
            expr: Expr::Ge { col: slot, val },
        };
        let ir = query::ir(&schema, &plan)?;
        let listing = query::build::<Text>(&schema, &plan)?;
        assert_eq!(listing, Text::compile(&ir)?);
        assert!(listing.contains(&format!(" = {val}\n")), "{listing}");
        assert!(listing.contains("filter_ge"));
        assert!(!query::build::<Text>(&schema, &Plan::Empty)?.contains("filter_ge"));
        check(query::build::<Lancy>(&schema, &plan)?, &schema, &plan, slot)?;
        check(Lancy::compile(&ir)?, &schema, &plan, slot)?;
        check(query::build::<Llvm>(&schema, &plan)?, &schema, &plan, slot)?;
        check(Llvm::compile(&ir)?, &schema, &plan, slot)?;
        check(
            query::build::<Cranelift>(&schema, &plan)?,
            &schema,
            &plan,
            slot,
        )?;
    }
    Ok(())
}

#[kernel]
fn unchecked_query(
    #[comptime] col: usize,
    #[columns([BufferKind::new(Element::F64, false)])] columns: &[BufferArg<'_>],
) -> i32 {
    columns[col].as_slice::<i32>()[0]
}

#[test]
fn invalid_column_casts_return_build_errors() {
    assert_eq!(
        unchecked_query::ir(0).unwrap_err(),
        "column 0 expects i32 elements"
    );
    assert_eq!(
        unchecked_query::build::<Lancy>(1).err().unwrap(),
        "unknown column 1"
    );
    assert_eq!(
        unchecked_query::print(0).unwrap_err(),
        "column 0 expects i32 elements"
    );
    assert_eq!(unchecked_query::print(1).unwrap_err(), "unknown column 1");
}
