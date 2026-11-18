use lancy_jit::{AdaptiveKernel, ScalarFn, kernel};

enum SqlExpr {
    Price,
    Quantity,
    Literal(i32),
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    AtLeast(Box<Self>, i32),
}

#[kernel]
fn keep_at_least(value: i32, floor: i32) -> i32 {
    if value >= floor { value } else { 0 }
}

#[kernel]
fn project(#[comptime] expr: &SqlExpr, price: i32, quantity: i32) -> i32 {
    match expr {
        SqlExpr::Price => price,
        SqlExpr::Quantity => quantity,
        SqlExpr::Literal(value) => *value,
        SqlExpr::Add(a, b) => project(a, price, quantity) + project(b, price, quantity),
        SqlExpr::Mul(a, b) => project(a, price, quantity) * project(b, price, quantity),
        SqlExpr::AtLeast(child, floor) => keep_at_least(project(child, price, quantity), *floor),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use SqlExpr::*;
    let expressions = [
        (
            "price + 5",
            Add(Box::new(Price), Box::new(Literal(5))),
            [15, 25, 35],
        ),
        (
            "price * quantity",
            Mul(Box::new(Price), Box::new(Quantity)),
            [20, 60, 120],
        ),
        (
            "CASE WHEN price >= 20 THEN price ELSE 0 END",
            AtLeast(Box::new(Price), 20),
            [0, 20, 30],
        ),
    ];
    for (sql, expr, expected) in expressions {
        let function = project::build(&expr)?;
        let single = ScalarFn::<2>::compile(&function)?;
        let prices = [10, 20, 30];
        let quantities = [2, 3, 4];
        for i in 0..expected.len() {
            let args = [prices[i], quantities[i]];
            assert_eq!(single.invoke(args), expected[i]);
            assert_eq!(project(&expr, args[0], args[1]), expected[i]);
        }
        let program = function
            .vectorize()?
            .ok_or("expected an independent SQL projection")?;
        println!(
            "{sql}  SIMD IR = {:?}",
            lancy_jit::hir::projection::optimize(&program, &[]).ops()
        );
        let jit = AdaptiveKernel::compile(program)?;
        let mut out = [0; 3];
        jit.run(&[&prices, &quantities], &mut out)?;
        assert_eq!(out, expected);
        println!("  => {out:?}");
    }
    Ok(())
}
