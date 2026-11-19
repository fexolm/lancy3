use splice::{Executable, kernel};
#[cfg(target_arch = "x86_64")]
use splice_lancy::Lancy as Backend;
#[cfg(not(target_arch = "x86_64"))]
use splice_llvm::Llvm as Backend;

#[derive(splice::ComptimeKey)]
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
        let single = project::build::<Backend>(&expr)?;
        let prices = [10, 20, 30];
        let quantities = [2, 3, 4];
        for i in 0..expected.len() {
            let args = [prices[i], quantities[i]];
            assert_eq!(single.call(&args, &mut [])?, expected[i]);
            assert_eq!(project(&expr, args[0], args[1]), expected[i]);
        }
        println!("{sql} => {expected:?}");
    }
    Ok(())
}
