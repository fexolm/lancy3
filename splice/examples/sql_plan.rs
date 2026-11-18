mod relational;

use relational::{Catalog, pricing_plan, query};
use splice::Codegen;
use splice_lancy::Lancy;
use splice_llvm::Llvm;

#[cfg(test)]
#[path = "relational/tests.rs"]
mod tests;

fn main() -> Result<(), String> {
    match std::env::args().nth(1).as_deref() {
        None | Some("lancy") => run::<Lancy>(),
        Some("llvm") => run::<Llvm>(),
        Some(other) => Err(format!("unknown backend {other}; choose lancy or llvm")),
    }
}

fn run<B: Codegen>() -> Result<(), String> {
    let code = query::build::<B>(&pricing_plan())?;
    let catalog = Catalog::new(&[
        &[(7, 3), (8, 1), (7, 4)],
        &[(7, 10), (8, 2)],
        &[(7, 1), (8, 0)],
        &[(7, 2)],
        &[(7, 5), (7, 20)],
    ])?;
    let mut workspace = vec![0; 256];
    let rows = catalog.run(&code, &mut workspace)?;
    assert_eq!(rows, [7, 61, 7, 91, 7, 71, 7, 101]);
    for [product, value] in rows.as_chunks::<2>().0 {
        println!("product_id={product} adjusted_value={value}");
    }
    Ok(())
}
