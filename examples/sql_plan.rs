mod relational;

use lancy_jit::AdaptiveBufferKernel;
use relational::{Catalog, pricing_plan, query};

#[cfg(test)]
#[path = "relational/tests.rs"]
mod tests;

fn main() -> Result<(), String> {
    let plan = pricing_plan();
    let function = query::build(&plan)?;
    let code = AdaptiveBufferKernel::compile(function)?;
    let catalog = Catalog::new(&[
        &[(7, 3), (8, 1), (7, 4)],
        &[(7, 10), (8, 2)],
        &[(7, 1), (8, 0)],
        &[(7, 2)],
        &[(7, 5), (7, 20)],
    ])?;
    let mut workspace = vec![0; 256];
    let rows = catalog.run(code.current(), &mut workspace)?;
    assert_eq!(rows, [7, 61, 7, 91, 7, 71, 7, 101]);
    for row in rows.chunks_exact(2) {
        println!("product_id={} adjusted_value={}", row[0], row[1]);
    }
    code.wait_for_optimization()?;
    assert_eq!(
        catalog.run(code.current(), &mut workspace)?,
        [7, 61, 7, 91, 7, 71, 7, 101]
    );
    Ok(())
}
