mod relational;

use relational::{Catalog, pricing_plan, query};
use splice::Executable;
use splice_cranelift::Cranelift;
#[cfg(target_arch = "x86_64")]
use splice_lancy::Lancy;
use splice_llvm::Llvm;
use splice_text::Text;

#[cfg(test)]
#[path = "relational/tests.rs"]
mod tests;

fn main() -> Result<(), String> {
    let plan = pricing_plan();
    match std::env::args().nth(1).as_deref() {
        #[cfg(target_arch = "x86_64")]
        None | Some("lancy") => run(query::build::<Lancy>(&plan)?),
        #[cfg(not(target_arch = "x86_64"))]
        None => run(query::build::<Llvm>(&plan)?),
        #[cfg(not(target_arch = "x86_64"))]
        Some("lancy") => Err("Lancy requires x86-64".into()),
        Some("cranelift") => run(query::build::<Cranelift>(&plan)?),
        Some("llvm") => run(query::build::<Llvm>(&plan)?),
        Some("text") => {
            print!("{}", query::build::<Text>(&plan)?);
            Ok(())
        }
        Some(other) => Err(format!(
            "unknown backend {other}; choose lancy, cranelift, llvm or text"
        )),
    }
}

fn run(code: impl Executable) -> Result<(), String> {
    let catalog = Catalog::new(&[
        &[(7, 3), (8, 1), (7, 4)],
        &[(7, 10), (8, 2)],
        &[(7, 1), (8, 0)],
        &[(7, 2)],
        &[(7, 5), (7, 20)],
    ])?;
    let mut workspace = vec![0; 256];
    let rows = catalog.run(&code, &mut workspace)?;
    let mut result = rows.as_chunks::<2>().0.to_vec();
    result.sort_unstable();
    assert_eq!(result, [[7, 61], [7, 71], [7, 91], [7, 101]]);
    for [product, value] in rows.as_chunks::<2>().0 {
        println!("product_id={product} adjusted_value={value}");
    }
    Ok(())
}
