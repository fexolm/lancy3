mod support;

use lancy::AdaptiveKernel;

fn main() -> Result<(), String> {
    let kernel = AdaptiveKernel::compile(support::revenue())?;
    let price: Vec<_> = (100..357).collect();
    let mut quantity = vec![3; price.len()];
    let stock: Vec<_> = (0..price.len()).map(|i| i32::from(i % 7 != 0)).collect();
    let mut output = vec![0; price.len()];

    kernel.run(&[&price, &quantity, &stock], &mut output)?;
    kernel.wait_for_optimization()?;
    println!("After profiling: {:#?}", kernel.stats());
    kernel.run(&[&price, &quantity, &stock], &mut output)?;

    // The first 128 rows still agree with the profiled quantity. The failing
    // block and every subsequent row must use the real quantity.
    quantity[130..].fill(9);
    kernel.run(&[&price, &quantity, &stock], &mut output)?;
    kernel.wait_for_optimization()?;
    for (row, actual) in output.iter().enumerate() {
        let expected = if stock[row] > 0 {
            price[row] * quantity[row]
        } else {
            0
        };
        assert_eq!(*actual, expected);
    }
    println!("After the data change: {:#?}", kernel.stats());
    println!("Revenue, rows 128..134: {:?}", &output[128..134]);

    // A later run of stable batches provides a new specialization opportunity.
    quantity.fill(9);
    for _ in 0..16 {
        kernel.run(&[&price, &quantity, &stock], &mut output)?;
    }
    kernel.wait_for_optimization()?;
    println!("After 16 stable batches: {:#?}", kernel.stats());
    Ok(())
}
