use lancy_jit::{Builder, Program};

/// Per-row revenue for items in stock: price * quantity, otherwise zero.
pub fn revenue() -> Program {
    let mut ir = Builder::new(3);
    let price = ir.input(0);
    let quantity = ir.input(1);
    let stock = ir.input(2);
    let zero = ir.constant(0);
    let available = ir.lt(zero, stock);
    let gross = ir.mul(price, quantity);
    let result = ir.select(available, gross, zero);
    ir.finish(result).expect("valid revenue expression")
}
