// A `char` method other than a comparison is not a kernel operation.
use splice::kernel;

#[kernel]
fn upper(c: char) -> char {
    c.to_ascii_uppercase()
}

fn main() {
    let _ = upper::ir();
}
