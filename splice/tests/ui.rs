//! Compile-time diagnostics of `#[kernel]`, checked against `tests/ui/*.stderr`.
//! Regenerate the snapshots with `TRYBUILD=overwrite cargo test --test ui`.
#[test]
fn diagnostics() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
