//! Compile-time diagnostics of `#[kernel]`, checked against `tests/ui/*.stderr`.
//! One case per `syn::Error::new_spanned` site of splice-macros; regenerate the snapshots with `TRYBUILD=overwrite cargo test --test ui`.
#[test]
fn diagnostics() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
