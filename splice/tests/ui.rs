//! Compile-time diagnostics of `#[kernel]`, checked against `tests/ui/*.stderr`.
//! One case per `syn::Error::new_spanned` site of splice-macros; regenerate the snapshots with `TRYBUILD=overwrite cargo test --test ui`.
//! The passing counterparts in `tests/ui-pass` make trybuild build (not
//! only check) the cases, so post-monomorphization errors are reported too.
#[test]
fn diagnostics() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/*.rs");
    cases.pass("tests/ui-pass/*.rs");
}
