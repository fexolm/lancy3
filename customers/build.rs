fn main() {
    let mut cpp = cc::Build::new();
    cpp.cpp(true).std("c++17").opt_level(3);
    let tool = cpp.get_compiler();
    let avx2 = std::env::var_os("CARGO_FEATURE_AVX2").is_some();
    if avx2 {
        cpp.flag(if tool.is_like_msvc() {
            "/arch:AVX2"
        } else {
            "-mavx2"
        });
    }
    for customer in ["rules", "decoder", "audio", "sql", "joins"] {
        let source = format!("{customer}/reference.cpp");
        println!("cargo:rerun-if-changed={source}");
        cpp.file(source);
    }
    println!("cargo:rustc-env=CUSTOMER_CXX={}", tool.path().display());
    cpp.compile("customer_references");
}
