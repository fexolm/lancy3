use splice::kernel;

#[kernel]
fn add_one(value: i32) -> i32 {
    value + 1
}

#[cfg(feature = "lancy")]
#[test]
fn default_backend_builds_a_kernel() {
    let code = add_one::build::<splice::lancy::Lancy>(add_one::BuildArgs {}).unwrap();
    assert_eq!(code.call(add_one::CallArgs { value: 41 }).unwrap(), 42);
}

#[cfg(feature = "cranelift")]
#[test]
fn cranelift_feature_builds_a_kernel() {
    let code = add_one::build::<splice::cranelift::Cranelift>(add_one::BuildArgs {}).unwrap();
    assert_eq!(code.call(add_one::CallArgs { value: 41 }).unwrap(), 42);
}

#[cfg(feature = "llvm")]
#[test]
fn llvm_feature_builds_a_kernel() {
    let code = add_one::build::<splice::llvm::Llvm>(add_one::BuildArgs {}).unwrap();
    assert_eq!(code.call(add_one::CallArgs { value: 41 }).unwrap(), 42);
}
