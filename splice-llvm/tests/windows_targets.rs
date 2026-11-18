#![cfg(windows)]

use inkwell::targets::{InitializationConfig, Target, TargetMachine};

#[test]
fn windows_import_library_supports_inkwell_all_and_native_initializers() {
    assert_eq!(splice_llvm::Llvm::version().unwrap().0, 19);
    let config = InitializationConfig::default();
    Target::initialize_all(&config);
    let host = Target::from_triple(&TargetMachine::get_default_triple()).unwrap();
    assert!(host.has_target_machine() && host.has_asm_backend());
    Target::initialize_native(&config).unwrap();
}
