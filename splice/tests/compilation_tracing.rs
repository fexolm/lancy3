use splice::{BuildOptions, Codegen, kernel};
use splice_cranelift::Cranelift;
use splice_lancy::Lancy;
use splice_llvm::Llvm;
use splice_text::Text;
use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};
use tracing_subscriber::fmt::format::FmtSpan;

#[kernel]
fn increment(value: i32) -> i32 {
    value.wrapping_add(1)
}

struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn staged_and_portable_compilation_report_stage_durations() {
    let output = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::clone(&output);
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_span_events(FmtSpan::CLOSE)
        .with_ansi(false)
        .with_writer(move || Capture(Arc::clone(&writer)))
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let options = BuildOptions::Release;
        let ir = increment::ir_with_options(options).unwrap();
        for kernel in [
            increment::build_with_options::<Lancy>(options, increment::BuildArgs {}).unwrap(),
            Lancy::compile(&ir).unwrap(),
        ] {
            assert_eq!(kernel.call(increment::CallArgs { value: 41 }), Ok(42));
        }
        for kernel in [
            increment::build_with_options::<Llvm>(options, increment::BuildArgs {}).unwrap(),
            Llvm::compile(&ir).unwrap(),
        ] {
            assert_eq!(kernel.call(increment::CallArgs { value: 41 }), Ok(42));
        }
        let kernel =
            increment::build_with_options::<Cranelift>(options, increment::BuildArgs {}).unwrap();
        assert_eq!(kernel.call(increment::CallArgs { value: 41 }), Ok(42));
        assert!(Text::compile(&ir).unwrap().starts_with("fn increment("));
    });
    let bytes = output.lock().unwrap();
    let log = std::str::from_utf8(&bytes).unwrap();
    for (stage, expected_count) in [
        ("materialize_ir", 2),
        ("stage_lancy_hir", 1),
        ("lower_lancy_hir", 1),
        ("verify_lancy_hir", 4),
        ("optimize_lancy_hir", 2),
        ("select_instructions", 2),
        ("schedule_instructions", 2),
        ("allocate_registers", 2),
        ("emit_machine_code", 2),
        ("install_code", 2),
        ("populate_llvm_ir", 2),
        ("verify_llvm_ir", 2),
        ("optimize_llvm_ir{auto_vectorize=true}", 2),
        ("emit_llvm_object", 2),
        ("link_llvm_object", 2),
        ("lower_cranelift_ir", 1),
        ("inline_cranelift_ir", 1),
        ("emit_cranelift_code", 1),
        ("finalize_cranelift_code", 1),
        ("render_text_ir", 1),
    ] {
        let close = format!("{stage}: splice::compile: close time.busy=");
        let count = log.lines().filter(|line| line.contains(&close)).count();
        assert_eq!(count, expected_count, "{stage}\n{log}");
    }
}
