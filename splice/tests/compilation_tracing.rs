use splice::{Codegen, Executable, kernel};
use splice_lancy::Lancy;
use splice_llvm::Llvm;
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
        let ir = increment::ir().unwrap();
        for kernel in [
            increment::build::<Lancy>().unwrap(),
            Lancy::compile(&ir).unwrap(),
        ] {
            assert_eq!(kernel.call(&[41], &mut []), Ok(42));
        }
        for kernel in [
            increment::build::<Llvm>().unwrap(),
            Llvm::compile(&ir).unwrap(),
        ] {
            assert_eq!(kernel.call(&[41], &mut []), Ok(42));
        }
    });
    let bytes = output.lock().unwrap();
    let log = std::str::from_utf8(&bytes).unwrap();
    for (stage, expected_count) in [
        ("stage_lancy_hir", 1),
        ("lower_lancy_hir", 1),
        ("select_instructions", 2),
        ("allocate_registers", 2),
        ("emit_machine_code", 2),
        ("install_code", 2),
        ("populate_llvm_ir", 2),
        ("optimize_llvm_ir", 2),
        ("emit_llvm_object", 2),
        ("link_llvm_object", 2),
    ] {
        let count = log
            .lines()
            .filter(|line| line.contains(&format!("{stage}:")) && line.contains("time.busy="))
            .count();
        assert_eq!(count, expected_count, "{stage}\n{log}");
    }
}
