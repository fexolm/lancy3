use splice::{BuildOptions, Codegen, kernel};
use splice_text::Text;
use std::collections::BTreeSet;
use std::fmt::{Debug, Write};
use std::sync::{Arc, Mutex};
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

#[kernel]
fn increment(value: i32) -> i32 {
    value.wrapping_add(1)
}

/// A span's name with its fields, such as `compile{backend="lancy"}`.
struct Label(String);

struct Fields(String);

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        let separator = if self.0.is_empty() { "" } else { "," };
        write!(self.0, "{separator}{}={value:?}", field.name()).unwrap();
    }
}

/// Records every closed span as its path of labels from the root, so the
/// test checks which stages run inside which compilation, not log text.
#[derive(Clone, Default)]
struct Closed(Arc<Mutex<Vec<Vec<String>>>>);

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Closed {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut fields = Fields(String::new());
        attrs.record(&mut fields);
        let name = attrs.metadata().name();
        let label = if fields.0.is_empty() {
            name.to_owned()
        } else {
            format!("{name}{{{}}}", fields.0)
        };
        ctx.span(id).unwrap().extensions_mut().insert(Label(label));
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).unwrap();
        let path = span
            .scope()
            .from_root()
            .map(|span| span.extensions().get::<Label>().unwrap().0.clone())
            .collect();
        self.0.lock().unwrap().push(path);
    }
}

#[test]
fn staged_and_portable_compilation_report_their_nested_stages() {
    let closed = Closed::default();
    let subscriber = tracing_subscriber::registry().with(closed.clone());
    tracing::subscriber::with_default(subscriber, || {
        let options = BuildOptions::Release;
        let ir = increment::ir_with_options(options).unwrap();
        splice_testkit::each_backend!(B => {
            for kernel in [
                increment::build_with_options::<B>(options, increment::BuildArgs {}).unwrap(),
                B::compile(&ir).unwrap(),
            ] {
                assert_eq!(kernel.call(increment::CallArgs { value: 41 }), Ok(42));
            }
        });
        assert!(Text::compile(&ir).unwrap().starts_with("fn increment("));
    });
    let closed: BTreeSet<String> = closed
        .0
        .lock()
        .unwrap()
        .iter()
        .map(|p| p.join(" / "))
        .collect();
    let mut expected = vec!["materialize_ir".to_owned(), "render_text_ir".to_owned()];
    let jit = "lancy_jit{inlining=Always,debug=false}";
    for (input, frontend) in [
        ("staged", "stage_lancy_hir"),
        ("portable", "lower_lancy_hir"),
    ] {
        let root = format!("compile{{backend=\"lancy\",input=\"{input}\"}}");
        expected.push(format!("{root} / {frontend}"));
        for stage in ["verify_lancy_hir", "install_code"] {
            expected.push(format!("{root} / {jit} / {stage}"));
        }
        for stage in [
            "verify_lancy_hir",
            "optimize_lancy_hir",
            "select_instructions",
            "schedule_instructions",
            "allocate_registers",
            "emit_machine_code",
        ] {
            expected.push(format!(
                "{root} / {jit} / optimize_and_lower_lancy / {stage}"
            ));
        }
        let root = format!("compile{{backend=\"llvm\",input=\"{input}\"}}");
        for stage in [
            "populate_llvm_ir",
            "verify_llvm_ir",
            "optimize_llvm_ir{auto_vectorize=true}",
            "optimize_llvm_ir{auto_vectorize=false}",
            "emit_llvm_object",
            "link_llvm_object",
        ]
        .into_iter()
        .filter(|_| cfg!(feature = "llvm"))
        {
            expected.push(format!("{root} / {stage}"));
        }
    }
    for stage in [
        "lower_cranelift_ir",
        "inline_cranelift_ir",
        "emit_cranelift_code",
        "finalize_cranelift_code",
    ] {
        expected.push(format!(
            "compile{{backend=\"cranelift\",input=\"portable\"}} / {stage}"
        ));
    }
    let listing = closed.iter().cloned().collect::<Vec<_>>().join("\n");
    for path in expected {
        assert!(closed.contains(&path), "missing {path} in\n{listing}");
    }
}
