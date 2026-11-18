use splice::{Codegen, hir::FunctionBuilder, kernel};
use splice_text::Text;
use std::sync::Arc;

enum Step {
    Scale(i32),
    Offset(i32),
}

#[kernel]
fn apply(#[comptime] step: &Step, x: i32) -> i32 {
    match step {
        Step::Scale(factor) => x.wrapping_mul(*factor),
        Step::Offset(delta) => x.wrapping_add(*delta),
    }
}

#[kernel]
fn adjust(#[comptime] steps: &[Step], data: &mut [i32]) -> i32 {
    for i in 0..data.len() as i32 {
        let mut x = data[i as usize];
        for step in steps {
            x = apply(step, x);
        }
        data[i as usize] = x;
    }
    data.len() as i32
}

#[test]
fn only_the_chosen_comptime_arm_is_printed() {
    fn specialize<B: Codegen>(step: &Step) -> B::Output {
        apply::build::<B>(step)
    }
    let scale: String = specialize::<Text>(&Step::Scale(13));
    assert_eq!(
        scale,
        "fn apply(x: i32) -> i32 {\n    %r2 = 13\n    %r3 = x * %r2\n    %r1 = %r3\n    goto L0\nL0:\n    return %r1\n}\n"
    );
    let offset: String = apply::build::<Text>(&Step::Offset(-9));
    assert!(offset.contains(" = -9\n"));
    assert!(offset.contains(" + "));
    assert!(!offset.contains(" * "));
}

#[test]
fn unrolled_calls_have_distinct_names_and_runtime_loop_is_visible() {
    let steps = [Step::Scale(3), Step::Offset(7)];
    let listing: String = adjust::build::<Text>(&steps);
    assert_eq!(listing, Text::compile(&adjust::ir(&steps).unwrap()));
    assert!(listing.starts_with("fn adjust(data: &mut [i32]) -> i32"));
    assert!(listing.contains("if %r"));
    assert!(listing.contains("data[%r"));
    assert!(listing.contains("data.len()"));
    for (id, op) in [(1, " * "), (2, " + ")] {
        let symbol = format!("apply.{id}");
        assert!(listing.contains(&format!(" = {symbol}(")));
        let body = listing
            .split_once(&format!("fn {symbol}(x: i32)"))
            .unwrap()
            .1;
        assert!(body.split_once('}').unwrap().0.contains(op));
    }
    assert_eq!(listing.matches("fn ").count(), 3);
}

#[test]
fn shared_callable_has_one_definition_and_two_uses() {
    let mut leaf = FunctionBuilder::new(1);
    leaf.ret(leaf.arg(0));
    let leaf = Arc::new(leaf.finish().unwrap());
    let mut root = FunctionBuilder::new(1);
    let a = root.call_shared(leaf.clone(), &[root.arg(0)]);
    let b = root.call_shared(leaf, &[a]);
    root.ret(b);
    let listing = Text::compile(&root.finish().unwrap());
    assert!(listing.starts_with("fn entry(%r0: i32)"));
    assert_eq!(listing.matches("fn function.1(").count(), 1);
    assert_eq!(listing.matches(" = function.1(").count(), 2);
}

#[test]
fn staging_error_produces_a_diagnostic_instead_of_a_partial_listing() {
    let text: String = Text::build(0, &[], Box::new(|_| Err("invalid specialization".into())));
    assert_eq!(
        text,
        "// Splice specialization failed: invalid specialization\n"
    );
}
