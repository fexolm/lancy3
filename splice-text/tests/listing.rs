use splice::{Codegen, hir::FunctionBuilder, kernel};
use splice_text::Text;
use std::sync::Arc;

#[kernel]
fn stop_early(#[comptime] bail: bool, x: i32) -> i32 {
    if bail {
        return 17;
    }
    if x < 0 {
        return 23;
    }
    x.wrapping_add(42)
}

#[kernel]
fn unroll_until(#[comptime] steps: &[i32], mut x: i32) -> i32 {
    for step in steps {
        if *step == 0 {
            continue;
        }
        if *step == 9 {
            break;
        }
        x += *step;
    }
    x
}

#[test]
fn rust_source_prunes_host_transfers_but_keeps_runtime_control() -> splice::Result<()> {
    let early = stop_early::print(true)?;
    assert!(early.contains("return 17"));
    assert!(!early.contains("23"));
    assert!(!early.contains("42"));
    let normal = stop_early::print(false)?;
    assert!(!normal.contains("17"));
    assert!(normal.contains("if x < 0"));
    assert!(normal.contains("return 23"));
    assert!(normal.contains("wrapping_add(42)"));
    let unrolled = unroll_until::print(&[7, 0, 8, 9, 111])?;
    assert!(unrolled.contains("x += 7"));
    assert!(unrolled.contains("x += 8"));
    assert!(!unrolled.contains("111"));
    assert!(!unrolled.contains("match block"));
    Ok(())
}

#[derive(splice::ComptimeKey)]
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
fn only_the_chosen_comptime_arm_is_printed() -> splice::Result<()> {
    fn specialize<B: Codegen>(step: &Step) -> splice::Result<B::Output> {
        apply::build::<B>(step)
    }
    let scale: String = specialize::<Text>(&Step::Scale(13))?;
    assert_eq!(
        scale,
        "fn apply(x: i32) -> i32 {\n    %r2 = 13\n    %r3 = x * %r2\n    %r1 = %r3\n    goto L0\nL0:\n    return %r1\n}\n"
    );
    let offset: String = apply::build::<Text>(&Step::Offset(-9))?;
    assert!(offset.contains(" = -9\n"));
    assert!(offset.contains(" + "));
    assert!(!offset.contains(" * "));
    Ok(())
}

#[test]
fn unrolled_calls_reuse_equal_steps_and_keep_the_runtime_loop() -> splice::Result<()> {
    let steps = [Step::Scale(3), Step::Offset(7), Step::Scale(3)];
    let listing: String = adjust::build::<Text>(&steps)?;
    assert_eq!(listing, Text::compile(&adjust::ir(&steps)?)?);
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
    assert_eq!(listing.matches(" = apply.1(").count(), 2);
    Ok(())
}

#[test]
fn shared_callable_has_one_definition_and_two_uses() -> splice::Result<()> {
    let mut leaf = FunctionBuilder::new(1);
    leaf.ret(leaf.arg(0));
    let leaf = Arc::new(leaf.finish()?);
    let mut root = FunctionBuilder::new(1);
    let a = root.call_shared(leaf.clone(), &[root.arg(0)]);
    let b = root.call_shared(leaf, &[a]);
    root.ret(b);
    let listing = Text::compile(&root.finish()?)?;
    assert!(listing.starts_with("fn entry(%r0: i32)"));
    assert_eq!(listing.matches("fn function.1(").count(), 1);
    assert_eq!(listing.matches(" = function.1(").count(), 2);
    Ok(())
}

#[test]
fn build_propagates_the_original_staging_error() {
    let result = Text::build(0, &[], Box::new(|_| Err("invalid specialization".into())));
    assert_eq!(result, Err("invalid specialization".into()));
}
