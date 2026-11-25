use splice_core::frontend::{KernelBuilder, Signature};
use splice_core::hir::ScalarType;
use splice_core::{Codegen, hir::FunctionBuilder, kernel};
use splice_text::Text;
use std::sync::Arc;

/// This machine's `SPLICE_SIMD` target in the mode of this test build.
fn options() -> splice_core::BuildOptions {
    splice_core::BuildOptions::host(splice_core::Mode::for_profile(cfg!(debug_assertions))).unwrap()
}

#[kernel(crate = splice_core)]
fn stop_early(#[comptime] bail: bool, x: i32) -> i32 {
    if bail {
        return 17;
    }
    if x < 0 {
        return 23;
    }
    x.wrapping_add(42)
}

#[kernel(crate = splice_core)]
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
fn rust_source_prunes_host_transfers_but_keeps_runtime_control() -> splice_core::Result<()> {
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

#[kernel(crate = splice_core)]
fn host_iterations(#[comptime] n: i32, x: i32) -> i32 {
    let mut k = 0;
    while k < n {
        k += 1;
    }
    for _i in 0..n {
        k += 1;
    }
    loop {
        k += 1;
        if k > n * 2 {
            break;
        }
    }
    x + k
}

#[kernel(crate = splice_core)]
fn print_division(#[comptime] d: i32, x: i32) -> i32 {
    if x > 0 {
        let q = 100 / d;
        x + q
    } else {
        9
    }
}

#[kernel(crate = splice_core)]
fn print_remainder(#[comptime] a: i32, #[comptime] d: i32, marks: &mut [i32], x: i32) -> i32 {
    match x {
        0 => 9,
        _ => {
            ({
                marks[0] = x;
                x
            }) + (a % d).wrapping_sub(0)
        }
    }
}

mod __splice_source_child_0 {
    pub use splice_core::simd::Lane;
}

#[kernel(crate = splice_core)]
fn identifier_in_path(x: i32) -> i32 {
    __splice_source_child_0::Lane::<i32>::splat(x).last()
}

#[test]
fn printer_and_builder_specialize_host_iterations() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(host_iterations(3, 5), 12);
    assert!(host_iterations::print(3)?.contains("x + 7"));
    assert!(
        host_iterations::build::<Text>(host_iterations::BuildArgs { n: 3 })?.contains(" = 7i32\n")
    );
    Ok(())
}

#[test]
fn division_under_runtime_control_faults_only_in_the_selected_arm()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(print_division(0, 0), 9);
    let text = print_division::print(0)?;
    assert!(text.contains("if x > 0"));
    assert!(text.contains("panic!") && text.contains("else") && text.contains('9'));
    assert!(print_division::build::<Text>(print_division::BuildArgs { d: 0 })?.contains(" / "));
    assert!(print_division::print(4)?.contains("x + 25"));
    let remainder = print_remainder::print(i32::MIN, -1)?;
    let (prefix, _) = remainder.split_once("signed division overflow").unwrap();
    assert!(prefix.contains("marks [0] = x"));
    assert!(
        print_remainder::build::<Text>(print_remainder::BuildArgs { a: i32::MIN, d: -1 })?
            .contains(" % ")
    );
    Ok(())
}

#[test]
fn source_substitutions_leave_other_identifiers_intact() -> Result<(), Box<dyn std::error::Error>> {
    assert!(identifier_in_path::print()?.contains("__splice_source_child_0"));
    assert!(
        identifier_in_path::build::<Text>(identifier_in_path::BuildArgs {})?.contains("return")
    );
    Ok(())
}

#[derive(splice_core::ComptimeKey)]
#[splice(crate = splice_core)]
enum Step {
    Scale(i32),
    Offset(i32),
}

#[kernel(crate = splice_core)]
fn apply(#[comptime] step: &Step, x: i32) -> i32 {
    match step {
        Step::Scale(factor) => x.wrapping_mul(*factor),
        Step::Offset(delta) => x.wrapping_add(*delta),
    }
}

#[kernel(crate = splice_core)]
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
fn only_the_chosen_comptime_arm_is_printed() -> Result<(), Box<dyn std::error::Error>> {
    fn specialize<B: Codegen>(step: &Step) -> Result<B::Output, Box<dyn std::error::Error>> {
        Ok(apply::build::<B>(apply::BuildArgs { step })?.into_inner())
    }
    let scale: String = specialize::<Text>(&Step::Scale(13))?;
    assert_eq!(
        scale,
        "fn apply(x: i32) -> i32 {\n    %r2 = 13i32\n    %r3 = x * %r2\n    %r1 = %r3\n    goto L0\nL0:\n    return %r1\n}\n"
    );
    let offset = apply::build::<Text>(apply::BuildArgs {
        step: &Step::Offset(-9),
    })?;
    assert!(offset.contains(" = -9i32\n"));
    assert!(offset.contains(" + "));
    assert!(!offset.contains(" * "));
    Ok(())
}

#[test]
fn unrolled_calls_reuse_equal_steps_and_keep_the_runtime_loop()
-> Result<(), Box<dyn std::error::Error>> {
    let steps = [Step::Scale(3), Step::Offset(7), Step::Scale(3)];
    let listing = adjust::build::<Text>(adjust::BuildArgs { steps: &steps })?;
    assert_eq!(
        listing.as_inner(),
        Text::compile(&adjust::ir(&steps)?)?.as_inner()
    );
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
fn shared_callable_has_one_definition_and_two_uses() -> splice_core::Result<()> {
    let scalar = Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]);
    let mut leaf = FunctionBuilder::new(options(), scalar);
    leaf.ret(leaf.arg(0)?)?;
    let leaf = Arc::new(leaf.finish()?);
    let mut root = FunctionBuilder::new(options(), scalar);
    let a = root.call_function(&leaf, &[root.arg(0)?], &[])?;
    let b = root.call_function(&leaf, &[a], &[])?;
    root.ret(b)?;
    let listing = Text::compile(&root.finish()?)?;
    assert!(listing.starts_with("fn entry(%r0: i32)"));
    assert_eq!(listing.matches("fn function.1(").count(), 1);
    assert_eq!(listing.matches(" = function.1(").count(), 2);
    Ok(())
}

#[test]
fn build_propagates_the_original_staging_error() {
    let result = Text::build_with_options(
        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]),
        Box::new(|_| {
            Err(splice_core::BuildError::unsupported(
                "invalid specialization",
            ))
        }),
        options(),
    );
    assert!(
        matches!(result, Err(splice_core::BuildError::Unsupported(feature)) if feature == "invalid specialization")
    );
}
