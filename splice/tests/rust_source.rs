//! `kernel::print` renders the specialized kernel as Rust; `kernel::ir`
//! keeps the same specialization as portable IR.
use splice::frontend::{KernelBuilder, Signature};
use splice::hir::{Binary, Constant, FunctionBuilder, Instr, ScalarType};
use splice::{BuildError, Codegen, Function, kernel};
use std::sync::Arc;

/// This machine's `SPLICE_SIMD` target in the mode of this test build.
fn options() -> splice::BuildOptions {
    splice::BuildOptions::host(splice::Mode::for_profile(cfg!(debug_assertions))).unwrap()
}

fn has(function: &Function, pattern: impl FnMut(&Instr) -> bool) -> bool {
    splice_testkit::instructions(function).any(pattern)
}

fn has_constant(function: &Function, value: i32) -> bool {
    has(
        function,
        |instruction| matches!(instruction, Instr::Const(_, constant) if *constant == Constant::from(value)),
    )
}

fn has_binary(function: &Function, op: Binary) -> bool {
    has(
        function,
        |instruction| matches!(instruction, Instr::Bin(_, o, ..) if *o == op),
    )
}

/// Calls to `callee` from `function` and the bodies it reaches.
fn calls_to(function: &Function, callee: &Arc<Function>) -> usize {
    splice_testkit::instructions(function)
        .filter(|instruction| matches!(instruction, Instr::Call { function, .. } if Arc::ptr_eq(function, callee)))
        .count()
}

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

#[kernel]
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

#[kernel]
fn print_division(#[comptime] d: i32, x: i32) -> i32 {
    if x > 0 {
        let q = 100 / d;
        x + q
    } else {
        9
    }
}

#[kernel]
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
    pub use splice::simd::Lane;
}

#[kernel]
fn identifier_in_path(x: i32) -> i32 {
    __splice_source_child_0::Lane::<i32>::splat(x).last()
}

#[test]
fn printer_and_builder_specialize_host_iterations() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(host_iterations(3, 5), 12);
    assert!(host_iterations::print(3)?.contains("x + 7"));
    assert!(has_constant(&*host_iterations::ir(3)?, 7));
    Ok(())
}

#[test]
fn division_under_runtime_control_faults_only_in_the_selected_arm()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(print_division(0, 0), 9);
    let text = print_division::print(0)?;
    assert!(text.contains("if x > 0"));
    assert!(text.contains("panic!") && text.contains("else") && text.contains('9'));
    assert!(has_binary(&*print_division::ir(0)?, Binary::Div));
    assert!(print_division::print(4)?.contains("x + 25"));
    let remainder = print_remainder::print(i32::MIN, -1)?;
    let (prefix, _) = remainder.split_once("signed division overflow").unwrap();
    assert!(prefix.contains("marks [0] = x"));
    assert!(has_binary(
        &*print_remainder::ir(i32::MIN, -1)?,
        Binary::Rem
    ));
    Ok(())
}

#[test]
fn source_substitutions_leave_other_identifiers_intact() -> Result<(), Box<dyn std::error::Error>> {
    assert!(identifier_in_path::print()?.contains("__splice_source_child_0"));
    assert!(has(&*identifier_in_path::ir()?, |instruction| matches!(
        instruction,
        Instr::Return(_)
    )));
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
fn only_the_chosen_comptime_arm_is_staged() -> Result<(), Box<dyn std::error::Error>> {
    let scale = apply::ir(&Step::Scale(13))?;
    assert!(has_constant(&scale, 13));
    assert!(has_binary(&scale, Binary::Mul) && !has_binary(&scale, Binary::Add));
    let offset = apply::ir(&Step::Offset(-9))?;
    assert!(has_constant(&offset, -9));
    assert!(has_binary(&offset, Binary::Add) && !has_binary(&offset, Binary::Mul));
    Ok(())
}

#[test]
fn unrolled_calls_reuse_equal_steps_and_keep_the_runtime_loop()
-> Result<(), Box<dyn std::error::Error>> {
    let steps = [Step::Scale(3), Step::Offset(7), Step::Scale(3)];
    let ir = adjust::ir(&steps)?;
    for pattern in [
        |i: &Instr| matches!(i, Instr::Branch { .. }),
        |i: &Instr| matches!(i, Instr::Load { .. }),
        |i: &Instr| matches!(i, Instr::BufferLen { .. }),
    ] {
        assert!(ir.instructions().iter().any(pattern));
    }
    let bodies = ir.reachable_functions();
    let [_, scale, offset] = bodies[..] else {
        panic!("expected the root and two specializations of apply, got {bodies:?}");
    };
    assert!(has_binary(scale, Binary::Mul) && has_binary(offset, Binary::Add));
    let callee = |body: &Function| {
        ir.instructions()
            .iter()
            .find_map(|i| match i {
                Instr::Call { function, .. } if std::ptr::eq(&**function, body) => {
                    Some(Arc::clone(function))
                }
                _ => None,
            })
            .unwrap()
    };
    assert_eq!(calls_to(&ir, &callee(scale)), 2);
    assert_eq!(calls_to(&ir, &callee(offset)), 1);
    Ok(())
}

#[test]
fn shared_callable_has_one_definition_and_two_uses() -> splice::Result<()> {
    let scalar = Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]);
    let mut leaf = FunctionBuilder::new(options(), scalar);
    leaf.ret(leaf.arg(0)?)?;
    let leaf = Arc::new(leaf.finish()?);
    let mut root = FunctionBuilder::new(options(), scalar);
    let a = root.call_function(&leaf, &[root.arg(0)?], &[])?;
    let b = root.call_function(&leaf, &[a], &[])?;
    root.ret(b)?;
    let root = root.finish()?;
    assert_eq!(root.reachable_functions().len(), 2);
    assert_eq!(calls_to(&root, &leaf), 2);
    Ok(())
}

#[test]
fn build_propagates_the_original_staging_error() {
    fn check<B: Codegen>() {
        let result = B::build_with_options(
            Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]),
            Box::new(|_| Err(BuildError::unsupported("invalid specialization"))),
            options(),
        );
        assert!(
            matches!(result, Err(BuildError::Unsupported(feature)) if feature == "invalid specialization")
        );
    }
    splice_testkit::each_backend!(B => check::<B>());
}
