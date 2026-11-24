use splice::{
    BuildOptions, CallError, Codegen, Executable, Mode, Trap, debug, hir::Instr, kernel, simd::Lane,
};

#[kernel]
fn update(dst: &mut [i32], index: i32, value: i32) -> i32 {
    dst[index as usize] = value;
    value
}

#[kernel]
fn test_kernel(dst: &mut [i32], bias: i32) -> i32 {
    let step = 1;
    let lanes = Lane::<i32>::splat(bias);
    let mut result = 0;
    for i in 0..2 {
        let next = lanes.extract_wrapping(0) + i;
        let result = update(dst, i, next);
        dst[i as usize] = result + step;
    }
    result += bias;
    result
}

#[kernel]
fn pattern_scopes(#[comptime] plan: Option<i32>, flag: i32) -> i32 {
    use Option::None as empty;
    match plan {
        Some(amount @ 9) => match flag {
            0 if flag + 1 == 1 => flag + amount,
            _ => flag + 2,
        },
        Some(_) => flag,
        empty => flag,
    }
}

#[kernel]
fn irrefutable_pattern(#[comptime] plan: Option<std::convert::Infallible>, flag: i32) -> i32 {
    use Option::None as empty;
    let empty = plan;
    flag
}

fn exercise<B: Codegen<Output: Executable>>() {
    let debug = BuildOptions::host(Mode::Debug).unwrap();
    let ir = test_kernel::ir_with_options(debug).unwrap();
    for kernel in [
        test_kernel::build_with_options::<B>(debug, test_kernel::BuildArgs {}).unwrap(),
        B::compile(&ir).unwrap(),
        B::compile_with_options(&ir, BuildOptions::host(Mode::Release).unwrap()).unwrap(),
    ] {
        let mut buf = [0, 0];
        assert_eq!(
            kernel.call(test_kernel::CallArgs {
                bias: 12,
                dst: &mut buf
            }),
            Ok(test_kernel(&mut [0, 0], 12))
        );
        assert_eq!(buf, [13, 14]);
        let mut short = [0];
        assert_eq!(
            kernel
                .call(test_kernel::CallArgs {
                    bias: 12,
                    dst: &mut short
                })
                .unwrap_err(),
            CallError::Trap(Trap::Bounds),
        );
        assert_eq!(short, [13]);
    }
}

#[test]
fn debug_and_release_follow_the_same_execution_contract() {
    splice_testkit::each_backend!(B => exercise::<B>());
}

#[test]
fn staging_preserves_names_types_and_locations_in_debug_only() {
    let default_ir = test_kernel::ir().unwrap();
    assert_eq!(
        default_ir.options(),
        BuildOptions::host(Mode::for_profile(cfg!(debug_assertions))).unwrap()
    );
    let ir = test_kernel::ir_with_options(BuildOptions::host(Mode::Debug).unwrap()).unwrap();
    let points: Vec<_> = ir
        .instructions()
        .iter()
        .filter_map(|inst| match inst {
            Instr::Debug(point) => Some(point),
            _ => None,
        })
        .collect();
    assert!(
        points
            .iter()
            .all(|p| p.location.file.ends_with("debug_symbols.rs") && p.location.line > 10)
    );
    assert!(points.iter().any(|p| {
        p.slots(&ir, ir.options().target.vector_bytes())
            .any(|(var, ty, _)| var.name == "lanes" && matches!(ty, debug::Type::Vector { .. }))
    }));
    assert!(
        points
            .iter()
            .any(|p| p.variables.iter().any(|v| v.name == "i"))
    );
    assert!(
        !points
            .last()
            .unwrap()
            .variables
            .iter()
            .any(|v| v.name == "i" || v.name == "next")
    );
    assert!(points.iter().any(|p| p.variables.iter().any(|v| {
        v.name == "step" && matches!(v.value, debug::Value::Constant(Some(value)) if value.parts() == (debug::ConstantType::Signed(4), 1))
    })));
    let release = test_kernel::ir_with_options(BuildOptions::host(Mode::Release).unwrap()).unwrap();
    assert!(
        release
            .instructions()
            .iter()
            .all(|inst| !matches!(inst, Instr::Debug(_)))
    );
}

#[test]
fn selected_match_patterns_and_guards_have_their_own_source_points() {
    irrefutable_pattern::ir_with_options(BuildOptions::host(Mode::Debug).unwrap(), None).unwrap();
    let ir =
        pattern_scopes::ir_with_options(BuildOptions::host(Mode::Debug).unwrap(), Some(9)).unwrap();
    let locations: std::collections::BTreeSet<_> = ir.instructions().iter().filter_map(|inst| {
        let Instr::Debug(point) = inst else { return None; };
        point.variables.iter().any(|var| {
            var.name == "amount" && matches!(var.value, debug::Value::Constant(Some(value)) if value.parts() == (debug::ConstantType::Signed(4), 9))
        }).then_some((point.location.line, point.location.column))
    }).collect();
    // The selected outer arm, guard and body of the first runtime arm, and
    // the fallback body each have a distinct source position.
    assert_eq!(locations.len(), 4);
}

/// `value` is assigned only when `flag` is nonzero, but a source point after
/// the join names it. Debug reads are not uses, so this is valid in debug
/// mode exactly as in release mode (CORE-18).
fn partially_assigned(b: &mut dyn splice::frontend::KernelBuilder) -> splice::Result<()> {
    observe_partially_assigned(b, false)
}

/// Returning `value` is a real use, which stays an error.
fn partially_returned(b: &mut dyn splice::frontend::KernelBuilder) -> splice::Result<()> {
    observe_partially_assigned(b, true)
}

fn observe_partially_assigned(
    b: &mut dyn splice::frontend::KernelBuilder,
    use_value: bool,
) -> splice::Result<()> {
    let flag = b.arg(0)?;
    let value = b.local(ValueType::Scalar)?;
    let (assign, join) = (b.label()?, b.label()?);
    b.branch(flag, assign, join)?;
    b.mark(assign)?;
    let five = b.constant(5)?;
    b.assign(value, five)?;
    b.jump(join)?;
    b.mark(join)?;
    let location = debug::Location {
        file: file!(),
        directory: env!("CARGO_MANIFEST_DIR"),
        line: line!(),
        column: 1,
    };
    let variables = [debug::Variable {
        name: "value",
        value: debug::Value::Register(value, debug::Scalar::I32),
    }];
    b.debug_point(location, &variables);
    b.ret(if use_value { value } else { flag })?;
    Ok(())
}

fn exercise_partial<B: Codegen<Output: Executable>>() {
    let options = BuildOptions::host(Mode::Debug).unwrap();
    let ir = splice::frontend::materialize(1, &[], Box::new(partially_assigned), options).unwrap();
    assert!(
        ir.instructions()
            .iter()
            .any(|op| matches!(op, Instr::Debug(_)))
    );
    for kernel in [
        B::build_with_options(1, &[], Box::new(partially_assigned), options).unwrap(),
        B::compile(&ir).unwrap(),
    ] {
        for flag in [0, 3] {
            assert_eq!(kernel.call(&[flag], &mut []), Ok(flag));
        }
    }
    for options in [
        BuildOptions::host(Mode::Debug).unwrap(),
        BuildOptions::host(Mode::Release).unwrap(),
    ] {
        assert!(B::build_with_options(1, &[], Box::new(partially_returned), options).is_err());
    }
}

#[test]
fn source_points_may_name_partially_assigned_variables() {
    splice_testkit::each_backend!(B => exercise_partial::<B>());
}

#[kernel]
fn flagged(x: i32) -> i32 {
    let positive = x > 0;
    let doubled = x * 2; // gdb: positive is assigned
    if positive { doubled } else { x }
}

/// A runtime `bool` reads as a Rust `bool` in GDB on every native backend
/// (CORE-16). Needs `gdb`; run with `--ignored`.
#[test]
#[ignore = "requires gdb"]
fn gdb_shows_runtime_bools_as_rust_bools() {
    const CHILD: &str = "SPLICE_GDB_BOOL_CHILD";
    if std::env::var_os(CHILD).is_some() {
        fn run<B: Codegen<Output: Executable>>() {
            let options = BuildOptions::host(Mode::Debug).unwrap();
            let code = flagged::build_with_options::<B>(options, flagged::BuildArgs {}).unwrap();
            assert_eq!(code.call(flagged::CallArgs { x: 4 }), Ok(8));
        }
        run::<splice_lancy::Lancy>();
        run::<splice_cranelift::Cranelift>();
        #[cfg(feature = "llvm")]
        run::<splice_llvm::Llvm>();
        return;
    }
    let backends = if cfg!(feature = "llvm") { 3 } else { 2 };
    let source = include_str!("debug_symbols.rs");
    let line = 1 + source
        .lines()
        .position(|l| l.ends_with("// gdb: positive is assigned"))
        .unwrap();
    let output = std::process::Command::new("gdb")
        .args(["-q", "-batch", "-ex", "set breakpoint pending on"])
        .args([
            "-ex",
            &format!("break debug_symbols.rs:{line}"),
            "-ex",
            "run",
        ])
        .args(["-ex", "print positive", "-ex", "continue"].repeat(backends))
        .arg("--args")
        .arg(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "gdb_shows_runtime_bools_as_rust_bools",
            "--ignored",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let values: Vec<_> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix('$')?.split_once(" = ").map(|(_, v)| v))
        .collect();
    assert_eq!(values, vec!["true"; backends], "{stdout}");
}
