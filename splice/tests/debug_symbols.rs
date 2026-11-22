use splice::{BuildOptions, Codegen, Executable, debug, hir::Instr, kernel, simd::Lane};

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
    let debug = BuildOptions::Debug;
    let ir = test_kernel::ir_with_options(debug).unwrap();
    for kernel in [
        test_kernel::build_with_options::<B>(debug, test_kernel::BuildArgs {}).unwrap(),
        B::compile(&ir).unwrap(),
        B::compile_with_options(&ir, BuildOptions::Release).unwrap(),
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
                .unwrap_err()
                .to_string(),
            "kernel slice index out of bounds",
        );
        assert_eq!(short, [13]);
    }
}

#[test]
fn debug_and_release_follow_the_same_execution_contract() {
    exercise::<splice_lancy::Lancy>();
    exercise::<splice_cranelift::Cranelift>();
    exercise::<splice_llvm::Llvm>();
    exercise::<splice_llvm::LlvmManualSimd>();
}

#[test]
fn staging_preserves_names_types_and_locations_in_debug_only() {
    let default_ir = test_kernel::ir().unwrap();
    assert_eq!(
        default_ir.options(),
        BuildOptions::for_profile(cfg!(debug_assertions))
    );
    let ir = test_kernel::ir_with_options(BuildOptions::Debug).unwrap();
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
        p.slots(&ir)
            .any(|(var, ty, _)| var.name == "lanes" && matches!(ty, debug::Type::Vector(_)))
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
    let release = test_kernel::ir_with_options(BuildOptions::Release).unwrap();
    assert!(
        release
            .instructions()
            .iter()
            .all(|inst| !matches!(inst, Instr::Debug(_)))
    );
}

#[test]
fn selected_match_patterns_and_guards_have_their_own_source_points() {
    irrefutable_pattern::ir_with_options(BuildOptions::Debug, None).unwrap();
    let ir = pattern_scopes::ir_with_options(BuildOptions::Debug, Some(9)).unwrap();
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
