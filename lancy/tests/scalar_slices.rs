use lancy::hir::{Buffer, BufferArg, BufferKind, Instr};
use lancy::{MappedKernel, ScalarFn, ScalarKernel, kernel};

#[kernel]
fn append_score(out: &mut [i32], #[comptime] factor: i32, values: &[i32], row: i32) -> i32 {
    let score = values[row as usize]
        .wrapping_mul(factor)
        .min(90)
        .wrapping_add(1)
        .max(-20);
    out[row as usize] += score;
    1
}

#[kernel]
fn lookup(input: &[i32], row: i32) -> i32 {
    let index = row as usize;
    if !input.is_empty() && index < input.len() {
        input[index]
    } else {
        71
    }
}

#[test]
fn negative_indices_compare_as_usize_before_loading() {
    let f = lookup::build().unwrap();
    let jit = ScalarKernel::compile(&f).unwrap();
    for input in [&[][..], &[7, 8][..]] {
        for row in [-1, i32::MIN, 0, 1, 2] {
            let mut args = [BufferArg::Read(input)];
            assert_eq!(
                jit.call_with_buffers(&[row], &mut args).unwrap(),
                lookup(input, row)
            );
        }
    }
}

#[kernel]
fn score_column(values: &[i32], out: &mut [i32], #[comptime] factor: i32) -> i32 {
    let mut count = 0;
    for row in 0..values.len() {
        count += append_score(out, factor, values, row as i32);
    }
    count
}

#[test]
fn slice_calls_remap_arguments_and_keep_row_updates() {
    let values = [i32::MAX, -13, 3, 100, i32::MIN];
    let mut expected = [2; 5];
    let expected_count = score_column(&values, &mut expected, 2);
    let f = score_column::build(2).unwrap();
    assert_eq!(f.buffer_kinds(), &[BufferKind::Read, BufferKind::Write]);
    assert!(f.instructions().iter().any(|i| matches!(i,
        Instr::Call { buffers, .. } if buffers == &[Buffer(1), Buffer(0)]
    )));
    for native in [ScalarKernel::baseline, ScalarKernel::compile] {
        let k = native(&f).unwrap();
        let mut out = [2; 5];
        let mut args = [BufferArg::Read(&values), BufferArg::Write(&mut out)];
        assert_eq!(k.call_with_buffers(&[], &mut args).unwrap(), expected_count);
        assert_eq!(out, expected);
    }
}

#[test]
fn entry_points_check_buffer_signatures_before_executing() {
    let f = score_column::build(3).unwrap();
    assert!(ScalarFn::<0>::baseline(&f).is_err());
    assert!(MappedKernel::compile(&f).is_err());
    let kernel = ScalarKernel::compile(&f).unwrap();
    let mut caller = kernel.caller();
    assert!(kernel.call(&[]).is_err());
    assert!(caller.call(&[]).is_err());
    assert!(caller.call_with_buffers(&[], &mut []).is_err());
    let mut out = [8];
    assert!(
        caller
            .call_with_buffers(&[], &mut [BufferArg::Read(&[5]), BufferArg::Read(&out)])
            .is_err()
    );
    assert_eq!(
        caller
            .call_with_buffers(
                &[],
                &mut [BufferArg::Read(&[5]), BufferArg::Write(&mut out)]
            )
            .unwrap(),
        1
    );
    assert_eq!(out, [24]);
}

#[kernel]
fn assign_in_order(buffer: &mut [i32], value: i32) -> i32 {
    buffer[{
        buffer[1] = buffer[0];
        value as usize
    }] = {
        buffer[0] = 4;
        7
    };
    buffer[value as usize] += {
        buffer[value as usize] = 3;
        value
    };
    buffer[1]
}

#[test]
fn indexed_assignments_evaluate_rhs_before_place() {
    let f = assign_in_order::build().unwrap();
    let mut expected = [0; 3];
    assert_eq!(assign_in_order(&mut expected, 2), 4);
    assert_eq!(expected, [4, 4, 5]);
    let mut out = [0; 3];
    let mut buffers = [BufferArg::Write(&mut out)];
    assert_eq!(f.interpret_with_buffers(&[2], &mut buffers).unwrap(), 4);
    assert_eq!(out, expected);
}

#[kernel]
fn read_at(#[comptime] index: usize, input: &[i32]) -> i32 {
    input[index]
}

#[test]
fn host_indices_outside_register_width_stay_out_of_bounds() {
    let f = read_at::build(1usize << 32).unwrap();
    let mut buffers = [BufferArg::Read(&[12])];
    assert!(f.interpret_with_buffers(&[], &mut buffers).is_err());
    let k = ScalarKernel::compile(&f).unwrap();
    assert!(k.call_with_buffers(&[], &mut buffers).is_err());
}

#[kernel]
fn divide(mode: i32, mut value: i32, divisor: i32) -> i32 {
    match mode {
        0 => value / divisor,
        1 => value % divisor,
        2 => {
            value /= divisor;
            value
        }
        3 => {
            value %= divisor;
            value
        }
        _ => 19,
    }
}

#[test]
fn division_errors_follow_runtime_control_flow() {
    let f = divide::build().unwrap();
    let typed = ScalarFn::<3>::compile(&f).unwrap();
    assert!(typed.try_invoke([2, 9, 0]).is_err());
    assert_eq!(typed.try_invoke([1, -17, 5]).unwrap(), -2);
    let mapped = MappedKernel::compile(&f).unwrap();
    let mut out = [41; 3];
    assert!(
        mapped
            .run(&[&[0, 1, 0], &[9, 9, 9], &[2, 0, 3]], &mut out)
            .is_err()
    );
    assert_eq!(out, [4, 41, 41]);
    for jit in [ScalarKernel::baseline(&f), ScalarKernel::compile(&f)] {
        let jit = jit.unwrap();
        for mode in 0..4 {
            for (n, d) in [(-17, 5), (17, -5), (-3, -7), (i32::MIN, 3)] {
                let expected = divide(mode, n, d);
                assert_eq!(
                    f.interpret(&[mode, n, d]).unwrap(),
                    expected,
                    "IR mode={mode} n={n} d={d}"
                );
                assert_eq!(
                    jit.call(&[mode, n, d]).unwrap(),
                    expected,
                    "JIT mode={mode} n={n} d={d}"
                );
            }
            for (n, d) in [(23, 0), (i32::MIN, -1)] {
                assert!(jit.call(&[mode, n, d]).is_err());
                assert!(f.interpret(&[mode, n, d]).is_err());
            }
        }
        assert_eq!(jit.call(&[4, 2, 0]).unwrap(), 19);
    }
}

#[kernel]
fn scale(#[comptime] multiplier: i32, row: i32) -> i32 {
    multiplier.wrapping_mul(row)
}

#[kernel]
fn staged_divide(#[comptime] a: i32, #[comptime] n: i32, row: i32) -> i32 {
    if row == 0 {
        0
    } else if row == 1 {
        let coefficient = a / n;
        scale(coefficient, row)
    } else {
        (a % n).wrapping_sub(0)
    }
}

#[kernel]
fn staged_usize(#[comptime] offset: usize, input: &[i32], row: i32) -> i32 {
    if row == 0 { 0 } else { input[offset / 2] }
}

#[test]
fn failing_comptime_i32_arithmetic_in_a_runtime_arm_faults_only_when_selected() {
    for (a, n) in [(17, 0), (i32::MIN, -1), (17, 4)] {
        let f = staged_divide::build(a, n).unwrap();
        for jit in [ScalarKernel::baseline(&f), ScalarKernel::compile(&f)] {
            let jit = jit.unwrap();
            for row in [0, 1, 2] {
                let expected = match row {
                    0 => Some(0),
                    1 => a.checked_div(n),
                    _ => a.checked_rem(n),
                };
                assert_eq!(jit.call(&[row]).ok(), expected);
                assert_eq!(f.interpret(&[row]).ok(), expected);
            }
        }
    }
    let f = staged_usize::build(4).unwrap();
    let input = [10, 11, 12];
    let mut args = [BufferArg::Read(&input)];
    assert_eq!(f.interpret_with_buffers(&[1], &mut args).unwrap(), 12);
}
