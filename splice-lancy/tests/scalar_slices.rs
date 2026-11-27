use splice_core::hir::{Buffer, Instr};
use splice_core::{BufferArg, BufferKind, Executable, kernel};
use splice_lancy::Lancy;

#[kernel(crate = splice_core)]
fn append_score(out: &mut [i32], #[comptime] factor: i32, values: &[i32], row: i32) -> i32 {
    let score = values[row as usize]
        .wrapping_mul(factor)
        .min(90)
        .wrapping_add(1)
        .max(-20);
    out[row as usize] += score;
    1
}

#[kernel(crate = splice_core)]
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
    let jit = lookup::build::<Lancy>(lookup::BuildArgs {}).unwrap();
    for input in [&[][..], &[7, 8][..]] {
        for row in [-1, i32::MIN, 0, 1, 2] {
            assert_eq!(
                jit.call(lookup::CallArgs { input, row }).unwrap(),
                lookup(input, row)
            );
        }
    }
}

#[kernel(crate = splice_core)]
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
    let f = score_column::ir(2).unwrap();
    assert_eq!(
        f.buffer_kinds(),
        &[BufferKind::READ_I32, BufferKind::WRITE_I32]
    );
    assert!(f.instructions().iter().any(|i| matches!(i,
        Instr::Call { buffers, .. } if buffers == &[Buffer(1), Buffer(0)]
    )));
    let k = score_column::build::<Lancy>(score_column::BuildArgs { factor: 2 }).unwrap();
    let mut out = [2; 5];
    assert_eq!(
        k.call(score_column::CallArgs {
            values: &values,
            out: &mut out
        })
        .unwrap(),
        expected_count
    );
    assert_eq!(out, expected);
}

#[test]
fn entry_points_check_buffer_signatures_before_executing() {
    let kernel = score_column::build::<Lancy>(score_column::BuildArgs { factor: 3 })
        .unwrap()
        .into_inner();
    assert!(kernel.call(&[], &mut []).is_err());
    let mut out = [8];
    assert!(
        kernel
            .call(
                &[],
                &mut [BufferArg::read::<i32>(&[5]), BufferArg::read(&out)]
            )
            .is_err()
    );
    assert_eq!(
        kernel
            .call(
                &[],
                &mut [BufferArg::read::<i32>(&[5]), BufferArg::write(&mut out)]
            )
            .unwrap(),
        1
    );
    assert_eq!(out, [24]);
}

#[kernel(crate = splice_core)]
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
    let f = assign_in_order::build::<Lancy>(assign_in_order::BuildArgs {}).unwrap();
    let mut expected = [0; 3];
    assert_eq!(assign_in_order(&mut expected, 2), 4);
    assert_eq!(expected, [4, 4, 5]);
    let mut out = [0; 3];
    assert_eq!(
        f.call(assign_in_order::CallArgs {
            buffer: &mut out,
            value: 2
        })
        .unwrap(),
        4
    );
    assert_eq!(out, expected);
}

#[kernel(crate = splice_core)]
fn read_at(#[comptime] index: usize, input: &[i32]) -> i32 {
    input[index]
}

#[test]
fn host_indices_outside_register_width_stay_out_of_bounds() {
    let f = read_at::build::<Lancy>(read_at::BuildArgs {
        index: 1usize << 32,
    })
    .unwrap();
    assert!(f.call(read_at::CallArgs { input: &[12] }).is_err());
}

#[kernel(crate = splice_core)]
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
    let jit = divide::build::<Lancy>(divide::BuildArgs {}).unwrap();
    for mode in 0..4 {
        for (n, d) in [(-17, 5), (17, -5), (-3, -7), (i32::MIN, 3)] {
            assert_eq!(
                jit.call(divide::CallArgs {
                    mode,
                    value: n,
                    divisor: d
                })
                .unwrap(),
                divide(mode, n, d),
                "mode={mode} n={n} d={d}"
            );
        }
        for (n, d) in [(23, 0), (i32::MIN, -1)] {
            assert!(
                jit.call(divide::CallArgs {
                    mode,
                    value: n,
                    divisor: d
                })
                .is_err()
            );
        }
    }
    assert_eq!(
        jit.call(divide::CallArgs {
            mode: 4,
            value: 2,
            divisor: 0
        })
        .unwrap(),
        19
    );
}

#[kernel(crate = splice_core)]
fn scale(#[comptime] multiplier: i32, row: i32) -> i32 {
    multiplier.wrapping_mul(row)
}

#[kernel(crate = splice_core)]
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

#[kernel(crate = splice_core)]
fn staged_usize(#[comptime] offset: usize, input: &[i32], row: i32) -> i32 {
    if row == 0 { 0 } else { input[offset / 2] }
}

#[test]
fn failing_comptime_i32_arithmetic_in_a_runtime_arm_faults_only_when_selected() {
    for (a, n) in [(17, 0), (i32::MIN, -1), (17, 4)] {
        let jit = staged_divide::build::<Lancy>(staged_divide::BuildArgs { a, n }).unwrap();
        for row in [0, 1, 2] {
            let expected = match row {
                0 => Some(0),
                1 => a.checked_div(n),
                _ => a.checked_rem(n),
            };
            assert_eq!(jit.call(staged_divide::CallArgs { row }).ok(), expected);
        }
    }
    let f = staged_usize::build::<Lancy>(staged_usize::BuildArgs { offset: 4 }).unwrap();
    let input = [10, 11, 12];
    assert_eq!(
        f.call(staged_usize::CallArgs {
            input: &input,
            row: 1
        })
        .unwrap(),
        12
    );
}
