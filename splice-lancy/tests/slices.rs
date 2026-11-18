use splice::{BufferArg, Codegen, Executable, kernel};
use splice_lancy::Lancy;

#[kernel]
fn project(#[comptime] gain: i32, values: &[i32], second: &[i32], out: &mut [i32]) -> i32 {
    for i in 0..out.len() as i32 {
        let value = values[i as usize];
        let _observed = second[i as usize];
        out[i as usize] = value.wrapping_mul(gain).max(-8);
    }
    out.len() as i32
}

#[kernel]
fn reciprocal(values: &[i32], out: &mut [i32]) -> i32 {
    for i in 0..out.len() as i32 {
        out[i as usize] = 64 / values[i as usize];
    }
    out.len() as i32
}

#[kernel]
fn constant(out: &mut [i32]) -> i32 {
    for i in 0..out.len() as i32 {
        out[i as usize] = 9;
    }
    out.len() as i32
}

#[test]
fn partial_columns_have_the_same_store_order_through_both_splice_entry_points() {
    let direct = project::build::<Lancy>(3).unwrap();
    let imported = Lancy::compile(&project::ir(3).unwrap()).unwrap();
    let values: Vec<i32> = (-7..20).collect();
    for (n, short) in [(9, 9), (9, 5), (16, 0), (0, 4), (17, 27)] {
        for code in [&direct, &imported] {
            let mut out = vec![71; n];
            // A writable input can be borrowed read-only by the kernel.
            let mut unused = vec![0; short];
            let result = code.call(
                &[],
                &mut [
                    BufferArg::Read(&values),
                    BufferArg::Write(&mut unused),
                    BufferArg::Write(&mut out),
                ],
            );
            if short < n {
                assert!(result.unwrap_err().contains("bounds"));
            } else {
                assert_eq!(result.unwrap(), n as i32);
            }
            for (i, &value) in out.iter().enumerate() {
                assert_eq!(
                    value,
                    if i < short {
                        values[i].wrapping_mul(3).max(-8)
                    } else {
                        71
                    }
                );
            }
        }
    }
}

#[test]
fn checks_the_full_signature_before_the_first_store() {
    let code = project::build::<Lancy>(2).unwrap();
    let value = [4; 8];
    let mut out = [17; 8];
    assert!(code.call(&[], &mut [BufferArg::Write(&mut out)]).is_err());
    assert!(
        code.call(
            &[],
            &mut [
                BufferArg::Read(&value),
                BufferArg::Read(&value),
                BufferArg::Read(&out),
            ]
        )
        .is_err()
    );
    assert!(
        code.call(
            &[0],
            &mut [
                BufferArg::Read(&value),
                BufferArg::Read(&value),
                BufferArg::Write(&mut out),
            ]
        )
        .is_err()
    );
    assert_eq!(out, [17; 8]);
}

#[test]
fn fallible_row_uses_the_scalar_loop_and_retains_store_order() {
    let code = reciprocal::build::<Lancy>().unwrap();
    let mut out = [55; 4];
    assert!(
        code.call(
            &[],
            &mut [BufferArg::Read(&[8, -4, 0]), BufferArg::Write(&mut out),]
        )
        .unwrap_err()
        .contains("division")
    );
    assert_eq!(out, [8, -16, 55, 55]);
}

#[test]
fn kernel_with_no_inputs_fills_the_output() {
    let code = constant::build::<Lancy>().unwrap();
    let mut out = [0; 7];
    assert_eq!(
        code.call(&[], &mut [BufferArg::Write(&mut out)]).unwrap(),
        7
    );
    assert_eq!(out, [9; 7]);
}
