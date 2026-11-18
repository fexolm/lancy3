use splice::{BufferArg, Executable, kernel};
use splice_lancy::{Lancy, LancyBaseline};

#[kernel(batch)]
fn project(#[comptime] gain: i32, value: i32, _unused: i32) -> i32 {
    value.wrapping_mul(gain).max(-8)
}

#[kernel(batch)]
fn reciprocal(value: i32) -> i32 {
    64 / value
}

#[kernel(batch)]
fn constant() -> i32 {
    9
}

#[test]
fn simd_batch_matches_the_portable_loop_for_partial_columns() {
    let baseline = project::build::<LancyBaseline>(3).unwrap();
    let simd = project::build::<Lancy>(3).unwrap();
    assert!(!baseline.is_vectorized());
    assert!(simd.is_vectorized());
    let values: Vec<i32> = (-7..20).collect();
    for (n, short) in [(9, 9), (9, 5), (16, 0), (0, 4), (17, 27)] {
        for code in [&simd, &baseline] {
            let mut out = vec![71; n];
            // A writable input can be borrowed read-only by the batch.
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
                        project(3, values[i], 0)
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
    assert!(!code.is_vectorized());
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
fn batch_with_no_inputs_fills_the_output() {
    let code = constant::build::<Lancy>().unwrap();
    assert!(code.is_vectorized());
    let mut out = [0; 7];
    assert_eq!(
        code.call(&[], &mut [BufferArg::Write(&mut out)]).unwrap(),
        7
    );
    assert_eq!(out, [9; 7]);
}
