use splice::{BufferArg, Codegen, Executable, kernel};
use splice_lancy::Lancy;
use splice_llvm::LlvmManualSimd;

#[kernel]
fn apply_coefficients(src: &[i32], coefficients: &[i32], dst: &mut [i32]) -> i32 {
    if coefficients.len() < 2 {
        return -1;
    }
    for i in 0..src.len() as i32 {
        dst[i as usize] = src[i as usize]
            .wrapping_mul(coefficients[0])
            .wrapping_add(coefficients[1]);
    }
    src.len() as i32
}

#[kernel]
fn checked_passes(src: &[i32], first: &mut [i32], second: &mut [i32]) -> i32 {
    for i in 0..src.len() as i32 {
        first[i as usize] = src[i as usize].wrapping_add(1);
    }
    for i in 0..src.len() as i32 {
        second[i as usize] = first[i as usize].wrapping_mul(2);
    }
    src.len() as i32
}

fn check<B: Codegen<Output: Executable>>() {
    let src = [i32::MAX, 3, -2];
    for kernel in [
        apply_coefficients::build::<B>().unwrap(),
        B::compile(&apply_coefficients::ir().unwrap()).unwrap(),
    ] {
        let mut dst = [0; 3];
        assert_eq!(
            kernel.call(
                &[],
                &mut [
                    BufferArg::Read(&src),
                    BufferArg::Read(&[2, 7]),
                    BufferArg::Write(&mut dst),
                ]
            ),
            Ok(3)
        );
        assert_eq!(dst, [5, 13, 3]);

        let mut short = [99];
        assert!(
            kernel
                .call(
                    &[],
                    &mut [
                        BufferArg::Read(&src),
                        BufferArg::Read(&[2, 7]),
                        BufferArg::Write(&mut short),
                    ]
                )
                .is_err()
        );
        assert_eq!(short, [5]);

        assert_eq!(
            kernel.call(
                &[],
                &mut [
                    BufferArg::Read(&[]),
                    BufferArg::Read(&[]),
                    BufferArg::Write(&mut []),
                ]
            ),
            Ok(-1)
        );
    }

    for kernel in [
        checked_passes::build::<B>().unwrap(),
        B::compile(&checked_passes::ir().unwrap()).unwrap(),
    ] {
        let mut first = [99; 3];
        let mut second = [99];
        assert!(
            kernel
                .call(
                    &[],
                    &mut [
                        BufferArg::Read(&src),
                        BufferArg::Write(&mut first),
                        BufferArg::Write(&mut second),
                    ]
                )
                .is_err()
        );
        assert_eq!(first, [i32::MIN, 4, -1]);
        assert_eq!(second, [0]);
    }
}

#[test]
fn loop_memory_optimizations_preserve_partial_writes() {
    check::<Lancy>();
    check::<LlvmManualSimd>();
}
