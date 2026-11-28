use splice::{
    BuildOptions, CallError, Codegen, Executable, Mode, Trap, kernel, std::intrinsics::simd::Lane,
};

#[kernel]
fn conditional(guard: i32, value: i32, fallback: i32) -> i32 {
    if guard >= 0 {
        value.wrapping_mul(11).wrapping_add(7)
    } else {
        fallback
    }
}

#[kernel]
fn guarded_load(guard: i32, src: &[i32], out: &mut [i32], index: usize) -> i32 {
    out[0] = 7;
    let value = if guard < 0 { 42 } else { src[index] };
    out[1] = 9;
    value
}

#[kernel]
fn interleaved_packets(#[comptime] guarded: bool, src: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if guarded && n > (out.len() as i32) / 2 {
        return -1;
    }
    let width = Lane::<i32>::element_width() as i32;
    let mut i = 0;
    while n - i >= width {
        Lane::<i32>::splat(41).store(out, (2 * i) as usize);
        Lane::<i32>::splat(53).store(out, (2 * i + width) as usize);
        i += width;
    }
    i
}

#[kernel]
fn fused_packet(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) -> i32 {
    Lane::load(a, 0)
        .mul_add(Lane::load(b, 0), Lane::load(c, 0))
        .store(out, 0);
    0
}

fn check_fused<B: Codegen<Output: Executable>>(options: BuildOptions) {
    let width = options.target.vector_bytes() / 4;
    let lanes = |values: [u32; 4]| {
        values
            .into_iter()
            .cycle()
            .take(width)
            .map(f32::from_bits)
            .collect::<Vec<_>>()
    };
    let a = lanes([0x33800001, 0x80000000, 0x00000001, 0x7f7fffff]);
    let b = lanes([0x3f7ffffe, 0x3f800000, 0x3f000000, 0x40000000]);
    let c = lanes([0x3f800001, 0x80000000, 0x007fffff, 0xff7fffff]);
    let want = lanes([0x3f800001, 0x80000000, 0x00800000, 0x7f7fffff]);
    let ir = fused_packet::ir_with_options(options).unwrap();
    for code in [
        fused_packet::build_with_options::<B>(options, fused_packet::BuildArgs {}).unwrap(),
        B::compile_with_options(&ir, options).unwrap(),
    ] {
        let mut out = vec![0.0; width];
        let args = fused_packet::CallArgs {
            a: &a,
            b: &b,
            c: &c,
            out: &mut out,
        };
        assert_eq!(code.call(args), Ok(0));
        assert!(
            out.iter()
                .zip(&want)
                .all(|(out, want)| out.to_bits() == want.to_bits())
        );
    }
}

fn check<B: Codegen<Output: Executable>>(options: BuildOptions) {
    let conditional = [
        conditional::build_with_options::<B>(options, conditional::BuildArgs {}).unwrap(),
        B::compile_with_options(&conditional::ir_with_options(options).unwrap(), options).unwrap(),
    ];
    for code in conditional {
        for (guard, value, fallback) in [(0, 9, 21), (-1, i32::MIN, 42), (1, i32::MAX, 10)] {
            let expected = if guard >= 0 {
                value.wrapping_mul(11).wrapping_add(7)
            } else {
                fallback
            };
            assert_eq!(
                code.call(conditional::CallArgs {
                    guard,
                    value,
                    fallback
                }),
                Ok(expected)
            );
        }
    }
    let loads = [
        guarded_load::build_with_options::<B>(options, guarded_load::BuildArgs {}).unwrap(),
        B::compile_with_options(&guarded_load::ir_with_options(options).unwrap(), options).unwrap(),
    ];
    for code in loads {
        for (guard, result, expected) in [
            (-1, Ok(42), [7, 9]),
            (0, Err(CallError::Trap(Trap::Bounds)), [7, 0]),
        ] {
            let mut out = [0; 2];
            let actual = code.call(guarded_load::CallArgs {
                guard,
                src: &[3],
                out: &mut out,
                index: 1,
            });
            assert_eq!((actual, out), (result, expected));
        }
    }

    let width = options.target.vector_bytes() / 4;
    let src = vec![0; width];
    for guarded in [false, true] {
        let ir = interleaved_packets::ir_with_options(options, guarded).unwrap();
        for code in [
            interleaved_packets::build_with_options::<B>(
                options,
                interleaved_packets::BuildArgs { guarded },
            )
            .unwrap(),
            B::compile_with_options(&ir, options).unwrap(),
        ] {
            let mut out = vec![0; 2 * width + 1];
            let args = interleaved_packets::CallArgs {
                src: &src,
                out: &mut out[..2 * width],
            };
            assert_eq!(code.call(args), Ok(width as i32));
            assert!(out[..width].iter().all(|&word| word == 41));
            assert!(out[width..2 * width].iter().all(|&word| word == 53));
            assert_eq!(out[2 * width], 0);
            out.fill(0);
            let args = interleaved_packets::CallArgs {
                src: &src,
                out: &mut out[..2 * width - 1],
            };
            if guarded {
                assert_eq!(code.call(args), Ok(-1));
                assert!(out.iter().all(|&word| word == 0));
            } else {
                assert_eq!(code.call(args), Err(CallError::Trap(Trap::Bounds)));
                assert!(out[..width].iter().all(|&word| word == 41));
                assert!(out[width..2 * width - 1].iter().all(|&word| word == 53));
                assert_eq!(&out[2 * width - 1..], &[0, 0]);
            }
        }
    }
}

#[test]
fn speculative_arithmetic_and_packet_proofs_preserve_public_results() {
    for mode in [Mode::Debug, Mode::Release] {
        let options = BuildOptions::host(mode).unwrap();
        splice_testkit::each_backend!(B => check::<B>(options));
        splice_testkit::each_backend!(B => check_fused::<B>(options));
    }
}
