#[path = "../benches/large/biquad.rs"]
mod biquad;
#[path = "../benches/large/csv.rs"]
mod csv;
#[path = "../benches/large/delta.rs"]
mod delta;
#[path = "../benches/large/vector/hash.rs"]
mod hash;
#[path = "../benches/large/vector/integral.rs"]
mod integral;
#[path = "../benches/large/vector/remap.rs"]
mod remap;
#[path = "../benches/large/requantize.rs"]
mod requantize;
#[path = "../benches/large/vector/yuv.rs"]
mod yuv;

use splice::{BufferArg, Codegen, Executable, Function};
use splice_cranelift::Cranelift;
use splice_lancy::Lancy;
use splice_llvm::Llvm;

const SENTINEL: i32 = -123_456_789;

fn check_backends(ir: Function, check: impl Fn(&str, &dyn Executable)) {
    check("Lancy", &Lancy::compile(&ir).unwrap());
    check("Cranelift", &Cranelift::compile(&ir).unwrap());
    check("LLVM", &Llvm::compile(&ir).unwrap());
}

#[test]
fn delta_handles_first_value_and_rejects_a_truncated_block_before_writing() {
    use delta::{BLOCK_VALUES, BLOCK_WORDS, decode};
    check_backends(decode::ir().unwrap(), |name, kernel| {
        let mut out = [SENTINEL; BLOCK_VALUES as usize + 1];
        let packed = [0; BLOCK_WORDS as usize];
        let decode = |minima: &[i32], packed: &[i32], out: &mut [i32]| {
            kernel
                .call(
                    &[i32::MIN],
                    &mut [
                        BufferArg::Read(minima),
                        BufferArg::Read(packed),
                        BufferArg::Write(out),
                    ],
                )
                .unwrap()
        };
        let count = decode(&[], &[], &mut out[..1]);
        assert_eq!((count, out[0], out[1]), (1, i32::MIN, SENTINEL), "{name}");
        let count = decode(&[-1], &packed, &mut out);
        assert_eq!(count, BLOCK_VALUES + 1, "{name}");
        assert_eq!(
            out,
            std::array::from_fn(|i| i32::MIN.wrapping_sub(i as i32)),
            "{name}"
        );
        out.fill(SENTINEL);
        for (words, capacity) in [(packed.len() - 1, out.len()), (packed.len(), out.len() - 1)] {
            let count = decode(&[-1], &packed[..words], &mut out[..capacity]);
            assert_eq!(
                (count, out),
                (-1, [SENTINEL; BLOCK_VALUES as usize + 1]),
                "{name}"
            );
        }
    });
}

#[test]
fn csv_preserves_the_written_prefix_on_invalid_value() {
    use csv::{FIELD_WORDS, FIELDS, export};
    let records = [0, -1, i32::MIN, 1];
    let expected = b"+000000.000,-000000.001,".as_chunks::<4>().0;
    check_backends(export::ir().unwrap(), |name, kernel| {
        let mut out = [SENTINEL; FIELDS * FIELD_WORDS];
        let short = out.len() - 1;
        let mut export = |capacity: usize| {
            let count = kernel
                .call(
                    &[],
                    &mut [
                        BufferArg::Read(&records),
                        BufferArg::Write(&mut out[..capacity]),
                    ],
                )
                .unwrap();
            (count, out)
        };
        assert_eq!(
            export(short),
            (-1, [SENTINEL; FIELDS * FIELD_WORDS]),
            "{name}"
        );
        let (count, actual) = export(FIELDS * FIELD_WORDS);
        assert_eq!(count, -1, "{name}");
        for (&word, &bytes) in actual.iter().zip(expected) {
            assert_eq!(word, i32::from_le_bytes(bytes), "{name}");
        }
        assert_eq!(
            actual[expected.len()..],
            [SENTINEL; 2 * FIELD_WORDS],
            "{name}"
        );
    });
}

#[test]
fn requantize_rounds_signed_ties_and_clips_per_channel() {
    use self::requantize::{SCALE, requantize};
    let channels = [(1, 1, 0), (SCALE, SCALE, 1)];
    check_backends(requantize::ir(&channels).unwrap(), |name, kernel| {
        let accum = [2048, 127, -2048, -129, 2047, 4, -2047, 4];
        let residual = [0, 127, 0, -129, 2048, -8, -2048, 8];
        let mut out = [SENTINEL; 9];
        let mut run = |residual: &[i32]| {
            let count = kernel
                .call(
                    &[],
                    &mut [
                        BufferArg::Read(&accum),
                        BufferArg::Read(residual),
                        BufferArg::Write(&mut out),
                    ],
                )
                .unwrap();
            (count, out)
        };
        assert_eq!(run(&residual[..7]), (-1, [SENTINEL; 9]), "{name}");
        assert_eq!(
            run(&residual),
            (8, [1, 127, -1, -128, 1, -3, -1, 13, SENTINEL]),
            "{name}"
        );
    });
}

#[test]
fn biquad_starts_with_zero_state_on_every_call() {
    use biquad::{Q14, low_pass};
    let sections = [(Q14 / 2, 0, -Q14 / 2, 0)];
    check_backends(low_pass::ir(&sections).unwrap(), |name, kernel| {
        let pcm = [3, -5, 6, 0, 0, 0];
        let mut out = [SENTINEL; 7];
        let mut state = [SENTINEL; 2];
        assert_eq!(
            kernel.call(
                &[],
                &mut [
                    BufferArg::Read(&pcm),
                    BufferArg::Write(&mut out),
                    BufferArg::Write(&mut state[..1]),
                ]
            ),
            Ok(-1),
            "{name}"
        );
        assert_eq!(out, [SENTINEL; 7], "{name}");
        assert_eq!(state, [SENTINEL; 2], "{name}");
        let mut run = |capacity: usize| {
            let count = kernel
                .call(
                    &[],
                    &mut [
                        BufferArg::Read(&pcm),
                        BufferArg::Write(&mut out[..capacity]),
                        BufferArg::Write(&mut state),
                    ],
                )
                .unwrap();
            (count, out)
        };
        assert_eq!(run(5), (-1, [SENTINEL; 7]), "{name}");
        for _ in 0..2 {
            assert_eq!(run(7), (6, [1, -2, 3, -1, 2, 1, SENTINEL]), "{name}");
        }
    });
}

#[test]
fn yuv_converts_known_colors_across_packets_and_tail() {
    use splice::simd::Lane;
    let colors = [
        ([16, 128, 128, 0], [0, 0, 0, 255]),
        ([235, 128, 128, 17], [255, 255, 255, 255]),
        ([81, 90, 240, 255], [255, 0, 0, 255]),
    ];
    let count = 2 * Lane::<i32>::lanes() + 3;
    let pixels: Vec<_> = colors
        .iter()
        .cycle()
        .take(count)
        .map(|&(yuv, _)| i32::from_le_bytes(yuv))
        .collect();
    let expected: Vec<_> = colors
        .iter()
        .cycle()
        .take(count)
        .map(|&(_, rgba)| i32::from_le_bytes(rgba))
        .collect();
    let check = |name: &str, kernel: &dyn Executable| {
        let mut out = vec![SENTINEL; count + 1];
        let short = kernel
            .call(
                &[],
                &mut [
                    BufferArg::Read(&pixels),
                    BufferArg::Write(&mut out[..count - 1]),
                ],
            )
            .unwrap();
        assert_eq!(short, -1, "{name}");
        assert!(out.iter().all(|&pixel| pixel == SENTINEL), "{name}");
        let width = Lane::<i32>::lanes();
        for len in [0, width - 1, width, width + 1, count] {
            out.fill(SENTINEL);
            let written = kernel
                .call(
                    &[],
                    &mut [BufferArg::Read(&pixels[..len]), BufferArg::Write(&mut out)],
                )
                .unwrap();
            assert_eq!(written, len as i32, "{name} at {len}");
            assert_eq!(out[..len], expected[..len], "{name} at {len}");
            assert!(out[len..].iter().all(|&x| x == SENTINEL), "{name} at {len}");
        }
    };
    check_backends(yuv::yuv_to_rgba::ir().unwrap(), check);
    check("direct Lancy", &yuv::yuv_to_rgba::build::<Lancy>().unwrap());
    check("direct LLVM", &yuv::yuv_to_rgba::build::<Llvm>().unwrap());
}

#[test]
fn integral_image_keeps_both_planes_separate_and_checks_capacity() {
    check_backends(integral::integral_moments::ir().unwrap(), |name, kernel| {
        let pixels = [1, 2, 3, 4, 5, 6];
        let mut out = [SENTINEL; 13];
        let short = kernel
            .call(
                &[3, 2],
                &mut [BufferArg::Read(&pixels), BufferArg::Write(&mut out[..11])],
            )
            .unwrap();
        assert_eq!((short, out), (-1, [SENTINEL; 13]), "{name}");
        let written = kernel
            .call(
                &[3, 2],
                &mut [BufferArg::Read(&pixels), BufferArg::Write(&mut out)],
            )
            .unwrap();
        assert_eq!(
            (written, out),
            (12, [1, 3, 6, 5, 12, 21, 1, 5, 14, 17, 46, 91, SENTINEL]),
            "{name}"
        );
    });
}

#[test]
fn murmur_hash_matches_a_known_key_and_rejects_incomplete_columns() {
    // MurmurHash3_x86_32 with seed zero and a key containing bytes 0 through 31.
    let key = std::array::from_fn::<_, { hash::COLUMNS }, _>(|column| {
        i32::from_le_bytes(std::array::from_fn(|byte| (4 * column + byte) as u8))
    });
    check_backends(hash::hash_keys::ir().unwrap(), |name, kernel| {
        let mut out = [SENTINEL; 2];
        let short = kernel
            .call(
                &[0],
                &mut [BufferArg::Read(&key[..7]), BufferArg::Write(&mut out)],
            )
            .unwrap();
        assert_eq!((short, out), (-1, [SENTINEL; 2]), "{name}");
        let written = kernel
            .call(
                &[0],
                &mut [BufferArg::Read(&key), BufferArg::Write(&mut out)],
            )
            .unwrap();
        assert_eq!(
            (written, out),
            (1, [0xcac3_7638_u32 as i32, SENTINEL]),
            "{name}"
        );
    });
}

#[test]
fn rgb16_remap_supports_unsigned_indices_and_stops_on_an_invalid_plane() {
    let check = |name: &str, kernel: &dyn Executable| {
        let mut src = vec![0u16; 65536];
        src[1] = 200;
        src[32768] = 300;
        src[65535] = 500;
        let mut out = [u16::MAX; 4];
        let map = [65535, 32768, 1];
        let written = kernel
            .call(
                &[],
                &mut [
                    BufferArg::ReadU16(&src),
                    BufferArg::ReadU16(&map),
                    BufferArg::WriteU16(&mut out),
                ],
            )
            .unwrap();
        assert_eq!((written, out), (3, [500, 300, 200, u16::MAX]), "{name}");

        out.fill(u16::MAX);
        let map = [32768, 65535, 1];
        let result = kernel.call(
            &[],
            &mut [
                BufferArg::ReadU16(&src[..65535]),
                BufferArg::ReadU16(&map),
                BufferArg::WriteU16(&mut out),
            ],
        );
        assert!(result.is_err(), "{name}");
        assert_eq!(out, [300, u16::MAX, u16::MAX, u16::MAX], "{name}");

        let width = splice::simd::Lane::<u16>::lanes();
        let mut map = vec![1u16; 6 * width];
        map[width + 1] = u16::MAX;
        let mut out = vec![u16::MAX; map.len()];
        let result = kernel.call(
            &[],
            &mut [
                BufferArg::ReadU16(&src[..65535]),
                BufferArg::ReadU16(&map),
                BufferArg::WriteU16(&mut out),
            ],
        );
        assert!(result.is_err(), "{name}");
        for plane in out.chunks_exact(2 * width) {
            assert!(plane[..width].iter().all(|&v| v == 200), "{name}");
            assert!(plane[width..].iter().all(|&v| v == u16::MAX), "{name}");
        }
    };
    check_backends(remap::remap_rgb16::ir().unwrap(), check);
    check(
        "direct Lancy",
        &remap::remap_rgb16::build::<Lancy>().unwrap(),
    );
    check("direct LLVM", &remap::remap_rgb16::build::<Llvm>().unwrap());
}
