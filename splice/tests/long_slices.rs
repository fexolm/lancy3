//! Slices past `i32::MAX` elements, as Rust allows them. Cranelift and LLVM
//! (both SIMD modes) index them with 64-bit offsets on both build paths and
//! in both modes; Lancy indexes with `i32` and rejects a longer slice before
//! the call. The zeroed buffers are mapped lazily: a test touches only the
//! pages it reads or writes.
#![cfg(all(target_os = "linux", target_pointer_width = "64"))]
use splice::std::intrinsics::simd::Lane;
use splice::{BuildOptions, CallError, Codegen, Executable, Mode, SignatureError, Trap, kernel};

/// Bytes past `i32::MAX`: the last ones lie beyond every `i32` offset.
const LONG: usize = (1 << 31) + 16;

#[kernel]
fn far_bytes(xs: &mut [u8], at: u64, value: u8) -> u64 {
    let i = at as usize;
    xs[i] = value;
    let tail = &xs[i..];
    let n = tail.len() as u64;
    let mut sum = 0_u64;
    for &x in tail {
        sum += x as u64;
    }
    let end = xs.len() - Lane::<u8>::element_width();
    let v = Lane::load(xs, end) + Lane::splat(1);
    v.store(xs, end);
    n * 1_000 + sum + xs[xs.len() - 1] as u64 * 1_000_000
}

#[kernel]
fn far_words(xs: &[u32], out: &mut [u32], at: u32) -> u32 {
    let v = Lane::<u32>::gather(xs, Lane::splat(at));
    v.store(out, 0);
    xs.len() as u32
}

/// Every built form of a kernel: path and mode.
macro_rules! builds {
    ($kernel:ident, $b:ident) => {{
        let mut codes = Vec::new();
        for mode in [Mode::Release, Mode::Debug] {
            let options = BuildOptions::host(mode).unwrap();
            codes.push($kernel::build_with_options::<$b>(options, $kernel::BuildArgs {}).unwrap());
            codes.push($b::compile(&$kernel::ir_with_options(options).unwrap()).unwrap());
        }
        codes
    }};
}

fn bytes<B: Codegen<Output: Executable>>(xs: &mut [u8], limited: bool) {
    for code in builds!(far_bytes, B) {
        let at = (LONG - 3) as u64;
        let result = code.call(far_bytes::CallArgs { xs, at, value: 5 });
        if limited {
            let error = CallError::Signature(SignatureError::Length {
                buffer: 0,
                limit: i32::MAX as usize,
            });
            assert_eq!(result, Err(error));
            assert_eq!(xs[LONG - 3], 0, "nothing runs");
            continue;
        }
        assert_eq!(result, Ok(1_003_005), "{}", std::any::type_name::<B>());
        assert_eq!(xs[LONG - 3], 6, "the store adds one to the last lanes");
        assert_eq!(xs[LONG - 1], 1);
        let past = code.call(far_bytes::CallArgs {
            xs,
            at: LONG as u64,
            value: 9,
        });
        assert_eq!(past, Err(CallError::Trap(Trap::Bounds)));
        assert_eq!(xs[LONG - 1], 1, "no write before the trap");
        xs[LONG - 64..].fill(0);
    }
}

fn words<B: Codegen<Output: Executable>>(xs: &[u32], limited: bool) {
    let lanes = splice::Target::host().unwrap().vector_bytes() / size_of::<u32>();
    for code in builds!(far_words, B) {
        for at in [(1u32 << 31) + 1, xs.len() as u32] {
            let mut out = vec![0; lanes];
            let result = code.call(far_words::CallArgs {
                xs,
                out: &mut out,
                at,
            });
            let expected = if limited {
                Err(CallError::Signature(SignatureError::Length {
                    buffer: 0,
                    limit: i32::MAX as usize,
                }))
            } else if (at as usize) < xs.len() {
                assert_eq!(out, vec![7; lanes], "{}", std::any::type_name::<B>());
                Ok(xs.len() as u32)
            } else {
                Err(CallError::Trap(Trap::Bounds))
            };
            assert_eq!(result, expected, "{} at {at}", std::any::type_name::<B>());
        }
    }
}

#[test]
fn byte_slices_past_i32_max_index_with_64_bit_offsets() {
    let mut xs = vec![0u8; LONG];
    bytes::<splice::lancy::Lancy>(&mut xs, true);
    bytes::<splice::cranelift::Cranelift>(&mut xs, false);
    #[cfg(feature = "llvm")]
    {
        bytes::<splice::llvm::Llvm>(&mut xs, false);
        bytes::<splice::llvm::LlvmManualSimd>(&mut xs, false);
    }
}

/// A `u32` gather index past `i32::MAX`, which a sign-extended 32-bit
/// offset would turn negative. The 8 GiB of the slice stay unmapped.
#[test]
fn unsigned_gather_indices_reach_past_i32_max() {
    let mut xs = vec![0u32; LONG];
    xs[(1 << 31) + 1] = 7;
    words::<splice::lancy::Lancy>(&xs, true);
    words::<splice::cranelift::Cranelift>(&xs, false);
    #[cfg(feature = "llvm")]
    {
        words::<splice::llvm::Llvm>(&xs, false);
        words::<splice::llvm::LlvmManualSimd>(&xs, false);
    }
}
