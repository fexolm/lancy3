use super::{Llvm, build_module, jit, lower, target};
use inkwell::{context::Context, targets::FileType};
use splice::{Codegen, kernel, simd::Lane};
use splice_target::{Profile, simd};

#[kernel]
fn xor_three(a: &[i32], b: &[i32], c: &[i32], out: &mut [i32]) -> i32 {
    let x = Lane::load(a, 0);
    let y = Lane::load(b, 0);
    let z = Lane::load(c, 0);
    (x ^ y ^ z).store(out, 0);
    Lane::<i32>::lanes() as i32
}

#[test]
fn emits_aarch64_simd_and_preserves_the_logical_width() {
    let config = simd().unwrap();
    let ir = xor_three::ir().unwrap();
    let target = target::host_machine().unwrap();
    let context = Context::create();
    let module = build_module(&context, &target, |ctx, m| lower::populate(ctx, m, &ir)).unwrap();
    jit::optimize(&module, &target, true).unwrap();
    let assembly = target
        .write_to_memory_buffer(&module, FileType::Assembly)
        .unwrap();
    let assembly = std::str::from_utf8(assembly.as_slice()).unwrap();
    match config.profile {
        Profile::Sve | Profile::Sve2 if config.bytes >= 32 => {
            assert!(assembly.contains("ld1w\t{ z"), "{assembly}");
            assert!(assembly.contains("st1w\t{ z"), "{assembly}");
            assert_eq!(
                assembly.contains("eor3\tz"),
                config.profile == Profile::Sve2,
                "{assembly}"
            );
        }
        Profile::Neon => {
            assert!(assembly.contains("eor\tv"), "{assembly}");
            assert!(!assembly.contains("eor3\tz"), "{assembly}");
        }
        _ => {}
    }

    let n = Lane::<i32>::lanes();
    // Leave enough space for a whole architectural SVE register so the
    // canary detects a full-register store at a VL above the logical width.
    let a: Vec<_> = (0..64).collect();
    let b: Vec<_> = a.iter().map(|&x| 0x7654_3210 ^ (x << 16)).collect();
    let c: Vec<_> = a.iter().map(|&x| !(x << 8)).collect();
    let mut expected = vec![77; a.len()];
    for i in 0..n {
        expected[i] = a[i] ^ b[i] ^ c[i];
    }
    for kernel in [
        xor_three::build::<Llvm>(xor_three::BuildArgs {}).unwrap(),
        Llvm::compile(&ir).unwrap(),
    ] {
        let mut out = vec![77; a.len()];
        assert_eq!(
            kernel.call(xor_three::CallArgs {
                a: &a,
                b: &b,
                c: &c,
                out: &mut out
            }),
            Ok(n as i32)
        );
        assert_eq!(out, expected);
    }
}

#[test]
fn rejects_a_shorter_sve_thread_before_writing() {
    let config = simd().unwrap();
    if !matches!(config.profile, Profile::Sve | Profile::Sve2) || config.bytes == 16 {
        return;
    }
    let kernel = xor_three::build::<Llvm>(xor_three::BuildArgs {}).unwrap();
    std::thread::spawn(move || {
        // Linux uapi/linux/prctl.h.
        const PR_SVE_SET_VL: libc::c_int = 50;
        // SAFETY: PR_SVE_SET_VL takes a byte count, no pointers, and affects
        // only this test thread. Variadic arguments are unsigned long.
        let vl = unsafe { libc::prctl(PR_SVE_SET_VL, 16usize, 0usize, 0usize, 0usize) };
        assert_eq!(vl, 16);
        let n = config.bytes / 4;
        let input = vec![10; n];
        let mut out = vec![77; n];
        let result = kernel.call(xor_three::CallArgs {
            a: &input,
            b: &input,
            c: &input,
            out: &mut out,
        });
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("SVE vector length")
        );
        assert_eq!(out, vec![77; n]);
    })
    .join()
    .unwrap();
}
