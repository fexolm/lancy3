use super::{Llvm, build_module, jit, lower, target};
use inkwell::{context::Context, targets::FileType};
use splice_core::{Codegen, Target, kernel, simd::Lane};

#[kernel(crate = splice_core)]
fn xor_three(a: &[i32], b: &[i32], c: &[i32], out: &mut [i32]) -> i32 {
    let x = Lane::load(a, 0);
    let y = Lane::load(b, 0);
    let z = Lane::load(c, 0);
    (x ^ y ^ z).store(out, 0);
    Lane::<i32>::lanes() as i32
}

#[test]
fn emits_aarch64_simd_and_preserves_the_logical_width() {
    let config = Target::host().unwrap();
    let ir = xor_three::ir().unwrap();
    let machine = target::Machine::new(&config);
    let target = target::machine(&config, false).unwrap();
    let context = Context::create();
    let module = build_module(&context, &target, machine, |ctx, m| {
        lower::populate_function(ctx, m, machine, &ir, false).map(drop)
    })
    .unwrap();
    jit::optimize(&module, &target, true).unwrap();
    let assembly = target
        .write_to_memory_buffer(&module, FileType::Assembly)
        .unwrap();
    let assembly = std::str::from_utf8(assembly.as_slice()).unwrap();
    if config.sve() && config.vector_bytes() >= 32 {
        if target::scalable(&config) {
            assert!(assembly.contains("ld1w\t{ z"), "{assembly}");
            assert!(assembly.contains("st1w\t{ z"), "{assembly}");
            assert_eq!(
                assembly.contains("eor3\tz"),
                config.arm64().unwrap().sve2,
                "{assembly}"
            );
        } else {
            // A non-power-of-two vector length gets NEON code.
            assert!(assembly.contains("eor\tv"), "{assembly}");
            assert!(!assembly.contains("\tz"), "{assembly}");
        }
    } else if !config.sve() {
        assert!(assembly.contains("eor\tv"), "{assembly}");
        assert!(!assembly.contains("eor3\tz"), "{assembly}");
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
    let config = Target::host().unwrap();
    if !config.sve() || config.vector_bytes() == 16 {
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
        let n = config.vector_bytes() / 4;
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

/// Set this thread's SVE vector length; `None` when the CPU lacks it.
fn set_sve_vl(bytes: usize) -> Option<()> {
    // Linux uapi/linux/prctl.h.
    const PR_SVE_SET_VL: libc::c_int = 50;
    // SAFETY: PR_SVE_SET_VL takes a byte count, no pointers, and affects
    // only the calling thread. Variadic arguments are unsigned long.
    let vl = unsafe { libc::prctl(PR_SVE_SET_VL, bytes, 0usize, 0usize, 0usize) };
    (vl == bytes as libc::c_int).then_some(())
}

#[test]
fn sve_code_is_only_generated_and_run_at_power_of_two_vector_lengths() {
    let config = Target::host().unwrap();
    if !config.sve() || config.vector_bytes() != 32 {
        return;
    }
    let build = |vl| {
        std::thread::spawn(move || {
            set_sve_vl(vl)?;
            Some(xor_three::build::<Llvm>(xor_three::BuildArgs {}).unwrap())
        })
        .join()
        .unwrap()
    };
    let (Some(sve), Some(neon)) = (build(32), build(48)) else {
        return;
    };
    let input: Vec<i32> = (0..16).collect();
    let call = |kernel: &splice_core::TypedKernel<_, crate::LlvmKernel>, vl| {
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    set_sve_vl(vl).unwrap();
                    let mut out = vec![77; 16];
                    let result = kernel.call(xor_three::CallArgs {
                        a: &input,
                        b: &input,
                        c: &input,
                        out: &mut out,
                    });
                    (result.map_err(|error| error.to_string()), out)
                })
                .join()
                .unwrap()
        })
    };
    let mut expected = input.clone();
    expected[8..].fill(77);
    for vl in [32, 48] {
        assert_eq!(call(&neon, vl), (Ok(8), expected.clone()));
    }
    assert_eq!(call(&sve, 32), (Ok(8), expected));
    let (result, out) = call(&sve, 48);
    assert!(result.unwrap_err().contains("power-of-two"));
    assert_eq!(out, vec![77; 16]);
}
