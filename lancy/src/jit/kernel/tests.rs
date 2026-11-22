use super::{Entry, Executable, ScalarKernel};
use crate::hir::{Binary, Function, Op, Terminator};
use iced_x86::{IcedError, code_asm::*};
use std::sync::Arc;

#[test]
fn scratch_pointers_require_the_raw_call_contract_transitively() {
    let mut callee = Function::scalar(0);
    let block = callee.entry();
    let address = callee.append(block, Op::CallScratch(8)).unwrap();
    let value = callee.append(block, Op::Const(19)).unwrap();
    callee.append(block, Op::Store(address, value));
    let read = callee
        .append(block, Op::Load(crate::hir::Type::I32, address))
        .unwrap();
    callee.set_terminator(block, Terminator::Return(read));
    let mut caller = Function::scalar(0);
    let block = caller.entry();
    let call = caller.add_call(Arc::new(callee), &[]);
    let returned = caller.append(block, Op::Call(call)).unwrap();
    caller.set_terminator(block, Terminator::Return(returned));

    for policy in [crate::Inlining::Always, crate::Inlining::Never] {
        let kernel = ScalarKernel::compile_with_inlining(&caller, policy).unwrap();
        assert_eq!(kernel.call(&[]), Err(crate::CallError::PointerSignature));
        assert_eq!(
            kernel.call_words(&[], &mut [0]),
            Err(crate::CallError::PointerSignature)
        );
        // SAFETY: the only access is the initialized first word of this
        // live frame's aligned scratch area. No pointers cross the ABI.
        assert_eq!(unsafe { kernel.call_raw(&[]) }, Ok(19));
    }
}

/// Call the real JIT entry with host register canaries. Return one if any
/// nonvolatile GPR changed, after restoring the test caller's own registers.
fn abi_probe(entry: Entry) -> Result<Vec<u8>, IcedError> {
    let regs: &[AsmRegister64] = if cfg!(windows) {
        &[rbx, rbp, rsi, rdi, r12, r13, r14, r15]
    } else {
        &[rbx, rbp, r12, r13, r14, r15]
    };
    let mut a = CodeAssembler::new(64)?;
    a.endbr64()?;
    let canary = |i: usize| 0x1234_abcd_9876_4000u64 + i as u64;
    for (i, &reg) in regs.iter().enumerate() {
        a.push(reg)?;
        a.mov(reg, canary(i))?;
    }
    // Both lists have even length. Align RSP before CALL and provide
    // the 32 bytes of outgoing shadow space required by Windows x64.
    let outgoing = if cfg!(windows) { 40 } else { 8 };
    a.sub(rsp, outgoing)?;
    a.mov(rax, entry as usize as u64)?;
    a.call(rax)?;
    a.add(rsp, outgoing)?;
    a.xor(eax, eax)?;
    for (i, &reg) in regs.iter().enumerate().rev() {
        a.mov(rdx, canary(i))?;
        a.cmp(reg, rdx)?;
        a.setne(cl)?;
        a.or(al, cl)?;
        a.pop(reg)?;
    }
    a.ret()?;
    a.assemble(0)
}

#[test]
fn nested_normal_and_trapping_returns_restore_host_registers() {
    let mut f = Function::scalar(24);
    let b = f.entry();
    let args = f.params(b).to_vec();
    let mut sum = f.append(b, Op::Bin(Binary::Div, args[0], args[1])).unwrap();
    for &v in &args[2..] {
        sum = f.append(b, Op::Bin(Binary::Add, sum, v)).unwrap();
    }
    f.set_terminator(b, Terminator::Return(sum));
    for _ in 0..2 {
        let mut caller = Function::scalar(24);
        let b = caller.entry();
        let args = caller.params(b).to_vec();
        let site = caller.add_call(Arc::new(f), &args);
        let result = caller.append(b, Op::Call(site)).unwrap();
        let result = caller
            .append(b, Op::Bin(Binary::Sub, result, args[2]))
            .unwrap();
        caller.set_terminator(b, Terminator::Return(result));
        f = caller;
    }
    for policy in [crate::Inlining::Always, crate::Inlining::Never] {
        let kernel = ScalarKernel::compile_with_inlining(&f, policy).unwrap();
        let probe = Executable::new(&abi_probe(kernel.entry).unwrap()).unwrap();
        // SAFETY: abi_probe emits the same two-argument C ABI, forwards the
        // arguments unchanged and retains stack alignment at the nested call.
        let check =
            unsafe { std::mem::transmute::<*mut std::ffi::c_void, Entry>(probe.address.as_ptr()) };
        for (divisor, expected_status) in [(3, 0), (0, 2)] {
            let mut args = [7u64; 24];
            args[1] = divisor;
            let mut status = 0i32;
            let mut workspace = vec![0u64; kernel.words];
            workspace[0] = std::ptr::from_mut(&mut status) as u64;
            // SAFETY: both executable owners and all three data buffers live
            // across this call. The verified kernel only accepts integer inputs.
            assert_eq!(unsafe { check(args.as_ptr(), workspace.as_mut_ptr()) }, 0);
            assert_eq!(status, expected_status);
        }
    }
}

// Windows preserves the low 128 bits of XMM6..15. Give the kernel ten distinct
// 128-bit canaries and copy the observed values back before returning to Rust.
#[cfg(windows)]
fn xmm_abi_probe(entry: Entry) -> Result<Vec<u8>, IcedError> {
    let regs = [
        xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15,
    ];
    let mut a = CodeAssembler::new(64)?;
    a.endbr64()?;
    // Shadow space, ten original XMM values, and the canary pointer. Subtracting
    // 200 aligns RSP to 16 bytes for the outgoing Windows x64 call.
    a.sub(rsp, 200)?;
    a.mov(qword_ptr(rsp + 192), r8)?;
    for (i, &reg) in regs.iter().enumerate() {
        let off = 16 * i as i32;
        a.movdqu(xmmword_ptr(rsp + 32 + off), reg)?;
        a.movdqu(reg, xmmword_ptr(r8 + off))?;
    }
    a.mov(rax, entry as usize as u64)?;
    a.call(rax)?;
    a.mov(r8, qword_ptr(rsp + 192))?;
    for (i, &reg) in regs.iter().enumerate() {
        let off = 16 * i as i32;
        a.movdqu(xmmword_ptr(r8 + off), reg)?;
        a.movdqu(reg, xmmword_ptr(rsp + 32 + off))?;
    }
    a.add(rsp, 200)?;
    a.ret()?;
    a.assemble(0)
}

#[cfg(windows)]
#[test]
fn outlined_vector_pressure_preserves_windows_xmm_registers_on_normal_and_trapping_exits() {
    use crate::{
        Inlining,
        hir::{Lane, Scale, Simd, Type},
    };
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, Register};

    const VECTORS: usize = 20;
    let lanes = ScalarKernel::vector_bytes() / 8;
    let params = [Type::Ptr, Type::Ptr, Type::I32];
    let mut leaf = Function::new(&params, Type::I32);
    let b = leaf.entry();
    let [src, dst, index] = leaf.params(b).try_into().unwrap();
    // All vectors are live before the first add. The visible store must
    // precede the bounds check even when the child traps.
    let mut values = Vec::new();
    for i in 0..VECTORS {
        let offset = leaf.append(b, Op::Const((i * lanes) as i32)).unwrap();
        let ptr = leaf
            .append(b, Op::PtrOffset(src, offset, Scale::Eight))
            .unwrap();
        values.push(
            leaf.append(b, Op::Simd(Simd::Load(Lane::F64, ptr)))
                .unwrap(),
        );
    }
    let mut sum = values[0];
    for &v in &values[1..] {
        sum = leaf
            .append(b, Op::Simd(Simd::Bin(Binary::Add, sum, v)))
            .unwrap();
    }
    leaf.append(b, Op::Simd(Simd::Store(dst, sum)));
    let one = leaf.append(b, Op::Const(1)).unwrap();
    leaf.append(b, Op::BoundsCheck(index, one));
    leaf.set_terminator(b, Terminator::Return(one));

    let mut root = Function::new(&params, Type::I32);
    let b = root.entry();
    let args = root.params(b).to_vec();
    let site = root.add_call(Arc::new(leaf), &args);
    let result = root.append(b, Op::Call(site)).unwrap();
    let answer = root
        .append(b, Op::Bin(Binary::Add, result, result))
        .unwrap();
    root.set_terminator(b, Terminator::Return(answer));
    let kernel = ScalarKernel::compile_with_inlining(&root, Inlining::Never).unwrap();

    // SAFETY: the owner keeps this readable executable mapping alive. This
    // kernel only uses immediate integer constants, so the mapping is all text.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            kernel.memory.address.as_ptr().cast::<u8>(),
            kernel.code_size(),
        )
    };
    let mut used = [false; 10];
    for inst in Decoder::new(64, bytes, DecoderOptions::NONE) {
        if matches!(inst.mnemonic(), Mnemonic::Addpd | Mnemonic::Vaddpd) {
            for i in 0..inst.op_count() {
                let reg = inst.op_register(i);
                for first in [Register::XMM6, Register::YMM6, Register::ZMM6] {
                    let n = (reg as usize).wrapping_sub(first as usize);
                    if n < used.len() {
                        used[n] = true;
                    }
                }
            }
        }
    }
    assert_eq!(
        used, [true; 10],
        "the vector sum must actually use XMM6..15"
    );

    let probe = Executable::new(&xmm_abi_probe(kernel.entry).unwrap()).unwrap();
    type Probe = unsafe extern "C" fn(*const u64, *mut u64, *mut [u64; 2]) -> u64;
    // SAFETY: xmm_abi_probe implements this exact three-argument Windows ABI,
    // forwards the first two arguments to the live kernel and preserves RAX.
    let invoke =
        unsafe { std::mem::transmute::<*mut std::ffi::c_void, Probe>(probe.address.as_ptr()) };
    let input: Vec<f64> = (0..VECTORS * lanes)
        .map(|i| (9123 + i * 123) as f64)
        .collect();
    let expected: Vec<f64> = (0..lanes)
        .map(|i| input.iter().skip(i).step_by(lanes).sum())
        .collect();
    let canaries: [[u64; 2]; 10] = std::array::from_fn(|i| {
        [
            0xc813_abcd_0000_0123 + i as u64,
            0xf634_5678_9876_5400 - i as u64,
        ]
    });
    for (index, status, value) in [(0, 0, 2), (1, 1, 0)] {
        let mut actual = canaries;
        let mut output = vec![0.0; lanes];
        let mut actual_status = 0i32;
        let mut workspace = vec![0; kernel.words];
        workspace[0] = std::ptr::from_mut(&mut actual_status) as u64;
        let args = [input.as_ptr() as u64, output.as_mut_ptr() as u64, index];
        // SAFETY: both executable owners, the exact-size workspace, all vector
        // inputs, the writable output and the in/out canary array span this call.
        assert_eq!(
            unsafe { invoke(args.as_ptr(), workspace.as_mut_ptr(), actual.as_mut_ptr()) },
            value
        );
        assert_eq!(actual_status, status);
        assert_eq!(output, expected);
        assert_eq!(actual, canaries);
    }
}
