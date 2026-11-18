use super::{Entry, Executable, ScalarKernel};
use crate::hir::{Binary, Function, Op, Terminator};
use iced_x86::{IcedError, code_asm::*};
use std::sync::Arc;

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
    let kernel = ScalarKernel::compile(&f).unwrap();
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
