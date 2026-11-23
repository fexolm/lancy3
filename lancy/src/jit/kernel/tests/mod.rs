use super::ScalarKernel;
use crate::hir::{Binary, Function, Op, Terminator};
use crate::{CallError, CompileError, Inlining, SignatureError, Trap};
use std::mem::MaybeUninit;
use std::sync::Arc;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64::call_with_canaries;
#[cfg(target_arch = "x86_64")]
mod x64;
#[cfg(target_arch = "x86_64")]
use x64::call_with_canaries;

#[test]
fn scratch_pointers_require_the_raw_call_contract_transitively() {
    let mut callee = Function::scalar(0);
    let block = callee.entry();
    let address = callee
        .append(
            block,
            Op::CallScratch {
                offset: 0,
                bytes: 8,
            },
        )
        .unwrap();
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
        assert_eq!(
            kernel.call(&[]),
            Err(crate::CallError::Signature(
                SignatureError::PointerSignature
            ))
        );
        assert_eq!(
            kernel.call_words(&[], &mut [0]),
            Err(crate::CallError::Signature(
                SignatureError::PointerSignature
            ))
        );
        // SAFETY: the only access is the initialized first word of this
        // live frame's aligned scratch area. No pointers cross the ABI.
        assert_eq!(unsafe { kernel.call_raw(&[]) }, Ok(19));
    }
}
/// Two nested call levels around a division that traps on a zero divisor,
/// with 24 arguments to put pressure on the nonvolatile registers.
fn nested_division() -> Function {
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
    f
}

#[test]
fn nested_normal_and_trapping_returns_restore_host_registers() {
    let f = nested_division();
    for policy in [Inlining::Always, Inlining::Never] {
        let kernel = ScalarKernel::compile_with_inlining(&f, policy).unwrap();
        for (divisor, expected) in [(3, Ok(0)), (0, Err(CallError::Trap(Trap::DivisionByZero)))] {
            let mut args = [7u64; 24];
            args[1] = divisor;
            let mut workspace = vec![MaybeUninit::uninit(); kernel.words];
            let mut changed = u64::MAX;
            // SAFETY: the verified kernel only takes integer inputs; `args`
            // and `workspace` live across the call, and the canary caller
            // enters the entry once with the C ABI.
            let status = unsafe {
                kernel.invoke_with(args.as_ptr(), &mut workspace, |entry, args, workspace| {
                    changed = call_with_canaries(entry, args, workspace);
                    0
                })
            };
            assert_eq!(status, expected, "{policy:?} divisor {divisor}");
            assert_eq!(
                changed, 0,
                "{policy:?} divisor {divisor}: changed {changed:#b}"
            );
        }
    }
}

#[test]
fn one_kernel_serves_concurrent_callers_and_drops_on_another_thread() {
    let f = nested_division();
    let kernel = ScalarKernel::compile(&f).unwrap();
    let expected = |i: u64| -> Result<i32, CallError> {
        if i == 0 {
            return Err(CallError::Trap(Trap::DivisionByZero));
        }
        // The leaf adds the 22 sevens of args[2..]; both callers subtract one.
        Ok((100 / i + 20 * 7) as i32)
    };
    std::thread::scope(|scope| {
        for t in 0..8u64 {
            let kernel = &kernel;
            scope.spawn(move || {
                for round in 0..200u64 {
                    let divisor = (t + round) % 5;
                    let mut args = [7i32; 24];
                    args[0] = 100;
                    args[1] = divisor as i32;
                    assert_eq!(kernel.call(&args), expected(divisor));
                }
            });
        }
    });
    std::thread::spawn(move || drop(kernel)).join().unwrap();
}

#[test]
fn empty_machine_code_is_not_mapped() {
    assert!(matches!(
        super::Executable::new(&[]),
        Err(CompileError::EmptyCode)
    ));
}
