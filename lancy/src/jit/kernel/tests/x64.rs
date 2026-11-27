use super::super::{Entry, Executable};
use iced_x86::{IcedError, code_asm::*};

/// Call the real JIT entry with host register canaries. Return one if any
/// nonvolatile GPR of the entry's convention changed, after restoring the
/// test caller's own registers. The probe has the entry's convention.
fn abi_probe(entry: Entry) -> Result<Vec<u8>, IcedError> {
    let win64 = matches!(entry, Entry::Win64(_));
    let regs: &[AsmRegister64] = if win64 {
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
    let outgoing = if win64 { 40 } else { 8 };
    a.sub(rsp, outgoing)?;
    a.mov(rax, entry.address() as u64)?;
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

/// Enter `entry` with canaries in the nonvolatile GPRs; nonzero if any changed.
pub(super) fn call_with_canaries(entry: Entry, args: *const u64, workspace: *mut u64) -> u64 {
    let probe = Executable::new(&abi_probe(entry).unwrap()).unwrap();
    // SAFETY: abi_probe emits the entry's two-argument convention, forwards
    // the arguments unchanged and retains stack alignment at the nested
    // call. `probe` owns the mapping until this function returns.
    let check = unsafe { entry.at(probe.address.as_ptr().cast_const().cast()) };
    // SAFETY: forwarded from the caller of the kernel entry.
    unsafe { check.call(args, workspace) }
}

// Windows preserves the low 128 bits of XMM6..15. Give the kernel ten distinct
// 128-bit canaries and copy the observed values back before returning to Rust.
// The probe itself has the Windows convention, which Rust can call on any OS.
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
    a.mov(rax, entry.address() as u64)?;
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

#[test]
fn outlined_vector_pressure_preserves_windows_xmm_registers_on_normal_and_trapping_exits() {
    use super::super::Kernel;
    use crate::{
        Inlining,
        hir::{Binary, Function, Lane, Op, Scale, Terminator, Type},
    };
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, Register};
    use splice_target::CallingConvention;
    use std::sync::Arc;

    const VECTORS: usize = 20;
    let lanes = crate::test_target::width() / 8;
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
            leaf.append(b, Op::Load(Type::Vector(Lane::F64), ptr))
                .unwrap(),
        );
    }
    let mut sum = values[0];
    for &v in &values[1..] {
        sum = leaf.append(b, Op::Bin(Binary::Add, sum, v)).unwrap();
    }
    leaf.append_inst(b, Op::Store(dst, sum)).unwrap();
    let one = leaf.append(b, Op::Const(1)).unwrap();
    leaf.append_inst(b, Op::BoundsCheck(index, one)).unwrap();
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
    let target = crate::test_target::target()
        .with_calling_convention(CallingConvention::Win64)
        .unwrap();
    let options = crate::CompileOptions::new(target).with_inlining(Inlining::Never);
    let kernel = Kernel::compile(&root, &options).unwrap();

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
    type Probe = unsafe extern "win64" fn(*const u64, *mut u64, *mut [u64; 2]) -> u64;
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
        let returned =
            unsafe { invoke(args.as_ptr(), workspace.as_mut_ptr(), actual.as_mut_ptr()) };
        assert_eq!(returned, value);
        assert_eq!(actual_status, status);
        assert_eq!(output, expected);
        assert_eq!(actual, canaries);
    }
}
