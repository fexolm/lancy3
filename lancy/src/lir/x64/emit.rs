//! Emit the allocated stream including regalloc2's ordered edge and spill moves.
use super::{Features, inst::class, inst::simd};
use super::{
    compile::{CompiledFunction, compile},
    frame::{CHILD_ARGS, Frame},
    inst::{Address, Alu, Cond, Inst, Quotient, Rhs},
};
use crate::hir::operation::Action;
use crate::{
    CompileError,
    hir::{Function, Type, ssa::VerifiedBody},
    module::{FunctionId, FunctionIds, Module},
    support::slotmap::DenseMap,
};
use iced_x86::{IcedError, code_asm::*};
use regalloc2::RegClass;
use regalloc2::{Allocation, Block, Edit, Function as _, InstOrEdit};

pub(super) type Constants = Vec<(CodeLabel, [u8; 64], usize)>;
type AsmResult = Result<(), IcedError>;

pub(super) const R64: [AsmRegister64; 16] = [
    rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15,
];
pub(super) const R32: [AsmRegister32; 16] = [
    eax, ecx, edx, ebx, esp, ebp, esi, edi, r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d,
];
pub(super) const R16: [AsmRegister16; 16] = [
    ax, cx, dx, bx, sp, bp, si, di, r8w, r9w, r10w, r11w, r12w, r13w, r14w, r15w,
];
pub(super) const R8: [AsmRegister8; 16] = [
    al, cl, dl, bl, spl, bpl, sil, dil, r8b, r9b, r10b, r11b, r12b, r13b, r14b, r15b,
];

fn gpr(a: Allocation) -> usize {
    a.as_reg().expect("register-constrained operand").hw_enc()
}
// Compilation bounds both the complete workspace and root argument table by disp32.
fn disp(index: usize) -> i32 {
    (index * 8) as i32
}
fn slot(index: usize) -> AsmMemoryOperand {
    qword_ptr(r11 + disp(index))
}

impl Frame {
    fn spill(&self, a: Allocation) -> AsmMemoryOperand {
        slot(self.spill_base + a.as_stack().expect("allocated spill").index())
    }

    fn emit_move(&self, a: &mut CodeAssembler, from: Allocation, to: Allocation) -> AsmResult {
        if from == to {
            return Ok(());
        }
        match (from.as_reg(), to.as_reg()) {
            (Some(s), Some(d)) if s.class() == RegClass::Vector => {
                simd::copy(a, d.hw_enc(), s.hw_enc(), Features::host().bytes)
            }
            (Some(s), None) if s.class() == RegClass::Vector => {
                simd::store(a, self.spill(to), s.hw_enc(), Features::host().bytes)
            }
            (None, Some(d)) if d.class() == RegClass::Vector => {
                simd::load(a, d.hw_enc(), self.spill(from), Features::host().bytes)
            }
            (Some(s), Some(d)) => a.mov(R64[d.hw_enc()], R64[s.hw_enc()]),
            (Some(s), None) => a.mov(self.spill(to), R64[s.hw_enc()]),
            (None, Some(d)) => a.mov(R64[d.hw_enc()], self.spill(from)),
            (None, None) => unreachable!("regalloc2 resolves memory-to-memory edits"),
        }
    }
}

// iced attaches a label to the next instruction. Its zero-byte pseudo-op
// lets a coalesced copy or a fallthrough block have an address of its own.
pub(super) fn bind(a: &mut CodeAssembler, mut label: CodeLabel) -> AsmResult {
    a.set_label(&mut label)?;
    a.zero_bytes()
}

pub(crate) struct Code {
    pub bytes: Vec<u8>,
    #[cfg(test)]
    pub text_len: usize,
    pub words: usize,
}

pub(crate) fn emit(
    source: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<Code, CompileError> {
    let module = compile(source, ids)?;
    emit_compiled(&module)
}

pub(super) fn emit_compiled(module: &Module<CompiledFunction>) -> Result<Code, CompileError> {
    let entry = module.call_graph.root();
    let mut a = CodeAssembler::new(64)?;
    let entries = DenseMap::from_fn(module.functions.len(), |_| a.create_label());
    let epilogue = a.create_label();
    a.endbr64()?;
    // Windows passes (argv, workspace) in RCX/RDX; SysV uses RDI/RSI.
    // The allocator models argv's incoming register. R11 is reserved.
    a.mov(r11, if cfg!(windows) { rdx } else { rsi })?;
    let root = &module.functions[entry];
    let saved = root.saved;
    let words: usize = saved.into_iter().map(super::frame::save_words).sum();
    let saves = || {
        saved
            .into_iter()
            .scan(root.frame.slots - words, |off, reg| {
                let at = slot(*off);
                *off += super::frame::save_words(reg);
                Some((at, reg))
            })
    };
    for (mem, reg) in saves() {
        if reg.class() == RegClass::Vector {
            simd::store(&mut a, mem, reg.hw_enc(), 16)?;
        } else {
            a.mov(mem, R64[reg.hw_enc()])?;
        }
    }
    let mut constants = Vec::new();
    // The root (last in postorder) follows the native entry prologue.
    for (id, machine) in module.functions.iter().rev() {
        bind(&mut a, entries[id])?;
        Emitter {
            a: &mut a,
            machine,
            constants: &mut constants,
            root: id == entry,
            entries: &entries,
            epilogue,
            failures: [None; 3],
        }
        .function()?;
    }
    bind(&mut a, epilogue)?;
    for (mem, reg) in saves() {
        if reg.class() == RegClass::Vector {
            simd::load(&mut a, reg.hw_enc(), mem, 16)?;
        } else {
            a.mov(R64[reg.hw_enc()], mem)?;
        }
    }
    a.ret()?;
    #[cfg(test)]
    let pool_bytes: usize = constants.iter().map(|(_, _, size)| size).sum();
    for (label, bits, size) in constants {
        bind(&mut a, label)?;
        a.db(&bits[..size])?;
    }
    let bytes = a.assemble(0)?;
    // Executable::new may install at any address. All label references must
    // fit RIP-relative addressing, including the LEA for each continuation.
    if bytes.len() > i32::MAX as usize {
        return Err(CompileError::BranchOutOfRange {
            offset: 0,
            target: bytes.len(),
        });
    }
    Ok(Code {
        #[cfg(test)]
        text_len: bytes.len() - pool_bytes,
        bytes,
        words: root.workspace,
    })
}

struct Emitter<'a> {
    constants: &'a mut Constants,
    a: &'a mut CodeAssembler,
    machine: &'a CompiledFunction,
    root: bool,
    entries: &'a DenseMap<FunctionId, CodeLabel>,
    epilogue: CodeLabel,
    failures: [Option<CodeLabel>; 3],
}

impl Emitter<'_> {
    fn function(&mut self) -> AsmResult {
        let body = &self.machine.body;
        let labels: Vec<_> = (0..body.num_blocks())
            .map(|_| self.a.create_label())
            .collect();
        for (i, &label) in labels.iter().enumerate() {
            let b = Block::new(i);
            bind(self.a, label)?;
            for op in self.machine.allocation.block_insts_and_edits(body, b) {
                match op {
                    InstOrEdit::Edit(&Edit::Move { from, to }) => {
                        self.machine.frame.emit_move(self.a, from, to)?
                    }
                    InstOrEdit::Inst(j) => {
                        let allocs = self.machine.allocation.inst_allocs(j);
                        match body.insts()[j.index()] {
                            Inst::Jump => {
                                let target = body.block_succs(b)[0].index();
                                if target != i + 1 {
                                    self.a.jmp(labels[target])?;
                                }
                            }
                            inst @ (Inst::Branch(_) | Inst::BranchAny(_) | Inst::BrCmp(..)) => {
                                let succ = body.block_succs(b);
                                let (yes, no) = (succ[0].index(), succ[1].index());
                                let condition = if let Inst::BranchAny(_) = inst {
                                    simd::test_any(
                                        self.a,
                                        super::Features::host(),
                                        gpr(allocs[0]),
                                    )?;
                                    Cond::Ne
                                } else if let Inst::BrCmp(cc, _, rhs) = inst {
                                    cmp(self.a, &self.machine.frame, allocs, rhs)?;
                                    cc
                                } else {
                                    let r = R32[gpr(allocs[0])];
                                    self.a.test(r, r)?;
                                    Cond::Ne
                                };
                                if yes == i + 1 {
                                    jcc(self.a, condition.inverse(), labels[no])?;
                                } else {
                                    jcc(self.a, condition, labels[yes])?;
                                    if no != i + 1 {
                                        self.a.jmp(labels[no])?;
                                    }
                                }
                            }
                            inst => self.instruction(inst, allocs)?,
                        }
                    }
                }
            }
        }
        for (i, label) in self.failures.into_iter().enumerate() {
            if let Some(label) = label {
                bind(self.a, label)?;
                // A trap can bypass every internal continuation. Recover the
                // root workspace, report through the live status pointer
                // supplied by ScalarKernel::invoke, and restore the host ABI.
                if !self.root {
                    self.a.mov(r11, slot(1))?;
                }
                self.a.mov(r10, slot(0))?;
                self.a.mov(dword_ptr(r10), (i + 1) as i32)?;
                self.a.xor(eax, eax)?;
                self.a.jmp(self.epilogue)?;
            }
        }
        Ok(())
    }

    fn lanes(
        &mut self,
        action: Action,
        input: Type,
        output: Type,
        args: [Option<Type>; 3],
        allocs: &[Allocation],
    ) -> AsmResult {
        use super::inst::lanes::{self, FRAME, INPUT, OUTPUT, WORKSPACE};
        let a = &mut self.a;
        a.sub(rsp, FRAME)?;
        a.mov(qword_ptr(rsp + WORKSPACE), r11)?;
        let mut allocs = allocs.iter().copied();
        for (arg, offset) in args.into_iter().zip(INPUT) {
            if let Some(ty) = arg {
                store(a, ty, ptr(rsp + offset), gpr(allocs.next().unwrap()))?;
            }
        }
        let mut faults = [None; 3];
        lanes::emit(a, action, input, output, &mut faults)?;
        load(a, output, gpr(allocs.next().unwrap()), ptr(rsp + OUTPUT))?;
        a.mov(r11, qword_ptr(rsp + WORKSPACE))?;
        a.add(rsp, FRAME)?;
        if faults.iter().any(Option::is_some) {
            let done = a.create_label();
            a.jmp(done)?;
            for (i, local) in faults.into_iter().enumerate() {
                if let Some(label) = local {
                    bind(a, label)?;
                    a.mov(r11, qword_ptr(rsp + WORKSPACE))?;
                    a.add(rsp, FRAME)?;
                    let fail = *self.failures[i].get_or_insert_with(|| a.create_label());
                    a.jmp(fail)?;
                }
            }
            bind(a, done)?;
        }
        Ok(())
    }

    fn instruction(&mut self, inst: Inst, allocs: &[Allocation]) -> AsmResult {
        let a = &mut *self.a;
        let frame = &self.machine.frame;
        let reg = |i: usize| gpr(allocs[i]);
        let r = |i| R32[reg(i)];
        let q = |i| R64[reg(i)];
        // x86's r/m operand can directly consume a spilled RHS.
        macro_rules! rm32 {
            ($op:ident, $lhs:expr, $index:expr) => {{
                let rhs = allocs[$index];
                if let Some(p) = rhs.as_reg() {
                    a.$op($lhs, R32[p.hw_enc()])
                } else {
                    a.$op(
                        $lhs,
                        dword_ptr(r11 + disp(frame.spill_base + rhs.as_stack().unwrap().index())),
                    )
                }
            }};
        }
        match inst {
            Inst::Argv(_) | Inst::Copy(..) => Ok(()), // incoming fixed def / tied operands
            Inst::LoadArg {
                argv: Some(_),
                ty,
                index,
                ..
            } => load(a, ty, reg(1), ptr(q(0) + disp(index))),
            Inst::LoadArg {
                argv: None,
                ty,
                index,
                ..
            } => load(a, ty, reg(0), slot(CHILD_ARGS + index)),
            Inst::Native(native, ..) => {
                let mut regs = [0; 10];
                for (dst, &loc) in regs.iter_mut().zip(allocs) {
                    *dst = gpr(loc);
                }
                let (operands, temps) =
                    regs[..allocs.len()].split_at(allocs.len() - native.temporaries());
                native.emit_allocated(a, operands, temps, Some(self.constants))
            }
            Inst::Lanes {
                action,
                input,
                output,
                args,
                ..
            } => self.lanes(
                action,
                input,
                output,
                args.map(|arg| arg.map(|(ty, _)| ty)),
                allocs,
            ),
            Inst::Literal(l, bits, _) => match l.bytes() {
                16 => {
                    let label = a.create_label();
                    let mut data = [0; 64];
                    data[..16].copy_from_slice(&bits.to_le_bytes());
                    self.constants.push((label, data, 16));
                    simd::load(a, reg(0), ptr(label), 16)
                }
                8 if l.float() => {
                    a.mov(r10, bits as u64)?;
                    a.movq(simd::XMM[reg(0)], r10)
                }
                4 if l.float() => {
                    a.mov(r10d, bits as u32)?;
                    a.movd(simd::XMM[reg(0)], r10d)
                }
                8 => a.mov(q(0), bits as u64),
                size => a.mov(r(0), (bits & (u128::MAX >> (128 - size * 8))) as u32),
            },
            Inst::Iota(l, _) => {
                let label = a.create_label();
                let width = Features::host().bytes;
                let bits = super::super::constants::iota_bytes(l, width);
                self.constants.push((label, bits, width));
                simd::load(a, reg(0), ptr(label), width)
            }
            Inst::BitMove(from, to, ..) => match (class(to), from.bytes(Features::host().bytes)) {
                (RegClass::Vector, 4) => a.movd(simd::XMM[reg(1)], r(0)),
                (RegClass::Vector, 8) => a.movq(simd::XMM[reg(1)], q(0)),
                (RegClass::Int, 4) => a.movd(r(1), simd::XMM[reg(0)]),
                (RegClass::Int, 8) => a.movq(q(1), simd::XMM[reg(0)]),
                _ => unreachable!(),
            },
            Inst::IntBinary(op, ty, ..) => {
                use crate::hir::Binary;
                let size = ty.lane().unwrap().bytes();
                macro_rules! integer {
                    ($mn:ident) => {
                        if size == 8 {
                            a.$mn(q(0), q(1))
                        } else {
                            a.$mn(r(0), r(1))
                        }
                    };
                }
                match op {
                    Binary::Add => integer!(add),
                    Binary::Sub => integer!(sub),
                    Binary::Mul => integer!(imul_2),
                    Binary::And => integer!(and),
                    Binary::Or => integer!(or),
                    Binary::Xor => integer!(xor),
                    _ => unreachable!(),
                }?;
                match size {
                    1 => a.movzx(r(0), R8[reg(0)]),
                    2 => a.movzx(r(0), R16[reg(0)]),
                    _ => Ok(()),
                }
            }
            Inst::Imm(_, n) => a.mov(r(0), n),
            Inst::Alu(op, _, _, Rhs::Imm(n)) => match op {
                Alu::Add => a.add(r(0), n),
                Alu::Sub => a.sub(r(0), n),
                Alu::And => a.and(r(0), n),
                Alu::Or => a.or(r(0), n),
                Alu::Xor => a.xor(r(0), n),
                Alu::Mul => a.imul_3(r(0), r(0), n),
                Alu::Min | Alu::Max => unreachable!("cmov requires a register or memory"),
            },
            Inst::Alu(op, _, _, Rhs::Reg(_)) => match op {
                Alu::Add => rm32!(add, r(2), 1),
                Alu::Sub => rm32!(sub, r(2), 1),
                Alu::And => rm32!(and, r(2), 1),
                Alu::Or => rm32!(or, r(2), 1),
                Alu::Xor => rm32!(xor, r(2), 1),
                Alu::Mul => rm32!(imul_2, r(2), 1),
                Alu::Min | Alu::Max => {
                    rm32!(cmp, r(2), 1)?;
                    if matches!(op, Alu::Min) {
                        rm32!(cmovg, r(2), 1)
                    } else {
                        rm32!(cmovl, r(2), 1)
                    }
                }
            },
            Inst::Shift { left, count, .. } => match (left, count) {
                (true, Rhs::Reg(_)) => a.shl(r(0), cl),
                (false, Rhs::Reg(_)) => a.sar(r(0), cl),
                (true, Rhs::Imm(n)) => a.shl(r(0), n & 31),
                (false, Rhs::Imm(n)) => a.sar(r(0), n & 31),
            },
            Inst::Compare(cc, _, _, rhs) => {
                cmp(a, frame, allocs, rhs)?;
                let dst = allocs.len() - 1;
                let byte = R8[reg(dst)];
                match cc {
                    Cond::Eq => a.sete(byte),
                    Cond::Ne => a.setne(byte),
                    Cond::Lt => a.setl(byte),
                    Cond::Le => a.setle(byte),
                    Cond::Gt => a.setg(byte),
                    Cond::Ge => a.setge(byte),
                }?;
                a.movzx(r(dst), byte)?;
                a.neg(r(dst))
            }
            Inst::Cdq { .. } => a.cdq(),
            Inst::IDiv { .. } => {
                let zero = *self.failures[1].get_or_insert_with(|| a.create_label());
                let overflow = *self.failures[2].get_or_insert_with(|| a.create_label());
                let divide = a.create_label();
                a.test(r(2), r(2))?;
                a.jz(zero)?;
                a.cmp(r(2), -1)?;
                a.jne(divide)?;
                a.cmp(eax, i32::MIN)?;
                a.je(overflow)?;
                bind(a, divide)?;
                a.idiv(r(2))
            }
            Inst::DivImm {
                remainder,
                divisor,
                quotient,
                ..
            } => {
                let input = r(0);
                let dst = r(1);
                match quotient {
                    Quotient::PowerOfTwo(shift) => {
                        let bias = ((1u32 << shift) - 1) as i32;
                        a.lea(dst, ptr(q(0) + bias))?;
                        a.test(input, input)?;
                        a.cmovns(dst, input)?;
                        a.sar(dst, shift)?;
                    }
                    Quotient::Reciprocal { multiplier, shift } => {
                        a.movsxd(q(1), input)?;
                        if let Ok(multiplier) = i32::try_from(multiplier) {
                            a.imul_3(q(1), q(1), multiplier)?;
                        } else {
                            // R10 is reserved for scratch and cannot hold an input.
                            a.mov(r10d, multiplier)?;
                            a.imul_2(q(1), r10)?;
                        }
                        a.sar(q(1), shift)?;
                        a.mov(r10d, input)?;
                        a.shr(r10d, 31)?;
                        a.add(dst, r10d)?;
                    }
                }
                if remainder {
                    // The product used the positive magnitude. Its wrapped
                    // negation also covers a divisor of i32::MIN.
                    a.imul_3(dst, dst, (divisor.unsigned_abs() as i32).wrapping_neg())?;
                    a.add(dst, input)?;
                } else if divisor < 0 {
                    a.neg(dst)?;
                }
                Ok(())
            }
            Inst::CheckMask(_) => {
                let fail = *self.failures[0].get_or_insert_with(|| a.create_label());
                simd::test_any(a, super::Features::host(), reg(0))?;
                a.jne(fail)
            }
            Inst::CheckBounds(..) => {
                let fail = *self.failures[0].get_or_insert_with(|| a.create_label());
                rm32!(cmp, r(0), 1)?;
                a.jae(fail)
            }
            Inst::Movsxd(..) => a.movsxd(q(1), r(0)),
            Inst::Lea(_, addr) => a.lea(q(allocs.len() - 1), memory(addr, allocs)),
            Inst::Masked {
                ty,
                indices,
                dst,
                temp,
                ..
            } => {
                let i = 1 + usize::from(indices.is_some());
                simd::MaskedMemory {
                    ty,
                    base: reg(0),
                    index: indices.map(|_| reg(1)),
                    mask: reg(i),
                    value: reg(i + 1),
                    dst: dst.map(|_| reg(i + 2)),
                    temp: temp.map(|_| reg(allocs.len() - 1)),
                }
                .emit(a, super::Features::host())
            }
            Inst::Load(ty, _, addr) => load(a, ty, reg(allocs.len() - 1), memory(addr, allocs)),
            Inst::Store(ty, addr, _) => store(a, ty, memory(addr, allocs), reg(allocs.len() - 1)),
            Inst::StoreArg(i, ty, _) => store(a, ty, slot(frame.slots + CHILD_ARGS + i), reg(0)),
            Inst::Call { target, .. } => {
                if self.root {
                    a.mov(slot(frame.slots + 1), r11)?;
                } else {
                    a.mov(r10, slot(1))?;
                    a.mov(slot(frame.slots + 1), r10)?;
                }
                let resume = a.create_label();
                a.add(r11, disp(frame.slots))?;
                a.lea(r10, ptr(resume))?;
                a.mov(slot(0), r10)?;
                a.jmp(self.entries[target])?;
                bind(a, resume)?;
                a.endbr64()?;
                a.sub(r11, disp(frame.slots))
            }
            Inst::Ret(_) if self.root => {
                let ty = self.machine.result;
                match ty.bytes(Features::host().bytes) {
                    16.. => store(a, ty, slot(1), 0)?,
                    4 if ty == Type::F32 => a.movd(eax, simd::XMM[0])?,
                    8 if ty == Type::F64 => a.movq(rax, simd::XMM[0])?,
                    _ => {}
                }
                a.jmp(self.epilogue)
            }
            Inst::Ret(_) => a.jmp(slot(0)),
            Inst::Jump | Inst::Branch(_) | Inst::BranchAny(_) | Inst::BrCmp(..) => {
                unreachable!("emitted with block successors")
            }
        }
    }
}

fn load(a: &mut CodeAssembler, ty: Type, d: usize, mem: AsmMemoryOperand) -> AsmResult {
    let size = ty.bytes(Features::host().bytes);
    if class(ty) == RegClass::Vector {
        return simd::load(a, d, mem, size);
    }
    match size {
        1 => a.movzx(R32[d], byte_ptr(mem)),
        2 => a.movzx(R32[d], word_ptr(mem)),
        4 => a.mov(R32[d], dword_ptr(mem)),
        8 => a.mov(R64[d], qword_ptr(mem)),
        _ => unreachable!(),
    }
}
fn store(a: &mut CodeAssembler, ty: Type, mem: AsmMemoryOperand, s: usize) -> AsmResult {
    let size = ty.bytes(Features::host().bytes);
    if class(ty) == RegClass::Vector {
        return simd::store(a, mem, s, size);
    }
    match size {
        1 => a.mov(byte_ptr(mem), R8[s]),
        2 => a.mov(word_ptr(mem), R16[s]),
        4 => a.mov(dword_ptr(mem), R32[s]),
        8 => a.mov(qword_ptr(mem), R64[s]),
        _ => unreachable!(),
    }
}

fn memory(addr: Address, allocs: &[Allocation]) -> AsmMemoryOperand {
    let base = R64[gpr(allocs[0])];
    match addr.index {
        Some((_, scale)) => ptr(base + R64[gpr(allocs[1])] * (1i32 << scale as u8) + addr.disp),
        None => ptr(base + addr.disp),
    }
}

fn cmp(a: &mut CodeAssembler, frame: &Frame, allocs: &[Allocation], rhs: Rhs) -> AsmResult {
    let lhs = R32[gpr(allocs[0])];
    match rhs {
        Rhs::Imm(n) => a.cmp(lhs, n),
        Rhs::Reg(_) => match allocs[1].as_reg() {
            Some(p) => a.cmp(lhs, R32[p.hw_enc()]),
            None => a.cmp(
                lhs,
                dword_ptr(r11 + disp(frame.spill_base + allocs[1].as_stack().unwrap().index())),
            ),
        },
    }
}

fn jcc(a: &mut CodeAssembler, cc: Cond, label: CodeLabel) -> AsmResult {
    match cc {
        Cond::Eq => a.je(label),
        Cond::Ne => a.jne(label),
        Cond::Lt => a.jl(label),
        Cond::Le => a.jle(label),
        Cond::Gt => a.jg(label),
        Cond::Ge => a.jge(label),
    }
}
