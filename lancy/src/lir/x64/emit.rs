//! Emit the allocated stream including regalloc2's ordered edge and spill moves.
use super::{
    compile::{CompiledFunction, compile},
    frame::{CHILD_ARGS, Frame},
    inst::{Address, Alu, Cond, Inst, NONVOLATILE, Rhs},
};
use crate::{
    CompileError,
    hir::{Function, Type, ssa::VerifiedBody},
    module::{FunctionId, FunctionIds, Module},
    support::slotmap::DenseMap,
};
use iced_x86::{IcedError, code_asm::*};
use regalloc2::{Allocation, Block, Edit, Function as _, InstOrEdit, PReg};

type AsmResult = Result<(), IcedError>;

const R64: [AsmRegister64; 16] = [
    rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15,
];
const R32: [AsmRegister32; 16] = [
    eax, ecx, edx, ebx, esp, ebp, esi, edi, r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d,
];
const R8: [AsmRegister8; 16] = [
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
            (Some(s), Some(d)) => a.mov(R64[d.hw_enc()], R64[s.hw_enc()]),
            (Some(s), None) => a.mov(self.spill(to), R64[s.hw_enc()]),
            (None, Some(d)) => a.mov(R64[d.hw_enc()], self.spill(from)),
            (None, None) => unreachable!("regalloc2 resolves memory-to-memory edits"),
        }
    }
}

// iced attaches a label to the next instruction. Its zero-byte pseudo-op
// lets a coalesced copy or a fallthrough block have an address of its own.
fn bind(a: &mut CodeAssembler, mut label: CodeLabel) -> AsmResult {
    a.set_label(&mut label)?;
    a.zero_bytes()
}

pub(crate) fn emit(
    source: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<(Vec<u8>, usize), CompileError> {
    let module = compile(source, ids)?;
    emit_compiled(&module)
}

pub(super) fn emit_compiled(
    module: &Module<CompiledFunction>,
) -> Result<(Vec<u8>, usize), CompileError> {
    let entry = module.call_graph.root();
    let mut a = CodeAssembler::new(64)?;
    let entries = DenseMap::from_fn(module.functions.len(), |_| a.create_label());
    let epilogue = a.create_label();
    a.endbr64()?;
    // Windows passes (argv, workspace) in RCX/RDX; SysV uses RDI/RSI.
    // The allocator models argv's incoming register. R11 is reserved.
    a.mov(r11, if cfg!(windows) { rdx } else { rsi })?;
    let root = &module.functions[entry];
    let saved = root.written & NONVOLATILE;
    let save_base = root.frame.slots - saved.len() as usize;
    let saves = || {
        saved
            .into_iter()
            .enumerate()
            .map(|(i, r): (usize, PReg)| (slot(save_base + i), R64[r.hw_enc()]))
    };
    for (mem, reg) in saves() {
        a.mov(mem, reg)?;
    }
    // The root (last in postorder) follows the native entry prologue.
    for (id, machine) in module.functions.iter().rev() {
        bind(&mut a, entries[id])?;
        Emitter {
            a: &mut a,
            machine,
            root: id == entry,
            entries: &entries,
            epilogue,
            failures: [None; 3],
        }
        .function()?;
    }
    bind(&mut a, epilogue)?;
    for (mem, reg) in saves() {
        a.mov(reg, mem)?;
    }
    a.ret()?;
    let bytes = a.assemble(0)?;
    // Executable::new may install at any address. All label references must
    // fit RIP-relative addressing, including the LEA for each continuation.
    if bytes.len() > i32::MAX as usize {
        return Err(CompileError::BranchOutOfRange {
            offset: 0,
            target: bytes.len(),
        });
    }
    Ok((bytes, root.workspace))
}

struct Emitter<'a> {
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
                            inst @ (Inst::Branch(_) | Inst::BrCmp(..)) => {
                                let succ = body.block_succs(b);
                                let (yes, no) = (succ[0].index(), succ[1].index());
                                let condition = if let Inst::BrCmp(cc, _, rhs) = inst {
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
                index,
                ..
            } => a.mov(q(1), qword_ptr(q(0) + disp(index))),
            Inst::LoadArg {
                argv: None, index, ..
            } => a.mov(q(0), slot(CHILD_ARGS + index)),
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
            Inst::CheckBounds(..) => {
                let fail = *self.failures[0].get_or_insert_with(|| a.create_label());
                rm32!(cmp, r(0), 1)?;
                a.jae(fail)
            }
            Inst::Movsxd(..) => a.movsxd(q(1), r(0)),
            Inst::Lea(_, addr) => a.lea(q(allocs.len() - 1), memory(addr, allocs)),
            Inst::Load(Type::I32, _, addr) => {
                a.mov(r(allocs.len() - 1), dword_ptr(memory(addr, allocs)))
            }
            Inst::Load(Type::Ptr, _, addr) => {
                a.mov(q(allocs.len() - 1), qword_ptr(memory(addr, allocs)))
            }
            Inst::Store(Type::I32, addr, _) => {
                a.mov(dword_ptr(memory(addr, allocs)), r(allocs.len() - 1))
            }
            Inst::Store(Type::Ptr, addr, _) => {
                a.mov(qword_ptr(memory(addr, allocs)), q(allocs.len() - 1))
            }
            Inst::StoreArg(i, _) => a.mov(slot(frame.slots + CHILD_ARGS + i), q(0)),
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
            Inst::Ret(_) if self.root => a.jmp(self.epilogue),
            Inst::Ret(_) => a.jmp(slot(0)),
            Inst::Jump | Inst::Branch(_) | Inst::BrCmp(..) => {
                unreachable!("emitted with block successors")
            }
        }
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
