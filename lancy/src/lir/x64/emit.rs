//! Emit the allocated stream including regalloc2's ordered edge and spill moves.
use super::{Features, inst::class, inst::simd};
use super::{
    compile::{CompiledFunction, compile},
    frame::{CHILD_ARGS, Frame},
    inst::{Address, Alu, Cond, Inst, Quotient, Rhs, SelectCond, VectorMask},
};
use crate::hir::operation::Action;
use crate::{
    CompileError,
    hir::{Binary, Function, Type, ssa::VerifiedBody},
    jit::{DebugFrame, DebugFunction},
    lir::layout,
    module::{FunctionId, FunctionIds, Module},
    support::slotmap::DenseMap,
};
use iced_x86::{BlockEncoderOptions, IcedError, code_asm::*};
use regalloc2::RegClass;
use regalloc2::{Allocation, Edit, Function as _, InstOrEdit, VReg};
use simd::{K, XMM, YMM, ZMM, opmask};

pub(super) type Constants = Vec<(CodeLabel, [u8; 64], usize)>;
type AsmResult = Result<(), IcedError>;

fn allocate_mask(mask: VectorMask<VReg>, regs: &[Allocation]) -> (VectorMask<usize>, usize) {
    let mut next = 0;
    let mask = mask.map(|_| {
        let reg = gpr(regs[next]);
        next += 1;
        reg
    });
    (mask, next)
}

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
            (Some(s), Some(d)) if s.class() == RegClass::Float => {
                if Features::host().bw {
                    a.kmovq(K[d.hw_enc()], K[s.hw_enc()])
                } else {
                    a.kmovw(K[d.hw_enc()], K[s.hw_enc()])
                }
            }
            (Some(s), None) if s.class() == RegClass::Float => {
                if Features::host().bw {
                    a.kmovq(self.spill(to), K[s.hw_enc()])
                } else {
                    a.kmovw(word_ptr(self.spill(to)), K[s.hw_enc()])
                }
            }
            (None, Some(d)) if d.class() == RegClass::Float => {
                if Features::host().bw {
                    a.kmovq(K[d.hw_enc()], self.spill(from))
                } else {
                    a.kmovw(K[d.hw_enc()], word_ptr(self.spill(from)))
                }
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
    pub debug: Vec<DebugFunction>,
}

pub(crate) fn emit(
    source: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<Code, CompileError> {
    emit_mode(&compile(source, ids)?, false)
}

pub(crate) fn emit_debug(
    source: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<Code, CompileError> {
    emit_mode(&compile(source, ids)?, true)
}

#[cfg(test)]
pub(super) fn emit_compiled(module: &Module<CompiledFunction>) -> Result<Code, CompileError> {
    emit_mode(module, false)
}

// The encoder translates these instruction indices after shortening branches.
fn position(a: &mut CodeAssembler) -> Result<u64, IcedError> {
    let index = a.instructions().len() as u64;
    a.zero_bytes()?;
    Ok(index)
}

fn emit_mode(module: &Module<CompiledFunction>, debug: bool) -> Result<Code, CompileError> {
    let _span = tracing::debug_span!(target: "splice::compile", "emit_machine_code").entered();
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
    let restore = |a: &mut CodeAssembler| {
        bind(a, epilogue)?;
        for (mem, reg) in saves() {
            if reg.class() == RegClass::Vector {
                simd::load(a, reg.hw_enc(), mem, 16)?;
            } else {
                a.mov(R64[reg.hw_enc()], mem)?;
            }
        }
        a.ret()
    };
    let mut constants = Vec::new();
    let mut debug_functions = Vec::new();
    // The root (last in postorder) follows the native entry prologue.
    for (id, machine) in module.functions.iter().rev() {
        bind(&mut a, entries[id])?;
        let is_root = id == entry;
        let start = if debug { position(&mut a)? } else { 0 };
        let mut points = Vec::new();
        Emitter {
            a: &mut a,
            machine,
            constants: &mut constants,
            root: is_root,
            entries: &entries,
            epilogue,
            failures: [None; 3],
            debug,
            points: &mut points,
        }
        .function()?;
        if debug {
            let frame_end = position(&mut a)?;
            if is_root {
                restore(&mut a)?;
            }
            let end = position(&mut a)?;
            let (cfa, saved, workspace_saved) = if is_root {
                let mut offset = (root.frame.slots - words) * 8;
                let workspace_saved = saved
                    .into_iter()
                    .map(|reg| {
                        const GPRS: [u16; 16] =
                            [0, 2, 1, 3, 7, 6, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15];
                        let dwarf = if reg.class() == RegClass::Vector {
                            17 + reg.hw_enc() as u16
                        } else {
                            GPRS[reg.hw_enc()]
                        };
                        let at = offset as i64;
                        offset += super::frame::save_words(reg) * 8;
                        (dwarf, at)
                    })
                    .collect();
                (8, vec![(16, -8)], workspace_saved)
            } else {
                (16, vec![(16, -16), (11, -8)], Vec::new())
            };
            debug_functions.push(DebugFunction {
                start,
                end,
                workspace_offset: machine.frame.debug_base * 8,
                points,
                frame: DebugFrame {
                    start,
                    end: frame_end,
                    cfa_from_sp: cfa,
                    saved,
                    workspace_saved,
                },
            });
        }
    }
    if !debug {
        restore(&mut a)?;
    }
    #[cfg(test)]
    let pool_bytes: usize = constants.iter().map(|(_, _, size)| size).sum();
    for (label, bits, size) in constants {
        bind(&mut a, label)?;
        a.db(&bits[..size])?;
    }
    let options = if debug {
        BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS
    } else {
        0
    };
    let assembly = a.assemble_options(0, options)?.inner;
    if debug {
        let offset = |index: u64| u64::from(assembly.new_instruction_offsets[index as usize]);
        for (i, f) in debug_functions.iter_mut().enumerate() {
            f.start = if i == 0 { 0 } else { offset(f.start) };
            f.end = offset(f.end);
            f.frame.start = offset(f.frame.start);
            f.frame.end = offset(f.frame.end);
            for (_, at) in &mut f.points {
                *at = offset(*at);
            }
        }
    }
    let bytes = assembly.code_buffer;
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
        debug: debug_functions,
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
    debug: bool,
    points: &'a mut Vec<(u32, u64)>,
}

impl Emitter<'_> {
    fn function(&mut self) -> AsmResult {
        let body = &self.machine.body;
        let labels: Vec<_> = (0..body.num_blocks())
            .map(|_| self.a.create_label())
            .collect();
        let order = layout::place(body);
        for (i, &b) in order.iter().enumerate() {
            let next = order.get(i + 1).map(|b| b.index());
            bind(self.a, labels[b.index()])?;
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
                                if Some(target) != next {
                                    self.a.jmp(labels[target])?;
                                }
                            }
                            inst @ (Inst::Branch(_)
                            | Inst::BranchAny(_)
                            | Inst::BranchAll(..)
                            | Inst::BrCmp(..)) => {
                                let succ = body.block_succs(b);
                                let (yes, no) = (succ[0].index(), succ[1].index());
                                let condition = if let Inst::BranchAny(mask) = inst {
                                    simd::test_vector_mask(
                                        self.a,
                                        super::Features::host(),
                                        allocate_mask(mask, allocs).0,
                                    )?;
                                    Cond::Ne
                                } else if let Inst::BranchAll(lane, mask) = inst {
                                    let cpu = Features::host();
                                    if mask.class() == RegClass::Float {
                                        opmask::test_all(self.a, cpu, lane, gpr(allocs[0]))?;
                                    } else {
                                        simd::test_all(self.a, cpu, lane, gpr(allocs[0]))?;
                                    }
                                    Cond::Eq
                                } else if let Inst::BrCmp(cc, _, rhs) = inst {
                                    cmp(self.a, &self.machine.frame, allocs, rhs)?;
                                    cc
                                } else {
                                    let r = R32[gpr(allocs[0])];
                                    self.a.test(r, r)?;
                                    Cond::Ne
                                };
                                if Some(yes) == next {
                                    jcc(self.a, condition.inverse(), labels[no])?;
                                } else {
                                    jcc(self.a, condition, labels[yes])?;
                                    if Some(no) != next {
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
                    if self.debug {
                        self.a.mov(r10, slot(1))?;
                        self.a.mov(r10, qword_ptr(r10))?;
                        self.a.mov(dword_ptr(r10), (i + 1) as i32)?;
                        self.a.xor(eax, eax)?;
                        self.a.ret()?;
                        continue;
                    }
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
            Inst::ExtractLane(lane, _, _, index) => simd::packed::rearrange::extract_lane(
                a,
                Features::host(),
                lane,
                reg(1),
                reg(0),
                index,
            ),
            Inst::ShiftDwords(_, _, count) => simd::packed::rearrange::shift_dwords(
                a,
                Features::host().bytes,
                reg(1),
                reg(0),
                count,
            ),
            Inst::PairedShuffle(form, ..) => form.emit(
                a,
                self.constants,
                Features::host().bytes,
                [reg(2), reg(0), reg(1)],
            ),
            Inst::DwordLane {
                lane,
                checked,
                replacement,
                temp,
                ..
            } => {
                let width = Features::host().bytes;
                if checked {
                    let fail = *self.failures[0].get_or_insert_with(|| a.create_label());
                    a.cmp(r(1), (width / 4) as i32)?;
                    a.jae(fail)?;
                }
                let registers = [reg(2 + usize::from(replacement.is_some())), reg(0), reg(1)];
                if replacement.is_some() {
                    simd::packed::rearrange::insert_dword(
                        a,
                        self.constants,
                        width,
                        lane,
                        registers,
                        reg(2),
                        temp.map(|_| reg(4)),
                    )
                } else {
                    simd::packed::rearrange::extract_dword(a, width, lane, registers)
                }
            }
            Inst::VectorShiftImm(kind, lane, _, _, count) => {
                simd::packed::integer::shift_dword_bits(
                    a,
                    Features::host(),
                    kind,
                    lane,
                    reg(1),
                    reg(0),
                    count,
                )
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
            Inst::SelectI32 { cond, .. } => {
                let cc = match cond {
                    SelectCond::Nonzero(_) => {
                        a.test(r(2), r(2))?;
                        Cond::Ne
                    }
                    SelectCond::Compare(cc, _, rhs) => {
                        cmp(a, frame, &allocs[2..], rhs)?;
                        cc
                    }
                };
                let dst = r(allocs.len() - 1);
                match cc {
                    Cond::Eq => rm32!(cmove, dst, 1),
                    Cond::Ne => rm32!(cmovne, dst, 1),
                    Cond::Lt => rm32!(cmovl, dst, 1),
                    Cond::Le => rm32!(cmovle, dst, 1),
                    Cond::Gt => rm32!(cmovg, dst, 1),
                    Cond::Ge => rm32!(cmovge, dst, 1),
                }
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
            Inst::CheckMask(mask) => {
                let fail = *self.failures[0].get_or_insert_with(|| a.create_label());
                simd::test_vector_mask(a, Features::host(), allocate_mask(mask, allocs).0)?;
                a.jne(fail)
            }
            Inst::AnyMask(_, mask) => {
                let (mask, dst) = allocate_mask(mask, allocs);
                simd::test_vector_mask(a, Features::host(), mask)?;
                a.mov(r(dst), 0)?;
                a.setne(R8[reg(dst)])?;
                a.neg(r(dst))
            }
            Inst::AllMask(lane, ..) => {
                opmask::test_all(a, Features::host(), lane, reg(0))?;
                a.mov(r(1), 0)?;
                a.sete(R8[reg(1)])?;
                a.neg(r(1))
            }
            Inst::SelectMask { lane, mask, .. } => {
                let (mask, first) = allocate_mask(mask, allocs);
                simd::blend_comparison(
                    a,
                    Features::host(),
                    lane,
                    mask,
                    reg(first + 2),
                    reg(first),
                    reg(first + 1),
                )
            }
            Inst::MaskCompare(lane, cc, ..) => simd::compare_mask(
                a,
                Features::host().bytes,
                K[reg(2)],
                lane,
                reg(0),
                reg(1),
                cc,
            ),
            Inst::MaskLogic(lane, kind, ..) => opmask::logic(
                a,
                Features::host().bytes / lane.bytes(),
                kind,
                reg(2),
                reg(0),
                reg(1),
            ),
            Inst::MaskNot(lane, ..) => opmask::not(a, Features::host(), lane, reg(1), reg(0)),
            Inst::MaskImmediate(_, bits) => opmask::immediate(a, reg(0), bits),
            Inst::MaskPrefix(..) => {
                a.mov(r10d, -1)?;
                a.bzhi(r10d, r10d, r(0))?;
                a.kmovw(K[reg(1)], r10d)
            }
            Inst::MaskCount(..) => {
                a.kmovw(r(1), K[reg(0)])?;
                a.popcnt(r(1), r(1))
            }
            Inst::MaskCompact(lane, expand, ..) => simd::compact(
                a,
                Features::host().bytes,
                lane,
                expand,
                reg(2),
                reg(0),
                reg(1),
            ),
            Inst::CompactLookup(expand, ..) => simd::packed::compact::emit(
                a,
                Features::host(),
                expand,
                [reg(2), reg(0), reg(1), reg(3)],
            ),
            Inst::CompactStore { .. } => {
                a.kmovw(r(4), K[reg(2)])?;
                a.kmovw(r10d, K[reg(3)])?;
                a.pdep(r10d, r10d, r(4))?;
                a.kmovw(k1, r10d)?;
                let address = ptr(q(0)).k1();
                match Features::host().bytes {
                    16 => a.vpcompressd(address, XMM[reg(1)]),
                    32 => a.vpcompressd(address, YMM[reg(1)]),
                    _ => a.vpcompressd(address, ZMM[reg(1)]),
                }
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
                mask,
                value,
                ..
            } => {
                let i = 1 + usize::from(indices.is_some());
                let (mask, operands) = allocate_mask(mask, &allocs[i..]);
                let first = i + operands;
                simd::MaskedMemory {
                    ty,
                    base: reg(0),
                    index: indices.map(|_| reg(1)),
                    mask,
                    value: value.map(|_| reg(first)),
                    dst: dst.map(|_| reg(first + usize::from(value.is_some()))),
                    temp: temp.map(|_| reg(allocs.len() - 1)),
                }
                .emit(a, super::Features::host())
            }
            Inst::Load(ty, _, addr) => load(a, ty, reg(allocs.len() - 1), memory(addr, allocs)),
            Inst::PackedMemory(kind, lane, _, _, addr) => {
                let src = reg(allocs.len() - 2);
                let dst = reg(allocs.len() - 1);
                let addr = memory(addr, allocs);
                if kind == Binary::Add {
                    return match (Features::host().bytes, lane.bytes()) {
                        (32, 4) => a.vpaddd(YMM[dst], YMM[src], addr),
                        (32, 8) => a.vpaddq(YMM[dst], YMM[src], addr),
                        (64, 4) => a.vpaddd(ZMM[dst], ZMM[src], addr),
                        (64, 8) => a.vpaddq(ZMM[dst], ZMM[src], addr),
                        _ => unreachable!(),
                    };
                }
                match (Features::host().bytes, kind) {
                    (32, Binary::And) => a.vpand(YMM[dst], YMM[src], addr),
                    (32, Binary::Or) => a.vpor(YMM[dst], YMM[src], addr),
                    (32, Binary::Xor) => a.vpxor(YMM[dst], YMM[src], addr),
                    (64, Binary::And) => a.vpandd(ZMM[dst], ZMM[src], addr),
                    (64, Binary::Or) => a.vpord(ZMM[dst], ZMM[src], addr),
                    (64, Binary::Xor) => a.vpxord(ZMM[dst], ZMM[src], addr),
                    _ => unreachable!(),
                }
            }
            Inst::BroadcastMemory(lane, _, addr) => {
                let dst = reg(allocs.len() - 1);
                let addr = memory(addr, allocs);
                match (Features::host().bytes, lane.bytes()) {
                    (32, 4) => a.vpbroadcastd(YMM[dst], addr),
                    (32, 8) => a.vpbroadcastq(YMM[dst], addr),
                    (64, 4) => a.vpbroadcastd(ZMM[dst], addr),
                    (64, 8) => a.vpbroadcastq(ZMM[dst], addr),
                    _ => unreachable!(),
                }
            }
            Inst::BroadcastLane(_, _, lane) => {
                let src = reg(0);
                let dst = reg(1);
                let width = Features::host().bytes;
                let imm = i32::from(lane % 4) * 0x55;
                if width == 16 {
                    return a.pshufd(XMM[dst], XMM[src], imm);
                }
                let chunk = i32::from(lane / 4);
                let low = if chunk == 0 {
                    XMM[src]
                } else {
                    if width == 32 {
                        a.vextracti128(XMM[dst], YMM[src], chunk)?;
                    } else {
                        a.vextracti32x4(XMM[dst], ZMM[src], chunk)?;
                    }
                    XMM[dst]
                };
                let lane = if imm == 0 {
                    low
                } else {
                    a.vpshufd(XMM[dst], low, imm)?;
                    XMM[dst]
                };
                if width == 32 {
                    a.vpbroadcastd(YMM[dst], lane)
                } else {
                    a.vpbroadcastd(ZMM[dst], lane)
                }
            }
            Inst::Store(ty, addr, _) => store(a, ty, memory(addr, allocs), reg(allocs.len() - 1)),
            Inst::DebugStore(ty, offset, _) => store(
                a,
                ty,
                ptr(r11 + (disp(frame.debug_base) + offset as i32)),
                reg(0),
            ),
            Inst::DebugPoint(id) => {
                if self.debug {
                    self.points.push((id, position(a)?));
                }
                a.nop()
            }
            Inst::StoreArg(i, ty, _) => store(a, ty, slot(frame.slots + CHILD_ARGS + i), reg(0)),
            Inst::Call { target, .. } => {
                if self.root {
                    a.mov(slot(frame.slots + 1), r11)?;
                } else {
                    a.mov(r10, slot(1))?;
                    a.mov(slot(frame.slots + 1), r10)?;
                }
                if self.debug {
                    // Leave an actual return address and caller workspace on
                    // the stack, with the same alignment at child entry.
                    a.push(r11)?;
                    a.add(r11, disp(frame.slots))?;
                    a.call(self.entries[target])?;
                    a.pop(r11)?;
                    if self.root {
                        a.mov(r10, slot(0))?;
                        a.cmp(dword_ptr(r10), 0)?;
                        return a.jne(self.epilogue);
                    }
                    let ok = a.create_label();
                    a.mov(r10, slot(1))?;
                    a.mov(r10, qword_ptr(r10))?;
                    a.cmp(dword_ptr(r10), 0)?;
                    a.je(ok)?;
                    a.ret()?;
                    return bind(a, ok);
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
            Inst::Ret(_) if self.debug => a.ret(),
            Inst::Ret(_) => a.jmp(slot(0)),
            Inst::Jump
            | Inst::Branch(_)
            | Inst::BranchAny(_)
            | Inst::BranchAll(..)
            | Inst::BrCmp(..) => {
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
