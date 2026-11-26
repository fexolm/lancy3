//! Emit the allocated stream including regalloc2's ordered edge and spill moves.
use super::{Cpu, Machine, abi::Abi, inst::class, inst::simd};
use super::{
    compile::{CompiledFunction, compile},
    frame::{CHILD_ARGS, Frame},
    inst::{
        Address, Alu, Cond, Expansion, Inst, MaskSelect, NativeInst, Quotient, Rhs, SelectCond,
    },
};
use crate::lir::regalloc::Allocations;
use crate::{
    CompileError, Limit, Trap,
    hir::{Binary, Type, ssa::VerifiedBody},
    jit::{DebugFrame, DebugFunction},
    lir::layout,
    module::{FunctionId, Module},
    support::slotmap::DenseMap,
};
use iced_x86::{BlockEncoderOptions, IcedError, code_asm::*};
use regalloc2::RegClass;
use regalloc2::{Allocation, Edit, Function as _, InstOrEdit, VReg};
use simd::{K, XMM, YMM, ZMM, opmask};

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

    fn spill32(&self, a: Allocation) -> AsmMemoryOperand {
        dword_ptr(r11 + disp(self.spill_base + a.as_stack().expect("allocated spill").index()))
    }

    fn emit_move(
        &self,
        cpu: Cpu,
        a: &mut CodeAssembler,
        from: Allocation,
        to: Allocation,
    ) -> AsmResult {
        if from == to {
            return Ok(());
        }
        match (from.as_reg(), to.as_reg()) {
            (Some(s), Some(d)) if s.class() == RegClass::Vector => {
                simd::copy(a, d.hw_enc(), s.hw_enc(), cpu.bytes)
            }
            (Some(s), None) if s.class() == RegClass::Vector => {
                simd::store(a, self.spill(to), s.hw_enc(), cpu.bytes)
            }
            (None, Some(d)) if d.class() == RegClass::Vector => {
                simd::load(a, d.hw_enc(), self.spill(from), cpu.bytes)
            }
            (Some(s), Some(d)) if s.class() == RegClass::Float => {
                if cpu.isa.bw {
                    a.kmovq(K[d.hw_enc()], K[s.hw_enc()])
                } else {
                    a.kmovw(K[d.hw_enc()], K[s.hw_enc()])
                }
            }
            (Some(s), None) if s.class() == RegClass::Float => {
                if cpu.isa.bw {
                    a.kmovq(self.spill(to), K[s.hw_enc()])
                } else {
                    a.kmovw(word_ptr(self.spill(to)), K[s.hw_enc()])
                }
            }
            (None, Some(d)) if d.class() == RegClass::Float => {
                if cpu.isa.bw {
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
    pub sites: crate::lir::Sites,
}

pub(crate) fn emit(
    target: &splice_target::Target,
    source: Module<VerifiedBody<'_>>,
) -> Result<Code, CompileError> {
    emit_mode(&compile(Machine::new(target), source)?, false)
}

pub(crate) fn emit_debug(
    target: &splice_target::Target,
    source: Module<VerifiedBody<'_>>,
) -> Result<Code, CompileError> {
    emit_mode(&compile(Machine::new(target), source)?, true)
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

/// An assembly failure: either a limit of the target or an encoder error,
/// which can only be a Lancy defect.
enum Failure {
    Assembly(IcedError),
    Limit(Limit),
}

impl From<IcedError> for Failure {
    fn from(error: IcedError) -> Self {
        Self::Assembly(error)
    }
}

impl From<Failure> for CompileError {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Assembly(error) => Self::internal(error),
            Failure::Limit(limit) => limit.into(),
        }
    }
}

fn emit_mode(module: &Module<CompiledFunction>, debug: bool) -> Result<Code, CompileError> {
    Ok(assemble(module, debug)?)
}

fn assemble(module: &Module<CompiledFunction>, debug: bool) -> Result<Code, Failure> {
    let _span = tracing::debug_span!(target: "lancy::compile", "emit_machine_code").entered();
    let entry = module.call_graph.root();
    let mut a = CodeAssembler::new(64)?;
    let entries = DenseMap::from_fn(module.functions.len(), |_| a.create_label());
    let epilogue = a.create_label();
    a.endbr64()?;
    // The entry receives (argv, workspace) in the first two argument
    // registers. The allocator models argv's incoming register. R11 is reserved.
    let root = &module.functions[entry];
    a.mov(r11, R64[root.body.machine().abi.args[1].hw_enc()])?;
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
    let mut sites = crate::lir::Sites::default();
    let mut debug_functions = Vec::new();
    // The root (last in postorder) follows the native entry prologue.
    for (id, machine) in module.functions.iter().rev() {
        bind(&mut a, entries[id])?;
        let is_root = id == entry;
        let start = if debug { position(&mut a)? } else { 0 };
        let mut points = Vec::new();
        Emitter {
            cpu: machine.body.machine().cpu,
            abi: machine.body.machine().abi,
            a: &mut a,
            machine,
            constants: &mut constants,
            root: is_root,
            entries: &entries,
            epilogue,
            exits: Exits::default(),
            debug,
            points: &mut points,
            sites: &mut sites,
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
        return Err(Failure::Limit(Limit::CodeSize { bytes: bytes.len() }));
    }
    Ok(Code {
        #[cfg(test)]
        text_len: bytes.len() - pool_bytes,
        bytes,
        words: root.workspace,
        debug: debug_functions,
        sites,
    })
}

/// Side exits of one function, created on first use and emitted at its end.
#[derive(Default)]
struct Exits {
    /// Indexed by `Trap::code() - 1`.
    traps: [Option<CodeLabel>; Trap::ALL.len()],
    /// The shared tail: report the trap code held in R10D.
    report: Option<CodeLabel>,
}

impl Exits {
    fn trap(&mut self, a: &mut CodeAssembler, trap: Trap) -> CodeLabel {
        *self.traps[trap.code() as usize - 1].get_or_insert_with(|| a.create_label())
    }

    fn report(&mut self, a: &mut CodeAssembler) -> CodeLabel {
        *self.report.get_or_insert_with(|| a.create_label())
    }
}

struct Emitter<'a> {
    cpu: Cpu,
    abi: &'static Abi,
    constants: &'a mut Constants,
    a: &'a mut CodeAssembler,
    machine: &'a CompiledFunction,
    root: bool,
    entries: &'a DenseMap<FunctionId, CodeLabel>,
    epilogue: CodeLabel,
    exits: Exits,
    debug: bool,
    points: &'a mut Vec<(u32, u64)>,
    sites: &'a mut crate::lir::Sites,
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
                        self.machine.frame.emit_move(self.cpu, self.a, from, to)?
                    }
                    InstOrEdit::Inst(j) => {
                        let allocs = Allocations::of(body, &self.machine.allocation, j);
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
                                let condition = match inst {
                                    Inst::BranchAny(mask) => {
                                        let mask = mask.map(|v| allocs.reg(v));
                                        simd::test_vector_mask(self.a, self.cpu, mask)?;
                                        Cond::Ne
                                    }
                                    Inst::BranchAll(lane, mask) => {
                                        let (cpu, mask_reg) = (self.cpu, allocs.reg(mask));
                                        if mask.class() == RegClass::Float {
                                            opmask::test_all(self.a, cpu, lane, mask_reg)?;
                                        } else {
                                            simd::test_all(self.a, cpu, lane, mask_reg)?;
                                        }
                                        Cond::Eq
                                    }
                                    Inst::BrCmp(cc, lhs, rhs) => {
                                        cmp(self.a, &self.machine.frame, allocs, lhs, rhs)?;
                                        cc
                                    }
                                    Inst::Branch(v) => {
                                        let r = R32[allocs.reg(v)];
                                        self.a.test(r, r)?;
                                        Cond::Ne
                                    }
                                    _ => unreachable!("matched as a branch"),
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
        let Exits { traps, report } = self.exits;
        let mut report = report;
        for (trap, exit) in Trap::ALL.into_iter().zip(traps) {
            if let Some(exit) = exit {
                bind(self.a, exit)?;
                self.a.mov(r10d, trap.code())?;
                let tail = match report {
                    Some(tail) => tail,
                    None => *report.insert(self.a.create_label()),
                };
                self.a.jmp(tail)?;
            }
        }
        if let Some(report) = report {
            bind(self.a, report)?;
            // A trap can bypass every internal continuation. Recover the
            // root workspace, report the code in R10D through the live status
            // pointer supplied by ScalarKernel::invoke, and restore the host ABI.
            if !self.root {
                if self.debug {
                    self.a.mov(rax, slot(1))?;
                    self.a.mov(rax, qword_ptr(rax))?;
                    self.a.mov(dword_ptr(rax), r10d)?;
                    self.a.xor(eax, eax)?;
                    self.a.ret()?;
                    return Ok(());
                }
                self.a.mov(r11, slot(1))?;
            }
            self.a.mov(rax, slot(0))?;
            self.a.mov(dword_ptr(rax), r10d)?;
            self.a.xor(eax, eax)?;
            self.a.jmp(self.epilogue)?;
        }
        Ok(())
    }

    fn lanes(&mut self, expansion: Expansion, allocs: Allocations) -> AsmResult {
        let Expansion {
            action,
            input,
            output,
            dst,
            args,
        } = expansion;
        use super::inst::lanes::{self, FRAME, INPUT, OUTPUT, WORKSPACE};
        let width = self.cpu.bytes;
        let a = &mut self.a;
        a.sub(rsp, FRAME)?;
        a.mov(qword_ptr(rsp + WORKSPACE), r11)?;
        for (arg, offset) in args.into_iter().zip(INPUT) {
            if let Some((ty, v)) = arg {
                store(a, width, ty, ptr(rsp + offset), allocs.reg(v))?;
            }
        }
        let mut faults = [None; 3];
        lanes::emit(a, width, action, input, output, &mut faults)?;
        load(a, width, output, allocs.reg(dst), ptr(rsp + OUTPUT))?;
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
                    let fail = self.exits.trap(
                        a,
                        [Trap::Bounds, Trap::DivisionByZero, Trap::DivisionOverflow][i],
                    );
                    a.jmp(fail)?;
                }
            }
            bind(a, done)?;
        }
        Ok(())
    }

    /// Call `lir::fallback` with the area below the Win64 shadow space and
    /// report its trap code. Every generated frame enters with RSP 8 mod 16.
    fn fallback(&mut self, expansion: Expansion, allocs: Allocations) -> AsmResult {
        let Expansion {
            action,
            input,
            output,
            dst,
            args,
        } = expansion;
        use crate::lir::fallback::{AREA_BYTES, RESULT, SLOT_BYTES, Site, entry};
        let (width, abi) = (self.cpu.bytes, self.abi);
        let area = abi.shadow;
        let saved = area + AREA_BYTES as i32;
        let frame = saved + 8;
        let types = args.map(|arg| arg.map(|(ty, _)| ty));
        let site = self
            .sites
            .add(Site::new(action, input, output, types, width));
        let a = &mut *self.a;
        a.sub(rsp, frame)?;
        a.mov(qword_ptr(rsp + saved), r11)?;
        for (i, arg) in args.into_iter().enumerate() {
            if let Some((ty, v)) = arg {
                let at = ptr(rsp + area + (i * SLOT_BYTES) as i32);
                store(a, width, ty, at, allocs.reg(v))?;
            }
        }
        let [site_arg, area_arg] = [0, 1].map(|i| R64[abi.args[i].hw_enc()]);
        a.mov(site_arg, site)?;
        a.lea(area_arg, ptr(rsp + area))?;
        a.mov(r10, entry())?;
        a.call(r10)?;
        // The result register may be RAX: keep the code in R10D.
        a.mov(r10d, eax)?;
        let result = ptr(rsp + area + RESULT as i32);
        load(a, width, output, allocs.reg(dst), result)?;
        a.mov(r11, qword_ptr(rsp + saved))?;
        a.add(rsp, frame)?;
        let report = self.exits.report(a);
        a.test(r10d, r10d)?;
        a.jne(report)
    }

    fn instruction(&mut self, inst: Inst, allocs: Allocations) -> AsmResult {
        let (cpu, abi) = (self.cpu, self.abi);
        let width = cpu.bytes;
        let a = &mut *self.a;
        let frame = &self.machine.frame;
        let details = self.machine.body.details();
        let reg = |v: VReg| allocs.reg(v);
        let r = |v| R32[reg(v)];
        let q = |v| R64[reg(v)];
        // x86's r/m operand can directly consume a spilled RHS.
        macro_rules! rm32 {
            ($op:ident, $lhs:expr, $rhs:expr) => {{
                let rhs = allocs.get($rhs);
                if let Some(p) = rhs.as_reg() {
                    a.$op($lhs, R32[p.hw_enc()])
                } else {
                    a.$op($lhs, frame.spill32(rhs))
                }
            }};
        }
        match inst {
            Inst::Argv(_) | Inst::Copy(..) => Ok(()), // incoming fixed def / tied operands
            Inst::LoadArg {
                argv: Some(argv),
                dst,
                ty,
                index,
            } => load(a, width, ty, reg(dst), ptr(q(argv) + disp(index))),
            Inst::LoadArg {
                argv: None,
                dst,
                ty,
                index,
            } => load(a, width, ty, reg(dst), slot(CHILD_ARGS + index)),
            Inst::Native(id) => {
                let NativeInst {
                    native,
                    dst,
                    args,
                    temps,
                } = details.natives[id];
                let operands: Vec<usize> =
                    args.into_iter().flatten().chain([dst]).map(reg).collect();
                let temps: Vec<usize> = temps.into_iter().flatten().map(reg).collect();
                native.emit_allocated(a, &operands, &temps, self.constants)
            }
            Inst::ExtractLane(lane, dst, src, index) => {
                simd::packed::rearrange::extract_lane(a, cpu, lane, reg(dst), reg(src), index)
            }
            Inst::ShiftDwords(dst, src, count) => {
                simd::packed::rearrange::shift_dwords(a, width, reg(dst), reg(src), count)
            }
            Inst::PairedShuffle(form, dst, x, y) => {
                form.emit(a, self.constants, width, [reg(dst), reg(x), reg(y)])
            }
            Inst::DwordLane {
                lane,
                checked,
                dst,
                value,
                index,
                replacement,
                temp,
            } => {
                if checked {
                    let fail = self.exits.trap(a, Trap::Bounds);
                    a.cmp(r(index), (width / 4) as i32)?;
                    a.jae(fail)?;
                }
                let registers = [reg(dst), reg(value), reg(index)];
                if let Some(replacement) = replacement {
                    simd::packed::rearrange::insert_dword(
                        a,
                        self.constants,
                        width,
                        lane,
                        registers,
                        reg(replacement),
                        temp.map(reg),
                    )
                } else {
                    simd::packed::rearrange::extract_dword(a, width, lane, registers)
                }
            }
            Inst::VectorShiftImm(kind, lane, dst, src, count) => {
                let (dst, src) = (reg(dst), reg(src));
                simd::packed::integer::shift_dword_bits(a, cpu, kind, lane, dst, src, count)
            }
            Inst::Fallback(id) => self.fallback(details.expansions[id], allocs),
            Inst::Lanes(id) => self.lanes(details.expansions[id], allocs),
            Inst::Literal(l, bits, dst) => match l.bytes() {
                16 => {
                    let label = a.create_label();
                    let mut data = [0; 64];
                    data[..16].copy_from_slice(&bits.to_le_bytes());
                    self.constants.push((label, data, 16));
                    simd::load(a, reg(dst), ptr(label), 16)
                }
                8 if l.float() => {
                    a.mov(r10, bits as u64)?;
                    a.movq(simd::XMM[reg(dst)], r10)
                }
                4 if l.float() => {
                    a.mov(r10d, bits as u32)?;
                    a.movd(simd::XMM[reg(dst)], r10d)
                }
                8 => a.mov(q(dst), bits as u64),
                size => a.mov(r(dst), (bits & (u128::MAX >> (128 - size * 8))) as u32),
            },
            Inst::Iota(l, dst) => {
                let label = a.create_label();
                let bits = super::super::constants::iota_bytes(l, width);
                self.constants.push((label, bits, width));
                simd::load(a, reg(dst), ptr(label), width)
            }
            Inst::BitMove(from, to, dst, src) => match (class(to), from.bytes(width)) {
                (RegClass::Vector, 4) => a.movd(simd::XMM[reg(dst)], r(src)),
                (RegClass::Vector, 8) => a.movq(simd::XMM[reg(dst)], q(src)),
                (RegClass::Int, 4) => a.movd(r(dst), simd::XMM[reg(src)]),
                (RegClass::Int, 8) => a.movq(q(dst), simd::XMM[reg(src)]),
                _ => unreachable!(),
            },
            // The destination reuses the left operand's register.
            Inst::IntBinary(op, ty, dst, _, rhs) => {
                use crate::hir::Binary;
                let size = ty.lane().unwrap().bytes();
                macro_rules! integer {
                    ($mn:ident) => {
                        if size == 8 {
                            a.$mn(q(dst), q(rhs))
                        } else {
                            a.$mn(r(dst), r(rhs))
                        }
                    };
                }
                // Narrow results keep the carry-out above their size (see `Inst`).
                match op {
                    Binary::Add => integer!(add),
                    Binary::Sub => integer!(sub),
                    Binary::Mul => integer!(imul_2),
                    Binary::And => integer!(and),
                    Binary::Or => integer!(or),
                    Binary::Xor => integer!(xor),
                    _ => unreachable!(),
                }
            }
            Inst::Imm(dst, n) => a.mov(r(dst), n),
            // The destination reuses the left operand's register.
            Inst::Alu(op, dst, _, Rhs::Imm(n)) => match op {
                Alu::Add => a.add(r(dst), n),
                Alu::Sub => a.sub(r(dst), n),
                Alu::And => a.and(r(dst), n),
                Alu::Or => a.or(r(dst), n),
                Alu::Xor => a.xor(r(dst), n),
                Alu::Mul => a.imul_3(r(dst), r(dst), n),
                Alu::Min | Alu::Max => unreachable!("cmov requires a register or memory"),
            },
            Inst::Alu(op, dst, _, Rhs::Reg(rhs)) => match op {
                Alu::Add => rm32!(add, r(dst), rhs),
                Alu::Sub => rm32!(sub, r(dst), rhs),
                Alu::And => rm32!(and, r(dst), rhs),
                Alu::Or => rm32!(or, r(dst), rhs),
                Alu::Xor => rm32!(xor, r(dst), rhs),
                Alu::Mul => rm32!(imul_2, r(dst), rhs),
                Alu::Min | Alu::Max => {
                    rm32!(cmp, r(dst), rhs)?;
                    if matches!(op, Alu::Min) {
                        rm32!(cmovg, r(dst), rhs)
                    } else {
                        rm32!(cmovl, r(dst), rhs)
                    }
                }
            },
            Inst::Shift {
                left, dst, count, ..
            } => match (left, count) {
                (true, Rhs::Reg(_)) => a.shl(r(dst), cl),
                (false, Rhs::Reg(_)) => a.sar(r(dst), cl),
                (true, Rhs::Imm(n)) => a.shl(r(dst), n & 31),
                (false, Rhs::Imm(n)) => a.sar(r(dst), n & 31),
            },
            Inst::Compare(cc, dst, lhs, rhs) => {
                cmp(a, frame, allocs, lhs, rhs)?;
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
            // The destination reuses the register of `no`.
            Inst::SelectI32 { dst, cond, yes, .. } => {
                let cc = match cond {
                    SelectCond::Nonzero(v) => {
                        a.test(r(v), r(v))?;
                        Cond::Ne
                    }
                    SelectCond::Compare(cc, lhs, rhs) => {
                        cmp(a, frame, allocs, lhs, rhs)?;
                        cc
                    }
                };
                let dst = r(dst);
                match cc {
                    Cond::Eq => rm32!(cmove, dst, yes),
                    Cond::Ne => rm32!(cmovne, dst, yes),
                    Cond::Lt => rm32!(cmovl, dst, yes),
                    Cond::Le => rm32!(cmovle, dst, yes),
                    Cond::Gt => rm32!(cmovg, dst, yes),
                    Cond::Ge => rm32!(cmovge, dst, yes),
                }
            }
            Inst::Cdq { .. } => a.cdq(),
            Inst::IDiv { divisor, .. } => {
                let zero = self.exits.trap(a, Trap::DivisionByZero);
                let overflow = self.exits.trap(a, Trap::DivisionOverflow);
                let divide = a.create_label();
                let divisor = r(divisor);
                a.test(divisor, divisor)?;
                a.jz(zero)?;
                a.cmp(divisor, -1)?;
                a.jne(divide)?;
                a.cmp(eax, i32::MIN)?;
                a.je(overflow)?;
                bind(a, divide)?;
                a.idiv(divisor)
            }
            Inst::DivImm {
                remainder,
                dst: result,
                value,
                divisor,
                quotient,
            } => {
                let input = r(value);
                let dst = r(result);
                match quotient {
                    Quotient::PowerOfTwo(shift) => {
                        let bias = ((1u32 << shift) - 1) as i32;
                        a.lea(dst, ptr(q(value) + bias))?;
                        a.test(input, input)?;
                        a.cmovns(dst, input)?;
                        a.sar(dst, shift)?;
                    }
                    Quotient::Reciprocal { multiplier, shift } => {
                        let wide = q(result);
                        a.movsxd(wide, input)?;
                        if let Ok(multiplier) = i32::try_from(multiplier) {
                            a.imul_3(wide, wide, multiplier)?;
                        } else {
                            // R10 is reserved for scratch and cannot hold an input.
                            a.mov(r10d, multiplier)?;
                            a.imul_2(wide, r10)?;
                        }
                        a.sar(wide, shift)?;
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
                let fail = self.exits.trap(a, Trap::Bounds);
                simd::test_vector_mask(a, cpu, mask.map(reg))?;
                a.jne(fail)
            }
            Inst::AnyMask(dst, mask) => {
                simd::test_vector_mask(a, cpu, mask.map(reg))?;
                a.mov(r(dst), 0)?;
                a.setne(R8[reg(dst)])?;
                a.neg(r(dst))
            }
            Inst::AllMask(lane, dst, mask) => {
                opmask::test_all(a, cpu, lane, reg(mask))?;
                a.mov(r(dst), 0)?;
                a.sete(R8[reg(dst)])?;
                a.neg(r(dst))
            }
            Inst::SelectMask(id) => {
                let MaskSelect {
                    lane,
                    mask,
                    yes,
                    no,
                    dst,
                } = details.selects[id];
                simd::blend_comparison(a, cpu, lane, mask.map(reg), reg(dst), reg(yes), reg(no))
            }
            Inst::MaskCompare(lane, cc, dst, x, y) => {
                simd::compare_mask(a, width, K[reg(dst)], lane, reg(x), reg(y), cc)
            }
            Inst::MaskLogic(lane, kind, dst, x, y) => {
                opmask::logic(a, width / lane.bytes(), kind, reg(dst), reg(x), reg(y))
            }
            Inst::MaskNot(lane, dst, x) => opmask::not(a, cpu, lane, reg(dst), reg(x)),
            Inst::MaskImmediate(dst, bits) => opmask::immediate(a, reg(dst), bits),
            Inst::MaskPrefix(lanes, dst, count, temp) => {
                prefix_bits(a, r(temp), r(count), lanes)?;
                a.kmovw(K[reg(dst)], r10d)
            }
            Inst::MaskCount(dst, mask) => {
                a.kmovw(r(dst), K[reg(mask)])?;
                a.popcnt(r(dst), r(dst))
            }
            Inst::MaskCompact(lane, expand, dst, x, y) => {
                simd::compact(a, width, lane, expand, reg(dst), reg(x), reg(y))
            }
            Inst::CompactLookup(expand, dst, x, mask, index) => {
                let registers = [reg(dst), reg(x), reg(mask), reg(index)];
                simd::packed::compact::emit(a, cpu, expand, registers)
            }
            Inst::CompactStore {
                base,
                data,
                mask,
                limit,
                temps: [bits, prefix],
            } => {
                // Deposit the written prefix on the mask lanes, so the
                // compressing store writes the first `limit` active lanes.
                prefix_bits(a, r(prefix), r(limit), (width / 4) as u8)?;
                a.kmovw(r(bits), K[reg(mask)])?;
                a.pdep(r10d, r10d, r(bits))?;
                a.kmovw(k1, r10d)?;
                let address = ptr(q(base)).k1();
                match width {
                    16 => a.vpcompressd(address, XMM[reg(data)]),
                    32 => a.vpcompressd(address, YMM[reg(data)]),
                    _ => a.vpcompressd(address, ZMM[reg(data)]),
                }
            }
            Inst::CheckBounds(index, length) => {
                let fail = self.exits.trap(a, Trap::Bounds);
                rm32!(cmp, r(index), length)?;
                a.jae(fail)
            }
            Inst::TrapIf(v, trap) => {
                let fail = self.exits.trap(a, trap);
                a.test(r(v), r(v))?;
                a.jne(fail)
            }
            Inst::Trap(trap) => {
                let fail = self.exits.trap(a, trap);
                a.jmp(fail)
            }
            Inst::Movsxd(dst, src) => a.movsxd(q(dst), r(src)),
            Inst::Lea(dst, addr) => a.lea(q(dst), memory(addr, allocs)),
            Inst::Masked(id) => details.masked[id].map(reg).emit(a, cpu),
            Inst::Load(ty, dst, addr) => load(a, width, ty, reg(dst), memory(addr, allocs)),
            Inst::PackedMemory(kind, lane, dst, src, addr) => {
                let src = reg(src);
                let dst = reg(dst);
                let addr = memory(addr, allocs);
                if kind == Binary::Add {
                    return match (width, lane.bytes()) {
                        (32, 4) => a.vpaddd(YMM[dst], YMM[src], addr),
                        (32, 8) => a.vpaddq(YMM[dst], YMM[src], addr),
                        (64, 4) => a.vpaddd(ZMM[dst], ZMM[src], addr),
                        (64, 8) => a.vpaddq(ZMM[dst], ZMM[src], addr),
                        _ => unreachable!(),
                    };
                }
                match (width, kind) {
                    (32, Binary::And) => a.vpand(YMM[dst], YMM[src], addr),
                    (32, Binary::Or) => a.vpor(YMM[dst], YMM[src], addr),
                    (32, Binary::Xor) => a.vpxor(YMM[dst], YMM[src], addr),
                    (64, Binary::And) => a.vpandd(ZMM[dst], ZMM[src], addr),
                    (64, Binary::Or) => a.vpord(ZMM[dst], ZMM[src], addr),
                    (64, Binary::Xor) => a.vpxord(ZMM[dst], ZMM[src], addr),
                    _ => unreachable!(),
                }
            }
            Inst::BroadcastMemory(lane, dst, addr) => {
                let dst = reg(dst);
                let addr = memory(addr, allocs);
                match (width, lane.bytes()) {
                    (32, 4) => a.vpbroadcastd(YMM[dst], addr),
                    (32, 8) => a.vpbroadcastq(YMM[dst], addr),
                    (64, 4) => a.vpbroadcastd(ZMM[dst], addr),
                    (64, 8) => a.vpbroadcastq(ZMM[dst], addr),
                    _ => unreachable!(),
                }
            }
            Inst::BroadcastLane(dst, src, lane) => {
                let src = reg(src);
                let dst = reg(dst);
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
            Inst::Store(ty, addr, value) => store(a, width, ty, memory(addr, allocs), reg(value)),
            Inst::DebugStore(ty, offset, value) => store(
                a,
                width,
                ty,
                ptr(r11 + (disp(frame.debug_base) + offset as i32)),
                reg(value),
            ),
            // A debug point gets its own address for the line table; release
            // code emits nothing.
            Inst::DebugPoint(id) if self.debug => {
                self.points.push((id, position(a)?));
                a.nop()
            }
            Inst::DebugPoint(_) => Ok(()),
            Inst::StoreArg(i, ty, value) => {
                store(a, width, ty, slot(frame.slots + CHILD_ARGS + i), reg(value))
            }
            Inst::CallScratch(offset, _, dst) => a.lea(
                q(dst),
                ptr(r11 + (disp(frame.scratch_base) + offset as i32)),
            ),
            Inst::RustCall(function, ..) => {
                // Every generated frame enters with RSP 8 mod 16. Leave the
                // Win64 shadow space intact and save the volatile workspace
                // outside it. SysV requires no shadow space.
                let shadow = abi.shadow;
                a.sub(rsp, shadow + 8)?;
                a.mov(qword_ptr(rsp + shadow), r11)?;
                a.mov(r10, function.entry as usize as u64)?;
                a.call(r10)?;
                a.mov(r11, qword_ptr(rsp + shadow))?;
                a.add(rsp, shadow + 8)?;
                // The status's upper word is the entry's trap code.
                let report = self.exits.report(a);
                a.mov(r10, rax)?;
                a.shr(r10, 32)?;
                a.jne(report)
            }
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
                match ty.bytes(width) {
                    16.. => store(a, width, ty, slot(1), 0)?,
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

fn load(
    a: &mut CodeAssembler,
    width: usize,
    ty: Type,
    d: usize,
    mem: AsmMemoryOperand,
) -> AsmResult {
    let size = ty.bytes(width);
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
fn store(
    a: &mut CodeAssembler,
    width: usize,
    ty: Type,
    mem: AsmMemoryOperand,
    s: usize,
) -> AsmResult {
    let size = ty.bytes(width);
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

fn memory(addr: Address, allocs: Allocations) -> AsmMemoryOperand {
    let base = R64[allocs.reg(addr.base)];
    match addr.index {
        Some((index, scale)) => {
            ptr(base + R64[allocs.reg(index)] * (1i32 << scale as u8) + addr.disp)
        }
        None => ptr(base + addr.disp),
    }
}

/// Compare `lhs` with `rhs`, which may stay in its spill slot.
fn cmp(
    a: &mut CodeAssembler,
    frame: &Frame,
    allocs: Allocations,
    lhs: VReg,
    rhs: Rhs,
) -> AsmResult {
    let lhs = R32[allocs.reg(lhs)];
    match rhs {
        Rhs::Imm(n) => a.cmp(lhs, n),
        Rhs::Reg(v) => match allocs.get(v).as_reg() {
            Some(p) => a.cmp(lhs, R32[p.hw_enc()]),
            None => a.cmp(lhs, frame.spill32(allocs.get(v))),
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

/// `r10d` = the low `count` of `lanes` bits, with `count` clamped to
/// `0..=lanes` in `temp`: `bzhi` reads only the low byte of its index.
fn prefix_bits(
    a: &mut CodeAssembler,
    temp: AsmRegister32,
    count: AsmRegister32,
    lanes: u8,
) -> AsmResult {
    a.mov(temp, count)?;
    a.xor(r10d, r10d)?;
    a.test(temp, temp)?;
    a.cmovs(temp, r10d)?;
    a.mov(r10d, i32::from(lanes))?;
    a.cmp(temp, r10d)?;
    a.cmova(temp, r10d)?;
    a.mov(r10d, -1)?;
    a.bzhi(r10d, r10d, temp)
}
