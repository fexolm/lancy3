//! Lower verified HIR, allocate target registers and emit AAPCS64 code.
mod compact;
mod compile;
mod divide;
mod groups;
mod half;
mod inst;
mod isel;
mod memory;
mod operations;
mod reduce;
mod rotate;
mod shuffle;
mod special;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::check_machine;
mod unary;
mod wide;
mod width;

use super::{
    Arrangement as A, Code as C, ConditionCode as Cond, Operand as O, PredQual, Register as R,
    ShiftKind,
};
use crate::lir::regalloc::Allocations;
use crate::{
    CompileError, Trap,
    hir::{Type, ssa::VerifiedBody},
    jit::{DebugFrame, DebugFunction},
    lir::layout,
    module::{FunctionId, Module},
    support::slotmap::DenseMap,
};
use compile::{Compiled, Frame, compile};
use inst::Inst;
use regalloc2::{Allocation, Block, Edit, Function as _, InstOrEdit, PReg, RegClass, VReg};
use splice_target::Profile;

/// The vector extension and width one compilation selects for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    pub profile: Profile,
    pub bytes: usize,
}

impl Config {
    pub fn new(target: &splice_target::Target) -> Self {
        let features = target
            .arm64()
            .expect("AArch64 code generation for an AArch64 target");
        let profile = if features.sve2 {
            Profile::Sve2
        } else if features.sve {
            Profile::Sve
        } else {
            Profile::Neon
        };
        Self {
            profile,
            bytes: target.vector_bytes(),
        }
    }
}

// Status pointer and eight result words.
const ROOT_BYTES: usize = 9 * 8;
const ROOT_SAVES: [u8; 4] = [19, 20, 22, 30];
const CHILD_SAVES: [u8; 3] = [19, 22, 30];
const HOST_STACK_BYTES: u64 = 32;
const RESULT: (R, usize) = (R::X20, 8);

pub(crate) struct Code {
    pub bytes: Vec<u8>,
    pub words: usize,
    pub sites: crate::lir::fallback::Sites,
    pub debug: Vec<DebugFunction>,
}

type Result<T = ()> = std::result::Result<T, CompileError>;
type Label = usize;

fn sve(profile: Profile) -> bool {
    matches!(profile, Profile::Sve | Profile::Sve2)
}

fn arrangements(bytes: usize) -> (A, A) {
    let index = bytes.trailing_zeros() as usize;
    (
        [A::B16, A::H8, A::S4, A::D2][index],
        [A::B, A::H, A::S, A::D][index],
    )
}

#[derive(Default)]
struct Assembler {
    words: Vec<u32>,
    labels: Vec<Option<u64>>,
    fixups: Vec<(usize, C, Label)>,
}

impl Assembler {
    fn put(&mut self, code: C, operands: &[O]) -> Result {
        self.words.push(
            code.encode(0, operands, self.words.len() as u64 * 4)
                .map_err(super::Error::from)?,
        );
        Ok(())
    }

    fn label(&mut self) -> Label {
        self.labels.push(None);
        self.labels.len() - 1
    }

    fn bind(&mut self, label: Label) {
        debug_assert!(self.labels[label].is_none());
        self.labels[label] = Some(self.words.len() as u64 * 4);
    }

    fn branch(&mut self, code: C, label: Label) {
        self.fixups.push((self.words.len(), code, label));
        self.words.push(0);
    }

    fn jump(&mut self, label: Label) {
        self.branch(C::B_only_branch_imm, label);
    }

    fn skip_if(&mut self, cond: Cond) -> Result {
        self.put(
            C::B_only_condbranch,
            &[O::cond(cond), O::label(self.words.len() as u64 * 4 + 8)],
        )
    }

    fn jump_if(&mut self, cond: Cond, label: Label) -> Result {
        self.skip_if(invert(cond))?;
        self.jump(label);
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<u8>> {
        for (at, code, label) in self.fixups {
            let target = self.labels[label].expect("branch target emitted");
            let pc = at as u64 * 4;
            self.words[at] = code.encode(0, &[O::label(target)], pc).map_err(|error| {
                if matches!(error, super::EncodeError::Branch { .. }) {
                    CompileError::from(crate::Limit::BranchRange {
                        offset: pc as usize,
                        target: target as usize,
                    })
                } else {
                    super::Error::from(error).into()
                }
            })?;
        }
        Ok(self.words.into_iter().flat_map(u32::to_le_bytes).collect())
    }

    fn imm(&mut self, dst: R, value: u64) -> Result {
        let wide = dst.bank == super::operand::Bank::X;
        let (movz, movk, end) = if wide {
            (C::Movz_64_movewide, C::Movk_64_movewide, 64)
        } else {
            (C::Movz_32_movewide, C::Movk_32_movewide, 32)
        };
        let mut chunks = (0..end).step_by(16).filter_map(|shift| {
            let chunk = (value >> shift) & 65535;
            (chunk != 0).then_some((chunk, shift))
        });
        let (first, shift) = chunks.next().unwrap_or((0, 0));
        self.put(movz, &[O::reg(dst), O::imm_lsl(first, shift)])?;
        for (chunk, shift) in chunks {
            self.put(movk, &[O::reg(dst), O::imm_lsl(chunk, shift)])?;
        }
        Ok(())
    }

    fn addr(&mut self, dst: R, base: R, offset: usize) -> Result {
        if dst == base && offset == 0 {
            return Ok(());
        }
        let (imm, shift) = if offset < 4096 {
            (offset, 0)
        } else if offset < (1 << 24) && offset.is_multiple_of(4096) {
            (offset >> 12, 12)
        } else {
            debug_assert_ne!(base, R::X12);
            self.imm(R::X12, offset as u64)?;
            return self.put(
                C::Add_64_addsub_shift,
                &[O::reg(dst), O::reg(base), O::reg(R::X12)],
            );
        };
        self.put(
            C::Add_64_addsub_imm,
            &[O::reg(dst), O::reg(base), O::imm_lsl(imm as u64, shift)],
        )
    }

    fn access(&mut self, load: bool, size: usize, n: u8, (base, offset): (R, usize)) -> Result {
        use C::*;
        let (code, reg) = match (load, size) {
            (true, 1) => (Ldrb_32_ldst_pos, R::w(n)),
            (false, 1) => (Strb_32_ldst_pos, R::w(n)),
            (true, 2) => (Ldrh_32_ldst_pos, R::w(n)),
            (false, 2) => (Strh_32_ldst_pos, R::w(n)),
            (true, 4) => (Ldr_32_ldst_pos, R::w(n)),
            (false, 4) => (Str_32_ldst_pos, R::w(n)),
            (true, 8) => (Ldr_64_ldst_pos, R::x(n)),
            (false, 8) => (Str_64_ldst_pos, R::x(n)),
            (true, 16) => (Ldr_q_ldst_pos, R::v(n)),
            (false, 16) => (Str_q_ldst_pos, R::v(n)),
            _ => unreachable!("invalid memory access width"),
        };
        self.mem(code, reg.unwrap(), size, (base, offset))
    }

    fn mem(&mut self, code: C, reg: R, size: usize, (base, offset): (R, usize)) -> Result {
        let addr = if offset.is_multiple_of(size) && offset / size < 4096 {
            O::mem_off(base, offset as i64)
        } else {
            self.addr(R::X11, base, offset)?;
            O::mem(R::X11)
        };
        self.put(code, &[O::reg(reg), addr])
    }
}

struct Emitter<'a> {
    a: &'a mut Assembler,
    frame: &'a Frame,
    sites: &'a mut crate::lir::fallback::Sites,
    entries: &'a DenseMap<FunctionId, Label>,
    config: Config,
    epilogue: Label,
    byte_predicate: bool,
    /// Per-trap side exits, indexed by `Trap::code() - 1`.
    failures: [Option<Label>; Trap::ALL.len()],
    /// The shared tail of every side exit: report the trap code in W16.
    report: Option<Label>,
}

impl Emitter<'_> {
    fn value_access(&mut self, load: bool, ty: Type, n: u8, address: (R, usize)) -> Result {
        let size = ty.bytes(self.config.bytes);
        let code = match (load, ty) {
            (true, Type::F32) => C::Ldr_s_ldst_pos,
            (false, Type::F32) => C::Str_s_ldst_pos,
            (true, Type::F64) => C::Ldr_d_ldst_pos,
            (false, Type::F64) => C::Str_d_ldst_pos,
            _ => return self.access(load, size, n, address),
        };
        self.a.mem(code, R::v(n).unwrap(), size, address)
    }

    fn check_status(&mut self) -> Result {
        self.a.access(true, 8, 10, (R::X20, 0))?;
        self.a.access(true, 4, 9, (R::X10, 0))?;
        self.a.put(
            C::Subs_32_addsub_imm,
            &[O::reg(R::WZR), O::reg(R::W9), O::imm(0)],
        )?;
        self.a.skip_if(Cond::Eq)?;
        self.a.jump(self.epilogue);
        Ok(())
    }

    fn trap_if(&mut self, cond: Cond, trap: Trap) -> Result {
        self.a.skip_if(invert(cond))?;
        self.trap(trap);
        Ok(())
    }

    fn trap(&mut self, trap: Trap) {
        let exit = *self.failures[trap.code() as usize - 1].get_or_insert_with(|| self.a.label());
        self.a.jump(exit);
    }

    /// Trap with the code a Rust entry or runtime helper left in W16.
    fn report_if(&mut self, cond: Cond) -> Result {
        let report = *self.report.get_or_insert_with(|| self.a.label());
        self.a.skip_if(invert(cond))?;
        self.a.jump(report);
        Ok(())
    }

    fn traps(&mut self) -> Result {
        let mut report = self.report;
        for (trap, exit) in Trap::ALL.into_iter().zip(self.failures) {
            if let Some(exit) = exit {
                self.a.bind(exit);
                self.a.imm(R::W16, trap.code().into())?;
                let tail = match report {
                    Some(tail) => tail,
                    None => *report.insert(self.a.label()),
                };
                self.a.jump(tail);
            }
        }
        if let Some(report) = report {
            self.a.bind(report);
            self.a.access(true, 8, 10, (R::X20, 0))?;
            self.a.access(false, 4, 16, (R::X10, 0))?;
            self.a.jump(self.epilogue);
        }
        Ok(())
    }

    fn access(&mut self, load: bool, size: usize, n: u8, (base, offset): (R, usize)) -> Result {
        if size <= 16 {
            return self.a.access(load, size, n, (base, offset));
        }
        if !self.byte_predicate {
            self.predicate(R::P0, A::B, self.config.bytes)?;
            self.byte_predicate = true;
        }
        let base = if offset != 0 {
            self.a.addr(R::X11, base, offset)?;
            R::X11
        } else {
            base
        };
        let (code, pred) = if load {
            (C::Ld1b_z_p_bi_u8, PredQual::Zero)
        } else {
            (C::St1b_z_p_bi_, PredQual::None)
        };
        self.a.put(
            code,
            &[
                O::vec(R::z(n).unwrap(), A::B),
                O::pred(R::P0, pred),
                O::mem(base),
            ],
        )
    }

    fn predicate(&mut self, p: R, a: A, lanes: usize) -> Result {
        // PTRUE's VL1..VL8 are literal patterns, VL16..VL64 are 9..11.
        let pattern = if lanes <= 8 {
            lanes
        } else {
            lanes.trailing_zeros() as usize + 5
        };
        self.a
            .put(C::Ptrue_p_s_, &[O::vec(p, a), O::named(pattern as u32)])
    }

    fn move_regs(&mut self, from: PReg, to: PReg) -> Result {
        let (s, d) = (from.hw_enc() as u8, to.hw_enc() as u8);
        if s == d {
            return Ok(());
        }
        if to.class() == RegClass::Vector {
            self.copy_vec(d, s, self.config.bytes > 16)
        } else {
            self.a.put(
                C::Orr_64_log_shift,
                &[
                    O::reg(R::x(d).unwrap()),
                    O::reg(R::XZR),
                    O::reg(R::x(s).unwrap()),
                ],
            )
        }
    }

    fn copy_vec(&mut self, dst: u8, src: u8, sve: bool) -> Result {
        if src == dst {
            return Ok(());
        }
        let (code, bank, arr): (_, fn(u8) -> Option<R>, _) = if sve {
            (C::Orr_z_zz_, R::z, A::D)
        } else {
            (C::Orr_asimdsame_only, R::v, A::B16)
        };
        let s = O::vec(bank(src).unwrap(), arr);
        self.a.put(code, &[O::vec(bank(dst).unwrap(), arr), s, s])
    }

    fn edit(&mut self, from: Allocation, to: Allocation) -> Result {
        let spill = |a: Allocation| {
            (
                R::X19,
                self.frame.spills + 8 * a.as_stack().unwrap().index(),
            )
        };
        match (from.as_reg(), to.as_reg()) {
            (Some(s), Some(d)) => self.move_regs(s, d),
            (Some(s), None) => {
                let size = if s.class() == RegClass::Vector {
                    self.config.bytes
                } else {
                    8
                };
                self.access(false, size, s.hw_enc() as u8, spill(to))
            }
            (None, Some(d)) => {
                let size = if d.class() == RegClass::Vector {
                    self.config.bytes
                } else {
                    8
                };
                self.access(true, size, d.hw_enc() as u8, spill(from))
            }
            (None, None) => unreachable!("regalloc2 resolves memory-to-memory moves"),
        }
    }

    fn literal(&mut self, ty: Type, n: u8, bits: u128) -> Result {
        let size = ty.bytes(self.config.bytes);
        if matches!(ty, Type::F32 | Type::F64) {
            self.a
                .imm(if size == 4 { R::W9 } else { R::X9 }, bits as u64)?;
            return self.float_move(ty, true, n, 9);
        }
        match size {
            16 => {
                self.a.imm(R::X9, bits as u64)?;
                self.a.imm(R::X10, (bits >> 64) as u64)?;
                self.a
                    .put(C::Fmov_d_x, &[O::reg(R::v(n).unwrap()), O::reg(R::X9)])?;
                self.a
                    .put(C::Mov_v_d1_x, &[O::reg(R::v(n).unwrap()), O::reg(R::X10)])
            }
            8 => self.a.imm(R::x(n).unwrap(), bits as u64),
            _ => self.a.imm(
                R::w(n).unwrap(),
                bits as u64 & (u64::MAX >> (64 - size * 8)),
            ),
        }
    }

    fn inst(
        &mut self,
        op: Inst,
        allocs: Allocations,
        succs: &[Block],
        labels: &[Label],
        next: Option<Block>,
    ) -> Result {
        let n = |v: VReg| allocs.reg(v) as u8;
        let g = |v: VReg| R::x(n(v)).unwrap();
        let word = |v: VReg| R::w(n(v)).unwrap();
        let preg = |v: VReg| allocs.get(v).as_reg().expect("register operand");
        let r = O::reg;
        let bytes = self.config.bytes;
        let size = |ty: Type| ty.bytes(bytes);
        match op {
            Inst::Arg(ty, offset, d) => self.value_access(true, ty, n(d), (R::X22, offset)),
            Inst::Literal(ty, bits, d) => self.literal(ty, n(d), bits),
            Inst::Copy(d, v) => self.move_regs(preg(v), preg(d)),
            Inst::BitMove(from, to, d, v) => {
                if matches!(to, Type::F32 | Type::F64) {
                    self.float_move(to, true, n(d), n(v))
                } else {
                    self.float_move(from, false, n(v), n(d))
                }
            }
            Inst::Imm(code, bits, shift, d, v) => self.a.put(
                code,
                &[r(word(d)), r(word(v)), O::imm_lsl(bits.into(), shift)],
            ),
            Inst::Int(int, d, a, b) => self.int_binary(int, n(a), n(b), n(d)),
            Inst::WideAdd {
                subtract,
                scalable,
                dst,
                a,
                b,
            } => self.wide_add(subtract, scalable, n(a), n(b), n(dst)),
            Inst::WideMath(form, d, a, b) => self.wide_math(form, n(a), n(b), n(d)),
            Inst::WideUnary(op, scalable, d, v) => self.wide_unary(op, scalable, n(v), n(d)),
            Inst::Unary(form, d, v) => self.unary(form, n(v), n(d)),
            Inst::Half(form, d, a, extra) => self.half(form, n(a), extra.map(|e| e.map(n)), n(d)),
            Inst::HalfCast(from, to, d, v) => self.half_cast(from, to, n(v), n(d)),
            Inst::IntCast(from, to, d, v) => self.int_cast(from, to, n(v), n(d)),
            Inst::WideCast(from, to, d, v) => self.wide_cast(from, to, n(v), n(d)),
            Inst::FloatCast(code, from, to, d, v) => self.float_cast(code, from, to, n(v), n(d)),
            Inst::PackedCast(code, arr, d, v) => self.packed_cast(code, arr, n(v), n(d)),
            Inst::Select(form, d, m, a, b) => self.select(form, n(m), n(a), n(b), n(d)),
            Inst::Fma(code, arr, d, args) => self.fma(code, arr, args.map(n), n(d)),
            Inst::Sve2Ternary(code, d, [x0, x1, x2]) => {
                let z = |v| O::vec(R::z(n(v)).unwrap(), A::D);
                self.a.put(code, &[z(d), z(x0), z(x1), z(x2)])
            }
            Inst::Shuffle(form, d, args) => self.shuffle(form, args.map(n), n(d)),
            Inst::Compact {
                size,
                scalable,
                dst,
                value,
                mask,
            } => self.compact(size, scalable, n(value), n(mask), n(dst)),
            Inst::VectorDivide(form, d, a, b) => self.vector_divide(form, n(a), n(b), n(d)),
            Inst::Rotate(form, d, a, b) => self.rotate(form, n(a), n(b), n(d)),
            Inst::ScalarSaturate(code, size, d, a, b) => {
                self.scalar_saturate(code, size, n(a), n(b), n(d))
            }
            Inst::NeonMul64(d, a, b) => self.neon_mul64(n(a), n(b), n(d)),
            Inst::FloatSpecial(form, d, a, b) => self.float_special(form, n(a), b.map(n), n(d)),
            Inst::WideMaskTest(all, scalable, d, v) => {
                self.wide_mask_test(all, scalable, n(v), n(d))
            }
            Inst::Reduce(form, d, v) => self.reduce(form, n(v), n(d)),
            Inst::Expand(form, d, data, mask) => self.expand(form, n(data), n(mask), n(d)),
            Inst::Widen(form, d, v) => self.widen(form, n(v), n(d)),
            Inst::Narrow(form, d, a, b) => self.narrow(form, n(a), n(b), n(d)),
            Inst::Float(code, arr, d, a, b) => {
                let op = |v| O::vec(R::v(n(v)).unwrap(), arr);
                self.a.put(code, &[op(d), op(a), op(b)])
            }
            Inst::FloatCompare(cond, arr, d, a, b) => {
                let op = |v| O::vec(R::v(n(v)).unwrap(), arr);
                self.a.put(C::Fcmp_scalar, &[op(a), op(b)])?;
                self.bool_mask(cond, n(d))
            }
            Inst::SveIota(lane, d) => self.sve_iota(lane, n(d)),
            Inst::MaskTest(all, arr, d, v) => self.mask_test(all, arr, n(v), n(d)),
            Inst::LaneAccess {
                lane,
                src,
                index,
                insert,
                dst,
            } => self.lane_access(lane, n(src), word(index), insert.map(n), n(dst)),
            Inst::Splat(code, arr, ty, d, v) => {
                let input = if matches!(ty, Type::F32 | Type::F64) {
                    self.float_move(ty, false, n(v), 9)?;
                    9
                } else {
                    n(v)
                };
                let src = if size(ty) == 8 {
                    R::x(input)
                } else {
                    R::w(input)
                }
                .unwrap();
                let dst = if code == C::Dup_z_r {
                    R::z(n(d))
                } else {
                    R::v(n(d))
                }
                .unwrap();
                self.a.put(code, &[O::vec(dst, arr), r(src)])
            }
            Inst::SveSplat128(d, v) => self.a.put(
                C::Dup_q0_z,
                &[r(R::z(n(d)).unwrap()), r(R::z(n(v)).unwrap())],
            ),
            Inst::Packed(op, d, a, b) => self.packed(op, n(a), n(b), n(d)),
            Inst::PackedShiftImm(code, arr, count, d, v) => {
                let reg = if matches!(arr, A::B | A::H | A::S | A::D) {
                    R::z
                } else {
                    R::v
                };
                let operand = |v| O::vec(reg(n(v)).unwrap(), arr);
                self.a
                    .put(code, &[operand(d), operand(v), O::imm(count.into())])
            }
            Inst::Check(index, limit) => {
                self.a.put(
                    C::Subs_32_addsub_shift,
                    &[r(R::WZR), r(word(index)), r(word(limit))],
                )?;
                self.trap_if(Cond::Cs, Trap::Bounds)
            }
            Inst::TrapIf(cond, trap) => {
                self.a.put(
                    C::Subs_32_addsub_shift,
                    &[r(R::WZR), r(word(cond)), r(R::WZR)],
                )?;
                self.trap_if(Cond::Ne, trap)
            }
            Inst::Trap(trap) => {
                self.trap(trap);
                Ok(())
            }
            Inst::CallScratch(offset, _, d) => {
                self.a
                    .addr(g(d), R::X19, self.frame.scratch_base + offset as usize)
            }
            Inst::RustCall(function, ..) => {
                // The frame has already saved LR and keeps the workspace in
                // callee-saved registers across this AAPCS64 call.
                self.a.imm(R::X16, function.entry as usize as u64)?;
                self.a.put(C::Blr_64_branch_reg, &[r(R::X16)])?;
                self.byte_predicate = false;
                self.a.put(
                    C::Ubfm_64m_bitfield,
                    &[r(R::X16), r(R::X0), O::imm(32), O::imm(63)],
                )?;
                // The status's upper word is the entry's trap code.
                self.a
                    .put(C::Subs_32_addsub_imm, &[r(R::WZR), r(R::W16), O::imm(0)])?;
                self.report_if(Cond::Ne)
            }
            Inst::Offset(scale, d, base, index) => self.a.put(
                C::Add_64_addsub_sxtw,
                &[
                    r(g(d)),
                    r(g(base)),
                    O::shifted(word(index), ShiftKind::Lsl, scale as u8),
                ],
            ),
            Inst::Load(ty, d, address) => self.value_access(true, ty, n(d), (g(address), 0)),
            Inst::Store(ty, address, value) => {
                self.value_access(false, ty, n(value), (g(address), 0))
            }
            Inst::DebugStore(ty, offset, v) => self.value_access(
                false,
                ty,
                n(v),
                (R::X19, self.frame.debug_base + offset as usize),
            ),
            Inst::DebugPoint(_) => self.a.put(C::Nop_HI_hints, &[]),
            Inst::Masked {
                kind,
                ty,
                base,
                indices,
                mask,
                value,
                dst,
            } => {
                let (base, mask, value) = (g(base), n(mask), n(value));
                let (indices, dst) = (indices.map(n), dst.map(n));
                match kind {
                    inst::Masked::Neon => self.neon_masked(
                        ty.lane().unwrap().bytes(),
                        base,
                        mask,
                        value,
                        indices,
                        dst,
                    ),
                    inst::Masked::Sve { code, arrangement } => {
                        self.sve_masked((code, arrangement), base, mask, value, indices, dst)
                    }
                    inst::Masked::SveNarrow(code) => self.sve_narrow_masked(
                        (code, ty.lane().unwrap().bytes()),
                        base,
                        [mask, value, indices.unwrap()],
                        dst,
                    ),
                    inst::Masked::SveWide(code) => {
                        self.sve_wide_masked(code, base, mask, value, indices, dst)
                    }
                }
            }
            Inst::StoreArg(ty, offset, v) => {
                self.value_access(false, ty, n(v), (R::X19, self.frame.outgoing + offset))
            }
            Inst::Call(target, ..) => {
                self.a.addr(R::X0, R::X19, self.frame.outgoing)?;
                self.a.addr(R::X1, R::X19, self.frame.bytes)?;
                self.a.branch(C::Bl_only_branch_imm, self.entries[target]);
                self.byte_predicate = false;
                self.check_status()
            }
            Inst::LibArg(ty, index, v) => self.value_access(false, ty, n(v), self.temp(index)),
            Inst::Libcall(action, input, output, args) => {
                self.primitive(action, input, output, args)
            }
            Inst::LibResult(ty, d) => self.value_access(true, ty, n(d), self.temp(3)),
            Inst::Jump => {
                if Some(succs[0]) != next {
                    self.a.jump(labels[succs[0].index()]);
                }
                Ok(())
            }
            Inst::Branch(cond) => {
                self.a.put(
                    C::Subs_32_addsub_imm,
                    &[r(R::WZR), r(word(cond)), O::imm(0)],
                )?;
                let [yes, no] = [succs[0], succs[1]];
                if Some(yes) == next {
                    self.a.jump_if(Cond::Eq, labels[no.index()])
                } else {
                    self.a.jump_if(Cond::Ne, labels[yes.index()])?;
                    if Some(no) != next {
                        self.a.jump(labels[no.index()]);
                    }
                    Ok(())
                }
            }
            Inst::Return(ty, v, root) => {
                if root && size(ty) > 8 {
                    self.value_access(false, ty, n(v), RESULT)?;
                } else if root && matches!(ty, Type::F32 | Type::F64) {
                    self.float_move(ty, false, n(v), 0)?;
                }
                self.a.jump(self.epilogue);
                Ok(())
            }
        }
    }
    fn temp(&self, index: usize) -> (R, usize) {
        (
            R::X19,
            self.frame.temporary + index * crate::lir::fallback::SLOT_BYTES,
        )
    }
}

fn invert(c: Cond) -> Cond {
    use Cond::*;
    match c {
        Eq => Ne,
        Ne => Eq,
        Cs => Cc,
        Cc => Cs,
        Mi => Pl,
        Pl => Mi,
        Vs => Vc,
        Vc => Vs,
        Hi => Ls,
        Ls => Hi,
        Ge => Lt,
        Lt => Ge,
        Gt => Le,
        Le => Gt,
        Al => Nv,
        Nv => Al,
    }
}

pub(crate) fn emit(
    target: &splice_target::Target,
    source: Module<VerifiedBody<'_>>,
) -> Result<Code> {
    emit_mode(Config::new(target), source, false)
}

pub(crate) fn emit_debug(
    target: &splice_target::Target,
    source: Module<VerifiedBody<'_>>,
) -> Result<Code> {
    emit_mode(Config::new(target), source, true)
}

fn emit_mode(config: Config, source: Module<VerifiedBody<'_>>, debug: bool) -> Result<Code> {
    let module = compile(source, config)?;
    emit_compiled(&module, config, debug)
}

fn emit_compiled(module: &Module<Compiled>, config: Config, debug: bool) -> Result<Code> {
    let _span = tracing::debug_span!(target: "lancy::compile", "emit_machine_code").entered();
    let root = module.call_graph.root();
    let mut a = Assembler::default();
    let mut sites = crate::lir::fallback::Sites::default();
    let mut debug_functions = Vec::new();
    let entries = DenseMap::from_fn(module.functions.len(), |_| a.label());
    for (id, machine) in module.functions.iter().rev() {
        let epilogue = a.label();
        let labels: Vec<_> = (0..machine.body.num_blocks()).map(|_| a.label()).collect();
        a.bind(entries[id]);
        let start = a.words.len() as u64 * 4;

        let entry = id == root;
        let host_stack = entry || debug;
        let saved = if entry {
            &ROOT_SAVES[..]
        } else {
            &CHILD_SAVES[..]
        };
        let save_base = if host_stack {
            a.put(
                C::Sub_64_addsub_imm,
                &[O::reg(R::SP), O::reg(R::SP), O::imm(HOST_STACK_BYTES)],
            )?;
            R::SP
        } else {
            R::X1
        };
        for (i, &reg) in saved.iter().enumerate() {
            a.access(false, 8, reg, (save_base, i * 8))?;
        }
        if entry {
            a.addr(R::X20, R::X1, 0)?;
            a.addr(R::X19, R::X1, ROOT_BYTES)?;
            for (i, reg) in machine.written.into_iter().enumerate() {
                a.access(
                    false,
                    8,
                    reg.hw_enc() as u8,
                    (R::X19, machine.frame.saves + i * 8),
                )?;
            }
        } else {
            a.addr(R::X19, R::X1, 0)?;
        }
        a.addr(R::X22, R::X0, 0)?;
        let body_start = a.words.len() as u64 * 4;
        let mut points = Vec::new();

        let mut out = Emitter {
            a: &mut a,
            frame: &machine.frame,
            sites: &mut sites,
            entries: &entries,
            config,
            epilogue,
            byte_predicate: false,
            failures: [None; Trap::ALL.len()],
            report: None,
        };
        let order = layout::place(&machine.body);
        for (index, &block) in order.iter().enumerate() {
            out.a.bind(labels[block.index()]);
            out.byte_predicate = false;
            let succs = machine.body.block_succs(block);
            for item in machine
                .allocation
                .block_insts_and_edits(&machine.body, block)
            {
                match item {
                    InstOrEdit::Edit(&Edit::Move { from, to }) => out.edit(from, to)?,
                    InstOrEdit::Inst(i) => {
                        let inst = machine.body.insts()[i.index()];
                        // A debug point gets its own address for the line
                        // table; release code emits nothing.
                        if let Inst::DebugPoint(id) = inst {
                            if !debug {
                                continue;
                            }
                            points.push((id, out.a.words.len() as u64 * 4));
                        }
                        out.inst(
                            inst,
                            Allocations::of(&machine.body, &machine.allocation, i),
                            succs,
                            &labels,
                            order.get(index + 1).copied(),
                        )?;
                    }
                }
            }
        }
        out.traps()?;
        a.bind(epilogue);
        let frame_end = a.words.len() as u64 * 4;
        if entry {
            for (i, reg) in machine.written.into_iter().enumerate() {
                a.access(
                    true,
                    8,
                    reg.hw_enc() as u8,
                    (R::X19, machine.frame.saves + i * 8),
                )?;
            }
        }
        let restore_base = if host_stack { R::SP } else { R::X19 };
        // Restore X19 last because it addresses the private frame on outlined calls.
        for (i, &reg) in saved.iter().enumerate().rev() {
            a.access(true, 8, reg, (restore_base, i * 8))?;
        }
        if host_stack {
            a.addr(R::SP, R::SP, HOST_STACK_BYTES as usize)?;
        }
        a.put(C::Ret_64r_branch_reg, &[O::reg(R::X30)])?;
        if debug {
            let workspace_saved = if entry {
                machine
                    .written
                    .into_iter()
                    .enumerate()
                    .map(|(i, reg)| (reg.hw_enc() as u16, (machine.frame.saves + i * 8) as i64))
                    .collect()
            } else {
                Vec::new()
            };
            debug_functions.push(DebugFunction {
                start,
                end: a.words.len() as u64 * 4,
                workspace_offset: machine.frame.debug_base,
                points,
                frame: DebugFrame {
                    start: body_start,
                    end: frame_end,
                    cfa_from_sp: HOST_STACK_BYTES as i32,
                    saved: saved
                        .iter()
                        .enumerate()
                        .map(|(i, &reg)| (u16::from(reg), i as i32 * 8 - HOST_STACK_BYTES as i32))
                        .collect(),
                    workspace_saved,
                },
            });
        }
    }
    let bytes = ROOT_BYTES
        .checked_add(module.functions[root].peak)
        .ok_or(CompileError::Unsupported(crate::Limit::Frame))?;
    Ok(Code {
        bytes: a.finish()?,
        words: bytes / 8,
        sites,
        debug: debug_functions,
    })
}
