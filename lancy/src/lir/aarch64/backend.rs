//! Lower verified HIR, allocate target registers and emit AAPCS64 code.
mod compact;
mod compile;
mod divide;
mod groups;
mod half;
mod inst;
mod integer_float;
mod isel;
mod memory;
mod operations;
mod reduce;
mod remainder;
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
use crate::{
    CompileError,
    hir::{Function, Type, ssa::VerifiedBody},
    jit::{DebugFrame, DebugFunction},
    lir::layout,
    module::{FunctionId, FunctionIds, Module},
    support::slotmap::DenseMap,
};
use compile::{Compiled, Frame, TEMP_WORD_BYTES, compile};
use inst::Inst;
use regalloc2::{Allocation, Block, Edit, Function as _, InstOrEdit, PReg, RegClass};
use splice_target::{Profile, Simd};

pub(crate) use operations::Site;

// Status pointer, eight result words, and the address of the immutable libcall table.
pub(crate) const SITE_POINTER: usize = 9;
const ROOT_BYTES: usize = (SITE_POINTER + 1) * 8;
const ROOT_SAVES: [u8; 4] = [19, 20, 22, 30];
const CHILD_SAVES: [u8; 3] = [19, 22, 30];
const HOST_STACK_BYTES: u64 = 32;
const RESULT: (R, usize) = (R::X20, 8);

pub(crate) struct Code {
    pub bytes: Vec<u8>,
    pub words: usize,
    pub sites: Box<[Site]>,
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
                    CompileError::BranchOutOfRange {
                        offset: pc as usize,
                        target: target as usize,
                    }
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

fn number(a: Allocation) -> u8 {
    a.as_reg().expect("register operand").hw_enc() as u8
}

fn x(a: Allocation) -> R {
    R::x(number(a)).unwrap()
}

fn w(a: Allocation) -> R {
    R::w(number(a)).unwrap()
}

struct Emitter<'a> {
    a: &'a mut Assembler,
    frame: &'a Frame,
    sites: &'a mut Vec<Site>,
    entries: &'a DenseMap<FunctionId, Label>,
    config: Simd,
    epilogue: Label,
    byte_predicate: bool,
    failures: [Option<Label>; 5],
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

    fn trap_if(&mut self, cond: Cond, status: usize) -> Result {
        let trap = *self.failures[status - 1].get_or_insert_with(|| self.a.label());
        self.a.skip_if(invert(cond))?;
        self.a.jump(trap);
        Ok(())
    }

    fn traps(&mut self) -> Result {
        for (i, label) in self.failures.into_iter().enumerate() {
            if let Some(label) = label {
                self.a.bind(label);
                self.a.access(true, 8, 10, (R::X20, 0))?;
                self.a.imm(R::W9, (i + 1) as u64)?;
                self.a.access(false, 4, 9, (R::X10, 0))?;
                self.a.jump(self.epilogue);
            }
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
        allocs: &[Allocation],
        succs: &[Block],
        labels: &[Label],
        next: Option<Block>,
    ) -> Result {
        let n = |i| number(allocs[i]);
        let g = |i| x(allocs[i]);
        let word = |i| w(allocs[i]);
        let r = O::reg;
        let bytes = self.config.bytes;
        let size = |ty: Type| ty.bytes(bytes);
        match op {
            Inst::Arg(ty, offset, _) => self.value_access(true, ty, n(0), (R::X22, offset)),
            Inst::Literal(ty, bits, _) => self.literal(ty, n(0), bits),
            Inst::Copy(..) => {
                self.move_regs(allocs[0].as_reg().unwrap(), allocs[1].as_reg().unwrap())
            }
            Inst::BitMove(from, to, ..) => {
                if matches!(to, Type::F32 | Type::F64) {
                    self.float_move(to, true, n(1), n(0))
                } else {
                    self.float_move(from, false, n(0), n(1))
                }
            }
            Inst::Imm(code, bits, shift, ..) => self.a.put(
                code,
                &[r(word(1)), r(word(0)), O::imm_lsl(bits.into(), shift)],
            ),
            Inst::Int(int, ..) => self.int_binary(int, n(0), n(1), n(2)),
            Inst::WideAdd {
                subtract, scalable, ..
            } => self.wide_add(subtract, scalable, n(0), n(1), n(2)),
            Inst::WideMath(form, ..) => self.wide_math(form, n(0), n(1), n(2)),
            Inst::WideUnary(op, scalable, ..) => self.wide_unary(op, scalable, n(0), n(1)),
            Inst::Unary(form, ..) => self.unary(form, n(0), n(1)),
            Inst::Half(form, _, _, extra) => {
                let extra = std::array::from_fn(|i| extra[i].map(|_| n(i + 1)));
                self.half(form, n(0), extra, n(allocs.len() - 1))
            }
            Inst::HalfCast(from, to, ..) => self.half_cast(from, to, n(0), n(1)),
            Inst::IntegerFloatCast(form, ..) => self.integer_float_cast(form, n(0), n(1)),
            Inst::IntCast(from, to, ..) => self.int_cast(from, to, n(0), n(1)),
            Inst::WideCast(from, to, ..) => self.wide_cast(from, to, n(0), n(1)),
            Inst::FloatCast(code, from, to, ..) => self.float_cast(code, from, to, n(0), n(1)),
            Inst::PackedCast(code, arr, ..) => self.packed_cast(code, arr, n(0), n(1)),
            Inst::Select(form, ..) => self.select(form, n(0), n(1), n(2), n(3)),
            Inst::Fma(code, arr, ..) => self.fma(code, arr, [n(0), n(1), n(2)], n(3)),
            Inst::Sve2Ternary(code, ..) => {
                let z = |i| O::vec(R::z(n(i)).unwrap(), A::D);
                self.a.put(code, &[z(3), z(0), z(1), z(2)])
            }
            Inst::Shuffle(form, ..) => self.shuffle(form, [n(0), n(1), n(2)], n(3)),
            Inst::Compact { size, scalable, .. } => self.compact(size, scalable, n(0), n(1), n(2)),
            Inst::VectorDivide(form, ..) => self.vector_divide(form, n(0), n(1), n(2)),
            Inst::Rotate(form, ..) => self.rotate(form, n(0), n(1), n(2)),
            Inst::ScalarSaturate(code, size, ..) => {
                self.scalar_saturate(code, size, n(0), n(1), n(2))
            }
            Inst::NeonMul64(..) => self.neon_mul64(n(0), n(1), n(2)),
            Inst::FloatRemainder(form, ..) => self.float_remainder(form, n(0), n(1), n(2)),
            Inst::FloatSpecial(form, _, _, b) => {
                self.float_special(form, n(0), b.map(|_| n(1)), n(allocs.len() - 1))
            }
            Inst::WideMaskTest(all, scalable, ..) => self.wide_mask_test(all, scalable, n(0), n(1)),
            Inst::Reduce(form, ..) => self.reduce(form, n(0), n(1)),
            Inst::Expand(form, ..) => self.expand(form, n(0), n(1), n(2)),
            Inst::Widen(form, ..) => self.widen(form, n(0), n(1)),
            Inst::Narrow(form, ..) => self.narrow(form, n(0), n(1), n(2)),
            Inst::Float(code, arr, ..) => {
                let op = |i| O::vec(R::v(n(i)).unwrap(), arr);
                self.a.put(code, &[op(2), op(0), op(1)])
            }
            Inst::FloatCompare(cond, arr, ..) => {
                let op = |i| O::vec(R::v(n(i)).unwrap(), arr);
                self.a.put(C::Fcmp_scalar, &[op(0), op(1)])?;
                self.bool_mask(cond, n(2))
            }
            Inst::SveIota(lane, _) => self.sve_iota(lane, n(0)),
            Inst::MaskTest(all, arr, ..) => self.mask_test(all, arr, n(0), n(1)),
            Inst::LaneAccess { lane, insert, .. } => self.lane_access(
                lane,
                n(0),
                word(1),
                insert.map(|_| n(2)),
                n(allocs.len() - 1),
            ),
            Inst::Splat(code, arr, ty, ..) => {
                let input = if matches!(ty, Type::F32 | Type::F64) {
                    self.float_move(ty, false, n(0), 9)?;
                    9
                } else {
                    n(0)
                };
                let src = if size(ty) == 8 {
                    R::x(input)
                } else {
                    R::w(input)
                }
                .unwrap();
                let dst = if code == C::Dup_z_r {
                    R::z(n(1))
                } else {
                    R::v(n(1))
                }
                .unwrap();
                self.a.put(code, &[O::vec(dst, arr), r(src)])
            }
            Inst::SveSplat128(..) => self.a.put(
                C::Dup_q0_z,
                &[r(R::z(n(1)).unwrap()), r(R::z(n(0)).unwrap())],
            ),
            Inst::Packed(op, ..) => self.packed(op, n(0), n(1), n(2)),
            Inst::PackedShiftImm(code, arr, count, ..) => {
                let reg = if matches!(arr, A::B | A::H | A::S | A::D) {
                    R::z
                } else {
                    R::v
                };
                let operand = |i| O::vec(reg(n(i)).unwrap(), arr);
                self.a
                    .put(code, &[operand(1), operand(0), O::imm(count.into())])
            }
            Inst::Check(..) => {
                self.a.put(
                    C::Subs_32_addsub_shift,
                    &[r(R::WZR), r(word(0)), r(word(1))],
                )?;
                self.trap_if(Cond::Cs, 1)
            }
            Inst::CheckOverflow(..) => {
                self.a
                    .put(C::Subs_32_addsub_shift, &[r(R::WZR), r(word(0)), r(R::WZR)])?;
                self.trap_if(Cond::Ne, 4)
            }
            Inst::CallScratch(..) => self.a.addr(g(0), R::X19, self.frame.scratch_base),
            Inst::StorePacketWord(offset, _) => self.value_access(
                false,
                Type::I32,
                n(0),
                (R::X19, self.frame.scratch_base + offset as usize),
            ),
            Inst::RustCall(..) | Inst::RustPacketCall(..) | Inst::RustContextCall(..) => {
                let entry = match op {
                    Inst::RustCall(import, ..) => (import.resolve)() as usize,
                    Inst::RustPacketCall(import, _) => {
                        self.a.imm(R::X0, std::ptr::from_ref(import) as u64)?;
                        self.a.addr(R::X1, R::X19, self.frame.scratch_base)?;
                        self.a.imm(R::X2, u64::from(import.arity))?;
                        splice_target::abi::invoke_packet_import as *const () as usize
                    }
                    Inst::RustContextCall(import, ..) => import.address(),
                    _ => unreachable!(),
                };
                // The frame has already saved LR and keeps the workspace in
                // callee-saved registers across this AAPCS64 call.
                self.a.imm(R::X16, entry as u64)?;
                self.a.put(C::Blr_64_branch_reg, &[r(R::X16)])?;
                self.byte_predicate = false;
                self.a.put(
                    C::Ubfm_64m_bitfield,
                    &[r(R::X16), r(R::X0), O::imm(32), O::imm(63)],
                )?;
                self.a
                    .put(C::Subs_32_addsub_imm, &[r(R::WZR), r(R::W16), O::imm(0)])?;
                self.trap_if(Cond::Ne, 5)
            }
            Inst::Offset(scale, ..) => self.a.put(
                C::Add_64_addsub_sxtw,
                &[
                    r(g(2)),
                    r(g(0)),
                    O::shifted(word(1), ShiftKind::Lsl, scale as u8),
                ],
            ),
            Inst::Load(ty, ..) => self.value_access(true, ty, n(1), (g(0), 0)),
            Inst::Store(ty, ..) => self.value_access(false, ty, n(1), (g(0), 0)),
            Inst::DebugStore(ty, offset, _) => self.value_access(
                false,
                ty,
                n(0),
                (R::X19, self.frame.debug_base + offset as usize),
            ),
            Inst::DebugPoint(_) => self.a.put(C::Nop_HI_hints, &[]),
            Inst::Masked {
                kind,
                ty,
                indices,
                dst,
                ..
            } => {
                let indices = indices.map(|_| n(3));
                let dst = dst.map(|_| n(allocs.len() - 1));
                match kind {
                    inst::Masked::Neon => {
                        self.neon_masked(ty.lane().unwrap().bytes(), g(0), n(1), n(2), indices, dst)
                    }
                    inst::Masked::Sve { code, arrangement } => {
                        self.sve_masked((code, arrangement), g(0), n(1), n(2), indices, dst)
                    }
                    inst::Masked::SveNarrow(code) => self.sve_narrow_masked(
                        (code, ty.lane().unwrap().bytes()),
                        g(0),
                        [n(1), n(2), indices.unwrap()],
                        dst,
                    ),
                    inst::Masked::SveWide(code) => {
                        self.sve_wide_masked(code, g(0), n(1), n(2), indices, dst)
                    }
                }
            }
            Inst::StoreArg(ty, offset, _) => {
                self.value_access(false, ty, n(0), (R::X19, self.frame.outgoing + offset))
            }
            Inst::Call(target, ..) => {
                self.a.addr(R::X0, R::X19, self.frame.outgoing)?;
                self.a.addr(R::X1, R::X19, self.frame.bytes)?;
                self.a.branch(C::Bl_only_branch_imm, self.entries[target]);
                self.byte_predicate = false;
                self.check_status()
            }
            Inst::LibArg(ty, index, _) => self.value_access(false, ty, n(0), self.temp(index)),
            Inst::Libcall(action, input, output, sizes) => {
                self.primitive(action, input, output, sizes)
            }
            Inst::LibResult(ty, _) => self.value_access(true, ty, n(0), self.temp(3)),
            Inst::Jump => {
                if Some(succs[0]) != next {
                    self.a.jump(labels[succs[0].index()]);
                }
                Ok(())
            }
            Inst::Branch(_) => {
                self.a
                    .put(C::Subs_32_addsub_imm, &[r(R::WZR), r(word(0)), O::imm(0)])?;
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
            Inst::Return(ty, _, root) => {
                if root && size(ty) > 8 {
                    self.value_access(false, ty, n(0), RESULT)?;
                } else if root && matches!(ty, Type::F32 | Type::F64) {
                    self.float_move(ty, false, n(0), 0)?;
                }
                self.a.jump(self.epilogue);
                Ok(())
            }
        }
    }
    fn temp(&self, index: usize) -> (R, usize) {
        (R::X19, self.frame.temporary + index * TEMP_WORD_BYTES)
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

pub(crate) fn emit(source: Module<VerifiedBody<'_>>, ids: &FunctionIds<Function>) -> Result<Code> {
    emit_mode(source, ids, false)
}

pub(crate) fn emit_debug(
    source: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<Code> {
    emit_mode(source, ids, true)
}

fn emit_mode(
    source: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
    debug: bool,
) -> Result<Code> {
    let config = splice_target::simd().map_err(CompileError::SimdProfile)?;
    let module = compile(source, ids, config)?;
    emit_compiled(&module, config, debug)
}

fn emit_compiled(module: &Module<Compiled>, config: Simd, debug: bool) -> Result<Code> {
    let _span = tracing::debug_span!(target: "splice::compile", "emit_machine_code").entered();
    let root = module.call_graph.root();
    let mut a = Assembler::default();
    let mut sites = Vec::new();
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
            failures: [None; 5],
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
                        if debug && let Inst::DebugPoint(id) = inst {
                            points.push((id, out.a.words.len() as u64 * 4));
                        }
                        out.inst(
                            inst,
                            machine.allocation.inst_allocs(i),
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
        .ok_or(CompileError::FrameOverflow)?;
    Ok(Code {
        bytes: a.finish()?,
        words: bytes / 8,
        sites: sites.into_boxed_slice(),
        debug: debug_functions,
    })
}
