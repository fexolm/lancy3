//! Selected x64 instructions. R11 addresses the current workspace frame.
//! R10 is the allocator's move scratch and is available inside a pseudo-op.
pub(super) mod convert;
pub(super) mod int;
mod names;
pub(super) mod simd;
pub(super) use super::abi::SCRATCH;
use super::abi::{MASKS, Machine, gprs, vectors};
use crate::hir::{Binary, Lane, RustFunction, Scale, Type};
use crate::support::numeric::Action;
use crate::support::slotmap::DenseMap;
use crate::{
    lir::{TargetInst, frame::FrameUse},
    module::FunctionId,
};
pub(super) use int::{Alu, Cond, Int, IntUnary, Shift};
use regalloc2::{MachineEnv, Operand, PReg, PRegSet, RegClass, VReg};
use simd::packed::rearrange::PairedShuffle;
use simd::{MaskedMemory, Native};

#[derive(Clone, Copy, Debug)]
pub(super) enum VectorMask<R> {
    Lanes(R),
    Register(Lane, R),
    Compare {
        lhs: R,
        rhs: R,
        lane: Lane,
        predicate: i32,
        active: Option<R>,
    },
}

impl<R> VectorMask<R> {
    pub(super) fn map<T>(self, mut reg: impl FnMut(R) -> T) -> VectorMask<T> {
        match self {
            Self::Lanes(v) => VectorMask::Lanes(reg(v)),
            Self::Register(lane, v) => VectorMask::Register(lane, reg(v)),
            Self::Compare {
                lhs,
                rhs,
                lane,
                predicate,
                active,
            } => VectorMask::Compare {
                lhs: reg(lhs),
                rhs: reg(rhs),
                lane,
                predicate,
                active: active.map(reg),
            },
        }
    }
}

impl VectorMask<VReg> {
    fn operands(self, out: &mut Vec<Operand>) {
        match self {
            Self::Lanes(v) | Self::Register(_, v) => out.push(Operand::reg_use(v)),
            Self::Compare {
                lhs, rhs, active, ..
            } => {
                out.extend([Operand::reg_use(lhs), Operand::reg_use(rhs)]);
                out.extend(active.map(Operand::reg_use));
            }
        }
    }
}

const AX: PReg = PReg::new(0, RegClass::Int);
const CX: PReg = PReg::new(1, RegClass::Int);
const DX: PReg = PReg::new(2, RegClass::Int);

// XMM, YMM and ZMM are aliases. Scalars and packed values must share one class.
pub(super) fn class(ty: Type) -> RegClass {
    if matches!(
        ty,
        Type::Vector(_) | Type::F32 | Type::F64 | Type::U128 | Type::I128
    ) {
        RegClass::Vector
    } else {
        RegClass::Int
    }
}
fn return_reg(v: VReg) -> PReg {
    PReg::new(0, v.class())
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Rhs {
    Reg(VReg),
    Imm(i32),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Quotient {
    PowerOfTwo(i32),
    Reciprocal { multiplier: u32, shift: i32 },
}

impl Rhs {
    fn operands(self, out: &mut Vec<Operand>) {
        if let Self::Reg(r) = self {
            out.push(Operand::any_use(r));
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Address {
    pub base: VReg,
    pub index: Option<(VReg, Scale)>,
    pub disp: i32,
}

impl Address {
    fn operands(self, out: &mut Vec<Operand>) {
        out.push(Operand::reg_use(self.base));
        if let Some((index, _)) = self.index {
            out.push(Operand::reg_use(index));
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum SelectCond {
    Nonzero(VReg),
    /// An integer comparison of the operand size in bytes.
    Compare(Cond, u8, VReg, Rhs),
}

/// A scalar integer narrower than its register holds its value in the low
/// `size` bytes only; the upper bits are undefined. Producers leave whatever
/// is convenient there (a signed `Cast` sign-extends, a load zero-extends,
/// `Alu` keeps the carry-out), and every consumer reads just the low
/// bytes: a narrow store, a `Cast` through `movsx`/`movzx`, an operation at
/// the type's operand size or after extending in place, lanes inserted by size, and a root return the caller truncates to the result type.
#[derive(Clone, Copy, Debug)]
pub(super) enum Inst {
    Argv(VReg),
    LoadArg {
        dst: VReg,
        argv: Option<VReg>,
        ty: Type,
        index: usize,
    },
    Native(NativeId),
    /// Call the shared numeric fallback (`lir::fallback`).
    Fallback(ExpansionId),
    Literal(Lane, u128, VReg),
    Iota(Lane, VReg),
    /// Every lane holds the low `lane.bytes()` bytes of the bits: a load
    /// from the constant pool, or a zeroing idiom.
    SplatLiteral(Lane, u64, VReg),
    BitMove(Type, Type, VReg, VReg),
    Imm(VReg, i32),
    Copy(VReg, VReg),
    Alu(Alu, Int, VReg, VReg, Rhs),
    Shift {
        op: Shift,
        int: Int,
        dst: VReg,
        value: VReg,
        count: Rhs,
    },
    IntUnary(IntUnary, Int, VReg, VReg, Option<VReg>),
    Saturating {
        sub: bool,
        int: Int,
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    /// A cast from or to a float scalar (see [`convert`]).
    Convert {
        from: Lane,
        to: Lane,
        dst: VReg,
        src: VReg,
        temp: Option<VReg>,
    },
    /// The class of a 16-bit float, in place, as an `i32` mask.
    Classify(crate::hir::Unary, Lane, VReg, VReg),
    /// Sign- or zero-extend an integer to the whole register.
    Extend(Int, VReg, VReg),
    /// The comparison of integers of the given size, as a mask.
    Compare(Cond, u8, VReg, VReg, Rhs),
    /// Choose between general registers, of 64 bits when `wide`.
    Select {
        wide: bool,
        dst: VReg,
        no: VReg,
        cond: SelectCond,
        yes: VReg,
    },
    /// Choose between vector registers by a scalar condition.
    SelectVector {
        dst: VReg,
        no: VReg,
        cond: VReg,
        yes: VReg,
    },
    BrCmp(Cond, u8, VReg, Rhs),
    /// The high half of a dividend: its sign, or zero when unsigned.
    DivideHigh {
        int: Int,
        lo: VReg,
        hi: VReg,
    },
    /// Hardware division of operands extended to 32 or 64 bits; the type
    /// decides signedness and the overflowing dividend.
    Divide {
        int: Int,
        remainder: bool,
        dst: VReg,
        lo: VReg,
        hi: VReg,
        divisor: VReg,
    },
    DivImm {
        remainder: bool,
        dst: VReg,
        value: VReg,
        divisor: i32,
        quotient: Quotient,
    },
    CheckBounds(VReg, VReg),
    TrapIf(VReg, crate::Trap),
    Trap(crate::Trap),
    CheckMask(VectorMask<VReg>),
    BranchAny(VectorMask<VReg>),
    BranchAll(Lane, VReg),
    AllMask(Lane, VReg, VReg),
    AnyMask(VReg, VectorMask<VReg>),
    SelectMask(SelectId),
    MaskCompare(Lane, i32, VReg, VReg, VReg),
    MaskLogic(Lane, Binary, VReg, VReg, VReg),
    MaskNot(Lane, VReg, VReg),
    MaskImmediate(VReg, u64),
    /// The predicate of the first `count` of `lanes` lanes: dst, count, temp.
    MaskPrefix(u8, VReg, VReg, VReg),
    MaskCount(VReg, VReg),
    MaskCompact(Lane, bool, VReg, VReg, VReg),
    CompactLookup(bool, VReg, VReg, VReg, VReg),
    /// Store the `mask` lanes of `data` packed at `base`, at most `limit`.
    CompactStore {
        base: VReg,
        data: VReg,
        mask: VReg,
        limit: VReg,
        temps: [VReg; 2],
    },
    Lea(VReg, Address),
    Load(Type, VReg, Address),
    PackedMemory(Binary, Lane, VReg, VReg, Address),
    BroadcastMemory(Lane, VReg, Address),
    BroadcastLane(VReg, VReg, u8),
    ExtractLane(Lane, VReg, VReg, u8),
    ShiftDwords(VReg, VReg, u8),
    PairedShuffle(PairedShuffle, VReg, VReg, VReg),
    DynamicLane {
        lane: Lane,
        checked: bool,
        dst: VReg,
        value: VReg,
        index: VReg,
        replacement: Option<VReg>,
        temp: Option<VReg>,
    },
    VectorShiftImm(Binary, Lane, VReg, VReg, u8),
    Masked(MaskedId),
    Store(Type, Address, VReg),
    DebugStore(Type, u32, VReg),
    DebugPoint(u32),
    StoreArg(usize, Type, VReg),
    Call {
        target: FunctionId,
        dst: VReg,
        clobbers: ClobberId,
    },
    RustCall(&'static RustFunction, [VReg; 4], VReg),
    /// Address of the frame's scratch memory at a byte offset, with the
    /// region's size in bytes.
    CallScratch(u32, u32, VReg),
    Jump,
    Branch(VReg),
    Ret(VReg),
}

crate::support::slotmap::slot_key!(pub(super) NativeId);
crate::support::slotmap::slot_key!(pub(super) ExpansionId);
crate::support::slotmap::slot_key!(pub(super) SelectId);
crate::support::slotmap::slot_key!(pub(super) MaskedId);
crate::support::slotmap::slot_key!(pub(super) ClobberId);

/// An operation with a native vector or scalar form.
#[derive(Clone, Copy, Debug)]
pub(super) struct NativeInst {
    pub native: Native,
    pub dst: VReg,
    pub args: [Option<VReg>; 3],
    pub temps: [Option<VReg>; 6],
}

/// An operation computed out of line by the numeric fallback, from typed
/// arguments.
#[derive(Clone, Copy, Debug)]
pub(super) struct Expansion {
    pub action: Action,
    pub input: Type,
    pub output: Type,
    pub dst: VReg,
    pub args: [Option<(Type, VReg)>; 3],
}

/// A lane select by a vector mask.
#[derive(Clone, Copy, Debug)]
pub(super) struct MaskSelect<R = VReg> {
    pub lane: Lane,
    pub mask: VectorMask<R>,
    pub yes: R,
    pub no: R,
    pub dst: R,
}

/// The function's instruction payloads too large to keep in [`Inst`],
/// which holds their keys. Selection stores a payload and pushes the key.
#[derive(Default)]
pub(super) struct Details {
    pub natives: DenseMap<NativeId, NativeInst>,
    pub expansions: DenseMap<ExpansionId, Expansion>,
    pub selects: DenseMap<SelectId, MaskSelect>,
    pub masked: DenseMap<MaskedId, MaskedMemory<VReg>>,
    pub clobbers: DenseMap<ClobberId, PRegSet>,
}

impl Details {
    pub(super) fn native(&mut self, native: NativeInst) -> Inst {
        Inst::Native(self.natives.insert(native))
    }
    pub(super) fn fallback(&mut self, expansion: Expansion) -> Inst {
        Inst::Fallback(self.expansions.insert(expansion))
    }
    pub(super) fn select(&mut self, select: MaskSelect) -> Inst {
        Inst::SelectMask(self.selects.insert(select))
    }
    pub(super) fn masked(&mut self, access: MaskedMemory<VReg>) -> Inst {
        Inst::Masked(self.masked.insert(access))
    }
    /// A call writes what the callee writes, except its result register.
    pub(super) fn call(&mut self, target: FunctionId, dst: VReg, mut written: PRegSet) -> Inst {
        written.remove(return_reg(dst));
        Inst::Call {
            target,
            dst,
            clobbers: self.clobbers.insert(written),
        }
    }
}

impl TargetInst for Inst {
    type Machine = Machine;
    type Details = Details;
    fn operands(&self, machine: &Machine, details: &Details, out: &mut Vec<Operand>) {
        use Operand as O;
        match *self {
            Self::Argv(v) => out.push(O::reg_fixed_def(v, machine.abi.args[0])),
            Self::LoadArg { dst, argv, .. } => {
                out.extend(argv.map(O::reg_use));
                out.push(O::reg_def(dst));
            }
            Self::Native(id) => {
                let NativeInst {
                    native,
                    dst: d,
                    args,
                    temps,
                } = details.natives[id];
                out.extend(args.into_iter().flatten().map(O::reg_use));
                out.push(if native.separate_output() {
                    O::reg_def_at_start(d)
                } else if native.tied() {
                    O::reg_reuse_def(d, 0)
                } else {
                    O::reg_def(d)
                });
                out.extend(temps.into_iter().flatten().map(O::reg_temp));
            }
            Self::Fallback(id) => {
                let Expansion { dst, args, .. } = details.expansions[id];
                out.extend(args.into_iter().flatten().map(|(_, reg)| O::reg_use(reg)));
                out.push(O::reg_fixed_def(dst, return_reg(dst)));
            }
            Self::Literal(_, _, d) | Self::Iota(_, d) | Self::SplatLiteral(_, _, d) => {
                out.push(O::reg_def(d))
            }
            Self::BitMove(_, _, d, a) => out.extend([O::reg_use(a), O::reg_def(d)]),
            Self::Imm(d, _) => out.push(O::reg_def(d)),
            Self::Copy(d, a) => out.extend([O::reg_use(a), O::reg_reuse_def(d, 0)]),
            Self::Alu(_, _, d, a, b) => {
                out.push(O::reg_use(a));
                b.operands(out);
                out.push(O::reg_reuse_def(d, 0));
            }
            Self::IntUnary(_, _, d, a, temp) => {
                out.extend([O::reg_use(a), O::reg_reuse_def(d, 0)]);
                out.extend(temp.map(O::reg_temp));
            }
            Self::Convert {
                from,
                to,
                dst,
                src,
                temp,
            } => {
                out.push(O::reg_use(src));
                out.push(if convert::soft(from, to) && dst.class() == RegClass::Int {
                    O::reg_fixed_def(dst, AX)
                } else {
                    O::reg_def(dst)
                });
                out.extend(temp.map(O::reg_temp));
            }
            Self::Classify(_, _, d, a) => out.extend([O::reg_use(a), O::reg_reuse_def(d, 0)]),
            Self::Saturating { dst, lhs, rhs, .. } => {
                out.extend([O::reg_use(lhs), O::reg_use(rhs), O::reg_reuse_def(dst, 0)])
            }
            Self::Shift {
                dst, value, count, ..
            } => {
                out.push(O::reg_use(value));
                if let Rhs::Reg(r) = count {
                    out.push(O::reg_fixed_use(r, CX));
                }
                out.push(O::reg_reuse_def(dst, 0));
            }
            Self::Compare(_, _, d, a, b) => {
                out.push(O::reg_use(a));
                b.operands(out);
                out.push(O::reg_def(d));
            }
            Self::Select {
                dst, no, cond, yes, ..
            } => {
                out.extend([O::reg_use(no), O::any_use(yes)]);
                match cond {
                    SelectCond::Nonzero(v) => out.push(O::reg_use(v)),
                    SelectCond::Compare(_, _, a, b) => {
                        out.push(O::reg_use(a));
                        b.operands(out);
                    }
                }
                out.push(O::reg_reuse_def(dst, 0));
            }
            Self::SelectVector { dst, no, cond, yes } => out.extend([
                O::reg_use(no),
                O::reg_use(yes),
                O::reg_use(cond),
                O::reg_reuse_def(dst, 0),
            ]),
            Self::BrCmp(_, _, a, b) => {
                out.push(O::reg_use(a));
                b.operands(out);
            }
            Self::DivideHigh { lo, hi, .. } => {
                out.extend([O::reg_fixed_use(lo, AX), O::reg_fixed_def(hi, DX)])
            }
            Self::Divide {
                remainder,
                dst,
                lo,
                hi,
                divisor,
                ..
            } => out.extend([
                O::reg_fixed_use(lo, AX),
                O::reg_fixed_use(hi, DX),
                O::reg_use(divisor),
                O::reg_fixed_def(dst, if remainder { DX } else { AX }),
            ]),
            Self::DivImm { dst, value, .. } => {
                out.extend([O::reg_use(value), O::reg_def_at_start(dst)])
            }
            Self::CheckBounds(i, n) => out.extend([O::reg_use(i), O::any_use(n)]),
            Self::TrapIf(v, _) => out.push(O::reg_use(v)),
            Self::Extend(_, d, a) => out.extend([O::reg_use(a), O::reg_def(d)]),
            Self::Lea(d, addr) | Self::Load(_, d, addr) | Self::BroadcastMemory(_, d, addr) => {
                addr.operands(out);
                out.push(O::reg_def(d));
            }
            Self::PackedMemory(_, _, d, value, addr) => {
                addr.operands(out);
                out.extend([O::reg_use(value), O::reg_def(d)]);
            }
            Self::BroadcastLane(d, src, _)
            | Self::ExtractLane(_, d, src, _)
            | Self::VectorShiftImm(_, _, d, src, _)
            | Self::ShiftDwords(d, src, _) => out.extend([O::reg_use(src), O::reg_def(d)]),
            Self::AnyMask(d, mask) => {
                mask.operands(out);
                out.push(O::reg_def(d));
            }
            Self::SelectMask(id) => {
                let MaskSelect {
                    mask, yes, no, dst, ..
                } = details.selects[id];
                mask.operands(out);
                out.extend([O::reg_use(yes), O::reg_use(no), O::reg_def(dst)]);
            }
            Self::MaskCompare(_, _, d, x, y)
            | Self::MaskLogic(_, _, d, x, y)
            | Self::PairedShuffle(_, d, x, y)
            | Self::MaskCompact(_, _, d, x, y) => {
                out.extend([O::reg_use(x), O::reg_use(y), O::reg_def(d)]);
            }
            Self::MaskNot(_, d, x) | Self::AllMask(_, d, x) | Self::MaskCount(d, x) => {
                out.extend([O::reg_use(x), O::reg_def(d)]);
            }
            Self::MaskPrefix(_, d, count, temp) => {
                out.extend([O::reg_use(count), O::reg_def(d), O::reg_temp(temp)]);
            }
            Self::MaskImmediate(d, _) => out.push(O::reg_def(d)),
            Self::DynamicLane {
                dst,
                value,
                index,
                replacement,
                temp,
                ..
            } => {
                out.extend([O::reg_use(value), O::reg_use(index)]);
                out.extend(replacement.map(O::reg_use));
                out.push(O::reg_def(dst));
                out.extend(temp.map(O::reg_temp));
            }
            Self::CompactLookup(_, d, x, m, index) => {
                let cpu = machine.cpu;
                let dst = if cpu.bytes == 16 && !cpu.isa.ssse3 {
                    // The SSE2 sequence reads the source after writing the result.
                    O::reg_def_at_start(d)
                } else {
                    O::reg_def(d)
                };
                out.extend([O::reg_use(x), O::reg_use(m), dst, O::reg_temp(index)]);
            }
            Self::CompactStore {
                base,
                data,
                mask,
                limit,
                temps: [bits, prefix],
            } => out.extend([
                O::reg_use(base),
                O::reg_use(data),
                O::reg_use(mask),
                O::reg_use(limit),
                O::reg_temp(bits),
                O::reg_temp(prefix),
            ]),
            Self::Masked(id) => {
                let MaskedMemory {
                    base,
                    index,
                    temp,
                    mask,
                    value,
                    dst,
                    ..
                } = details.masked[id];
                out.push(O::reg_use(base));
                out.extend(index.map(O::reg_use));
                mask.operands(out);
                out.extend(value.map(O::reg_use));
                out.extend(dst.map(O::reg_def));
                out.extend(temp.map(O::reg_temp));
            }
            Self::Store(_, addr, value) => {
                addr.operands(out);
                out.push(O::reg_use(value));
            }
            Self::CheckMask(mask) | Self::BranchAny(mask) => mask.operands(out),
            Self::BranchAll(_, v)
            | Self::StoreArg(_, _, v)
            | Self::DebugStore(_, _, v)
            | Self::Branch(v) => out.push(O::reg_use(v)),
            Self::Call { dst, .. } => out.push(O::reg_fixed_def(dst, return_reg(dst))),
            Self::RustCall(_, args, dst) => {
                out.extend(
                    args.into_iter()
                        .zip(machine.abi.args)
                        .map(|(arg, reg)| O::reg_fixed_use(arg, reg)),
                );
                out.push(O::reg_fixed_def(dst, AX));
            }
            Self::CallScratch(_, _, dst) => out.push(O::reg_def(dst)),
            Self::Ret(v) => out.push(O::reg_fixed_use(v, return_reg(v))),
            Self::Jump | Self::DebugPoint(_) | Self::Trap(_) => {}
        }
    }
    fn clobbers(&self, machine: &Machine, details: &Details) -> PRegSet {
        let abi = machine.abi;
        // A wider value in a preserved vector register loses its upper lanes.
        let wide = machine.register_bytes > 16;
        let callee = || {
            let mut regs = abi.volatile | MASKS | abi.volatile_vectors;
            if wide {
                regs.union_from(abi.saved & vectors(0, 16));
            }
            regs
        };
        match self {
            Self::Call { clobbers, .. } => details.clobbers[*clobbers],
            Self::RustCall(..) => {
                let mut regs = callee();
                regs.remove(AX);
                regs
            }
            Self::Fallback(id) => {
                let mut set = callee();
                set.remove(return_reg(details.expansions[*id].dst));
                set
            }
            Self::Divide { remainder, .. } => {
                PRegSet::empty().with(if *remainder { AX } else { DX })
            }
            Self::Convert { from, to, dst, .. } if convert::soft(*from, *to) => {
                let mut set = gprs(&[0, 1, 2, 8, 9]);
                if dst.class() == RegClass::Int {
                    set.remove(AX);
                }
                set
            }
            _ => PRegSet::empty(),
        }
    }
    fn frame_use(&self, machine: &Machine) -> Option<FrameUse> {
        Some(match *self {
            Self::DebugStore(ty, offset, _) => {
                FrameUse::Debug(offset as usize + ty.bytes(machine.cpu.bytes))
            }
            Self::CallScratch(offset, bytes, _) => {
                FrameUse::Scratch(offset as usize + bytes as usize)
            }
            // Outgoing arguments go to the callee's frame above this one;
            // the fallback and dynamic lane accesses reserve stack below RSP.
            _ => return None,
        })
    }
    fn is_branch(&self) -> bool {
        matches!(
            self,
            Self::Jump
                | Self::Branch(_)
                | Self::BranchAny(_)
                | Self::BranchAll(..)
                | Self::BrCmp(..)
        )
    }
    fn is_ret(&self) -> bool {
        matches!(self, Self::Ret(_))
    }
    fn machine_env(machine: &Machine) -> &'static MachineEnv {
        machine.abi.env()
    }
    fn spillslot_size(machine: &Machine, class: RegClass) -> usize {
        if class == RegClass::Vector {
            machine.register_bytes / 8
        } else {
            1
        }
    }
}

#[cfg(test)]
#[test]
fn instructions_stay_small() {
    // `Function` stores and the scheduler permutes instructions by value.
    assert!(
        std::mem::size_of::<Inst>() <= 32,
        "{}",
        std::mem::size_of::<Inst>()
    );
}
