//! Selected x64 instructions. R11 addresses the current workspace frame.
//! R10 is the allocator's move scratch and is available inside a pseudo-op.
pub(super) mod lanes;
pub(super) mod simd;
pub(super) use super::abi::SCRATCH;
use super::abi::{MASKS, Machine, gprs, vectors};
use crate::hir::operation::Action;
use crate::hir::{Binary, Lane, RustFunction, Scale, Type};
use crate::{lir::TargetInst, module::FunctionId};
use regalloc2::{MachineEnv, Operand, PReg, PRegSet, RegClass, VReg};
use simd::Native;
use simd::packed::rearrange::PairedShuffle;

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
pub(super) enum Alu {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Cond {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cond {
    pub fn inverse(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Ge => Self::Lt,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
        }
    }
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
    Compare(Cond, VReg, Rhs),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Inst {
    Argv(VReg),
    LoadArg {
        dst: VReg,
        argv: Option<VReg>,
        ty: Type,
        index: usize,
    },
    Native(Native, VReg, [Option<VReg>; 3], [Option<VReg>; 6]),
    Lanes {
        action: Action,
        input: Type,
        output: Type,
        dst: VReg,
        args: [Option<(Type, VReg)>; 3],
    },
    Literal(Lane, u128, VReg),
    Iota(Lane, VReg),
    BitMove(Type, Type, VReg, VReg),
    IntBinary(Binary, Type, VReg, VReg, VReg),
    Imm(VReg, i32),
    Copy(VReg, VReg),
    Alu(Alu, VReg, VReg, Rhs),
    Shift {
        left: bool,
        dst: VReg,
        value: VReg,
        count: Rhs,
    },
    Compare(Cond, VReg, VReg, Rhs),
    SelectI32 {
        dst: VReg,
        no: VReg,
        cond: SelectCond,
        yes: VReg,
    },
    BrCmp(Cond, VReg, Rhs),
    Cdq {
        lo: VReg,
        hi: VReg,
    },
    IDiv {
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
    SelectMask {
        lane: Lane,
        mask: VectorMask<VReg>,
        yes: VReg,
        no: VReg,
        dst: VReg,
    },
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
    Movsxd(VReg, VReg),
    Lea(VReg, Address),
    Load(Type, VReg, Address),
    PackedMemory(Binary, Lane, VReg, VReg, Address),
    BroadcastMemory(Lane, VReg, Address),
    BroadcastLane(VReg, VReg, u8),
    ExtractLane(Lane, VReg, VReg, u8),
    ShiftDwords(VReg, VReg, u8),
    PairedShuffle(PairedShuffle, VReg, VReg, VReg),
    DwordLane {
        lane: Lane,
        checked: bool,
        dst: VReg,
        value: VReg,
        index: VReg,
        replacement: Option<VReg>,
        temp: Option<VReg>,
    },
    VectorShiftImm(Binary, Lane, VReg, VReg, u8),
    Masked {
        ty: Type,
        base: VReg,
        indices: Option<VReg>,
        temp: Option<VReg>,
        mask: VectorMask<VReg>,
        value: Option<VReg>,
        dst: Option<VReg>,
    },
    Store(Type, Address, VReg),
    DebugStore(Type, u32, VReg),
    DebugPoint(u32),
    StoreArg(usize, Type, VReg),
    Call {
        target: FunctionId,
        dst: VReg,
        clobbers: PRegSet,
    },
    RustCall(&'static RustFunction, [VReg; 4], VReg),
    /// Address of the frame's scratch memory at a byte offset, with the
    /// region's size in bytes.
    CallScratch(u32, u32, VReg),
    Jump,
    Branch(VReg),
    Ret(VReg),
}

impl Inst {
    pub fn call(target: FunctionId, dst: VReg, mut written: PRegSet) -> Self {
        written.remove(return_reg(dst));
        let clobbers = written;
        Self::Call {
            target,
            dst,
            clobbers,
        }
    }
}

impl TargetInst for Inst {
    type Machine = Machine;
    fn operands(&self, machine: &Machine, out: &mut Vec<Operand>) {
        use Operand as O;
        match *self {
            Self::Argv(v) => out.push(O::reg_fixed_def(v, machine.abi.args[0])),
            Self::LoadArg { dst, argv, .. } => {
                out.extend(argv.map(O::reg_use));
                out.push(O::reg_def(dst));
            }
            Self::Native(native, d, args, temps) => {
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
            Self::Lanes { dst, args, .. } => {
                out.extend(args.into_iter().flatten().map(|(_, reg)| O::reg_use(reg)));
                out.push(O::reg_fixed_def(dst, return_reg(dst)));
            }
            Self::Literal(_, _, d) | Self::Iota(_, d) => out.push(O::reg_def(d)),
            Self::BitMove(_, _, d, a) => out.extend([O::reg_use(a), O::reg_def(d)]),
            Self::IntBinary(_, _, d, a, b) => {
                out.extend([O::reg_use(a), O::reg_use(b), O::reg_reuse_def(d, 0)])
            }
            Self::Imm(d, _) => out.push(O::reg_def(d)),
            Self::Copy(d, a) => out.extend([O::reg_use(a), O::reg_reuse_def(d, 0)]),
            Self::Alu(_, d, a, b) => {
                out.push(O::reg_use(a));
                b.operands(out);
                out.push(O::reg_reuse_def(d, 0));
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
            Self::Compare(_, d, a, b) => {
                out.push(O::reg_use(a));
                b.operands(out);
                out.push(O::reg_def(d));
            }
            Self::SelectI32 { dst, no, cond, yes } => {
                out.extend([O::reg_use(no), O::any_use(yes)]);
                match cond {
                    SelectCond::Nonzero(v) => out.push(O::reg_use(v)),
                    SelectCond::Compare(_, a, b) => {
                        out.push(O::reg_use(a));
                        b.operands(out);
                    }
                }
                out.push(O::reg_reuse_def(dst, 0));
            }
            Self::BrCmp(_, a, b) => {
                out.push(O::reg_use(a));
                b.operands(out);
            }
            Self::Cdq { lo, hi } => {
                out.extend([O::reg_fixed_use(lo, AX), O::reg_fixed_def(hi, DX)])
            }
            Self::IDiv {
                remainder,
                dst,
                lo,
                hi,
                divisor,
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
            Self::Movsxd(d, a) => out.extend([O::reg_use(a), O::reg_def(d)]),
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
            Self::SelectMask {
                mask, yes, no, dst, ..
            } => {
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
            Self::DwordLane {
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
            Self::Masked {
                base,
                indices,
                temp,
                mask,
                value,
                dst,
                ..
            } => {
                out.push(O::reg_use(base));
                out.extend(indices.map(O::reg_use));
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
    fn clobbers(&self, machine: &Machine) -> PRegSet {
        let abi = machine.abi;
        // A wider value in a preserved vector register loses its upper lanes.
        let wide = machine.cpu.bytes > 16;
        let callee = || {
            let mut regs = abi.volatile | MASKS | abi.volatile_vectors;
            if wide {
                regs.union_from(abi.saved & vectors(0, 16));
            }
            regs
        };
        match self {
            Self::Call { clobbers, .. } => *clobbers,
            Self::RustCall(..) => {
                let mut regs = callee();
                regs.remove(AX);
                regs
            }
            Self::Lanes {
                action, input, dst, ..
            } => {
                // The inline expansions use only these scratch registers. A real
                // math call may also overwrite the upper halves of Windows YMM/ZMM6..15.
                let mut set = if lanes::calls_library(*action, *input) {
                    callee()
                } else {
                    gprs(&[0, 1, 2, 8, 9]) | vectors(0, 3)
                };
                set.remove(return_reg(*dst));
                set
            }
            Self::IDiv { remainder, .. } => PRegSet::empty().with(if *remainder { AX } else { DX }),
            _ => PRegSet::empty(),
        }
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
            machine.cpu.bytes / 8
        } else {
            1
        }
    }
}
