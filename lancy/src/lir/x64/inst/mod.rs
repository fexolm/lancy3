//! Selected x64 instructions. R11 addresses the current workspace frame.
//! R10 is the allocator's move scratch and is available inside a pseudo-op.
pub(super) mod lanes;
pub(super) mod simd;
use crate::hir::{Binary, Lane, Scale, Type};
use crate::numbers::Math;
use crate::{lir::TargetInst, module::FunctionId};
use regalloc2::{MachineEnv, Operand, PReg, PRegSet, RegClass, VReg};
use simd::Native;

const fn gpr(enc: usize) -> PReg {
    PReg::new(enc, RegClass::Int)
}
const AX: PReg = gpr(0);
const CX: PReg = gpr(1);
const DX: PReg = gpr(2);
const ARGV: PReg = if cfg!(windows) { CX } else { gpr(7) };

const fn gprs(encodings: &[usize]) -> PRegSet {
    let mut set = PRegSet::empty();
    let mut i = 0;
    while i < encodings.len() {
        set.add(gpr(encodings[i]));
        i += 1;
    }
    set
}

const VOLATILE: PRegSet = if cfg!(windows) {
    gprs(&[0, 1, 2, 8, 9])
} else {
    gprs(&[0, 1, 2, 6, 7, 8, 9])
};
const SAVED_GPRS: PRegSet = if cfg!(windows) {
    gprs(&[3, 5, 6, 7, 12, 13, 14, 15])
} else {
    gprs(&[3, 5, 12, 13, 14, 15])
};

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
const SCRATCH: usize = if cfg!(windows) { 5 } else { 15 };
const fn vectors(mut start: usize, end: usize) -> PRegSet {
    let mut regs = PRegSet::empty();
    while start < end {
        regs.add(PReg::new(start, RegClass::Vector));
        start += 1;
    }
    regs
}
const SAVED_SIMD: PRegSet = if cfg!(windows) {
    vectors(6, 16)
} else {
    PRegSet::empty()
};
pub(super) fn nonvolatile() -> PRegSet {
    SAVED_GPRS | SAVED_SIMD
}
const SIMD: PRegSet = {
    let mut set = PRegSet::empty();
    let mut i = 0;
    while i < SCRATCH {
        set.add(PReg::new(i, RegClass::Vector));
        i += 1;
    }
    set
};

static ENV: MachineEnv = MachineEnv {
    preferred_regs_by_class: [VOLATILE, PRegSet::empty(), SIMD],
    non_preferred_regs_by_class: [SAVED_GPRS, PRegSet::empty(), SAVED_SIMD],
    scratch_by_class: [
        Some(gpr(10)),
        None,
        Some(PReg::new(SCRATCH, RegClass::Vector)),
    ],
    fixed_stack_slots: Vec::new(),
};

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
    pub fn from_binary(op: Binary) -> Option<Self> {
        Some(match op {
            Binary::Eq => Self::Eq,
            Binary::Ne => Self::Ne,
            Binary::Lt => Self::Lt,
            Binary::Le => Self::Le,
            Binary::Gt => Self::Gt,
            Binary::Ge => Self::Ge,
            _ => return None,
        })
    }
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

impl Quotient {
    // The caller has handled ±1 and zero.
    pub fn for_divisor(divisor: i32) -> Self {
        let magnitude = divisor.unsigned_abs();
        let log = magnitude.ilog2();
        if magnitude.is_power_of_two() {
            return Self::PowerOfTwo(log as i32);
        }
        let d = u64::from(magnitude);
        let mut shift = 31 + log;
        let scale = 1u64 << shift;
        let mut multiplier = scale.div_ceil(d);
        // For n >= 0, the error in n*m/scale must be strictly below 1/d.
        // For n < 0 the result is rounded down, then incremented to
        // truncate toward zero. The reciprocal rounds up strictly for all
        // non-powers of two. One extra bit always meets the error bound.
        if multiplier * d - scale > 1u64 << log {
            shift += 1;
            multiplier = (1u64 << shift).div_ceil(d);
        }
        Self::Reciprocal {
            multiplier: multiplier as u32,
            shift: shift as i32,
        }
    }
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
pub(super) enum Inst {
    Argv(VReg),
    LoadArg {
        dst: VReg,
        argv: Option<VReg>,
        ty: Type,
        index: usize,
    },
    Native(Native, VReg, [Option<VReg>; 3], [Option<VReg>; 6]),
    Lanes(Math, VReg, [Option<VReg>; 3]),
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
    CheckMask(VReg),
    BranchAny(VReg),
    Movsxd(VReg, VReg),
    Lea(VReg, Address),
    Load(Type, VReg, Address),
    Masked {
        ty: Type,
        base: VReg,
        indices: Option<VReg>,
        temp: Option<VReg>,
        mask: VReg,
        value: VReg,
        dst: Option<VReg>,
    },
    Store(Type, Address, VReg),
    StoreArg(usize, Type, VReg),
    Call {
        target: FunctionId,
        dst: VReg,
        clobbers: PRegSet,
    },
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
    fn operands(&self, out: &mut Vec<Operand>) {
        use Operand as O;
        match *self {
            Self::Argv(v) => out.push(O::reg_fixed_def(v, ARGV)),
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
            Self::Lanes(_, d, args) => {
                out.extend(args.into_iter().flatten().map(O::reg_use));
                out.push(O::reg_fixed_def(d, return_reg(d)));
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
            Self::Movsxd(d, a) => out.extend([O::reg_use(a), O::reg_def(d)]),
            Self::Lea(d, addr) | Self::Load(_, d, addr) => {
                addr.operands(out);
                out.push(O::reg_def(d));
            }
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
                out.extend([O::reg_use(mask), O::reg_use(value)]);
                out.extend(dst.map(O::reg_def));
                out.extend(temp.map(O::reg_temp));
            }
            Self::Store(_, addr, value) => {
                addr.operands(out);
                out.push(O::reg_use(value));
            }
            Self::CheckMask(v) | Self::BranchAny(v) | Self::StoreArg(_, _, v) | Self::Branch(v) => {
                out.push(O::reg_use(v))
            }
            Self::Call { dst, .. } => out.push(O::reg_fixed_def(dst, return_reg(dst))),
            Self::Ret(v) => out.push(O::reg_fixed_use(v, return_reg(v))),
            Self::Jump => {}
        }
    }
    fn clobbers(&self) -> PRegSet {
        match self {
            Self::Call { clobbers, .. } => *clobbers,
            Self::Lanes(spec, dst, _) => {
                // The inline expansions use only these scratch registers. A real
                // math call may also overwrite the upper halves of Windows YMM/ZMM6..15.
                let mut set = if lanes::calls_library(*spec) {
                    let wide = super::Features::host().bytes > 16;
                    VOLATILE | SIMD | if wide { SAVED_SIMD } else { PRegSet::empty() }
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
            Self::Jump | Self::Branch(_) | Self::BranchAny(_) | Self::BrCmp(..)
        )
    }
    fn is_ret(&self) -> bool {
        matches!(self, Self::Ret(_))
    }
    fn machine_env() -> &'static MachineEnv {
        &ENV
    }
    fn spillslot_size(class: RegClass) -> usize {
        if class == RegClass::Vector {
            super::Features::host().bytes / 8
        } else {
            1
        }
    }
}
