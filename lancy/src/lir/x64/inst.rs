//! Selected x64 instructions. R11 addresses the current workspace frame.
//! R10 is the allocator's move scratch and is available inside a pseudo-op.
use crate::hir::{Scale, Type};
use crate::{lir::TargetInst, module::FunctionId};
use regalloc2::{MachineEnv, Operand, PReg, PRegSet, RegClass, VReg};

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
pub(super) const NONVOLATILE: PRegSet = if cfg!(windows) {
    gprs(&[3, 5, 6, 7, 12, 13, 14, 15])
} else {
    gprs(&[3, 5, 12, 13, 14, 15])
};

static ENV: MachineEnv = MachineEnv {
    preferred_regs_by_class: [VOLATILE, PRegSet::empty(), PRegSet::empty()],
    non_preferred_regs_by_class: [NONVOLATILE, PRegSet::empty(), PRegSet::empty()],
    scratch_by_class: [Some(gpr(10)), None, None],
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

#[derive(Clone, Copy, Debug)]
pub(super) enum Inst {
    Argv(VReg),
    LoadArg {
        dst: VReg,
        argv: Option<VReg>,
        index: usize,
    },
    Imm(VReg, i32),
    Copy(VReg, VReg),
    Alu(Alu, VReg, VReg, VReg),
    Shift {
        left: bool,
        dst: VReg,
        value: VReg,
        count: VReg,
    },
    Compare(Cond, VReg, VReg, VReg),
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
    CheckBounds(VReg, VReg),
    Movsxd(VReg, VReg),
    Lea(VReg, VReg, VReg, Scale),
    Load(Type, VReg, VReg),
    Store(Type, VReg, VReg),
    StoreArg(usize, VReg),
    Call {
        target: FunctionId,
        dst: VReg,
        clobbers: u16,
    },
    Jump,
    Branch(VReg),
    Ret(VReg),
}

impl Inst {
    pub fn call(target: FunctionId, dst: VReg, mut written: PRegSet) -> Self {
        written.remove(AX); // The return value is a fixed definition.
        let clobbers = written
            .into_iter()
            .fold(0, |bits, r| bits | (1 << r.hw_enc()));
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
            Self::Imm(d, _) => out.push(O::reg_def(d)),
            Self::Copy(d, a) => out.extend([O::reg_use(a), O::reg_reuse_def(d, 0)]),
            Self::Alu(_, d, a, b) => {
                out.extend([O::reg_use(a), O::any_use(b), O::reg_reuse_def(d, 0)])
            }
            Self::Shift {
                dst, value, count, ..
            } => out.extend([
                O::reg_use(value),
                O::reg_fixed_use(count, CX),
                O::reg_reuse_def(dst, 0),
            ]),
            Self::Compare(_, d, a, b) => out.extend([O::reg_use(a), O::any_use(b), O::reg_def(d)]),
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
            Self::CheckBounds(i, n) => out.extend([O::reg_use(i), O::any_use(n)]),
            Self::Movsxd(d, a) | Self::Load(_, d, a) => out.extend([O::reg_use(a), O::reg_def(d)]),
            Self::Lea(d, base, i, _) => {
                out.extend([O::reg_use(base), O::reg_use(i), O::reg_def(d)])
            }
            Self::Store(_, a, v) => out.extend([O::reg_use(a), O::reg_use(v)]),
            Self::StoreArg(_, v) | Self::Branch(v) => out.push(O::reg_use(v)),
            Self::Call { dst, .. } => out.push(O::reg_fixed_def(dst, AX)),
            Self::Ret(v) => out.push(O::reg_fixed_use(v, AX)),
            Self::Jump => {}
        }
    }
    fn clobbers(&self) -> PRegSet {
        match self {
            Self::Call { clobbers, .. } => {
                let mut set = PRegSet::empty();
                let mut bits = *clobbers;
                while bits != 0 {
                    set.add(gpr(bits.trailing_zeros() as usize));
                    bits &= bits - 1;
                }
                set
            }
            Self::IDiv { remainder, .. } => PRegSet::empty().with(if *remainder { AX } else { DX }),
            _ => PRegSet::empty(),
        }
    }
    fn is_branch(&self) -> bool {
        matches!(self, Self::Jump | Self::Branch(_))
    }
    fn is_ret(&self) -> bool {
        matches!(self, Self::Ret(_))
    }
    fn machine_env() -> &'static MachineEnv {
        &ENV
    }
    fn spillslot_size(class: RegClass) -> usize {
        debug_assert_eq!(class, RegClass::Int);
        1 // one 64-bit workspace word for either i32 or ptr
    }
}
