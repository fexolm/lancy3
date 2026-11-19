use super::{A, C, Cond};
use crate::{
    hir::{Scale, Type},
    lir::TargetInst,
    module::FunctionId,
    numbers::Math,
};
use regalloc2::{MachineEnv, Operand, PReg, PRegSet, RegClass, VReg};
use std::sync::OnceLock;

pub(super) fn class(ty: Type) -> RegClass {
    if ty.bytes(16) <= 8 && !matches!(ty, Type::F32 | Type::F64) {
        RegClass::Int
    } else {
        RegClass::Vector
    }
}

fn registers(class: RegClass, numbers: impl IntoIterator<Item = usize>) -> PRegSet {
    let mut set = PRegSet::empty();
    for n in numbers {
        set.add(PReg::new(n, class));
    }
    set
}

fn volatile() -> [PRegSet; 3] {
    [
        registers(RegClass::Int, 0..=8),
        PRegSet::empty(),
        registers(RegClass::Vector, (0..=7).chain(17..=31)),
    ]
}

pub(super) fn call_clobbers() -> PRegSet {
    let [ints, _, vectors] = Inst::machine_env().preferred_regs_by_class;
    ints | vectors
}

pub(super) fn nonvolatile() -> PRegSet {
    Inst::machine_env().non_preferred_regs_by_class[RegClass::Int as usize]
}

fn return_reg(ty: Type) -> PReg {
    PReg::new(0, class(ty))
}

#[derive(Clone, Copy, Debug)]
pub(super) enum IntOp {
    Three(C),
    Multiply(C),
    Divide { signed: bool, remainder: bool },
    Compare(Cond),
    MinMax(Cond),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct IntBinary {
    pub op: IntOp,
    pub size: usize,
    pub signed_inputs: bool,
    pub wrap_shift: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Packed {
    Neon(C, A),
    Sve(C, A),
    SveDestructive(C, A, bool),
    NeonCompare {
        code: C,
        arrangement: A,
        invert: bool,
    },
    SveCompare {
        code: C,
        arrangement: A,
    },
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Masked {
    Lanes,
    Sve { code: C, arrangement: A },
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Select {
    Scalar(Type),
    Neon(Option<A>),
    Sve(Option<A>),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Inst {
    Arg(Type, usize, VReg),
    Literal(Type, u128, VReg),
    Copy(VReg, VReg),
    BitMove(Type, Type, VReg, VReg),
    Imm(C, u16, u8, VReg, VReg),
    Int(IntBinary, VReg, VReg, VReg),
    IntCast(Type, Type, VReg, VReg),
    FloatCast(C, Type, Type, VReg, VReg),
    PackedCast(C, A, VReg, VReg),
    Select(Select, VReg, VReg, VReg, VReg),
    Fma(C, A, VReg, [VReg; 3]),
    Compact(A, VReg, VReg, VReg),
    Sum(A, usize, VReg, VReg),
    Float(C, A, VReg, VReg, VReg),
    FloatCompare(Cond, A, VReg, VReg, VReg),
    Splat(C, A, Type, VReg, VReg),
    SveIota(A, VReg),
    MaskTest(bool, A, VReg, VReg),
    LaneAccess {
        lane: Type,
        src: VReg,
        index: VReg,
        insert: Option<VReg>,
        dst: VReg,
    },
    Packed(Packed, VReg, VReg, VReg),
    Check(VReg, VReg),
    Offset(Scale, VReg, VReg, VReg),
    Load(Type, VReg, VReg),
    Store(Type, VReg, VReg),
    Masked {
        kind: Masked,
        ty: Type,
        base: VReg,
        indices: Option<VReg>,
        mask: VReg,
        value: VReg,
        dst: Option<VReg>,
    },
    StoreArg(Type, usize, VReg),
    Call(FunctionId, PRegSet, Type, VReg),
    LibArg(Type, usize, VReg),
    Libcall(Math, [usize; 3]),
    LibResult(Type, VReg),
    Jump,
    Branch(VReg),
    Return(Type, VReg, bool),
}

impl TargetInst for Inst {
    fn operands(&self, out: &mut Vec<Operand>) {
        let input = Operand::reg_use;
        let output = Operand::reg_def;
        match *self {
            Self::Arg(_, _, d)
            | Self::Literal(_, _, d)
            | Self::LibResult(_, d)
            | Self::SveIota(_, d) => out.push(output(d)),
            Self::Call(_, _, ty, d) => {
                out.push(Operand::reg_fixed_def(d, return_reg(ty)));
            }
            Self::Copy(d, v)
            | Self::Imm(_, _, _, d, v)
            | Self::Load(_, d, v)
            | Self::IntCast(_, _, d, v)
            | Self::FloatCast(_, _, _, d, v)
            | Self::PackedCast(_, _, d, v)
            | Self::Sum(_, _, d, v)
            | Self::BitMove(_, _, d, v)
            | Self::MaskTest(_, _, d, v)
            | Self::Splat(_, _, _, d, v) => {
                out.extend([input(v), output(d)]);
            }
            Self::Int(_, d, a, b)
            | Self::Packed(_, d, a, b)
            | Self::Compact(_, d, a, b)
            | Self::Offset(_, d, a, b)
            | Self::Float(_, _, d, a, b)
            | Self::FloatCompare(_, _, d, a, b) => {
                out.extend([input(a), input(b), output(d)]);
            }
            Self::Check(a, b) | Self::Store(_, a, b) => out.extend([input(a), input(b)]),
            Self::Select(_, d, m, a, b) => out.extend([input(m), input(a), input(b), output(d)]),
            Self::Fma(_, _, d, args) => out.extend(args.map(input).into_iter().chain([output(d)])),
            Self::LaneAccess {
                src,
                index,
                insert,
                dst,
                ..
            } => {
                out.extend([input(src), input(index)]);
                out.extend(insert.map(input));
                out.push(output(dst));
            }
            Self::Masked {
                base,
                indices,
                mask,
                value,
                dst,
                ..
            } => {
                out.extend([input(base), input(mask), input(value)]);
                out.extend(indices.map(input));
                out.extend(dst.map(output));
            }
            Self::StoreArg(_, _, v) | Self::LibArg(_, _, v) | Self::Branch(v) => out.push(input(v)),
            Self::Return(ty, v, root) => {
                out.push(if !root || class(ty) == RegClass::Int {
                    Operand::reg_fixed_use(v, return_reg(ty))
                } else {
                    input(v)
                });
            }
            Self::Libcall(..) | Self::Jump => {}
        }
    }

    fn clobbers(&self) -> PRegSet {
        match *self {
            Self::Call(_, written, ty, _) => {
                let mut regs = call_clobbers() | written;
                regs.remove(return_reg(ty));
                regs
            }
            Self::Libcall(..) => call_clobbers(),
            _ => PRegSet::empty(),
        }
    }

    fn is_branch(&self) -> bool {
        matches!(self, Self::Jump | Self::Branch(_))
    }

    fn is_ret(&self) -> bool {
        matches!(self, Self::Return(..))
    }

    fn machine_env() -> &'static MachineEnv {
        static ENV: OnceLock<MachineEnv> = OnceLock::new();
        ENV.get_or_init(|| MachineEnv {
            preferred_regs_by_class: volatile(),
            non_preferred_regs_by_class: [
                registers(RegClass::Int, 23..=28),
                PRegSet::empty(),
                PRegSet::empty(),
            ],
            scratch_by_class: [
                Some(PReg::new(9, RegClass::Int)),
                None,
                Some(PReg::new(16, RegClass::Vector)),
            ],
            fixed_stack_slots: vec![],
        })
    }

    fn spillslot_size(class: RegClass) -> usize {
        if class == RegClass::Vector {
            splice_target::vector_bytes() / 8
        } else {
            1
        }
    }
}
