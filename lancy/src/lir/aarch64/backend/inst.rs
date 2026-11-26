use super::half::Form as HalfForm;
use super::reduce::Form as ReduceForm;
use super::unary::Form as UnaryForm;
use super::wide::MathForm as WideMath;
use super::width::{Narrow, Widen};
use super::{A, C, Cond, Config};
use crate::{
    hir::{Lane, RustFunction, Scale, Type, Unary},
    lir::{
        TargetInst,
        schedule::{Schedulable, Scheduling},
    },
    module::FunctionId,
    support::numeric::Action,
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
    let [ints, _, vectors] = machine_env().preferred_regs_by_class;
    ints | vectors
}

pub(super) fn nonvolatile() -> PRegSet {
    machine_env().non_preferred_regs_by_class[RegClass::Int as usize]
}

/// AAPCS64 is the only convention, whatever the vector width.
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
    NeonMinMax64(C, bool),
    NeonShift(C, A, bool),
    SveShift(C, A),
    FloatOrder(A, bool),
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
    Neon,
    Sve { code: C, arrangement: A },
    SveNarrow(C),
    SveWide(C),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Select {
    Scalar(Type),
    ScalarBits,
    Neon(Option<A>),
    Sve(Option<A>),
    Wide(bool),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Shuffle {
    Neon(A, usize),
    Sve(A),
    Wide(bool),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Expand {
    Neon(A, usize),
    Sve(A),
    Sve2(A),
    Wide(splice_target::Profile),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct VectorDivide {
    pub size: usize,
    pub signed: bool,
    pub remainder: bool,
    pub scalable: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Inst {
    Arg(Type, usize, VReg),
    Literal(Type, u128, VReg),
    Copy(VReg, VReg),
    BitMove(Type, Type, VReg, VReg),
    Imm(C, u16, u8, VReg, VReg),
    Int(IntBinary, VReg, VReg, VReg),
    WideMath(WideMath, VReg, VReg, VReg),
    WideUnary(Unary, bool, VReg, VReg),
    WideAdd {
        subtract: bool,
        scalable: bool,
        dst: VReg,
        a: VReg,
        b: VReg,
    },
    Unary(UnaryForm, VReg, VReg),
    Half(HalfForm, VReg, VReg, [Option<VReg>; 2]),
    HalfCast(Lane, Lane, VReg, VReg),
    IntCast(Type, Type, VReg, VReg),
    WideCast(Type, Type, VReg, VReg),
    FloatCast(C, Type, Type, VReg, VReg),
    PackedCast(C, A, VReg, VReg),
    Select(Select, VReg, VReg, VReg, VReg),
    Fma(C, A, VReg, [VReg; 3]),
    Sve2Ternary(C, VReg, [VReg; 3]),
    Shuffle(Shuffle, VReg, [VReg; 3]),
    Compact {
        size: usize,
        scalable: bool,
        dst: VReg,
        value: VReg,
        mask: VReg,
    },
    VectorDivide(VectorDivide, VReg, VReg, VReg),
    Rotate(super::rotate::Form, VReg, VReg, VReg),
    ScalarSaturate(C, usize, VReg, VReg, VReg),
    NeonMul64(VReg, VReg, VReg),
    FloatSpecial(super::special::Form, VReg, VReg, Option<VReg>),
    Expand(Expand, VReg, VReg, VReg),
    Widen(Widen, VReg, VReg),
    Narrow(Narrow, VReg, VReg, VReg),
    Reduce(ReduceForm, VReg, VReg),
    Float(C, A, VReg, VReg, VReg),
    FloatCompare(Cond, A, VReg, VReg, VReg),
    Splat(C, A, Type, VReg, VReg),
    SveIota(Lane, VReg),
    SveSplat128(VReg, VReg),
    MaskTest(bool, A, VReg, VReg),
    WideMaskTest(bool, bool, VReg, VReg),
    LaneAccess {
        lane: Type,
        src: VReg,
        index: VReg,
        insert: Option<VReg>,
        dst: VReg,
    },
    Packed(Packed, VReg, VReg, VReg),
    PackedShiftImm(C, A, u8, VReg, VReg),
    Check(VReg, VReg),
    TrapIf(VReg, crate::Trap),
    Trap(crate::Trap),
    Offset(Scale, VReg, VReg, VReg),
    Load(Type, VReg, VReg),
    Store(Type, VReg, VReg),
    DebugStore(Type, u32, VReg),
    DebugPoint(u32),
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
    Libcall(Action, Type, Type, [Option<Type>; 3]),
    RustCall(&'static RustFunction, [VReg; 4], VReg),
    /// Address of the frame's scratch memory at a byte offset, with the
    /// region's size in bytes.
    CallScratch(u32, u32, VReg),
    LibResult(Type, VReg),
    Jump,
    Branch(VReg),
    Return(Type, VReg, bool),
}

impl TargetInst for Inst {
    type Machine = Config;
    fn operands(&self, _: &Config, out: &mut Vec<Operand>) {
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
            | Self::Unary(_, d, v)
            | Self::IntCast(_, _, d, v)
            | Self::WideCast(_, _, d, v)
            | Self::HalfCast(_, _, d, v)
            | Self::SveSplat128(d, v)
            | Self::FloatCast(_, _, _, d, v)
            | Self::PackedCast(_, _, d, v)
            | Self::PackedShiftImm(_, _, _, d, v)
            | Self::BitMove(_, _, d, v)
            | Self::MaskTest(_, _, d, v)
            | Self::WideMaskTest(_, _, d, v)
            | Self::Splat(_, _, _, d, v) => {
                out.extend([input(v), output(d)]);
            }
            Self::Widen(Widen::Convert(..), d, v) => {
                out.extend([Operand::reg_use_at_end(v), Operand::reg_def_at_start(d)]);
            }
            Self::Widen(_, d, v) => out.extend([input(v), output(d)]),
            Self::Int(_, d, a, b)
            | Self::ScalarSaturate(_, _, d, a, b)
            | Self::WideAdd { dst: d, a, b, .. }
            | Self::Packed(_, d, a, b)
            | Self::Compact {
                size: 4..,
                scalable: true,
                dst: d,
                value: a,
                mask: b,
            }
            | Self::Expand(Expand::Sve2(_), d, a, b)
            | Self::Offset(_, d, a, b)
            | Self::Float(_, _, d, a, b)
            | Self::FloatCompare(_, _, d, a, b) => {
                out.extend([input(a), input(b), output(d)]);
            }
            Self::Check(a, b) | Self::Store(_, a, b) => out.extend([input(a), input(b)]),
            Self::TrapIf(cond, _) => out.push(input(cond)),
            Self::Select(_, d, m, a, b) => out.extend([input(m), input(a), input(b), output(d)]),
            Self::Fma(_, _, d, args) | Self::Shuffle(Shuffle::Sve(_), d, args) => {
                out.extend(args.map(input).into_iter().chain([output(d)]));
            }
            Self::Shuffle(Shuffle::Neon(..) | Shuffle::Wide(_), d, args) => {
                // The emitter uses the output and, for wider lanes, V17 as scratch.
                out.extend(args.map(Operand::reg_use_at_end));
                out.push(Operand::reg_def_at_start(d));
            }
            Self::Expand(_, d, data, mask) => {
                out.extend([input(data), input(mask), Operand::reg_def_at_start(d)]);
            }
            Self::Compact {
                dst: d,
                value: a,
                mask: b,
                ..
            }
            | Self::VectorDivide(_, d, a, b)
            | Self::Rotate(super::rotate::Form { packed: true, .. }, d, a, b)
            | Self::NeonMul64(d, a, b) => {
                out.extend([
                    Operand::reg_use_at_end(a),
                    Operand::reg_use_at_end(b),
                    Operand::reg_def_at_start(d),
                ]);
            }
            Self::Rotate(_, d, a, b) => out.extend([input(a), input(b), output(d)]),
            Self::FloatSpecial(form, d, a, b) => {
                let used = if form.packed {
                    Operand::reg_use_at_end
                } else {
                    input
                };
                let defined = if form.packed {
                    Operand::reg_def_at_start
                } else {
                    output
                };
                out.extend(
                    [Some(used(a)), b.map(used), Some(defined(d))]
                        .into_iter()
                        .flatten(),
                );
            }
            Self::Narrow(Narrow::Convert(_), d, a, b) => {
                out.extend([
                    Operand::reg_use_at_end(a),
                    Operand::reg_use_at_end(b),
                    Operand::reg_def_at_start(d),
                ]);
            }
            Self::Narrow(_, d, a, b) => out.extend([input(a), input(b), output(d)]),
            Self::Reduce(_, d, v) => {
                out.extend([input(v), Operand::reg_def_at_start(d)]);
            }
            Self::WideMath(form, d, a, b) => {
                let def = if form.scalable {
                    Operand::reg_def_at_start
                } else {
                    output
                };
                out.extend([input(a), input(b), def(d)]);
            }
            Self::WideUnary(_, scalable, d, v) => {
                let def = if scalable {
                    Operand::reg_def_at_start
                } else {
                    output
                };
                out.extend([input(v), def(d)]);
            }
            Self::Half(form, d, a, extra) => {
                if form.packed {
                    let live = Operand::reg_use_at_end;
                    out.push(live(a));
                    out.extend(extra.into_iter().flatten().map(live));
                    out.push(Operand::reg_def_at_start(d));
                } else {
                    out.push(input(a));
                    out.extend(extra.into_iter().flatten().map(input));
                    out.push(output(d));
                }
            }
            Self::Sve2Ternary(_, d, args) => {
                out.extend(args.map(input));
                out.push(Operand::reg_reuse_def(d, 0));
            }
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
                kind,
                base,
                indices,
                mask,
                value,
                dst,
                ..
            } => {
                out.push(input(base));
                if matches!(kind, Masked::Sve { .. }) {
                    out.extend([input(mask), input(value)]);
                    out.extend(indices.map(input));
                    out.extend(dst.map(output));
                } else {
                    let live = Operand::reg_use_at_end;
                    out.extend([live(mask), live(value)]);
                    out.extend(indices.map(live));
                    out.extend(dst.map(Operand::reg_def_at_start));
                }
            }
            Self::StoreArg(_, _, v)
            | Self::DebugStore(_, _, v)
            | Self::LibArg(_, _, v)
            | Self::Branch(v) => out.push(input(v)),
            Self::Return(ty, v, root) => {
                out.push(if !root || class(ty) == RegClass::Int {
                    Operand::reg_fixed_use(v, return_reg(ty))
                } else {
                    input(v)
                });
            }
            Self::RustCall(_, args, dst) => {
                out.extend(
                    args.into_iter().enumerate().map(|(reg, arg)| {
                        Operand::reg_fixed_use(arg, PReg::new(reg, RegClass::Int))
                    }),
                );
                out.push(Operand::reg_fixed_def(dst, return_reg(Type::I32)));
            }
            Self::CallScratch(_, _, dst) => out.push(Operand::reg_def(dst)),
            Self::Libcall(..) | Self::Jump | Self::DebugPoint(_) | Self::Trap(_) => {}
        }
    }

    fn clobbers(&self, _: &Config) -> PRegSet {
        match *self {
            Self::Call(_, written, ty, _) => {
                let mut regs = call_clobbers() | written;
                regs.remove(return_reg(ty));
                regs
            }
            Self::Libcall(..) => call_clobbers(),
            Self::RustCall(..) => {
                let mut regs = call_clobbers();
                regs.remove(return_reg(Type::I32));
                regs
            }
            Self::Half(form, ..) if matches!(form.action, super::half::Action::MulAdd) => {
                registers(RegClass::Vector, 17..=19)
            }
            Self::Half(..) => registers(RegClass::Vector, [17]),
            Self::ScalarSaturate(..)
            | Self::Rotate(super::rotate::Form { packed: true, .. }, ..) => {
                registers(RegClass::Vector, [17])
            }
            Self::FloatSpecial(
                super::special::Form {
                    action: super::special::Action::MinMax(_),
                    ..
                },
                ..,
            ) => registers(RegClass::Vector, [17]),
            Self::Reduce(ReduceForm::Ordered(_, Lane::F16 | Lane::BF16, _), ..) => {
                registers(RegClass::Vector, [17])
            }
            Self::VectorDivide(
                VectorDivide {
                    size: 1 | 2,
                    scalable: true,
                    ..
                },
                ..,
            ) => registers(RegClass::Vector, [17, 18]),
            Self::Compact {
                size: 1 | 2,
                scalable: true,
                ..
            } => registers(RegClass::Vector, [17]),
            Self::Masked {
                kind: Masked::SveNarrow(_),
                ..
            }
            | Self::Masked {
                kind: Masked::SveWide(_),
                indices: Some(_),
                ..
            } => registers(RegClass::Vector, [17]),
            Self::Shuffle(Shuffle::Neon(_, 2..) | Shuffle::Wide(true), ..)
            | Self::WideMaskTest(true, true, ..) => registers(RegClass::Vector, [17]),
            _ => PRegSet::empty(),
        }
    }

    fn is_branch(&self) -> bool {
        matches!(self, Self::Jump | Self::Branch(_))
    }

    fn is_ret(&self) -> bool {
        matches!(self, Self::Return(..))
    }

    fn machine_env(_: &Config) -> &'static MachineEnv {
        machine_env()
    }

    fn spillslot_size(config: &Config, class: RegClass) -> usize {
        if class == RegClass::Vector {
            config.bytes / 8
        } else {
            1
        }
    }
}

impl Schedulable for Inst {
    fn scheduling(&self) -> Scheduling {
        use Scheduling::{Barrier, Compute, Ordered};
        match *self {
            Self::Load(..) => Ordered(5),
            Self::Masked {
                dst: Some(_),
                indices,
                ..
            } => Ordered(if indices.is_some() { 9 } else { 5 }),
            Self::Check(..)
            | Self::TrapIf(..)
            | Self::Trap(_)
            | Self::Store(..)
            | Self::Masked { .. } => Ordered(1),
            Self::Int(
                IntBinary {
                    op: IntOp::Divide { .. },
                    ..
                },
                ..,
            )
            | Self::VectorDivide(..)
            | Self::LaneAccess { .. } => Ordered(12),
            Self::Int(
                IntBinary {
                    op: IntOp::Multiply(_),
                    ..
                },
                ..,
            )
            | Self::Float(..)
            | Self::Fma(..)
            | Self::Packed(..) => Compute(4),
            Self::Literal(..)
            | Self::Copy(..)
            | Self::BitMove(..)
            | Self::Imm(..)
            | Self::Int(..)
            | Self::Offset(..)
            | Self::IntCast(..)
            | Self::Select(..)
            | Self::Splat(..)
            | Self::SveIota(..)
            | Self::PackedShiftImm(..) => Compute(1),
            _ => Barrier,
        }
    }
}
