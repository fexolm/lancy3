//! Dispatch HIR operations to AArch64 instruction selection rules.

mod control;
mod conversions;
mod half;
mod memory;
mod rearrange;
mod reduce;
mod scalar_binary;
mod ternary;
mod unary;
mod values;
mod vector_binary;
mod wide;

use crate::{
    hir::operation::operation,
    hir::{Binary, Function, Lane, Op, Simd, Ternary, Type, Unary, Value},
    lir::{
        aarch64::backend::{
            inst::{Inst, class},
            remainder, rotate, special, sve,
        },
        isel::{Context, Root},
    },
    module::FunctionId,
};
use regalloc2::{PRegSet, VReg};
use splice_target::Simd as Config;

pub(super) fn reg(f: &Function, v: Value) -> VReg {
    VReg::new(v.index(), class(f.value_type(v)))
}

pub(super) fn select(
    ctx: &mut Context,
    f: &Function,
    root: Root,
    out: &mut Vec<Inst>,
    config: Config,
    entry: bool,
    callee: impl Fn(&Function) -> (FunctionId, PRegSet),
) {
    let mut rules = Rules {
        ctx,
        f,
        out,
        config,
        result: None,
    };
    let inst = match root {
        Root::Inst(i) => {
            rules.result = f.result(i);
            rules.instruction(*f.op(i), callee)
        }
        Root::Term(term) => rules.terminator(term, entry),
    };
    out.push(inst);
}

struct Rules<'a> {
    ctx: &'a mut Context,
    f: &'a Function,
    out: &'a mut Vec<Inst>,
    config: Config,
    result: Option<Value>,
}

impl Rules<'_> {
    fn ty(&self, v: Value) -> Type {
        self.f.value_type(v)
    }

    fn result(&self) -> Value {
        self.result.expect("value-producing operation")
    }

    fn dst(&self) -> VReg {
        reg(self.f, self.result())
    }

    fn input(&mut self, v: Value) -> VReg {
        // Only the selected form should mark its register operands live.
        self.ctx.require(v);
        reg(self.f, v)
    }

    fn float_remainder(&mut self, op: crate::hir::Binary, a: Value, b: Value) -> Option<Inst> {
        let ty = self.ty(a);
        let lane = ty.lane().filter(|lane| lane.float())?;
        if op != crate::hir::Binary::Rem {
            return None;
        }
        let packed = matches!(ty, Type::Vector(_));
        let form = remainder::Form {
            lane,
            packed,
            scalable: packed && sve(self.config.profile),
        };
        Some(Inst::FloatRemainder(
            form,
            self.dst(),
            self.input(a),
            self.input(b),
        ))
    }

    fn float_special(
        &mut self,
        action: special::Action,
        a: Value,
        b: Option<Value>,
    ) -> Option<Inst> {
        let ty = self.ty(a);
        let lane = ty.lane()?;
        if !lane.float() || (lane.bytes() == 2 && matches!(action, special::Action::MinMax(_))) {
            return None;
        }
        let packed = matches!(ty, Type::Vector(_));
        let form = special::Form {
            lane,
            packed,
            scalable: packed && sve(self.config.profile),
            action,
        };
        Some(Inst::FloatSpecial(
            form,
            self.dst(),
            self.input(a),
            b.map(|b| self.input(b)),
        ))
    }

    fn float_binary_special(&mut self, op: Binary, a: Value, b: Value) -> Option<Inst> {
        let action = match op {
            Binary::Min => special::Action::MinMax(true),
            Binary::Max => special::Action::MinMax(false),
            Binary::CopySign => special::Action::CopySign,
            _ => return None,
        };
        self.float_special(action, a, Some(b))
    }

    fn float_unary_special(&mut self, op: Unary, v: Value) -> Option<Inst> {
        let action = match op {
            Unary::IsNan
            | Unary::IsInfinite
            | Unary::IsFinite
            | Unary::IsNormal
            | Unary::IsSubnormal
            | Unary::SignBit => special::Action::Classify(op),
            Unary::Signum => special::Action::Signum,
            Unary::NextUp => special::Action::Next(true),
            Unary::NextDown => special::Action::Next(false),
            _ => return None,
        };
        self.float_special(action, v, None)
    }

    fn rotate(&mut self, op: Binary, a: Value, b: Value) -> Option<Inst> {
        if !matches!(op, Binary::RotateLeft | Binary::RotateRight) {
            return None;
        }
        let ty = self.ty(a);
        let lane = ty.lane().filter(|lane| scalar_int(*lane))?;
        let packed = matches!(ty, Type::Vector(_));
        let form = rotate::Form {
            size: lane.bytes(),
            packed,
            scalable: packed && sve(self.config.profile),
            right: op == Binary::RotateRight,
        };
        Some(Inst::Rotate(form, self.dst(), self.input(a), self.input(b)))
    }

    fn instruction(&mut self, op: Op, callee: impl Fn(&Function) -> (FunctionId, PRegSet)) -> Inst {
        let native = match op {
            Op::Const(n) => return Inst::Literal(Type::I32, n as u32 as u128, self.dst()),
            Op::Literal(lane, bits) => return Inst::Literal(lane.scalar(), bits, self.dst()),
            Op::Simd(Simd::LaneCount(lane)) => {
                return Inst::Literal(
                    Type::I32,
                    (self.config.bytes / lane.bytes()) as u128,
                    self.dst(),
                );
            }
            Op::Copy(v) | Op::Simd(Simd::Bitcast(_, v)) => {
                return Inst::Copy(self.dst(), self.input(v));
            }
            Op::Bitcast(to, v) => return self.bitcast(to, v),
            Op::BoundsCheck(index, length) => {
                return Inst::Check(self.input(index), self.input(length));
            }
            Op::PtrOffset(base, index, scale) => {
                return Inst::Offset(scale, self.dst(), self.input(base), self.input(index));
            }
            Op::Load(ty, ptr) => return Inst::Load(ty, self.dst(), self.input(ptr)),
            Op::Simd(Simd::Load(lane, ptr)) => {
                return Inst::Load(Type::Vector(lane), self.dst(), self.input(ptr));
            }
            Op::Store(ptr, v) | Op::Simd(Simd::Store(ptr, v)) => {
                return Inst::Store(self.ty(v), self.input(ptr), self.input(v));
            }
            Op::DebugStore(offset, value) => {
                return Inst::DebugStore(self.ty(value), offset, self.input(value));
            }
            Op::DebugPoint(id) => return Inst::DebugPoint(id),
            Op::Simd(Simd::MaskedLoad {
                address,
                mask,
                passthrough,
            }) => {
                return self.masked_memory(address, mask, passthrough);
            }
            Op::Simd(Simd::MaskedStore {
                address,
                mask,
                value,
            }) => {
                return self.masked_memory(address, mask, value);
            }
            Op::Call(site) => return self.call(site, callee),
            Op::Bin(kind, a, b) => self.scalar_binary(kind, a, b),
            Op::Simd(Simd::Bin(kind, a, b)) => self.vector_binary(kind, a, b),
            Op::Unary(kind, v) | Op::Simd(Simd::Unary(kind, v)) => self.unary(kind, v),
            Op::Ternary(Ternary::MulAdd, a, b, c) => self.scalar_fma(a, b, c),
            Op::Ternary(Ternary::Select, mask, a, b) => return self.scalar_select(mask, a, b),
            Op::Simd(Simd::Ternary(Ternary::MulAdd, a, b, c)) => self.vector_fma(a, b, c),
            Op::Simd(Simd::Ternary(kind @ (Ternary::Select | Ternary::BitSelect), mask, a, b)) => {
                return self.vector_select(kind, mask, a, b);
            }
            Op::Cast(to, v) => self.scalar_cast(to, v),
            Op::Simd(Simd::Cast(to, v)) => self.vector_cast(to, v),
            Op::Simd(Simd::Widen { to, high, value }) => return self.widen(to, high, value),
            Op::Simd(Simd::Narrow(to, a, b)) => return self.narrow(to, a, b),
            Op::Simd(Simd::Iota(lane)) => return self.iota(lane),
            Op::Simd(Simd::Splat(v)) => return self.splat(v),
            Op::Simd(Simd::ExtractLane(src, index)) => return self.lane_access(src, index, None),
            Op::Simd(Simd::InsertLane(src, index, value)) => {
                return self.lane_access(src, index, Some(value));
            }
            Op::Simd(Simd::Compress(value, mask)) => return self.compress(value, mask),
            Op::Simd(Simd::Expand(value, mask)) => return self.expand(value, mask),
            Op::Simd(Simd::Shuffle(a, b, indices)) => return self.shuffle(a, b, indices),
            Op::Simd(Simd::Reduce(kind, v)) => return self.reduce(kind, v),
            Op::Simd(Simd::AnyTrue(v)) => return self.mask_test(v, false),
            Op::Simd(Simd::AllTrue(v)) => return self.mask_test(v, true),
            Op::Ternary(Ternary::BitSelect, mask, a, b) => {
                return self.scalar_bit_select(mask, a, b);
            }
        };
        native.unwrap_or_else(|| self.libcall(op))
    }

    fn libcall(&mut self, op: Op) -> Inst {
        let (action, input, output, args) =
            operation(op, |v| self.ty(v)).expect("verified numeric operation");
        for (n, arg) in args.into_iter().enumerate() {
            if let Some(arg) = arg {
                let inst = Inst::LibArg(self.ty(arg), n, self.input(arg));
                self.out.push(inst);
            }
        }
        let sizes = args.map(|arg| arg.map_or(0, |v| self.ty(v).bytes(self.config.bytes)));
        self.out.push(Inst::Libcall(action, input, output, sizes));
        Inst::LibResult(self.ty(self.result()), self.dst())
    }
}

fn scalar_int(lane: Lane) -> bool {
    !lane.float() && lane.bytes() <= 8
}

fn fp(ty: Type) -> bool {
    matches!(ty, Type::F32 | Type::F64)
}
