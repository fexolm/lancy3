//! Dispatch HIR operations to x64 instruction selection rules.

mod binary;
mod control;
mod division;
mod experimental;
mod memory;
mod native;
mod numeric;
mod packed;
mod packed_float;
mod packed_integer;
mod packed_rearrange;

pub(in crate::lir::x64) use packed::packed_half;

use super::super::inst::{Cond, Inst, SelectCond, class};
use crate::{
    CompileError,
    hir::{Function, Op, Simd, Ternary, Type, Value},
    lir::{
        self,
        isel::{Context, Root},
    },
    module::FunctionId,
};
use regalloc2::{PRegSet, VReg};
use splice_target::Features;

pub(super) fn reg(f: &Function, v: Value) -> VReg {
    VReg::new(v.index(), class(f.value_type(v)))
}

pub(super) fn select(
    ctx: &mut Context,
    f: &Function,
    body: &mut lir::Function<Inst>,
    out: &mut Vec<Inst>,
    cpu: Features,
    root: Root,
    callee: impl Fn(&Function) -> (FunctionId, PRegSet),
) -> Result<(), CompileError> {
    let mut rules = Rules {
        ctx,
        f,
        body,
        out,
        cpu,
        result: None,
    };
    let inst = match root {
        Root::Inst(i) => {
            rules.result = f.result(i);
            rules.instruction(*f.op(i), callee)?
        }
        Root::Term(term) => rules.terminator(term),
    };
    out.push(inst);
    Ok(())
}

struct Rules<'a> {
    ctx: &'a mut Context,
    f: &'a Function,
    body: &'a mut lir::Function<Inst>,
    out: &'a mut Vec<Inst>,
    cpu: Features,
    result: Option<Value>,
}

impl Rules<'_> {
    fn instruction(
        &mut self,
        op: Op,
        callee: impl Fn(&Function) -> (FunctionId, PRegSet),
    ) -> Result<Inst, CompileError> {
        Ok(match op {
            Op::Const(n) => Inst::Imm(self.dst(), n),
            Op::Literal(lane, bits) => Inst::Literal(lane, bits, self.dst()),
            Op::Simd(Simd::Iota(lane)) => Inst::Iota(lane, self.dst()),
            Op::Simd(Simd::LaneCount(lane)) => {
                Inst::Imm(self.dst(), (self.cpu.bytes / lane.bytes()) as i32)
            }
            Op::Bitcast(to, v) if class(to) != class(self.ty(v)) => {
                Inst::BitMove(self.ty(v), to, self.dst(), self.input(v))
            }
            Op::Copy(v) | Op::Bitcast(_, v) | Op::Simd(Simd::Bitcast(_, v)) => {
                Inst::Copy(self.dst(), self.input(v))
            }
            Op::Bin(kind, a, b) => self.binary(kind, a, b)?,
            Op::Ternary(Ternary::Select, cond, yes, no) if self.ty(yes) == Type::I32 => {
                let cond = if let Some(Op::Bin(kind, a, b)) = self.ctx.single_use(cond)
                    && self.ty(a) == Type::I32
                    && let Some(cc) = Cond::from_binary(kind)
                {
                    SelectCond::Compare(cc, self.input(a), self.rhs(b))
                } else {
                    SelectCond::Nonzero(self.input(cond))
                };
                Inst::SelectI32 {
                    dst: self.dst(),
                    no: self.input(no),
                    cond,
                    yes: self.input(yes),
                }
            }
            Op::BoundsCheck(index, length) => self.bounds_check(index, length),
            Op::PtrOffset(base, index, scale) => self.pointer_offset(base, index, scale)?,
            Op::Load(ty, ptr) => self.load(ty, ptr)?,
            Op::Simd(Simd::Load(lane, ptr)) => self.load(Type::Vector(lane), ptr)?,
            Op::Store(ptr, value) | Op::Simd(Simd::Store(ptr, value)) => self.store(ptr, value)?,
            Op::DebugStore(offset, value) => {
                Inst::DebugStore(self.ty(value), offset, self.input(value))
            }
            Op::DebugPoint(id) => Inst::DebugPoint(id),
            Op::Simd(Simd::MaskedLoad {
                address,
                mask,
                passthrough,
            }) => self.masked_memory(address, mask, passthrough)?,
            Op::Simd(Simd::MaskedStore {
                address,
                mask,
                value,
            }) => self.masked_memory(address, mask, value)?,
            Op::Call(site) => self.call(site, callee),
            op => self.numeric(op)?,
        })
    }

    fn ty(&self, v: Value) -> Type {
        self.f.value_type(v)
    }

    fn dst(&self) -> VReg {
        reg(self.f, self.result.expect("value-producing instruction"))
    }

    fn input(&mut self, v: Value) -> VReg {
        // Only the selected form should mark its register operands live.
        self.ctx.require(v);
        reg(self.f, v)
    }
}
