//! Dispatch HIR operations to x64 instruction selection rules.

mod binary;
mod control;
mod division;
mod experimental;
mod mask;
mod memory;
mod native;
mod numeric;
mod packed;
mod packed_float;
mod packed_integer;
mod packed_rearrange;
mod scalar;

pub(in crate::lir::x64) use packed::packed_half;

use super::super::inst::{Inst, SelectCond, class};
use super::masks::{Masks, native_compress_store, native_count};
use crate::lir::x64::Cpu;
use crate::{
    CompileError,
    hir::{Call, Function, Inst as HirInst, Lane, Op, Simd, Ternary, Type, Value},
    lir::{
        self,
        isel::{Context, Root},
    },
    module::FunctionId,
};
use regalloc2::{PRegSet, RegClass, VReg};

/// Whether `inst`, a canonical predicate operation, has a rule here:
/// its predicate operands and result live in mask registers, and the
/// machine has the instructions. `canonical::expand` defines the others.
pub(super) fn native_canonical(f: &Function, inst: HirInst, cpu: Cpu, masks: &Masks) -> bool {
    let lane = |v| f.value_type(v).lane().unwrap();
    match *f.op(inst) {
        Op::Simd(Simd::FirstLanes(lane, _)) => {
            cpu.isa.bmi2
                && cpu.bytes / lane.bytes() <= 16
                && f.result(inst).is_some_and(|v| masks.contains(v))
        }
        Op::Simd(Simd::MaskCount(mask)) => native_count(cpu, lane(mask)) && masks.contains(mask),
        Op::Simd(Simd::CompressStore { mask, value, .. }) => {
            native_compress_store(cpu, lane(value)) && masks.contains(mask)
        }
        _ => false,
    }
}

pub(super) fn select(
    ctx: &mut Context,
    f: &Function,
    body: &mut lir::Function<Inst>,
    out: &mut Vec<Inst>,
    target: (Cpu, &Masks),
    root: Root,
    callee: impl Fn(Call) -> (FunctionId, PRegSet),
) -> Result<(), CompileError> {
    let mut rules = Rules {
        ctx,
        f,
        body,
        out,
        cpu: target.0,
        masks: target.1,
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
    cpu: Cpu,
    masks: &'a Masks,
    result: Option<Value>,
}

impl Rules<'_> {
    fn instruction(
        &mut self,
        op: Op,
        callee: impl Fn(Call) -> (FunctionId, PRegSet),
    ) -> Result<Inst, CompileError> {
        if let Op::Simd(Simd::Splat(x)) = op
            && let Some((lane, bits)) = self.splat_literal(x)
        {
            return Ok(Inst::SplatLiteral(lane, bits, self.dst()));
        }
        Ok(match op {
            Op::Const(n) => Inst::Imm(self.dst(), n),
            Op::CallScratch { offset, bytes } => Inst::CallScratch(offset, bytes, self.dst()),
            Op::Literal(lane, bits) => Inst::Literal(lane, bits, self.dst()),
            Op::Simd(Simd::Iota(lane)) => Inst::Iota(lane, self.dst()),
            Op::Simd(Simd::LaneCount(lane)) => {
                Inst::Imm(self.dst(), (self.cpu.bytes / lane.bytes()) as i32)
            }
            Op::Bitcast(to, v) if class(to) != class(self.ty(v)) => {
                Inst::BitMove(self.ty(v), to, self.dst(), self.input(v))
            }
            Op::Copy(v) | Op::Bitcast(_, v) => Inst::Copy(self.dst(), self.input(v)),
            Op::Bin(kind, a, b) if !matches!(self.ty(a), Type::Vector(_)) => {
                self.binary(kind, a, b)?
            }
            Op::Ternary(Ternary::Select, cond, yes, no)
                if !matches!(self.ty(cond), Type::Vector(_))
                    && class(self.ty(yes)) == RegClass::Int =>
            {
                let cond = if let Some(Op::Bin(kind, a, b)) = self.ctx.single_use(cond)
                    && let Some((cc, int)) = self.integer_condition(kind, a)
                {
                    SelectCond::Compare(cc, int.bytes, self.input(a), self.rhs(b))
                } else {
                    SelectCond::Nonzero(self.input(cond))
                };
                Inst::Select {
                    wide: self.ty(yes).bytes(self.cpu.bytes) == 8,
                    dst: self.dst(),
                    no: self.input(no),
                    cond,
                    yes: self.input(yes),
                }
            }
            Op::Ternary(Ternary::Select, cond, yes, no)
                if !matches!(self.ty(cond), Type::Vector(_)) =>
            {
                Inst::SelectVector {
                    dst: self.dst(),
                    no: self.input(no),
                    cond: self.input(cond),
                    yes: self.input(yes),
                }
            }
            Op::BoundsCheck(index, length) => self.bounds_check(index, length),
            Op::TrapIf(cond, trap) => Inst::TrapIf(self.input(cond), trap),
            Op::Trap(trap) => Inst::Trap(trap),
            Op::PtrOffset(base, index, scale) => self.pointer_offset(base, index, scale)?,
            Op::Load(ty, ptr) => self.load(ty, ptr)?,
            Op::Store(ptr, value) => self.store(ptr, value)?,
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
            Op::Simd(Simd::FirstLanes(lane, count)) => self.first_lanes(lane, count)?,
            Op::Simd(Simd::MaskCount(mask)) => Inst::MaskCount(self.dst(), self.input(mask)),
            Op::Simd(Simd::CompressStore {
                base,
                mask,
                value,
                limit,
            }) => self.compress_store(base, mask, value, limit)?,
            Op::Call(site) => self.call(site, callee),
            Op::RustCall(function, args) => {
                Inst::RustCall(function, args.map(|arg| self.input(arg)), self.dst())
            }
            op => self.numeric(op)?,
        })
    }

    fn ty(&self, v: Value) -> Type {
        self.f.value_type(v)
    }

    fn dst(&self) -> VReg {
        self.masks
            .reg(self.f, self.result.expect("value-producing instruction"))
    }

    fn input(&mut self, v: Value) -> VReg {
        // Only the selected form should mark its register operands live.
        self.ctx.require(v);
        self.masks.reg(self.f, v)
    }

    /// The lane and bits of a vector splat of a constant scalar `x`. One load
    /// from the constant pool replaces moving the scalar into a register and
    /// broadcasting it, which costs two shuffle-port operations.
    fn splat_literal(&self, x: Value) -> Option<(Lane, u64)> {
        let result = self.result?;
        if self.masks.contains(result) {
            return None;
        }
        let (lane, bits) = match self.ctx.defining_op(x)? {
            Op::Const(n) => (Lane::I32, u64::from(n as u32)),
            Op::Literal(lane, bits) if lane.bytes() <= 8 => (lane, bits as u64),
            _ => return None,
        };
        (self.ty(result) == Type::Vector(lane)).then_some((lane, bits))
    }

    fn dword_constant(&self, value: Value) -> Option<u32> {
        match self.ctx.defining_op(value)? {
            Op::Const(n) => Some(n as u32),
            Op::Literal(Lane::I32 | Lane::U32, bits) => Some(bits as u32),
            Op::Simd(Simd::LaneCount(lane)) => Some((self.cpu.bytes / lane.bytes()) as u32),
            _ => None,
        }
    }
}
