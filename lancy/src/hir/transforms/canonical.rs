//! The fallback lowering of the canonical predicate operations
//! (`FirstLanes`, `MaskCount`, `CompressStore`). A backend selects one of
//! them natively where its machine has the instruction; every other use
//! becomes the primitive operations that define it, so both backends share
//! one fallback and no rule has to recognize the primitive form again.
use crate::hir::{
    Binary, Block, Function, Inst, Lane, Op, Simd, Type, Unary, Value, VectorAddress,
};
use crate::support::slotmap::DenseMap;

/// A copy of `f` with every canonical operation for which `native` is false
/// replaced by its definition, or `None` when `native` takes them all.
/// `native` also decides the canonical operations a definition introduces.
pub(crate) fn expand(f: &Function, native: impl Fn(&Function, Inst) -> bool) -> Option<Function> {
    let pending = |f: &Function, inst| is_canonical(*f.op(inst)) && !native(f, inst);
    let blocks: Vec<_> = f.blocks().collect();
    if !blocks
        .iter()
        .any(|&b| f.instructions(b).any(|inst| pending(f, inst)))
    {
        return None;
    }
    let mut defs = DenseMap::filled(f.value_count(), None);
    for &b in &blocks {
        for inst in f.instructions(b) {
            if let Some(v) = f.result(inst) {
                defs[v] = Some(*f.op(inst));
            }
        }
    }
    let mut f = f.clone();
    for block in blocks {
        let mut prev = None;
        let instructions: Vec<_> = f.instructions(block).collect();
        for inst in instructions {
            if pending(&f, inst) {
                Expansion {
                    f: &mut f,
                    defs: &defs,
                    block,
                    prev,
                    native: &native,
                }
                .lower(inst);
            }
            prev = Some(inst);
        }
    }
    Some(f)
}

fn is_canonical(op: Op) -> bool {
    matches!(
        op,
        Op::Simd(Simd::FirstLanes(..) | Simd::MaskCount(_) | Simd::CompressStore { .. })
    )
}

struct Expansion<'a, N> {
    f: &'a mut Function,
    /// The defining operations of the values of the original function.
    defs: &'a DenseMap<Value, Option<Op>>,
    block: Block,
    /// The instruction the next emitted operation follows.
    prev: Option<Inst>,
    native: &'a N,
}

impl<N: Fn(&Function, Inst) -> bool> Expansion<'_, N> {
    /// Insert the definition of `inst` before it and give `inst` the last
    /// operation of the definition, which keeps its result.
    fn lower(&mut self, inst: Inst) {
        let op = match *self.f.op(inst) {
            Op::Simd(Simd::FirstLanes(lane, count)) => self.first_lanes(lane, count),
            Op::Simd(Simd::MaskCount(mask)) => self.mask_count(mask),
            Op::Simd(Simd::CompressStore {
                base,
                mask,
                value,
                limit,
            }) => {
                let packed = self.value(Op::Simd(Simd::Compress(value, mask)));
                let count = self.value(Op::Simd(Simd::MaskCount(mask)));
                let count = self.value(Op::Bin(Binary::Min, count, limit));
                let lane = self.f.value_type(mask).lane().unwrap();
                let written = self.value(Op::Simd(Simd::FirstLanes(lane, count)));
                Op::Simd(Simd::MaskedStore {
                    address: VectorAddress {
                        base,
                        indices: None,
                    },
                    mask: written,
                    value: packed,
                })
            }
            op => unreachable!("{op:?} is not a canonical operation"),
        };
        self.f.replace_op_preserving_type(inst, op);
    }

    /// `iota < splat(count)` in signed lanes of the same width. The count
    /// is clamped first, so narrow lanes see it exactly.
    fn first_lanes(&mut self, lane: Lane, count: Value) -> Op {
        let signed = signed(lane);
        let count = if signed.bytes() < 4 {
            let lanes = self.value(Op::Simd(Simd::LaneCount(lane)));
            let zero = self.value(Op::Const(0));
            let count = self.value(Op::Bin(Binary::Max, count, zero));
            self.value(Op::Bin(Binary::Min, count, lanes))
        } else {
            count
        };
        let count = self.cast(signed.scalar(), count);
        let iota = self.value(Op::Simd(Simd::Iota(signed)));
        let limit = self.value(Op::Simd(Simd::Splat(count)));
        let below = Op::Bin(Binary::Lt, iota, limit);
        if signed == lane {
            below
        } else {
            Op::Bitcast(Type::Vector(lane), self.value(below))
        }
    }

    /// The sum of the negated predicate lanes. A lane that is not all ones
    /// or zero is first made a predicate from its sign.
    fn mask_count(&mut self, mask: Value) -> Op {
        let lane = self.f.value_type(mask).lane().unwrap();
        let signed = signed(lane);
        let mut lanes = self.reinterpret(signed, mask);
        if !is_predicate(self.f, self.defs, mask) {
            let zero = self.value(Op::literal(signed, 0));
            let zero = self.value(Op::Simd(Simd::Splat(zero)));
            lanes = self.value(Op::Bin(Binary::Lt, lanes, zero));
        }
        let flags = self.value(Op::Unary(Unary::Neg, lanes));
        let sum = Op::Simd(Simd::Reduce(Binary::Add, flags));
        if signed == Lane::I32 {
            sum
        } else {
            Op::Cast(Type::I32, self.value(sum))
        }
    }

    fn reinterpret(&mut self, lane: Lane, v: Value) -> Value {
        if self.f.value_type(v) == Type::Vector(lane) {
            v
        } else {
            self.value(Op::Bitcast(Type::Vector(lane), v))
        }
    }

    fn cast(&mut self, ty: Type, v: Value) -> Value {
        if self.f.value_type(v) == ty {
            v
        } else {
            self.value(Op::Cast(ty, v))
        }
    }

    /// Emit `op` before the instruction being lowered. A canonical operation
    /// the backend does not take is lowered in turn.
    fn value(&mut self, op: Op) -> Value {
        let inst = self
            .f
            .insert_after(self.block, self.prev, op)
            .expect("a fallback definition is well typed");
        if is_canonical(op) && !(self.native)(self.f, inst) {
            self.lower(inst);
        }
        self.prev = Some(inst);
        self.f.result(inst).unwrap()
    }
}

fn signed(lane: Lane) -> Lane {
    match lane.bytes() {
        1 => Lane::I8,
        2 => Lane::I16,
        4 => Lane::I32,
        8 => Lane::I64,
        _ => Lane::I128,
    }
}

/// Whether every lane of `v` is all ones or zero: a comparison, a
/// `FirstLanes`, or bitwise logic of such values.
fn is_predicate(f: &Function, defs: &DenseMap<Value, Option<Op>>, v: Value) -> bool {
    use Binary::*;
    let lane_bytes = |v| f.value_type(v).lane().map_or(0, Lane::bytes);
    let mut pending = vec![(v, 4)];
    while let Some((v, depth)) = pending.pop() {
        match defs.get(v).copied().flatten() {
            Some(Op::Simd(Simd::FirstLanes(..))) => {}
            Some(Op::Bin(Eq | Ne | Lt | Le | Gt | Ge | Ordered | Unordered, ..)) => {}
            Some(Op::Bin(And | Or | Xor, a, b)) if depth > 0 => {
                pending.extend([(a, depth - 1), (b, depth - 1)]);
            }
            // Splitting a lane of ones or zeros keeps it uniform.
            Some(Op::Unary(Unary::Not, a) | Op::Copy(a) | Op::Bitcast(_, a))
                if depth > 0 && lane_bytes(v) <= lane_bytes(a) =>
            {
                pending.push((a, depth - 1));
            }
            _ => return false,
        }
    }
    true
}
