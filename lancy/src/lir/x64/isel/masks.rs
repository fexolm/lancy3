//! Canonical predicates can share a register class through SSA copies and edges.
use super::super::inst::{class, simd};
use crate::hir::transforms::ControlFlowGraph;
use crate::hir::{Binary, Function, Lane, Op, Simd, Terminator, Ternary, Type, Unary, Value};
use crate::lir::x64::Cpu;
use regalloc2::{RegClass, VReg};

pub(super) struct Masks(Vec<bool>);

impl Masks {
    pub fn analyze(f: &Function, cfg: &ControlFlowGraph, cpu: Cpu) -> Self {
        if !simd::evex(cpu) {
            return Self(Vec::new());
        }
        let mut groups = Groups::new(f.value_count());
        let mut literals = vec![None; f.value_count()];
        for &v in f.params(f.entry()) {
            groups.reject(v);
        }
        let lane = |v| f.value_type(v).lane().unwrap();
        for &b in cfg.reverse_postorder() {
            for i in f.instructions(b) {
                let result = f.result(i);
                let result_is_mask = result.is_some_and(|v| match f.value_type(v) {
                    Type::Vector(lane) => !lane.float() && simd::mask_lanes(cpu, lane),
                    _ => false,
                });
                let form = result
                    .filter(|_| result_is_mask)
                    .and_then(|v| mask_form(f.op(i), lane(v), lane, |v| literals[v.index()]));
                if let Some(form) = form {
                    let v = result.unwrap();
                    match form {
                        MaskForm::Copy(a) | MaskForm::Not(a) => groups.join(v, a),
                        MaskForm::Logic(_, a, b) => {
                            groups.join(v, a);
                            groups.join(v, b);
                        }
                        MaskForm::Compare(_, _, a, b) => {
                            groups.reject(a);
                            groups.reject(b);
                        }
                        MaskForm::Constant(_) => {}
                    }
                    continue;
                }
                match *f.op(i) {
                    Op::Const(n) => {
                        literals[result.unwrap().index()] = Some(u128::from(n as u32));
                        groups.reject(result.unwrap());
                    }
                    Op::Literal(_, bits) => {
                        literals[result.unwrap().index()] = Some(bits);
                        groups.reject(result.unwrap());
                    }
                    Op::Simd(Simd::AnyTrue(_) | Simd::AllTrue(_)) => {}
                    // A predicate producer, like a comparison; its count is a scalar.
                    Op::Simd(Simd::FirstLanes(..)) if result_is_mask => {}
                    Op::Simd(Simd::MaskCount(m)) if native_count(cpu, lane(m)) => {}
                    Op::Simd(Simd::CompressStore { mask, value, .. }) => {
                        groups.reject(value);
                        // The mask stays a predicate for the native store and
                        // for the fallback's compress and count.
                        let lane = lane(value);
                        let kept = native_compress_store(cpu, lane)
                            || simd::compact_lanes(cpu, lane) && native_count(cpu, lane);
                        if !kept {
                            groups.reject(mask);
                        }
                    }
                    Op::Ternary(Ternary::Select, _, yes, no)
                        if simd::mask_lanes(cpu, lane(yes)) =>
                    {
                        groups.reject(yes);
                        groups.reject(no);
                        groups.reject(result.unwrap());
                    }
                    Op::Simd(Simd::Compress(v, _) | Simd::Expand(v, _))
                        if simd::compact_lanes(cpu, lane(v)) =>
                    {
                        groups.reject(v);
                        groups.reject(result.unwrap());
                    }
                    Op::Simd(
                        Simd::MaskedLoad {
                            address,
                            passthrough: value,
                            ..
                        }
                        | Simd::MaskedStore { address, value, .. },
                    ) if simd::evex_masked_memory(
                        cpu,
                        lane(value).bytes(),
                        address.indices.is_some(),
                    ) =>
                    {
                        groups.reject(value);
                        if let Some(indices) = address.indices {
                            groups.reject(indices);
                        }
                        if let Some(v) = result {
                            groups.reject(v);
                        }
                    }
                    _ => {
                        if let Some(v) = result {
                            groups.reject(v);
                        }
                        f.visit_operands(i, |v| groups.reject(v));
                    }
                }
            }
            let term = *f.terminator(b);
            if let Terminator::Return(v) = term {
                groups.reject(v);
            }
            for edge in term.edges() {
                let (to, args) = f.edge(edge);
                for (&param, &arg) in f.params(to).iter().zip(args) {
                    groups.join(param, arg);
                }
            }
        }
        Self(groups.finish())
    }

    pub fn contains(&self, v: Value) -> bool {
        self.0.get(v.index()).copied().unwrap_or(false)
    }

    pub fn reg(&self, f: &Function, v: Value) -> VReg {
        let ty = f.value_type(v);
        let class = if matches!(ty, Type::Vector(_)) && self.contains(v) {
            RegClass::Float
        } else {
            class(ty)
        };
        VReg::new(v.index(), class)
    }
}

/// How a predicate result is computed in a k-register. The analysis keeps a
/// value in k-registers exactly when the selection rules have a form for it.
#[derive(Clone, Copy)]
pub(super) enum MaskForm {
    /// Same predicate in another value of equal lane size.
    Copy(Value),
    Not(Value),
    Logic(Binary, Value, Value),
    /// A comparison with its `vpcmp` predicate.
    Compare(Lane, i32, Value, Value),
    /// A splat of all-zero or all-one lanes.
    Constant(bool),
}

/// The k-register form of `op` producing a predicate with `result` lanes,
/// given each value's lane and the bits of its constant definition.
pub(super) fn mask_form(
    op: &Op,
    result: Lane,
    lane: impl Fn(Value) -> Lane,
    constant: impl Fn(Value) -> Option<u128>,
) -> Option<MaskForm> {
    Some(match *op {
        Op::Copy(v) if !lane(v).float() && lane(v).bytes() == result.bytes() => MaskForm::Copy(v),
        Op::Unary(Unary::Not, v) => MaskForm::Not(v),
        Op::Bin(kind @ (Binary::And | Binary::Or | Binary::Xor), a, b) => {
            MaskForm::Logic(kind, a, b)
        }
        Op::Bin(kind, a, b) if !matches!(lane(a), Lane::F16 | Lane::BF16) => {
            MaskForm::Compare(lane(a), simd::comparison_predicate(lane(a), kind)?, a, b)
        }
        Op::Simd(Simd::Splat(v)) => MaskForm::Constant(constant_predicate(lane(v), constant(v)?)?),
        _ => return None,
    })
}

pub(super) fn constant_predicate(lane: Lane, bits: u128) -> Option<bool> {
    if lane.float() || lane.bytes() > 8 {
        return None;
    }
    let mask = (1u128 << (8 * lane.bytes())) - 1;
    match bits & mask {
        0 => Some(false),
        n if n == mask => Some(true),
        _ => None,
    }
}

/// `kmovw` and `popcnt` count a predicate of at most 16 lanes.
pub(super) fn native_count(cpu: Cpu, lane: Lane) -> bool {
    cpu.isa.popcnt && cpu.bytes / lane.bytes() <= 16
}

/// `pdep` places the written prefix on the mask and `vpcompressd` stores it.
pub(super) fn native_compress_store(cpu: Cpu, lane: Lane) -> bool {
    simd::evex(cpu) && cpu.isa.bmi2 && lane.bytes() == 4
}

// Each component must have one representation. Rejecting any noncanonical
// producer or vector consumer also rejects copies, boolean ops and phi inputs.
struct Groups {
    parent: Vec<usize>,
    rejected: Vec<bool>,
}

impl Groups {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rejected: vec![false; len],
        }
    }

    fn find(&mut self, mut v: usize) -> usize {
        while self.parent[v] != v {
            self.parent[v] = self.parent[self.parent[v]];
            v = self.parent[v];
        }
        v
    }

    fn join(&mut self, a: Value, b: Value) {
        let a = self.find(a.index());
        let b = self.find(b.index());
        self.parent[b] = a;
        self.rejected[a] |= self.rejected[b];
    }

    fn reject(&mut self, v: Value) {
        let group = self.find(v.index());
        self.rejected[group] = true;
    }

    fn finish(mut self) -> Vec<bool> {
        (0..self.parent.len())
            .map(|v| {
                let group = self.find(v);
                !self.rejected[group]
            })
            .collect()
    }
}
