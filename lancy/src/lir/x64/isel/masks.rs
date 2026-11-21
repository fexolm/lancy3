//! Canonical predicates can share a register class through SSA copies and edges.
use super::super::inst::{class, simd};
use crate::hir::transforms::ControlFlowGraph;
use crate::hir::{
    Binary, Block, Function, Lane, Op, Simd, Terminator, Ternary, Type, Unary, Value,
};
use regalloc2::{RegClass, VReg};
use splice_target::Features;

pub(super) struct Masks(Vec<bool>);

impl Masks {
    pub fn analyze(f: &Function, cfg: &ControlFlowGraph, cpu: Features) -> Self {
        if !simd::evex(cpu) {
            return Self(Vec::new());
        }
        let mut groups = Groups::new(f.value_count());
        let mut literals = vec![None; f.value_count()];
        let uses = if is_x86_feature_detected!("popcnt") {
            count_only_uses(f, cfg)
        } else {
            Vec::new()
        };
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
                match *f.op(i) {
                    Op::Const(n) => {
                        literals[result.unwrap().index()] = Some(u128::from(n as u32));
                        groups.reject(result.unwrap());
                    }
                    Op::Literal(_, bits) => {
                        literals[result.unwrap().index()] = Some(bits);
                        groups.reject(result.unwrap());
                    }
                    Op::Simd(Simd::Splat(v))
                        if result_is_mask
                            && literals[v.index()]
                                .and_then(|bits| constant_predicate(lane(v), bits))
                                .is_some() => {}
                    Op::Copy(v) | Op::Simd(Simd::Bitcast(_, v))
                        if result_is_mask
                            && !lane(v).float()
                            && lane(v).bytes() == lane(result.unwrap()).bytes() =>
                    {
                        groups.join(result.unwrap(), v);
                    }
                    Op::Simd(Simd::Bin(Binary::And | Binary::Or | Binary::Xor, a, b))
                        if result_is_mask =>
                    {
                        groups.join(result.unwrap(), a);
                        groups.join(result.unwrap(), b);
                    }
                    Op::Simd(Simd::Unary(Unary::Not, v)) if result_is_mask => {
                        groups.join(result.unwrap(), v);
                    }
                    Op::Simd(Simd::Unary(Unary::Neg, _))
                        if uses.get(result.unwrap().index()) == Some(&Use::Count(b)) =>
                    {
                        // Code selection consumes the predicate instead of the
                        // negated vector when the predicate stays in K registers.
                        groups.reject(result.unwrap());
                    }
                    Op::Simd(Simd::Bin(op, a, b))
                        if result_is_mask
                            && !matches!(lane(a), Lane::F16 | Lane::BF16)
                            && simd::comparison_predicate(lane(a), op).is_some() =>
                    {
                        groups.reject(a);
                        groups.reject(b);
                    }
                    Op::Simd(Simd::AnyTrue(_) | Simd::AllTrue(_)) => {}
                    Op::Simd(Simd::Ternary(Ternary::Select, _, yes, no))
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    None,
    Count(Block),
    Other,
}

fn count_only_uses(f: &Function, cfg: &ControlFlowGraph) -> Vec<Use> {
    let mut uses = vec![Use::None; f.value_count()];
    for &b in cfg.reverse_postorder() {
        for i in f.instructions(b) {
            if let Op::Simd(Simd::Reduce(Binary::Add, v)) = *f.op(i)
                && f.value_type(v) == Type::Vector(Lane::I32)
                && uses[v.index()] == Use::None
            {
                uses[v.index()] = Use::Count(b);
            } else {
                f.visit_operands(i, |v| uses[v.index()] = Use::Other);
            }
        }
        let term = *f.terminator(b);
        if let Terminator::Return(v) = term {
            uses[v.index()] = Use::Other;
        }
        for edge in term.edges() {
            for &v in f.edge(edge).1 {
                uses[v.index()] = Use::Other;
            }
        }
    }
    uses
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
