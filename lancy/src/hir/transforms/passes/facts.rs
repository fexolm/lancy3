use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{
    Binary, Block, Effects, Function, Inst, Lane, MIN_VECTOR_BYTES, Op, Simd, TrapKind, Type, Value,
};
use crate::support::numeric::integer;
use crate::support::slotmap::DenseMap;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

#[derive(Clone, Copy)]
pub(super) struct Definition {
    pub(super) block: Block,
    pub(super) inst: Inst,
}

fn definitions(f: &Function) -> DenseMap<Value, Definition> {
    let mut defs = DenseMap::filled(
        f.value_count(),
        Definition {
            block: Block::INVALID,
            inst: Inst::INVALID,
        },
    );
    for block in f.blocks() {
        for &value in f.params(block) {
            defs[value].block = block;
        }
        for inst in f.instructions(block) {
            if let Some(value) = f.result(inst) {
                defs[value] = Definition { block, inst };
            }
        }
    }
    defs
}

fn root(roots: &mut DenseMap<Value, Value>, mut value: Value) -> Value {
    while roots[value] != value {
        roots[value] = roots[roots[value]];
        value = roots[value];
    }
    value
}

pub(super) fn flatten(roots: &mut DenseMap<Value, Value>) {
    for value in roots.keys() {
        roots[value] = root(roots, value);
    }
}

fn forwarded_params(
    f: &Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    defs: &DenseMap<Value, Definition>,
) -> DenseMap<Value, Value> {
    let mut roots = DenseMap::from_fn(f.value_count(), |v| v);
    if cfg.has_cycle() {
        forward_cyclic(f, cfg, dom, defs, &mut roots);
        return roots;
    }

    // In a DAG every predecessor already uses canonical values.
    let mut common = Vec::new();
    for &block in &cfg.reverse_postorder()[1..] {
        let params = f.params(block);
        if params.is_empty() {
            continue;
        }
        let mut incoming = cfg.incoming_edges(block).map(|(_, edge)| f.edge(edge).1);
        common.clear();
        common.extend(incoming.next().unwrap().iter().map(|&v| roots[v]));
        for args in incoming {
            for (value, &arg) in common.iter_mut().zip(args) {
                if *value != roots[arg] {
                    *value = Value::INVALID;
                }
            }
        }
        for (&param, &value) in params.iter().zip(&common) {
            if value != Value::INVALID && strictly_dominates(dom, defs[value].block, block) {
                roots[param] = value;
            }
        }
    }
    roots
}

fn strictly_dominates(dom: &DomTree, from: Block, to: Block) -> bool {
    from != to && dom.dominates(from, to)
}

const NONE: usize = usize::MAX;

struct Users {
    head: usize,
    tail: usize,
}

impl Default for Users {
    fn default() -> Self {
        Self {
            head: NONE,
            tail: NONE,
        }
    }
}

struct User {
    parameter: usize,
    next: usize,
}

struct Parameter {
    value: Value,
    inputs: Range<usize>,
    users: Users,
    queued: bool,
}

fn forward_cyclic(
    f: &Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    defs: &DenseMap<Value, Definition>,
    roots: &mut DenseMap<Value, Value>,
) {
    let mut positions = DenseMap::filled(f.value_count(), NONE);
    let mut params = Vec::new();
    let mut inputs = Vec::new();
    let mut incoming = Vec::new();
    for &block in &cfg.reverse_postorder()[1..] {
        if f.params(block).is_empty() {
            continue;
        }
        incoming.clear();
        incoming.extend(cfg.incoming_edges(block).map(|(_, edge)| f.edge(edge).1));
        for (column, &value) in f.params(block).iter().enumerate() {
            let start = inputs.len();
            inputs.extend(incoming.iter().map(|args| args[column]));
            positions[value] = params.len();
            params.push(Parameter {
                value,
                inputs: start..inputs.len(),
                users: Users::default(),
                queued: true,
            });
        }
    }

    let mut users = Vec::new();
    for p in 0..params.len() {
        for &arg in &inputs[params[p].inputs.clone()] {
            let source = positions[arg];
            if source == NONE || source == p {
                continue;
            }
            let list = &mut params[source].users;
            let node = users.len();
            users.push(User {
                parameter: p,
                next: list.head,
            });
            list.head = node;
            if list.tail == NONE {
                list.tail = node;
            }
        }
    }

    let mut pending: Vec<_> = (0..params.len()).rev().collect();
    while let Some(p) = pending.pop() {
        let param = &mut params[p];
        param.queued = false;
        let value = param.value;
        let mut args = inputs[param.inputs.clone()]
            .iter()
            .map(|&arg| root(roots, arg))
            .filter(|&arg| arg != value);
        // Self inputs hold the previous value. All remaining inputs must
        // agree on a definition available from outside this block.
        let Some(replacement) = args.next() else {
            continue;
        };
        if !strictly_dominates(dom, defs[replacement].block, defs[value].block)
            || !args.all(|arg| arg == replacement)
        {
            continue;
        }
        roots[value] = replacement;
        let list = std::mem::take(&mut param.users);
        if list.head == NONE {
            continue;
        }

        let mut node = list.head;
        while let Some(user) = users.get(node) {
            let dest = &mut params[user.parameter];
            if roots[dest.value] == dest.value && !std::mem::replace(&mut dest.queued, true) {
                pending.push(user.parameter);
            }
            node = user.next;
        }

        // Consumers of the old root also depend on any later replacement
        // of the new root. Link the whole list without copying its entries.
        if let Some(target) = params.get_mut(positions[replacement]) {
            users[list.tail].next = target.users.head;
            target.users.head = list.head;
            if target.users.tail == NONE {
                target.users.tail = list.tail;
            }
        }
    }
    for param in params {
        roots[param.value] = root(roots, param.value);
    }
}

pub(super) struct Facts {
    pub(super) defs: DenseMap<Value, Definition>,
    roots: DenseMap<Value, Value>,
    /// Loads the memory pass proved inside a declared slice: they cannot
    /// trap, so an unused one is dead (HIR-19).
    in_bounds_loads: HashSet<Value>,
}

/// Unrolled accesses can compute the same wrapping index in separate SSA
/// values. Canonicalize i32 additions before checking address equality. The
/// constant is part of the i32 index, never a displacement of the pointer.
fn offset_roots(
    f: &Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    defs: &DenseMap<Value, Definition>,
) -> DenseMap<Value, Value> {
    let mut roots = forwarded_params(f, cfg, dom, defs);
    let constant = |v: Value| {
        let inst = defs[v].inst;
        if inst != Inst::INVALID
            && let Op::Const(n) = *f.op(inst)
        {
            return Some(n);
        }
        None
    };
    let offset = |op, roots: &mut DenseMap<Value, Value>| {
        let Op::Bin(Binary::Add, a, b) = op else {
            return None;
        };
        if f.value_type(a) != Type::I32 {
            return None;
        }
        let (a, b) = (root(roots, a), root(roots, b));
        constant(b)
            .map(|n| (a, n))
            .or_else(|| constant(a).map(|n| (b, n)))
    };
    let mut offsets = HashMap::<(Value, i32), Value>::new();
    let mut changed = false;
    for &block in dom.preorder() {
        for inst in f.instructions(block) {
            let Some((base, displacement)) = offset(*f.op(inst), &mut roots) else {
                continue;
            };
            let previous = defs[base].inst;
            let key = if previous != Inst::INVALID
                && let Some((base, prefix)) = offset(*f.op(previous), &mut roots)
            {
                (base, prefix.wrapping_add(displacement))
            } else {
                (base, displacement)
            };
            let result = f.result(inst).unwrap();
            let first = offsets.entry(key).or_insert(result);
            if *first != result && dom.dominates(defs[*first].block, block) {
                roots[result] = *first;
                changed = true;
            } else {
                *first = result;
            }
        }
    }
    if changed {
        flatten(&mut roots);
    }
    roots
}

impl Facts {
    pub(super) fn new(f: &Function, cfg: &ControlFlowGraph, dom: &DomTree) -> Self {
        let defs = definitions(f);
        let roots = offset_roots(f, cfg, dom, &defs);
        Self {
            defs,
            roots,
            in_bounds_loads: HashSet::new(),
        }
    }

    /// Record that the load defining `value` stays inside a declared slice.
    pub(super) fn load_in_bounds(&mut self, value: Value) {
        self.in_bounds_loads.insert(value);
    }

    pub(super) fn root(&self, v: Value) -> Value {
        self.roots[v]
    }

    /// Record that `value` equals the earlier `root`, a root itself.
    pub(super) fn merge_value(&mut self, value: Value, root: Value) {
        debug_assert_eq!(self.roots[root], root);
        self.roots[value] = root;
    }

    /// Replace every use of the values `aliases` maps elsewhere by their
    /// images: each one an equal value whose definition dominates the
    /// replaced one's uses. Parameters that forwarded a replaced value
    /// forward its image.
    pub(super) fn merge_values(&mut self, f: &mut Function, aliases: &DenseMap<Value, Value>) {
        f.substitute_values(aliases);
        for root in self.roots.values_mut() {
            *root = aliases[*root];
        }
    }

    pub(super) fn forward_acyclic_params(&self, f: &mut Function, cfg: &ControlFlowGraph) {
        debug_assert!(!cfg.has_cycle());
        if cfg.reverse_postorder()[1..]
            .iter()
            .flat_map(|&block| f.params(block))
            .any(|&param| self.root(param) != param)
        {
            f.retain_block_params(|param| self.root(param) == param);
            f.substitute_values(&self.roots);
        }
    }

    pub(super) fn forward_loads(
        &mut self,
        f: &mut Function,
        cfg: &ControlFlowGraph,
        dom: &DomTree,
        aliases: &DenseMap<Value, Value>,
    ) {
        // Formerly different arguments may now agree at a merge. Revisit
        // parameter forwarding only when an unresolved parameter can change.
        let changed_inputs = cfg.reverse_postorder().iter().any(|&block| {
            f.terminator(block).edges().any(|edge| {
                let (target, args) = f.edge(edge);
                f.params(target).iter().zip(args).any(|(&param, &arg)| {
                    let root = self.root(arg);
                    self.root(param) == param && aliases[root] != root
                })
            })
        });
        f.substitute_values(aliases);
        if changed_inputs {
            self.roots = offset_roots(f, cfg, dom, &self.defs);
        } else {
            // Every replacement is an earlier canonical value.
            for root in self.roots.values_mut() {
                *root = aliases[*root];
            }
        }
    }

    pub(super) fn available(&self, v: Value, at: Block, dom: &DomTree) -> Option<Value> {
        let v = self.root(v);
        dom.dominates(self.defs[v].block, at).then_some(v)
    }

    pub(super) fn op(&self, f: &Function, v: Value) -> Option<Op> {
        let mut value = v;
        loop {
            let inst = self.defs[self.root(value)].inst;
            if inst == Inst::INVALID {
                return None;
            }
            match *f.op(inst) {
                Op::Copy(input) => value = input,
                op => return Some(op),
            }
        }
    }

    pub(super) fn constant(&self, f: &Function, v: Value) -> Option<i32> {
        match self.op(f, v) {
            Some(Op::Const(n)) => Some(n),
            _ => None,
        }
    }

    fn literal_bits(&self, f: &Function, value: Value) -> Option<(Lane, u128)> {
        let constant = match self.op(f, value) {
            Some(Op::Simd(Simd::Splat(scalar))) => self.op(f, scalar),
            op => op,
        };
        let (lane, bits) = match constant {
            Some(Op::Const(n)) => (Lane::I32, u128::from(n as u32)),
            Some(Op::Literal(lane, bits)) if !lane.float() => (lane, bits),
            _ => return None,
        };
        Some((lane, bits & integer::mask(lane)))
    }

    /// Whether evaluation can be skipped or moved without observing a trap
    /// or any memory or debug effect.
    #[inline]
    pub(super) fn is_total(&self, f: &Function, inst: Inst) -> bool {
        let op = *f.op(inst);
        let effects = op.effects(f);
        if effects.is_none() {
            return true;
        }
        if let Op::Load(..) = op {
            return f
                .result(inst)
                .is_some_and(|value| self.in_bounds_loads.contains(&value));
        }
        // Only a trap that the operands rule out can be skipped.
        if effects
            != (Effects {
                trap: effects.trap,
                ..Effects::default()
            })
        {
            return false;
        }
        match (effects.trap, op) {
            (Some(TrapKind::Division), Op::Bin(_, dividend, divisor)) => {
                let Some((lane, bits)) = self.literal_bits(f, divisor) else {
                    return false;
                };
                bits != 0
                    && (!lane.signed()
                        || bits != integer::mask(lane)
                        || self
                            .literal_bits(f, dividend)
                            .is_some_and(|(_, bits)| bits != 1 << (lane.bytes() * 8 - 1)))
            }
            (
                Some(TrapKind::LaneIndex),
                Op::Simd(Simd::ExtractLane(vector, index) | Simd::InsertLane(vector, index, _)),
            ) => {
                let lanes = MIN_VECTOR_BYTES / f.value_type(vector).lane().unwrap().bytes();
                self.constant(f, index)
                    .is_some_and(|i| (i as u32) < lanes as u32)
            }
            _ => false,
        }
    }

    pub(super) fn append(&mut self, f: &mut Function, block: Block, op: Op) -> Value {
        let inst = f.append_inst(block, op).expect("well-typed operation");
        let value = f.result(inst).expect("value-producing operation");
        debug_assert_eq!(self.defs.len(), value.index());
        self.defs.insert(Definition { block, inst });
        self.roots.insert(value);
        value
    }
}
