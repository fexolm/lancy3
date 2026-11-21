use crate::hir::transforms::ControlFlowGraph;
use crate::hir::{
    Binary, Block, Function, Lane, Op, Simd, Terminator, Ternary, Type, Value,
    ssa::{Edge, Inst},
};
use crate::support::slotmap::DenseMap;

use crate::hir::integer::{binary as integer_binary, mask as lane_mask};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fact {
    Unknown,
    Constant(u128),
    Varying,
}

impl Fact {
    pub(super) fn bits(self) -> Option<u128> {
        match self {
            Self::Constant(bits) => Some(bits),
            _ => None,
        }
    }

    fn meet(self, incoming: Self) -> Self {
        match (self, incoming) {
            (Self::Unknown, x) | (x, Self::Unknown) => x,
            (a, b) if a == b => a,
            _ => Self::Varying,
        }
    }
}

pub(super) fn literal(lane: Lane, bits: u128) -> Op {
    if lane == Lane::I32 {
        Op::Const(bits as i32)
    } else {
        Op::Literal(lane, bits)
    }
}

pub(super) enum Reduced {
    Constant(u128),
    Copy(Value),
}

fn integer(ty: Type) -> bool {
    ty.lane().is_some_and(|lane| !lane.float())
}

/// Exact integer constants and total arithmetic identities for one scalar
/// or a uniform vector. Discarded operands that can trap remain effects.
pub(super) fn reduce(
    kind: Binary,
    ty: Type,
    lhs: Value,
    rhs: Value,
    a: Option<u128>,
    b: Option<u128>,
) -> Option<Reduced> {
    use Binary::*;
    use Reduced::*;
    let lane = ty.lane().filter(|lane| !lane.float())?;
    let filled = lane_mask(lane);
    if let (Some(a), Some(b)) = (a, b) {
        let bits = integer_binary(kind, lane, a, b).ok()?;
        return Some(Constant(
            bits & lane_mask(kind.result_type(ty).lane().unwrap()),
        ));
    }
    if lhs == rhs {
        match kind {
            Sub | Xor | Ne | Lt | Gt => return Some(Constant(0)),
            Eq | Le | Ge => return Some(Constant(lane_mask(kind.result_type(ty).lane().unwrap()))),
            And | Or | Min | Max => return Some(Copy(lhs)),
            _ => {}
        }
    }
    match (kind, a, b) {
        (Add | Or | Xor, Some(0), _) | (Mul, Some(1), _) => Some(Copy(rhs)),
        (Add | Sub | Or | Xor | SaturatingAdd | SaturatingSub, _, Some(0))
        | (Mul | Div, _, Some(1)) => Some(Copy(lhs)),
        (And, Some(n), _) if n == filled => Some(Copy(rhs)),
        (And, _, Some(n)) if n == filled => Some(Copy(lhs)),
        (Shl | Shr | RotateLeft | RotateRight, _, Some(n))
            if n & (lane.bytes() as u128 * 8 - 1) == 0 =>
        {
            Some(Copy(lhs))
        }
        (Mul | And, Some(0), _) | (Mul | And, _, Some(0)) | (Rem, _, Some(1)) => Some(Constant(0)),
        (Or, Some(n), _) | (Or, _, Some(n)) if n == filled => Some(Constant(filled)),
        _ => None,
    }
}

fn evaluate(f: &Function, facts: &DenseMap<Value, Fact>, op: Op) -> Fact {
    use Fact::*;
    match op {
        Op::Const(n) => Constant(u128::from(n as u32)),
        Op::Literal(lane, bits) => Constant(bits & lane_mask(lane)),
        Op::Copy(v) | Op::Simd(Simd::Splat(v)) => facts[v],
        Op::Bin(kind, a, b) | Op::Simd(Simd::Bin(kind, a, b)) if integer(f.value_type(a)) => {
            let (x, y) = (facts[a], facts[b]);
            match reduce(kind, f.value_type(a), a, b, x.bits(), y.bits()) {
                Some(Reduced::Constant(bits)) => Constant(bits),
                Some(Reduced::Copy(v)) => facts[v],
                _ if x == Unknown || y == Unknown => Unknown,
                _ => Varying,
            }
        }
        Op::Ternary(Ternary::Select, c, yes, no)
        | Op::Simd(Simd::Ternary(Ternary::Select, c, yes, no)) => {
            if yes == no {
                return facts[yes];
            }
            let vector = matches!(op, Op::Simd(_));
            match facts[c] {
                Constant(bits) => {
                    let cond = if vector {
                        bits >> (f.value_type(c).lane().unwrap().bytes() * 8 - 1) != 0
                    } else {
                        bits != 0
                    };
                    facts[if cond { yes } else { no }]
                }
                Varying => facts[yes].meet(facts[no]),
                Unknown => Unknown,
            }
        }
        _ => Varying,
    }
}

fn successors(term: Terminator, facts: &DenseMap<Value, Fact>) -> [Option<Edge>; 2] {
    match term {
        Terminator::Jump(edge) => [Some(edge), None],
        Terminator::Branch { cond, yes, no } => match facts[cond] {
            Fact::Constant(0) => [Some(no), None],
            Fact::Constant(_) => [Some(yes), None],
            Fact::Varying => [Some(no), Some(yes)],
            Fact::Unknown => [None, None],
        },
        Terminator::Return(_) => [None, None],
    }
}

#[derive(Clone, Copy)]
enum User {
    Inst(Block, Inst),
    Branch(Block),
    Argument(Edge, Value),
}

struct Use {
    user: User,
    next: usize,
}

struct Solver<'a> {
    f: &'a Function,
    facts: DenseMap<Value, Fact>,
    heads: DenseMap<Value, usize>,
    uses: Vec<Use>,
    reachable: DenseMap<Block, bool>,
    executable: DenseMap<Edge, bool>,
    blocks: Vec<Block>,
    changed: Vec<Value>,
}

impl Solver<'_> {
    fn add_use(&mut self, value: Value, user: User) {
        let next = std::mem::replace(&mut self.heads[value], self.uses.len());
        self.uses.push(Use { user, next });
    }

    fn build_uses(&mut self) {
        for block in self.f.blocks() {
            for inst in self.f.instructions(block) {
                let user = User::Inst(block, inst);
                match *self.f.op(inst) {
                    Op::Copy(v) | Op::Simd(Simd::Splat(v)) => self.add_use(v, user),
                    Op::Bin(_, a, b) | Op::Simd(Simd::Bin(_, a, b))
                        if integer(self.f.value_type(a)) =>
                    {
                        self.add_use(a, user);
                        if a != b {
                            self.add_use(b, user);
                        }
                    }
                    Op::Ternary(Ternary::Select, c, a, b)
                    | Op::Simd(Simd::Ternary(Ternary::Select, c, a, b)) => {
                        self.add_use(c, user);
                        self.add_use(a, user);
                        self.add_use(b, user);
                    }
                    _ => {}
                }
            }
            let term = *self.f.terminator(block);
            if let Terminator::Branch { cond, .. } = term {
                self.add_use(cond, User::Branch(block));
            }
            for edge in term.edges() {
                let (target, args) = self.f.edge(edge);
                for (&arg, &param) in args.iter().zip(self.f.params(target)) {
                    self.add_use(arg, User::Argument(edge, param));
                }
            }
        }
    }

    fn update(&mut self, value: Value, incoming: Fact) {
        let next = self.facts[value].meet(incoming);
        if next != self.facts[value] {
            self.facts[value] = next;
            if self.heads[value] != usize::MAX {
                self.changed.push(value);
            }
        }
    }

    fn enter(&mut self, block: Block) {
        if !std::mem::replace(&mut self.reachable[block], true) {
            self.blocks.push(block);
        }
    }

    fn enable(&mut self, edge: Edge) {
        if std::mem::replace(&mut self.executable[edge], true) {
            return;
        }
        let (target, args) = self.f.edge(edge);
        for (&arg, &param) in args.iter().zip(self.f.params(target)) {
            self.update(param, self.facts[arg]);
        }
        self.enter(target);
    }

    fn instruction(&mut self, inst: Inst) {
        if let Some(result) = self.f.result(inst)
            && self.facts[result] != Fact::Varying
        {
            self.update(result, evaluate(self.f, &self.facts, *self.f.op(inst)));
        }
    }

    fn terminator(&mut self, block: Block) {
        for edge in successors(*self.f.terminator(block), &self.facts)
            .into_iter()
            .flatten()
        {
            self.enable(edge);
        }
    }

    fn solve(mut self) -> DenseMap<Value, Fact> {
        self.build_uses();
        let entry = self.f.entry();
        self.enter(entry);
        loop {
            if let Some(block) = self.blocks.pop() {
                for inst in self.f.instructions(block) {
                    self.instruction(inst);
                }
                self.terminator(block);
            } else if let Some(value) = self.changed.pop() {
                let mut cursor = self.heads[value];
                while let Some(site) = self.uses.get(cursor) {
                    let user = site.user;
                    cursor = site.next;
                    match user {
                        User::Inst(block, inst) if self.reachable[block] => self.instruction(inst),
                        User::Branch(block) if self.reachable[block] => self.terminator(block),
                        User::Argument(edge, param) if self.executable[edge] => {
                            self.update(param, self.facts[value]);
                        }
                        _ => {}
                    }
                }
            } else {
                return self.facts;
            }
        }
    }
}

/// Sparse conditional constant propagation over SSA users and executable edges.
/// Each edge activates once and each scalar fact changes at most twice.
pub(super) fn analyze(f: &Function, cfg: &ControlFlowGraph) -> DenseMap<Value, Fact> {
    let mut facts = DenseMap::filled(f.value_count(), Fact::Unknown);
    for &v in f.params(f.entry()) {
        facts[v] = Fact::Varying;
    }
    if !cfg.has_cycle() {
        // RPO is a topological order for a DAG. All executable predecessors
        // have contributed before visiting a block, so one scan suffices.
        let mut reachable = DenseMap::filled(f.block_capacity(), false);
        reachable[f.entry()] = true;
        for &block in cfg.reverse_postorder() {
            if !reachable[block] {
                continue;
            }
            for inst in f.instructions(block) {
                if let Some(result) = f.result(inst) {
                    facts[result] = evaluate(f, &facts, *f.op(inst));
                }
            }
            for edge in successors(*f.terminator(block), &facts)
                .into_iter()
                .flatten()
            {
                let (target, args) = f.edge(edge);
                reachable[target] = true;
                for (&arg, &param) in args.iter().zip(f.params(target)) {
                    facts[param] = facts[param].meet(facts[arg]);
                }
            }
        }
        return facts;
    }
    Solver {
        f,
        facts,
        heads: DenseMap::filled(f.value_count(), usize::MAX),
        uses: Vec::new(),
        reachable: DenseMap::filled(f.block_capacity(), false),
        executable: DenseMap::filled(f.edge_count(), false),
        blocks: Vec::new(),
        changed: Vec::new(),
    }
    .solve()
}
