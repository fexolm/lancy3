use crate::hir::transforms::ControlFlowGraph;
use crate::hir::{
    Binary, Block, Function, Op, Terminator, Ternary, Type, Value,
    ssa::{Edge, Inst},
};
use crate::support::slotmap::DenseMap;

use super::simplify::fold_i32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fact {
    Unknown,
    Constant(i32),
    Varying,
}

impl Fact {
    pub(super) fn constant(self) -> Option<i32> {
        match self {
            Self::Constant(n) => Some(n),
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

/// Cheap, total integer identities. The producers of discarded operands
/// remain live if they can trap or have other observable effects.
pub(super) fn reduce(
    kind: Binary,
    lhs: Value,
    rhs: Value,
    a: Option<i32>,
    b: Option<i32>,
) -> Option<Op> {
    use Binary::*;
    if let (Some(a), Some(b)) = (a, b) {
        return fold_i32(kind, a, b).map(Op::Const);
    }
    if lhs == rhs {
        match kind {
            Sub | Xor | Ne | Lt | Gt => return Some(Op::Const(0)),
            Eq | Le | Ge => return Some(Op::Const(-1)),
            And | Or | Min | Max => return Some(Op::Copy(lhs)),
            _ => {}
        }
    }
    match (kind, a, b) {
        (Add | Or | Xor, Some(0), _) | (Mul, Some(1), _) | (And, Some(-1), _) => {
            Some(Op::Copy(rhs))
        }
        (Add | Sub | Or | Xor | SaturatingAdd | SaturatingSub, _, Some(0))
        | (Mul | Div, _, Some(1))
        | (And, _, Some(-1)) => Some(Op::Copy(lhs)),
        (Shl | Shr | RotateLeft | RotateRight, _, Some(n)) if n & 31 == 0 => Some(Op::Copy(lhs)),
        (Mul | And, Some(0), _) | (Mul | And, _, Some(0)) | (Rem, _, Some(1)) => Some(Op::Const(0)),
        (Or, Some(-1), _) | (Or, _, Some(-1)) => Some(Op::Const(-1)),
        _ => None,
    }
}

fn evaluate(f: &Function, facts: &DenseMap<Value, Fact>, op: Op) -> Fact {
    use Fact::*;
    match op {
        Op::Const(n) => Constant(n),
        Op::Copy(v) => facts[v],
        Op::Bin(kind, a, b) if f.value_type(a) == Type::I32 => {
            let (x, y) = (facts[a], facts[b]);
            match reduce(kind, a, b, x.constant(), y.constant()) {
                Some(Op::Const(n)) => Constant(n),
                Some(Op::Copy(v)) => facts[v],
                _ if x == Unknown || y == Unknown => Unknown,
                _ => Varying,
            }
        }
        Op::Ternary(Ternary::Select, c, yes, no) if f.value_type(yes) == Type::I32 => {
            if yes == no {
                return facts[yes];
            }
            match facts[c] {
                Constant(0) => facts[no],
                Constant(_) => facts[yes],
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
                    Op::Copy(v) => self.add_use(v, user),
                    Op::Bin(_, a, b) if self.f.value_type(a) == Type::I32 => {
                        self.add_use(a, user);
                        if a != b {
                            self.add_use(b, user);
                        }
                    }
                    Op::Ternary(Ternary::Select, c, a, b) if self.f.value_type(a) == Type::I32 => {
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
