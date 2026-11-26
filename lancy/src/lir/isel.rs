//! Consumer-first selection over the SSA use-def graph. Rules inspect definitions
//! without committing to a cover. Only requesting a register makes its producer live.
use crate::CompileError;
use crate::hir::{
    Block, Function, Inst, Op, Terminator, Value,
    ssa::VerifiedBody,
    transforms::{ControlFlowGraph, DomTree, NaturalLoops},
};
use crate::support::slotmap::DenseMap;
use regalloc2::{Block as MachineBlock, VReg};

use super::TargetInst;

/// Number machine blocks in reverse postorder and insert a bridge on each
/// critical edge so the allocator can emit its parallel copies unconditionally.
pub(super) struct Edges<'a> {
    f: &'a Function,
    cfg: &'a ControlFlowGraph,
    dom: &'a DomTree,
    loops: Option<(NaturalLoops, DenseMap<Block, usize>)>,
    mapping: DenseMap<Block, MachineBlock>,
    count: usize,
}

impl<'a> Edges<'a> {
    pub fn new(source: &'a VerifiedBody<'_>) -> Self {
        let VerifiedBody {
            function: f,
            cfg,
            dom,
            ..
        } = source;
        let mut edges = Self {
            f,
            cfg,
            dom,
            loops: cfg.has_cycle().then(|| {
                let loops = NaturalLoops::discover(cfg, dom);
                let depths = loops.header_depths();
                (loops, depths)
            }),
            mapping: DenseMap::filled(f.block_capacity(), MachineBlock::invalid()),
            count: 0,
        };
        for &b in cfg.reverse_postorder() {
            edges.mapping[b] = MachineBlock::new(edges.count);
            edges.count += 1 + cfg.successors(b).filter(|&t| edges.split(b, t)).count();
        }
        edges
    }

    fn split(&self, source: Block, target: Block) -> bool {
        matches!(self.f.terminator(source), Terminator::Branch { .. })
            && self.cfg.predecessors(target).len() > 1
    }

    pub fn function<I: TargetInst>(
        &self,
        machine: I::Machine,
    ) -> Result<super::Function<I>, CompileError> {
        super::Function::new(machine, self.count, self.f.value_count())
    }

    pub fn finish<I: TargetInst>(
        &self,
        body: &mut super::Function<I>,
        b: Block,
        reg: impl Fn(Value) -> VReg,
        jump: impl Fn() -> I,
    ) -> Result<(), CompileError> {
        let term = *self.f.terminator(b);
        let mut bridge = self.mapping[b].index() + 1;
        for e in term.edges() {
            let (target, args) = self.f.edge(e);
            let backedge = self.dom.dominates(target, b);
            if self.split(b, target) {
                body.edge(MachineBlock::new(bridge), [], backedge);
                bridge += 1;
            } else {
                body.edge(
                    self.mapping[target],
                    args.iter().copied().map(&reg),
                    backedge,
                );
            }
        }
        let params = if b == self.f.entry() {
            &[]
        } else {
            self.f.params(b)
        };
        let depth = self.loops.as_ref().map_or(0, |(loops, depths)| {
            depths.get(loops.owner[b]).copied().unwrap_or(0)
        });
        body.end_block(params.iter().copied().map(&reg), depth)?;
        for e in term.edges() {
            let (target, args) = self.f.edge(e);
            if self.split(b, target) {
                body.push(jump());
                let backedge = self.dom.dominates(target, b);
                body.edge(
                    self.mapping[target],
                    args.iter().copied().map(&reg),
                    backedge,
                );
                let depth = self
                    .loops
                    .as_ref()
                    .map_or(0, |(loops, depths)| loops.edge_depth(depths, b, target));
                body.end_block([], depth)?;
            }
        }
        Ok(())
    }
}

pub(super) enum Root {
    Inst(Inst),
    Term(Terminator),
}

#[derive(Default, Clone)]
struct Definition {
    source: Option<(Block, Op)>,
    /// The definition has no effects, so it may move or disappear.
    pure: bool,
    uses: usize,
    register: bool,
}

pub(super) struct Context {
    defs: DenseMap<Value, Definition>,
    current: Block,
    /// The value defined right before the root being selected, if any.
    previous: Option<(Value, Op)>,
    covered: Option<Value>,
}

impl Context {
    pub fn require(&mut self, v: Value) {
        self.defs[v].register = true;
    }
    pub fn constant(&self, v: Value) -> Option<i32> {
        match self.defining_op(v) {
            Some(Op::Const(n)) => Some(n),
            _ => None,
        }
    }

    pub fn defining_op(&self, v: Value) -> Option<Op> {
        self.defs[v].source.map(|(_, op)| op)
    }
    /// Only sink total computations. Memory accesses, checks and division
    /// keep their original execution point, including when their result is dead.
    pub fn single_use(&self, v: Value) -> Option<Op> {
        let def = &self.defs[v];
        let (b, op) = def.source?;
        (b == self.current && def.uses == 1 && def.pure).then_some(op)
    }

    /// The definition of `v` if it is the instruction right before the
    /// root and the root is its only use: nothing executes between them, so
    /// the root may perform it, even a memory access, after [`Self::cover`].
    pub fn adjacent_single_use(&self, v: Value) -> Option<Op> {
        let (previous, op) = self.previous?;
        (previous == v && self.defs[v].uses == 1).then_some(op)
    }

    /// The root performs the adjacent definition of `v` itself; selection
    /// skips that instruction.
    pub fn cover(&mut self, v: Value) {
        debug_assert!(self.adjacent_single_use(v).is_some());
        self.covered = Some(v);
    }
}

pub(super) fn select<I: Copy>(
    f: &Function,
    cfg: &ControlFlowGraph,
    mut rule: impl FnMut(&mut Context, Root, &mut Vec<I>) -> Result<(), CompileError>,
) -> Result<DenseMap<Block, Vec<I>>, CompileError> {
    let mut ctx = Context {
        defs: DenseMap::filled(f.value_count(), Definition::default()),
        current: f.entry(),
        previous: None,
        covered: None,
    };
    for &b in cfg.reverse_postorder() {
        for i in f.instructions(b) {
            let op = *f.op(i);
            if let Some(v) = f.result(i) {
                ctx.defs[v].source = Some((b, op));
                ctx.defs[v].pure = op.effects(f).is_none();
            }
            f.visit_operands(i, |v| ctx.defs[v].uses += 1);
        }
        let term = *f.terminator(b);
        if let Terminator::Return(v) | Terminator::Branch { cond: v, .. } = term {
            ctx.defs[v].uses += 1;
        }
        for e in term.edges() {
            for &v in f.edge(e).1 {
                ctx.defs[v].uses += 1;
                ctx.require(v);
            }
        }
    }
    let mut blocks = DenseMap::from_fn(f.block_capacity(), |_| Vec::new());
    let mut instructions = Vec::new();
    // Roots are selected from the terminator backwards; each rule appends
    // its instructions in execution order, and the groups are then laid out
    // in program order.
    let mut selected = Vec::new();
    let mut groups = Vec::new();
    for &b in cfg.reverse_postorder().iter().rev() {
        ctx.current = b;
        selected.clear();
        groups.clear();
        let mut emit = |ctx: &mut Context, root| -> Result<(), CompileError> {
            let first = selected.len();
            rule(ctx, root, &mut selected)?;
            groups.push(first..selected.len());
            Ok(())
        };
        ctx.previous = f
            .instructions(b)
            .last()
            .and_then(|i| Some((f.result(i)?, *f.op(i))));
        emit(&mut ctx, Root::Term(*f.terminator(b)))?;
        instructions.extend(f.instructions(b));
        while let Some(i) = instructions.pop() {
            if let Some(covered) = ctx.covered.take() {
                debug_assert_eq!(f.result(i), Some(covered));
                continue;
            }
            ctx.previous = instructions
                .last()
                .and_then(|&prev| Some((f.result(prev)?, *f.op(prev))));
            if f.result(i)
                .is_some_and(|v| !ctx.defs[v].register && ctx.defs[v].pure)
            {
                continue;
            }
            emit(&mut ctx, Root::Inst(i))?;
        }
        let out = &mut blocks[b];
        out.reserve(selected.len());
        for group in groups.iter().rev() {
            out.extend_from_slice(&selected[group.clone()]);
        }
    }
    Ok(blocks)
}
