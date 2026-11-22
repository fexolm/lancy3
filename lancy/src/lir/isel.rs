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

    pub fn function<I: TargetInst>(&self) -> Result<super::Function<I>, CompileError> {
        super::Function::new(self.count, self.f.value_count())
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
    uses: usize,
    register: bool,
}

pub(super) struct Context {
    defs: DenseMap<Value, Definition>,
    current: Block,
    #[cfg(target_arch = "x86_64")]
    previous: Option<(Value, Op)>,
    #[cfg(target_arch = "x86_64")]
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
        (b == self.current && def.uses == 1 && op.pure(None)).then_some(op)
    }

    #[cfg(target_arch = "x86_64")]
    pub fn adjacent_single_use(&self, v: Value) -> Option<Op> {
        let (previous, op) = self.previous?;
        (previous == v && self.defs[v].uses == 1).then_some(op)
    }

    #[cfg(target_arch = "x86_64")]
    pub fn cover_adjacent_load(&mut self, v: Value) {
        debug_assert!(matches!(
            self.adjacent_single_use(v),
            Some(Op::Load(..) | Op::Simd(crate::hir::Simd::Load(..)))
        ));
        self.covered = Some(v);
    }
}

pub(super) fn select<I>(
    f: &Function,
    cfg: &ControlFlowGraph,
    mut rule: impl FnMut(&mut Context, Root, &mut Vec<I>) -> Result<(), CompileError>,
) -> Result<DenseMap<Block, Vec<I>>, CompileError> {
    let mut ctx = Context {
        defs: DenseMap::filled(f.value_count(), Definition::default()),
        current: f.entry(),
        #[cfg(target_arch = "x86_64")]
        previous: None,
        #[cfg(target_arch = "x86_64")]
        covered: None,
    };
    for &b in cfg.reverse_postorder() {
        for i in f.instructions(b) {
            let op = *f.op(i);
            if let Some(v) = f.result(i) {
                ctx.defs[v].source = Some((b, op));
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
    for &b in cfg.reverse_postorder().iter().rev() {
        ctx.current = b;
        let out: &mut Vec<I> = &mut blocks[b];
        let mut emit = |ctx: &mut Context, root| -> Result<(), CompileError> {
            let first = out.len();
            rule(ctx, root, out)?;
            out[first..].reverse();
            Ok(())
        };
        emit(&mut ctx, Root::Term(*f.terminator(b)))?;
        instructions.extend(f.instructions(b));
        while let Some(i) = instructions.pop() {
            #[cfg(target_arch = "x86_64")]
            if let Some(covered) = ctx.covered.take() {
                debug_assert_eq!(f.result(i), Some(covered));
                continue;
            }
            #[cfg(all(target_arch = "x86_64", feature = "experimental-simd"))]
            {
                ctx.previous = instructions
                    .last()
                    .and_then(|&prev| Some((f.result(prev)?, *f.op(prev))));
            }
            if f.result(i)
                .is_some_and(|v| !ctx.defs[v].register && f.op(i).pure(Some(f.value_type(v))))
            {
                continue;
            }
            emit(&mut ctx, Root::Inst(i))?;
        }
        #[cfg(all(target_arch = "x86_64", feature = "experimental-simd"))]
        {
            ctx.previous = None;
        }
        out.reverse();
    }
    Ok(blocks)
}
