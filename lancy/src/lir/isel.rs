//! Consumer-first selection over the SSA use-def graph. Rules inspect definitions
//! without committing to a cover. Only requesting a register makes its producer live.
use crate::CompileError;
use crate::hir::{Block, Function, Inst, Op, Terminator, Value, transforms::ControlFlowGraph};
use crate::support::slotmap::DenseMap;

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
}

impl Context {
    pub fn require(&mut self, v: Value) {
        self.defs[v].register = true;
    }
    pub fn constant(&self, v: Value) -> Option<i32> {
        match self.defs[v].source {
            Some((_, Op::Const(n))) => Some(n),
            _ => None,
        }
    }
    /// Only sink total computations. Memory accesses, checks and division
    /// keep their original execution point, including when their result is dead.
    pub fn single_use(&self, v: Value) -> Option<Op> {
        let def = &self.defs[v];
        let (b, op) = def.source?;
        (b == self.current && def.uses == 1 && op.pure(None)).then_some(op)
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
        for i in instructions.drain(..).rev() {
            if f.result(i)
                .is_some_and(|v| !ctx.defs[v].register && f.op(i).pure(Some(f.value_type(v))))
            {
                continue;
            }
            emit(&mut ctx, Root::Inst(i))?;
        }
        out.reverse();
    }
    Ok(blocks)
}
