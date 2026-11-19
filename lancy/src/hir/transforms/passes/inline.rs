//! Expand the verified call DAG in callee-first order. A template is built once
//! per distinct function, including when many call sites share its body.
use crate::CompileError;
use crate::hir::transforms::ControlFlowGraph;
use crate::hir::{Block, Edge, Function, Op, Terminator, Value, ssa::VerifiedBody};
use crate::module::{FunctionId, FunctionIds, Module};
use crate::support::slotmap::DenseMap;

struct Template {
    body: Function,
    cfg: ControlFlowGraph,
    returns: usize,
}

pub(crate) fn inline_all(
    module: &Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<Function, CompileError> {
    check_expansion(module)?;
    let mut uses = DenseMap::filled(module.functions.len(), 0usize);
    for id in module.functions.keys() {
        for &child in module.call_graph.callees(id) {
            uses[child] += 1;
        }
    }
    let mut templates: DenseMap<FunctionId, Option<Template>> = DenseMap::new();
    for (id, source) in module.functions.iter() {
        let src = source.function;
        let mut body = Function::new(src.param_types(), src.return_type());
        let entry = body.entry();
        let args = body.params(entry).to_vec();
        let mut copy = Copier::new(src, &source.cfg, &mut body, entry, &args);
        copy.copy_blocks(
            &mut body,
            &source.cfg,
            Exit::Root,
            |out, at, callee, args| {
                let id = ids[&std::ptr::from_ref(callee)];
                let template = templates[id].as_ref().unwrap();
                let result = template.expand(out, at, args);
                uses[id] -= 1;
                if uses[id] == 0 {
                    templates[id] = None;
                }
                result
            },
        );
        // A callee can loop forever. Its call continuation, and even other
        // caller blocks dominated by the call, then become unreachable.
        body.prune_unreachable()?;
        if id == module.call_graph.root() {
            return Ok(body);
        }
        let cfg = ControlFlowGraph::compute(&body)?;
        let returns = body
            .blocks()
            .filter(|&b| matches!(body.terminator(b), Terminator::Return(_)))
            .count();
        templates.insert(Some(Template { body, cfg, returns }));
    }
    unreachable!("a verified module contains its root")
}

/// Account for the emitted entities at every call site before copying a shared
/// DAG. A diamond of small functions can exceed a 32-bit index in a few levels.
#[derive(Default)]
struct Size {
    insts: u64,
    blocks: u64,
    values: u64,
    edges: u64,
    args: u64,
    returns: u64,
}

fn check_expansion(module: &Module<VerifiedBody<'_>>) -> Result<(), CompileError> {
    let mut sizes: DenseMap<FunctionId, Size> = DenseMap::new();
    for (id, src) in module.functions.iter() {
        let f = src.function;
        let mut size = Size::default();
        for &b in src.cfg.reverse_postorder() {
            size.blocks += 1;
            size.values += f.params(b).len() as u64;
            for i in f.instructions(b) {
                if !matches!(f.op(i), Op::Call(_)) {
                    size.insts += 1;
                    size.values += u64::from(f.result(i).is_some());
                }
            }
            let term = *f.terminator(b);
            size.returns += u64::from(matches!(term, Terminator::Return(_)));
            for e in term.edges() {
                size.edges += 1;
                size.args += f.edge(e).1.len() as u64;
            }
        }
        for &child in module.call_graph.callees(id) {
            let c = &sizes[child];
            let join = u64::from(c.returns != 1);
            size.insts += c.insts;
            size.blocks += c.blocks - 1 + join;
            size.values += c.values - module.functions[child].function.arity() as u64 + join;
            size.edges += c.edges + join * c.returns;
            size.args += c.args + join * c.returns;
        }
        if size.insts + size.blocks > i32::MAX as u64
            || size.values > regalloc2::VReg::MAX as u64
            || size.edges > u32::MAX as u64
            || size.args > u32::MAX as u64
        {
            return Err(CompileError::FunctionTooLarge);
        }
        sizes.insert(size);
    }
    Ok(())
}

enum Exit {
    Root,
    Single,
    Join(Block),
}

impl Template {
    fn expand(&self, out: &mut Function, at: Block, args: &[Value]) -> (Block, Value) {
        let mut copy = Copier::new(&self.body, &self.cfg, out, at, args);
        let join = (self.returns != 1).then(|| {
            let b = out.create_block();
            (b, out.append_typed_param(b, self.body.return_type()))
        });
        let exit = join.map_or(Exit::Single, |(b, _)| Exit::Join(b));
        copy.copy_blocks(out, &self.cfg, exit, |_, _, _, _| {
            unreachable!("expanded template contains a call")
        })
        .or(join)
        .expect("single return or continuation")
    }
}

struct Copier<'a> {
    src: &'a Function,
    blocks: DenseMap<Block, Block>,
    values: DenseMap<Value, Value>,
}

impl<'a> Copier<'a> {
    fn new(
        src: &'a Function,
        cfg: &ControlFlowGraph,
        out: &mut Function,
        entry: Block,
        args: &[Value],
    ) -> Self {
        let mut blocks = DenseMap::filled(src.block_capacity(), Block::INVALID);
        let mut values = DenseMap::filled(src.value_count(), Value::INVALID);
        blocks[src.entry()] = entry;
        for (&param, &arg) in src.params(src.entry()).iter().zip(args) {
            values[param] = arg;
        }
        for &b in &cfg.reverse_postorder()[1..] {
            let target = out.create_block();
            blocks[b] = target;
            for &v in src.params(b) {
                values[v] = out.append_typed_param(target, src.value_type(v));
            }
        }
        Self {
            src,
            blocks,
            values,
        }
    }

    fn edge(&self, out: &mut Function, edge: Edge) -> Edge {
        let (target, args) = self.src.edge(edge);
        out.add_edge(self.blocks[target], &self.arguments(args))
    }

    fn arguments(&self, args: &[Value]) -> Vec<Value> {
        args.iter().map(|&v| self.values[v]).collect()
    }

    fn copy_blocks(
        &mut self,
        out: &mut Function,
        cfg: &ControlFlowGraph,
        exit: Exit,
        mut call: impl FnMut(&mut Function, Block, &Function, &[Value]) -> (Block, Value),
    ) -> Option<(Block, Value)> {
        let mut single = None;
        for &b in cfg.reverse_postorder() {
            let mut at = self.blocks[b];
            for i in self.src.instructions(b) {
                let mut op = *self.src.op(i);
                if let Op::Call(site) = op {
                    let (callee, args) = self.src.call(site);
                    let result = self.src.result(i).unwrap();
                    (at, self.values[result]) = call(out, at, callee, &self.arguments(args));
                } else {
                    op.map_operands(|v| self.values[v]);
                    if let Some(result) = out.append(at, op) {
                        self.values[self.src.result(i).unwrap()] = result;
                    }
                }
            }
            let term = match *self.src.terminator(b) {
                Terminator::Return(v) => match exit {
                    Exit::Root => Terminator::Return(self.values[v]),
                    Exit::Join(target) => Terminator::Jump(out.add_edge(target, &[self.values[v]])),
                    Exit::Single => {
                        single = Some((at, self.values[v]));
                        continue;
                    }
                },
                Terminator::Jump(e) => Terminator::Jump(self.edge(out, e)),
                Terminator::Branch { cond, yes, no } => Terminator::Branch {
                    cond: self.values[cond],
                    yes: self.edge(out, yes),
                    no: self.edge(out, no),
                },
            };
            out.set_terminator(at, term);
        }
        single
    }
}
