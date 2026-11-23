//! Expand the verified call DAG in callee-first order. A template is built once
//! per distinct function, including when many call sites share its body.
//!
//! Scratch memory and the debugger workspace belong to a frame. An inlined
//! callee's regions move above the caller's own, so caller data stored there
//! survives the call exactly as it did with separate frames. Calls of one
//! caller never overlap in time, so they share the region above it.
use crate::CompileError;
use crate::hir::transforms::ControlFlowGraph;
use crate::hir::{Block, Edge, Function, Op, Terminator, Value, ssa::VerifiedBody};
use crate::module::{FunctionId, FunctionIds, Module};
use crate::support::slotmap::DenseMap;
use std::sync::Arc;

struct Template {
    body: Function,
    cfg: ControlFlowGraph,
    returns: usize,
}

pub(crate) fn inline_all(
    module: &Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<Function, CompileError> {
    check_expansion(module, ids)?;
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
        body.copy_slice_params(src);
        let entry = body.entry();
        let args = body.params(entry).to_vec();
        let mut copy = Copier::new(src, &source.cfg, &mut body, entry, &args, Shift::default());
        let above = Shift::above(src);
        copy.copy_blocks(
            &mut body,
            &source.cfg,
            Exit::Root,
            |out, at, callee, args| {
                let id = ids[&std::ptr::from_ref(callee)];
                let template = templates[id].as_ref().unwrap();
                let result = template.expand(out, at, args, above);
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
        super::checkpoint(&body, "inline");
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
#[derive(Clone, Copy, Default)]
struct Size {
    insts: u64,
    blocks: u64,
    values: u64,
    edges: u64,
    args: u64,
    returns: u64,
}

impl Size {
    fn block(&mut self, params: usize) {
        self.blocks += 1;
        self.values += params as u64;
    }

    fn inst(&mut self, has_result: bool) {
        self.insts += 1;
        self.values += u64::from(has_result);
    }

    fn terminator(&mut self, f: &Function, term: Terminator) {
        self.returns += u64::from(matches!(term, Terminator::Return(_)));
        for edge in term.edges() {
            self.edges += 1;
            self.args += f.edge(edge).1.len() as u64;
        }
    }

    fn call(&mut self, callee: Self, arity: usize, join: bool) {
        let join = u64::from(join);
        self.insts += callee.insts;
        self.blocks += callee.blocks - 1 + join;
        self.values += callee.values - arity as u64 + join;
        self.edges += callee.edges + join * callee.returns;
        self.args += callee.args + join * callee.returns;
    }

    fn check(self) -> Result<(), CompileError> {
        if self.insts + self.blocks > i32::MAX as u64
            || self.values > regalloc2::VReg::MAX as u64
            || self.edges > u32::MAX as u64
            || self.args > u32::MAX as u64
        {
            Err(CompileError::Unsupported(crate::Limit::Registers))
        } else {
            Ok(())
        }
    }
}

/// Offsets added to an inlined callee's frame-local regions.
#[derive(Clone, Copy, Default)]
struct Shift {
    scratch: u32,
    debug: u32,
}

impl Shift {
    /// Place callee regions above everything `caller` addresses itself.
    /// `check_expansion` has bounded the nested sums by `u32::MAX`.
    fn above(caller: &Function) -> Self {
        Self {
            scratch: caller.scratch_bytes() as u32,
            debug: caller.debug_bytes() as u32,
        }
    }

    fn apply(self, op: Op) -> Op {
        match op {
            Op::CallScratch { offset, bytes } => Op::CallScratch {
                offset: offset + self.scratch,
                bytes,
            },
            Op::DebugStore(offset, value) => Op::DebugStore(offset + self.debug, value),
            op => op,
        }
    }
}

fn check_expansion(
    module: &Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<(), CompileError> {
    let mut sizes: DenseMap<FunctionId, Size> = DenseMap::new();
    // Frame-local bytes after inlining: own regions plus the largest callee.
    let mut frames: DenseMap<FunctionId, [u64; 2]> = DenseMap::new();
    for (id, src) in module.functions.iter() {
        let f = src.function;
        let own = [f.scratch_bytes(), f.debug_bytes()];
        let mut frame = own;
        for &child in module.call_graph.callees(id) {
            for i in 0..2 {
                frame[i] = frame[i].max(own[i] + frames[child][i]);
            }
        }
        if frame.iter().any(|&bytes| bytes > u64::from(u32::MAX)) {
            return Err(CompileError::Unsupported(crate::Limit::Frame));
        }
        frames.insert(frame);
        let callee = |site| {
            let (target, _) = f.call(site);
            (sizes[ids[&Arc::as_ptr(target)]], target.arity())
        };
        // The copier allocates even dead continuations in this body. Its
        // callers only copy the blocks reachable after pruning.
        let prunes = module
            .call_graph
            .callees(id)
            .iter()
            .any(|&c| sizes[c].returns == 0);
        let reachable = prunes.then(|| {
            let mut reachable = DenseMap::filled(f.block_capacity(), false);
            let mut pending = vec![f.entry()];
            while let Some(block) = pending.pop() {
                if std::mem::replace(&mut reachable[block], true) {
                    continue;
                }
                if !f.instructions(block).any(
                    |inst| matches!(*f.op(inst), Op::Call(site) if callee(site).0.returns == 0),
                ) {
                    pending.extend(src.cfg.successors(block));
                }
            }
            reachable
        });
        let mut emitted = Size::default();
        let mut retained = Size::default();
        for &b in src.cfg.reverse_postorder() {
            let mut live = reachable.as_ref().is_some_and(|reachable| reachable[b]);
            let params = f.params(b).len();
            emitted.block(params);
            if live {
                retained.block(params);
            }
            for i in f.instructions(b) {
                if let Op::Call(site) = *f.op(i) {
                    let (callee, arity) = callee(site);
                    emitted.call(callee, arity, callee.returns != 1);
                    emitted.check()?;
                    if live {
                        retained.call(callee, arity, callee.returns > 1);
                        live = callee.returns != 0;
                    }
                } else {
                    let has_result = f.result(i).is_some();
                    emitted.inst(has_result);
                    if live {
                        retained.inst(has_result);
                    }
                }
            }
            let term = *f.terminator(b);
            emitted.terminator(f, term);
            if live {
                retained.terminator(f, term);
            }
        }
        emitted.check()?;
        sizes.insert(if prunes { retained } else { emitted });
    }
    Ok(())
}

enum Exit {
    Root,
    Single,
    Join(Block),
}

impl Template {
    fn expand(
        &self,
        out: &mut Function,
        at: Block,
        args: &[Value],
        shift: Shift,
    ) -> (Block, Value) {
        let mut copy = Copier::new(&self.body, &self.cfg, out, at, args, shift);
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
    shift: Shift,
}

impl<'a> Copier<'a> {
    fn new(
        src: &'a Function,
        cfg: &ControlFlowGraph,
        out: &mut Function,
        entry: Block,
        args: &[Value],
        shift: Shift,
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
            shift,
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
                    if let Op::RustPacketCall(site) = op {
                        let (import, scratch, operands) = self.src.packet_call(site);
                        op = Op::RustPacketCall(out.add_packet_call(
                            import,
                            scratch + self.shift.scratch,
                            &self.arguments(operands),
                        ));
                    } else {
                        op.map_operands(|v| self.values[v]);
                        op = self.shift.apply(op);
                    }
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
