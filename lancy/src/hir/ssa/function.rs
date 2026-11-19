//! Typed SSA with block parameters. IDs are stable across instruction rewrites
//! and block merges. Instruction order lives in the layout, independently of IDs.

#[cfg(test)]
mod interpret;
mod simplify;

use super::{Op, Terminator, Type, Value};
use crate::hir::{
    IrError,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::support::slotmap::{DenseMap, SlotMap};
use crate::{
    CompileError,
    module::{FunctionIds, Module},
};
use std::{ops::Range, sync::Arc};

crate::slot_key!(pub Block);

pub(crate) struct VerifiedBody<'a> {
    pub function: &'a Function,
    pub cfg: ControlFlowGraph,
}

#[derive(Clone, Debug)]
struct BlockData {
    params: Vec<Value>,
    first: Inst,
    last: Inst,
    term: Option<Terminator>,
}

impl Default for BlockData {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            first: Inst::INVALID,
            last: Inst::INVALID,
            term: None,
        }
    }
}

crate::slot_key!(pub Inst);

#[derive(Clone, Debug)]
struct InstData {
    op: Op,
    // Value::INVALID for instructions with no result.
    result: Value,
    next: Inst,
}

#[derive(Clone, Copy, Debug)]
struct Span {
    start: u32,
    len: u32,
}

impl Span {
    fn append<T: Copy>(arena: &mut Vec<T>, slice: &[T]) -> Self {
        let start = u32::try_from(arena.len()).expect("exhausted operand arena");
        let len = u32::try_from(slice.len()).expect("operand list too large");
        start.checked_add(len).expect("exhausted operand arena");
        arena.extend_from_slice(slice);
        Self { start, len }
    }

    fn range(self) -> Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }
}

crate::slot_key!(pub Call);

#[derive(Clone, Debug)]
struct CallData {
    callee: Arc<Function>,
    args: Span,
}

crate::slot_key!(pub Edge);

#[derive(Clone, Debug)]
pub struct Function {
    arity: usize,
    return_type: Type,
    blocks: SlotMap<Block, BlockData>,
    insts: SlotMap<Inst, InstData>,
    edges: DenseMap<Edge, (Block, Span)>,
    calls: DenseMap<Call, CallData>,
    operands: Vec<Value>,
    types: DenseMap<Value, Type>,
}

impl Function {
    /// Create a function with typed parameters in ABI order.
    pub fn new(params: &[Type], return_type: Type) -> Self {
        Self::from_types(params.iter().copied(), return_type)
    }

    pub fn scalar(arity: usize) -> Self {
        Self::from_types(std::iter::repeat_n(Type::I32, arity), Type::I32)
    }

    fn from_types(params: impl ExactSizeIterator<Item = Type>, return_type: Type) -> Self {
        let arity = params.len();
        let mut f = Self {
            arity,
            return_type,
            blocks: SlotMap::new(),
            insts: SlotMap::new(),
            edges: DenseMap::new(),
            calls: DenseMap::new(),
            operands: Vec::new(),
            types: DenseMap::with_capacity(arity),
        };
        let entry = f.create_block();
        for ty in params {
            f.append_typed_param(entry, ty);
        }
        f
    }

    pub fn arity(&self) -> usize {
        self.arity
    }
    pub fn param_types(&self) -> &[Type] {
        &self.types.as_slice()[..self.arity]
    }
    pub fn return_type(&self) -> Type {
        self.return_type
    }
    pub fn value_type(&self, value: Value) -> Type {
        self.types[value]
    }
    pub fn entry(&self) -> Block {
        Block::from_index(0)
    }
    pub fn value_count(&self) -> usize {
        self.types.len()
    }
    /// Number of block IDs issued, including blocks retired by a merge.
    pub fn block_capacity(&self) -> usize {
        self.blocks.slot_count()
    }
    pub fn blocks(&self) -> impl Iterator<Item = Block> + '_ {
        self.blocks.keys()
    }
    pub fn params(&self, b: Block) -> &[Value] {
        &self.blocks[b].params
    }
    pub fn instructions(&self, b: Block) -> impl Iterator<Item = Inst> + '_ {
        let mut cursor = self.blocks[b].first;
        std::iter::from_fn(move || {
            if cursor == Inst::INVALID {
                return None;
            }
            let inst = cursor;
            cursor = self.insts[inst].next;
            Some(inst)
        })
    }
    pub fn op(&self, i: Inst) -> &Op {
        &self.insts[i].op
    }
    pub fn result(&self, i: Inst) -> Option<Value> {
        let result = self.insts[i].result;
        (result != Value::INVALID).then_some(result)
    }
    pub(crate) fn visit_operands(&self, i: Inst, mut visit: impl FnMut(Value)) {
        let mut op = *self.op(i);
        if let Op::Call(site) = op {
            self.call(site).1.iter().copied().for_each(visit);
        } else {
            op.map_operands(|v| {
                visit(v);
                v
            });
        }
    }
    pub fn terminator(&self, b: Block) -> &Terminator {
        self.blocks[b]
            .term
            .as_ref()
            .expect("unterminated SSA block")
    }
    pub fn edge(&self, edge: Edge) -> (Block, &[Value]) {
        let (block, args) = self.edges[edge];
        (block, &self.operands[args.range()])
    }
    pub fn call(&self, call: Call) -> (&Arc<Function>, &[Value]) {
        let data = &self.calls[call];
        (&data.callee, &self.operands[data.args.range()])
    }

    pub fn create_block(&mut self) -> Block {
        self.blocks.insert(BlockData::default())
    }

    pub fn append_param(&mut self, block: Block) -> Value {
        self.append_typed_param(block, Type::I32)
    }
    pub fn append_typed_param(&mut self, block: Block, ty: Type) -> Value {
        let params = &mut self.blocks[block].params;
        let value = self.types.insert(ty);
        params.push(value);
        value
    }
    pub fn append_inst(&mut self, b: Block, op: Op) -> Inst {
        let last = self.blocks[b].last;
        let result = if op.produces_value() {
            let ty = match op {
                Op::Simd(op) => op
                    .result_type(|v| self.types.get(v).copied().unwrap_or(Type::I32))
                    .expect("value-producing SIMD operation"),
                Op::Literal(lane, _) => lane.scalar(),
                Op::Ternary(_, _, v, _) => self.types.get(v).copied().unwrap_or(Type::I32),
                Op::Unary(op, v) => op.result_type(self.types.get(v).copied().unwrap_or(Type::I32)),
                Op::Cast(ty, _) | Op::Bitcast(ty, _) => ty,
                Op::Bin(op, v, _) => {
                    op.result_type(self.types.get(v).copied().unwrap_or(Type::I32))
                }
                Op::PtrOffset(..) => Type::Ptr,
                Op::Load(ty, _) => ty,
                Op::Copy(v) => self.types.get(v).copied().unwrap_or(Type::I32),
                Op::Call(c) => self
                    .calls
                    .get(c)
                    .map_or(Type::I32, |c| c.callee.return_type()),
                _ => Type::I32,
            };
            self.types.insert(ty)
        } else {
            Value::INVALID
        };
        let id = self.insts.insert(InstData {
            op,
            result,
            next: Inst::INVALID,
        });
        if last == Inst::INVALID {
            self.blocks[b].first = id;
        } else {
            self.insts[last].next = id;
        }
        self.blocks[b].last = id;
        id
    }
    pub fn append(&mut self, b: Block, op: Op) -> Option<Value> {
        let id = self.append_inst(b, op);
        self.result(id)
    }
    pub fn add_call(&mut self, callee: Arc<Function>, args: &[Value]) -> Call {
        let args = Span::append(&mut self.operands, args);
        self.calls.insert(CallData { callee, args })
    }
    pub fn add_edge(&mut self, target: Block, args: &[Value]) -> Edge {
        let args = Span::append(&mut self.operands, args);
        self.edges.insert((target, args))
    }
    pub fn set_terminator(&mut self, b: Block, term: Terminator) {
        self.blocks[b].term = Some(term);
    }

    pub fn jump(&mut self, from: Block, to: Block, args: &[Value]) {
        let edge = self.add_edge(to, args);
        self.set_terminator(from, Terminator::Jump(edge));
    }

    /// Replace an instruction and retain its position, ID and result ID.
    /// A rejected edit leaves the original instruction in place.
    pub fn replace_op(&mut self, inst: Inst, op: Op) -> Result<(), IrError> {
        self.replace_ops(&[(inst, op)])
    }

    /// Apply an entire pass worth of replacements in place with one verifier
    /// run. Repeated edits of the same instruction are replayed in order.
    pub fn replace_ops(&mut self, edits: &[(Inst, Op)]) -> Result<(), IrError> {
        for &(inst, op) in edits {
            let Some(old) = self.insts.get(inst) else {
                return Err(if inst.index() < self.insts.slot_count() {
                    IrError::RetiredInstruction { inst }
                } else {
                    IrError::UnknownInstruction { inst }
                });
            };
            if old.op.produces_value() != op.produces_value() {
                return Err(IrError::ReplacementResultCount { inst });
            }
        }
        let mut undo = Vec::with_capacity(edits.len());
        for &(inst, op) in edits {
            let old = std::mem::replace(&mut self.insts[inst].op, op);
            undo.push((inst, old));
        }
        if let Err(err) = self.verify() {
            for (inst, previous) in undo.into_iter().rev() {
                self.insts[inst].op = previous;
            }
            return Err(err);
        }
        Ok(())
    }

    /// Splice a successor with exactly one incoming edge into its predecessor.
    /// Live IDs are stable. The successor block and its block-parameter IDs
    /// are retired. Arguments substitute parameters simultaneously, including
    /// references in dominated blocks, calls and successor edges.
    pub fn merge_blocks(&mut self, pred: Block, succ: Block) -> Result<(), IrError> {
        self.verify()?;
        if pred == succ {
            return Err(IrError::MergeSameBlock { block: pred });
        }
        if self.blocks.get(succ).is_none() {
            return Err(IrError::UnknownBlock { block: succ });
        }
        let pred_block = self
            .blocks
            .get(pred)
            .ok_or(IrError::UnknownBlock { block: pred })?;
        let Some(Terminator::Jump(edge)) = pred_block.term else {
            return Err(IrError::ExpectedJump { block: pred });
        };
        let (target, _) = self.edge(edge);
        if target != succ {
            return Err(IrError::WrongJumpTarget {
                from: pred,
                expected: succ,
                actual: target,
            });
        }
        let incoming = self
            .blocks()
            .flat_map(|b| self.terminator(b).edges())
            .filter(|&e| self.edges[e].0 == succ)
            .count();
        if incoming != 1 {
            return Err(IrError::MergePredecessors {
                block: succ,
                incoming,
            });
        }

        let (_, args) = self.edges[edge];
        let succ = self.blocks.remove(succ).unwrap();
        let last = self.blocks[pred].last;
        if succ.first != Inst::INVALID {
            if last == Inst::INVALID {
                self.blocks[pred].first = succ.first;
            } else {
                self.insts[last].next = succ.first;
            }
            self.blocks[pred].last = succ.last;
        }
        self.blocks[pred].term = succ.term;
        if !succ.params.is_empty() {
            let mut aliases = DenseMap::from_fn(self.types.len(), |v| v);
            for (&param, &actual) in succ.params.iter().zip(&self.operands[args.range()]) {
                aliases[param] = actual;
            }
            self.substitute_params(&aliases);
        }
        Ok(())
    }

    fn substitute_params(&mut self, aliases: &DenseMap<Value, Value>) {
        let map = |v: Value| aliases[v];
        for inst in self.insts.values_mut() {
            inst.op.map_operands(map);
        }
        for v in &mut self.operands {
            // Unreferenced call/edge records can hold speculative operands.
            if let Some(&alias) = aliases.get(*v) {
                *v = alias;
            }
        }
        for block in self.blocks.values_mut() {
            if let Some(Terminator::Return(v) | Terminator::Branch { cond: v, .. }) =
                &mut block.term
            {
                *v = map(*v);
            }
        }
    }

    /// Remove unreachable blocks without renumbering surviving instructions,
    /// blocks, or values. Only reachable blocks must have terminators.
    pub fn prune_unreachable(&mut self) -> Result<usize, IrError> {
        let mut pending = vec![self.entry()];
        let mut live = DenseMap::filled(self.block_capacity(), false);
        while let Some(b) = pending.pop() {
            if std::mem::replace(&mut live[b], true) {
                continue;
            }
            let term = self.blocks[b]
                .term
                .ok_or(IrError::UnterminatedBlock { block: b })?;
            for edge in term.edges() {
                let target = self
                    .edges
                    .get(edge)
                    .ok_or(IrError::UnknownEdge { from: b, edge })?
                    .0;
                if self.blocks.get(target).is_none() {
                    return Err(IrError::UnknownTarget {
                        from: b,
                        edge,
                        target,
                    });
                }
                pending.push(target);
            }
        }
        let mut count = 0;
        for (block, &reachable) in live.iter() {
            if !reachable && let Some(dead) = self.blocks.remove(block) {
                count += 1;
                let mut cursor = dead.first;
                while cursor != Inst::INVALID {
                    cursor = self.insts.remove(cursor).unwrap().next;
                }
            }
        }
        Ok(count)
    }

    pub(in crate::hir) fn checked_successors(
        &self,
        block: Block,
    ) -> Result<[Option<Block>; 2], IrError> {
        let term = self.blocks[block]
            .term
            .ok_or(IrError::UnterminatedBlock { block })?;
        let mut successors = [None; 2];
        for (i, edge) in term.edges().enumerate() {
            let &(target, args) = self
                .edges
                .get(edge)
                .ok_or(IrError::UnknownEdge { from: block, edge })?;
            let target_block = self.blocks.get(target).ok_or(IrError::UnknownTarget {
                from: block,
                edge,
                target,
            })?;
            if target == self.entry() {
                return Err(IrError::EntryBackedge { from: block });
            }
            if target_block.params.len() != args.len as usize {
                return Err(IrError::EdgeArity {
                    from: block,
                    edge,
                    target,
                    expected: target_block.params.len(),
                    actual: args.len as usize,
                });
            }
            successors[i] = Some(target);
        }
        Ok(successors)
    }

    /// Verify this body. Callee signatures participate in the check; code
    /// generation verifies every distinct callee body in the reachable graph.
    pub fn verify(&self) -> Result<(), IrError> {
        self.verified_cfg().map(drop)
    }

    fn verified_cfg(&self) -> Result<ControlFlowGraph, IrError> {
        if self.params(self.entry()).len() != self.arity {
            return Err(IrError::EntryArity {
                expected: self.arity,
                actual: self.params(self.entry()).len(),
            });
        }
        let cfg = ControlFlowGraph::compute(self)?;
        let dom = DomTree::compute(&cfg);
        let mut defs = DenseMap::filled(self.value_count(), None);
        for block in self.blocks() {
            for &param in self.params(block) {
                defs[param] = Some((block, 0));
            }
            for (i, inst) in self.instructions(block).enumerate() {
                if let Some(value) = self.result(inst) {
                    defs[value] = Some((block, i + 1));
                }
            }
        }
        for block in self.blocks() {
            let used = |val: Value, index: usize| -> Result<(), IrError> {
                let &(def, position) = defs
                    .get(val)
                    .and_then(Option::as_ref)
                    .ok_or(IrError::UndefinedValue { value: val, block })?;
                if (block == def && position >= index) || !dom.dominates(def, block) {
                    return Err(IrError::DoesNotDominate {
                        value: val,
                        definition: def,
                        block,
                    });
                }
                Ok(())
            };
            for (i, inst) in self.instructions(block).enumerate() {
                let operand = |v: Value| {
                    used(v, i + 1)?;
                    Ok::<_, IrError>(self.value_type(v))
                };
                let expect = |v, ty| {
                    let actual = operand(v)?;
                    if actual != ty {
                        return Err(IrError::OperandType {
                            inst,
                            value: v,
                            expected: ty,
                            actual,
                        });
                    }
                    Ok(())
                };
                let ty = match *self.op(inst) {
                    Op::Const(_) => Type::I32,
                    Op::Simd(op) => {
                        op.verify(inst, operand)?;
                        let Some(ty) = op.result_type(|v| self.value_type(v)) else {
                            continue;
                        };
                        ty
                    }
                    Op::Literal(lane, _) => lane.scalar(),
                    Op::Copy(v) => operand(v)?,
                    Op::Bin(op, a, b) => {
                        let ty = operand(a)?;
                        expect(b, ty)?;
                        if matches!(ty, Type::Vector(_)) || !op.supports(ty) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        op.result_type(ty)
                    }
                    Op::Unary(op, v) => {
                        let ty = operand(v)?;
                        if matches!(ty, Type::Vector(_)) || !op.supports(ty) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        op.result_type(ty)
                    }
                    Op::Ternary(op, a, b, c) => {
                        let ty = operand(b)?;
                        if matches!(ty, Type::Vector(_)) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        expect(
                            a,
                            if op == super::Ternary::Select {
                                ty.mask_type()
                            } else {
                                ty
                            },
                        )?;
                        expect(c, ty)?;
                        if ty
                            .lane()
                            .is_none_or(|l| op == super::Ternary::MulAdd && !l.float())
                        {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        ty
                    }
                    Op::Cast(to, v) | Op::Bitcast(to, v) => {
                        let ty = operand(v)?;
                        let valid = match (ty, to) {
                            // Expose the address bits for alignment arithmetic.
                            (Type::Ptr, Type::U64) => matches!(self.op(inst), Op::Bitcast(..)),
                            (Type::Ptr, _) | (_, Type::Ptr) => false,
                            (Type::Vector(_), _) | (_, Type::Vector(_)) => false,
                            _ => {
                                matches!(self.op(inst), Op::Cast(..))
                                    || ty.bytes(16) == to.bytes(16)
                            }
                        };
                        if !valid {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        to
                    }
                    Op::BoundsCheck(index, len) => {
                        expect(index, Type::I32)?;
                        expect(len, Type::I32)?;
                        continue;
                    }
                    Op::PtrOffset(base, index, _) => {
                        expect(base, Type::Ptr)?;
                        expect(index, Type::I32)?;
                        Type::Ptr
                    }
                    Op::Load(ty, addr) => {
                        if matches!(ty, Type::Vector(_)) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        expect(addr, Type::Ptr)?;
                        ty
                    }
                    Op::Store(addr, value) => {
                        expect(addr, Type::Ptr)?;
                        let ty = operand(value)?;
                        if matches!(ty, Type::Vector(_)) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        continue;
                    }
                    Op::Call(id) => {
                        let site = self
                            .calls
                            .get(id)
                            .ok_or(IrError::UnknownCall { inst, call: id })?;
                        let args = &self.operands[site.args.range()];
                        if args.len() != site.callee.arity() {
                            return Err(IrError::CallArity {
                                call: id,
                                expected: site.callee.arity(),
                                actual: args.len(),
                            });
                        }
                        for (&arg, &ty) in args.iter().zip(site.callee.param_types()) {
                            expect(arg, ty)?;
                        }
                        site.callee.return_type()
                    }
                };
                let actual = self.value_type(self.result(inst).unwrap());
                if actual != ty {
                    return Err(IrError::ResultType {
                        inst,
                        expected: ty,
                        actual,
                    });
                }
            }
            let term = *self.terminator(block);
            match term {
                Terminator::Branch { cond, .. } => {
                    used(cond, usize::MAX)?;
                    if self.value_type(cond) != Type::I32 {
                        return Err(IrError::BranchType {
                            block,
                            actual: self.value_type(cond),
                        });
                    }
                }
                Terminator::Return(value) => {
                    used(value, usize::MAX)?;
                    if self.value_type(value) != self.return_type {
                        return Err(IrError::ReturnType {
                            block,
                            expected: self.return_type,
                            actual: self.value_type(value),
                        });
                    }
                }
                Terminator::Jump(_) => {}
            }
            for edge in term.edges() {
                let (target, args) = self.edge(edge);
                for (&v, &param) in args.iter().zip(self.params(target)) {
                    used(v, usize::MAX)?;
                    if self.value_type(v) != self.value_type(param) {
                        return Err(IrError::EdgeArgumentType {
                            edge,
                            value: v,
                            expected: self.value_type(param),
                            actual: self.value_type(v),
                        });
                    }
                }
            }
        }
        Ok(cfg)
    }
}

impl Function {
    pub(crate) fn verified_module(
        &self,
    ) -> Result<(Module<VerifiedBody<'_>>, FunctionIds<Self>), CompileError> {
        Module::discover(self, |function, callees| {
            let cfg = function.verified_cfg()?;
            for inst in function.insts.values() {
                if let Op::Call(site) = inst.op {
                    callees.push(function.calls[site].callee.as_ref());
                }
            }
            Ok(VerifiedBody { function, cfg })
        })
    }
}

#[cfg(test)]
impl Function {
    pub fn interpret(&self, args: &[i32]) -> Result<i32, String> {
        if self.param_types().iter().any(|&ty| ty != Type::I32) || self.return_type != Type::I32 {
            return Err("expected an i32-only signature".into());
        }
        let args: Vec<_> = args.iter().map(|&x| x as u32 as u64).collect();
        // SAFETY: no pointer can enter a verified function through this signature.
        unsafe { self.interpret_raw(&args) }.map(|v| v as i32)
    }

    /// # Safety
    /// Pointer arguments and addresses read from memory must be valid for all
    /// accesses executed by the function and its callees, including writes.
    pub(crate) unsafe fn interpret_raw(&self, args: &[u64]) -> Result<u64, String> {
        if args.len() != self.arity() {
            return Err("wrong argument count".into());
        }
        self.verified_module().map_err(|error| error.to_string())?;
        // SAFETY: the caller supplies every pointer reachable during execution.
        unsafe { interpret::run(self, args, None) }
    }

    /// Execute a filetest with one buffer. Its pointer is represented by a
    /// byte offset so an invalid fixture is rejected before entering the JIT.
    pub(crate) fn interpret_buffer(&self, args: &[i32], buffer: &mut [u8]) -> Result<i32, String> {
        let Some((Type::Ptr, scalars)) = self.param_types().split_first() else {
            return Err("expected leading buffer argument".into());
        };
        if self.return_type != Type::I32
            || scalars.len() != args.len()
            || scalars.iter().any(|&t| t != Type::I32)
        {
            return Err("expected (ptr, i32...) -> i32 matching the run arguments".into());
        }
        self.verified_module().map_err(|e| e.to_string())?;
        let words: Vec<_> = std::iter::once(0)
            .chain(args.iter().map(|&x| u64::from(x as u32)))
            .collect();
        // SAFETY: buffer mode services every load/store through checked slice accesses.
        unsafe { interpret::run(self, &words, Some(buffer)) }.map(|v| v as i32)
    }
}
