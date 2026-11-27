//! Typed SSA with block parameters. IDs are stable across instruction rewrites
//! and block merges. Instruction order lives in the layout, independently of IDs.

#[cfg(test)]
mod interpret;
#[cfg(test)]
pub(crate) use interpret::{InterpretError, buffer_address};
mod verify;

pub(crate) use verify::VerifiedBody;

use super::{Op, Terminator, Type, Value};
use crate::hir::IrError;
use crate::support::slotmap::{DenseMap, SlotMap};
use std::{ops::Range, sync::Arc};

crate::support::slotmap::slot_key!(
    /// A basic block.
    pub Block
);

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

crate::support::slotmap::slot_key!(
    /// An instruction.
    pub Inst
);

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

crate::support::slotmap::slot_key!(
    /// A call site, recorded by [`Function::add_call`].
    pub Call
);
#[derive(Clone, Debug)]
struct CallData {
    callee: Arc<Function>,
    args: Span,
}

crate::support::slotmap::slot_key!(
    /// A control-flow edge with its arguments, recorded by [`Function::add_edge`].
    pub Edge
);

/// An SSA function under construction or verified by [`Function::verify`].
///
/// # Panics
///
/// `Block`, `Value`, `Inst`, `Edge` and `Call` handles index
/// the function that issued them. Every method that takes a handle panics
/// for a handle of another function or a block retired by
/// [`Function::merge_blocks`], and [`Function::add_edge`] and
/// [`Function::add_call`] panic past 2^32 operands in all. Ill-typed
/// operations are rejected by [`Function::append`] and structural errors
/// that a builder can make with valid handles by `verify`, both as an
/// [`IrError`].
///
/// Each kind of handle has its own type, so a block cannot stand for a value:
///
/// ```
/// # use lancy::hir::Function;
/// let mut f = Function::scalar(1);
/// let block = f.create_block();
/// assert!(f.params(block).is_empty());
/// ```
///
/// ```compile_fail
/// # use lancy::hir::Function;
/// let f = Function::scalar(1);
/// let value = f.params(f.entry())[0];
/// f.params(value);
/// ```
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
    slices: Vec<SliceParam>,
    /// The position in `slices` of each pointer and length parameter.
    slice_of: DenseMap<Value, Option<u32>>,
}

/// A slice that the caller passes as a pointer and an `i32` length entry
/// parameter, with `element` bytes per element. This is part of the raw call
/// contract: `len` elements are initialized, writable bytes are exclusively
/// borrowed, and shared bytes stay immutable throughout the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceParam {
    /// The pointer parameter.
    pub ptr: Value,
    /// The length parameter, in elements.
    pub len: Value,
    /// Bytes per element.
    pub element: u8,
    /// Whether the kernel may write the slice.
    pub writable: bool,
}

impl Function {
    /// Create a function with typed parameters in ABI order.
    pub fn new(params: &[Type], return_type: Type) -> Self {
        Self::from_types(params.iter().copied(), return_type)
    }

    /// A function of `arity` `i32` parameters and an `i32` result.
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
            slices: Vec::new(),
            slice_of: DenseMap::new(),
        };
        let entry = f.create_block();
        for ty in params {
            f.append_typed_param(entry, ty);
        }
        f
    }

    /// Number of parameters.
    pub fn arity(&self) -> usize {
        self.arity
    }
    /// Parameter types in ABI order.
    pub fn param_types(&self) -> &[Type] {
        &self.types.as_slice()[..self.arity]
    }
    /// The result type.
    pub fn return_type(&self) -> Type {
        self.return_type
    }
    /// The type of a parameter or result.
    pub fn value_type(&self, value: Value) -> Type {
        self.types[value]
    }

    /// Declare the slice parameters of the raw call contract (see
    /// [`SliceParam`]). Each is a distinct `(Ptr, I32)` pair of entry
    /// parameters, with elements of 2, 4 or 8 bytes: the element widths of
    /// the buffers Lancy's bounds proofs model. A rejected declaration
    /// leaves the previous one.
    pub fn set_slice_params(&mut self, slices: Vec<SliceParam>) -> Result<(), IrError> {
        let entry = self.params(self.entry());
        let mut slice_of = DenseMap::filled(self.value_count(), None);
        for (i, slice) in slices.iter().enumerate() {
            let is_param = |v: Value, ty| entry.contains(&v) && self.types[v] == ty;
            if !is_param(slice.ptr, Type::Ptr)
                || !is_param(slice.len, Type::I32)
                || !matches!(slice.element, 2 | 4 | 8)
                || slice_of[slice.ptr].is_some()
                || slice_of[slice.len].is_some()
            {
                return Err(IrError::InvalidSliceParam { slice: *slice });
            }
            slice_of[slice.ptr] = Some(i as u32);
            slice_of[slice.len] = Some(i as u32);
        }
        self.slices = slices;
        self.slice_of = slice_of;
        Ok(())
    }

    pub(crate) fn copy_slice_params(&mut self, source: &Self) {
        self.slices.clone_from(&source.slices);
        self.slice_of.clone_from(&source.slice_of);
    }

    fn slice(&self, value: Value) -> Option<&SliceParam> {
        let index = (*self.slice_of.get(value)?)?;
        Some(&self.slices[index as usize])
    }

    /// The declared slices, in declaration order.
    pub fn slice_params(&self) -> &[SliceParam] {
        &self.slices
    }

    /// The declared slice whose pointer is `ptr`.
    pub fn slice_param(&self, ptr: Value) -> Option<SliceParam> {
        self.slice(ptr).filter(|slice| slice.ptr == ptr).copied()
    }

    pub(in crate::hir) fn has_slice_params(&self) -> bool {
        !self.slices.is_empty()
    }

    pub(in crate::hir) fn has_disjoint_slice_params(&self) -> bool {
        self.slices.len() > 1 && self.slices.iter().any(|slice| slice.writable)
    }

    pub(in crate::hir) fn is_slice_length(&self, value: Value) -> bool {
        self.slice(value).is_some_and(|slice| slice.len == value)
    }

    /// Replace an operation without verifying the function: the passes'
    /// editor. Debug builds check that the new operation keeps the type of
    /// the result; `verify` checks the whole function after a pass.
    pub(in crate::hir) fn replace_op_preserving_type(&mut self, inst: Inst, op: Op) {
        self.insts[inst].op = op;
        self.debug_assert_typed(inst);
    }

    fn debug_assert_typed(&self, inst: Inst) {
        if cfg!(debug_assertions) {
            let ty = self
                .op(inst)
                .result_type(self, inst, |v| Ok(self.types[v]))
                .unwrap_or_else(|error| panic!("pass edit of {inst:?}: {error}"));
            assert_eq!(
                ty,
                self.result(inst).map(|v| self.types[v]),
                "pass edit of {inst:?} changes its result type"
            );
        }
    }

    pub(in crate::hir) fn disjoint_buffer_params(&self, a: Value, b: Value) -> bool {
        a != b
            && self
                .slice_param(a)
                .zip(self.slice_param(b))
                .is_some_and(|(a, b)| a.writable || b.writable)
    }
    /// The entry block; its parameters are the function's.
    pub fn entry(&self) -> Block {
        Block::from_index(0)
    }
    /// Number of value IDs issued.
    pub fn value_count(&self) -> usize {
        self.types.len()
    }
    /// The type of every value, indexed by `Value`.
    pub(crate) fn value_types(&self) -> &[Type] {
        self.types.as_slice()
    }
    /// Number of block IDs issued, including blocks retired by a merge.
    pub fn block_capacity(&self) -> usize {
        self.blocks.slot_count()
    }
    /// The live blocks in ID order.
    pub fn blocks(&self) -> impl Iterator<Item = Block> + '_ {
        self.blocks.keys()
    }
    /// The parameters of `b`.
    pub fn params(&self, b: Block) -> &[Value] {
        &self.blocks[b].params
    }
    /// The column of `value` among the parameters of `b`. Blocks have few
    /// parameters, and their IDs need not be in any order.
    pub fn param_index(&self, b: Block, value: Value) -> Option<usize> {
        self.params(b).iter().position(|&param| param == value)
    }
    /// The instructions of `b` in order.
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
    /// The operation of an instruction.
    pub fn op(&self, i: Inst) -> &Op {
        &self.insts[i].op
    }
    /// The value an instruction defines, if any.
    pub fn result(&self, i: Inst) -> Option<Value> {
        let result = self.insts[i].result;
        (result != Value::INVALID).then_some(result)
    }
    pub(crate) fn visit_operands(&self, i: Inst, mut visit: impl FnMut(Value)) {
        let mut op = *self.op(i);
        match op {
            Op::Call(site) => self.call(site).1.iter().copied().for_each(visit),
            _ => op.map_operands(|v| {
                visit(v);
                v
            }),
        }
    }
    /// # Panics
    ///
    /// If `b` has no terminator yet. A verified function terminates every block.
    pub fn terminator(&self, b: Block) -> &Terminator {
        self.blocks[b]
            .term
            .as_ref()
            .expect("unterminated SSA block")
    }
    /// The target and arguments of an edge.
    pub fn edge(&self, edge: Edge) -> (Block, &[Value]) {
        let (block, args) = self.edges[edge];
        (block, &self.operands[args.range()])
    }

    /// Operands held by the arena, and those that live edge and call
    /// records reference.
    #[cfg(test)]
    pub(crate) fn operand_arena(&self) -> (usize, usize) {
        let edges = self
            .blocks()
            .flat_map(|b| self.terminator(b).edges())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|e| self.edge(e).1.len());
        let calls = self.insts.values().map(|inst| match inst.op {
            Op::Call(call) => self.call(call).1.len(),
            _ => 0,
        });
        (self.operands.len(), edges.chain(calls).sum())
    }

    pub(in crate::hir) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Filter non-entry parameters and the corresponding column of every edge.
    /// Filter non-entry parameters and the corresponding column of every edge.
    pub(in crate::hir) fn retain_block_params(&mut self, keep: impl Fn(Value) -> bool) {
        // After compaction a live edge record's arguments match its target's
        // parameters. A record shared by several terminators is filtered once.
        self.compact();
        let entry = self.entry();
        for (target, span) in self.edges.values_mut() {
            if *target == Block::INVALID || *target == entry {
                continue;
            }
            debug_assert_eq!(span.len as usize, self.blocks[*target].params.len());
            let block = &self.blocks[*target];
            let args = &mut self.operands[span.range()];
            let mut kept = 0;
            for (i, &param) in block.params.iter().enumerate() {
                if keep(param) {
                    args[kept] = args[i];
                    kept += 1;
                }
            }
            span.len = kept as u32;
        }
        for block in self.blocks.values_mut().skip(1) {
            block.params.retain(|&p| keep(p));
        }
    }

    /// Rebuild the operand arena from the edge and call records that a
    /// terminator or instruction references. Unreferenced edge records are
    /// retired: their target becomes `Block::INVALID` and their
    /// arguments empty. Every ID stays valid, so analyses that name edges
    /// remain usable.
    pub(in crate::hir) fn compact(&mut self) {
        let old = std::mem::take(&mut self.operands);
        let mut referenced = DenseMap::filled(self.edges.len(), false);
        for block in self.blocks.values() {
            for edge in block.term.into_iter().flat_map(Terminator::edges) {
                referenced[edge] = true;
            }
        }
        for (edge, (target, span)) in self.edges.iter_mut() {
            if referenced[edge] {
                *span = Span::append(&mut self.operands, &old[span.range()]);
            } else {
                *target = Block::INVALID;
                *span = Span { start: 0, len: 0 };
            }
        }
        let mut calls = DenseMap::filled(self.calls.len(), false);
        for inst in self.insts.values() {
            if let Op::Call(call) = inst.op {
                calls[call] = true;
            }
        }
        for (call, data) in self.calls.iter_mut() {
            let live = if calls[call] {
                &old[data.args.range()]
            } else {
                &[]
            };
            data.args = Span::append(&mut self.operands, live);
        }
    }

    pub(in crate::hir) fn call_site(&self, call: Call) -> Option<(&Arc<Function>, &[Value])> {
        let data = self.calls.get(call)?;
        Some((&data.callee, &self.operands[data.args.range()]))
    }

    /// The callee and arguments of a call.
    pub fn call(&self, call: Call) -> (&Arc<Function>, &[Value]) {
        let data = &self.calls[call];
        (&data.callee, &self.operands[data.args.range()])
    }

    /// A new empty block without parameters.
    pub fn create_block(&mut self) -> Block {
        self.blocks.insert(BlockData::default())
    }

    /// Append an `i32` parameter to `block`.
    pub fn append_param(&mut self, block: Block) -> Value {
        self.append_typed_param(block, Type::I32)
    }
    /// Append a parameter of type `ty` to `block`.
    pub fn append_typed_param(&mut self, block: Block, ty: Type) -> Value {
        let params = &mut self.blocks[block].params;
        let value = self.types.insert(ty);
        params.push(value);
        value
    }
    /// Append `op` to `b` after checking its operand types.
    pub fn append_inst(&mut self, b: Block, op: Op) -> Result<Inst, IrError> {
        let id = self.new_instruction(op)?;
        let last = self.blocks[b].last;
        if last == Inst::INVALID {
            self.blocks[b].first = id;
        } else {
            self.insts[last].next = id;
        }
        self.blocks[b].last = id;
        Ok(id)
    }

    /// Append a value-defining `op` to `b` and return its result.
    pub fn append(&mut self, b: Block, op: Op) -> Result<Value, IrError> {
        let inst = self.append_inst(b, op)?;
        self.result(inst).ok_or(IrError::NoResult { inst })
    }

    /// Insert `op` after `after` in `b`, or first when `after` is `None`.
    /// Its operands must be available at that point.
    pub(in crate::hir) fn insert_after(
        &mut self,
        b: Block,
        after: Option<Inst>,
        op: Op,
    ) -> Result<Inst, IrError> {
        let id = self.new_instruction(op)?;
        let block = &mut self.blocks[b];
        let next = match after {
            Some(prev) => std::mem::replace(&mut self.insts[prev].next, id),
            None => std::mem::replace(&mut block.first, id),
        };
        self.insts[id].next = next;
        if next == Inst::INVALID {
            block.last = id;
        }
        Ok(id)
    }

    fn new_instruction(&mut self, op: Op) -> Result<Inst, IrError> {
        // Instruction IDs are issued in order and never reused.
        let inst = Inst::from_index(self.insts.slot_count());
        let ty = op.result_type(self, inst, |v| Ok(self.types[v]))?;
        debug_assert_eq!(ty.is_some(), op.has_result());
        let result = ty.map_or(Value::INVALID, |ty| self.types.insert(ty));
        let id = self.insts.insert(InstData {
            op,
            result,
            next: Inst::INVALID,
        });
        debug_assert_eq!(id, inst);
        Ok(id)
    }

    /// Record a call of `callee` with `args`, for an [`Op::Call`].
    pub fn add_call(&mut self, callee: Arc<Function>, args: &[Value]) -> Call {
        let args = Span::append(&mut self.operands, args);
        self.calls.insert(CallData { callee, args })
    }

    /// Record an edge to `target` passing `args` to its parameters.
    pub fn add_edge(&mut self, target: Block, args: &[Value]) -> Edge {
        let args = Span::append(&mut self.operands, args);
        self.edges.insert((target, args))
    }
    /// End `b` with `term`, replacing any earlier terminator.
    pub fn set_terminator(&mut self, b: Block, term: Terminator) {
        self.blocks[b].term = Some(term);
    }

    /// End `from` with a jump to `to` passing `args`.
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
            if old.op.has_result() != op.has_result() {
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
    /// references in dominated blocks, calls and successor edges. Only the
    /// two blocks and their edge are checked; `verify` checks the result.
    pub fn merge_blocks(&mut self, pred: Block, succ: Block) -> Result<(), IrError> {
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
        let mut incoming = 0;
        for b in self.blocks() {
            incoming += self
                .checked_terminator(b)?
                .edges()
                .filter(|&e| self.edges[e].0 == succ)
                .count();
        }
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
            self.substitute_values(&aliases);
        }
        Ok(())
    }

    /// Splice every block whose only incoming edge is a jump into the
    /// jumping block, as [`Self::merge_blocks`] does, in one pass: chains of
    /// straight-line blocks become one block. Returns whether any merged.
    pub(in crate::hir) fn merge_jump_chains(&mut self) -> bool {
        let entry = self.entry();
        let mut incoming = DenseMap::filled(self.block_capacity(), 0u32);
        let mut merged = false;
        for data in self.blocks.values() {
            if let Some(term) = &data.term {
                for e in term.edges() {
                    incoming[self.edges[e].0] += 1;
                }
            }
        }
        let mut aliases: Option<DenseMap<Value, Value>> = None;
        let blocks: Vec<Block> = self.blocks().collect();
        for pred in blocks {
            while let Some(Some(Terminator::Jump(edge))) =
                self.blocks.get(pred).map(|data| data.term)
            {
                let (succ, args) = self.edges[edge];
                if succ == pred || succ == entry || incoming[succ] != 1 {
                    break;
                }
                let succ = self.blocks.remove(succ).unwrap();
                merged = true;
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
                    let types = self.types.len();
                    let aliases = aliases.get_or_insert_with(|| DenseMap::from_fn(types, |v| v));
                    for (&param, &actual) in succ.params.iter().zip(&self.operands[args.range()]) {
                        aliases[param] = actual;
                    }
                }
            }
        }
        if let Some(mut aliases) = aliases {
            // An argument can be a parameter of a block merged before or
            // after: follow each chain to a value that stays.
            for v in aliases.keys() {
                let mut root = aliases[v];
                while aliases[root] != root {
                    root = aliases[root];
                }
                aliases[v] = root;
            }
            self.substitute_values(&aliases);
        }
        merged
    }

    pub(in crate::hir) fn substitute_values(&mut self, aliases: &DenseMap<Value, Value>) {
        let map = |v: Value| aliases[v];
        for inst in self.insts.values_mut() {
            inst.op.map_operands(map);
        }
        for v in &mut self.operands {
            // Unreferenced call/edge records can hold values created after
            // `aliases`.
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

    pub(in crate::hir) fn checked_terminator(&self, b: Block) -> Result<Terminator, IrError> {
        self.blocks[b]
            .term
            .ok_or(IrError::UnterminatedBlock { block: b })
    }

    pub(in crate::hir) fn checked_target(&self, from: Block, edge: Edge) -> Result<Block, IrError> {
        let &(target, _) = self
            .edges
            .get(edge)
            .ok_or(IrError::UnknownEdge { from, edge })?;
        self.blocks
            .get(target)
            .ok_or(IrError::UnknownTarget { from, edge, target })?;
        Ok(target)
    }

    pub(in crate::hir) fn remove_block(&mut self, block: Block) -> bool {
        let Some(dead) = self.blocks.remove(block) else {
            return false;
        };
        let mut cursor = dead.first;
        while cursor != Inst::INVALID {
            cursor = self.insts.remove(cursor).unwrap().next;
        }
        true
    }

    /// Rewrite a block in layout order, preserving the types and IDs of surviving results.
    /// The caller removes uses of any retired values.
    pub(in crate::hir) fn rewrite_instructions(
        &mut self,
        block: Block,
        mut rewrite: impl FnMut(Op, Option<Value>, &DenseMap<Value, Type>) -> Option<Op>,
    ) {
        let id = block;
        let block = &mut self.blocks[block];
        let mut prev = Inst::INVALID;
        let mut i = block.first;
        while i != Inst::INVALID {
            let inst = &mut self.insts[i];
            let next = inst.next;
            if let Some(op) = rewrite(
                inst.op,
                (inst.result != Value::INVALID).then_some(inst.result),
                &self.types,
            ) {
                inst.op = op;
                prev = i;
            } else {
                if prev == Inst::INVALID {
                    block.first = next;
                } else {
                    self.insts[prev].next = next;
                }
                self.insts.remove(i);
            }
            i = next;
        }
        block.last = prev;
        if cfg!(debug_assertions) {
            let rewritten: Vec<_> = self.instructions(id).collect();
            for inst in rewritten {
                self.debug_assert_typed(inst);
            }
        }
    }

    /// Move instructions to the ends of earlier dominating blocks.
    /// The caller proves operand availability and that speculation is safe.
    pub(in crate::hir) fn hoist_instructions(
        &mut self,
        from: Block,
        mut destination: impl FnMut(Op, Value, &DenseMap<Value, Type>) -> Option<(Block, Op)>,
    ) {
        let mut prev = Inst::INVALID;
        let mut i = self.blocks[from].first;
        while i != Inst::INVALID {
            let inst = &self.insts[i];
            let next = inst.next;
            if inst.result != Value::INVALID
                && let Some((to, op)) = destination(inst.op, inst.result, &self.types)
            {
                debug_assert_ne!(from, to);
                if prev == Inst::INVALID {
                    self.blocks[from].first = next;
                } else {
                    self.insts[prev].next = next;
                }
                let target = &mut self.blocks[to];
                if target.last == Inst::INVALID {
                    target.first = i;
                } else {
                    self.insts[target.last].next = i;
                }
                target.last = i;
                self.insts[i].op = op;
                self.insts[i].next = Inst::INVALID;
                self.debug_assert_typed(i);
            } else {
                prev = i;
            }
            i = next;
        }
        self.blocks[from].last = prev;
    }

    pub(in crate::hir) fn calls_mut(&mut self) -> impl Iterator<Item = (Call, &mut Arc<Function>)> {
        self.calls
            .iter_mut()
            .map(|(call, site)| (call, &mut site.callee))
    }

    /// Scratch bytes addressed by this frame's own operations, rounded up so
    /// that a region placed above them stays 8-byte aligned.
    pub(crate) fn scratch_bytes(&self) -> u64 {
        self.frame_extent(|op| match op {
            Op::CallScratch { offset, bytes } => Some(u64::from(offset) + u64::from(bytes)),
            _ => None,
        })
    }

    /// Debugger workspace bytes written by this frame at any vector width,
    /// rounded up like [`Self::scratch_bytes`].
    pub(crate) fn debug_bytes(&self) -> u64 {
        self.frame_extent(|op| match op {
            Op::DebugStore(offset, value) => {
                let bytes = self.types[value].bytes(crate::hir::MAX_VECTOR_BYTES);
                Some(u64::from(offset) + bytes as u64)
            }
            _ => None,
        })
    }

    fn frame_extent(&self, end: impl Fn(Op) -> Option<u64>) -> u64 {
        self.blocks()
            .flat_map(|block| self.instructions(block))
            .filter_map(|inst| end(self.insts[inst].op))
            .max()
            .unwrap_or(0)
            .next_multiple_of(8)
    }
}

#[cfg(test)]
impl Function {
    pub(crate) fn interpret(&self, args: &[i32]) -> Result<i32, InterpretError> {
        if self.param_types().iter().any(|&ty| ty != Type::I32) || self.return_type != Type::I32 {
            return Err(InterpretError::Unsupported(
                "a signature that is not all i32",
            ));
        }
        let args: Vec<_> = args.iter().map(|&x| x as u32 as u64).collect();
        // SAFETY: no pointer can enter a verified function through this signature.
        unsafe { self.interpret_raw(&args) }.map(|v| v as i32)
    }

    /// # Safety
    /// Pointer arguments and addresses read from memory must be valid for all
    /// accesses executed by the function and its callees, including writes.
    pub(crate) unsafe fn interpret_raw(&self, args: &[u64]) -> Result<u64, InterpretError> {
        if args.len() != self.arity() {
            return Err(crate::CallError::Signature(crate::SignatureError::Arity {
                expected: self.arity(),
                actual: args.len(),
            })
            .into());
        }
        let width = crate::test_target::width();
        if self.param_types().iter().any(|ty| ty.bytes(width) > 8)
            || self.return_type.bytes(width) > 8
        {
            return Err(InterpretError::Unsupported(
                "a root signature wider than a word",
            ));
        }
        self.verified_module()
            .map_err(|error| InterpretError::Invalid(error.to_string()))?;
        let args = args.iter().map(|&x| interpret::word(x)).collect();
        // SAFETY: the caller supplies every pointer reachable during execution.
        let result = unsafe { interpret::run(self, args, None) }?;
        Ok(u64::from_le_bytes(result[..8].try_into().unwrap()))
    }

    /// Execute a filetest. Each pointer argument is an
    /// [`interpret::buffer_address`] into `buffers`, so an invalid fixture is
    /// rejected before entering the JIT.
    pub(crate) fn interpret_buffers(
        &self,
        args: &[crate::support::numeric::Bits],
        buffers: &mut [Vec<u8>],
    ) -> Result<crate::support::numeric::Bits, InterpretError> {
        if args.len() != self.arity() {
            return Err(crate::CallError::Signature(crate::SignatureError::Arity {
                expected: self.arity(),
                actual: args.len(),
            })
            .into());
        }
        let params = self.params(self.entry());
        let word = |i: usize| u64::from_le_bytes(args[i][..8].try_into().unwrap());
        for (i, &ty) in self.param_types().iter().enumerate() {
            if ty != Type::Ptr {
                continue;
            }
            let buffer = (0..buffers.len())
                .find(|&k| interpret::buffer_address(k) == word(i))
                .ok_or(InterpretError::Unsupported(
                    "a pointer that is not a run buffer",
                ))?;
            if let Some(slice) = self.slice_param(params[i]) {
                let len = params.iter().position(|&param| param == slice.len).unwrap();
                let len = word(len) as u32 as i32;
                if len < 0 || len as usize > buffers[buffer].len() / usize::from(slice.element) {
                    return Err(InterpretError::OutOfBuffer {
                        buffer,
                        offset: buffers[buffer].len() as u64,
                    });
                }
            }
        }
        self.verified_module()
            .map_err(|error| InterpretError::Invalid(error.to_string()))?;
        if self.param_types().contains(&Type::Ptr) {
            // SAFETY: buffer mode services every load/store through checked accesses.
            unsafe { interpret::run(self, args.to_vec(), Some(buffers)) }
        } else {
            // SAFETY: no pointer enters a verified function without pointer
            // parameters; its only addresses come from CallScratch inside a
            // live interpreter frame.
            unsafe { interpret::run(self, args.to_vec(), None) }
        }
    }
}
