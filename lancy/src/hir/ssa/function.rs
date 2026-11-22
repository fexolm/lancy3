//! Typed SSA with block parameters. IDs are stable across instruction rewrites
//! and block merges. Instruction order lives in the layout, independently of IDs.

#[cfg(test)]
mod interpret;
mod verify;

pub(crate) use verify::VerifiedBody;

use super::{Op, RustPacketImport, Terminator, Type, Value};
use crate::hir::IrError;
use crate::support::slotmap::{DenseMap, SlotMap};
use std::{ops::Range, sync::Arc};

crate::slot_key!(pub Block);

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
crate::slot_key!(pub PacketCall);

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
    packets: DenseMap<PacketCall, (&'static RustPacketImport, Span)>,
    operands: Vec<Value>,
    types: DenseMap<Value, Type>,
    slice_base: usize,
    slices: Box<[(u8, bool)]>,
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
            packets: DenseMap::new(),
            operands: Vec::new(),
            types: DenseMap::with_capacity(arity),
            slice_base: arity,
            slices: Box::new([]),
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

    /// Declare consecutive (pointer, i32 length) parameters as initialized
    /// slices, given the (element byte width, writable) pair for each.
    /// This is part of the raw call contract. Writable bytes are exclusively
    /// borrowed and shared bytes stay immutable throughout the call.
    pub fn set_slice_params(&mut self, first: usize, slices: Box<[(u8, bool)]>) {
        let params = &self.param_types()[first..first + 2 * slices.len()];
        assert!(
            params
                .chunks_exact(2)
                .all(|pair| pair == [Type::Ptr, Type::I32])
        );
        assert!(slices.iter().all(|&(bytes, _)| matches!(bytes, 2 | 4 | 8)));
        self.slice_base = first;
        self.slices = slices;
    }

    pub(crate) fn copy_slice_params(&mut self, source: &Self) {
        self.slice_base = source.slice_base;
        self.slices.clone_from(&source.slices);
    }

    /// For a declared slice pointer, return its length parameter, element byte width
    /// and whether the caller grants exclusive writable access.
    pub fn slice_param(&self, ptr: Value) -> Option<(Value, u8, bool)> {
        let offset = ptr.index().checked_sub(self.slice_base)?;
        if !offset.is_multiple_of(2) {
            return None;
        }
        let &(bytes, writable) = self.slices.get(offset / 2)?;
        Some((Value::from_index(ptr.index() + 1), bytes, writable))
    }

    pub(in crate::hir) fn has_slice_params(&self) -> bool {
        !self.slices.is_empty()
    }

    pub(in crate::hir) fn has_disjoint_slice_params(&self) -> bool {
        self.slices.len() > 1 && self.slices.iter().any(|&(_, writable)| writable)
    }

    pub(in crate::hir) fn is_slice_length(&self, value: Value) -> bool {
        value
            .index()
            .checked_sub(1)
            .and_then(|index| self.slice_param(Value::from_index(index)))
            .is_some_and(|(len, _, _)| len == value)
    }

    pub(in crate::hir) fn replace_op_preserving_type(&mut self, inst: Inst, op: Op) {
        self.insts[inst].op = op;
    }

    pub(in crate::hir) fn disjoint_buffer_params(&self, a: Value, b: Value) -> bool {
        a != b
            && self
                .slice_param(a)
                .zip(self.slice_param(b))
                .is_some_and(|((_, _, write_a), (_, _, write_b))| write_a || write_b)
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
        match op {
            Op::Call(site) => self.call(site).1.iter().copied().for_each(visit),
            Op::RustPacketCall(site) => self.packet_call(site).1.iter().copied().for_each(visit),
            _ => op.map_operands(|v| {
                visit(v);
                v
            }),
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

    pub(in crate::hir) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Filter non-entry parameters and the corresponding column of every edge.
    pub(in crate::hir) fn retain_block_params(&mut self, keep: impl Fn(Value) -> bool) {
        // Include retired edges only when their spans still describe current parameters.
        // An edge can be referenced by multiple terminators, so compact each record once.
        let entry = self.entry();
        for (target, span) in self.edges.values_mut() {
            if *target == entry {
                continue;
            }
            let Some(block) = self.blocks.get(*target) else {
                continue;
            };
            if span.len as usize != block.params.len() {
                continue;
            }
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
    pub fn call(&self, call: Call) -> (&Arc<Function>, &[Value]) {
        let data = &self.calls[call];
        (&data.callee, &self.operands[data.args.range()])
    }

    pub fn packet_call(&self, call: PacketCall) -> (&'static RustPacketImport, &[Value]) {
        let (import, args) = self.packets[call];
        (import, &self.operands[args.range()])
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
        let id = self.new_instruction(op);
        let last = self.blocks[b].last;
        if last == Inst::INVALID {
            self.blocks[b].first = id;
        } else {
            self.insts[last].next = id;
        }
        self.blocks[b].last = id;
        id
    }

    /// Insert before the first instruction. Operands must already dominate
    /// the start of the block.
    pub(in crate::hir) fn prepend(&mut self, b: Block, op: Op) -> Option<Value> {
        let id = self.new_instruction(op);
        let block = &mut self.blocks[b];
        self.insts[id].next = block.first;
        if block.last == Inst::INVALID {
            block.last = id;
        }
        block.first = id;
        self.result(id)
    }

    fn new_instruction(&mut self, op: Op) -> Inst {
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
                Op::PtrOffset(..) | Op::CallScratch(_) => Type::Ptr,
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
        self.insts.insert(InstData {
            op,
            result,
            next: Inst::INVALID,
        })
    }
    pub fn append(&mut self, b: Block, op: Op) -> Option<Value> {
        let id = self.append_inst(b, op);
        self.result(id)
    }
    pub fn add_call(&mut self, callee: Arc<Function>, args: &[Value]) -> Call {
        let args = Span::append(&mut self.operands, args);
        self.calls.insert(CallData { callee, args })
    }

    pub fn add_packet_call(
        &mut self,
        import: &'static RustPacketImport,
        args: &[Value],
    ) -> PacketCall {
        let args = Span::append(&mut self.operands, args);
        self.packets.insert((import, args))
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
            self.substitute_values(&aliases);
        }
        Ok(())
    }

    pub(in crate::hir) fn substitute_values(&mut self, aliases: &DenseMap<Value, Value>) {
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
            } else {
                prev = i;
            }
            i = next;
        }
        self.blocks[from].last = prev;
    }

    pub(in crate::hir) fn callees_mut(&mut self) -> impl Iterator<Item = &mut Arc<Function>> {
        self.calls.values_mut().map(|site| &mut site.callee)
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
        if let Some((_, width, _)) = self.slice_param(Value::from_index(0))
            && (args[0] < 0 || args[0] as usize > buffer.len() / usize::from(width))
        {
            return Err("declared slice extends beyond the run buffer".into());
        }
        self.verified_module().map_err(|e| e.to_string())?;
        let words: Vec<_> = std::iter::once(0)
            .chain(args.iter().map(|&x| u64::from(x as u32)))
            .collect();
        // SAFETY: buffer mode services every load/store through checked slice accesses.
        unsafe { interpret::run(self, &words, Some(buffer)) }.map(|v| v as i32)
    }
}
