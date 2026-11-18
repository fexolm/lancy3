//! Scalar SSA with block parameters. IDs are stable across instruction rewrites
//! and block merges. Instruction order lives in the layout, independently of IDs.

mod dominance;
mod edit;
mod import;
mod interpret;
mod parameters;
pub use import::from_input;
#[cfg(test)]
mod tests;
mod verify;

use crate::hir::{Binary, Buffer, BufferKind};
use std::ops::Range;
use std::sync::Arc;

macro_rules! entity {
    ($($name:ident),+) => {$(
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        pub struct $name(pub u32);

        impl $name {
            pub fn index(self) -> usize { self.0 as usize }
        }
    )+};
}
entity!(Block, Inst, Value, Edge, Call);

const END: Inst = Inst(u32::MAX);

fn next_id(index: usize) -> u32 {
    let id = u32::try_from(index).expect("SSA entity index exceeds 32 bits");
    assert_ne!(id, u32::MAX, "exhausted SSA entity indices");
    id
}

/// Variable-sized operands live in side tables, keeping common operations inline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    Const(i32),
    Copy(Value),
    Bin(Binary, Value, Value),
    BufferLen(Buffer),
    Load(Buffer, Value),
    Store(Buffer, Value, Value),
    Call(Call),
}

impl Op {
    fn produces_value(self) -> bool {
        !matches!(self, Self::Store(..))
    }

    fn map_operands(&mut self, mut map: impl FnMut(Value) -> Value) {
        match self {
            Self::Copy(v) | Self::Load(_, v) => *v = map(*v),
            Self::Bin(_, a, b) | Self::Store(_, a, b) => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::Const(_) | Self::BufferLen(_) | Self::Call(_) => {}
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Terminator {
    Jump(Edge),
    Branch { cond: Value, yes: Edge, no: Edge },
    Return(Value),
}

impl Terminator {
    pub fn edges(self) -> impl Iterator<Item = Edge> {
        let empty = Edge(0);
        let (edges, count) = match self {
            Self::Jump(edge) => ([edge, empty], 1),
            Self::Branch { yes, no, .. } => ([yes, no], 2),
            Self::Return(_) => ([empty, empty], 0),
        };
        edges.into_iter().take(count)
    }
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
            first: END,
            last: END,
            term: None,
        }
    }
}

#[derive(Clone, Debug)]
struct InstData {
    op: Op,
    // u32::MAX for instructions with no result.
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

#[derive(Clone, Debug)]
struct CallData {
    callee: Arc<Function>,
    args: Span,
    buffers: Span,
}

#[derive(Clone, Debug)]
pub struct Function {
    arity: usize,
    buffers: Vec<BufferKind>,
    blocks: Vec<Option<BlockData>>,
    insts: Vec<InstData>,
    edges: Vec<(Block, Span)>,
    calls: Vec<CallData>,
    operands: Vec<Value>,
    buffer_operands: Vec<Buffer>,
    values: u32,
    fallible_ops: usize,
}

impl Function {
    /// The entry block has one parameter per scalar argument in ABI order.
    pub fn new(arity: usize, buffers: Vec<BufferKind>) -> Self {
        let mut f = Self {
            arity,
            buffers,
            blocks: vec![Some(BlockData::default())],
            insts: Vec::new(),
            edges: Vec::new(),
            calls: Vec::new(),
            operands: Vec::new(),
            buffer_operands: Vec::new(),
            values: 0,
            fallible_ops: 0,
        };
        for _ in 0..arity {
            f.append_param(f.entry());
        }
        f
    }

    pub fn arity(&self) -> usize {
        self.arity
    }
    pub fn buffer_kinds(&self) -> &[BufferKind] {
        &self.buffers
    }
    pub fn needs_status(&self) -> bool {
        !self.buffers.is_empty() || self.fallible_ops != 0
    }
    pub fn entry(&self) -> Block {
        Block(0)
    }
    pub fn value_count(&self) -> usize {
        self.values as usize
    }
    /// Number of block IDs issued, including blocks retired by a merge.
    pub fn block_capacity(&self) -> usize {
        self.blocks.len()
    }
    pub fn blocks(&self) -> impl Iterator<Item = Block> + '_ {
        self.blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.as_ref().map(|_| Block(i as u32)))
    }
    fn block(&self, b: Block) -> &BlockData {
        self.blocks[b.index()].as_ref().expect("retired SSA block")
    }
    fn block_mut(&mut self, b: Block) -> &mut BlockData {
        self.blocks[b.index()].as_mut().expect("retired SSA block")
    }
    pub fn params(&self, b: Block) -> &[Value] {
        &self.block(b).params
    }
    pub fn instructions(&self, b: Block) -> impl Iterator<Item = Inst> + '_ {
        let mut cursor = self.block(b).first;
        std::iter::from_fn(move || {
            if cursor == END {
                return None;
            }
            let inst = cursor;
            cursor = self.insts[inst.index()].next;
            Some(inst)
        })
    }
    pub fn op(&self, i: Inst) -> &Op {
        &self.insts[i.index()].op
    }
    pub fn result(&self, i: Inst) -> Option<Value> {
        let result = self.insts[i.index()].result;
        (result.0 != u32::MAX).then_some(result)
    }
    pub fn terminator(&self, b: Block) -> &Terminator {
        self.block(b).term.as_ref().expect("unterminated SSA block")
    }
    pub fn edge(&self, edge: Edge) -> (Block, &[Value]) {
        let (block, args) = self.edges[edge.index()];
        (block, &self.operands[args.range()])
    }
    pub fn call(&self, call: Call) -> (&Arc<Function>, &[Value], &[Buffer]) {
        let data = &self.calls[call.index()];
        (
            &data.callee,
            &self.operands[data.args.range()],
            &self.buffer_operands[data.buffers.range()],
        )
    }

    pub fn create_block(&mut self) -> Block {
        let id = Block(next_id(self.blocks.len()));
        self.blocks.push(Some(BlockData::default()));
        id
    }
    fn fresh_value(&mut self) -> Value {
        let value = Value(self.values);
        self.values = self.values.checked_add(1).expect("exhausted SSA values");
        value
    }
    pub fn append_param(&mut self, block: Block) -> Value {
        let value = self.fresh_value();
        self.block_mut(block).params.push(value);
        value
    }
    fn fallible(&self, op: Op) -> bool {
        match op {
            Op::Load(..) | Op::Store(..) | Op::Bin(Binary::Div | Binary::Rem, ..) => true,
            Op::Call(c) => self
                .calls
                .get(c.index())
                .is_some_and(|data| data.callee.needs_status()),
            _ => false,
        }
    }
    pub fn append_inst(&mut self, b: Block, op: Op) -> Inst {
        let result = if op.produces_value() {
            self.fresh_value()
        } else {
            Value(u32::MAX)
        };
        let id = Inst(next_id(self.insts.len()));
        let last = self.block(b).last;
        if last == END {
            self.block_mut(b).first = id;
        } else {
            self.insts[last.index()].next = id;
        }
        self.block_mut(b).last = id;
        self.fallible_ops += usize::from(self.fallible(op));
        self.insts.push(InstData {
            op,
            result,
            next: END,
        });
        id
    }
    pub fn append(&mut self, b: Block, op: Op) -> Option<Value> {
        let id = self.append_inst(b, op);
        self.result(id)
    }
    pub fn add_call(&mut self, callee: Arc<Function>, args: &[Value], buffers: &[Buffer]) -> Call {
        let id = Call(next_id(self.calls.len()));
        let args = Span::append(&mut self.operands, args);
        let buffers = Span::append(&mut self.buffer_operands, buffers);
        self.calls.push(CallData {
            callee,
            args,
            buffers,
        });
        id
    }
    pub fn add_edge(&mut self, target: Block, args: &[Value]) -> Edge {
        let id = Edge(next_id(self.edges.len()));
        let args = Span::append(&mut self.operands, args);
        self.edges.push((target, args));
        id
    }
    pub fn set_terminator(&mut self, b: Block, term: Terminator) {
        self.block_mut(b).term = Some(term);
    }
}
