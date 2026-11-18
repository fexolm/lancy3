//! Stage directly into Lancy. Scalar operations receive SSA values immediately.
//! Only mutable-name bindings and the operands of forward edges are deferred.
mod portable;
mod seal;
pub use portable::lower;
#[cfg(test)]
mod tests;

use lancy::hir::{self, Block, Function, Op, Value};
use splice::{
    BufferKind,
    frontend::{Body, KernelBuilder},
    hir::{Binary, Buffer, Label, Reg},
};
use std::{collections::HashMap, sync::Arc};

fn binary(op: Binary) -> hir::Binary {
    use hir::Binary as H;
    match op {
        Binary::Add => H::Add,
        Binary::Sub => H::Sub,
        Binary::Mul => H::Mul,
        Binary::Div => H::Div,
        Binary::Rem => H::Rem,
        Binary::And => H::And,
        Binary::Or => H::Or,
        Binary::Xor => H::Xor,
        Binary::Eq => H::Eq,
        Binary::Ne => H::Ne,
        Binary::Lt => H::Lt,
        Binary::Le => H::Le,
        Binary::Gt => H::Gt,
        Binary::Ge => H::Ge,
        Binary::Shl => H::Shl,
        Binary::Shr => H::Shr,
        Binary::Min => H::Min,
        Binary::Max => H::Max,
    }
}

pub(crate) fn build(
    arity: usize,
    kinds: &[BufferKind],
    body: Body<'_>,
) -> Result<Function, String> {
    let mut builder = Builder::new(arity, kinds)?;
    body(&mut builder)?;
    builder.seal()
}

impl<'a> Builder<'a> {
    fn new(arity: usize, kinds: &'a [BufferKind]) -> Result<Self, String> {
        let count = u32::try_from(arity).map_err(|_| "too many kernel parameters")?;
        if count == u32::MAX {
            return Err("too many kernel parameters".into());
        }
        let graph = Function::new(
            arity,
            kinds
                .iter()
                .map(|kind| match kind {
                    BufferKind::Read => hir::BufferKind::Read,
                    BufferKind::Write => hir::BufferKind::Write,
                })
                .collect(),
        );
        let entry = graph.entry();
        let mut first = State::default();
        first.defs.extend(
            graph
                .params(entry)
                .iter()
                .enumerate()
                .map(|(i, &v)| (Reg(i as u32), v)),
        );
        Ok(Self {
            graph,
            kinds,
            blocks: vec![first],
            current: Some(entry),
            count,
            demands: Vec::new(),
            labels: Vec::new(),
            immediate: None,
            error: None,
        })
    }
}

#[derive(Clone, Copy)]
enum Exit {
    Jump(Block),
    Branch(Value, Block, Block),
    Return(Value),
}
impl Exit {
    fn successors(self) -> impl Iterator<Item = Block> {
        let targets = match self {
            Self::Return(_) => [None, None],
            Self::Jump(b) => [Some(b), None],
            Self::Branch(_, t, f) => [Some(t), Some(f)],
        };
        targets.into_iter().flatten()
    }
}

#[derive(Default)]
struct State {
    defs: HashMap<Reg, Value>,
    incoming: Vec<Reg>,
    exit: Option<Exit>,
}
#[derive(Default)]
struct Destination {
    block: Option<Block>,
    marked: bool,
}

struct Builder<'a> {
    graph: Function,
    kinds: &'a [BufferKind],
    blocks: Vec<State>,
    current: Option<Block>,
    count: u32,
    demands: Vec<(Block, Reg)>,
    labels: Vec<Destination>,
    immediate: Option<(Reg, bool)>,
    error: Option<String>,
}

impl Builder<'_> {
    fn valid(&mut self, r: Reg) {
        if r.0 >= self.count {
            self.error
                .get_or_insert_with(|| format!("invalid register {r:?}"));
        }
    }
    fn buffer(&mut self, b: Buffer, write: bool) -> hir::Buffer {
        match self.kinds.get(b.index()) {
            None => {
                self.error
                    .get_or_insert_with(|| format!("unknown buffer {}", b.0));
            }
            Some(BufferKind::Read) if write => {
                self.error
                    .get_or_insert_with(|| format!("buffer {} is read-only", b.0));
            }
            _ => {}
        }
        hir::Buffer(b.0)
    }
    fn parameter(&mut self, block: Block, r: Reg) -> Value {
        let v = self.graph.append_param(block);
        let state = &mut self.blocks[block.index()];
        state.incoming.push(r);
        state.defs.insert(r, v);
        self.demands.push((block, r));
        v
    }
    fn read(&mut self, r: Reg) -> Value {
        self.valid(r);
        let Some(b) = self.current else {
            return Value(u32::MAX);
        };
        if let Some(&v) = self.blocks[b.index()].defs.get(&r) {
            return v;
        }
        if b == self.graph.entry() {
            self.error
                .get_or_insert_with(|| format!("uninitialized {r:?} at function entry"));
            return Value(u32::MAX);
        }
        self.parameter(b, r)
    }
    fn result(&mut self, op: Op) -> Reg {
        self.immediate = None;
        let r = self.local();
        self.define(r, op);
        r
    }
    fn define(&mut self, r: Reg, op: Op) {
        if let Some(b) = self.current {
            let v = self.graph.append(b, op).expect("operation defines a value");
            self.blocks[b.index()].defs.insert(r, v);
        }
    }
    fn call_op(&mut self, callee: Arc<Function>, args: &[Reg], buffers: &[Buffer]) -> Op {
        let args: Vec<_> = args.iter().map(|&r| self.read(r)).collect();
        let buffers: Vec<_> = buffers.iter().map(|b| hir::Buffer(b.0)).collect();
        Op::Call(self.graph.add_call(callee, &args, &buffers))
    }
    fn target(&mut self, label: Label) -> Block {
        let Some(dest) = self.labels.get_mut(label.0 as usize) else {
            self.error
                .get_or_insert_with(|| format!("invalid label {label:?}"));
            return self.graph.entry();
        };
        *dest.block.get_or_insert_with(|| {
            self.blocks.push(State::default());
            self.graph.create_block()
        })
    }
    fn end(&mut self, exit: Exit) {
        self.immediate = None;
        if let Some(b) = self.current.take() {
            self.blocks[b.index()].exit = Some(exit);
        }
    }
}

impl KernelBuilder for Builder<'_> {
    fn arg(&self, index: usize) -> Reg {
        Reg(if index < self.graph.arity() {
            index as u32
        } else {
            u32::MAX
        })
    }
    fn local(&mut self) -> Reg {
        let r = Reg(self.count);
        self.count = self
            .count
            .checked_add(1)
            .expect("exhausted scalar register handles");
        r
    }
    fn constant(&mut self, n: i32) -> Reg {
        let r = self.result(Op::Const(n));
        self.immediate = Some((r, n != 0));
        r
    }
    fn buffer_len(&mut self, buffer: Buffer) -> Reg {
        let b = self.buffer(buffer, false);
        self.result(Op::BufferLen(b))
    }
    fn load(&mut self, buffer: Buffer, index: Reg) -> Reg {
        let index = self.read(index);
        let b = self.buffer(buffer, false);
        self.result(Op::Load(b, index))
    }
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) {
        let (index, value) = (self.read(index), self.read(value));
        let buffer = self.buffer(buffer, true);
        self.immediate = None;
        if let Some(b) = self.current {
            self.graph.append(b, Op::Store(buffer, index, value));
        }
    }
    fn immediate_condition(&self, reg: Reg) -> Option<bool> {
        self.immediate.filter(|(r, _)| *r == reg).map(|(_, v)| v)
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg {
        let (a, b) = (self.read(a), self.read(b));
        self.result(Op::Bin(binary(op), a, b))
    }
    fn assign(&mut self, dst: Reg, src: Reg) {
        self.valid(dst);
        let v = self.read(src);
        self.immediate = None;
        if let Some(b) = self.current {
            self.blocks[b.index()].defs.insert(dst, v);
        }
    }
    fn label(&mut self) -> Label {
        let l = Label(u32::try_from(self.labels.len()).expect("exhausted labels"));
        self.labels.push(Destination::default());
        l
    }
    fn mark(&mut self, label: Label) {
        let block = self.target(label);
        let Some(dest) = self.labels.get_mut(label.0 as usize) else {
            return;
        };
        if std::mem::replace(&mut dest.marked, true) {
            self.error
                .get_or_insert_with(|| format!("duplicate label {label:?}"));
            return;
        }
        self.end(Exit::Jump(block));
        self.current = Some(block);
    }
    fn jump(&mut self, label: Label) {
        let b = self.target(label);
        self.end(Exit::Jump(b));
    }
    fn branch(&mut self, cond: Reg, yes: Label, no: Label) {
        let cond = self.read(cond);
        let (yes, no) = (self.target(yes), self.target(no));
        self.end(Exit::Branch(cond, yes, no));
    }
    fn ret(&mut self, value: Reg) {
        let v = self.read(value);
        self.end(Exit::Return(v));
    }
    fn call_kernel(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> Result<Reg, String> {
        if args.len() != arity {
            return Err(format!(
                "expected {arity} scalar arguments, got {}",
                args.len()
            ));
        }
        splice::hir::check_forwarding(self.kinds, kinds, buffers)?;
        let callee = Arc::new(build(arity, kinds, body)?);
        let op = self.call_op(callee, args, buffers);
        Ok(self.result(op))
    }
}
