//! Stage Splice's typed operations directly into Lancy SSA.
//! Only mutable-name bindings and the operands of forward edges are deferred.
mod portable;
mod seal;
mod simd;
pub use portable::lower;
#[cfg(test)]
mod tests;

use lancy::hir::{self, Block, Function, Op, Value};
use lancy::support::slotmap::DenseMap;
use splice::{
    BufferKind,
    frontend::{Body, KernelBuilder, Specialization},
    hir::{Binary, Buffer, Label, Reg, Simd, ValueType},
    simd::Element,
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
        let mut params = vec![hir::Type::I32; arity];
        for _ in kinds {
            params.extend([hir::Type::Ptr, hir::Type::I32]);
        }
        let graph = Function::new(&params, hir::Type::I32);
        let entry = graph.entry();
        let mut first = State::default();
        first.defs.extend(
            graph
                .params(entry)
                .iter()
                .take(arity)
                .enumerate()
                .map(|(i, &v)| (Reg(i as u32), v)),
        );
        let mut blocks = DenseMap::new();
        let state = blocks.insert(first);
        debug_assert_eq!(state, entry);
        Ok(Self {
            graph,
            callees: HashMap::new(),
            arity,
            kinds,
            blocks,
            current: Some(entry),
            types: vec![ValueType::Scalar; arity],
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
    callees: HashMap<Specialization, Arc<Function>>,
    arity: usize,
    kinds: &'a [BufferKind],
    blocks: DenseMap<Block, State>,
    current: Option<Block>,
    types: Vec<ValueType>,
    demands: Vec<(Block, Reg)>,
    labels: Vec<Destination>,
    immediate: Option<(Reg, bool)>,
    error: Option<String>,
}

impl Builder<'_> {
    fn valid(&mut self, r: Reg) {
        if r.0 as usize >= self.types.len() {
            self.error
                .get_or_insert_with(|| format!("invalid register {r:?}"));
        }
    }
    fn expect_type(&mut self, r: Reg, ty: ValueType) {
        if self.types.get(r.0 as usize) != Some(&ty) {
            self.error
                .get_or_insert_with(|| format!("expected {ty:?} register, got {r:?}"));
        }
    }
    fn scalar(&mut self, r: Reg) -> Value {
        self.expect_type(r, ValueType::Scalar);
        self.read(r)
    }
    fn slice(&mut self, buffer: Buffer, write: bool) -> (Value, Value) {
        match self.kinds.get(buffer.index()) {
            None => {
                self.error
                    .get_or_insert_with(|| format!("unknown buffer {}", buffer.0));
                return (Value::INVALID, Value::INVALID);
            }
            Some(kind) if write && !kind.writable() => {
                self.error
                    .get_or_insert_with(|| format!("buffer {} is read-only", buffer.0));
            }
            _ => {}
        }
        let first = self.arity + 2 * buffer.index();
        let params = self.graph.params(self.graph.entry());
        (params[first], params[first + 1])
    }

    fn address(&mut self, buffer: Buffer, index: Value, write: bool) -> Value {
        if self
            .kinds
            .get(buffer.index())
            .is_some_and(|k| k.element() != Element::I32)
        {
            self.error
                .get_or_insert_with(|| "scalar memory access requires an i32 buffer".into());
        }
        let (base, len) = self.slice(buffer, write);
        let Some(block) = self.current else {
            return Value::INVALID;
        };
        self.graph.append(block, Op::BoundsCheck(index, len));
        self.graph
            .append(block, Op::PtrOffset(base, index, hir::Scale::Four))
            .unwrap()
    }
    fn parameter(&mut self, block: Block, r: Reg) -> Value {
        let ty = self
            .types
            .get(r.0 as usize)
            .copied()
            .unwrap_or(ValueType::Scalar);
        let v = self.graph.append_typed_param(block, simd::native_type(ty));
        let state = &mut self.blocks[block];
        state.incoming.push(r);
        state.defs.insert(r, v);
        self.demands.push((block, r));
        v
    }
    fn read(&mut self, r: Reg) -> Value {
        self.valid(r);
        let Some(b) = self.current else {
            return Value::INVALID;
        };
        if let Some(&v) = self.blocks[b].defs.get(&r) {
            return v;
        }
        if b == self.graph.entry() {
            self.error
                .get_or_insert_with(|| format!("uninitialized {r:?} at function entry"));
            return Value::INVALID;
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
            self.blocks[b].defs.insert(r, v);
        }
    }
    fn call_op(&mut self, callee: Arc<Function>, args: &[Reg], buffers: &[Buffer]) -> Op {
        let mut values: Vec<_> = args.iter().map(|&r| self.scalar(r)).collect();
        for &buffer in buffers {
            let (ptr, len) = self.slice(buffer, false);
            values.extend([ptr, len]);
        }
        Op::Call(self.graph.add_call(callee, &values))
    }
    fn staged_call(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        key: Option<Specialization>,
        body: Body<'_>,
    ) -> splice::Result<Reg> {
        if args.len() != arity {
            return Err(format!(
                "expected {arity} scalar arguments, got {}",
                args.len()
            ));
        }
        splice::hir::check_forwarding(self.kinds, kinds, buffers)?;
        let callee = if let Some(hit) = key.as_ref().and_then(|k| self.callees.get(k)) {
            hit.clone()
        } else {
            let mut child = Builder::new(arity, kinds)?;
            std::mem::swap(&mut child.callees, &mut self.callees);
            let staged = body(&mut child);
            std::mem::swap(&mut child.callees, &mut self.callees);
            staged?;
            let callee = Arc::new(child.seal()?);
            if let Some(key) = key {
                self.callees.insert(key, callee.clone());
            }
            callee
        };
        let op = self.call_op(callee, args, buffers);
        Ok(self.result(op))
    }
    fn target(&mut self, label: Label) -> Block {
        let Some(dest) = self.labels.get_mut(label.0 as usize) else {
            self.error
                .get_or_insert_with(|| format!("invalid label {label:?}"));
            return self.graph.entry();
        };
        *dest.block.get_or_insert_with(|| {
            let block = self.graph.create_block();
            let state = self.blocks.insert(State::default());
            debug_assert_eq!(state, block);
            block
        })
    }
    fn end(&mut self, exit: Exit) {
        self.immediate = None;
        if let Some(b) = self.current.take() {
            self.blocks[b].exit = Some(exit);
        }
    }
}

impl KernelBuilder for Builder<'_> {
    fn reg_type(&self, r: Reg) -> Option<ValueType> {
        self.types.get(r.0 as usize).copied()
    }
    fn arg(&self, index: usize) -> Reg {
        Reg(if index < self.arity {
            index as u32
        } else {
            u32::MAX
        })
    }
    fn local(&mut self) -> Reg {
        self.typed_local(ValueType::Scalar)
    }
    fn typed_local(&mut self, ty: ValueType) -> Reg {
        let r = Reg(u32::try_from(self.types.len()).expect("exhausted register handles"));
        self.types.push(ty);
        r
    }
    fn simd(&mut self, inst: Simd, args: &[Reg]) -> splice::Result<Option<Reg>> {
        let value = self.emit_simd(inst, args)?;
        let dst = inst.output().map(|ty| self.typed_local(ty));
        self.bind_simd(dst, value);
        Ok(dst)
    }
    fn constant(&mut self, n: i32) -> Reg {
        let r = self.result(Op::Const(n));
        self.immediate = Some((r, n != 0));
        r
    }
    fn buffer_len(&mut self, buffer: Buffer) -> Reg {
        let (_, len) = self.slice(buffer, false);
        self.result(Op::Copy(len))
    }
    fn load(&mut self, buffer: Buffer, index: Reg) -> Reg {
        let index = self.scalar(index);
        let addr = self.address(buffer, index, false);
        self.result(Op::Load(hir::Type::I32, addr))
    }
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) {
        let (index, value) = (self.scalar(index), self.scalar(value));
        let addr = self.address(buffer, index, true);
        self.immediate = None;
        if let Some(b) = self.current {
            self.graph.append(b, Op::Store(addr, value));
        }
    }
    fn immediate_condition(&self, reg: Reg) -> Option<bool> {
        self.immediate.filter(|(r, _)| *r == reg).map(|(_, v)| v)
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg {
        let (a, b) = (self.scalar(a), self.scalar(b));
        self.result(Op::Bin(binary(op), a, b))
    }
    fn assign(&mut self, dst: Reg, src: Reg) {
        self.valid(dst);
        if let Some(&ty) = self.types.get(dst.0 as usize) {
            self.expect_type(src, ty);
        }
        let v = self.read(src);
        self.immediate = None;
        if let Some(b) = self.current {
            self.blocks[b].defs.insert(dst, v);
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
        let cond = self.scalar(cond);
        let (yes, no) = (self.target(yes), self.target(no));
        self.end(Exit::Branch(cond, yes, no));
    }
    fn ret(&mut self, value: Reg) {
        let v = self.scalar(value);
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
        self.staged_call(arity, kinds, args, buffers, None, body)
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> splice::Result<Reg> {
        let (arity, kinds) = key.signature();
        self.staged_call(arity, kinds, args, buffers, Some(key), body)
    }
}
