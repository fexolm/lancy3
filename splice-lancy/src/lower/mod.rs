//! Stage Splice's typed operations directly into Lancy SSA.
//! Only mutable-name bindings and the operands of forward edges are deferred.
mod portable;
mod seal;
mod simd;
pub use portable::lower;
pub(super) use portable::{DebugSources, lower_debug};
#[cfg(test)]
mod tests;

use lancy::hir::{self, Block, Function, Op, Value};
use lancy::support::slotmap::DenseMap;
use splice_core::{
    BufferKind, ContextArguments, StaticRustSymbol,
    frontend::{Body, KernelBuilder, KernelSource, Specialization},
    hir::{
        Binary, Buffer, ContextSlice, Label, Reg, RustCall, RustContextTarget, RustPacketSymbol,
        Simd, ValueType,
    },
    simd::Element,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

fn binary(op: Binary) -> hir::Binary {
    use hir::Binary as H;
    match op {
        Binary::Add | Binary::CheckedAdd => H::Add,
        Binary::Sub | Binary::CheckedSub => H::Sub,
        Binary::Mul | Binary::CheckedMul => H::Mul,
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
        Binary::Shl | Binary::CheckedShl => H::Shl,
        Binary::Shr | Binary::CheckedShr => H::Shr,
        Binary::Min => H::Min,
        Binary::Max => H::Max,
    }
}

#[cfg(test)]
pub(crate) fn build(
    arity: usize,
    kinds: &[BufferKind],
    body: Body<'_>,
) -> splice_core::Result<Function> {
    build_with_source(arity, kinds, body).map(|(function, _)| function)
}

pub(crate) fn build_with_source(
    arity: usize,
    kinds: &[BufferKind],
    body: Body<'_>,
) -> splice_core::Result<(Function, Option<&'static KernelSource>)> {
    splice_target::simd()?;
    let mut builder = Builder::new(arity, kinds)?;
    body(&mut builder)?;
    let source = builder.source;
    Ok((builder.seal()?, source))
}

impl<'a> Builder<'a> {
    fn new(arity: usize, kinds: &'a [BufferKind]) -> splice_core::Result<Self> {
        let count = u32::try_from(arity)
            .map_err(|_| splice_core::BuildError::unsupported("too many kernel parameters"))?;
        if count == u32::MAX {
            return Err(splice_core::BuildError::unsupported(
                "too many kernel parameters",
            ));
        }
        let mut params = vec![hir::Type::I32; arity];
        for _ in kinds {
            params.extend([hir::Type::Ptr, hir::Type::I32]);
        }
        let mut graph = Function::new(&params, hir::Type::I32);
        // BufferArg enforces these borrows at entry; call-site validation
        // preserves them when a nested kernel receives forwarded buffers.
        graph.set_slice_params(
            arity,
            kinds
                .iter()
                .map(|kind| (kind.element().bytes() as u8, kind.writable()))
                .collect(),
        );
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
            source: None,
            callees: HashMap::new(),
            arity,
            kinds,
            blocks,
            current: Some(entry),
            types: vec![ValueType::Scalar; arity],
            demands: Vec::new(),
            observed: HashMap::new(),
            unassigned: HashSet::new(),
            labels: Vec::new(),
            error: None,
            has_shared_i32: kinds.contains(&BufferKind::Read),
            constants: HashMap::new(),
            shared_loads: HashMap::new(),
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
    source: Option<&'static KernelSource>,
    callees: HashMap<Specialization, Arc<Function>>,
    arity: usize,
    kinds: &'a [BufferKind],
    blocks: DenseMap<Block, State>,
    current: Option<Block>,
    types: Vec<ValueType>,
    /// Blocks whose parameter for a register needs a value from every
    /// predecessor. `required` is false while only source points observe it.
    demands: Vec<Demand>,
    /// Parameters that only source points read, with their block.
    observed: HashMap<Value, (Block, Reg)>,
    /// Placeholders for a register that is unassigned on some path.
    unassigned: HashSet<Value>,
    labels: Vec<Destination>,
    error: Option<splice_core::BuildError>,
    has_shared_i32: bool,
    constants: HashMap<Value, i32>,
    shared_loads: HashMap<(Buffer, LoadIndex), Value>,
}

#[derive(Clone, Copy)]
struct Demand {
    block: Block,
    reg: Reg,
    required: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum LoadIndex {
    Constant(i32),
    Value(Value),
}

impl Builder<'_> {
    fn valid(&mut self, r: Reg) {
        if r.0 as usize >= self.types.len() {
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify(format!("invalid register {r:?}"))
            });
        }
    }
    fn expect_type(&mut self, r: Reg, ty: ValueType) {
        if self.types.get(r.0 as usize) != Some(&ty) {
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify(format!("expected {ty:?} register, got {r:?}"))
            });
        }
    }
    fn scalar(&mut self, r: Reg) -> Value {
        self.expect_type(r, ValueType::Scalar);
        self.read(r)
    }
    fn slice(&mut self, buffer: Buffer, write: bool) -> (Value, Value) {
        match self.kinds.get(buffer.index()) {
            None => {
                self.error.get_or_insert_with(|| {
                    splice_core::BuildError::verify(format!("unknown buffer {}", buffer.0))
                });
                return (Value::INVALID, Value::INVALID);
            }
            Some(kind) if write && !kind.writable() => {
                self.error.get_or_insert_with(|| {
                    splice_core::BuildError::verify(format!("buffer {} is read-only", buffer.0))
                });
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
            .is_some_and(|k| k.context_signature().is_some() || k.element() != Element::I32)
        {
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify("scalar memory access requires an i32 buffer")
            });
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
    fn parameter(&mut self, block: Block, r: Reg, required: bool) -> Value {
        let ty = self
            .types
            .get(r.0 as usize)
            .copied()
            .unwrap_or(ValueType::Scalar);
        let v = self.graph.append_typed_param(block, simd::native_type(ty));
        let state = &mut self.blocks[block];
        state.incoming.push(r);
        state.defs.insert(r, v);
        if !required {
            self.observed.insert(v, (block, r));
        }
        self.demands.push(Demand {
            block,
            reg: r,
            required,
        });
        v
    }
    /// A use of `v` requires every parameter it came from to be assigned.
    fn require(&mut self, v: Value) {
        if let Some((block, reg)) = self.observed.remove(&v) {
            self.demands.push(Demand {
                block,
                reg,
                required: true,
            });
        }
    }
    fn read(&mut self, r: Reg) -> Value {
        self.valid(r);
        let Some(b) = self.current else {
            return Value::INVALID;
        };
        if let Some(&v) = self.blocks[b].defs.get(&r) {
            self.require(v);
            return v;
        }
        if b == self.graph.entry() {
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify(format!("uninitialized {r:?} at function entry"))
            });
            return Value::INVALID;
        }
        self.parameter(b, r, true)
    }
    /// Read a scalar register for a source point. It need not be assigned on
    /// every path; where it is not, the point sees zero.
    fn observe(&mut self, r: Reg) -> Value {
        self.valid(r);
        let Some(b) = self.current else {
            return Value::INVALID;
        };
        if let Some(&v) = self.blocks[b].defs.get(&r) {
            return v;
        }
        if b == self.graph.entry() {
            return self.graph.append(b, Op::Const(0)).unwrap();
        }
        self.parameter(b, r, false)
    }
    fn result(&mut self, op: Op) -> Reg {
        let r = self.local();
        self.define(r, op);
        r
    }

    fn arithmetic(&mut self, dst: Reg, op: Binary, lhs: Reg, rhs: Reg) {
        let (a, b) = (self.scalar(lhs), self.scalar(rhs));
        let Some(block) = self.current else {
            return;
        };
        let kind = binary(op);
        let mut value = |op| self.graph.append(block, op).unwrap();
        match op {
            Binary::CheckedAdd | Binary::CheckedSub | Binary::CheckedMul => {
                let a = value(Op::Cast(hir::Type::I64, a));
                let b = value(Op::Cast(hir::Type::I64, b));
                let wide = value(Op::Bin(kind, a, b));
                let narrow = value(Op::Cast(hir::Type::I32, wide));
                let sign_extended = value(Op::Cast(hir::Type::I64, narrow));
                let overflow = value(Op::Bin(hir::Binary::Ne, wide, sign_extended));
                let overflow = value(Op::Cast(hir::Type::I32, overflow));
                self.graph.append(block, Op::CheckOverflow(overflow));
                self.blocks[block].defs.insert(dst, narrow);
            }
            Binary::CheckedShl | Binary::CheckedShr => {
                let high_mask = value(Op::Const(!31));
                let overflow = value(Op::Bin(hir::Binary::And, b, high_mask));
                self.graph.append(block, Op::CheckOverflow(overflow));
                self.define(dst, Op::Bin(kind, a, b));
            }
            _ => self.define(dst, Op::Bin(kind, a, b)),
        }
    }
    fn define(&mut self, r: Reg, op: Op) {
        if let Some(b) = self.current {
            let v = self.graph.append(b, op).expect("operation defines a value");
            if let Op::Const(n) = op
                && self.has_shared_i32
            {
                self.constants.insert(v, n);
            }
            self.blocks[b].defs.insert(r, v);
        }
    }
    fn define_load(&mut self, dst: Reg, buffer: Buffer, index: Value) {
        let Some(block) = self.current else {
            return;
        };
        let shared = self.kinds.get(buffer.index()) == Some(&BufferKind::Read);
        let key = if shared {
            // A Splice shared slice cannot be modified via another parameter
            // or a callee. The first load already checked this index.
            let index = self
                .constants
                .get(&index)
                .copied()
                .map_or(LoadIndex::Value(index), LoadIndex::Constant);
            let key = (buffer, index);
            if let Some(&value) = self.shared_loads.get(&key) {
                self.blocks[block].defs.insert(dst, value);
                return;
            }
            Some(key)
        } else {
            None
        };
        let address = self.address(buffer, index, false);
        self.define(dst, Op::Load(hir::Type::I32, address));
        if let Some(key) = key {
            self.shared_loads.insert(key, self.blocks[block].defs[&dst]);
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
    fn define_rust_call(
        &mut self,
        dst: Reg,
        import: RustCall,
        args: &[Reg],
    ) -> splice_core::Result<()> {
        import.validate(args.iter().map(|&arg| self.reg_type(arg)))?;
        let [a, b] = self.rust_operands(args);
        self.define(dst, Op::RustCall(import.symbol, a, b));
        Ok(())
    }

    fn define_packet_call(
        &mut self,
        dst: Reg,
        import: &'static RustPacketSymbol,
        args: &[Reg],
    ) -> splice_core::Result<()> {
        splice_core::hir::check_rust_packet_call(
            import,
            args.iter().map(|&arg| self.reg_type(arg)),
        )?;
        self.shared_loads.clear();
        if self.current.is_some() {
            let args: Vec<_> = args.iter().map(|&arg| self.scalar(arg)).collect();
            let call = self.graph.add_packet_call(import, 0, &args);
            self.define(dst, Op::RustPacketCall(call));
        }
        Ok(())
    }

    fn define_context_call(
        &mut self,
        dst: Reg,
        target: RustContextTarget,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<()> {
        splice_core::hir::check_rust_context_call(
            self.kinds,
            target,
            args.iter().map(|&arg| self.reg_type(arg)),
            slices,
        )?;
        let Some(block) = self.current else {
            return Ok(());
        };
        let (context, operation) = match target {
            RustContextTarget::Borrowed { context, operation } => {
                (Some(self.slice(context, false).0), operation)
            }
            RustContextTarget::Static(_) => (None, 0),
        };
        let operation = self.constant(operation as i32);
        let operation = self.scalar(operation);
        let (import, args) = if let Some(context) = context
            && slices.is_empty()
            && args.len() <= 2
        {
            let [a, b] = self.rust_operands(args);
            (
                hir::RustContextImport::Scalars(splice_core::invoke_runtime_context),
                [context, operation, a, b],
            )
        } else {
            self.shared_loads.clear();
            let layout = ContextArguments::packet_layout(args.len() as u32, slices.len() as u32)
                .expect("validated packet size");
            let packet = self
                .graph
                .append(
                    block,
                    Op::CallScratch {
                        offset: 0,
                        bytes: layout.bytes.max(8),
                    },
                )
                .unwrap();
            let store = |graph: &mut Function, offset, value| {
                let offset = graph.append(block, Op::Const(offset as i32)).unwrap();
                let ptr = graph
                    .append(block, Op::PtrOffset(packet, offset, hir::Scale::One))
                    .unwrap();
                graph.append(block, Op::Store(ptr, value));
            };
            for (index, &reg) in args.iter().enumerate() {
                let value = self.scalar(reg);
                store(&mut self.graph, index as u32 * 4, value);
            }
            for (index, &ContextSlice { buffer, writable }) in slices.iter().enumerate() {
                let (ptr, len) = self.slice(buffer, writable);
                let tag = ContextArguments::view_tag(self.kinds[buffer.index()].element(), writable)
                    as i32;
                let tag = self.graph.append(block, Op::Const(tag)).unwrap();
                let offset = layout.slices_at + index as u32 * ContextArguments::SLICE_BYTES;
                store(&mut self.graph, offset, ptr);
                store(
                    &mut self.graph,
                    offset + ContextArguments::LENGTH_OFFSET,
                    len,
                );
                store(
                    &mut self.graph,
                    offset + ContextArguments::ELEMENT_OFFSET,
                    tag,
                );
            }
            let zero = self.graph.append(block, Op::Const(0)).unwrap();
            let entry = match target {
                RustContextTarget::Borrowed { .. } => splice_core::invoke_packet_runtime_context,
                RustContextTarget::Static(symbol) => symbol.entry(),
            };
            // The static entry reads only the packet (arg 2). Reuse its live
            // pointer for the unused context slot in the native ABI.
            (
                hir::RustContextImport::Packet(entry),
                [context.unwrap_or(packet), operation, packet, zero],
            )
        };
        self.define(dst, Op::RustContextCall(import, args));
        Ok(())
    }

    fn rust_operands(&mut self, args: &[Reg]) -> [Value; 2] {
        let [a, b] = match args {
            [a, b] => [*a, *b],
            [a] => [*a, self.constant(0)],
            [] => [self.constant(0); 2],
            _ => unreachable!("validated Rust import arity"),
        };
        let (a, b) = (self.scalar(a), self.scalar(b));
        self.shared_loads.clear();
        [a, b]
    }
    fn staged_call(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        key: Option<Specialization>,
        body: Body<'_>,
    ) -> splice_core::Result<Reg> {
        splice_core::hir::check_call(
            arity,
            kinds,
            args.iter().map(|&r| self.reg_type(r)),
            self.kinds,
            buffers,
        )?;
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
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify(format!("invalid label {label:?}"))
            });
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
        if let Some(b) = self.current.take() {
            self.blocks[b].exit = Some(exit);
            self.constants.clear();
            self.shared_loads.clear();
        }
    }
}

impl KernelBuilder for Builder<'_> {
    fn set_source(&mut self, source: &'static KernelSource) {
        self.source = Some(source);
    }
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
    fn simd(&mut self, inst: Simd, args: &[Reg]) -> splice_core::Result<Option<Reg>> {
        let value = self.emit_simd(inst, args)?;
        let dst = inst.output().map(|ty| self.typed_local(ty));
        self.bind_simd(dst, value);
        Ok(dst)
    }
    fn constant(&mut self, n: i32) -> Reg {
        self.result(Op::Const(n))
    }
    fn buffer_len(&mut self, buffer: Buffer) -> Reg {
        if self
            .kinds
            .get(buffer.index())
            .is_some_and(|kind| kind.context_signature().is_some())
        {
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify("slice length cannot use Rust context")
            });
        }
        let (_, len) = self.slice(buffer, false);
        self.result(Op::Copy(len))
    }
    fn load(&mut self, buffer: Buffer, index: Reg) -> Reg {
        let index = self.scalar(index);
        let dst = self.local();
        self.define_load(dst, buffer, index);
        dst
    }
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) {
        let (index, value) = (self.scalar(index), self.scalar(value));
        let addr = self.address(buffer, index, true);
        if let Some(b) = self.current {
            self.graph.append(b, Op::Store(addr, value));
        }
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg {
        let dst = self.local();
        self.arithmetic(dst, op, a, b);
        dst
    }
    fn assign(&mut self, dst: Reg, src: Reg) {
        self.valid(dst);
        if let Some(&ty) = self.types.get(dst.0 as usize) {
            self.expect_type(src, ty);
        }
        let v = self.read(src);
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
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify(format!("duplicate label {label:?}"))
            });
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
    fn trap_bounds(&mut self) -> splice_core::Result<()> {
        if let Some(block) = self.current {
            let zero = self.graph.append(block, Op::Const(0)).unwrap();
            self.graph.append(block, Op::BoundsCheck(zero, zero));
            self.end(Exit::Return(zero));
        }
        Ok(())
    }
    fn rust_call(&mut self, import: RustCall, args: &[Reg]) -> splice_core::Result<Reg> {
        let dst = self.local();
        self.define_rust_call(dst, import, args)?;
        Ok(dst)
    }
    fn rust_packet_call(
        &mut self,
        import: &'static RustPacketSymbol,
        args: &[Reg],
    ) -> splice_core::Result<Reg> {
        let dst = self.local();
        self.define_packet_call(dst, import, args)?;
        Ok(dst)
    }
    fn rust_context_call_with_slices(
        &mut self,
        context: Buffer,
        operation: u32,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        let dst = self.local();
        self.define_context_call(
            dst,
            RustContextTarget::Borrowed { context, operation },
            args,
            slices,
        )?;
        Ok(dst)
    }
    fn rust_static_call(
        &mut self,
        symbol: &'static StaticRustSymbol,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        let dst = self.local();
        self.define_context_call(dst, RustContextTarget::Static(symbol), args, slices)?;
        Ok(dst)
    }
    fn call_kernel(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> splice_core::Result<Reg> {
        self.staged_call(arity, kinds, args, buffers, None, body)
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> splice_core::Result<Reg> {
        let (arity, kinds) = key.signature();
        self.staged_call(arity, kinds, args, buffers, Some(key), body)
    }
}
