//! Stage Splice's typed operations directly into Lancy SSA.
//! Only mutable-name bindings and the operands of forward edges are deferred.
//! The portable path replays a function into the same builder.
mod seal;
mod simd;
#[cfg(test)]
mod tests;

use lancy::hir::{self, Block, Function, Op, Value};
use lancy::support::slotmap::DenseMap;
use splice_core::{
    BufferKind, ContextArguments, StaticRustSymbol,
    frontend::{Body, KernelBuilder, KernelSource, Signature, Specialization},
    hir::{
        Binary, Buffer, ContextSlice, Label, Reg, RustCall, RustContextTarget, RustPacketSymbol,
        Simd, ValueType,
    },
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

/// A source point of a debug build: the function it belongs to and, after
/// the marker that opens each function, the point with its saved values.
pub(crate) struct DebugSource {
    pub function: Option<&'static KernelSource>,
    pub point: Option<(splice_core::debug::Point, Vec<splice_core::debug::Slot>)>,
}

/// Stage a validated body into Lancy SSA.
pub(crate) fn stage(
    signature: Signature<'_>,
    body: Body<'_>,
    vector_bytes: usize,
    debug: Option<&mut Vec<DebugSource>>,
) -> splice_core::Result<(Function, Option<&'static KernelSource>)> {
    let mut debug = debug;
    let sources = debug.as_deref_mut().map(std::mem::take);
    let mut builder = Builder::new(signature, vector_bytes, sources)?;
    let staged = body(&mut builder);
    if let (Some(out), Some(sources)) = (debug, builder.debug.take()) {
        *out = sources;
    }
    staged?;
    builder.finish()
}

/// Lower a verified portable function and its callees. Preserve shared
/// callee identities. Raw invocation of the result must uphold Splice's
/// slice contract: immutable buffers may overlap and writable buffers have
/// exclusive access.
pub fn lower(
    root: &splice_core::Function,
    target: &splice_core::Target,
) -> splice_core::Result<hir::Function> {
    lower_with(root, target.vector_bytes(), None)
}

pub(crate) fn lower_with(
    root: &splice_core::Function,
    vector_bytes: usize,
    debug: Option<&mut Vec<DebugSource>>,
) -> splice_core::Result<hir::Function> {
    let (graph, _) = stage(
        root.signature(),
        Box::new(|b| splice_core::hir::replay(root, b)),
        vector_bytes,
        debug,
    )?;
    Ok(graph)
}

impl<'a> Builder<'a> {
    fn new(
        signature: Signature<'a>,
        vector_bytes: usize,
        debug: Option<Vec<DebugSource>>,
    ) -> splice_core::Result<Self> {
        let Signature {
            arity,
            buffers: kinds,
        } = signature;
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
        let mut builder = Self {
            graph,
            source: None,
            callees: HashMap::new(),
            functions: HashMap::new(),
            arity,
            kinds,
            vector_bytes,
            blocks,
            current: Some(entry),
            types: vec![ValueType::Scalar; arity],
            demands: Vec::new(),
            observed: HashMap::new(),
            unassigned: HashSet::new(),
            labels: Vec::new(),
            has_shared_i32: kinds.contains(&BufferKind::Read),
            constants: HashMap::new(),
            shared_loads: HashMap::new(),
            debug,
            marker: 0,
        };
        if let Some(sources) = &mut builder.debug {
            // Each function opens with a marker that names it.
            builder.marker = sources.len();
            let id = u32::try_from(builder.marker)
                .map_err(|_| splice_core::BuildError::unsupported("too many debug points"))?;
            sources.push(DebugSource {
                function: None,
                point: None,
            });
            builder.graph.append(entry, Op::DebugPoint(id));
        }
        Ok(builder)
    }

    fn finish(self) -> splice_core::Result<(Function, Option<&'static KernelSource>)> {
        let source = self.source;
        Ok((self.seal()?, source))
    }

    /// Stage a nested function sharing this build's caches.
    fn nested(
        &mut self,
        signature: Signature<'_>,
        body: impl FnOnce(&mut dyn KernelBuilder) -> splice_core::Result<()>,
    ) -> splice_core::Result<Arc<Function>> {
        let mut child = Builder::new(signature, self.vector_bytes, self.debug.take())?;
        std::mem::swap(&mut child.callees, &mut self.callees);
        std::mem::swap(&mut child.functions, &mut self.functions);
        let staged = body(&mut child);
        std::mem::swap(&mut child.callees, &mut self.callees);
        std::mem::swap(&mut child.functions, &mut self.functions);
        self.debug = child.debug.take();
        staged?;
        Ok(Arc::new(child.seal()?))
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

struct Builder<'a> {
    graph: Function,
    source: Option<&'static KernelSource>,
    callees: HashMap<Specialization, Arc<Function>>,
    /// Lowered portable callees, by identity.
    functions: HashMap<*const splice_core::Function, (Arc<splice_core::Function>, Arc<Function>)>,
    arity: usize,
    kinds: &'a [BufferKind],
    /// Width of the target's vectors; fixes lane counts and alignment.
    vector_bytes: usize,
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
    labels: Vec<Option<Block>>,
    has_shared_i32: bool,
    constants: HashMap<Value, i32>,
    shared_loads: HashMap<(Buffer, LoadIndex), Value>,
    /// The source points of a debug build, shared by its functions.
    debug: Option<Vec<DebugSource>>,
    /// This function's entry in `debug`.
    marker: usize,
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
    fn slice(&self, buffer: Buffer) -> (Value, Value) {
        let first = self.arity + 2 * buffer.index();
        let params = self.graph.params(self.graph.entry());
        (params[first], params[first + 1])
    }

    fn address(&mut self, buffer: Buffer, index: Value) -> Value {
        let (base, len) = self.slice(buffer);
        let block = self.current.expect("open block");
        self.graph.append(block, Op::BoundsCheck(index, len));
        self.graph
            .append(block, Op::PtrOffset(base, index, hir::Scale::Four))
            .unwrap()
    }
    fn parameter(&mut self, block: Block, r: Reg, required: bool) -> Value {
        let ty = self.types[r.0 as usize];
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
        let b = self.current.expect("open block");
        if let Some(&v) = self.blocks[b].defs.get(&r) {
            self.require(v);
            return v;
        }
        if b == self.graph.entry() {
            // Assigned on no path: the verifier rejects the kernel once the
            // body finishes, so until then the read sees zero.
            let v = self.zero(self.types[r.0 as usize]);
            self.blocks[b].defs.insert(r, v);
            return v;
        }
        self.parameter(b, r, true)
    }
    /// Read a scalar register for a source point. It need not be assigned on
    /// every path; where it is not, the point sees zero.
    fn observe(&mut self, r: Reg) -> Value {
        let b = self.current.expect("open block");
        if let Some(&v) = self.blocks[b].defs.get(&r) {
            return v;
        }
        if b == self.graph.entry() {
            return self.zero(self.types[r.0 as usize]);
        }
        self.parameter(b, r, false)
    }
    /// Operations in a block that no edge reaches have no effect.
    fn reachable(&self) -> bool {
        self.current.is_some()
    }
    fn value(&mut self, op: Op) -> Value {
        self.graph
            .append(self.current.expect("open block"), op)
            .expect("operation defines a value")
    }
    fn result(&mut self, op: Op) -> splice_core::Result<Reg> {
        let r = self.local(ValueType::Scalar)?;
        if self.reachable() {
            self.define(r, op);
        }
        Ok(r)
    }

    fn arithmetic(&mut self, dst: Reg, op: Binary, lhs: Reg, rhs: Reg) {
        let (a, b) = (self.read(lhs), self.read(rhs));
        let kind = binary(op);
        match op {
            Binary::CheckedAdd | Binary::CheckedSub | Binary::CheckedMul => {
                let a = self.value(Op::Cast(hir::Type::I64, a));
                let b = self.value(Op::Cast(hir::Type::I64, b));
                let wide = self.value(Op::Bin(kind, a, b));
                let narrow = self.value(Op::Cast(hir::Type::I32, wide));
                let sign_extended = self.value(Op::Cast(hir::Type::I64, narrow));
                let overflow = self.value(Op::Bin(hir::Binary::Ne, wide, sign_extended));
                let overflow = self.value(Op::Cast(hir::Type::I32, overflow));
                self.effect(Op::CheckOverflow(overflow));
                self.bind(dst, narrow);
            }
            Binary::CheckedShl | Binary::CheckedShr => {
                let high_mask = self.value(Op::Const(!31));
                let overflow = self.value(Op::Bin(hir::Binary::And, b, high_mask));
                self.effect(Op::CheckOverflow(overflow));
                self.define(dst, Op::Bin(kind, a, b));
            }
            _ => self.define(dst, Op::Bin(kind, a, b)),
        }
    }
    fn bind(&mut self, r: Reg, v: Value) {
        let b = self.current.expect("open block");
        self.blocks[b].defs.insert(r, v);
    }
    fn define(&mut self, r: Reg, op: Op) {
        let v = self.value(op);
        if let Op::Const(n) = op
            && self.has_shared_i32
        {
            self.constants.insert(v, n);
        }
        self.bind(r, v);
    }
    fn define_load(&mut self, dst: Reg, buffer: Buffer, index: Value) {
        let shared = self.kinds[buffer.index()] == BufferKind::Read;
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
                self.bind(dst, value);
                return;
            }
            Some(key)
        } else {
            None
        };
        let address = self.address(buffer, index);
        self.define(dst, Op::Load(hir::Type::I32, address));
        if let Some(key) = key {
            let block = self.current.expect("open block");
            self.shared_loads.insert(key, self.blocks[block].defs[&dst]);
        }
    }
    fn call(
        &mut self,
        callee: Arc<Function>,
        args: &[Reg],
        buffers: &[Buffer],
    ) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar)?;
        if self.reachable() {
            let mut values: Vec<_> = args.iter().map(|&r| self.read(r)).collect();
            for &buffer in buffers {
                let (ptr, len) = self.slice(buffer);
                values.extend([ptr, len]);
            }
            let call = self.graph.add_call(callee, &values);
            self.define(dst, Op::Call(call));
        }
        Ok(dst)
    }

    fn context_call(
        &mut self,
        target: RustContextTarget,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar)?;
        if !self.reachable() {
            return Ok(dst);
        }
        let (context, operation) = match target {
            RustContextTarget::Borrowed { context, operation } => {
                (Some(self.slice(context).0), operation)
            }
            RustContextTarget::Static(_) => (None, 0),
        };
        let operation = self.value(Op::Const(operation as i32));
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
            let packet = self.value(Op::CallScratch {
                offset: 0,
                bytes: layout.bytes.max(8),
            });
            let store = |b: &mut Self, offset, value| {
                let offset = b.value(Op::Const(offset as i32));
                let ptr = b.value(Op::PtrOffset(packet, offset, hir::Scale::One));
                b.effect(Op::Store(ptr, value));
            };
            for (index, &reg) in args.iter().enumerate() {
                let value = self.read(reg);
                store(self, index as u32 * 4, value);
            }
            for (index, &ContextSlice { buffer, writable }) in slices.iter().enumerate() {
                let (ptr, len) = self.slice(buffer);
                let tag = ContextArguments::view_tag(self.kinds[buffer.index()].element(), writable)
                    as i32;
                let tag = self.value(Op::Const(tag));
                let offset = layout.slices_at + index as u32 * ContextArguments::SLICE_BYTES;
                store(self, offset, ptr);
                store(self, offset + ContextArguments::LENGTH_OFFSET, len);
                store(self, offset + ContextArguments::ELEMENT_OFFSET, tag);
            }
            let zero = self.value(Op::Const(0));
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
        Ok(dst)
    }

    /// The two ABI words of a scalar Rust call. Unused words are zero.
    fn rust_operands(&mut self, args: &[Reg]) -> [Value; 2] {
        let mut words = [None; 2];
        for (word, &arg) in words.iter_mut().zip(args) {
            *word = Some(self.read(arg));
        }
        self.shared_loads.clear();
        words.map(|word| word.unwrap_or_else(|| self.value(Op::Const(0))))
    }
    fn staged_call(
        &mut self,
        signature: Signature<'_>,
        args: &[Reg],
        buffers: &[Buffer],
        key: Option<Specialization>,
        body: Body<'_>,
    ) -> splice_core::Result<Reg> {
        let callee = if let Some(hit) = key.as_ref().and_then(|k| self.callees.get(k)) {
            hit.clone()
        } else {
            let callee = self.nested(signature, body)?;
            if let Some(key) = key {
                self.callees.insert(key, callee.clone());
            }
            callee
        };
        self.call(callee, args, buffers)
    }
    fn target(&mut self, label: Label) -> Block {
        if let Some(block) = self.labels[label.0 as usize] {
            return block;
        }
        let block = self.block();
        self.labels[label.0 as usize] = Some(block);
        block
    }
    fn end(&mut self, exit: Exit) {
        if let Some(b) = self.current.take() {
            self.blocks[b].exit = Some(exit);
            self.constants.clear();
            self.shared_loads.clear();
        }
    }
    fn effect(&mut self, op: Op) {
        self.graph.append(self.current.expect("open block"), op);
    }
    /// A block for control flow inside one operation.
    fn block(&mut self) -> Block {
        let block = self.graph.create_block();
        let state = self.blocks.insert(State::default());
        debug_assert_eq!(state, block);
        block
    }
    /// Continue in `block`, falling through from the open block.
    fn begin(&mut self, block: Block) {
        self.end(Exit::Jump(block));
        self.current = Some(block);
    }
}

impl KernelBuilder for Builder<'_> {
    fn set_source(&mut self, source: &'static KernelSource) {
        self.source = Some(source);
        if let Some(sources) = &mut self.debug {
            sources[self.marker].function = Some(source);
        }
    }
    fn debug_enabled(&self) -> bool {
        self.debug.is_some()
    }
    fn debug_point(
        &mut self,
        location: splice_core::debug::Location,
        variables: &[splice_core::debug::Variable],
    ) -> splice_core::Result<()> {
        if self.debug.is_none() || !self.reachable() {
            return Ok(());
        }
        let point = splice_core::debug::Point {
            location,
            variables: variables.into(),
        };
        let slots: Vec<_> = point
            .slots(|r| self.types[r.0 as usize], self.kinds, self.vector_bytes)
            .collect();
        for slot in &slots {
            let offset = u32::try_from(slot.offset)
                .ok()
                .filter(|offset| offset.checked_add(8).is_some())
                .ok_or_else(|| splice_core::BuildError::unsupported("debug frame too large"))?;
            match slot.variable.value {
                splice_core::debug::Value::Register(reg, _) => {
                    let mut value = self.observe(reg);
                    match slot.ty {
                        splice_core::debug::Type::Scalar(splice_core::debug::Scalar::Index) => {
                            value = self.value(Op::Cast(hir::Type::I64, value));
                        }
                        // A DWARF bool is one byte holding 0 or 1.
                        splice_core::debug::Type::Scalar(splice_core::debug::Scalar::Bool) => {
                            let one = self.value(Op::Const(1));
                            value = self.value(Op::Bin(hir::Binary::And, value, one));
                        }
                        _ => {}
                    }
                    self.effect(Op::DebugStore(offset, value));
                }
                splice_core::debug::Value::Slice(buffer) => {
                    let (data, len) = self.slice(buffer);
                    self.effect(Op::DebugStore(offset, data));
                    let len = self.value(Op::Cast(hir::Type::I64, len));
                    self.effect(Op::DebugStore(offset + 8, len));
                }
                splice_core::debug::Value::Constant(_) => {
                    unreachable!("only runtime values have slots")
                }
            }
        }
        let sources = self.debug.as_mut().expect("debug build");
        let id = u32::try_from(sources.len())
            .map_err(|_| splice_core::BuildError::unsupported("too many debug points"))?;
        sources.push(DebugSource {
            function: sources[self.marker].function,
            point: Some((point, slots)),
        });
        self.effect(Op::DebugPoint(id));
        Ok(())
    }
    fn reg_type(&self, r: Reg) -> Option<ValueType> {
        self.types.get(r.0 as usize).copied()
    }
    fn local(&mut self, ty: ValueType) -> splice_core::Result<Reg> {
        let r = u32::try_from(self.types.len())
            .map(Reg)
            .map_err(|_| splice_core::BuildError::unsupported("too many registers"))?;
        self.types.push(ty);
        Ok(r)
    }
    fn simd(&mut self, inst: Simd, args: &[Reg]) -> splice_core::Result<Option<Reg>> {
        let value = self.emit_simd(inst, args)?;
        let dst = inst.output().map(|ty| self.local(ty)).transpose()?;
        self.bind_simd(dst, value);
        Ok(dst)
    }
    fn constant(&mut self, n: i32) -> splice_core::Result<Reg> {
        self.result(Op::Const(n))
    }
    fn buffer_len(&mut self, buffer: Buffer) -> splice_core::Result<Reg> {
        let (_, len) = self.slice(buffer);
        self.result(Op::Copy(len))
    }
    fn load(&mut self, buffer: Buffer, index: Reg) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar)?;
        if self.reachable() {
            let index = self.read(index);
            self.define_load(dst, buffer, index);
        }
        Ok(dst)
    }
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) -> splice_core::Result<()> {
        if self.reachable() {
            let (index, value) = (self.read(index), self.read(value));
            let addr = self.address(buffer, index);
            self.effect(Op::Store(addr, value));
        }
        Ok(())
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar)?;
        if self.reachable() {
            self.arithmetic(dst, op, a, b);
        }
        Ok(dst)
    }
    fn assign(&mut self, dst: Reg, src: Reg) -> splice_core::Result<()> {
        if self.reachable() {
            let v = self.read(src);
            self.bind(dst, v);
        }
        Ok(())
    }
    fn label(&mut self) -> splice_core::Result<Label> {
        let label = u32::try_from(self.labels.len())
            .map(Label)
            .map_err(|_| splice_core::BuildError::unsupported("too many labels"))?;
        self.labels.push(None);
        Ok(label)
    }
    fn mark(&mut self, label: Label) -> splice_core::Result<()> {
        let block = self.target(label);
        self.begin(block);
        Ok(())
    }
    fn jump(&mut self, label: Label) -> splice_core::Result<()> {
        let b = self.target(label);
        self.end(Exit::Jump(b));
        Ok(())
    }
    fn branch(&mut self, cond: Reg, yes: Label, no: Label) -> splice_core::Result<()> {
        if self.reachable() {
            let cond = self.read(cond);
            let (yes, no) = (self.target(yes), self.target(no));
            self.end(Exit::Branch(cond, yes, no));
        }
        Ok(())
    }
    fn ret(&mut self, value: Reg) -> splice_core::Result<()> {
        if self.reachable() {
            let v = self.read(value);
            self.end(Exit::Return(v));
        }
        Ok(())
    }
    fn trap_bounds(&mut self) -> splice_core::Result<()> {
        if self.reachable() {
            let zero = self.value(Op::Const(0));
            self.effect(Op::BoundsCheck(zero, zero));
            self.end(Exit::Return(zero));
        }
        Ok(())
    }
    fn rust_call(&mut self, import: RustCall, args: &[Reg]) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar)?;
        if self.reachable() {
            let [a, b] = self.rust_operands(args);
            self.define(dst, Op::RustCall(import.symbol, a, b));
        }
        Ok(dst)
    }
    fn rust_packet_call(
        &mut self,
        import: &'static RustPacketSymbol,
        args: &[Reg],
    ) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar)?;
        if self.reachable() {
            self.shared_loads.clear();
            let args: Vec<_> = args.iter().map(|&arg| self.read(arg)).collect();
            let call = self.graph.add_packet_call(import, 0, &args);
            self.define(dst, Op::RustPacketCall(call));
        }
        Ok(dst)
    }
    fn rust_context_call(
        &mut self,
        context: Buffer,
        operation: u32,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        self.context_call(
            RustContextTarget::Borrowed { context, operation },
            args,
            slices,
        )
    }
    fn rust_static_call(
        &mut self,
        symbol: &'static StaticRustSymbol,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        self.context_call(RustContextTarget::Static(symbol), args, slices)
    }
    fn call_kernel(
        &mut self,
        signature: Signature<'_>,
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> splice_core::Result<Reg> {
        self.staged_call(signature, args, buffers, None, body)
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> splice_core::Result<Reg> {
        self.staged_call(key.signature(), args, buffers, Some(key), body)
    }
    fn call_function(
        &mut self,
        function: &Arc<splice_core::Function>,
        args: &[Reg],
        buffers: &[Buffer],
    ) -> splice_core::Result<Reg> {
        let key = Arc::as_ptr(function);
        let callee = match self.functions.get(&key) {
            Some((_, callee)) => callee.clone(),
            None => {
                let callee = self.nested(function.signature(), |b| {
                    splice_core::hir::replay(function, b)
                })?;
                self.functions
                    .insert(key, (Arc::clone(function), callee.clone()));
                callee
            }
        };
        self.call(callee, args, buffers)
    }
}
