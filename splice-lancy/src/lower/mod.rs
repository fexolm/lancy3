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
    BufferKind,
    frontend::{Body, KernelBuilder, KernelSource, Signature, Specialization},
    hir::{
        Binary, Buffer, Constant, ContextSlice, Label, PacketStore, PacketValue, Reg, RustTarget,
        ScalarType, Simd, ValueType,
    },
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// The Lancy type of a scalar register. A `bool` is an `i32` mask, the
/// form of Lancy comparisons: zero or all bits set.
fn scalar_type(ty: ScalarType) -> hir::Type {
    match ty {
        ScalarType::Bool | ScalarType::I32 => hir::Type::I32,
        ScalarType::U32 => hir::Type::U32,
        ScalarType::I64 => hir::Type::I64,
        ScalarType::U64 => hir::Type::U64,
        ScalarType::Index => hir::Type::Usize,
        ScalarType::F32 => hir::Type::F32,
        ScalarType::F64 => hir::Type::F64,
    }
}

/// The type twice as wide as an integer type, for overflow checks.
fn wide(ty: ScalarType) -> hir::Type {
    match ty {
        ScalarType::I32 => hir::Type::I64,
        ScalarType::U32 => hir::Type::U64,
        ScalarType::I64 => hir::Type::I128,
        _ => hir::Type::U128,
    }
}

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
    let mut builder = Builder::new(signature, true, vector_bytes, sources)?;
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
    /// A function with `signature`. The root function is the kernel's
    /// entry: its `bool` parameters and result are 0 or 1.
    fn new(
        signature: Signature<'a>,
        root: bool,
        vector_bytes: usize,
        debug: Option<Vec<DebugSource>>,
    ) -> splice_core::Result<Self> {
        let Signature {
            params: scalars,
            result,
            buffers: kinds,
        } = signature;
        let arity = scalars.len();
        let count = u32::try_from(arity)
            .map_err(|_| splice_core::BuildError::unsupported("too many kernel parameters"))?;
        if count == u32::MAX {
            return Err(splice_core::BuildError::unsupported(
                "too many kernel parameters",
            ));
        }
        let mut params: Vec<_> = scalars.iter().map(|&ty| scalar_type(ty)).collect();
        for _ in kinds {
            params.extend([hir::Type::Ptr, hir::Type::I32]);
        }
        let mut graph = Function::new(&params, scalar_type(result));
        // BufferArg enforces these borrows at entry; call-site validation
        // preserves them when a nested kernel receives forwarded buffers.
        let entry = graph.entry();
        let buffers = graph.params(entry)[arity..].chunks_exact(2);
        let slices = buffers
            .zip(kinds)
            .map(|(pair, kind)| hir::SliceParam {
                ptr: pair[0],
                len: pair[1],
                element: kind.element().bytes() as u8,
                writable: kind.writable(),
            })
            .collect();
        graph
            .set_slice_params(slices)
            .expect("buffer parameters are (Ptr, I32) pairs");
        let mut first = State::default();
        for (i, &ty) in scalars.iter().enumerate() {
            let mut v = graph.params(entry)[i];
            if root && ty == ScalarType::Bool {
                let zero = graph.append(entry, Op::Const(0)).unwrap();
                v = graph
                    .append(entry, Op::Bin(hir::Binary::Sub, zero, v))
                    .unwrap();
            }
            first.defs.insert(Reg(i as u32), v);
        }
        let mut blocks = DenseMap::new();
        let state = blocks.insert(first);
        debug_assert_eq!(state, entry);
        let mut builder = Self {
            graph,
            source: None,
            callees: HashMap::new(),
            functions: HashMap::new(),
            arity,
            root,
            result,
            kinds,
            vector_bytes,
            blocks,
            current: Some(entry),
            types: scalars.iter().map(|&ty| ValueType::Scalar(ty)).collect(),
            demands: Vec::new(),
            observed: HashMap::new(),
            unassigned: HashSet::new(),
            labels: Vec::new(),
            has_shared: kinds
                .iter()
                .any(|kind| !kind.writable() && kind.context_signature().is_none()),
            constants: HashMap::new(),
            indices: HashMap::new(),
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
            builder
                .graph
                .append_inst(entry, Op::DebugPoint(id))
                .expect("a debug point");
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
        let mut child = Builder::new(signature, false, self.vector_bytes, self.debug.take())?;
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

/// A Rust call returns an `i32` word.
const WORD: ValueType = ValueType::Scalar(ScalarType::I32);

struct Builder<'a> {
    graph: Function,
    source: Option<&'static KernelSource>,
    callees: HashMap<Specialization, Arc<Function>>,
    /// Lowered portable callees, by identity.
    functions: HashMap<*const splice_core::Function, (Arc<splice_core::Function>, Arc<Function>)>,
    arity: usize,
    /// Whether this is the kernel's entry rather than a callee.
    root: bool,
    result: ScalarType,
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
    /// Whether a read-only slice can share loads at one index.
    has_shared: bool,
    /// The `i32` constants of the open block.
    constants: HashMap<Value, i32>,
    /// The `i32` form of `Index` values (see `index32`).
    indices: HashMap<Value, Narrow>,
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

/// How an `Index` value relates to a Lancy `i32`.
#[derive(Clone, Copy)]
enum Narrow {
    /// The `Index` sign-extends `value`, which is nonnegative if `nonnegative`.
    Extends {
        value: Value,
        nonnegative: bool,
    },
    Constant(u64),
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

    /// The `i32` Lancy index of a `usize` slice index or offset, with the
    /// same bounds check: every slice length fits `i32`, so an `Index` past
    /// `i32::MAX` clamps to it. An `Index` that sign-extends an `i32` is out
    /// of bounds exactly when the `i32` is negative or too large.
    fn index32(&mut self, index: Value) -> Value {
        match self.indices.get(&index) {
            Some(&Narrow::Extends { value, .. }) => return value,
            Some(&Narrow::Constant(bits)) => {
                return self.const32(bits.min(i32::MAX as u64) as i32);
            }
            None => {}
        }
        let max = self.value(Op::Literal(hir::Lane::Usize, i32::MAX as u128));
        let clamped = self.value(Op::Bin(hir::Binary::Min, index, max));
        self.value(Op::Cast(hir::Type::I32, clamped))
    }
    fn const32(&mut self, n: i32) -> Value {
        let v = self.value(Op::Const(n));
        if self.has_shared {
            self.constants.insert(v, n);
        }
        v
    }
    /// The `i32` that `index` sign-extends, and whether it is nonnegative.
    fn exact32(&mut self, index: Value) -> Option<(Value, bool)> {
        match *self.indices.get(&index)? {
            Narrow::Extends { value, nonnegative } => Some((value, nonnegative)),
            Narrow::Constant(bits) => {
                let n = i32::try_from(bits).ok()?;
                Some((self.const32(n), true))
            }
        }
    }
    /// An `Index` comparison of two values with exact `i32` forms, done on
    /// those forms so that Lancy's `i32` bounds analysis sees the guard.
    /// Sign extension preserves unsigned order, so the unsigned `i32`
    /// comparison is exact; it is signed when both sides are nonnegative.
    fn compare32(&mut self, kind: hir::Binary, a: Value, b: Value) -> Option<Value> {
        // Check both before `exact32` materializes a constant.
        let exact =
            |n: Narrow| !matches!(n, Narrow::Constant(bits) if i32::try_from(bits).is_err());
        if !exact(*self.indices.get(&a)?) || !exact(*self.indices.get(&b)?) {
            return None;
        }
        let ((a, a_nonnegative), (b, b_nonnegative)) = (self.exact32(a)?, self.exact32(b)?);
        let ordered = !matches!(kind, hir::Binary::Eq | hir::Binary::Ne);
        let (a, b) = if ordered && !(a_nonnegative && b_nonnegative) {
            let bias = self.value(Op::Const(i32::MIN));
            (
                self.value(Op::Bin(hir::Binary::Xor, a, bias)),
                self.value(Op::Bin(hir::Binary::Xor, b, bias)),
            )
        } else {
            (a, b)
        };
        Some(self.value(Op::Bin(kind, a, b)))
    }
    fn address(&mut self, buffer: Buffer, index: Value) -> Value {
        let (base, len) = self.slice(buffer);
        let scale = simd::stride(self.kinds[buffer.index()].element());
        let block = self.current.expect("open block");
        self.graph
            .append_inst(block, Op::BoundsCheck(index, len))
            .expect("an i32 bounds check");
        self.graph
            .append(block, Op::PtrOffset(base, index, scale))
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
            .expect("a well-typed lowering of verified IR defines a value")
    }
    fn scalar(&self, r: Reg) -> ScalarType {
        match self.types[r.0 as usize] {
            ValueType::Scalar(ty) => ty,
            _ => unreachable!("validated scalar register"),
        }
    }
    /// A constant of `ty` with the given bits.
    fn literal(&mut self, ty: ScalarType, bits: u64) -> Value {
        match ty {
            ScalarType::Bool => self.value(Op::Const(-((bits & 1) as i32))),
            ScalarType::I32 => self.value(Op::Const(bits as u32 as i32)),
            _ => {
                let lane = scalar_type(ty).lane().expect("scalar lane");
                self.value(Op::literal(lane, bits.into()))
            }
        }
    }

    fn arithmetic(&mut self, dst: Reg, op: Binary, lhs: Reg, rhs: Reg) {
        let ty = self.scalar(lhs);
        let (a, b) = (self.read(lhs), self.read(rhs));
        let kind = binary(op);
        match op {
            Binary::CheckedAdd | Binary::CheckedSub | Binary::CheckedMul => {
                // The exact result in twice the width must survive narrowing.
                let (narrow_type, wide_type) = (scalar_type(ty), wide(ty));
                let a = self.value(Op::Cast(wide_type, a));
                let b = self.value(Op::Cast(wide_type, b));
                let exact = self.value(Op::Bin(kind, a, b));
                let narrow = self.value(Op::Cast(narrow_type, exact));
                let extended = self.value(Op::Cast(wide_type, narrow));
                let overflow = self.value(Op::Bin(hir::Binary::Ne, exact, extended));
                self.effect(Op::CheckOverflow(overflow));
                self.bind(dst, narrow);
            }
            Binary::CheckedShl | Binary::CheckedShr => {
                // The amount, as an unsigned value, must be below the width.
                let high_bits = self.literal(ty, !u64::from(ty.bits() - 1));
                let high = self.value(Op::Bin(hir::Binary::And, b, high_bits));
                let overflow = if ty == ScalarType::I32 {
                    high
                } else {
                    let zero = self.literal(ty, 0);
                    self.value(Op::Bin(hir::Binary::Ne, high, zero))
                };
                self.effect(Op::CheckOverflow(overflow));
                self.define(dst, Op::Bin(kind, a, b));
            }
            Binary::Eq | Binary::Ne | Binary::Lt | Binary::Le | Binary::Gt | Binary::Ge
                if ty == ScalarType::Index =>
            {
                let v = self
                    .compare32(kind, a, b)
                    .unwrap_or_else(|| self.value(Op::Bin(kind, a, b)));
                self.bind(dst, v);
            }
            _ => self.define(dst, Op::Bin(kind, a, b)),
        }
    }
    /// Rust's `src as to`.
    fn convert(&mut self, dst: Reg, src: Reg, to: ScalarType) {
        let from = self.scalar(src);
        let v = self.read(src);
        let value = if from == to {
            v
        } else if from == ScalarType::Bool {
            // The mask becomes 0 or 1.
            let one = self.value(Op::Const(1));
            let bit = self.value(Op::Bin(hir::Binary::And, v, one));
            if to == ScalarType::I32 {
                bit
            } else {
                self.value(Op::Cast(scalar_type(to), bit))
            }
        } else if let (ScalarType::Index, ScalarType::I32, Some(&Narrow::Extends { value, .. })) =
            (from, to, self.indices.get(&v))
        {
            value
        } else {
            self.value(Op::Cast(scalar_type(to), v))
        };
        if from == ScalarType::I32 && to == ScalarType::Index {
            let narrow = Narrow::Extends {
                value: v,
                nonnegative: false,
            };
            self.indices.insert(value, narrow);
        }
        self.bind(dst, value);
    }
    fn bind(&mut self, r: Reg, v: Value) {
        let b = self.current.expect("open block");
        self.blocks[b].defs.insert(r, v);
    }
    fn define(&mut self, r: Reg, op: Op) {
        let v = self.value(op);
        if let Op::Const(n) = op
            && self.has_shared
        {
            self.constants.insert(v, n);
        }
        self.bind(r, v);
    }
    fn define_load(&mut self, dst: Reg, buffer: Buffer, index: Value) {
        let shared = !self.kinds[buffer.index()].writable();
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
        let ty = scalar_type(self.scalar(dst));
        self.define(dst, Op::Load(ty, address));
        if let Some(key) = key {
            let block = self.current.expect("open block");
            self.shared_loads.insert(key, self.blocks[block].defs[&dst]);
        }
    }
    fn call(
        &mut self,
        callee: Arc<Function>,
        result: ScalarType,
        args: &[Reg],
        buffers: &[Buffer],
    ) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar(result))?;
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
        self.call(callee, signature.result, args, buffers)
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
        self.graph
            .append_inst(self.current.expect("open block"), op)
            .expect("a well-typed lowering of verified IR");
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
            let types = &self.types;
            let stores: Vec<_> = slot.stores(|r| types[r.0 as usize]).collect();
            for (at, store) in stores {
                let value = match store {
                    splice_core::debug::Store::Scalar(reg, ScalarType::Bool) => {
                        // A comparison mask: zero or all bits set.
                        let value = self.observe(reg);
                        let one = self.value(Op::Const(1));
                        let bit = self.value(Op::Bin(hir::Binary::And, value, one));
                        self.value(Op::Cast(hir::Type::U8, bit))
                    }
                    // A Lancy mask is already a vector of all-ones or zero
                    // index lanes.
                    splice_core::debug::Store::Scalar(reg, _)
                    | splice_core::debug::Store::Vector(reg)
                    | splice_core::debug::Store::Mask(reg, _) => self.observe(reg),
                    splice_core::debug::Store::Data(buffer) => self.slice(buffer).0,
                    splice_core::debug::Store::Length(buffer) => {
                        let len = self.slice(buffer).1;
                        self.value(Op::Cast(hir::Type::I64, len))
                    }
                };
                let at = u32::try_from(at)
                    .ok()
                    .filter(|at| at.checked_add(8).is_some())
                    .ok_or_else(|| splice_core::BuildError::unsupported("debug frame too large"))?;
                self.effect(Op::DebugStore(at, value));
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
    fn constant(&mut self, c: Constant) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar(c.ty()))?;
        if self.reachable() {
            let v = self.literal(c.ty(), c.bits());
            if c.ty() == ScalarType::Index {
                self.indices.insert(v, Narrow::Constant(c.bits()));
            }
            self.bind(dst, v);
        }
        Ok(dst)
    }
    fn cast(&mut self, src: Reg, to: ScalarType) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar(to))?;
        if self.reachable() {
            self.convert(dst, src, to);
        }
        Ok(dst)
    }
    fn buffer_len(&mut self, buffer: Buffer) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar(ScalarType::Index))?;
        if self.reachable() {
            let (_, len) = self.slice(buffer);
            let v = self.extend_index(len);
            self.bind(dst, v);
        }
        Ok(dst)
    }
    fn load(&mut self, buffer: Buffer, index: Reg) -> splice_core::Result<Reg> {
        let ty = ScalarType::of_element(self.kinds[buffer.index()].element())
            .expect("validated scalar element");
        let dst = self.local(ValueType::Scalar(ty))?;
        if self.reachable() {
            let index = self.read(index);
            let index = self.index32(index);
            self.define_load(dst, buffer, index);
        }
        Ok(dst)
    }
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) -> splice_core::Result<()> {
        if self.reachable() {
            let (index, value) = (self.read(index), self.read(value));
            let index = self.index32(index);
            let addr = self.address(buffer, index);
            self.effect(Op::Store(addr, value));
        }
        Ok(())
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> splice_core::Result<Reg> {
        let output = op.output(self.scalar(a)).expect("validated operation");
        let dst = self.local(ValueType::Scalar(output))?;
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
            let mut v = self.read(value);
            if self.root && self.result == ScalarType::Bool {
                let one = self.value(Op::Const(1));
                v = self.value(Op::Bin(hir::Binary::And, v, one));
            }
            self.end(Exit::Return(v));
        }
        Ok(())
    }
    fn trap_bounds(&mut self) -> splice_core::Result<()> {
        if self.reachable() {
            let zero = self.value(Op::Const(0));
            self.effect(Op::BoundsCheck(zero, zero));
            let result = self.zero(ValueType::Scalar(self.result));
            self.end(Exit::Return(result));
        }
        Ok(())
    }
    /// A Rust call as [`RustTarget::plan`] lays it out: the operands go to
    /// a call-scratch packet and the native entry receives its address.
    fn rust_call(
        &mut self,
        target: RustTarget,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        let dst = self.local(WORD)?;
        if !self.reachable() {
            return Ok(dst);
        }
        let plan = target.plan(self.kinds, args.len(), slices);
        // The callee may write any borrowed slice.
        self.shared_loads.clear();
        let packet = self.value(Op::CallScratch {
            offset: 0,
            bytes: plan.bytes,
        });
        let store = |b: &mut Self, at: u32, value| {
            let offset = b.value(Op::Const(at as i32));
            let ptr = b.value(Op::PtrOffset(packet, offset, hir::Scale::One));
            b.effect(Op::Store(ptr, value));
        };
        for PacketStore { at, value } in plan.stores {
            match value {
                PacketValue::Word(n) => {
                    let word = self.read(args[n]);
                    store(self, at, word);
                }
                PacketValue::Data(buffer) => {
                    let ptr = self.slice(buffer).0;
                    store(self, at, ptr);
                }
                PacketValue::Length(buffer) => {
                    let len = self.slice(buffer).1;
                    store(self, at, len);
                }
                PacketValue::Tag(tag) => {
                    let tag = self.value(Op::Const(tag as i32));
                    store(self, at, tag);
                }
                PacketValue::Symbol(address) => {
                    let address = self.value(Op::Literal(hir::Lane::U64, address as u128));
                    store(self, at, address);
                }
            }
        }
        let receiver = match plan.context {
            Some(context) => self.slice(context).0,
            None => packet,
        };
        let operation = self.value(Op::Const(plan.operation as i32));
        let zero = self.value(Op::Const(0));
        self.define(
            dst,
            Op::RustContextCall(
                hir::RustContextImport::Packet(plan.entry),
                [receiver, operation, packet, zero],
            ),
        );
        Ok(dst)
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
        self.call(callee, function.result(), args, buffers)
    }
}
