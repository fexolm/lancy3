//! The one verifier of Splice kernels. Every operation a backend's builder
//! receives was checked here first, on the direct build path and when a
//! portable function is verified (which replays it through this check).
use crate::frontend::{Body, KernelBuilder, KernelSource, Signature, Specialization};
use crate::hir::Intrinsic;
use crate::hir::{
    Binary, BufferKind, Constant, ContextSlice, Function, Label, Reg, RustTarget, ScalarType,
    Slice, ValueType,
};
use crate::{BuildError, Result};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Check `body` as it stages and when it completes.
/// `overflow_checks` is the build's overflow mode, passed on to nested
/// bodies (see [`KernelBuilder::overflow_checks`]).
pub fn validated<'a>(
    signature: Signature<'a>,
    body: Body<'a>,
    overflow_checks: Option<bool>,
) -> Body<'a> {
    Box::new(move |builder| {
        let mut validating = Validating::new(builder, signature)?;
        validating.overflow_checks = overflow_checks;
        body(&mut validating)?;
        validating.finish()
    })
}

/// A builder that checks each operation (types, registers, buffers,
/// labels and calls) before passing it to `inner`, and the definite
/// initialization of every register at [`finish`](Self::finish).
///
/// The position of an error is the index of its instruction in the
/// portable function the same operations record.
pub struct Validating<'a> {
    inner: &'a mut dyn KernelBuilder,
    /// The number of parameter registers: scalars, then slice pairs.
    params: usize,
    buffers: usize,
    result: ScalarType,
    types: Vec<ValueType>,
    /// The block each label starts, once marked.
    labels: Vec<Option<usize>>,
    blocks: Vec<Block>,
    current: Option<usize>,
    pc: usize,
    overflow_checks: Option<bool>,
    /// The registers `local` allocated: the only ones `assign` may write.
    /// Every other register has the one value its operation defines.
    locals: HashSet<Reg>,
    /// The length register of each slice pointer: a buffer parameter's
    /// pair, or the pair one [`KernelBuilder::slice_range`] produced. A
    /// slice operand must be one of these pairs, so the length a backend
    /// checks an access against is the length of the memory it addresses.
    /// Neither register of a pair is a local, so no `assign` changes one.
    pairs: HashMap<Reg, Reg>,
}

#[derive(Clone, Copy)]
enum Exit {
    /// The function ends without a terminator.
    Open,
    Return,
    Jump(Label, usize),
    Branch(Label, Label, usize),
}

/// The control-flow skeleton of one block: no operations are kept.
struct Block {
    defined: HashSet<Reg>,
    /// First use of each register the block reads before writing it.
    exposed: Vec<(Reg, usize)>,
    exit: Exit,
}

impl Block {
    fn new() -> Self {
        Self {
            defined: HashSet::new(),
            exposed: Vec::new(),
            exit: Exit::Open,
        }
    }
}

impl<'a> Validating<'a> {
    pub fn new(inner: &'a mut dyn KernelBuilder, signature: Signature<'a>) -> Result<Self> {
        if signature.params.len() >= u32::MAX as usize {
            return Err(BuildError::verify("too many scalar parameters"));
        }
        let params = signature.registers();
        if signature.buffers.len() >= u32::MAX as usize / 2 || params >= u32::MAX as usize - 1 {
            return Err(BuildError::verify("too many buffer parameters"));
        }
        Ok(Self {
            inner,
            params,
            buffers: signature.buffers.len(),
            result: signature.result,
            types: (0..params)
                .map(|r| signature.param_type(Reg(r as u32)).expect("a parameter"))
                .collect(),
            labels: Vec::new(),
            blocks: vec![Block::new()],
            current: Some(0),
            pc: 0,
            overflow_checks: None,
            locals: HashSet::new(),
            pairs: (0..signature.buffers.len())
                .map(|i| Slice::param(signature.params.len(), i))
                .map(|s| (s.ptr, s.len))
                .collect(),
        })
    }

    fn error(&self, reason: impl Into<String>) -> BuildError {
        BuildError::verify_at(self.pc, reason)
    }

    fn at<T>(&self, result: Result<T>) -> Result<T> {
        result.map_err(|e| e.at(self.pc))
    }

    fn ty(&self, r: Reg) -> Result<ValueType> {
        self.types
            .get(r.0 as usize)
            .copied()
            .ok_or_else(|| self.error("invalid register"))
    }

    fn types(&self, rs: &[Reg]) -> Result<Vec<ValueType>> {
        rs.iter().map(|&r| self.ty(r)).collect()
    }

    fn scalar(&self, r: Reg) -> Result<ScalarType> {
        match self.ty(r)? {
            ValueType::Scalar(ty) => Ok(ty),
            _ => Err(self.error("expected a scalar register")),
        }
    }

    fn expect(&self, r: Reg, want: ScalarType) -> Result<()> {
        let ty = self.scalar(r)?;
        if ty != want {
            return Err(self.error(format!("expected a register of type {want}, got {ty}")));
        }
        Ok(())
    }

    fn block(&mut self) -> &mut Block {
        let index = *self.current.get_or_insert_with(|| {
            // Operations after a terminator form a block no edge reaches.
            self.blocks.push(Block::new());
            self.blocks.len() - 1
        });
        &mut self.blocks[index]
    }

    /// Record the operands and the written register of the next instruction.
    fn record(&mut self, reads: &[Reg], written: Option<Reg>) {
        let pc = self.pc;
        let block = self.block();
        for &r in reads {
            if !block.defined.contains(&r) && !block.exposed.iter().any(|&(e, _)| e == r) {
                block.exposed.push((r, pc));
            }
        }
        block.defined.extend(written);
        self.pc += 1;
    }

    /// The builder allocated `r` for a value of type `ty`.
    fn allocated(&mut self, r: Reg, ty: ValueType) -> Reg {
        debug_assert_eq!(
            r.0 as usize,
            self.types.len(),
            "builders number registers densely"
        );
        self.types.push(ty);
        r
    }

    fn value(&mut self, reads: &[Reg], r: Reg, ty: ValueType) -> Reg {
        let r = self.allocated(r, ty);
        self.record(reads, Some(r));
        r
    }

    fn label_index(&self, l: Label) -> Result<usize> {
        let index = l.0 as usize;
        if index < self.labels.len() {
            Ok(index)
        } else {
            Err(self.error("invalid label index"))
        }
    }

    fn close(&mut self, reads: &[Reg], exit: Exit) {
        self.block().exit = exit;
        self.record(reads, None);
        self.current = None;
    }

    fn reg_types(&self) -> impl Fn(Reg) -> Option<ValueType> + '_ {
        |r: Reg| self.types.get(r.0 as usize).copied()
    }

    /// Record the fresh pair `dst` of a slice of `kind` that one operation
    /// reading `reads` produced.
    fn define_pair(&mut self, dst: Slice, kind: BufferKind, reads: &[Reg]) {
        self.allocated(dst.ptr, ValueType::Ptr(kind));
        self.allocated(dst.len, ValueType::Scalar(ScalarType::Index));
        self.pairs.insert(dst.ptr, dst.len);
        self.block().defined.insert(dst.len);
        self.record(reads, Some(dst.ptr));
    }

    /// Reject a slice whose pointer and length were not produced together.
    fn paired(&self, slice: Slice) -> Result<()> {
        if self.pairs.get(&slice.ptr) == Some(&slice.len) {
            Ok(())
        } else {
            Err(self.error(
                "a slice operand must be a buffer parameter or the result of one slice operation",
            ))
        }
    }

    /// The kind of a slice whose registers are a valid pair.
    fn slice_kind(&self, slice: Slice) -> Result<BufferKind> {
        let kind = slice
            .kind(&self.reg_types())
            .ok_or_else(|| self.error("invalid buffer"))?;
        self.paired(slice)?;
        Ok(kind)
    }

    fn buffer(&self, slice: Slice) -> Result<BufferKind> {
        let kind = self.slice_kind(slice)?;
        if kind.context_signature().is_some() {
            return Err(self.error("slice memory operation cannot use Rust context"));
        }
        Ok(kind)
    }

    /// The type of the elements of `slice`, accessed one at a time.
    fn element(&self, slice: Slice, write: bool) -> Result<ScalarType> {
        let kind = self.buffer(slice)?;
        let ty = kind
            .scalar()
            .ok_or_else(|| self.error("a slice element is read with `load_slice`"))?;
        if write && !kind.writable() {
            return Err(self.error("store to read-only buffer"));
        }
        Ok(ty)
    }

    fn call_arguments(&self, callee: Signature<'_>, args: &[Reg], buffers: &[Slice]) -> Result<()> {
        for &slice in buffers {
            self.paired(slice)?;
        }
        let types: Vec<_> = args
            .iter()
            .map(|&r| self.types.get(r.0 as usize).copied())
            .collect();
        self.at(crate::hir::check_call(
            callee,
            types.into_iter(),
            &self.reg_types(),
            buffers,
        ))
    }

    fn rust_operands(&self, args: &[Reg]) -> Vec<Option<ValueType>> {
        args.iter()
            .map(|&r| self.types.get(r.0 as usize).copied())
            .collect()
    }

    /// Check labels and the definite initialization of every register on
    /// every reachable path.
    pub fn finish(self) -> Result<()> {
        let resolve = |l: Label, pc: usize| {
            self.labels[l.0 as usize]
                .ok_or_else(|| BuildError::verify_at(pc, format!("undefined label {l:?}")))
        };
        let mut edges = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            edges.push(match block.exit {
                Exit::Open => None,
                Exit::Return => Some([None, None]),
                Exit::Jump(l, pc) => Some([Some(resolve(l, pc)?), None]),
                Exit::Branch(yes, no, pc) => {
                    Some([Some(resolve(yes, pc)?), Some(resolve(no, pc)?)])
                }
            });
        }
        let words = self.types.len().div_ceil(64);
        let bit = |bits: &[u64], r: Reg| bits[r.0 as usize / 64] & (1 << (r.0 % 64)) != 0;
        let mut entry = vec![0u64; words];
        for r in 0..self.params {
            entry[r / 64] |= 1 << (r % 64);
        }
        let mut states: Vec<Option<Vec<u64>>> = vec![None; self.blocks.len()];
        states[0] = Some(entry);
        let mut queue = VecDeque::from([0]);
        while let Some(index) = queue.pop_front() {
            let mut bits = states[index].clone().expect("queued block has a state");
            for &r in &self.blocks[index].defined {
                bits[r.0 as usize / 64] |= 1 << (r.0 % 64);
            }
            for next in edges[index].into_iter().flatten().flatten() {
                match &mut states[next] {
                    Some(state) => {
                        let mut changed = false;
                        for (a, b) in state.iter_mut().zip(&bits) {
                            changed |= *a & b != *a;
                            *a &= b;
                        }
                        if changed {
                            queue.push_back(next);
                        }
                    }
                    slot @ None => {
                        *slot = Some(bits.clone());
                        queue.push_back(next);
                    }
                }
            }
        }
        let uninitialized = self
            .blocks
            .iter()
            .zip(&states)
            .filter_map(|(block, state)| Some((block, state.as_ref()?)))
            .flat_map(|(block, state)| block.exposed.iter().filter(move |&&(r, _)| !bit(state, r)))
            .min_by_key(|&&(_, pc)| pc);
        if let Some(&(r, pc)) = uninitialized {
            return Err(BuildError::verify_at(pc, format!("uninitialized {r:?}")));
        }
        if self
            .blocks
            .iter()
            .zip(&states)
            .any(|(block, state)| state.is_some() && matches!(block.exit, Exit::Open))
        {
            return Err(BuildError::verify(
                "reachable path falls off the end of the function",
            ));
        }
        Ok(())
    }
}

impl KernelBuilder for Validating<'_> {
    fn set_source(&mut self, source: &'static KernelSource) {
        self.inner.set_source(source);
    }
    fn debug_enabled(&self) -> bool {
        self.inner.debug_enabled()
    }
    fn overflow_checks(&self) -> Option<bool> {
        self.overflow_checks
    }
    fn debug_point(
        &mut self,
        location: crate::debug::Location,
        variables: &[crate::debug::Variable],
    ) -> Result<()> {
        if !self.inner.debug_enabled() {
            return Ok(());
        }
        if location.file.is_empty()
            || location.directory.is_empty()
            || location.file.contains('\0')
            || location.directory.contains('\0')
            || location.line == 0
        {
            return Err(self.error("invalid debug source location"));
        }
        for var in variables {
            match var.value {
                crate::debug::Value::Register(r) if matches!(self.ty(r)?, ValueType::Ptr(_)) => {
                    return Err(self.error("a debug pointer is saved with its slice"));
                }
                crate::debug::Value::Register(_) => {}
                crate::debug::Value::Slice(slice)
                    if self.slice_kind(slice).map_or(true, |kind| {
                        kind.context_signature().is_some() || kind.scalar().is_none()
                    }) =>
                {
                    return Err(self.error("invalid debug buffer"));
                }
                _ => {}
            }
        }
        self.inner.debug_point(location, variables)?;
        // A source point observes its variables: they need not be assigned.
        self.record(&[], None);
        Ok(())
    }
    fn arg(&self, index: usize) -> Result<Reg> {
        if index < self.params {
            self.inner.arg(index)
        } else {
            Err(BuildError::verify("invalid parameter index"))
        }
    }
    fn buffer(&self, index: usize) -> Result<Slice> {
        if index < self.buffers {
            self.inner.buffer(index)
        } else {
            Err(BuildError::verify("invalid buffer index"))
        }
    }
    fn reg_type(&self, r: Reg) -> Option<ValueType> {
        self.types.get(r.0 as usize).copied()
    }
    fn local(&mut self, ty: ValueType) -> Result<Reg> {
        if let ValueType::Vector(e) | ValueType::Mask(e) = ty
            && !ty.is_supported()
        {
            return Err(BuildError::unsupported(format!(
                "SIMD lanes of {} are not supported",
                e.name()
            )));
        }
        if let ValueType::Ptr(_) = ty {
            return Err(BuildError::verify(
                "a slice pointer is produced with its length by a slice operation",
            ));
        }
        let r = self.inner.local(ty)?;
        self.locals.insert(r);
        Ok(self.allocated(r, ty))
    }
    fn call_intrinsic(&mut self, inst: Intrinsic, args: &[Reg]) -> Result<Option<Reg>> {
        let types = self.types(args)?;
        if let Intrinsic::Simd(simd) = inst
            && let Some(memory) = simd.op.info().memory
        {
            self.paired(memory.slice)?;
        }
        self.at(inst.validate(&types, &self.reg_types()))?;
        let dst = self.inner.call_intrinsic(inst, args)?;
        let mut reads = args.to_vec();
        if let Intrinsic::Simd(simd) = inst
            && let Some(memory) = simd.op.info().memory
        {
            reads.extend([memory.slice.ptr, memory.slice.len]);
        }
        match (dst, inst.output()) {
            (Some(dst), Some(ty)) => {
                self.value(&reads, dst, ty);
            }
            (None, None) => self.record(&reads, None),
            _ => return Err(self.error("invalid intrinsic result type")),
        }
        Ok(dst)
    }
    fn constant(&mut self, value: Constant) -> Result<Reg> {
        // A kernel's `char`s reach `&mut [char]` and the result word that
        // `KernelScalar::from_word` decodes: each must be a Unicode scalar value.
        if value.ty() == ScalarType::Char && char::from_u32(value.bits() as u32).is_none() {
            return Err(self.error(format!("{:#x} is not a char", value.bits())));
        }
        let r = self.inner.constant(value)?;
        Ok(self.value(&[], r, ValueType::Scalar(value.ty())))
    }
    fn cast(&mut self, value: Reg, to: ScalarType) -> Result<Reg> {
        let from = self.scalar(value)?;
        if !from.casts_to(to) {
            return Err(self.error(format!("Rust has no cast from {from} to {to}")));
        }
        let r = self.inner.cast(value, to)?;
        Ok(self.value(&[value], r, ValueType::Scalar(to)))
    }
    fn slice_range(&mut self, slice: Slice, start: Reg, end: Reg) -> Result<Slice> {
        let kind = self.buffer(slice)?;
        self.expect(start, ScalarType::Index)?;
        self.expect(end, ScalarType::Index)?;
        let dst = self.inner.slice_range(slice, start, end)?;
        self.define_pair(dst, kind, &[slice.ptr, slice.len, start, end]);
        Ok(dst)
    }
    fn load_slice(&mut self, slice: Slice, index: Reg) -> Result<Slice> {
        self.expect(index, ScalarType::Index)?;
        let crate::hir::Element::Slice(&inner) = self.buffer(slice)?.element() else {
            return Err(self.error("`load_slice` reads a slice of slices"));
        };
        let dst = self.inner.load_slice(slice, index)?;
        self.define_pair(dst, inner, &[slice.ptr, slice.len, index]);
        Ok(dst)
    }
    fn load(&mut self, slice: Slice, index: Reg) -> Result<Reg> {
        self.expect(index, ScalarType::Index)?;
        let ty = self.element(slice, false)?;
        let r = self.inner.load(slice, index)?;
        Ok(self.value(&[slice.ptr, slice.len, index], r, ValueType::Scalar(ty)))
    }
    fn store(&mut self, slice: Slice, index: Reg, value: Reg) -> Result<()> {
        self.expect(index, ScalarType::Index)?;
        let ty = self.element(slice, true)?;
        self.expect(value, ty)?;
        self.inner.store(slice, index, value)?;
        self.record(&[slice.ptr, slice.len, index, value], None);
        Ok(())
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Result<Reg> {
        let ty = self.scalar(a)?;
        self.expect(b, ty)?;
        let output = op
            .output(ty)
            .ok_or_else(|| self.error(format!("{op:?} does not apply to {ty}")))?;
        let r = self.inner.binary(op, a, b)?;
        Ok(self.value(&[a, b], r, ValueType::Scalar(output)))
    }
    fn assign(&mut self, dst: Reg, src: Reg) -> Result<()> {
        if self.ty(dst)? != self.ty(src)? {
            return Err(self.error("register copy type mismatch"));
        }
        // A buffer parameter's pair holds the caller's slice for the whole
        // body, so a backend may read it where it is defined.
        if (self.params - 2 * self.buffers..self.params).contains(&(dst.0 as usize)) {
            return Err(self.error("a buffer parameter's registers are read-only"));
        }
        // A backend may read an operation's result where it is defined;
        // scalar parameters and locals are variables.
        if dst.0 as usize >= self.params && !self.locals.contains(&dst) {
            return Err(self.error("only a local register can be assigned"));
        }
        self.inner.assign(dst, src)?;
        self.record(&[src], Some(dst));
        Ok(())
    }
    fn label(&mut self) -> Result<Label> {
        let l = self.inner.label()?;
        debug_assert_eq!(
            l.0 as usize,
            self.labels.len(),
            "builders number labels densely"
        );
        self.labels.push(None);
        Ok(l)
    }
    fn mark(&mut self, l: Label) -> Result<()> {
        let index = self.label_index(l)?;
        if self.labels[index].is_some() {
            return Err(self.error(format!("duplicate label {l:?}")));
        }
        self.inner.mark(l)?;
        if let Some(open) = self.current {
            // Execution falls through into the labeled block.
            self.blocks[open].exit = Exit::Jump(l, self.pc);
        }
        self.blocks.push(Block::new());
        self.current = Some(self.blocks.len() - 1);
        self.labels[index] = self.current;
        self.record(&[], None);
        Ok(())
    }
    fn jump(&mut self, l: Label) -> Result<()> {
        self.label_index(l)?;
        self.inner.jump(l)?;
        self.close(&[], Exit::Jump(l, self.pc));
        Ok(())
    }
    fn branch(&mut self, cond: Reg, yes: Label, no: Label) -> Result<()> {
        self.expect(cond, ScalarType::Bool)?;
        self.label_index(yes)?;
        self.label_index(no)?;
        self.inner.branch(cond, yes, no)?;
        self.close(&[cond], Exit::Branch(yes, no, self.pc));
        Ok(())
    }
    fn ret(&mut self, value: Reg) -> Result<()> {
        self.expect(value, self.result)?;
        self.inner.ret(value)?;
        self.close(&[value], Exit::Return);
        Ok(())
    }
    fn trap_bounds(&mut self) -> Result<()> {
        self.inner.trap_bounds()?;
        self.close(&[], Exit::Return);
        Ok(())
    }
    fn rust_call(
        &mut self,
        target: RustTarget,
        args: &[Reg],
        slices: &[ContextSlice],
        result: ScalarType,
    ) -> Result<Reg> {
        if let RustTarget::Context { context, .. } = target {
            self.paired(context)?;
        }
        for s in slices {
            if self.slice_kind(s.slice)?.scalar().is_none() {
                return Err(self.error("a Rust call takes slices of scalars"));
            }
        }
        let types = self.rust_operands(args);
        self.at(target.check(&self.reg_types(), types.into_iter(), slices))?;
        let r = self.inner.rust_call(target, args, slices, result)?;
        let mut reads = args.to_vec();
        if let RustTarget::Context { context, .. } = target {
            reads.extend([context.ptr, context.len]);
        }
        reads.extend(slices.iter().flat_map(|s| [s.slice.ptr, s.slice.len]));
        Ok(self.value(&reads, r, ValueType::Scalar(result)))
    }
    fn call_kernel(
        &mut self,
        signature: Signature<'_>,
        args: &[Reg],
        buffers: &[Slice],
        body: Body<'_>,
    ) -> Result<Reg> {
        self.call_arguments(signature, args, buffers)?;
        let r = self.inner.call_kernel(
            signature,
            args,
            buffers,
            validated(signature, body, self.overflow_checks),
        )?;
        Ok(self.value(
            &call_reads(args, buffers),
            r,
            ValueType::Scalar(signature.result),
        ))
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Slice],
        body: Body<'_>,
    ) -> Result<Reg> {
        let signature = key.signature();
        self.call_arguments(signature, args, buffers)?;
        let _staging = Staging::enter(&key)?;
        let r = self.inner.call_specialized(
            key,
            args,
            buffers,
            validated(signature, body, self.overflow_checks),
        )?;
        Ok(self.value(
            &call_reads(args, buffers),
            r,
            ValueType::Scalar(signature.result),
        ))
    }
    fn call_function(
        &mut self,
        function: &Arc<Function>,
        args: &[Reg],
        buffers: &[Slice],
    ) -> Result<Reg> {
        self.call_arguments(function.signature(), args, buffers)?;
        let r = self.inner.call_function(function, args, buffers)?;
        Ok(self.value(
            &call_reads(args, buffers),
            r,
            ValueType::Scalar(function.result()),
        ))
    }
}

/// The registers a call reads: its scalars and its slices.
fn call_reads(args: &[Reg], buffers: &[Slice]) -> Vec<Reg> {
    let slices = buffers.iter().flat_map(|s| [s.ptr, s.len]);
    args.iter().copied().chain(slices).collect()
}

thread_local! {
    /// The specializations whose bodies are being staged on this thread,
    /// outermost first.
    static STAGING: RefCell<Vec<Specialization>> = const { RefCell::new(Vec::new()) };
}

/// A specialization being staged. A call that needs the same one again
/// can only be staged by recursing forever: the IR has no recursive calls.
struct Staging;

impl Staging {
    fn enter(key: &Specialization) -> Result<Self> {
        STAGING.with_borrow_mut(|staging| {
            if staging.contains(key) {
                return Err(BuildError::staging(
                    "a kernel calls itself with the same specialization arguments; \
                     recursion must change a #[comptime] argument and end at specialization",
                ));
            }
            staging.push(key.clone());
            Ok(Self)
        })
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        STAGING.with_borrow_mut(Vec::pop);
    }
}

/// A builder that only numbers registers and labels, to verify a function.
pub(crate) struct Sink {
    arity: usize,
    buffers: usize,
    registers: u32,
    labels: u32,
    debug: bool,
}

impl Sink {
    pub(crate) fn new(signature: Signature<'_>, debug: bool) -> Self {
        Self {
            arity: signature.params.len(),
            buffers: signature.buffers.len(),
            registers: signature.registers() as u32,
            labels: 0,
            debug,
        }
    }
    fn pair(&mut self) -> Result<Slice> {
        Ok(Slice {
            ptr: self.next()?,
            len: self.next()?,
        })
    }
    fn next(&mut self) -> Result<Reg> {
        let r = Reg(self.registers);
        self.registers = self
            .registers
            .checked_add(1)
            .ok_or_else(|| BuildError::unsupported("too many registers"))?;
        Ok(r)
    }
}

impl KernelBuilder for Sink {
    fn debug_enabled(&self) -> bool {
        self.debug
    }
    fn buffer(&self, index: usize) -> Result<Slice> {
        if index < self.buffers {
            Ok(Slice::param(self.arity, index))
        } else {
            Err(BuildError::verify("invalid buffer index"))
        }
    }
    fn reg_type(&self, _: Reg) -> Option<ValueType> {
        None
    }
    fn local(&mut self, _: ValueType) -> Result<Reg> {
        self.next()
    }
    fn call_intrinsic(&mut self, inst: Intrinsic, _: &[Reg]) -> Result<Option<Reg>> {
        inst.output().map(|_| self.next()).transpose()
    }
    fn constant(&mut self, _: Constant) -> Result<Reg> {
        self.next()
    }
    fn cast(&mut self, _: Reg, _: ScalarType) -> Result<Reg> {
        self.next()
    }
    fn slice_range(&mut self, _: Slice, _: Reg, _: Reg) -> Result<Slice> {
        self.pair()
    }
    fn load_slice(&mut self, _: Slice, _: Reg) -> Result<Slice> {
        self.pair()
    }
    fn load(&mut self, _: Slice, _: Reg) -> Result<Reg> {
        self.next()
    }
    fn store(&mut self, _: Slice, _: Reg, _: Reg) -> Result<()> {
        Ok(())
    }
    fn binary(&mut self, _: Binary, _: Reg, _: Reg) -> Result<Reg> {
        self.next()
    }
    fn assign(&mut self, _: Reg, _: Reg) -> Result<()> {
        Ok(())
    }
    fn label(&mut self) -> Result<Label> {
        let l = Label(self.labels);
        self.labels += 1;
        Ok(l)
    }
    fn mark(&mut self, _: Label) -> Result<()> {
        Ok(())
    }
    fn jump(&mut self, _: Label) -> Result<()> {
        Ok(())
    }
    fn branch(&mut self, _: Reg, _: Label, _: Label) -> Result<()> {
        Ok(())
    }
    fn ret(&mut self, _: Reg) -> Result<()> {
        Ok(())
    }
    fn trap_bounds(&mut self) -> Result<()> {
        Ok(())
    }
    fn rust_call(
        &mut self,
        _: RustTarget,
        _: &[Reg],
        _: &[ContextSlice],
        _: ScalarType,
    ) -> Result<Reg> {
        self.next()
    }
    fn call_kernel(
        &mut self,
        signature: Signature<'_>,
        _: &[Reg],
        _: &[Slice],
        body: Body<'_>,
    ) -> Result<Reg> {
        body(&mut Self::new(signature, self.debug))?;
        self.next()
    }
    fn call_function(&mut self, _: &Arc<Function>, _: &[Reg], _: &[Slice]) -> Result<Reg> {
        self.next()
    }
}
