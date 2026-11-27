//! The one verifier of Splice kernels. Every operation a backend's builder
//! receives was checked here first, on the direct build path and when a
//! portable function is verified (which replays it through this check).
use crate::frontend::{Body, KernelBuilder, KernelSource, Signature, Specialization};
use crate::hir::{
    Binary, Constant, ContextSlice, Function, Label, Reg, RustTarget, ScalarType, Simd, Slice,
    ValueType,
};
use crate::intrinsic::IntrinsicId;
use crate::{BuildError, Result};
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
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
    arity: usize,
    result: ScalarType,
    signature: Signature<'a>,
    types: Vec<ValueType>,
    /// The block each label starts, once marked.
    labels: Vec<Option<usize>>,
    blocks: Vec<Block>,
    current: Option<usize>,
    pc: usize,
    overflow_checks: Option<bool>,
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
        if signature.buffers.len() > u32::MAX as usize {
            return Err(BuildError::verify("too many buffer parameters"));
        }
        Ok(Self {
            inner,
            arity: signature.params.len(),
            result: signature.result,
            signature,
            types: (0..signature.registers())
                .map(|r| signature.param_type(r).expect("a parameter register"))
                .collect(),
            labels: Vec::new(),
            blocks: vec![Block::new()],
            current: Some(0),
            pc: 0,
            overflow_checks: None,
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

    /// The type of the elements of `slice`, accessed one at a time.
    fn element(&self, slice: Slice, write: bool) -> Result<ScalarType> {
        let ValueType::Ptr(kind) = self.ty(slice.ptr)? else {
            return Err(self.error("expected a slice pointer register"));
        };
        if kind.context_signature().is_some() {
            return Err(self.error("slice memory operation cannot use Rust context"));
        }
        self.expect(slice.len, ScalarType::Index)?;
        if write && !kind.writable() {
            return Err(self.error("store to read-only buffer"));
        }
        Ok(kind.element())
    }

    fn call_arguments(&self, callee: Signature<'_>, args: &[Reg], slices: &[Slice]) -> Result<()> {
        self.at(crate::hir::check_call(callee, args, slices, |r| {
            self.reg_type(r)
        }))
    }

    /// The scalar arguments and slice registers a call reads.
    fn call_reads(args: &[Reg], slices: &[Slice]) -> Vec<Reg> {
        let slices = slices.iter().flat_map(|s| [s.ptr, s.len]);
        args.iter().copied().chain(slices).collect()
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
        for r in 0..self.signature.registers() {
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
                crate::debug::Value::Register(r) => {
                    self.ty(r)?;
                }
                crate::debug::Value::Slice(slice) => {
                    self.element(slice, false)
                        .map_err(|_| self.error("invalid debug slice"))?;
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
        if index < self.arity {
            self.inner.arg(index)
        } else {
            Err(BuildError::verify("invalid parameter index"))
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
        let r = self.inner.local(ty)?;
        Ok(self.allocated(r, ty))
    }
    fn simd(&mut self, inst: Simd, args: &[Reg]) -> Result<Option<Reg>> {
        let types = self.types(args)?;
        self.at(inst.validate(&types))?;
        let dst = self.inner.simd(inst, args)?;
        match (dst, inst.output()) {
            (Some(dst), Some(ty)) => {
                self.value(args, dst, ty);
            }
            (None, None) => self.record(args, None),
            _ => return Err(self.error("invalid SIMD result type")),
        }
        Ok(dst)
    }
    fn constant(&mut self, value: Constant) -> Result<Reg> {
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
    fn slice_arg(&self, index: usize) -> Result<Slice> {
        if index < self.signature.buffers.len() {
            self.inner.slice_arg(index)
        } else {
            Err(BuildError::verify("invalid buffer parameter index"))
        }
    }
    fn offset(&mut self, ptr: Reg, offset: Reg) -> Result<Reg> {
        let ty @ ValueType::Ptr(kind) = self.ty(ptr)? else {
            return Err(self.error("expected a slice pointer register"));
        };
        if kind.context_signature().is_some() {
            return Err(self.error("a Rust context has no elements"));
        }
        self.expect(offset, ScalarType::Index)?;
        let r = self.inner.offset(ptr, offset)?;
        Ok(self.value(&[ptr, offset], r, ty))
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
        self.at(target.check(args, slices, |r| self.reg_type(r)))?;
        let r = self.inner.rust_call(target, args, slices, result)?;
        let mut reads = Self::call_reads(args, &[]);
        reads.extend(slices.iter().flat_map(|s| [s.slice.ptr, s.slice.len]));
        if let RustTarget::Context { context, .. } = target {
            reads.push(context);
        }
        Ok(self.value(&reads, r, ValueType::Scalar(result)))
    }
    fn intrinsic(&mut self, id: &IntrinsicId, args: &[Reg]) -> Result<Reg> {
        let types = self.types(args)?;
        if types != id.params {
            return Err(self.error(format!(
                "intrinsic {id} takes operands of its parameter types, got {types:?}"
            )));
        }
        let r = self.inner.intrinsic(id, args)?;
        Ok(self.value(args, r, id.result))
    }
    fn call_kernel(
        &mut self,
        signature: Signature<'_>,
        args: &[Reg],
        slices: &[Slice],
        body: Body<'_>,
    ) -> Result<Reg> {
        self.call_arguments(signature, args, slices)?;
        let r = self.inner.call_kernel(
            signature,
            args,
            slices,
            validated(signature, body, self.overflow_checks),
        )?;
        let reads = Self::call_reads(args, slices);
        Ok(self.value(&reads, r, ValueType::Scalar(signature.result)))
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        slices: &[Slice],
        body: Body<'_>,
    ) -> Result<Reg> {
        let signature = key.signature();
        self.call_arguments(signature, args, slices)?;
        let _staging = Staging::enter(&key)?;
        let r = self.inner.call_specialized(
            key,
            args,
            slices,
            validated(signature, body, self.overflow_checks),
        )?;
        let reads = Self::call_reads(args, slices);
        Ok(self.value(&reads, r, ValueType::Scalar(signature.result)))
    }
    fn call_function(
        &mut self,
        function: &Arc<Function>,
        args: &[Reg],
        slices: &[Slice],
    ) -> Result<Reg> {
        self.call_arguments(function.signature(), args, slices)?;
        let r = self.inner.call_function(function, args, slices)?;
        let reads = Self::call_reads(args, slices);
        Ok(self.value(&reads, r, ValueType::Scalar(function.result())))
    }
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
    registers: u32,
    /// The scalar parameters.
    signature: usize,
    labels: u32,
    debug: bool,
}

impl Sink {
    pub(crate) fn new(signature: Signature<'_>, debug: bool) -> Self {
        Self {
            registers: signature.registers() as u32,
            signature: signature.params.len(),
            labels: 0,
            debug,
        }
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
    fn reg_type(&self, _: Reg) -> Option<ValueType> {
        None
    }
    fn local(&mut self, _: ValueType) -> Result<Reg> {
        self.next()
    }
    fn simd(&mut self, inst: Simd, _: &[Reg]) -> Result<Option<Reg>> {
        inst.output().map(|_| self.next()).transpose()
    }
    fn constant(&mut self, _: Constant) -> Result<Reg> {
        self.next()
    }
    fn cast(&mut self, _: Reg, _: ScalarType) -> Result<Reg> {
        self.next()
    }
    fn slice_arg(&self, index: usize) -> Result<Slice> {
        let ptr = (self.signature + 2 * index) as u32;
        Ok(Slice {
            ptr: Reg(ptr),
            len: Reg(ptr + 1),
        })
    }
    fn offset(&mut self, _: Reg, _: Reg) -> Result<Reg> {
        self.next()
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
    fn intrinsic(&mut self, _: &IntrinsicId, _: &[Reg]) -> Result<Reg> {
        self.next()
    }
}
