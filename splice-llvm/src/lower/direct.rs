//! Emit native LLVM IR during staging. Each nested kernel has its own Inkwell
//! insertion point and shares the enclosing kernel's module and target.
mod flow;

use super::{
    body::{self, Body},
    declare, emit_entry, permit_buffer_capture,
};
use flow::{Exit, Flow};
use inkwell::{builder::Builder, context::Context, module::Module, values::FunctionValue};
use splice_core::{
    BufferKind,
    frontend::{Body as StagedBody, KernelBuilder, KernelSource, Specialization},
    hir::{
        Binary, Buffer, ContextSlice, Label, Reg, RustContextTarget, Simd, ValueType, check_call,
    },
    simd::Element,
};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};

struct Graph<'a, 'ctx> {
    ctx: &'ctx Context,
    machine: crate::target::Machine,
    module: &'a Module<'ctx>,
    count: Cell<usize>,
    source: Cell<Option<&'static KernelSource>>,
    static_rust_slice_operands: Cell<bool>,
    callees: RefCell<HashMap<Specialization, (FunctionValue<'ctx>, bool)>>,
}

impl<'ctx> Graph<'_, 'ctx> {
    fn function(
        &self,
        arity: usize,
        kinds: &[BufferKind],
        key: Option<Specialization>,
        root: bool,
        stage: StagedBody<'_>,
    ) -> splice_core::Result<(FunctionValue<'ctx>, bool)> {
        if let Some(hit) = key
            .as_ref()
            .and_then(|k| self.callees.borrow().get(k).copied())
        {
            return Ok(hit);
        }
        let id = self.count.get();
        let function = declare(self.ctx, self.module, arity, kinds, id)?;
        self.count.set(id + 1);
        match self.body(function, arity, kinds, root, stage) {
            Ok(fallible) => {
                if let Some(key) = key {
                    self.callees.borrow_mut().insert(key, (function, fallible));
                }
                Ok((function, fallible))
            }
            Err(error) => {
                // SAFETY: body() dropped its builders and an incomplete function
                // never enters the cache or a caller. Its completed children
                // remain valid definitions that a later call can reuse.
                unsafe { function.delete() };
                Err(error)
            }
        }
    }

    fn body(
        &self,
        function: FunctionValue<'ctx>,
        arity: usize,
        kinds: &[BufferKind],
        root: bool,
        stage: StagedBody<'_>,
    ) -> splice_core::Result<bool> {
        let builder = self.ctx.create_builder();
        let unit = body::Unit {
            ctx: self.ctx,
            module: self.module,
            machine: self.machine,
        };
        let native = Body::start(
            unit,
            &builder,
            function,
            arity,
            std::iter::repeat_n(ValueType::Scalar, arity),
            kinds,
        )
        .map_err(crate::error)?;
        let mut direct = Direct {
            native,
            graph: self,
            root,
            allocas: self.ctx.create_builder(),
            arity,
            kinds,
            flow: Flow::new(),
            error: None,
        };
        stage(&mut direct)?;
        if let Some(err) = direct.error {
            return Err(err);
        }
        direct.flow.verify(arity, direct.native.register_count())?;
        if direct.flow.current.is_some() {
            // The verifier only permits fallthrough in a disconnected dead block.
            direct.native.unreachable().map_err(crate::error)?;
        }
        if direct.native.static_rust_slice_operands() {
            self.static_rust_slice_operands.set(true);
        }
        Ok(direct.native.needs_status())
    }
}

pub(crate) fn populate_direct<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    machine: crate::target::Machine,
    arity: usize,
    kinds: &[BufferKind],
    stage: StagedBody<'_>,
) -> splice_core::Result<Option<&'static KernelSource>> {
    let graph = Graph {
        ctx,
        machine,
        module,
        count: Cell::new(0),
        source: Cell::new(None),
        static_rust_slice_operands: Cell::new(false),
        callees: RefCell::default(),
    };
    let (root, _) = graph.function(arity, kinds, None, true, stage)?;
    emit_entry(ctx, module, &ctx.create_builder(), arity, kinds.len(), root)
        .map_err(crate::error)?;
    if graph.static_rust_slice_operands.get() {
        permit_buffer_capture(module);
    }
    Ok(graph.source.get())
}

struct Direct<'a, 'ctx> {
    graph: &'a Graph<'a, 'ctx>,
    root: bool,
    native: Body<'a, 'ctx>,
    allocas: Builder<'ctx>,
    arity: usize,
    kinds: &'a [BufferKind],
    flow: Flow,
    // Infallible frontend operations defer errors to the enclosing build.
    error: Option<splice_core::BuildError>,
}

impl<'a, 'ctx> Direct<'a, 'ctx> {
    fn emit(
        &mut self,
        reads: &[Reg],
        dst: Option<Reg>,
        generate: impl FnOnce(&mut Body<'a, 'ctx>) -> Result<(), body::Error>,
    ) {
        if reads.iter().copied().chain(dst).any(|r| {
            self.native
                .reg_type(r)
                .is_some_and(|ty| ty != ValueType::Scalar)
        }) {
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify("expected a scalar LLVM register")
            });
            return;
        }
        self.emit_typed(reads, dst, generate);
    }
    fn emit_typed(
        &mut self,
        reads: &[Reg],
        dst: Option<Reg>,
        generate: impl FnOnce(&mut Body<'a, 'ctx>) -> Result<(), body::Error>,
    ) {
        if self.error.is_some() {
            return;
        }
        if reads
            .iter()
            .copied()
            .chain(dst)
            .any(|r| r.0 as usize >= self.native.register_count())
        {
            self.error = Some(splice_core::BuildError::verify("invalid LLVM register"));
            return;
        }
        if self.flow.current.is_some() {
            self.flow.operands(reads, dst);
            if let Err(e) = generate(&mut self.native) {
                self.error = Some(crate::error(e));
            }
        }
    }
    fn result(
        &mut self,
        reads: &[Reg],
        generate: impl FnOnce(&mut Body<'a, 'ctx>, Reg) -> Result<(), body::Error>,
    ) -> Reg {
        let dst = self.local();
        self.emit(reads, Some(dst), |ir| generate(ir, dst));
        dst
    }
    fn buffer(&mut self, b: Buffer, write: bool) {
        match self.kinds.get(b.index()) {
            None => {
                self.error
                    .get_or_insert_with(|| splice_core::BuildError::verify("invalid LLVM buffer"));
            }
            Some(kind) if kind.context_signature().is_some() => {
                self.error.get_or_insert_with(|| {
                    splice_core::BuildError::verify(
                        "slice memory operation cannot use Rust context",
                    )
                });
            }
            Some(kind) if write && !kind.writable() => {
                self.error.get_or_insert_with(|| {
                    splice_core::BuildError::verify("store to read-only LLVM buffer")
                });
            }
            _ => {}
        }
    }
    fn scalar_buffer(&mut self, b: Buffer, write: bool) {
        self.buffer(b, write);
        if self
            .kinds
            .get(b.index())
            .is_some_and(|k| k.element() != Element::I32)
        {
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify("scalar access requires an i32 buffer")
            });
        }
    }
    fn target(&mut self, label: Label) {
        if let Err(err) = self.flow.reference(label) {
            self.error.get_or_insert(err);
        } else {
            self.native.prepare_label(label);
        }
    }
}

impl KernelBuilder for Direct<'_, '_> {
    fn set_source(&mut self, source: &'static KernelSource) {
        if self.root {
            self.graph.source.set(Some(source));
        }
    }
    fn reg_type(&self, r: Reg) -> Option<ValueType> {
        self.native.reg_type(r)
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
        if self.error.is_none() && self.native.register_count() < u32::MAX as usize {
            match self.native.new_local(&self.allocas, ty) {
                Ok(r) => return r,
                Err(e) => self.error = Some(crate::error(e)),
            }
        }
        self.error.get_or_insert_with(|| {
            splice_core::BuildError::unsupported("exhausted LLVM registers")
        });
        Reg(u32::MAX)
    }
    fn simd(&mut self, inst: Simd, args: &[Reg]) -> splice_core::Result<Option<Reg>> {
        let types = args
            .iter()
            .map(|&r| {
                self.native
                    .reg_type(r)
                    .ok_or_else(|| splice_core::BuildError::verify("invalid LLVM register"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        inst.validate(&types, self.kinds)?;
        let dst = inst.output().map(|ty| self.typed_local(ty));
        self.emit_typed(args, dst, |ir| ir.simd(dst, inst, args));
        Ok(dst)
    }
    fn constant(&mut self, n: i32) -> Reg {
        self.result(&[], |ir, d| ir.constant(d, n))
    }
    fn buffer_len(&mut self, b: Buffer) -> Reg {
        self.buffer(b, false);
        self.result(&[], |ir, d| ir.length(d, b))
    }
    fn load(&mut self, b: Buffer, index: Reg) -> Reg {
        self.scalar_buffer(b, false);
        self.result(&[index], |ir, d| ir.read_element(d, b, index))
    }
    fn store(&mut self, b: Buffer, index: Reg, value: Reg) {
        self.scalar_buffer(b, true);
        self.emit(&[index, value], None, |ir| {
            ir.write_element(b, index, value)
        });
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg {
        self.result(&[a, b], |ir, d| ir.arithmetic(d, op, a, b))
    }
    fn assign(&mut self, d: Reg, s: Reg) {
        if self.native.reg_type(d) != self.native.reg_type(s) {
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::verify("incompatible LLVM register assignment")
            });
            return;
        }
        self.emit_typed(&[s], Some(d), |ir| ir.assign(d, s));
    }
    fn label(&mut self) -> Label {
        if self.native.label_count() >= u32::MAX as usize {
            self.error.get_or_insert_with(|| {
                splice_core::BuildError::unsupported("exhausted LLVM labels")
            });
            return Label(u32::MAX);
        }
        self.flow.add_label();
        self.native.new_label()
    }
    fn mark(&mut self, l: Label) {
        self.target(l);
        if self.error.is_some() {
            return;
        }
        if self.flow.current.is_some() {
            self.jump(l);
        }
        if let Err(err) = self.flow.begin(l) {
            self.error = Some(err);
        } else {
            self.native.begin_block(l);
        }
    }
    fn jump(&mut self, l: Label) {
        self.target(l);
        self.emit(&[], None, |ir| ir.jump(l));
        self.flow.close(Exit::Jump(l));
    }
    fn branch(&mut self, cond: Reg, yes: Label, no: Label) {
        self.target(yes);
        self.target(no);
        self.emit(&[cond], None, |ir| ir.branch(cond, yes, no));
        self.flow.close(Exit::Branch(yes, no));
    }
    fn ret(&mut self, r: Reg) {
        self.emit(&[r], None, |ir| ir.ret(r));
        self.flow.close(Exit::Return);
    }
    fn trap_bounds(&mut self) -> splice_core::Result<()> {
        self.emit(&[], None, |ir| ir.trap_bounds());
        self.flow.close(Exit::Return);
        Ok(())
    }
    fn rust_call(
        &mut self,
        import: splice_core::hir::RustCall,
        args: &[Reg],
    ) -> splice_core::Result<Reg> {
        import.validate(args.iter().map(|&r| self.reg_type(r)))?;
        Ok(self.result(args, |ir, dst| ir.rust_call(dst, import, args)))
    }
    fn rust_packet_call(
        &mut self,
        import: &'static splice_core::hir::RustPacketSymbol,
        args: &[Reg],
    ) -> splice_core::Result<Reg> {
        splice_core::hir::check_rust_packet_call(
            import,
            args.iter().map(|&arg| self.reg_type(arg)),
        )?;
        Ok(self.result(args, |ir, dst| ir.rust_packet_call(dst, import, args)))
    }
    fn rust_context_call_with_slices(
        &mut self,
        context: Buffer,
        operation: u32,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        self.emit_rust_context_call(
            RustContextTarget::Borrowed { context, operation },
            args,
            slices,
        )
    }
    fn rust_static_call(
        &mut self,
        symbol: &'static splice_core::StaticRustSymbol,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        self.emit_rust_context_call(RustContextTarget::Static(symbol), args, slices)
    }
    fn call_kernel(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        stage: StagedBody<'_>,
    ) -> splice_core::Result<Reg> {
        self.staged_call(arity, kinds, args, buffers, None, stage)
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        stage: StagedBody<'_>,
    ) -> splice_core::Result<Reg> {
        let (arity, kinds) = key.signature();
        self.staged_call(arity, kinds, args, buffers, Some(key), stage)
    }
}

impl Direct<'_, '_> {
    fn emit_rust_context_call(
        &mut self,
        target: RustContextTarget,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        splice_core::hir::check_rust_context_call(
            self.kinds,
            target,
            args.iter().map(|&r| self.reg_type(r)),
            slices,
        )?;
        Ok(self.result(args, |ir, dst| {
            ir.rust_context_call(dst, target, args, slices)
        }))
    }

    fn staged_call(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        key: Option<Specialization>,
        stage: StagedBody<'_>,
    ) -> splice_core::Result<Reg> {
        check_call(
            arity,
            kinds,
            args.iter().map(|&r| self.reg_type(r)),
            self.kinds,
            buffers,
        )?;
        let (callee, fallible) = self.graph.function(arity, kinds, key, false, stage)?;
        let dst = self.result(args, |ir, d| {
            ir.call_function(d, callee, fallible, args, buffers)
        });
        if let Some(err) = self.error.take() {
            return Err(err);
        }
        Ok(dst)
    }
}
