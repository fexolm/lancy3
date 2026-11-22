//! Emit native LLVM IR during staging. Each nested kernel has its own Inkwell
//! insertion point and shares the enclosing kernel's module and target.
mod flow;

use super::{body::Body, declare, emit_entry, permit_buffer_capture};
use flow::{Exit, Flow};
use inkwell::{
    builder::{Builder, BuilderError},
    context::Context,
    module::Module,
    values::FunctionValue,
};
use splice::{
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
        stage: StagedBody<'_>,
    ) -> Result<(FunctionValue<'ctx>, bool), String> {
        splice::hir::check_context_layout(kinds)?;
        if let Some(hit) = key
            .as_ref()
            .and_then(|k| self.callees.borrow().get(k).copied())
        {
            return Ok(hit);
        }
        let id = self.count.get();
        let function = declare(self.ctx, self.module, arity, kinds, id)?;
        self.count.set(id + 1);
        match self.body(function, arity, kinds, stage) {
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
        stage: StagedBody<'_>,
    ) -> Result<bool, String> {
        let builder = self.ctx.create_builder();
        let native = Body::start(
            self.ctx,
            &builder,
            self.module,
            function,
            arity,
            std::iter::repeat_n(ValueType::Scalar, arity),
            kinds,
        )
        .map_err(|e| e.to_string())?;
        let mut direct = Direct {
            native,
            graph: self,
            root: function.get_name().to_bytes() == b"kernel0",
            allocas: self.ctx.create_builder(),
            arity,
            kinds,
            flow: Flow::new(),
            literal: None,
            error: None,
        };
        stage(&mut direct)?;
        if let Some(err) = direct.error {
            return Err(err);
        }
        direct.flow.verify(arity, direct.native.register_count())?;
        if direct.flow.current.is_some() {
            // The verifier only permits fallthrough in a disconnected dead block.
            direct.native.unreachable().map_err(|e| e.to_string())?;
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
    arity: usize,
    kinds: &[BufferKind],
    stage: StagedBody<'_>,
) -> Result<Option<&'static KernelSource>, String> {
    let graph = Graph {
        ctx,
        module,
        count: Cell::new(0),
        source: Cell::new(None),
        static_rust_slice_operands: Cell::new(false),
        callees: RefCell::default(),
    };
    let (root, _) = graph.function(arity, kinds, None, stage)?;
    emit_entry(ctx, module, &ctx.create_builder(), arity, kinds.len(), root)
        .map_err(|e| e.to_string())?;
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
    literal: Option<(Reg, bool)>,
    // Infallible frontend operations defer errors to the enclosing build.
    error: Option<String>,
}

impl<'a, 'ctx> Direct<'a, 'ctx> {
    fn emit(
        &mut self,
        reads: &[Reg],
        dst: Option<Reg>,
        generate: impl FnOnce(&mut Body<'a, 'ctx>) -> Result<(), BuilderError>,
    ) {
        if reads.iter().copied().chain(dst).any(|r| {
            self.native
                .reg_type(r)
                .is_some_and(|ty| ty != ValueType::Scalar)
        }) {
            self.error
                .get_or_insert_with(|| "expected a scalar LLVM register".into());
            return;
        }
        self.emit_typed(reads, dst, generate);
    }
    fn emit_typed(
        &mut self,
        reads: &[Reg],
        dst: Option<Reg>,
        generate: impl FnOnce(&mut Body<'a, 'ctx>) -> Result<(), BuilderError>,
    ) {
        self.literal = None;
        if self.error.is_some() {
            return;
        }
        if reads
            .iter()
            .copied()
            .chain(dst)
            .any(|r| r.0 as usize >= self.native.register_count())
        {
            self.error = Some("invalid LLVM register".into());
            return;
        }
        if self.flow.current.is_some() {
            self.flow.operands(reads, dst);
            if let Err(e) = generate(&mut self.native) {
                self.error = Some(e.to_string());
            }
        }
    }
    fn result(
        &mut self,
        reads: &[Reg],
        generate: impl FnOnce(&mut Body<'a, 'ctx>, Reg) -> Result<(), BuilderError>,
    ) -> Reg {
        let dst = self.local();
        self.emit(reads, Some(dst), |ir| generate(ir, dst));
        dst
    }
    fn buffer(&mut self, b: Buffer, write: bool) {
        match self.kinds.get(b.index()) {
            None => {
                self.error
                    .get_or_insert_with(|| "invalid LLVM buffer".into());
            }
            Some(kind) if kind.context_signature().is_some() => {
                self.error
                    .get_or_insert_with(|| "slice memory operation cannot use Rust context".into());
            }
            Some(kind) if write && !kind.writable() => {
                self.error
                    .get_or_insert_with(|| "store to read-only LLVM buffer".into());
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
            self.error
                .get_or_insert_with(|| "scalar access requires an i32 buffer".into());
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
                Err(e) => self.error = Some(e.to_string()),
            }
        }
        self.error
            .get_or_insert_with(|| "exhausted LLVM registers".into());
        Reg(u32::MAX)
    }
    fn simd(&mut self, inst: Simd, args: &[Reg]) -> splice::Result<Option<Reg>> {
        let types = args
            .iter()
            .map(|&r| self.native.reg_type(r).ok_or("invalid LLVM register"))
            .collect::<Result<Vec<_>, _>>()?;
        inst.validate(&types, self.kinds)?;
        let dst = inst.output().map(|ty| self.typed_local(ty));
        self.emit_typed(args, dst, |ir| ir.simd(dst, inst, args));
        Ok(dst)
    }
    fn constant(&mut self, n: i32) -> Reg {
        let r = self.result(&[], |ir, d| ir.constant(d, n));
        self.literal = Some((r, n != 0));
        r
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
    fn immediate_condition(&self, r: Reg) -> Option<bool> {
        self.literal.filter(|&(v, _)| v == r).map(|(_, n)| n)
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg {
        self.result(&[a, b], |ir, d| ir.arithmetic(d, op, a, b))
    }
    fn assign(&mut self, d: Reg, s: Reg) {
        if self.native.reg_type(d) != self.native.reg_type(s) {
            self.error
                .get_or_insert_with(|| "incompatible LLVM register assignment".into());
            return;
        }
        self.emit_typed(&[s], Some(d), |ir| ir.assign(d, s));
    }
    fn label(&mut self) -> Label {
        if self.native.label_count() >= u32::MAX as usize {
            self.error
                .get_or_insert_with(|| "exhausted LLVM labels".into());
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
        self.literal = None;
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
    fn trap_bounds(&mut self) -> splice::Result<()> {
        self.emit(&[], None, |ir| ir.trap_bounds());
        self.flow.close(Exit::Return);
        Ok(())
    }
    fn rust_call(&mut self, import: splice::hir::RustCall, args: &[Reg]) -> splice::Result<Reg> {
        import.validate(args.iter().map(|&r| self.reg_type(r)))?;
        Ok(self.result(args, |ir, dst| ir.rust_call(dst, import, args)))
    }
    fn rust_packet_call(
        &mut self,
        import: &'static splice::hir::RustPacketSymbol,
        args: &[Reg],
    ) -> splice::Result<Reg> {
        splice::hir::check_rust_packet_call(import, args.iter().map(|&arg| self.reg_type(arg)))?;
        Ok(self.result(args, |ir, dst| ir.rust_packet_call(dst, import, args)))
    }
    fn rust_context_call_with_slices(
        &mut self,
        context: Buffer,
        operation: u32,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice::Result<Reg> {
        self.emit_rust_context_call(
            RustContextTarget::Borrowed { context, operation },
            args,
            slices,
        )
    }
    fn rust_static_call(
        &mut self,
        symbol: &'static splice::StaticRustSymbol,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice::Result<Reg> {
        self.emit_rust_context_call(RustContextTarget::Static(symbol), args, slices)
    }
    fn call_kernel(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        stage: StagedBody<'_>,
    ) -> Result<Reg, String> {
        self.staged_call(arity, kinds, args, buffers, None, stage)
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        stage: StagedBody<'_>,
    ) -> splice::Result<Reg> {
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
    ) -> splice::Result<Reg> {
        splice::hir::check_rust_context_call(
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
    ) -> splice::Result<Reg> {
        check_call(
            arity,
            kinds,
            args.iter().map(|&r| self.reg_type(r)),
            self.kinds,
            buffers,
        )?;
        let (callee, fallible) = self.graph.function(arity, kinds, key, stage)?;
        let dst = self.result(args, |ir, d| {
            ir.call_function(d, callee, fallible, args, buffers)
        });
        if let Some(err) = &self.error {
            return Err(err.clone());
        }
        Ok(dst)
    }
}
