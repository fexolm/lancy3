//! Emit native LLVM IR while a kernel is staged. Each nested kernel has its
//! own Inkwell insertion point and shares the enclosing kernel's module and
//! target. The portable path replays a function into the same builder.
use super::{
    body::{self, Body},
    declare, emit_entry, permit_buffer_capture,
};
use inkwell::{
    attributes::{Attribute, AttributeLoc},
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    values::FunctionValue,
};
use splice_core::{
    Function,
    debug::{Location, Point, Slot, Variable},
    frontend::{Body as StagedBody, KernelBuilder, KernelSource, Signature, Specialization},
    hir::{Binary, Buffer, ContextSlice, Label, Reg, RustContextTarget, Simd, ValueType},
};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    sync::Arc,
};

/// What staging recorded about the module, for its symbols and debug info.
#[derive(Default)]
pub(crate) struct Staged {
    /// The root kernel's declaration.
    pub source: Option<&'static KernelSource>,
    /// The source points of a debug build, by function number (`kernel{id}`).
    pub functions: Vec<DebugFunction>,
}

#[derive(Default)]
pub(crate) struct DebugFunction {
    pub source: Option<&'static KernelSource>,
    pub points: Vec<(Point, Vec<Slot>)>,
}

/// A lowered callee: its LLVM function and whether a call can fail.
type Callee<'ctx> = (FunctionValue<'ctx>, bool);

struct Graph<'a, 'ctx> {
    ctx: &'ctx Context,
    machine: crate::target::Machine,
    module: &'a Module<'ctx>,
    debug: bool,
    count: Cell<usize>,
    staged: RefCell<Staged>,
    static_rust_slice_operands: Cell<bool>,
    callees: RefCell<HashMap<Specialization, Callee<'ctx>>>,
    /// Shared portable callees by identity. Each entry keeps its function
    /// alive, so a key cannot be reused by another allocation.
    functions: RefCell<HashMap<*const Function, (Arc<Function>, Callee<'ctx>)>>,
}

impl<'ctx> Graph<'_, 'ctx> {
    fn function(
        &self,
        signature: Signature<'_>,
        root: bool,
        stage: impl FnOnce(&mut dyn KernelBuilder) -> splice_core::Result<()>,
    ) -> splice_core::Result<(FunctionValue<'ctx>, bool)> {
        let id = self.count.get();
        let function = declare(
            self.ctx,
            self.module,
            signature.arity,
            signature.buffers,
            id,
        )?;
        if self.debug {
            function.set_linkage(Linkage::External);
            self.staged
                .borrow_mut()
                .functions
                .push(DebugFunction::default());
        }
        self.count.set(id + 1);
        match self.body(function, id, signature, root, stage) {
            Ok(fallible) => Ok((function, fallible)),
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
        id: usize,
        signature: Signature<'_>,
        root: bool,
        stage: impl FnOnce(&mut dyn KernelBuilder) -> splice_core::Result<()>,
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
            signature.arity,
            std::iter::repeat_n(ValueType::Scalar, signature.arity),
            signature.buffers,
        )
        .map_err(crate::error)?;
        let mut direct = Direct {
            native,
            graph: self,
            id,
            root,
            allocas: self.ctx.create_builder(),
            kinds: signature.buffers,
            open: true,
        };
        stage(&mut direct)?;
        if direct.open {
            // Validation only permits fallthrough in a disconnected dead block.
            direct.native.unreachable().map_err(crate::error)?;
        }
        if direct.native.static_rust_slice_operands() {
            self.static_rust_slice_operands.set(true);
        }
        Ok(direct.native.needs_status())
    }
}

/// Stage `stage` into `module` as the kernel and its callees, with the
/// public entry point.
pub(crate) fn populate<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    machine: crate::target::Machine,
    signature: Signature<'_>,
    stage: impl FnOnce(&mut dyn KernelBuilder) -> splice_core::Result<()>,
    debug: bool,
) -> splice_core::Result<Staged> {
    let graph = Graph {
        ctx,
        machine,
        module,
        debug,
        count: Cell::new(0),
        staged: RefCell::default(),
        static_rust_slice_operands: Cell::new(false),
        callees: RefCell::default(),
        functions: RefCell::default(),
    };
    let (root, _) = graph.function(signature, true, stage)?;
    emit_entry(
        ctx,
        module,
        &ctx.create_builder(),
        signature.arity,
        signature.buffers.len(),
        root,
    )
    .map_err(crate::error)?;
    if graph.static_rust_slice_operands.get() {
        permit_buffer_capture(module);
    }
    if debug {
        let preserve_frame = ctx.create_string_attribute("frame-pointer", "all");
        for f in module.get_functions() {
            f.add_attribute(AttributeLoc::Function, preserve_frame);
            #[cfg(target_os = "linux")]
            f.add_attribute(
                AttributeLoc::Function,
                ctx.create_enum_attribute(Attribute::get_named_enum_kind_id("uwtable"), 2),
            );
        }
    }
    Ok(graph.staged.into_inner())
}

/// Stage a verified portable function.
pub(crate) fn populate_function<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    machine: crate::target::Machine,
    function: &Function,
    debug: bool,
) -> splice_core::Result<Staged> {
    populate(
        ctx,
        module,
        machine,
        function.signature(),
        |b| splice_core::hir::replay(function, b),
        debug,
    )
}

struct Direct<'a, 'ctx> {
    graph: &'a Graph<'a, 'ctx>,
    /// This function's number.
    id: usize,
    root: bool,
    native: Body<'a, 'ctx>,
    allocas: Builder<'ctx>,
    kinds: &'a [splice_core::BufferKind],
    /// Whether the insertion point is in a block that has no terminator yet.
    /// Operations after a terminator are unreachable and emit nothing.
    open: bool,
}

impl<'a, 'ctx> Direct<'a, 'ctx> {
    fn emit(
        &mut self,
        generate: impl FnOnce(&mut Body<'a, 'ctx>) -> Result<(), body::Error>,
    ) -> splice_core::Result<()> {
        if self.open {
            generate(&mut self.native).map_err(crate::error)?;
        }
        Ok(())
    }
    fn result(
        &mut self,
        generate: impl FnOnce(&mut Body<'a, 'ctx>, Reg) -> Result<(), body::Error>,
    ) -> splice_core::Result<Reg> {
        let dst = self.local(ValueType::Scalar)?;
        self.emit(|ir| generate(ir, dst))?;
        Ok(dst)
    }
    fn close(
        &mut self,
        generate: impl FnOnce(&mut Body<'a, 'ctx>) -> Result<(), body::Error>,
    ) -> splice_core::Result<()> {
        self.emit(generate)?;
        self.open = false;
        Ok(())
    }
    fn call(
        &mut self,
        callee: FunctionValue<'ctx>,
        fallible: bool,
        args: &[Reg],
        buffers: &[Buffer],
    ) -> splice_core::Result<Reg> {
        self.result(|ir, d| ir.call_function(d, callee, fallible, args, buffers))
    }
}

impl KernelBuilder for Direct<'_, '_> {
    fn set_source(&mut self, source: &'static KernelSource) {
        let mut staged = self.graph.staged.borrow_mut();
        if self.root {
            staged.source = Some(source);
        }
        if let Some(function) = staged.functions.get_mut(self.id) {
            function.source = Some(source);
        }
    }
    fn debug_enabled(&self) -> bool {
        self.graph.debug
    }
    fn debug_point(
        &mut self,
        location: Location,
        variables: &[Variable],
    ) -> splice_core::Result<()> {
        if !self.graph.debug || !self.open {
            return Ok(());
        }
        let point = Point {
            location,
            variables: variables.into(),
        };
        let slots: Vec<_> = point
            .slots(
                |r| self.native.reg_type(r).expect("validated register"),
                self.kinds,
                self.graph.machine.vector_bytes,
            )
            .collect();
        let mut staged = self.graph.staged.borrow_mut();
        let points = &mut staged.functions[self.id].points;
        let ordinal = points.len();
        if u32::try_from(self.id).is_err() || u32::try_from(ordinal).is_err() {
            return Err(splice_core::BuildError::unsupported(
                "too many LLVM source points",
            ));
        }
        self.native
            .debug_point(&self.allocas, &slots, self.id, ordinal)
            .map_err(crate::error)?;
        points.push((point, slots));
        Ok(())
    }
    fn reg_type(&self, r: Reg) -> Option<ValueType> {
        self.native.reg_type(r)
    }
    fn local(&mut self, ty: ValueType) -> splice_core::Result<Reg> {
        if self.native.register_count() >= u32::MAX as usize {
            return Err(splice_core::BuildError::unsupported(
                "exhausted LLVM registers",
            ));
        }
        self.native
            .new_local(&self.allocas, ty)
            .map_err(crate::error)
    }
    fn simd(&mut self, inst: Simd, args: &[Reg]) -> splice_core::Result<Option<Reg>> {
        let dst = inst.output().map(|ty| self.local(ty)).transpose()?;
        self.emit(|ir| ir.simd(dst, inst, args))?;
        Ok(dst)
    }
    fn constant(&mut self, n: i32) -> splice_core::Result<Reg> {
        self.result(|ir, d| ir.constant(d, n))
    }
    fn buffer_len(&mut self, b: Buffer) -> splice_core::Result<Reg> {
        self.result(|ir, d| ir.length(d, b))
    }
    fn load(&mut self, b: Buffer, index: Reg) -> splice_core::Result<Reg> {
        self.result(|ir, d| ir.read_element(d, b, index))
    }
    fn store(&mut self, b: Buffer, index: Reg, value: Reg) -> splice_core::Result<()> {
        self.emit(|ir| ir.write_element(b, index, value))
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> splice_core::Result<Reg> {
        self.result(|ir, d| ir.arithmetic(d, op, a, b))
    }
    fn assign(&mut self, d: Reg, s: Reg) -> splice_core::Result<()> {
        self.emit(|ir| ir.assign(d, s))
    }
    fn label(&mut self) -> splice_core::Result<Label> {
        if self.native.label_count() >= u32::MAX as usize {
            return Err(splice_core::BuildError::unsupported(
                "exhausted LLVM labels",
            ));
        }
        Ok(self.native.new_label())
    }
    fn mark(&mut self, l: Label) -> splice_core::Result<()> {
        self.native.prepare_label(l);
        if self.open {
            self.native.jump(l).map_err(crate::error)?;
        }
        self.native.begin_block(l);
        self.open = true;
        Ok(())
    }
    fn jump(&mut self, l: Label) -> splice_core::Result<()> {
        self.native.prepare_label(l);
        self.close(|ir| ir.jump(l))
    }
    fn branch(&mut self, cond: Reg, yes: Label, no: Label) -> splice_core::Result<()> {
        self.native.prepare_label(yes);
        self.native.prepare_label(no);
        self.close(|ir| ir.branch(cond, yes, no))
    }
    fn ret(&mut self, r: Reg) -> splice_core::Result<()> {
        self.close(|ir| ir.ret(r))
    }
    fn trap_bounds(&mut self) -> splice_core::Result<()> {
        self.close(|ir| ir.trap_bounds())
    }
    fn rust_call(
        &mut self,
        import: splice_core::hir::RustCall,
        args: &[Reg],
    ) -> splice_core::Result<Reg> {
        self.result(|ir, dst| ir.rust_call(dst, import, args))
    }
    fn rust_packet_call(
        &mut self,
        import: &'static splice_core::hir::RustPacketSymbol,
        args: &[Reg],
    ) -> splice_core::Result<Reg> {
        self.result(|ir, dst| ir.rust_packet_call(dst, import, args))
    }
    fn rust_context_call(
        &mut self,
        context: Buffer,
        operation: u32,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        let target = RustContextTarget::Borrowed { context, operation };
        self.result(|ir, dst| ir.rust_context_call(dst, target, args, slices))
    }
    fn rust_static_call(
        &mut self,
        symbol: &'static splice_core::StaticRustSymbol,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> splice_core::Result<Reg> {
        let target = RustContextTarget::Static(symbol);
        self.result(|ir, dst| ir.rust_context_call(dst, target, args, slices))
    }
    fn call_kernel(
        &mut self,
        signature: Signature<'_>,
        args: &[Reg],
        buffers: &[Buffer],
        stage: StagedBody<'_>,
    ) -> splice_core::Result<Reg> {
        let (callee, fallible) = self.graph.function(signature, false, stage)?;
        self.call(callee, fallible, args, buffers)
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        stage: StagedBody<'_>,
    ) -> splice_core::Result<Reg> {
        let hit = self.graph.callees.borrow().get(&key).copied();
        let (callee, fallible) = match hit {
            Some(hit) => hit,
            None => {
                let signature = key.signature();
                let staged = self.graph.function(signature, false, stage)?;
                self.graph.callees.borrow_mut().insert(key, staged);
                staged
            }
        };
        self.call(callee, fallible, args, buffers)
    }
    fn call_function(
        &mut self,
        function: &Arc<Function>,
        args: &[Reg],
        buffers: &[Buffer],
    ) -> splice_core::Result<Reg> {
        let key = Arc::as_ptr(function);
        let hit = self
            .graph
            .functions
            .borrow()
            .get(&key)
            .map(|&(_, callee)| callee);
        let (callee, fallible) = match hit {
            Some(hit) => hit,
            None => {
                let (callee, fallible) = self.graph.function(function.signature(), false, |b| {
                    splice_core::hir::replay(function, b)
                })?;
                self.graph
                    .functions
                    .borrow_mut()
                    .insert(key, (Arc::clone(function), (callee, fallible)));
                (callee, fallible)
            }
        };
        self.call(callee, fallible, args, buffers)
    }
}
