//! Emit native LLVM IR during staging. Each nested kernel has its own Inkwell
//! insertion point and shares the enclosing kernel's module and target.
mod flow;

use super::{body::Body, declare, emit_entry};
use flow::{Exit, Flow};
use inkwell::{
    builder::{Builder, BuilderError},
    context::Context,
    module::Module,
    values::FunctionValue,
};
use splice::{
    BufferKind,
    frontend::{Body as StagedBody, KernelBuilder, Specialization},
    hir::{Binary, Buffer, Label, Reg, check_forwarding},
};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};

struct Graph<'a, 'ctx> {
    ctx: &'ctx Context,
    module: &'a Module<'ctx>,
    count: Cell<usize>,
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
        let native = Body::start(self.ctx, &builder, function, arity, arity, kinds.len())
            .map_err(|e| e.to_string())?;
        let mut direct = Direct {
            native,
            graph: self,
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
        Ok(direct.native.needs_status())
    }
}

pub(crate) fn populate_direct<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    arity: usize,
    kinds: &[BufferKind],
    stage: StagedBody<'_>,
) -> Result<(), String> {
    let graph = Graph {
        ctx,
        module,
        count: Cell::new(0),
        callees: RefCell::default(),
    };
    let (root, _) = graph.function(arity, kinds, None, stage)?;
    emit_entry(ctx, module, &ctx.create_builder(), arity, kinds.len(), root)
        .map_err(|e| e.to_string())
}

struct Direct<'a, 'ctx> {
    graph: &'a Graph<'a, 'ctx>,
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
            Some(BufferKind::Read) if write => {
                self.error
                    .get_or_insert_with(|| "store to read-only LLVM buffer".into());
            }
            _ => {}
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
    fn arg(&self, index: usize) -> Reg {
        Reg(if index < self.arity {
            index as u32
        } else {
            u32::MAX
        })
    }
    fn local(&mut self) -> Reg {
        if self.error.is_none() && self.native.register_count() < u32::MAX as usize {
            match self.native.new_local(&self.allocas) {
                Ok(r) => return r,
                Err(e) => self.error = Some(e.to_string()),
            }
        }
        self.error
            .get_or_insert_with(|| "exhausted LLVM registers".into());
        Reg(u32::MAX)
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
        self.buffer(b, false);
        self.result(&[index], |ir, d| ir.read_element(d, b, index))
    }
    fn store(&mut self, b: Buffer, index: Reg, value: Reg) {
        self.buffer(b, true);
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
        self.emit(&[s], Some(d), |ir| ir.assign(d, s));
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
    fn staged_call(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        key: Option<Specialization>,
        stage: StagedBody<'_>,
    ) -> splice::Result<Reg> {
        if arity != args.len() {
            return Err("wrong scalar arity at LLVM call".into());
        }
        if args
            .iter()
            .any(|r| r.0 as usize >= self.native.register_count())
        {
            return Err("invalid argument register in LLVM call".into());
        }
        check_forwarding(self.kinds, kinds, buffers)?;
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
