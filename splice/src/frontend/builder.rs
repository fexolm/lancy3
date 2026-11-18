use crate::hir::{Binary, Buffer, BufferKind, Function, Instr, Label, Reg, buffer, verify};
use std::sync::Arc;

/// Append typed scalar operations and control-flow edges, then verify the
/// function for all code generators at `finish`.
pub struct FunctionBuilder {
    source: Option<&'static super::KernelSource>,
    arity: usize,
    buffer_kinds: Vec<BufferKind>,
    registers: usize,
    labels: usize,
    code: Vec<Instr>,
}

impl FunctionBuilder {
    pub fn new(arity: usize) -> Self {
        Self::with_buffers(arity, Vec::new())
    }
    pub fn with_buffers(arity: usize, kinds: Vec<BufferKind>) -> Self {
        Self {
            source: None,
            arity,
            buffer_kinds: kinds,
            registers: arity,
            labels: 0,
            code: Vec::new(),
        }
    }
    pub fn arg(&self, index: usize) -> Reg {
        Reg(if index < self.arity {
            u32::try_from(index).unwrap_or(u32::MAX)
        } else {
            u32::MAX
        })
    }
    pub fn local(&mut self) -> Reg {
        let register = Reg(u32::try_from(self.registers).expect("exhausted virtual registers"));
        self.registers += 1;
        register
    }
    pub fn constant(&mut self, value: i32) -> Reg {
        let dst = self.local();
        self.code.push(Instr::Const(dst, value));
        dst
    }

    pub fn buffer_len(&mut self, buffer: Buffer) -> Reg {
        let dst = self.local();
        self.code.push(Instr::BufferLen { dst, buffer });
        dst
    }

    pub fn load(&mut self, buffer: Buffer, index: Reg) -> Reg {
        let dst = self.local();
        self.code.push(Instr::Load { dst, buffer, index });
        dst
    }

    pub fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) {
        self.code.push(Instr::Store {
            buffer,
            index,
            value,
        });
    }

    /// A mixed-stage condition may just have materialized its known arm.
    /// Looking at the preceding instruction is valid even for mutable registers:
    /// no label can transfer control between that definition and this use.
    #[doc(hidden)]
    pub fn immediate_condition(&self, reg: Reg) -> Option<bool> {
        match self.code.last()? {
            Instr::Const(dst, value) if *dst == reg => Some(*value != 0),
            _ => None,
        }
    }
    pub fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg {
        let dst = self.local();
        self.code.push(Instr::Bin(dst, op, a, b));
        dst
    }
    pub fn assign(&mut self, dst: Reg, src: Reg) {
        self.code.push(Instr::Copy(dst, src));
    }
    pub fn label(&mut self) -> Label {
        let label = Label(u32::try_from(self.labels).expect("exhausted labels"));
        self.labels += 1;
        label
    }
    pub fn mark(&mut self, label: Label) {
        self.code.push(Instr::Label(label));
    }
    pub fn jump(&mut self, label: Label) {
        self.code.push(Instr::Jump(label));
    }
    pub fn branch(&mut self, cond: Reg, yes: Label, no: Label) {
        self.code.push(Instr::Branch { cond, yes, no });
    }
    pub fn ret(&mut self, value: Reg) {
        self.code.push(Instr::Return(value));
    }
    pub fn call(&mut self, callee: Function, args: &[Reg]) -> Reg {
        self.call_shared(Arc::new(callee), args)
    }
    pub fn call_shared(&mut self, function: Arc<Function>, args: &[Reg]) -> Reg {
        self.emit_call(function, args, &[])
    }
    pub fn call_with_buffers(
        &mut self,
        function: Function,
        args: &[Reg],
        buffers: &[Buffer],
    ) -> Result<Reg, String> {
        if function.arity() != args.len() {
            return Err(format!(
                "expected {} scalar arguments, got {}",
                function.arity(),
                args.len()
            ));
        }
        buffer::check_forwarding(&self.buffer_kinds, function.buffer_kinds(), buffers)?;
        Ok(self.emit_call(Arc::new(function), args, buffers))
    }
    fn emit_call(&mut self, function: Arc<Function>, args: &[Reg], buffers: &[Buffer]) -> Reg {
        let dst = self.local();
        self.code.push(Instr::Call {
            dst,
            function,
            args: args.to_vec(),
            buffers: buffers.to_vec(),
        });
        dst
    }
    pub fn finish(self) -> Result<Function, String> {
        verify::finish(
            self.source,
            self.arity,
            self.buffer_kinds,
            self.registers,
            self.labels,
            self.code,
        )
    }
}

impl super::KernelBuilder for FunctionBuilder {
    fn set_source(&mut self, source: &'static super::KernelSource) {
        self.source = Some(source);
    }
    fn arg(&self, index: usize) -> Reg {
        self.arg(index)
    }
    fn local(&mut self) -> Reg {
        self.local()
    }
    fn constant(&mut self, value: i32) -> Reg {
        self.constant(value)
    }
    fn buffer_len(&mut self, buffer: Buffer) -> Reg {
        self.buffer_len(buffer)
    }
    fn load(&mut self, buffer: Buffer, index: Reg) -> Reg {
        self.load(buffer, index)
    }
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) {
        self.store(buffer, index, value)
    }
    fn immediate_condition(&self, reg: Reg) -> Option<bool> {
        self.immediate_condition(reg)
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg {
        self.binary(op, a, b)
    }
    fn assign(&mut self, dst: Reg, src: Reg) {
        self.assign(dst, src)
    }
    fn label(&mut self) -> Label {
        self.label()
    }
    fn mark(&mut self, l: Label) {
        self.mark(l)
    }
    fn jump(&mut self, l: Label) {
        self.jump(l)
    }
    fn branch(&mut self, cond: Reg, yes: Label, no: Label) {
        self.branch(cond, yes, no)
    }
    fn ret(&mut self, v: Reg) {
        self.ret(v)
    }
    fn call_kernel(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        body: super::Body<'_>,
    ) -> Result<Reg, String> {
        self.call_with_buffers(super::materialize(arity, kinds, body)?, args, buffers)
    }
}
