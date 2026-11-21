use super::{Binary, Buffer, BufferKind, Function, Instr, Label, Reg, Simd, ValueType};
use crate::frontend::{Body, KernelBuilder, KernelSource, Specialization};
use std::collections::HashMap;
use std::sync::Arc;

/// Append typed scalar operations and control-flow edges, then verify the
/// function for all code generators at `finish`.
pub struct FunctionBuilder {
    function: Function,
    callees: HashMap<Specialization, Arc<Function>>,
}

impl FunctionBuilder {
    pub fn new(arity: usize) -> Self {
        Self::with_buffers(arity, Vec::new())
    }
    pub fn with_buffers(arity: usize, kinds: Vec<BufferKind>) -> Self {
        Self::with_options(arity, kinds, crate::BuildOptions::default())
    }
    pub fn with_options(
        arity: usize,
        kinds: Vec<BufferKind>,
        options: crate::BuildOptions,
    ) -> Self {
        Self {
            callees: HashMap::new(),
            function: Function {
                source: None,
                options,
                needs_status: false,
                arity,
                buffer_kinds: kinds,
                locals: Vec::new(),
                labels: 0,
                code: Vec::new(),
            },
        }
    }
    pub fn arg(&self, index: usize) -> Reg {
        Reg(if index < self.function.arity {
            u32::try_from(index).unwrap_or(u32::MAX)
        } else {
            u32::MAX
        })
    }
    pub fn local(&mut self) -> Reg {
        self.typed_local(ValueType::Scalar)
    }
    pub fn typed_local(&mut self, ty: ValueType) -> Reg {
        let index = self.function.register_count();
        if index >= u32::MAX as usize {
            return Reg(u32::MAX);
        }
        self.function.locals.push(ty);
        Reg(index as u32)
    }
    pub fn simd(&mut self, inst: Simd, args: &[Reg]) -> crate::Result<Option<Reg>> {
        let types = args
            .iter()
            .map(|&r| {
                self.function
                    .checked_reg_type(r)
                    .ok_or("invalid SIMD register")
            })
            .collect::<Result<Vec<_>, _>>()?;
        inst.validate(&types, &self.function.buffer_kinds)?;
        let dst = inst.output().map(|ty| self.typed_local(ty));
        self.function.code.push(Instr::Simd {
            inst,
            dst,
            args: args.to_vec(),
        });
        Ok(dst)
    }
    fn emit_value(&mut self, instruction: impl FnOnce(Reg) -> Instr) -> Reg {
        let dst = self.local();
        self.function.code.push(instruction(dst));
        dst
    }
    pub fn constant(&mut self, value: i32) -> Reg {
        self.emit_value(|dst| Instr::Const(dst, value))
    }

    pub fn buffer_len(&mut self, buffer: Buffer) -> Reg {
        self.emit_value(|dst| Instr::BufferLen { dst, buffer })
    }

    pub fn load(&mut self, buffer: Buffer, index: Reg) -> Reg {
        self.emit_value(|dst| Instr::Load { dst, buffer, index })
    }

    pub fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) {
        self.function.code.push(Instr::Store {
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
        match self
            .function
            .code
            .iter()
            .rev()
            .find(|op| !matches!(op, Instr::Debug(_)))?
        {
            Instr::Const(dst, value) if *dst == reg => Some(*value != 0),
            _ => None,
        }
    }
    pub fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg {
        self.emit_value(|dst| Instr::Bin(dst, op, a, b))
    }
    pub fn assign(&mut self, dst: Reg, src: Reg) {
        self.function.code.push(Instr::Copy(dst, src));
    }
    pub fn label(&mut self) -> Label {
        let label = Label(u32::try_from(self.function.labels).expect("exhausted labels"));
        self.function.labels += 1;
        label
    }
    pub fn mark(&mut self, label: Label) {
        self.function.code.push(Instr::Label(label));
    }
    pub fn jump(&mut self, label: Label) {
        self.function.code.push(Instr::Jump(label));
    }
    pub fn branch(&mut self, cond: Reg, yes: Label, no: Label) {
        self.function.code.push(Instr::Branch { cond, yes, no });
    }
    pub fn ret(&mut self, value: Reg) {
        self.function.code.push(Instr::Return(value));
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
        super::check_call(
            function.arity(),
            function.buffer_kinds(),
            args.iter().map(|&r| self.reg_type(r)),
            &self.function.buffer_kinds,
            buffers,
        )?;
        Ok(self.emit_call(Arc::new(function), args, buffers))
    }
    fn emit_call(&mut self, function: Arc<Function>, args: &[Reg], buffers: &[Buffer]) -> Reg {
        self.emit_value(|dst| Instr::Call {
            dst,
            function,
            args: args.to_vec(),
            buffers: buffers.to_vec(),
        })
    }
    fn staged_call(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        key: Option<Specialization>,
        body: Body<'_>,
    ) -> crate::Result<Reg> {
        super::check_call(
            arity,
            kinds,
            args.iter().map(|&r| self.reg_type(r)),
            &self.function.buffer_kinds,
            buffers,
        )?;
        if let Some(callee) = key.as_ref().and_then(|k| self.callees.get(k)).cloned() {
            return Ok(self.emit_call(callee, args, buffers));
        }
        let mut child = Self::with_options(arity, kinds.to_vec(), self.function.options);
        std::mem::swap(&mut self.callees, &mut child.callees);
        let staged = body(&mut child);
        std::mem::swap(&mut self.callees, &mut child.callees);
        staged?;
        let callee = Arc::new(child.finish()?);
        if let Some(key) = key {
            self.callees.insert(key, callee.clone());
        }
        Ok(self.emit_call(callee, args, buffers))
    }
    pub fn finish(self) -> Result<Function, String> {
        self.function.verify()
    }
}

// Keep the inherent API usable without importing the streaming builder trait.
macro_rules! forward_mut {
    ($($method:ident($($arg:ident: $ty:ty),*) $(-> $out:ty)?;)*) => {$(
        fn $method(&mut self, $($arg: $ty),*) $(-> $out)? {
            FunctionBuilder::$method(self, $($arg),*)
        }
    )*};
}

impl KernelBuilder for FunctionBuilder {
    fn debug_enabled(&self) -> bool {
        self.function.options == crate::BuildOptions::Debug
    }
    fn debug_point(
        &mut self,
        location: crate::debug::Location,
        variables: &[crate::debug::Variable],
    ) {
        if self.debug_enabled() {
            self.function.code.push(Instr::Debug(crate::debug::Point {
                location,
                variables: variables.into(),
            }));
        }
    }
    fn reg_type(&self, r: Reg) -> Option<ValueType> {
        self.function.checked_reg_type(r)
    }
    fn set_source(&mut self, source: &'static KernelSource) {
        self.function.source = Some(source);
    }
    fn arg(&self, index: usize) -> Reg {
        self.arg(index)
    }
    forward_mut! {
        local() -> Reg;
        typed_local(ty:ValueType) -> Reg;
        simd(inst:Simd,args:&[Reg])->crate::Result<Option<Reg>>;
        constant(value: i32) -> Reg;
        buffer_len(buffer: Buffer) -> Reg;
        load(buffer: Buffer, index: Reg) -> Reg;
        store(buffer: Buffer, index: Reg, value: Reg);
        binary(op: Binary, a: Reg, b: Reg) -> Reg;
        assign(dst: Reg, src: Reg);
        label() -> Label;
        mark(label: Label);
        jump(label: Label);
        branch(cond: Reg, yes: Label, no: Label);
        ret(value: Reg);
    }
    fn immediate_condition(&self, reg: Reg) -> Option<bool> {
        self.immediate_condition(reg)
    }
    fn call_kernel(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> Result<Reg, String> {
        self.staged_call(arity, kinds, args, buffers, None, body)
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> crate::Result<Reg> {
        let (arity, kinds) = key.signature();
        self.staged_call(arity, kinds, args, buffers, Some(key), body)
    }
}
