use super::{
    Binary, Buffer, Constant, ContextSlice, Function, Instr, Label, Reg, ScalarType, ValueType,
};
use crate::frontend::{Body, KernelBuilder, KernelSource, Signature, Specialization};
use crate::hir::Intrinsic;
use std::collections::HashMap;
use std::sync::Arc;

/// Record a kernel body as a portable function. [`finish`](Self::finish)
/// verifies what was recorded; staging through
/// [`materialize`](crate::frontend::materialize) checks each operation as
/// it arrives instead.
pub struct FunctionBuilder {
    function: Function,
    callees: HashMap<Specialization, Arc<Function>>,
}

impl FunctionBuilder {
    pub fn new(options: crate::BuildOptions, signature: Signature<'_>) -> Self {
        Self {
            callees: HashMap::new(),
            function: Function {
                source: None,
                options,
                needs_status: false,
                params: signature.params.to_vec(),
                result: signature.result,
                buffer_kinds: signature.buffers.to_vec(),
                locals: Vec::new(),
                labels: 0,
                code: Vec::new(),
            },
        }
    }

    fn push(&mut self, instruction: Instr) -> crate::Result<()> {
        self.function.code.push(instruction);
        Ok(())
    }

    fn emit_value(
        &mut self,
        ty: ValueType,
        instruction: impl FnOnce(Reg) -> Instr,
    ) -> crate::Result<Reg> {
        let dst = self.local(ty)?;
        self.push(instruction(dst))?;
        Ok(dst)
    }

    fn scalar(
        &mut self,
        ty: ScalarType,
        instruction: impl FnOnce(Reg) -> Instr,
    ) -> crate::Result<Reg> {
        self.emit_value(ValueType::Scalar(ty), instruction)
    }

    /// The scalar type of `r`. An operation on an invalid operand records
    /// `I32`; verification rejects it.
    fn scalar_type(&self, r: Reg) -> ScalarType {
        match self.function.checked_reg_type(r) {
            Some(ValueType::Scalar(ty)) => ty,
            _ => ScalarType::I32,
        }
    }

    fn staged_call(
        &mut self,
        signature: Signature<'_>,
        args: &[Reg],
        buffers: &[Buffer],
        key: Option<Specialization>,
        body: Body<'_>,
    ) -> crate::Result<Reg> {
        if let Some(callee) = key.as_ref().and_then(|k| self.callees.get(k)).cloned() {
            return self.call_function(&callee, args, buffers);
        }
        let mut child = Self::new(self.function.options, signature);
        std::mem::swap(&mut self.callees, &mut child.callees);
        let staged = body(&mut child);
        std::mem::swap(&mut self.callees, &mut child.callees);
        staged?;
        // The body checked its own operations.
        let callee = Arc::new(child.finish_validated());
        if let Some(key) = key {
            self.callees.insert(key, callee.clone());
        }
        self.call_function(&callee, args, buffers)
    }

    /// Verify the recorded function.
    pub fn finish(self) -> crate::Result<Function> {
        self.function.verify()
    }

    /// The recorded operations passed [`crate::validate::Validating`].
    pub(crate) fn finish_validated(mut self) -> Function {
        self.function.needs_status = self.function.code.iter().any(Instr::may_trap);
        self.function
    }
}

impl KernelBuilder for FunctionBuilder {
    fn set_source(&mut self, source: &'static KernelSource) {
        self.function.source = Some(source);
    }
    fn debug_enabled(&self) -> bool {
        self.function.options.debug()
    }
    fn debug_point(
        &mut self,
        location: crate::debug::Location,
        variables: &[crate::debug::Variable],
    ) -> crate::Result<()> {
        self.push(Instr::Debug(crate::debug::Point {
            location,
            variables: variables.into(),
        }))
    }
    fn reg_type(&self, r: Reg) -> Option<ValueType> {
        self.function.checked_reg_type(r)
    }
    fn local(&mut self, ty: ValueType) -> crate::Result<Reg> {
        let index = u32::try_from(self.function.register_count())
            .ok()
            .filter(|&index| index != u32::MAX)
            .ok_or_else(|| crate::BuildError::unsupported("too many registers"))?;
        self.function.locals.push(ty);
        Ok(Reg(index))
    }
    fn call_intrinsic(&mut self, inst: Intrinsic, args: &[Reg]) -> crate::Result<Option<Reg>> {
        let dst = inst.output().map(|ty| self.local(ty)).transpose()?;
        self.push(Instr::CallIntrinsic {
            intrinsic: inst,
            dst,
            args: args.to_vec(),
        })?;
        Ok(dst)
    }
    fn constant(&mut self, value: Constant) -> crate::Result<Reg> {
        self.scalar(value.ty(), |dst| Instr::Const(dst, value))
    }
    fn cast(&mut self, src: Reg, to: ScalarType) -> crate::Result<Reg> {
        self.scalar(to, |dst| Instr::Cast { dst, src })
    }
    fn buffer_len(&mut self, buffer: Buffer) -> crate::Result<Reg> {
        self.scalar(ScalarType::Index, |dst| Instr::BufferLen { dst, buffer })
    }
    fn load(&mut self, buffer: Buffer, index: Reg) -> crate::Result<Reg> {
        let ty = self
            .function
            .buffer_kinds
            .get(buffer.index())
            .map_or(ScalarType::I32, |kind| kind.element());
        self.scalar(ty, |dst| Instr::Load { dst, buffer, index })
    }
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) -> crate::Result<()> {
        self.push(Instr::Store {
            buffer,
            index,
            value,
        })
    }
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> crate::Result<Reg> {
        let ty = self.scalar_type(a);
        let ty = op.output(ty).unwrap_or(ty);
        self.scalar(ty, |dst| Instr::Bin(dst, op, a, b))
    }
    fn assign(&mut self, dst: Reg, src: Reg) -> crate::Result<()> {
        self.push(Instr::Copy(dst, src))
    }
    fn label(&mut self) -> crate::Result<Label> {
        let label = u32::try_from(self.function.labels)
            .map(Label)
            .map_err(|_| crate::BuildError::unsupported("too many labels"))?;
        self.function.labels += 1;
        Ok(label)
    }
    fn mark(&mut self, label: Label) -> crate::Result<()> {
        self.push(Instr::Label(label))
    }
    fn jump(&mut self, label: Label) -> crate::Result<()> {
        self.push(Instr::Jump(label))
    }
    fn branch(&mut self, cond: Reg, yes: Label, no: Label) -> crate::Result<()> {
        self.push(Instr::Branch { cond, yes, no })
    }
    fn ret(&mut self, value: Reg) -> crate::Result<()> {
        self.push(Instr::Return(value))
    }
    fn trap_bounds(&mut self) -> crate::Result<()> {
        self.push(Instr::TrapBounds)
    }
    fn rust_call(
        &mut self,
        target: super::RustTarget,
        args: &[Reg],
        slices: &[ContextSlice],
        result: ScalarType,
    ) -> crate::Result<Reg> {
        self.scalar(result, |dst| Instr::RustCall {
            dst,
            target,
            args: args.to_vec(),
            slices: slices.to_vec(),
            result,
        })
    }
    fn call_kernel(
        &mut self,
        signature: Signature<'_>,
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> crate::Result<Reg> {
        self.staged_call(signature, args, buffers, None, body)
    }
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> crate::Result<Reg> {
        self.staged_call(key.signature(), args, buffers, Some(key), body)
    }
    fn call_function(
        &mut self,
        function: &Arc<Function>,
        args: &[Reg],
        buffers: &[Buffer],
    ) -> crate::Result<Reg> {
        self.scalar(function.result(), |dst| Instr::Call {
            dst,
            function: Arc::clone(function),
            args: args.to_vec(),
            buffers: buffers.to_vec(),
        })
    }
}

impl Function {
    pub fn signature(&self) -> Signature<'_> {
        Signature::new(&self.params, self.result, &self.buffer_kinds)
    }
}
