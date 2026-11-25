//! Replay a portable function into any kernel builder. A backend's portable
//! path is this replay over the builder that stages kernels directly.
use super::{Function, Instr, Label, Reg};
use crate::frontend::KernelBuilder;

/// Issue the operations of `function` to `builder`, in order.
///
/// A function recorded by [`super::FunctionBuilder`] replays as the same
/// stream of operations, with the same register and label numbers, so a
/// builder observes the same kernel through either build path.
pub fn replay(function: &Function, builder: &mut dyn KernelBuilder) -> crate::Result<()> {
    Replay::new(function, builder)?.run()
}

struct Replay<'f, 'b> {
    f: &'f Function,
    b: &'b mut dyn KernelBuilder,
    /// The builder's register for each register of the function.
    regs: Vec<Option<Reg>>,
    /// Locals that a single value-producing instruction writes. The builder
    /// allocates such a register when it emits the instruction.
    results: Vec<bool>,
    /// The next local of the function the builder has not yet allocated.
    next: usize,
    labels: Vec<Label>,
    /// The instruction being replayed.
    pc: usize,
}

impl<'f, 'b> Replay<'f, 'b> {
    fn new(f: &'f Function, b: &'b mut dyn KernelBuilder) -> crate::Result<Self> {
        let count = f.register_count();
        let mut writes = vec![0u32; count];
        let mut copied = vec![false; count];
        for instruction in f.instructions() {
            if let Some(r) = instruction.written()
                && let Some(n) = writes.get_mut(r.0 as usize)
            {
                *n += 1;
                copied[r.0 as usize] |= matches!(instruction, Instr::Copy(..));
            }
        }
        let results = (0..count)
            .map(|r| r >= f.arity() && writes[r] == 1 && !copied[r])
            .collect();
        let mut regs = vec![None; count];
        for (r, slot) in regs.iter_mut().enumerate().take(f.arity()) {
            *slot = Some(b.arg(r)?);
        }
        if let Some(source) = f.source() {
            b.set_source(source);
        }
        let labels = (0..f.label_count())
            .map(|_| b.label())
            .collect::<crate::Result<_>>()?;
        Ok(Self {
            f,
            b,
            regs,
            results,
            next: f.arity(),
            labels,
            pc: 0,
        })
    }

    /// Allocate the function's locals below `end` that the builder lacks.
    fn allocate_below(&mut self, end: usize) -> crate::Result<()> {
        while self.next < end.min(self.regs.len()) {
            let r = self.next;
            if self.regs[r].is_none() {
                self.regs[r] = Some(self.b.local(self.f.reg_type(Reg(r as u32)))?);
                self.results[r] = false;
            }
            self.next += 1;
        }
        Ok(())
    }

    /// A register out of range stays invalid for the builder to reject.
    fn reg(&mut self, r: Reg) -> crate::Result<Reg> {
        let index = r.0 as usize;
        if index >= self.regs.len() {
            return Ok(Reg(u32::MAX));
        }
        self.allocate_below(index + 1)?;
        Ok(self.regs[index].expect("allocated register"))
    }

    fn regs(&mut self, rs: &[Reg]) -> crate::Result<Vec<Reg>> {
        rs.iter().map(|&r| self.reg(r)).collect()
    }

    fn label(&self, l: Label) -> Label {
        self.labels
            .get(l.0 as usize)
            .copied()
            .unwrap_or(Label(u32::MAX))
    }

    /// Bind the result of a value-producing operation to `dst`.
    fn define(
        &mut self,
        dst: Reg,
        emit: impl FnOnce(&mut dyn KernelBuilder) -> crate::Result<Reg>,
    ) -> crate::Result<()> {
        let index = dst.0 as usize;
        if index >= self.regs.len() {
            return Err(crate::BuildError::verify_at(
                self.pc,
                "invalid result register",
            ));
        }
        if self.results[index] {
            self.allocate_below(index)?;
            if self.next == index {
                let value = emit(&mut *self.b)?;
                if self.b.reg_type(value) != Some(self.f.reg_type(dst)) {
                    return Err(crate::BuildError::verify_at(
                        self.pc,
                        "result register type mismatch",
                    ));
                }
                self.regs[index] = Some(value);
                self.next += 1;
                return Ok(());
            }
        }
        let value = emit(&mut *self.b)?;
        let dst = self.reg(dst)?;
        self.b.assign(dst, value)
    }

    fn run(mut self) -> crate::Result<()> {
        use crate::debug::{Value, Variable};
        for (pc, instruction) in self.f.instructions().iter().enumerate() {
            self.pc = pc;
            match instruction {
                Instr::Debug(point) => {
                    let variables = point
                        .variables
                        .iter()
                        .map(|&var| {
                            Ok(match var.value {
                                Value::Register(r) => Variable {
                                    value: Value::Register(self.reg(r)?),
                                    ..var
                                },
                                _ => var,
                            })
                        })
                        .collect::<crate::Result<Vec<_>>>()?;
                    self.b.debug_point(point.location, &variables)?;
                }
                Instr::Simd { dst, inst, args } => {
                    let args = self.regs(args)?;
                    match dst {
                        Some(dst) => self.define(*dst, |b| {
                            b.simd(*inst, &args)?.ok_or_else(|| {
                                crate::BuildError::verify_at(pc, "invalid SIMD result type")
                            })
                        })?,
                        None => {
                            if self.b.simd(*inst, &args)?.is_some() {
                                return Err(crate::BuildError::verify_at(
                                    pc,
                                    "invalid SIMD result type",
                                ));
                            }
                        }
                    }
                }
                &Instr::Const(dst, value) => self.define(dst, |b| b.constant(value))?,
                &Instr::Cast { dst, src } => {
                    let src = self.reg(src)?;
                    let to = match self.f.checked_reg_type(dst) {
                        Some(super::ValueType::Scalar(ty)) => ty,
                        // `define` rejects an invalid register.
                        None => super::ScalarType::I32,
                        Some(_) => {
                            return Err(crate::BuildError::verify_at(
                                pc,
                                "a cast produces a scalar register",
                            ));
                        }
                    };
                    self.define(dst, |b| b.cast(src, to))?;
                }
                &Instr::Copy(dst, src) => {
                    let (dst, src) = (self.reg(dst)?, self.reg(src)?);
                    self.b.assign(dst, src)?;
                }
                &Instr::Bin(dst, op, a, z) => {
                    let (a, z) = (self.reg(a)?, self.reg(z)?);
                    self.define(dst, |b| b.binary(op, a, z))?;
                }
                &Instr::BufferLen { dst, buffer } => {
                    self.define(dst, |b| b.buffer_len(buffer))?;
                }
                &Instr::Load { dst, buffer, index } => {
                    let index = self.reg(index)?;
                    self.define(dst, |b| b.load(buffer, index))?;
                }
                &Instr::Store {
                    buffer,
                    index,
                    value,
                } => {
                    let (index, value) = (self.reg(index)?, self.reg(value)?);
                    self.b.store(buffer, index, value)?;
                }
                &Instr::Label(l) => self.b.mark(self.label(l))?,
                &Instr::Jump(l) => self.b.jump(self.label(l))?,
                &Instr::Branch { cond, yes, no } => {
                    let cond = self.reg(cond)?;
                    self.b.branch(cond, self.label(yes), self.label(no))?;
                }
                &Instr::Return(value) => {
                    let value = self.reg(value)?;
                    self.b.ret(value)?;
                }
                Instr::TrapBounds => self.b.trap_bounds()?,
                Instr::RustCall { dst, import, args } => {
                    let args = self.regs(args)?;
                    self.define(*dst, |b| b.rust_call(*import, &args))?;
                }
                Instr::RustPacketCall { dst, import, args } => {
                    let args = self.regs(args)?;
                    self.define(*dst, |b| b.rust_packet_call(import, &args))?;
                }
                Instr::RustContextCall {
                    dst,
                    target,
                    args,
                    slices,
                } => {
                    let args = self.regs(args)?;
                    self.define(*dst, |b| match *target {
                        super::RustContextTarget::Borrowed { context, operation } => {
                            b.rust_context_call(context, operation, &args, slices)
                        }
                        super::RustContextTarget::Static(symbol) => {
                            b.rust_static_call(symbol, &args, slices)
                        }
                    })?;
                }
                Instr::Call {
                    dst,
                    function,
                    args,
                    buffers,
                } => {
                    let args = self.regs(args)?;
                    self.define(*dst, |b| b.call_function(function, &args, buffers))?;
                }
            }
        }
        self.allocate_below(self.regs.len())
    }
}
