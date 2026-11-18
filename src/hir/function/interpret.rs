use super::{Buffer, BufferArg, Function, Instr, validate_buffers};

fn out_of_bounds(buffer: Buffer, index: i32, len: usize) -> String {
    format!(
        "buffer {} index {index} out of bounds for length {len}",
        buffer.0
    )
}

impl Function {
    /// Reference execution using the same verified control-flow graph as the JIT.
    pub fn interpret(&self, args: &[i32]) -> Result<i32, String> {
        self.interpret_with_buffers(args, &mut [])
    }

    /// Execute with borrowed buffers. On a bounds error all earlier stores
    /// remain visible, and no subsequent instruction executes.
    pub fn interpret_with_buffers(
        &self,
        args: &[i32],
        buffers: &mut [BufferArg<'_>],
    ) -> Result<i32, String> {
        if args.len() != self.arity {
            return Err(format!(
                "expected {} arguments, got {}",
                self.arity,
                args.len()
            ));
        }
        validate_buffers(self.buffer_kinds(), buffers)?;
        let map: Vec<_> = (0..buffers.len()).map(|i| Buffer(i as u32)).collect();
        self.interpret_frame(args, buffers, &map)
    }

    // Buffer identifiers stay local to each function. Translate them to
    // root slice indices so forwarding never creates aliased mutable borrows.
    fn interpret_frame(
        &self,
        args: &[i32],
        buffers: &mut [BufferArg<'_>],
        map: &[Buffer],
    ) -> Result<i32, String> {
        let mut regs = vec![0; self.registers];
        regs[..args.len()].copy_from_slice(args);
        let mut pc = 0;
        loop {
            let instruction = &self.code[pc];
            pc += 1;
            match instruction {
                Instr::Const(d, n) => regs[d.0 as usize] = *n,
                Instr::Copy(d, s) => regs[d.0 as usize] = regs[s.0 as usize],
                Instr::Bin(d, op, a, b) => {
                    let rhs = regs[b.0 as usize];
                    let Some(value) = op.checked_eval(regs[a.0 as usize], rhs) else {
                        let error = if rhs == 0 {
                            "integer division by zero"
                        } else {
                            "signed division overflow"
                        };
                        return Err(error.into());
                    };
                    regs[d.0 as usize] = value;
                }
                Instr::BufferLen { dst, buffer } => {
                    regs[dst.0 as usize] = buffers[map[buffer.index()].index()].len() as i32;
                }
                Instr::Load { dst, buffer, index } => {
                    let index = regs[index.0 as usize];
                    let slice = buffers[map[buffer.index()].index()].as_slice();
                    regs[dst.0 as usize] = *slice
                        .get(index as usize)
                        .ok_or_else(|| out_of_bounds(*buffer, index, slice.len()))?;
                }
                Instr::Store {
                    buffer,
                    index,
                    value,
                } => {
                    let index = regs[index.0 as usize];
                    let BufferArg::Write(slice) = &mut buffers[map[buffer.index()].index()] else {
                        unreachable!("verified store into read-only buffer")
                    };
                    let len = slice.len();
                    *slice
                        .get_mut(index as usize)
                        .ok_or_else(|| out_of_bounds(*buffer, index, len))? =
                        regs[value.0 as usize];
                }
                Instr::Label(_) => {}
                Instr::Jump(l) => pc = self.targets[l.0 as usize],
                Instr::Branch { cond, yes, no } => {
                    let target = if regs[cond.0 as usize] != 0 { yes } else { no };
                    pc = self.targets[target.0 as usize];
                }
                Instr::Return(r) => return Ok(regs[r.0 as usize]),
                Instr::Call {
                    dst,
                    function,
                    args,
                    buffers: forwarded,
                } => {
                    let actual: Vec<_> = args.iter().map(|v| regs[v.0 as usize]).collect();
                    let child_map: Vec<_> = forwarded.iter().map(|b| map[b.index()]).collect();
                    regs[dst.0 as usize] =
                        function.interpret_frame(&actual, buffers, &child_map)?;
                }
            }
        }
    }
}
