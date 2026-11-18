//! Reference execution of SSA block arguments, checked operations and calls.

use super::{Block, END, Edge, Function, Inst, Op, Terminator, Value};
use crate::hir::{Buffer, BufferArg, validate_buffers};

struct Frame<'a> {
    function: &'a Function,
    block: Block,
    next: Inst,
    values: Vec<i32>,
    /// Local buffer number -> root borrow. Nested calls never reborrow slices.
    buffers: Vec<Buffer>,
    return_to: Option<Value>,
    transfer: Vec<i32>,
}

impl<'a> Frame<'a> {
    fn new(
        function: &'a Function,
        args: impl IntoIterator<Item = i32>,
        buffers: Vec<Buffer>,
        return_to: Option<Value>,
    ) -> Self {
        let block = function.entry();
        let mut values = vec![0; function.value_count()];
        for (&param, arg) in function.params(block).iter().zip(args) {
            values[param.index()] = arg;
        }
        Self {
            function,
            block,
            next: function.block(block).first,
            values,
            buffers,
            return_to,
            transfer: Vec::new(),
        }
    }

    fn read(&self, value: Value) -> i32 {
        self.values[value.index()]
    }

    fn jump(&mut self, edge: Edge) {
        let (target, args) = self.function.edge(edge);
        // An edge can swap values on a backedge; read all sources first.
        self.transfer.clear();
        self.transfer
            .extend(args.iter().map(|a| self.values[a.index()]));
        for (&param, &actual) in self.function.params(target).iter().zip(&self.transfer) {
            self.values[param.index()] = actual;
        }
        self.block = target;
        self.next = self.function.block(target).first;
    }
}

fn bounds_error(buffer: Buffer, index: i32, len: usize) -> String {
    format!(
        "buffer {} index {index} out of bounds for length {len}",
        buffer.0
    )
}

impl Function {
    pub fn interpret(&self, args: &[i32]) -> Result<i32, String> {
        self.interpret_with_buffers(args, &mut [])
    }

    /// Check the entire program before executing. Stores are visible to the
    /// caller even if a later operation or nested call traps.
    pub fn interpret_with_buffers(
        &self,
        args: &[i32],
        buffers: &mut [BufferArg<'_>],
    ) -> Result<i32, String> {
        if args.len() != self.arity() {
            return Err(format!(
                "expected {} arguments, got {}",
                self.arity(),
                args.len()
            ));
        }
        validate_buffers(self.buffer_kinds(), buffers)?;
        self.verify_graph()?;
        let root_buffers = (0..buffers.len()).map(|i| Buffer(i as u32)).collect();
        let mut frames = vec![Frame::new(self, args.iter().copied(), root_buffers, None)];
        loop {
            let f = frames.last_mut().unwrap();
            let graph = f.function;
            let inst = f.next;
            if inst == END {
                match *graph.terminator(f.block) {
                    Terminator::Return(result) => {
                        let result = f.read(result);
                        let ret = f.return_to;
                        frames.pop();
                        if let Some(ret) = ret {
                            frames.last_mut().unwrap().values[ret.index()] = result;
                        } else {
                            return Ok(result);
                        }
                    }
                    Terminator::Jump(edge) => f.jump(edge),
                    Terminator::Branch { cond, yes, no } => {
                        f.jump(if f.read(cond) != 0 { yes } else { no });
                    }
                }
                continue;
            }
            f.next = graph.insts[inst.index()].next;
            let result = match *graph.op(inst) {
                Op::Const(n) => n,
                Op::Copy(src) => f.read(src),
                Op::Bin(op, a, b) => {
                    let rhs = f.read(b);
                    op.checked_eval(f.read(a), rhs).ok_or(if rhs == 0 {
                        "integer division by zero"
                    } else {
                        "signed division overflow"
                    })?
                }
                Op::BufferLen(b) => buffers[f.buffers[b.index()].index()].len() as i32,
                Op::Load(b, index) => {
                    let index = f.read(index);
                    let slice = buffers[f.buffers[b.index()].index()].as_slice();
                    *slice
                        .get(index as usize)
                        .ok_or_else(|| bounds_error(b, index, slice.len()))?
                }
                Op::Store(b, index, value) => {
                    let index = f.read(index);
                    let BufferArg::Write(slice) = &mut buffers[f.buffers[b.index()].index()] else {
                        unreachable!("verified writable slice")
                    };
                    let len = slice.len();
                    *slice
                        .get_mut(index as usize)
                        .ok_or_else(|| bounds_error(b, index, len))? = f.read(value);
                    continue;
                }
                Op::Call(call) => {
                    let (callee, args, forwarded) = graph.call(call);
                    let buffers = forwarded.iter().map(|b| f.buffers[b.index()]).collect();
                    let child = Frame::new(
                        callee,
                        args.iter().map(|a| f.read(*a)),
                        buffers,
                        graph.result(inst),
                    );
                    frames.push(child);
                    continue;
                }
            };
            f.values[graph.result(inst).unwrap().index()] = result;
        }
    }
}
