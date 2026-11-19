use crate::hir::{Block, Edge, Function, Inst, Op, Terminator, Type, Value};
use crate::support::slotmap::DenseMap;

struct Frame<'a> {
    function: &'a Function,
    block: Block,
    next: Inst,
    values: DenseMap<Value, u64>,
    return_to: Option<Value>,
    transfer: Vec<u64>,
}

impl<'a> Frame<'a> {
    fn new(
        function: &'a Function,
        args: impl IntoIterator<Item = u64>,
        return_to: Option<Value>,
    ) -> Self {
        let block = function.entry();
        let mut values = DenseMap::filled(function.value_count(), 0);
        for (&param, arg) in function.params(block).iter().zip(args) {
            values[param] = arg;
        }
        Self {
            function,
            block,
            next: function.blocks[block].first,
            values,
            return_to,
            transfer: Vec::new(),
        }
    }

    fn read(&self, value: Value) -> u64 {
        self.values[value]
    }

    fn jump(&mut self, edge: Edge) {
        let (target, args) = self.function.edge(edge);
        // An edge can swap values on a backedge; read all sources first.
        self.transfer.clear();
        self.transfer.extend(args.iter().map(|a| self.values[*a]));
        for (&param, &actual) in self.function.params(target).iter().zip(&self.transfer) {
            self.values[param] = actual;
        }
        self.block = target;
        self.next = self.function.blocks[target].first;
    }
}

fn buffer_word(bytes: &mut [u8], addr: u64, ty: Type) -> Result<&mut [u8; 4], String> {
    if ty != Type::I32 {
        return Err("filetest buffer holds i32 data only".into());
    }
    bytes
        .get_mut(addr as usize..)
        .and_then(|tail| tail.first_chunk_mut())
        .ok_or_else(|| format!("filetest access outside buffer at offset {addr}"))
}

pub(super) unsafe fn run(
    function: &Function,
    args: &[u64],
    mut buffer: Option<&mut [u8]>,
) -> Result<u64, String> {
    let mut frames = vec![Frame::new(function, args.iter().copied(), None)];
    loop {
        let f = frames.last_mut().unwrap();
        let graph = f.function;
        let inst = f.next;
        if inst == Inst::INVALID {
            match *graph.terminator(f.block) {
                Terminator::Return(result) => {
                    let result = f.read(result);
                    let ret = f.return_to;
                    frames.pop();
                    if let Some(ret) = ret {
                        frames.last_mut().unwrap().values[ret] = result;
                    } else {
                        return Ok(result);
                    }
                }
                Terminator::Jump(edge) => f.jump(edge),
                Terminator::Branch { cond, yes, no } => {
                    f.jump(if f.read(cond) as i32 != 0 { yes } else { no });
                }
            }
            continue;
        }
        f.next = graph.insts[inst].next;
        let result = match *graph.op(inst) {
            Op::Const(n) => n as u32 as u64,
            Op::Copy(src) => f.read(src),
            Op::Bin(op, a, b) => {
                let rhs = f.read(b) as i32;
                op.checked_eval(f.read(a) as i32, rhs).ok_or(if rhs == 0 {
                    "integer division by zero"
                } else {
                    "signed division overflow"
                })? as u32 as u64
            }
            Op::BoundsCheck(index, len) => {
                if f.read(index) as u32 >= f.read(len) as u32 {
                    return Err("kernel slice index out of bounds".into());
                }
                continue;
            }
            Op::PtrOffset(base, index, scale) => f
                .read(base)
                .wrapping_add((f.read(index) as i32 as i64 as u64) << scale as u8),
            Op::Load(ty, addr) => {
                let addr = f.read(addr);
                if let Some(bytes) = buffer.as_deref_mut() {
                    u32::from_ne_bytes(*buffer_word(bytes, addr, ty)?) as u64
                } else {
                    // SAFETY: covered by interpret_raw's caller contract.
                    unsafe {
                        match ty {
                            Type::I32 => (addr as *const u32).read_unaligned() as u64,
                            Type::Ptr => (addr as *const u64).read_unaligned(),
                        }
                    }
                }
            }
            Op::Store(addr, value) => {
                let addr = f.read(addr);
                if let Some(bytes) = buffer.as_deref_mut() {
                    *buffer_word(bytes, addr, graph.value_type(value))? =
                        (f.read(value) as u32).to_ne_bytes();
                } else {
                    // SAFETY: covered by interpret_raw's caller contract.
                    unsafe {
                        match graph.value_type(value) {
                            Type::I32 => (addr as *mut u32).write_unaligned(f.read(value) as u32),
                            Type::Ptr => (addr as *mut u64).write_unaligned(f.read(value)),
                        }
                    }
                }
                continue;
            }
            Op::Call(call) => {
                let (callee, args) = graph.call(call);
                let child = Frame::new(callee, args.iter().map(|a| f.read(*a)), graph.result(inst));
                frames.push(child);
                continue;
            }
        };
        f.values[graph.result(inst).unwrap()] = result;
    }
}
