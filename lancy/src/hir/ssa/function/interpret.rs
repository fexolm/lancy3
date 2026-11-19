use crate::hir::{Block, Edge, Function, Inst, Op, Simd, Terminator, Type, Value};
use crate::numbers::{Bits, Math, read};
use crate::support::slotmap::DenseMap;

struct Frame<'a> {
    function: &'a Function,
    block: Block,
    next: Inst,
    values: DenseMap<Value, Bits>,
    return_to: Option<Value>,
    transfer: Vec<Bits>,
}

impl<'a> Frame<'a> {
    fn new(
        function: &'a Function,
        args: impl IntoIterator<Item = Bits>,
        return_to: Option<Value>,
    ) -> Self {
        let block = function.entry();
        let mut values = DenseMap::filled(function.value_count(), [0; 64]);
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
        read(&self.values[value][..8]) as u64
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

fn buffer_value(bytes: &mut [u8], addr: u64, ty: Type, width: usize) -> Result<&mut [u8], String> {
    if ty == Type::Ptr {
        return Err("filetest buffer cannot contain native pointers".into());
    }
    bytes
        .get_mut(addr as usize..)
        .and_then(|tail| tail.get_mut(..ty.bytes(width)))
        .ok_or_else(|| format!("filetest access outside buffer at offset {addr}"))
}

fn word(x: u64) -> Bits {
    let mut bits = [0; 64];
    bits[..8].copy_from_slice(&x.to_le_bytes());
    bits
}

pub(super) unsafe fn run(
    function: &Function,
    args: &[u64],
    mut buffer: Option<&mut [u8]>,
) -> Result<u64, String> {
    let width = crate::ScalarKernel::vector_bytes();
    if function.param_types().iter().any(|ty| ty.bytes(width) > 8)
        || function.return_type().bytes(width) > 8
    {
        return Err("expected word-sized root signature".into());
    }
    let mut frames = vec![Frame::new(function, args.iter().copied().map(word), None)];
    loop {
        let f = frames.last_mut().unwrap();
        let graph = f.function;
        let inst = f.next;
        if inst == Inst::INVALID {
            match *graph.terminator(f.block) {
                Terminator::Return(result) => {
                    let result = f.values[result];
                    let ret = f.return_to;
                    frames.pop();
                    if let Some(ret) = ret {
                        frames.last_mut().unwrap().values[ret] = result;
                    } else {
                        return Ok(read(&result[..8]) as u64);
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
            Op::Const(n) => word(n as u32 as u64),
            Op::Literal(lane, bits) => {
                let mut value = [0; 64];
                let size = lane.bytes();
                value[..size].copy_from_slice(&bits.to_le_bytes()[..size]);
                value
            }
            Op::Simd(Simd::LaneCount(l)) => word((width / l.bytes()) as u64),
            Op::Copy(v) => f.values[v],
            Op::BoundsCheck(index, len) => {
                if f.read(index) as u32 >= f.read(len) as u32 {
                    return Err("kernel slice index out of bounds".into());
                }
                continue;
            }
            Op::PtrOffset(base, index, scale) => word(
                f.read(base)
                    .wrapping_add((f.read(index) as i32 as i64 as u64) << scale as u8),
            ),
            Op::Simd(Simd::MaskedLoad {
                address,
                mask,
                passthrough: v,
            })
            | Op::Simd(Simd::MaskedStore {
                address,
                mask,
                value: v,
            }) => {
                let load = matches!(graph.op(inst), Op::Simd(Simd::MaskedLoad { .. }));
                let lane = graph.value_type(v).lane().unwrap();
                let size = lane.bytes();
                let mut result = f.values[v];
                for (i, data) in result[..width].chunks_exact_mut(size).enumerate() {
                    if f.values[mask][(i + 1) * size - 1] & 0x80 == 0 {
                        continue;
                    }
                    let offset = address
                        .indices
                        .map_or(i as u64, |idx| {
                            let raw = read(&f.values[idx][i * size..(i + 1) * size]);
                            let shift = 128 - 8 * size;
                            (((raw << shift) as i128) >> shift) as u64
                        })
                        .wrapping_mul(size as u64);
                    let addr = f.read(address.base).wrapping_add(offset);
                    if let Some(buf) = buffer.as_deref_mut() {
                        let target = buffer_value(buf, addr, lane.scalar(), width)?;
                        if load {
                            data.copy_from_slice(target);
                        } else {
                            target.copy_from_slice(data);
                        }
                    } else {
                        // SAFETY: interpret_raw's caller covers the executed (active) lane range.
                        unsafe {
                            if load {
                                std::ptr::copy_nonoverlapping(
                                    addr as *const u8,
                                    data.as_mut_ptr(),
                                    size,
                                )
                            } else {
                                std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, size)
                            }
                        }
                    }
                }
                if !load {
                    continue;
                }
                result
            }
            Op::Load(_, addr) | Op::Simd(Simd::Load(_, addr)) => {
                let ty = graph.value_type(graph.result(inst).unwrap());
                let addr = f.read(addr);
                let mut bits = [0; 64];
                let dest = &mut bits[..ty.bytes(width)];
                if let Some(buf) = buffer.as_deref_mut() {
                    dest.copy_from_slice(buffer_value(buf, addr, ty, width)?);
                } else {
                    // SAFETY: interpret_raw's caller supplies a live readable range of the declared width.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            addr as *const u8,
                            dest.as_mut_ptr(),
                            dest.len(),
                        )
                    };
                }
                bits
            }
            Op::Store(addr, v) | Op::Simd(Simd::Store(addr, v)) => {
                let addr = f.read(addr);
                let ty = graph.value_type(v);
                let src = &f.values[v][..ty.bytes(width)];
                if let Some(buf) = buffer.as_deref_mut() {
                    buffer_value(buf, addr, ty, width)?.copy_from_slice(src);
                } else {
                    // SAFETY: interpret_raw's caller supplies a writable range disjoint from interpreter storage.
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), addr as *mut u8, src.len())
                    };
                }
                continue;
            }
            Op::Call(call) => {
                let (callee, args) = graph.call(call);
                let child = Frame::new(
                    callee,
                    args.iter().map(|a| f.values[*a]),
                    graph.result(inst),
                );
                frames.push(child);
                continue;
            }
            op => {
                let (math, inputs) = Math::from_op(op, |v| graph.value_type(v)).unwrap();
                let args = inputs.map(|v| v.map_or([0; 64], |v| f.values[v]));
                math.eval(&args, width).map_err(|e| match e {
                    crate::CallError::DivisionOverflow => "signed division overflow".to_owned(),
                    crate::CallError::DivisionByZero => "integer division by zero".to_owned(),
                    e => e.to_string(),
                })?
            }
        };
        f.values[graph.result(inst).unwrap()] = result;
    }
}
