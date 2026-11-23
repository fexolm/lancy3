use crate::hir::{Block, Edge, Function, Inst, Op, Simd, Terminator, Type, Value};
use crate::hir::{
    eval::{Bits, evaluate, read},
    operation::operation,
};
use crate::support::slotmap::DenseMap;
use crate::{CallError, Trap};

/// Why the test interpreter stopped without a result.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum InterpretError {
    /// The trap native code raises for the same input.
    #[error(transparent)]
    Trap(#[from] CallError),
    /// A fixture access left its run buffer.
    #[error("access outside buffer {buffer} at offset {offset}")]
    OutOfBuffer { buffer: usize, offset: u64 },
    /// The interpreter cannot model this signature or operation.
    #[error("unsupported by the interpreter: {0}")]
    Unsupported(&'static str),
    /// The function does not verify.
    #[error("invalid IR: {0}")]
    Invalid(String),
}

impl From<Trap> for InterpretError {
    fn from(trap: Trap) -> Self {
        Self::Trap(trap.into())
    }
}

type Result<T> = std::result::Result<T, InterpretError>;

struct Frame<'a> {
    function: &'a Function,
    block: Block,
    next: Inst,
    values: DenseMap<Value, Bits>,
    return_to: Option<Value>,
    transfer: Vec<Bits>,
    /// This invocation's scratch memory, in words to keep it 8-byte aligned.
    scratch: Vec<u64>,
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
            scratch: vec![0; (function.scratch_bytes() / 8) as usize],
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

/// Run buffer `k` starts at address `k << 32`, so a pointer names its buffer
/// and every access is checked against that buffer alone.
pub(crate) fn buffer_address(buffer: usize) -> u64 {
    (buffer as u64) << 32
}

fn buffer_value(buffers: &mut [Vec<u8>], addr: u64, ty: Type, width: usize) -> Result<&mut [u8]> {
    if ty == Type::Ptr {
        return Err(InterpretError::Unsupported(
            "native pointers in a run buffer",
        ));
    }
    let (buffer, offset) = ((addr >> 32) as usize, addr & u64::from(u32::MAX));
    buffers
        .get_mut(buffer)
        .and_then(|bytes| bytes.get_mut(offset as usize..))
        .and_then(|tail| tail.get_mut(..ty.bytes(width)))
        .ok_or(InterpretError::OutOfBuffer { buffer, offset })
}

pub(super) fn word(x: u64) -> Bits {
    let mut bits = [0; 64];
    bits[..8].copy_from_slice(&x.to_le_bytes());
    bits
}

/// # Safety
/// Without `buffer`, pointer arguments and addresses read from memory must be
/// valid for every executed access. With it, pointers are
/// [`buffer_address`]es and every access is checked.
pub(super) unsafe fn run(
    function: &Function,
    args: Vec<Bits>,
    mut buffer: Option<&mut [Vec<u8>]>,
) -> Result<Bits> {
    let width = crate::ScalarKernel::vector_bytes();
    let mut frames = vec![Frame::new(function, args, None)];
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
                    return Err(Trap::Bounds.into());
                }
                continue;
            }
            Op::CheckOverflow(cond) => {
                if f.read(cond) as u32 != 0 {
                    return Err(Trap::Overflow.into());
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
                    // SAFETY: the address comes from interpret_raw's caller, which supplies
                    // a live readable range of the declared width, or from CallScratch,
                    // whose verified range lies in a live frame's `scratch`.
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
                    // SAFETY: interpret_raw's caller supplies a writable range disjoint from
                    // interpreter storage; a CallScratch range lies in a live frame's `scratch`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), addr as *mut u8, src.len())
                    };
                }
                continue;
            }
            Op::DebugStore(..) | Op::DebugPoint(_) => continue,
            Op::RustCall(import, a, b) => {
                let status = ((import.resolve)())(f.read(a) as i32, f.read(b) as i32);
                word(u64::from(splice_target::abi::decode_status(status)?))
            }
            Op::RustPacketCall(site) => {
                let (import, _, words) = graph.packet_call(site);
                let arguments: Vec<_> = words.iter().map(|&value| f.read(value) as i32).collect();
                let status = import.invoke(&arguments);
                word(u64::from(splice_target::abi::decode_status(status)?))
            }
            Op::RustContextCall(..) => {
                return Err(InterpretError::Unsupported("Rust context calls"));
            }
            Op::CallScratch { .. } if buffer.is_some() => {
                return Err(InterpretError::Unsupported(
                    "call scratch with a run buffer",
                ));
            }
            Op::CallScratch { offset, .. } => {
                word(f.scratch.as_mut_ptr() as u64 + u64::from(offset))
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
                let (action, input, output, inputs) =
                    operation(op, |v| graph.value_type(v)).unwrap();
                let args = inputs.map(|v| v.map_or([0; 64], |v| f.values[v]));
                evaluate(action, input, output, &args, width)?
            }
        };
        f.values[graph.result(inst).unwrap()] = result;
    }
}
