//! One Splice vector occupies consecutive 128-bit CLIF values. This keeps
//! the lane count identical across backends even when the selected Cranelift
//! ISA legalizes a smaller vector than the host width.
use super::return_if;
use crate::lower::trap;
use cranelift_codegen::ir::{
    InstBuilder, MemFlags, StackSlot, StackSlotData, StackSlotKind, Type, Value,
    condcodes::{FloatCC, IntCC},
    types::{I8X16, I32, I32X4, I64},
};
use cranelift_frontend::{FunctionBuilder, Variable};
use splice_core::Trap;
use splice_core::{
    Function,
    hir::{
        Buffer, Reg, ScalarType, Simd, SimdOp, ValueType,
        memory::{self, Address, BeforeTrap, Masked, Slice},
    },
};
use std::cell::Cell;

const CHUNK_BYTES: usize = 16;

/// What lowering needs from the build target.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Machine {
    pub(crate) vector_bytes: usize,
    /// SSSE3 `pshufb`, which zeroes lanes whose control byte has its high bit set.
    pshufb: bool,
}

impl Machine {
    pub(crate) fn new(target: &splice_core::Target) -> Self {
        Self {
            vector_bytes: target.vector_bytes(),
            pshufb: target.x86().is_some_and(|isa| isa.ssse3),
        }
    }
    fn chunks(self) -> usize {
        self.vector_bytes / CHUNK_BYTES
    }
}

/// The 16-byte CLIF vector of lanes of `element`.
fn chunk_type(element: ScalarType) -> Type {
    super::scalar_type(element)
        .by((CHUNK_BYTES / element.bytes()) as u32)
        .expect("a lane element fills a 16-byte vector")
}

/// An unsigned integer element (`is_signed` is also false for floats).
fn unsigned(e: ScalarType) -> bool {
    e.is_integer() && !e.is_signed()
}

/// The lanes of `element` in one chunk.
fn chunk_lanes(element: ScalarType) -> usize {
    CHUNK_BYTES / element.bytes()
}

pub(super) struct Registers {
    starts: Vec<usize>,
    vars: Vec<Variable>,
    /// One vector-sized stack slot that `ExtractWrapping`, `ReplaceWrapping`,
    /// `Expand` and `Compress` share: each uses it only inside the
    /// instruction (BE-12).
    scratch: Cell<Option<StackSlot>>,
}

impl Registers {
    pub(super) fn new(b: &mut FunctionBuilder, f: &Function, m: Machine) -> Self {
        let mut starts = Vec::with_capacity(f.register_count() + 1);
        let mut vars = Vec::with_capacity(f.register_count());
        for index in 0..f.register_count() {
            starts.push(vars.len());
            let (ty, count) = match f.reg_type(Reg(index as u32)) {
                ValueType::Scalar(ty) => (super::scalar_type(ty), 1),
                ValueType::Vector(e) => (chunk_type(e), m.chunks()),
                ValueType::Mask(e) => (chunk_type(e.index()), m.chunks()),
            };
            vars.extend((0..count).map(|_| b.declare_var(ty)));
        }
        starts.push(vars.len());
        Self {
            starts,
            vars,
            scratch: Cell::new(None),
        }
    }

    fn scratch(&self, b: &mut FunctionBuilder, m: Machine) -> StackSlot {
        let slot = self.scratch.get().unwrap_or_else(|| {
            b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                m.vector_bytes as u32,
                4,
            ))
        });
        self.scratch.set(Some(slot));
        slot
    }

    pub(super) fn get(&self, reg: Reg) -> &[Variable] {
        let index = reg.0 as usize;
        &self.vars[self.starts[index]..self.starts[index + 1]]
    }
}

/// A contiguous access start. Every slice length fits `i32`, so clamping
/// keeps an out-of-range start out of range and lets `start + lanes`
/// not wrap.
fn offset(b: &mut FunctionBuilder, start: Value) -> Value {
    let limit = b.ins().iconst(I64, i64::from(i32::MAX));
    b.ins().umin(start, limit)
}

fn constant(b: &mut FunctionBuilder, ty: Type, bytes: &[u8; CHUNK_BYTES]) -> Value {
    let handle = b.func.dfg.constants.insert(bytes.as_slice().into());
    b.ins().vconst(ty, handle)
}

fn lane_constant(
    b: &mut FunctionBuilder,
    element: ScalarType,
    part: usize,
    bits_at: impl Fn(usize) -> u64,
) -> Value {
    let size = element.bytes();
    let mut bytes = [0; CHUNK_BYTES];
    for (i, lane) in bytes.chunks_exact_mut(size).enumerate() {
        let bits = bits_at(part * CHUNK_BYTES / size + i);
        lane.copy_from_slice(&bits.to_le_bytes()[..size]);
    }
    constant(b, chunk_type(element), &bytes)
}

pub(super) fn emit(
    b: &mut FunctionBuilder,
    m: Machine,
    inst: Simd,
    vars: &Registers,
    dst: Option<Reg>,
    args: &[Reg],
    buffers: &[[Value; 2]],
) {
    use SimdOp::*;
    if inst.op.may_trap() {
        let Ok(result) = memory::lower(
            &mut Slices {
                b,
                m,
                vars,
                buffers,
            },
            inst,
            args,
        );
        define(b, vars, dst, result.map_or_else(Vec::new, |v| v.to_vec()));
        return;
    }
    let inputs: Vec<Vec<Value>> = args
        .iter()
        .map(|&r| vars.get(r).iter().map(|&v| b.use_var(v)).collect())
        .collect();
    let element = inst.element;
    let byte_order = MemFlags::new().with_endianness(cranelift_codegen::ir::Endianness::Little);
    let results = match inst.op {
        AlignmentPrefix(buf) => {
            // Only the address modulo the vector width matters.
            let base = b.ins().ireduce(I32, buffers[buf.index()][0]);
            let offset = b.ins().ireduce(I32, inputs[0][0]);
            let scaled = b.ins().ishl_imm(offset, 2);
            let addr = b.ins().iadd(base, scaled);
            let neg = b.ins().ineg(addr);
            let bytes = b.ins().band_imm(neg, (m.vector_bytes - 1) as i64);
            let lanes = b.ins().ushr_imm(bytes, 2);
            vec![b.ins().uextend(I64, lanes)]
        }
        Swizzle => swizzle_words(b, m, &inputs[0], &inputs[1], byte_order),
        Expand => {
            let slot = vars.scratch(b, m);
            expand_words(b, m, slot, &inputs, byte_order)
        }
        Compress => {
            let slot = vars.scratch(b, m);
            compress_words(b, m, slot, &inputs[0], &inputs[1], byte_order)
        }
        Count => {
            let mut total = b.ins().iconst(I32, 0);
            for &mask in &inputs[0] {
                let bits = b.ins().vhigh_bits(I32, mask);
                let count = b.ins().popcnt(bits);
                total = b.ins().iadd(total, count);
            }
            vec![total]
        }
        ReduceSum => {
            let mut total = b.ins().iconst(super::scalar_type(element), 0);
            for &chunk in &inputs[0] {
                for i in 0..chunk_lanes(element) {
                    let lane = b.ins().extractlane(chunk, i as u8);
                    total = b.ins().iadd(total, lane);
                }
            }
            vec![total]
        }
        Widen { to, high } => convert_width(
            b,
            m,
            element,
            to,
            &inputs[0],
            usize::from(high) * to.lanes(m.vector_bytes),
        ),
        Narrow(to) => {
            let sources: Vec<_> = inputs.iter().flatten().copied().collect();
            convert_width(b, m, element, to, &sources, 0)
        }
        ByteWidth => vec![b.ins().iconst(I64, m.vector_bytes as i64)],
        ElementWidth => vec![b.ins().iconst(I64, element.lanes(m.vector_bytes) as i64)],
        FirstN => {
            // Clamp the full unsigned index before narrowing it to a mask lane.
            let width = b.ins().iconst(I64, element.lanes(m.vector_bytes) as i64);
            let count = b.ins().umin(inputs[0][0], width);
            let ty = chunk_type(element.index());
            let count = if ty.lane_type() == I64 {
                count
            } else {
                b.ins().ireduce(ty.lane_type(), count)
            };
            let limit = b.ins().splat(ty, count);
            (0..m.chunks())
                .map(|part| {
                    let indices = lane_constant(b, element.index(), part, |i| i as u64);
                    b.ins().icmp(IntCC::SignedLessThan, indices, limit)
                })
                .collect()
        }
        SplatScalar => {
            let chunk = b.ins().splat(chunk_type(element), inputs[0][0]);
            vec![chunk; m.chunks()]
        }
        Shuffle(order) => {
            let n = element.lanes(m.vector_bytes);
            let chunks: Vec<_> = inputs.iter().flatten().copied().collect();
            (0..n / 4)
                .map(|part| {
                    let idx = std::array::from_fn::<_, 4, _>(|j| order.index(4 * part + j, n));
                    let first = idx[0] / 4;
                    let second = idx
                        .iter()
                        .map(|j| j / 4)
                        .find(|&c| c != first)
                        .unwrap_or(first);
                    let mask: Vec<_> = (0..16)
                        .map(|i| {
                            let src = idx[i / 4];
                            debug_assert!(src / 4 == first || src / 4 == second);
                            (usize::from(src / 4 != first) * 16 + 4 * (src % 4) + i % 4) as u8
                        })
                        .collect();
                    let mask = b.func.dfg.immediates.push(mask.into());
                    let a = b.ins().bitcast(I8X16, byte_order, chunks[first]);
                    let z = b.ins().bitcast(I8X16, byte_order, chunks[second]);
                    let data = b.ins().shuffle(a, z, mask);
                    b.ins().bitcast(I32X4, byte_order, data)
                })
                .collect()
        }
        ExtractWrapping | ReplaceWrapping => {
            let bytes = m.vector_bytes;
            let slot = vars.scratch(b, m);
            for (part, &chunk) in inputs[0].iter().enumerate() {
                b.ins()
                    .stack_store(chunk, slot, (part * CHUNK_BYTES) as i32);
            }
            let index = b
                .ins()
                .band_imm(inputs[1][0], element.lanes(bytes) as i64 - 1);
            let index = b.ins().uextend(I64, index);
            let offset = b
                .ins()
                .ishl_imm(index, i64::from(element.bytes().trailing_zeros()));
            let base = b.ins().stack_addr(I64, slot, 0);
            let addr = b.ins().iadd(base, offset);
            if inst.op == ExtractWrapping {
                let ty = super::scalar_type(element);
                vec![b.ins().load(ty, byte_order, addr, 0)]
            } else {
                b.ins().store(byte_order, inputs[2][0], addr, 0);
                let ty = chunk_type(element);
                (0..inputs[0].len())
                    .map(|part| b.ins().stack_load(ty, slot, (part * CHUNK_BYTES) as i32))
                    .collect()
            }
        }
        Last => {
            let last = (chunk_lanes(element) - 1) as u8;
            vec![b.ins().extractlane(*inputs[0].last().unwrap(), last)]
        }
        PrefixSum => {
            // A CLIF chunk has four i32 lanes. Propagate the previous chunk's
            // final sum after the two intra-chunk shifts.
            let zero = lane_constant(b, element, 0, |_| 0);
            let zero = b.ins().bitcast(I8X16, byte_order, zero);
            let mut carry = None;
            inputs[0]
                .iter()
                .map(|&part| {
                    let mut sum = part;
                    for distance in [4, 8] {
                        let bytes: Vec<u8> = (0..16u8)
                            .map(|i| if i < distance { 16 } else { i - distance })
                            .collect();
                        let imm = b.func.dfg.immediates.push(bytes.into());
                        let bytes = b.ins().bitcast(I8X16, byte_order, sum);
                        let low = b.ins().shuffle(bytes, zero, imm);
                        let low = b.ins().bitcast(I32X4, byte_order, low);
                        sum = b.ins().iadd(sum, low);
                    }
                    if let Some(previous) = carry {
                        let previous = b.ins().splat(I32X4, previous);
                        sum = b.ins().iadd(sum, previous);
                    }
                    carry = Some(b.ins().extractlane(sum, 3));
                    sum
                })
                .collect()
        }
        Any | All => {
            let mut combined = None;
            for &part in &inputs[0] {
                let flag = if inst.op == Any {
                    b.ins().vany_true(part)
                } else {
                    b.ins().vall_true(part)
                };
                combined = Some(match combined {
                    None => flag,
                    Some(prior) if inst.op == Any => b.ins().bor(prior, flag),
                    Some(prior) => b.ins().band(prior, flag),
                });
            }
            vec![combined.unwrap()]
        }
        _ => (0..m.chunks())
            .map(|i| packed(b, inst, &inputs, i))
            .collect(),
    };
    define(b, vars, dst, results);
}

fn define(b: &mut FunctionBuilder, vars: &Registers, dst: Option<Reg>, results: Vec<Value>) {
    if let Some(dst) = dst {
        let output = vars.get(dst);
        debug_assert_eq!(output.len(), results.len());
        for (&v, result) in output.iter().zip(results) {
            b.def_var(v, result);
        }
    }
}

/// Each CLIF swizzle addresses sixteen bytes. Select the relevant source
/// chunk for each output lane and OR the zero-filled partial results.
fn swizzle_words(
    b: &mut FunctionBuilder,
    m: Machine,
    parts: &[Value],
    indices: &[Value],
    order: MemFlags,
) -> Vec<Value> {
    let repeat = b
        .func
        .dfg
        .immediates
        .push(Vec::from([0u8, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]).into());
    let offsets = constant(b, I8X16, &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    let three = lane_constant(b, ScalarType::I32, 0, |_| 3);
    indices
        .iter()
        .map(|&index| {
            // Each source chunk uses the same intra-chunk byte selector. The
            // remaining index bits identify the one chunk allowed to contribute.
            let word = b.ins().band(index, three);
            let byte = b.ins().ishl_imm(word, 2);
            let byte = b.ins().bitcast(I8X16, order, byte);
            let low = b.ins().shuffle(byte, byte, repeat);
            let selector = b.ins().iadd(low, offsets);
            let owner = b.ins().ushr_imm(index, 2);
            let mut result = constant(b, I32X4, &[0; 16]);
            for (part, &source) in parts.iter().enumerate() {
                let id = lane_constant(b, ScalarType::I32, 0, |_| part as u64);
                let valid = b.ins().icmp(IntCC::Equal, owner, id);
                let source = b.ins().bitcast(I8X16, order, source);
                let shuffled = shuffle_bytes(b, m, source, selector);
                let words = b.ins().bitcast(I32X4, order, shuffled);
                let active = b.ins().band(words, valid);
                result = b.ins().bor(result, active);
            }
            result
        })
        .collect()
}

// Our control bytes are either in 0..16 or have their high bit set.
// PSHUFB implements these cases directly; CLIF swizzle also handles 16..127.
fn shuffle_bytes(b: &mut FunctionBuilder, m: Machine, bytes: Value, selector: Value) -> Value {
    if m.pshufb {
        return b.ins().x86_pshufb(bytes, selector);
    }
    b.ins().swizzle(bytes, selector)
}

// One pshufb control vector per four-bit comparison mask. The immutable
// table has process lifetime, including after the JIT builder has been dropped.
static COMPACT_BYTES: [[u8; 16]; 16] = shuffle_controls(false);
static EXPAND_BYTES: [[u8; 16]; 16] = shuffle_controls(true);

const fn shuffle_controls(expand: bool) -> [[u8; 16]; 16] {
    let mut table = [[0x80; 16]; 16];
    let mut mask = 0;
    while mask < 16 {
        let mut lane = 0;
        let mut out = 0;
        while lane < 4 {
            if mask & (1 << lane) != 0 {
                let mut byte = 0;
                while byte < 4 {
                    let input = lane * 4 + byte;
                    if expand {
                        table[mask][input] = out as u8;
                    } else {
                        table[mask][out] = input as u8;
                    }
                    byte += 1;
                    out += 1;
                }
            }
            lane += 1;
        }
        mask += 1;
    }
    table
}

fn expand_words(
    b: &mut FunctionBuilder,
    m: Machine,
    slot: StackSlot,
    inputs: &[Vec<Value>],
    order: MemFlags,
) -> Vec<Value> {
    for (i, &part) in inputs[0].iter().enumerate() {
        b.ins().stack_store(part, slot, (i * 16) as i32);
    }
    let table = b.ins().iconst(I64, EXPAND_BYTES.as_ptr() as usize as i64);
    let mut src = b.ins().stack_addr(I64, slot, 0);
    inputs[1]
        .iter()
        .zip(&inputs[2])
        .map(|(&mask, &fallback)| {
            let bits = b.ins().vhigh_bits(I64, mask);
            let off = b.ins().ishl_imm(bits, 4);
            let ptr = b.ins().iadd(table, off);
            let control = b.ins().load(I8X16, MemFlags::new(), ptr, 0);
            // Each previous chunk consumed at most 16 bytes of the stack slot.
            let bytes = b.ins().load(I8X16, MemFlags::new(), src, 0);
            let expanded = shuffle_bytes(b, m, bytes, control);
            let values = b.ins().bitcast(I32X4, order, expanded);
            let count = b.ins().popcnt(bits);
            let length = b.ins().ishl_imm(count, 2);
            src = b.ins().iadd(src, length);
            b.ins().bitselect(mask, values, fallback)
        })
        .collect()
}

fn compress_words(
    b: &mut FunctionBuilder,
    m: Machine,
    slot: StackSlot,
    parts: &[Value],
    masks: &[Value],
    order: MemFlags,
) -> Vec<Value> {
    let zero = constant(b, I32X4, &[0; 16]);
    for i in 0..parts.len() {
        b.ins().stack_store(zero, slot, (16 * i) as i32);
    }
    let table = b.ins().iconst(I64, COMPACT_BYTES.as_ptr() as usize as i64);
    let mut dest = b.ins().stack_addr(I64, slot, 0);
    for (&data, &mask) in parts.iter().zip(masks) {
        let bits = b.ins().vhigh_bits(I64, mask);
        let off = b.ins().ishl_imm(bits, 4);
        let ptr = b.ins().iadd(table, off);
        let control = b.ins().load(I8X16, MemFlags::new(), ptr, 0);
        let bytes = b.ins().bitcast(I8X16, order, data);
        let compact = shuffle_bytes(b, m, bytes, control);
        // At most four words per previous chunk precede this 16-byte store.
        b.ins().store(MemFlags::new(), compact, dest, 0);
        let count = b.ins().popcnt(bits);
        let length = b.ins().ishl_imm(count, 2);
        dest = b.ins().iadd(dest, length);
    }
    (0..parts.len())
        .map(|i| b.ins().stack_load(I32X4, slot, (16 * i) as i32))
        .collect()
}

/// The wrapping product of byte lanes: CLIF multiplies 16-bit and wider
/// lanes, so multiply the zero-extended halves and keep the low bytes.
fn byte_product(b: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    let order = MemFlags::new().with_endianness(cranelift_codegen::ir::Endianness::Little);
    let halves = [true, false].map(|low| {
        let (x, y) = if low {
            (b.ins().uwiden_low(x), b.ins().uwiden_low(y))
        } else {
            (b.ins().uwiden_high(x), b.ins().uwiden_high(y))
        };
        let product = b.ins().imul(x, y);
        b.ins().bitcast(I8X16, order, product)
    });
    let even: Vec<u8> = (0..16).map(|i| 2 * i).collect();
    let even = b.func.dfg.immediates.push(even.into());
    b.ins().shuffle(halves[0], halves[1], even)
}

/// Saturating sums and differences. CLIF has them for 8- and 16-bit
/// lanes; wider lanes detect the overflow of the wrapping result.
fn saturating(b: &mut FunctionBuilder, inst: Simd, x: Value, y: Value) -> Value {
    let add = inst.op == SimdOp::SaturatingAdd;
    let e = inst.element;
    if e.bytes() <= 2 {
        return match (add, e.is_signed()) {
            (true, true) => b.ins().sadd_sat(x, y),
            (true, false) => b.ins().uadd_sat(x, y),
            (false, true) => b.ins().ssub_sat(x, y),
            (false, false) => b.ins().usub_sat(x, y),
        };
    }
    if !e.is_signed() {
        // An unsigned sum overflows below an operand; a difference is
        // zero unless `x` is larger.
        return if add {
            let sum = b.ins().iadd(x, y);
            let wrapped = b.ins().icmp(IntCC::UnsignedLessThan, sum, x);
            b.ins().bor(sum, wrapped)
        } else {
            let larger = b.ins().umax(x, y);
            b.ins().isub(larger, y)
        };
    }
    // A signed result overflows when its sign differs from both operands
    // of a sum (from `x` and from `-y` for a difference); it saturates
    // toward the sign of `x`.
    let wrapped = if add {
        b.ins().iadd(x, y)
    } else {
        b.ins().isub(x, y)
    };
    let from_x = b.ins().bxor(wrapped, x);
    let other = if add {
        b.ins().bxor(wrapped, y)
    } else {
        b.ins().bxor(x, y)
    };
    let overflow = b.ins().band(from_x, other);
    let bits = i64::from(e.bits()) - 1;
    let overflow = b.ins().sshr_imm(overflow, bits);
    let sign = b.ins().sshr_imm(x, bits);
    let ones = b.ins().icmp(IntCC::Equal, x, x);
    let max = b.ins().ushr_imm(ones, 1);
    let limit = b.ins().bxor(sign, max);
    b.ins().bitselect(overflow, limit, wrapped)
}

fn packed(b: &mut FunctionBuilder, inst: Simd, inputs: &[Vec<Value>], part: usize) -> Value {
    use SimdOp::*;
    let op = inst.op;
    let ty = chunk_type(inst.element);
    let float = ty.lane_type().is_float();
    let arg = |i: usize| inputs[i][part];
    match op {
        Splat(bits) => lane_constant(b, inst.element, part, |_| bits),
        Iota => lane_constant(b, inst.element, part, |i| match inst.element {
            ScalarType::F32 => u64::from((i as f32).to_bits()),
            ScalarType::F64 => (i as f64).to_bits(),
            _ => i as u64,
        }),
        Shr(n) if unsigned(inst.element) => b.ins().ushr_imm(arg(0), i64::from(n & 31)),
        Shr(n) => b.ins().sshr_imm(arg(0), i64::from(n & 31)),
        RotateLeft(n) => {
            let count = n & 31;
            let x = arg(0);
            if count == 0 {
                return x;
            }
            let high = b.ins().ishl_imm(x, i64::from(count));
            let low = b.ins().ushr_imm(x, i64::from(32 - count));
            b.ins().bor(high, low)
        }
        Add if float => b.ins().fadd(arg(0), arg(1)),
        Sub if float => b.ins().fsub(arg(0), arg(1)),
        Mul if float => b.ins().fmul(arg(0), arg(1)),
        Div => b.ins().fdiv(arg(0), arg(1)),
        Sqrt => b.ins().sqrt(arg(0)),
        MulAdd => b.ins().fma(arg(0), arg(1), arg(2)),
        Cast(to) => convert(b, inst.element, to, chunk_type(to), arg(0)),
        Add => b.ins().iadd(arg(0), arg(1)),
        Sub => b.ins().isub(arg(0), arg(1)),
        Mul if inst.element.bytes() == 1 => byte_product(b, arg(0), arg(1)),
        Mul => b.ins().imul(arg(0), arg(1)),
        SaturatingAdd | SaturatingSub => saturating(b, inst, arg(0), arg(1)),
        Min if unsigned(inst.element) => b.ins().umin(arg(0), arg(1)),
        Max if unsigned(inst.element) => b.ins().umax(arg(0), arg(1)),
        Min => b.ins().smin(arg(0), arg(1)),
        Max => b.ins().smax(arg(0), arg(1)),
        And | MaskAnd => b.ins().band(arg(0), arg(1)),
        Or | MaskOr => b.ins().bor(arg(0), arg(1)),
        Xor | MaskXor => b.ins().bxor(arg(0), arg(1)),
        Not | MaskNot => b.ins().bnot(arg(0)),
        Eq | Ne | Lt | Le | Gt | Ge if float => {
            let cc = match op {
                Eq => FloatCC::Equal,
                Ne => FloatCC::NotEqual,
                Lt => FloatCC::LessThan,
                Le => FloatCC::LessThanOrEqual,
                Gt => FloatCC::GreaterThan,
                Ge => FloatCC::GreaterThanOrEqual,
                _ => unreachable!(),
            };
            b.ins().fcmp(cc, arg(0), arg(1))
        }
        Eq | Ne | Lt | Le | Gt | Ge => {
            let unsigned = unsigned(inst.element);
            let cc = match op {
                Eq => IntCC::Equal,
                Ne => IntCC::NotEqual,
                Lt if unsigned => IntCC::UnsignedLessThan,
                Le if unsigned => IntCC::UnsignedLessThanOrEqual,
                Gt if unsigned => IntCC::UnsignedGreaterThan,
                Ge if unsigned => IntCC::UnsignedGreaterThanOrEqual,
                Lt => IntCC::SignedLessThan,
                Le => IntCC::SignedLessThanOrEqual,
                Gt => IntCC::SignedGreaterThan,
                Ge => IntCC::SignedGreaterThanOrEqual,
                _ => unreachable!(),
            };
            b.ins().icmp(cc, arg(0), arg(1))
        }
        AddMasked | Select => {
            let (mask, yes, no) = if op == AddMasked {
                let sum = if float {
                    b.ins().fadd(arg(0), arg(2))
                } else {
                    b.ins().iadd(arg(0), arg(2))
                };
                (arg(1), sum, arg(3))
            } else {
                (arg(0), arg(1), arg(2))
            };
            let mask = if float {
                b.ins().bitcast(ty, MemFlags::new(), mask)
            } else {
                mask
            };
            b.ins().bitselect(mask, yes, no)
        }
        _ => unreachable!("memory and scalar results are emitted for the whole vector"),
    }
}

fn convert(b: &mut FunctionBuilder, from: ScalarType, to: ScalarType, ty: Type, x: Value) -> Value {
    if b.func.dfg.value_type(x) == ty {
        return x;
    }
    match (from.is_float(), to.is_float()) {
        (true, true) if from.bytes() < to.bytes() => b.ins().fpromote(ty, x),
        (true, true) => b.ins().fdemote(ty, x),
        (false, true) if unsigned(from) => b.ins().fcvt_from_uint(ty, x),
        (false, true) => b.ins().fcvt_from_sint(ty, x),
        // The x64 backend lowers saturating float-to-int conversions only
        // into 32-bit lanes (BE-02); convert 64-bit lanes one at a time.
        (true, false) if cfg!(target_arch = "x86_64") && ty.is_vector() && ty.lane_bits() == 64 => {
            let lanes: Vec<_> = (0..ty.lane_count() as u8)
                .map(|i| {
                    let lane = b.ins().extractlane(x, i);
                    convert(b, from, to, ty.lane_type(), lane)
                })
                .collect();
            gather_lanes(b, ty, &lanes)
        }
        (true, false) if unsigned(to) => b.ins().fcvt_to_uint_sat(ty, x),
        (true, false) => b.ins().fcvt_to_sint_sat(ty, x),
        (false, false) if from.bytes() > to.bytes() => b.ins().ireduce(ty, x),
        (false, false) if unsigned(from) => b.ins().uextend(ty, x),
        (false, false) => b.ins().sextend(ty, x),
    }
}

fn convert_width(
    b: &mut FunctionBuilder,
    machine: Machine,
    from: ScalarType,
    to: ScalarType,
    sources: &[Value],
    start: usize,
) -> Vec<Value> {
    let out = chunk_type(to);
    let n = CHUNK_BYTES / from.bytes();
    let m = CHUNK_BYTES / to.bytes();
    (0..to.lanes(machine.vector_bytes) / m)
        .map(|part| {
            let mut lane = |i| {
                let source = start + part * m + i;
                let x = b.ins().extractlane(sources[source / n], (source % n) as u8);
                convert(b, from, to, out.lane_type(), x)
            };
            let lanes: Vec<_> = (0..m).map(&mut lane).collect();
            gather_lanes(b, out, &lanes)
        })
        .collect()
}

fn gather_lanes(b: &mut FunctionBuilder, ty: Type, lanes: &[Value]) -> Value {
    let mut result = b.ins().splat(ty, lanes[0]);
    for (i, &value) in lanes.iter().enumerate().skip(1) {
        result = b.ins().insertlane(result, value, i as u8);
    }
    result
}

/// The 128-bit parts of one vector, up to the widest scalable vector.
#[derive(Clone, Copy)]
struct Chunks {
    parts: [Value; 16],
    len: usize,
}

impl std::ops::Deref for Chunks {
    type Target = [Value];
    fn deref(&self) -> &[Value] {
        &self.parts[..self.len]
    }
}

impl FromIterator<Value> for Chunks {
    fn from_iter<I: IntoIterator<Item = Value>>(values: I) -> Self {
        use cranelift_codegen::entity::EntityRef;
        let mut chunks = Self {
            parts: [Value::new(0); 16],
            len: 0,
        };
        for value in values {
            chunks.parts[chunks.len] = value;
            chunks.len += 1;
        }
        chunks
    }
}

/// The slices of the function being lowered, for [`memory::lower`].
struct Slices<'a, 'f> {
    b: &'a mut FunctionBuilder<'f>,
    m: Machine,
    vars: &'a Registers,
    buffers: &'a [[Value; 2]],
}

type Lowered<T> = Result<T, std::convert::Infallible>;

impl Slices<'_, '_> {
    fn length(&mut self, buffer: Buffer) -> Value {
        let len = self.buffers[buffer.index()][1];
        self.b.ins().uextend(I64, len)
    }
    /// An access to the lanes at `at`.
    fn access<'v>(&self, s: Slice, at: &'v Address<Value, Chunks>) -> Memory<'v> {
        Memory {
            m: self.m,
            data: self.buffers[s.buffer.index()][0],
            element: s.element,
            address: match at {
                Address::Offset(offset) => Lanes::Contiguous(*offset),
                Address::Indices(indices) => Lanes::Indexed(indices),
            },
            mask: None,
            values: None,
            fallback: None,
        }
    }
    /// The bits of the lanes of `mask`, lane 0 lowest.
    fn mask_bits(&mut self, mask: Chunks) -> Value {
        let b = &mut self.b;
        let per_chunk = b.func.dfg.value_type(mask[0]).lane_count();
        let mut bits = b.ins().iconst(I64, 0);
        for (part, &chunk) in mask.iter().enumerate() {
            let high = b.ins().vhigh_bits(I64, chunk);
            let high = b.ins().ishl_imm(high, i64::from(per_chunk) * part as i64);
            bits = b.ins().bor(bits, high);
        }
        bits
    }
    /// A scalar count or length as a lane of `ty`, splatted.
    fn splat_lane(&mut self, ty: Type, x: Value) -> Value {
        let x = if ty.lane_type() == I64 {
            x
        } else {
            self.b.ins().ireduce(ty.lane_type(), x)
        };
        self.b.ins().splat(ty, x)
    }
}

impl memory::Primitives for Slices<'_, '_> {
    type Scalar = Value;
    type Vector = Chunks;
    type Error = std::convert::Infallible;

    fn read_offset(&mut self, r: Reg) -> Lowered<Value> {
        let start = self.b.use_var(self.vars.get(r)[0]);
        Ok(offset(self.b, start))
    }
    fn read_vector(&mut self, r: Reg) -> Lowered<Chunks> {
        Ok(self
            .vars
            .get(r)
            .iter()
            .map(|&v| self.b.use_var(v))
            .collect())
    }
    fn require_full(
        &mut self,
        s: Slice,
        offset: Value,
        before_trap: BeforeTrap<'_, Self>,
    ) -> Lowered<()> {
        // offset() clamped the start, so this addition cannot wrap into the slice.
        let lanes = s.element.lanes(self.m.vector_bytes) as i64;
        let end = self.b.ins().iadd_imm(offset, lanes);
        let len = self.length(s.buffer);
        let outside = self.b.ins().icmp(IntCC::UnsignedGreaterThan, end, len);
        self.trap_if(outside, before_trap)
    }
    fn room(&mut self, s: Slice, offset: Value) -> Lowered<Value> {
        let len = self.length(s.buffer);
        let end = self.b.ins().umax(offset, len);
        Ok(self.b.ins().isub(end, offset))
    }
    fn first_lanes(&mut self, e: ScalarType, count: Value) -> Lowered<Chunks> {
        // A room can exceed the lanes, and the narrowest lane type.
        let lanes = self
            .b
            .ins()
            .iconst(I64, e.lanes(self.m.vector_bytes) as i64);
        let count = self.b.ins().umin(count, lanes);
        let limit = self.splat_lane(chunk_type(e.index()), count);
        Ok((0..self.m.chunks())
            .map(|part| {
                let ord = lane_constant(self.b, e.index(), part, |i| i as u64);
                self.b.ins().icmp(IntCC::SignedLessThan, ord, limit)
            })
            .collect())
    }
    fn outside(&mut self, s: Slice, indices: Chunks) -> Lowered<Chunks> {
        let index = s.element.index();
        let mut len = self.length(s.buffer);
        // Compare narrow lanes against the length capped where it still fits
        // them: at the first negative index (unsigned, every negative index is
        // at least that), or at the largest unsigned index, where `memory`
        // ignores the result for a longer slice.
        if index.bits() < 32 {
            let cap = if index.is_signed() {
                1i64 << (index.bits() - 1)
            } else {
                (1i64 << index.bits()) - 1
            };
            let ty = self.b.func.dfg.value_type(len);
            let cap = self.b.ins().iconst(ty, cap);
            len = self.b.ins().umin(len, cap);
        }
        let limit = self.splat_lane(chunk_type(index), len);
        Ok(indices
            .iter()
            .map(|&chunk| {
                self.b
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, chunk, limit)
            })
            .collect())
    }
    fn length_at_most(&mut self, buffer: Buffer, n: u64) -> Lowered<Value> {
        let len = self.length(buffer);
        let ins = self.b.ins();
        Ok(ins.icmp_imm(IntCC::UnsignedLessThanOrEqual, len, n as i64))
    }
    fn and(&mut self, a: Value, b: Value) -> Lowered<Value> {
        Ok(self.b.ins().band(a, b))
    }
    fn mask_and(&mut self, a: Chunks, b: Chunks) -> Lowered<Chunks> {
        Ok(a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| self.b.ins().band(x, y))
            .collect())
    }
    fn mask_xor(&mut self, a: Chunks, b: Chunks) -> Lowered<Chunks> {
        Ok(a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| self.b.ins().bxor(x, y))
            .collect())
    }
    fn any(&mut self, mask: Chunks) -> Lowered<Value> {
        let b = &mut self.b;
        let all = mask
            .iter()
            .copied()
            .reduce(|x, y| b.ins().bor(x, y))
            .unwrap();
        Ok(b.ins().vany_true(all))
    }
    fn before_first(&mut self, e: ScalarType, mask: Chunks) -> Lowered<Chunks> {
        // lowbit(bits) - 1 has every bit below the first lane, and all bits
        // when the mask is empty. Lane `i` of each part tests bit `i`.
        let bits = self.mask_bits(mask);
        let negated = self.b.ins().ineg(bits);
        let low = self.b.ins().band(bits, negated);
        let before = self.b.ins().iadd_imm(low, -1);
        let ty = chunk_type(e.index());
        let per_chunk = ty.lane_count() as usize;
        let lane = lane_constant(self.b, e.index(), 0, |i| 1 << i);
        let zero = constant(self.b, ty, &[0; CHUNK_BYTES]);
        Ok((0..mask.len())
            .map(|part| {
                let part = self.b.ins().ushr_imm(before, (per_chunk * part) as i64);
                let part = self.splat_lane(ty, part);
                let set = self.b.ins().band(part, lane);
                self.b.ins().icmp(IntCC::NotEqual, set, zero)
            })
            .collect())
    }
    fn exceeds(&mut self, mask: Chunks, count: Value) -> Lowered<Value> {
        let bits = self.mask_bits(mask);
        let lanes = self.b.ins().popcnt(bits);
        Ok(self.b.ins().icmp(IntCC::UnsignedGreaterThan, lanes, count))
    }
    fn trap_if(&mut self, condition: Value, before_trap: BeforeTrap<'_, Self>) -> Lowered<()> {
        let Some(before_trap) = before_trap else {
            let fault = trap(self.b, Trap::Bounds);
            return_if(self.b, condition, fault);
            return Ok(());
        };
        let fail = self.b.create_block();
        let next = self.b.create_block();
        self.b.ins().brif(condition, fail, &[], next, &[]);
        self.b.switch_to_block(fail);
        before_trap(self)?;
        // Every block of the path that traps is cold.
        let blocks = self.b.func.dfg.num_blocks() as u32;
        for block in (fail.as_u32()..blocks).map(cranelift_codegen::ir::Block::from_u32) {
            if block != next {
                self.b.set_cold_block(block);
            }
        }
        let fault = trap(self.b, Trap::Bounds);
        self.b.ins().return_(&[fault]);
        self.b.switch_to_block(next);
        Ok(())
    }
    fn load(
        &mut self,
        s: Slice,
        at: Address<Value, Chunks>,
        masked: Option<Masked<Chunks>>,
    ) -> Lowered<Chunks> {
        let mut access = self.access(s, &at);
        let Some(Masked { mask, passthrough }) = &masked else {
            return Ok(match at {
                Address::Offset(_) => access.full(self.b),
                Address::Indices(_) => access.ordered(self.b),
            }
            .into_iter()
            .collect());
        };
        (access.mask, access.fallback) = (Some(&mask[..]), Some(&passthrough[..]));
        Ok(match at {
            Address::Offset(_) => access.masked(self.b),
            Address::Indices(ref indices) => {
                access.masked_gather(self.b, indices, mask, passthrough)
            }
        }
        .into_iter()
        .collect())
    }
    fn store(
        &mut self,
        s: Slice,
        at: Address<Value, Chunks>,
        mask: Option<Chunks>,
        value: Chunks,
    ) -> Lowered<()> {
        let mut access = self.access(s, &at);
        (access.mask, access.values) = (mask.as_deref(), Some(&value[..]));
        match (&at, mask) {
            (_, Some(_)) => access.masked(self.b),
            (Address::Offset(_), None) => access.full(self.b),
            (Address::Indices(_), None) => access.ordered(self.b),
        };
        Ok(())
    }
    fn compress_store(
        &mut self,
        s: Slice,
        offset: Value,
        mask: Chunks,
        value: Chunks,
        room: Value,
    ) -> Lowered<()> {
        let data = self.buffers[s.buffer.index()][0];
        let b = &mut self.b;
        let end = b.ins().iadd(offset, room);
        let mut cursor = offset;
        for (&mask, &values) in mask.iter().zip(value.iter()) {
            for lane in 0..4 {
                let active = b.ins().extractlane(mask, lane);
                let active = b.ins().icmp_imm(IntCC::NotEqual, active, 0);
                let fits = b.ins().icmp(IntCC::UnsignedLessThan, cursor, end);
                let write = b.ins().band(active, fits);
                let store = b.create_block();
                let next = b.create_block();
                let advanced = b.append_block_param(next, I64);
                b.ins().brif(write, store, &[], next, &[cursor.into()]);
                b.switch_to_block(store);
                let bytes = b.ins().imul_imm(cursor, 4);
                let ptr = b.ins().iadd(data, bytes);
                let value = b.ins().extractlane(values, lane);
                b.ins().store(MemFlags::new(), value, ptr, 0);
                let after = b.ins().iadd_imm(cursor, 1);
                b.ins().jump(next, &[after.into()]);
                b.switch_to_block(next);
                cursor = advanced;
            }
        }
        Ok(())
    }
}

enum Lanes<'a> {
    Contiguous(Value),
    Indexed(&'a [Value]),
}

/// One unchecked access: every lane it touches is in bounds.
struct Memory<'a> {
    m: Machine,
    data: Value,
    element: ScalarType,
    address: Lanes<'a>,
    mask: Option<&'a [Value]>,
    values: Option<&'a [Value]>,
    fallback: Option<&'a [Value]>,
}

impl Memory<'_> {
    fn pointer(&self, b: &mut FunctionBuilder, index: Value) -> Value {
        let bytes = b.ins().imul_imm(index, self.element.bytes() as i64);
        b.ins().iadd(self.data, bytes)
    }

    /// Every lane of a contiguous vector.
    fn full(&self, b: &mut FunctionBuilder) -> Vec<Value> {
        let Lanes::Contiguous(start) = self.address else {
            unreachable!("a contiguous access")
        };
        let base = self.pointer(b, start);
        let mut out = Vec::new();
        for part in 0..self.m.chunks() {
            let offset = (part * CHUNK_BYTES) as i32;
            if let Some(values) = self.values {
                b.ins().store(MemFlags::new(), values[part], base, offset);
            } else {
                out.push(
                    b.ins()
                        .load(chunk_type(self.element), MemFlags::new(), base, offset),
                );
            }
        }
        out
    }
    // A disabled lane can form an arbitrary address. Select a readable static
    // word before the load and merge its caller-provided fallback afterwards.
    // The word is aligned for every supported element and outlives the JIT code.
    fn masked_gather(
        &self,
        b: &mut FunctionBuilder,
        indices: &[Value],
        masks: &[Value],
        fallback: &[Value],
    ) -> Vec<Value> {
        static ZERO: u64 = 0;
        let ty = chunk_type(self.element);
        let index_ty = chunk_type(self.element.index());
        let zero = b
            .ins()
            .iconst(I64, std::ptr::from_ref(&ZERO) as usize as i64);
        let mut result = Vec::new();
        for ((&chunk, &mask), &old) in indices.iter().zip(masks).zip(fallback) {
            let active = b.ins().vany_true(mask);
            let some = b.create_block();
            let done = b.create_block();
            let merged = b.append_block_param(done, ty);
            b.ins().brif(active, some, &[], done, &[old.into()]);
            b.switch_to_block(some);
            let mut value = constant(b, ty, &[0; CHUNK_BYTES]);
            for lane in 0..ty.lane_count() as u8 {
                let idx = b.ins().extractlane(chunk, lane);
                let idx = if index_ty.lane_type() == I64 {
                    idx
                } else {
                    b.ins().uextend(I64, idx)
                };
                let addr = self.pointer(b, idx);
                let on = b.ins().extractlane(mask, lane);
                let ptr = b.ins().select(on, addr, zero);
                let item = b.ins().load(ty.lane_type(), MemFlags::new(), ptr, 0);
                value = b.ins().insertlane(value, item, lane);
            }
            let mask = b.ins().bitcast(ty, MemFlags::new(), mask);
            let value = b.ins().bitselect(mask, value, old);
            b.ins().jump(done, &[value.into()]);
            b.switch_to_block(done);
            result.push(merged);
        }
        result
    }

    fn masked(&self, b: &mut FunctionBuilder) -> Vec<Value> {
        let mask = self.mask.expect("masked access");
        let ty = chunk_type(self.element);
        let lanes = ty.lane_count();
        let mut results = Vec::new();
        for (part, &m) in mask.iter().enumerate() {
            let initial = self.initial(b, part);
            let some = b.create_block();
            let done = b.create_block();
            let merged_value =
                initial.map(|v| b.append_block_param(done, b.func.dfg.value_type(v)));
            let any = b.ins().vany_true(m);
            if let Some(value) = initial {
                b.ins().brif(any, some, &[], done, &[value.into()]);
            } else {
                b.ins().brif(any, some, &[], done, &[]);
            }
            b.switch_to_block(some);
            if let Lanes::Contiguous(base) = self.address {
                let start = b.ins().iadd_imm(base, (part * lanes as usize) as i64);
                let full = b.ins().vall_true(m);
                let packed = b.create_block();
                let partial = b.create_block();
                b.ins().brif(full, packed, &[], partial, &[]);
                b.switch_to_block(packed);
                let ptr = self.pointer(b, start);
                if let Some(values) = self.values {
                    b.ins().store(MemFlags::new(), values[part], ptr, 0);
                    b.ins().jump(done, &[]);
                } else {
                    let value = b.ins().load(ty, MemFlags::new(), ptr, 0);
                    b.ins().jump(done, &[value.into()]);
                }
                b.switch_to_block(partial);
            }
            let result = self.ordered_part(b, part, initial);
            if let Some(v) = result {
                b.ins().jump(done, &[v.into()]);
            } else {
                b.ins().jump(done, &[]);
            }
            b.switch_to_block(done);
            results.extend(merged_value);
        }
        results
    }

    fn initial(&self, b: &mut FunctionBuilder, part: usize) -> Option<Value> {
        self.values.is_none().then(|| {
            self.fallback.map_or_else(
                || constant(b, chunk_type(self.element), &[0; CHUNK_BYTES]),
                |v| v[part],
            )
        })
    }
    fn ordered(&self, b: &mut FunctionBuilder) -> Vec<Value> {
        (0..self.m.chunks())
            .filter_map(|part| {
                let value = self.initial(b, part);
                self.ordered_part(b, part, value)
            })
            .collect()
    }
    fn ordered_part(
        &self,
        b: &mut FunctionBuilder,
        part: usize,
        mut value: Option<Value>,
    ) -> Option<Value> {
        let ty = chunk_type(self.element);
        let chunk_lanes = ty.lane_count() as u8;
        for lane in 0..chunk_lanes {
            // An inactive lane skips both its bounds check and its memory access.
            let next = self.mask.map(|mask| {
                let active = b.ins().extractlane(mask[part], lane);
                let access = b.create_block();
                let next = b.create_block();
                if let Some(value) = value {
                    b.append_block_param(next, ty);
                    b.ins().brif(active, access, &[], next, &[value.into()]);
                } else {
                    b.ins().brif(active, access, &[], next, &[]);
                }
                b.switch_to_block(access);
                next
            });
            let index = match self.address {
                Lanes::Contiguous(base) => {
                    let i = part * usize::from(chunk_lanes) + usize::from(lane);
                    b.ins().iadd_imm(base, i as i64)
                }
                Lanes::Indexed(parts) => {
                    let i = b.ins().extractlane(parts[part], lane);
                    if self.element.bytes() == 8 {
                        i
                    } else {
                        b.ins().uextend(I64, i)
                    }
                }
            };
            let ptr = self.pointer(b, index);
            if let Some(values) = self.values {
                let item = b.ins().extractlane(values[part], lane);
                b.ins().store(MemFlags::new(), item, ptr, 0);
            } else {
                let item = b.ins().load(ty.lane_type(), MemFlags::new(), ptr, 0);
                value = Some(b.ins().insertlane(value.unwrap(), item, lane));
            }
            if let Some(next) = next {
                if let Some(v) = value {
                    b.ins().jump(next, &[v.into()]);
                } else {
                    b.ins().jump(next, &[]);
                }
                b.switch_to_block(next);
                value = b.block_params(next).first().copied();
            }
        }
        value
    }
}

/// Cranelift 0.132 folds a vector load into the memory operand of legacy-SSE
/// `cmpps`/`cmppd` without requiring 16-byte alignment, so an unaligned slice
/// load faults (BE-01). Every other legacy-SSE form copies such a load to a
/// register first. Without AVX, route compared loads through a vector bitcast,
/// a register no-op that the x64 lowering cannot fold.
pub(crate) fn keep_compared_loads_in_registers(func: &mut cranelift_codegen::ir::Function) {
    use cranelift_codegen::cursor::{Cursor, FuncCursor};
    use cranelift_codegen::ir::{InstructionData, Opcode, ValueDef};
    let order = MemFlags::new().with_endianness(cranelift_codegen::ir::Endianness::Little);
    let mut pos = FuncCursor::new(func);
    while pos.next_block().is_some() {
        while let Some(inst) = pos.next_inst() {
            let InstructionData::FloatCompare { args, .. } = pos.func.dfg.insts[inst] else {
                continue;
            };
            for (operand, value) in args.into_iter().enumerate() {
                let ty = pos.func.dfg.value_type(value);
                let loaded = matches!(pos.func.dfg.value_def(value),
                    ValueDef::Result(def, _) if pos.func.dfg.insts[def].opcode() == Opcode::Load);
                if ty.is_vector() && loaded {
                    let bits = pos.ins().bitcast(ty.as_int(), order, value);
                    let copy = pos.ins().bitcast(ty, order, bits);
                    pos.func.dfg.inst_args_mut(inst)[operand] = copy;
                }
            }
        }
    }
}
