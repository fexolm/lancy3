//! One Splice vector occupies consecutive 128-bit CLIF values. This keeps
//! the lane count identical across backends even when the selected Cranelift
//! ISA legalizes a smaller vector than the host width.
use super::return_if;
use crate::lower::trap;
use cranelift_codegen::ir::{
    InstBuilder, MemFlags, Type, Value,
    condcodes::{FloatCC, IntCC},
    types::{F32X4, F64X2, I8X16, I16X8, I32, I32X4, I64, I64X2},
};
use cranelift_frontend::{FunctionBuilder, Variable};
use splice_core::Trap;
use splice_core::{
    Function,
    hir::{Reg, Simd, SimdOp, ValueType},
    simd::Element,
};

const CHUNK_BYTES: usize = 16;

fn chunk_type(element: Element) -> Type {
    match element {
        Element::U16 => I16X8,
        Element::I32 | Element::U32 => I32X4,
        Element::I64 => I64X2,
        Element::F32 => F32X4,
        Element::F64 => F64X2,
    }
}

pub(super) struct Registers {
    starts: Vec<usize>,
    vars: Vec<Variable>,
}

impl Registers {
    pub(super) fn new(b: &mut FunctionBuilder, f: &Function) -> Self {
        let mut starts = Vec::with_capacity(f.register_count() + 1);
        let mut vars = Vec::with_capacity(f.register_count());
        for index in 0..f.register_count() {
            starts.push(vars.len());
            let (ty, count) = match f.reg_type(Reg(index as u32)) {
                ValueType::Scalar => (I32, 1),
                ValueType::Vector(e) => (
                    chunk_type(e),
                    splice_core::simd::vector_bytes() / CHUNK_BYTES,
                ),
                ValueType::Mask(e) => (
                    chunk_type(e.index()),
                    splice_core::simd::vector_bytes() / CHUNK_BYTES,
                ),
            };
            vars.extend((0..count).map(|_| b.declare_var(ty)));
        }
        starts.push(vars.len());
        Self { starts, vars }
    }

    pub(super) fn get(&self, reg: Reg) -> &[Variable] {
        let index = reg.0 as usize;
        &self.vars[self.starts[index]..self.starts[index + 1]]
    }
}

fn constant(b: &mut FunctionBuilder, ty: Type, bytes: &[u8; CHUNK_BYTES]) -> Value {
    let handle = b.func.dfg.constants.insert(bytes.as_slice().into());
    b.ins().vconst(ty, handle)
}

fn lane_constant(
    b: &mut FunctionBuilder,
    element: Element,
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
    inst: Simd,
    vars: &Registers,
    dst: Option<Reg>,
    args: &[Reg],
    buffers: &[[Value; 2]],
) {
    use SimdOp::*;
    let inputs: Vec<Vec<Value>> = args
        .iter()
        .map(|&r| vars.get(r).iter().map(|&v| b.use_var(v)).collect())
        .collect();
    let element = inst.element;
    let byte_order = MemFlags::new().with_endianness(cranelift_codegen::ir::Endianness::Little);
    let results = match inst.op {
        Load(buf) | Store(buf) | LoadSelect(buf) | StoreSelect(buf) | Gather(buf)
        | GatherSelect(buf) | Scatter(buf) | ScatterSelect(buf) => {
            let [data, len] = buffers[buf.index()];
            let indexed = matches!(
                inst.op,
                Gather(_) | GatherSelect(_) | Scatter(_) | ScatterSelect(_)
            );
            let masked = matches!(
                inst.op,
                LoadSelect(_) | StoreSelect(_) | GatherSelect(_) | ScatterSelect(_)
            );
            let write = matches!(
                inst.op,
                Store(_) | StoreSelect(_) | Scatter(_) | ScatterSelect(_)
            );
            let address = if indexed {
                Address::Indexed(&inputs[0])
            } else {
                Address::Contiguous(b.ins().uextend(I64, inputs[0][0]))
            };
            let access = Memory {
                data,
                len: b.ins().uextend(I64, len),
                element,
                address,
                mask: masked.then(|| inputs[1].as_slice()),
                values: write.then(|| inputs.last().unwrap().as_slice()),
                fallback: (masked && !write).then(|| inputs[2].as_slice()),
            };
            access.emit(b)
        }
        AlignmentPrefix(buf) => {
            let base = b.ins().ireduce(I32, buffers[buf.index()][0]);
            let scaled = b.ins().ishl_imm(inputs[0][0], 2);
            let addr = b.ins().iadd(base, scaled);
            let neg = b.ins().ineg(addr);
            let bytes = b
                .ins()
                .band_imm(neg, (splice_core::simd::vector_bytes() - 1) as i64);
            vec![b.ins().ushr_imm(bytes, 2)]
        }
        Swizzle => swizzle_words(b, &inputs[0], &inputs[1], byte_order),
        Expand => expand_words(b, &inputs, byte_order),
        Compress => compress_words(b, &inputs[0], &inputs[1], byte_order),
        CompressStore(buf) => {
            let [data, len] = buffers[buf.index()];
            let len = b.ins().uextend(I64, len);
            let mut offset = b.ins().uextend(I64, inputs[0][0]);
            for (&mask, &values) in inputs[1].iter().zip(&inputs[2]) {
                for lane in 0..4 {
                    let active = b.ins().extractlane(mask, lane);
                    let store = b.create_block();
                    let next = b.create_block();
                    let cursor = b.append_block_param(next, I64);
                    b.ins().brif(active, store, &[], next, &[offset.into()]);
                    b.switch_to_block(store);
                    let outside = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, offset, len);
                    let fault = trap(b, Trap::Bounds);
                    return_if(b, outside, fault);
                    let bytes = b.ins().imul_imm(offset, 4);
                    let ptr = b.ins().iadd(data, bytes);
                    let value = b.ins().extractlane(values, lane);
                    b.ins().store(MemFlags::new(), value, ptr, 0);
                    let advanced = b.ins().iadd_imm(offset, 1);
                    b.ins().jump(next, &[advanced.into()]);
                    b.switch_to_block(next);
                    offset = cursor;
                }
            }
            vec![]
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
            let mut total = b.ins().iconst(I32, 0);
            for &chunk in &inputs[0] {
                for i in 0..4 {
                    let lane = b.ins().extractlane(chunk, i);
                    total = b.ins().iadd(total, lane);
                }
            }
            vec![total]
        }
        Widen { to, high } => {
            convert_width(b, element, to, &inputs[0], usize::from(high) * to.lanes())
        }
        Narrow(to) => {
            let sources: Vec<_> = inputs.iter().flatten().copied().collect();
            convert_width(b, element, to, &sources, 0)
        }
        Lanes => vec![b.ins().iconst(I32, element.lanes() as i64)],
        FirstN => {
            // Clamp the full unsigned index before narrowing it to a mask lane.
            let width = b.ins().iconst(I32, element.lanes() as i64);
            let count = b.ins().umin(inputs[0][0], width);
            let ty = chunk_type(element.index());
            let count = match element.bytes() {
                2 => b.ins().ireduce(ty.lane_type(), count),
                8 => b.ins().uextend(I64, count),
                _ => count,
            };
            let limit = b.ins().splat(ty, count);
            (0..splice_core::simd::vector_bytes() / CHUNK_BYTES)
                .map(|part| {
                    let indices = lane_constant(b, element.index(), part, |i| i as u64);
                    b.ins().icmp(IntCC::SignedLessThan, indices, limit)
                })
                .collect()
        }
        SplatScalar => {
            let chunk = b.ins().splat(chunk_type(element), inputs[0][0]);
            vec![chunk; splice_core::simd::vector_bytes() / CHUNK_BYTES]
        }
        Shuffle(order) => {
            let n = element.lanes();
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
            use cranelift_codegen::ir::{StackSlotData, StackSlotKind};
            let bytes = splice_core::simd::vector_bytes();
            let slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                bytes as u32,
                4,
            ));
            for (part, &chunk) in inputs[0].iter().enumerate() {
                b.ins()
                    .stack_store(chunk, slot, (part * CHUNK_BYTES) as i32);
            }
            let index = b.ins().band_imm(inputs[1][0], (bytes / 4 - 1) as i64);
            let index = b.ins().uextend(I64, index);
            let offset = b.ins().ishl_imm(index, 2);
            let base = b.ins().stack_addr(I64, slot, 0);
            let addr = b.ins().iadd(base, offset);
            if inst.op == ExtractWrapping {
                vec![b.ins().load(I32, byte_order, addr, 0)]
            } else {
                b.ins().store(byte_order, inputs[2][0], addr, 0);
                (0..inputs[0].len())
                    .map(|part| b.ins().stack_load(I32X4, slot, (part * CHUNK_BYTES) as i32))
                    .collect()
            }
        }
        Last => vec![b.ins().extractlane(*inputs[0].last().unwrap(), 3)],
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
            vec![b.ins().bmask(I32, combined.unwrap())]
        }
        _ => (0..splice_core::simd::vector_bytes() / CHUNK_BYTES)
            .map(|i| packed(b, inst, &inputs, i))
            .collect(),
    };
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
    let three = lane_constant(b, Element::I32, 0, |_| 3);
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
                let id = lane_constant(b, Element::I32, 0, |_| part as u64);
                let valid = b.ins().icmp(IntCC::Equal, owner, id);
                let source = b.ins().bitcast(I8X16, order, source);
                let shuffled = shuffle_bytes(b, source, selector);
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
fn shuffle_bytes(b: &mut FunctionBuilder, bytes: Value, selector: Value) -> Value {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("ssse3") {
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

fn expand_words(b: &mut FunctionBuilder, inputs: &[Vec<Value>], order: MemFlags) -> Vec<Value> {
    use cranelift_codegen::ir::{StackSlotData, StackSlotKind};
    let slot = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        (inputs[0].len() * 16) as u32,
        4,
    ));
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
            let expanded = shuffle_bytes(b, bytes, control);
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
    parts: &[Value],
    masks: &[Value],
    order: MemFlags,
) -> Vec<Value> {
    use cranelift_codegen::ir::{StackSlotData, StackSlotKind};
    let slot = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        (parts.len() * 16) as u32,
        4,
    ));
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
        let compact = shuffle_bytes(b, bytes, control);
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

fn packed(b: &mut FunctionBuilder, inst: Simd, inputs: &[Vec<Value>], part: usize) -> Value {
    use SimdOp::*;
    let op = inst.op;
    let ty = chunk_type(inst.element);
    let float = ty.lane_type().is_float();
    let arg = |i: usize| inputs[i][part];
    match op {
        Splat(bits) => lane_constant(b, inst.element, part, |_| bits),
        Iota => lane_constant(b, inst.element, part, |i| match inst.element {
            Element::F32 => u64::from((i as f32).to_bits()),
            Element::F64 => (i as f64).to_bits(),
            _ => i as u64,
        }),
        Shr(n) if inst.element.is_unsigned() => b.ins().ushr_imm(arg(0), i64::from(n & 31)),
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
        Mul => b.ins().imul(arg(0), arg(1)),
        Min if inst.element.is_unsigned() => b.ins().umin(arg(0), arg(1)),
        Max if inst.element.is_unsigned() => b.ins().umax(arg(0), arg(1)),
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
            let unsigned = inst.element.is_unsigned();
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
        Select => {
            let mask = if float {
                b.ins().bitcast(ty, MemFlags::new(), arg(0))
            } else {
                arg(0)
            };
            b.ins().bitselect(mask, arg(1), arg(2))
        }
        _ => unreachable!("memory and scalar results are emitted for the whole vector"),
    }
}

fn convert(b: &mut FunctionBuilder, from: Element, to: Element, ty: Type, x: Value) -> Value {
    if b.func.dfg.value_type(x) == ty {
        return x;
    }
    match (from.is_float(), to.is_float()) {
        (true, true) if from.bytes() < to.bytes() => b.ins().fpromote(ty, x),
        (true, true) => b.ins().fdemote(ty, x),
        (false, true) if from.is_unsigned() => b.ins().fcvt_from_uint(ty, x),
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
        (true, false) if to.is_unsigned() => b.ins().fcvt_to_uint_sat(ty, x),
        (true, false) => b.ins().fcvt_to_sint_sat(ty, x),
        (false, false) if from.bytes() > to.bytes() => b.ins().ireduce(ty, x),
        (false, false) if from.is_unsigned() => b.ins().uextend(ty, x),
        (false, false) => b.ins().sextend(ty, x),
    }
}

fn convert_width(
    b: &mut FunctionBuilder,
    from: Element,
    to: Element,
    sources: &[Value],
    start: usize,
) -> Vec<Value> {
    let out = chunk_type(to);
    let n = CHUNK_BYTES / from.bytes();
    let m = CHUNK_BYTES / to.bytes();
    (0..to.lanes() / m)
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

enum Address<'a> {
    Contiguous(Value),
    Indexed(&'a [Value]),
}

struct Memory<'a> {
    data: Value,
    len: Value,
    element: Element,
    address: Address<'a>,
    mask: Option<&'a [Value]>,
    values: Option<&'a [Value]>,
    fallback: Option<&'a [Value]>,
}

impl Memory<'_> {
    fn pointer(&self, b: &mut FunctionBuilder, index: Value) -> Value {
        let bytes = b.ins().imul_imm(index, self.element.bytes() as i64);
        b.ins().iadd(self.data, bytes)
    }

    fn emit(&self, b: &mut FunctionBuilder) -> Vec<Value> {
        let start = match self.address {
            Address::Indexed(indices) => {
                let invalid = self.invalid_indices(b, indices);
                self.guard(b, invalid);
                return match (self.mask, self.fallback) {
                    (Some(mask), Some(fallback)) => self.masked_gather(b, indices, mask, fallback),
                    (Some(_), None) => self.masked(b),
                    (None, _) => self.ordered(b, false),
                };
            }

            Address::Contiguous(start) => start,
        };
        // Widen before addition so even an invalid i32 start cannot wrap into the slice.
        let end = b.ins().iadd_imm(start, self.element.lanes() as i64);
        let invalid = b.ins().icmp(IntCC::UnsignedGreaterThan, end, self.len);
        if let Some(mask) = self.mask {
            // The common case covers the whole vector. When it crosses the
            // end, only enabled lanes need to fit; an empty mask can start anywhere.
            let boundary = b.create_block();
            let access = b.create_block();
            b.ins().brif(invalid, boundary, &[], access, &[]);
            b.switch_to_block(boundary);
            let end = b.ins().umax(start, self.len);
            let room = b.ins().isub(end, start);
            let elem = self.element.index();
            let ty = chunk_type(elem);
            let count = if ty.lane_type() == I64 {
                room
            } else {
                b.ins().ireduce(ty.lane_type(), room)
            };
            let limit = b.ins().splat(ty, count);
            let mut bad = constant(b, ty, &[0; CHUNK_BYTES]);
            for (part, &enabled) in mask.iter().enumerate() {
                let ord = lane_constant(b, elem, part, |i| i as u64);
                let outside = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, ord, limit);
                let failed = b.ins().band(outside, enabled);
                bad = b.ins().bor(bad, failed);
            }
            let failed = b.ins().vany_true(bad);
            self.guard(b, failed);
            b.ins().jump(access, &[]);
            b.switch_to_block(access);
            return self.masked(b);
        }
        self.guard(b, invalid);
        let base = self.pointer(b, start);
        let mut out = Vec::new();
        for part in 0..splice_core::simd::vector_bytes() / CHUNK_BYTES {
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

    fn guard(&self, b: &mut FunctionBuilder, invalid: Value) {
        let fault = trap(b, Trap::Bounds);
        if self.values.is_some() {
            // An invalid store can still write a valid prefix. Cold execution
            // checks and writes each lane in source order, up to the first fault.
            let partial = b.create_block();
            let full = b.create_block();
            b.set_cold_block(partial);
            b.ins().brif(invalid, partial, &[], full, &[]);
            b.switch_to_block(partial);
            self.ordered(b, true);
            b.ins().return_(&[fault]);
            b.switch_to_block(full);
        } else {
            return_if(b, invalid, fault);
        }
    }

    // A read fault exposes no partial result. A write fault takes the ordered
    // cold path in guard() so stores before the first bad active lane survive.
    fn invalid_indices(&self, b: &mut FunctionBuilder, indices: &[Value]) -> Value {
        let ty = chunk_type(self.element.index());
        let lane = ty.lane_type();
        let len = if lane == I64 {
            self.len
        } else {
            b.ins().ireduce(lane, self.len)
        };
        let limit = b.ins().splat(ty, len);
        let bad = if self.mask.is_none() && lane != I64 {
            // Only the largest value in each column of 128-bit chunks matters.
            let max = indices
                .iter()
                .copied()
                .reduce(|a, z| b.ins().umax(a, z))
                .unwrap();
            b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, max, limit)
        } else {
            let mut bad = constant(b, ty, &[0; CHUNK_BYTES]);
            for (i, &chunk) in indices.iter().enumerate() {
                let outside = b
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, chunk, limit);
                let enabled = match self.mask {
                    Some(mask) => b.ins().band(outside, mask[i]),
                    None => outside,
                };
                bad = b.ins().bor(bad, enabled);
            }
            bad
        };
        let mut invalid = b.ins().vany_true(bad);
        if self.element == Element::U16 {
            // Every u16 offset fits in a slice with at least 65536 elements.
            let narrow =
                b.ins()
                    .icmp_imm(IntCC::UnsignedLessThanOrEqual, self.len, u16::MAX as i64);
            invalid = b.ins().band(invalid, narrow);
        }
        invalid
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
            if let Address::Contiguous(base) = self.address {
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
            let result = self.ordered_part(b, part, initial, false);
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
    fn ordered(&self, b: &mut FunctionBuilder, check_lanes: bool) -> Vec<Value> {
        (0..splice_core::simd::vector_bytes() / CHUNK_BYTES)
            .filter_map(|part| {
                let value = self.initial(b, part);
                self.ordered_part(b, part, value, check_lanes)
            })
            .collect()
    }
    fn ordered_part(
        &self,
        b: &mut FunctionBuilder,
        part: usize,
        mut value: Option<Value>,
        check_lanes: bool,
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
                Address::Contiguous(base) => {
                    let i = part * usize::from(chunk_lanes) + usize::from(lane);
                    b.ins().iadd_imm(base, i as i64)
                }
                Address::Indexed(parts) => {
                    let i = b.ins().extractlane(parts[part], lane);
                    if self.element.bytes() == 8 {
                        i
                    } else {
                        b.ins().uextend(I64, i)
                    }
                }
            };
            if check_lanes {
                let invalid = b
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, index, self.len);
                let fault = trap(b, Trap::Bounds);
                return_if(b, invalid, fault);
            }
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
