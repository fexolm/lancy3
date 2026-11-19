//! One Splice vector occupies consecutive 128-bit CLIF values. This keeps
//! the lane count identical across backends even when the selected Cranelift
//! ISA legalizes a smaller vector than the host width.
use super::return_if;
use crate::BOUNDS;
use cranelift_codegen::ir::{
    InstBuilder, MemFlags, Type, Value,
    condcodes::{FloatCC, IntCC},
    types::{F32X4, F64X2, I16X8, I32, I32X4, I64, I64X2},
};
use cranelift_frontend::{FunctionBuilder, Variable};
use splice::{
    Function,
    hir::{Reg, Simd, SimdOp, ValueType},
    simd::Element,
};

const CHUNK_BYTES: usize = 16;

fn chunk_type(element: Element) -> Type {
    match element {
        Element::U16 => I16X8,
        Element::I32 => I32X4,
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
                ValueType::Vector(e) => (chunk_type(e), splice::simd::vector_bytes() / CHUNK_BYTES),
                ValueType::Mask(e) => (
                    chunk_type(e.index()),
                    splice::simd::vector_bytes() / CHUNK_BYTES,
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
            (0..splice::simd::vector_bytes() / CHUNK_BYTES)
                .map(|part| {
                    let indices = lane_constant(b, element.index(), part, |i| i as u64);
                    b.ins().icmp(IntCC::SignedLessThan, indices, limit)
                })
                .collect()
        }
        SplatScalar => {
            let chunk = b.ins().splat(chunk_type(element), inputs[0][0]);
            vec![chunk; splice::simd::vector_bytes() / CHUNK_BYTES]
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
        _ => (0..splice::simd::vector_bytes() / CHUNK_BYTES)
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
        Add if float => b.ins().fadd(arg(0), arg(1)),
        Sub if float => b.ins().fsub(arg(0), arg(1)),
        Mul if float => b.ins().fmul(arg(0), arg(1)),
        Div => b.ins().fdiv(arg(0), arg(1)),
        Add => b.ins().iadd(arg(0), arg(1)),
        Sub => b.ins().isub(arg(0), arg(1)),
        Mul => b.ins().imul(arg(0), arg(1)),
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
            let unsigned = inst.element == Element::U16;
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
        if self.mask.is_some() {
            return self.masked(b);
        }
        let Address::Contiguous(start) = self.address else {
            return self.ordered(b);
        };
        // Widen before addition so even an invalid i32 start cannot wrap into the slice.
        let end = b.ins().iadd_imm(start, self.element.lanes() as i64);
        let invalid = b.ins().icmp(IntCC::UnsignedGreaterThan, end, self.len);
        let fault = b.ins().iconst(I64, BOUNDS);
        if self.values.is_some() {
            // An invalid store can still write a valid prefix. Cold execution
            // checks and writes each lane in source order, up to the first fault.
            let partial = b.create_block();
            let full = b.create_block();
            b.set_cold_block(partial);
            b.ins().brif(invalid, partial, &[], full, &[]);
            b.switch_to_block(partial);
            self.ordered(b);
            b.ins().return_(&[fault]); // the full interval was out of bounds
            b.switch_to_block(full);
        } else {
            return_if(b, invalid, fault);
        }
        let base = self.pointer(b, start);
        let mut out = Vec::new();
        for part in 0..splice::simd::vector_bytes() / CHUNK_BYTES {
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
                let end = b.ins().iadd_imm(start, i64::from(lanes));
                let fits = b.ins().icmp(IntCC::UnsignedLessThanOrEqual, end, self.len);
                let all = b.ins().vall_true(m);
                let full = b.ins().band(all, fits);
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
        (0..splice::simd::vector_bytes() / CHUNK_BYTES)
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
            let invalid = b
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, index, self.len);
            let fault = b.ins().iconst(I64, BOUNDS);
            return_if(b, invalid, fault);
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
