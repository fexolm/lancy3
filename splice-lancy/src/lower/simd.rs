//! Bounds-checked slice operations over one scalable Lancy vector.
use super::{Builder, KernelBuilder};
use lancy::hir::{Binary, Lane, Op, Scale, Simd as V, Ternary, Type, Unary, Value, VectorAddress};
use splice::{
    hir::{Reg, Simd, SimdOp, ValueType},
    simd::Element,
};

fn lane(e: Element) -> Lane {
    match e {
        Element::U16 => Lane::U16,
        Element::U32 => Lane::U32,
        Element::I32 => Lane::I32,
        Element::I64 => Lane::I64,
        Element::F32 => Lane::F32,
        Element::F64 => Lane::F64,
    }
}

pub(super) fn native_type(ty: ValueType) -> Type {
    match ty {
        ValueType::Scalar => Type::I32,
        ValueType::Vector(e) => Type::Vector(lane(e)),
        ValueType::Mask(e) => Type::Vector(lane(e.index())),
    }
}

fn stride(e: Element) -> Scale {
    match e {
        Element::U16 => Scale::Two,
        Element::I32 | Element::U32 | Element::F32 => Scale::Four,
        Element::I64 | Element::F64 => Scale::Eight,
    }
}

impl Builder<'_> {
    pub(super) fn emit_simd(&mut self, inst: Simd, args: &[Reg]) -> splice::Result<Option<Value>> {
        use SimdOp::*;
        let types = args
            .iter()
            .map(|r| {
                self.types
                    .get(r.0 as usize)
                    .copied()
                    .ok_or_else(|| format!("invalid SIMD register {r:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        inst.validate(&types, self.kinds)?;
        if self.current.is_none() {
            return Ok(None);
        }
        let vs: Vec<_> = args.iter().map(|&r| self.read(r)).collect();
        let e = inst.element;
        let op = match inst.op {
            Lanes => V::LaneCount(lane(e)),
            Iota => V::Iota(lane(e)),
            SimdOp::FirstN => {
                // Splice indices occupy an unsigned 32-bit word. Clamp at this
                // width before converting the limit to a narrower lane.
                let count = self.val(Op::Bitcast(Type::U32, vs[0]));
                let width = self.vec_op(V::LaneCount(lane(e)));
                let width = self.val(Op::Bitcast(Type::U32, width));
                let count = self.val(Op::Bin(Binary::Min, count, width));
                let count = self.val(Op::Bitcast(Type::I32, count));
                return Ok(Some(self.first_lanes(e, count)));
            }
            SimdOp::SplatScalar => V::Splat(vs[0]),
            Sqrt => V::Unary(Unary::Sqrt, vs[0]),
            MulAdd => V::Ternary(Ternary::MulAdd, vs[0], vs[1], vs[2]),
            ReduceSum => V::Reduce(Binary::Add, vs[0]),
            Cast(to) if to == e || !to.is_float() && !e.is_float() => V::Bitcast(lane(to), vs[0]),
            Cast(to) => V::Cast(lane(to), vs[0]),
            Widen { to, high } => V::Widen {
                to: lane(to),
                high,
                value: vs[0],
            },
            Narrow(to) => V::Narrow(lane(to), vs[0], vs[1]),
            Splat(bits) => {
                let x = self.val(Op::Literal(lane(e), bits.into()));
                V::Splat(x)
            }
            Load(b) | Store(b) | LoadSelect(b) | StoreSelect(b) | Gather(b) | GatherSelect(b)
            | Scatter(b) | ScatterSelect(b) => {
                let write = matches!(
                    inst.op,
                    Store(_) | StoreSelect(_) | Scatter(_) | ScatterSelect(_)
                );
                let indexed = matches!(
                    inst.op,
                    Gather(_) | GatherSelect(_) | Scatter(_) | ScatterSelect(_)
                );
                let masked = matches!(
                    inst.op,
                    LoadSelect(_) | StoreSelect(_) | GatherSelect(_) | ScatterSelect(_)
                );
                let (base, len) = self.slice(b, write);
                let result = if !indexed && !masked {
                    self.contiguous(e, base, len, vs[0], vs.get(1).copied())
                } else {
                    let mask = if masked {
                        vs[1]
                    } else {
                        self.splat_bits(lane(e.index()), u64::MAX)
                    };
                    let value = if write || masked {
                        *vs.last().unwrap()
                    } else {
                        self.splat_bits(lane(e), 0)
                    };
                    if indexed {
                        self.indexed(e, base, len, vs[0], mask, Access { write, value })
                    } else {
                        self.masked_contiguous(e, base, len, vs[0], mask, Access { write, value })
                    }
                };
                return Ok(result);
            }
            Shuffle(order) => {
                use splice::simd::Shuffle as S;
                let ids = self.vec_op(V::Iota(Lane::I32));
                let one = self.splat_bits(Lane::I32, 1);
                let indices = match order {
                    S::DeinterleaveEven | S::DeinterleaveOdd => {
                        let even = self.vec_op(V::Bin(Binary::Add, ids, ids));
                        if order == S::DeinterleaveEven {
                            even
                        } else {
                            self.vec_op(V::Bin(Binary::Add, even, one))
                        }
                    }
                    S::InterleaveLow | S::InterleaveHigh => {
                        let pair = self.vec_op(V::Bin(Binary::Shr, ids, one));
                        let odd = self.vec_op(V::Bin(Binary::And, ids, one));
                        let width = self.splat_bits(Lane::I32, e.lanes() as u64);
                        let base = self.vec_op(V::Bin(Binary::Mul, odd, width));
                        let idx = self.vec_op(V::Bin(Binary::Add, pair, base));
                        if order == S::InterleaveLow {
                            idx
                        } else {
                            let half = self.splat_bits(Lane::I32, e.lanes() as u64 / 2);
                            self.vec_op(V::Bin(Binary::Add, idx, half))
                        }
                    }
                };
                V::Shuffle(vs[0], vs[1], indices)
            }
            Shr(n) | RotateLeft(n) => {
                let shift = self.splat_bits(lane(e), (n & 31).into());
                let op = if matches!(inst.op, Shr(_)) {
                    Binary::Shr
                } else {
                    Binary::RotateLeft
                };
                V::Bin(op, vs[0], shift)
            }
            Count => return Ok(Some(self.mask_count(vs[0]))),
            Compress => V::Compress(vs[0], vs[1]),
            Expand => {
                let expanded = self.vec_op(V::Expand(vs[0], vs[1]));
                V::Ternary(Ternary::Select, vs[1], expanded, vs[2])
            }
            Swizzle => {
                let zero = self.splat_bits(Lane::I32, 0);
                V::Shuffle(vs[0], zero, vs[1])
            }
            AlignmentPrefix(b) => {
                let (base, _) = self.slice(b, false);
                let ptr = self.val(Op::PtrOffset(base, vs[0], Scale::Four));
                let addr = self.val(Op::Bitcast(Type::U64, ptr));
                let low = self.val(Op::Cast(Type::I32, addr));
                let neg = self.val(Op::Unary(Unary::Neg, low));
                let mask = self.val(Op::Const((splice::simd::vector_bytes() - 1) as i32));
                let bytes = self.val(Op::Bin(Binary::And, neg, mask));
                let two = self.val(Op::Const(2));
                return Ok(Some(self.val(Op::Bin(Binary::Shr, bytes, two))));
            }
            CompressStore(b) => {
                let (base, len) = self.slice(b, true);
                let data = self.vec_op(V::Compress(vs[2], vs[1]));
                let count = self.mask_count(vs[1]);
                let selected = self.first_lanes(e, count);
                return Ok(self.masked_contiguous(
                    e,
                    base,
                    len,
                    vs[0],
                    selected,
                    Access {
                        write: true,
                        value: data,
                    },
                ));
            }
            ExtractWrapping | ReplaceWrapping => {
                let mask = self.val(Op::Const(e.lanes() as i32 - 1));
                let i = self.val(Op::Bin(Binary::And, vs[1], mask));
                if inst.op == ExtractWrapping {
                    V::ExtractLane(vs[0], i)
                } else {
                    V::InsertLane(vs[0], i, vs[2])
                }
            }
            Last => {
                let last = self.val(Op::Const(e.lanes() as i32 - 1));
                V::ExtractLane(vs[0], last)
            }
            PrefixSum => {
                let indices = self.vec_op(V::Iota(Lane::I32));
                let mut sums = vs[0];
                let mut shift = 1;
                while shift < e.lanes() {
                    let offset = self.splat_bits(Lane::I32, shift as u64);
                    let previous = self.vec_op(V::Bin(Binary::Sub, indices, offset));
                    // Negative indices select zero in a two-table Lancy shuffle.
                    let low = self.vec_op(V::Shuffle(sums, sums, previous));
                    sums = self.vec_op(V::Bin(Binary::Add, sums, low));
                    shift *= 2;
                }
                return Ok(Some(sums));
            }
            Any => V::AnyTrue(vs[0]),
            All => V::AllTrue(vs[0]),
            Not | MaskNot => V::Unary(Unary::Not, vs[0]),
            Select => V::Ternary(Ternary::Select, vs[0], vs[1], vs[2]),
            op => V::Bin(
                match op {
                    Add => Binary::Add,
                    Sub => Binary::Sub,
                    Mul => Binary::Mul,
                    Min => Binary::Min,
                    Max => Binary::Max,
                    SimdOp::Div => Binary::Div,
                    And | MaskAnd => Binary::And,
                    Or | MaskOr => Binary::Or,
                    Xor | MaskXor => Binary::Xor,
                    Eq => Binary::Eq,
                    Ne => Binary::Ne,
                    Lt => Binary::Lt,
                    Le => Binary::Le,
                    Gt => Binary::Gt,
                    Ge => Binary::Ge,
                    _ => unreachable!("non-binary SIMD operation handled above"),
                },
                vs[0],
                vs[1],
            ),
        };
        Ok(Some(self.vec_op(op)))
    }
    pub(super) fn bind_simd(&mut self, dst: Option<Reg>, value: Option<Value>) {
        if let Some(value) = value {
            self.blocks[self.current.unwrap()]
                .defs
                .insert(dst.expect("SIMD destination"), value);
        }
    }

    fn effect(&mut self, op: Op) {
        self.graph.append(self.current.unwrap(), op);
    }
    fn val(&mut self, op: Op) -> Value {
        self.graph
            .append(self.current.unwrap(), op)
            .expect("operation defines a value")
    }
    fn vec_op(&mut self, op: V) -> Value {
        self.val(Op::Simd(op))
    }
    fn splat_bits(&mut self, ty: Lane, bits: u64) -> Value {
        let x = self.val(Op::Literal(ty, bits.into()));
        self.vec_op(V::Splat(x))
    }
    fn splat_index(&mut self, e: Element, x: Value) -> Value {
        let ty = lane(e.index()).scalar();
        let x = if ty == Type::I32 {
            x
        } else {
            self.val(Op::Cast(ty, x))
        };
        self.vec_op(V::Splat(x))
    }
    fn mask_count(&mut self, mask: Value) -> Value {
        // Splice predicates are all-ones or zero in each Lancy lane.
        let flags = self.vec_op(V::Unary(Unary::Neg, mask));
        self.vec_op(V::Reduce(Binary::Add, flags))
    }
    fn first_lanes(&mut self, e: Element, count: Value) -> Value {
        let iota = self.vec_op(V::Iota(lane(e.index())));
        let limit = self.splat_index(e, count);
        self.vec_op(V::Bin(Binary::Lt, iota, limit))
    }

    /// A full load has no observable partial result. A short store must first
    /// commit the in-bounds prefix, so it has a separate cold path.
    fn contiguous(
        &mut self,
        e: Element,
        base: Value,
        len: Value,
        offset: Value,
        value: Option<Value>,
    ) -> Option<Value> {
        self.effect(Op::BoundsCheck(offset, len));
        let ptr = self.val(Op::PtrOffset(base, offset, stride(e)));
        let count = self.vec_op(V::LaneCount(lane(e)));
        let remaining = self.val(Op::Bin(Binary::Sub, len, offset));
        if let Some(value) = value {
            let full = self.val(Op::Bin(Binary::Ge, remaining, count));
            let (yes, short, join) = (self.label(), self.label(), self.label());
            let (t, f) = (self.target(yes), self.target(short));
            self.end(super::Exit::Branch(full, t, f));
            self.mark(yes);
            self.effect(Op::Simd(V::Store(ptr, value)));
            self.jump(join);
            self.mark(short);
            let prefix = self.first_lanes(e, remaining);
            Access { write: true, value }.emit(
                self,
                VectorAddress {
                    base: ptr,
                    indices: None,
                },
                prefix,
            );
            self.effect(Op::BoundsCheck(len, len));
            self.mark(join);
            None
        } else {
            let one = self.val(Op::Const(1));
            let last = self.val(Op::Bin(Binary::Sub, count, one));
            self.effect(Op::BoundsCheck(last, remaining));
            Some(self.vec_op(V::Load(lane(e), ptr)))
        }
    }

    fn masked_contiguous(
        &mut self,
        e: Element,
        base: Value,
        len: Value,
        offset: Value,
        active: Value,
        access: Access,
    ) -> Option<Value> {
        let zero = self.val(Op::Const(0));
        let nonnegative = self.val(Op::Bin(Binary::Ge, offset, zero));
        let left = self.val(Op::Bin(Binary::Sub, len, offset));
        let left = self.val(Op::Bin(Binary::And, left, nonnegative));
        let left = self.val(Op::Bin(Binary::Max, left, zero));
        let width = self.vec_op(V::LaneCount(lane(e)));
        let count = self.val(Op::Bin(Binary::Min, left, width));
        let valid = self.first_lanes(e, count);
        let enabled = self.vec_op(V::Bin(Binary::And, active, valid));
        let bad = self.vec_op(V::Bin(Binary::Xor, active, enabled));
        let ptr = self.val(Op::PtrOffset(base, offset, stride(e)));
        // All invalid contiguous lanes are a suffix. The mask suppresses every
        // out-of-bounds address, even when the slice is empty.
        let result = access.emit(
            self,
            VectorAddress {
                base: ptr,
                indices: None,
            },
            enabled,
        );
        self.check_mask(bad);
        result
    }

    fn check_mask(&mut self, bad: Value) {
        let fail = self.vec_op(V::AnyTrue(bad));
        let one = self.val(Op::Const(1));
        // A true Lancy comparison is all ones, which fails unsigned bounds.
        self.effect(Op::BoundsCheck(fail, one));
    }

    fn indexed(
        &mut self,
        e: Element,
        base: Value,
        len: Value,
        indices: Value,
        active: Value,
        access: Access,
    ) -> Option<Value> {
        let idx = lane(e.index());
        let limit = self.splat_index(e, len);
        let (invalid, address) = if idx == Lane::U16 {
            // For longer slices every u16 fits, including 65535. Rebase unsigned
            // offsets around 32768 because Lancy's gather uses signed indices.
            let max = self.val(Op::Const(i32::from(u16::MAX)));
            let short = self.val(Op::Bin(Binary::Le, len, max));
            let short = self.splat_index(e, short);
            let over = self.vec_op(V::Bin(Binary::Ge, indices, limit));
            let invalid = self.vec_op(V::Bin(Binary::And, over, short));
            let half = self.val(Op::Const(1 << 15));
            let centered = self.val(Op::PtrOffset(base, half, Scale::Two));
            let sign = self.splat_bits(idx, 1 << 15);
            let biased = self.vec_op(V::Bin(Binary::Xor, indices, sign));
            let offsets = self.vec_op(V::Bitcast(Lane::I16, biased));
            (
                invalid,
                VectorAddress {
                    base: centered,
                    indices: Some(offsets),
                },
            )
        } else {
            // Buffer validation caps every length at i32::MAX. Reinterpreting
            // negative indices as unsigned puts them above every valid length.
            let unsigned = if idx.bytes() == 4 {
                Lane::U32
            } else {
                Lane::U64
            };
            let offsets = self.vec_op(V::Bitcast(unsigned, indices));
            let limit = self.vec_op(V::Bitcast(unsigned, limit));
            let invalid = self.vec_op(V::Bin(Binary::Ge, offsets, limit));
            let invalid = self.vec_op(V::Bitcast(idx, invalid));
            let indices = if idx == Lane::U32 {
                self.vec_op(V::Bitcast(Lane::I32, indices))
            } else {
                indices
            };
            (
                invalid,
                VectorAddress {
                    base,
                    indices: Some(indices),
                },
            )
        };
        let bad = self.vec_op(V::Bin(Binary::And, invalid, active));
        if !access.write {
            self.check_mask(bad);
            return access.emit(self, address, active);
        }
        // Find the first active invalid index, or the lane count if all fit.
        // Ordered masked scatter resolves duplicate indices inside this prefix.
        let iota = self.vec_op(V::Iota(idx));
        let count = self.vec_op(V::LaneCount(idx));
        let count = self.splat_index(e, count);
        let faults = self.vec_op(V::Ternary(Ternary::Select, bad, iota, count));
        let first = self.vec_op(V::Reduce(Binary::Min, faults));
        let first = self.vec_op(V::Splat(first));
        let prefix = self.vec_op(V::Bin(Binary::Lt, iota, first));
        let enabled = self.vec_op(V::Bin(Binary::And, active, prefix));
        access.emit(self, address, enabled);
        self.check_mask(bad);
        None
    }
}

#[derive(Clone, Copy)]
struct Access {
    write: bool,
    value: Value,
}

impl Access {
    fn emit(self, b: &mut Builder<'_>, address: VectorAddress, mask: Value) -> Option<Value> {
        if self.write {
            b.effect(Op::Simd(V::MaskedStore {
                address,
                mask,
                value: self.value,
            }));
            None
        } else {
            Some(b.vec_op(V::MaskedLoad {
                address,
                mask,
                passthrough: self.value,
            }))
        }
    }
}
