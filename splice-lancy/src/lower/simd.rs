//! Bounds-checked slice operations over one scalable Lancy vector.
use super::{Builder, Narrow};
use lancy::Trap;
use lancy::hir::{
    Binary, Block, Lane, Op, Scale, Simd as V, Ternary, Type, Unary, Value, VectorAddress,
};
use splice_core::{
    hir::{
        Buffer, Reg, Simd, SimdOp, ValueType,
        memory::{self, Address, BeforeTrap, Masked, Slice},
    },
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
        ValueType::Scalar(ty) => super::scalar_type(ty),
        ValueType::Vector(e) => Type::Vector(lane(e)),
        ValueType::Mask(e) => Type::Vector(lane(e.index())),
    }
}

pub(super) fn stride(e: Element) -> Scale {
    match e {
        Element::U16 => Scale::Two,
        Element::I32 | Element::U32 | Element::F32 => Scale::Four,
        Element::I64 | Element::F64 => Scale::Eight,
    }
}

impl Builder<'_> {
    pub(super) fn emit_simd(
        &mut self,
        inst: Simd,
        args: &[Reg],
    ) -> splice_core::Result<Option<Value>> {
        use SimdOp::*;
        if self.current.is_none() {
            return Ok(None);
        }
        if inst.op.may_trap() {
            return memory::lower(self, inst, args);
        }
        let mut vs: Vec<_> = args.iter().map(|&r| self.read(r)).collect();
        let e = inst.element;
        if let AlignmentPrefix(_) = inst.op {
            vs[0] = self.index32(vs[0]);
        }
        let op = match inst.op {
            ElementWidth | ByteWidth => {
                let unit = if inst.op == ByteWidth {
                    Lane::U8
                } else {
                    lane(e)
                };
                let lanes = self.value(Op::Simd(V::LaneCount(unit)));
                return Ok(Some(self.extend_index(lanes)));
            }
            Iota => Op::Simd(V::Iota(lane(e))),
            SimdOp::FirstN => {
                // `FirstLanes` takes every lane for a larger nonnegative
                // `i32` count. Clamp any other usize count to the lane count
                // before narrowing it; sign extension preserves unsigned
                // order, so an `i32` form clamps as `u32`.
                let width = self.value(Op::Simd(V::LaneCount(lane(e))));
                let count = match self.exact32(vs[0]) {
                    Some((count, true)) => count,
                    Some((count, false)) => {
                        let count = self.value(Op::Bitcast(Type::U32, count));
                        let width = self.value(Op::Bitcast(Type::U32, width));
                        let count = self.value(Op::Bin(Binary::Min, count, width));
                        self.value(Op::Bitcast(Type::I32, count))
                    }
                    None => {
                        let width = self.value(Op::Cast(Type::Usize, width));
                        let count = self.value(Op::Bin(Binary::Min, vs[0], width));
                        self.value(Op::Cast(Type::I32, count))
                    }
                };
                return Ok(Some(self.first_lanes(e, count)));
            }
            SimdOp::SplatScalar => Op::Simd(V::Splat(vs[0])),
            Sqrt => Op::Unary(Unary::Sqrt, vs[0]),
            MulAdd => Op::Ternary(Ternary::MulAdd, vs[0], vs[1], vs[2]),
            ReduceSum => Op::Simd(V::Reduce(Binary::Add, vs[0])),
            Cast(to) if to == e || !to.is_float() && !e.is_float() => {
                Op::Bitcast(Type::Vector(lane(to)), vs[0])
            }
            Cast(to) => Op::Cast(Type::Vector(lane(to)), vs[0]),
            Widen { to, high } => Op::Simd(V::Widen {
                to: lane(to),
                high,
                value: vs[0],
            }),
            Narrow(to) => Op::Simd(V::Narrow(lane(to), vs[0], vs[1])),
            Splat(bits) => {
                let x = self.value(Op::literal(lane(e), bits.into()));
                Op::Simd(V::Splat(x))
            }
            Shuffle(order) => {
                use splice_core::simd::Shuffle as S;
                let ids = self.value(Op::Simd(V::Iota(Lane::I32)));
                let one = self.splat_bits(Lane::I32, 1);
                let indices = match order {
                    S::DeinterleaveEven | S::DeinterleaveOdd => {
                        let even = self.value(Op::Bin(Binary::Add, ids, ids));
                        if order == S::DeinterleaveEven {
                            even
                        } else {
                            self.value(Op::Bin(Binary::Add, even, one))
                        }
                    }
                    S::InterleaveLow | S::InterleaveHigh => {
                        let pair = self.value(Op::Bin(Binary::Shr, ids, one));
                        let odd = self.value(Op::Bin(Binary::And, ids, one));
                        let width = self.splat_bits(Lane::I32, e.lanes(self.vector_bytes) as u64);
                        let base = self.value(Op::Bin(Binary::Mul, odd, width));
                        let idx = self.value(Op::Bin(Binary::Add, pair, base));
                        if order == S::InterleaveLow {
                            idx
                        } else {
                            let half =
                                self.splat_bits(Lane::I32, e.lanes(self.vector_bytes) as u64 / 2);
                            self.value(Op::Bin(Binary::Add, idx, half))
                        }
                    }
                };
                Op::Simd(V::Shuffle(vs[0], vs[1], indices))
            }
            Shr(n) | RotateLeft(n) => {
                let shift = self.splat_bits(lane(e), (n & 31).into());
                let op = if matches!(inst.op, Shr(_)) {
                    Binary::Shr
                } else {
                    Binary::RotateLeft
                };
                Op::Bin(op, vs[0], shift)
            }
            Count => return Ok(Some(self.mask_count(vs[0]))),
            Compress => Op::Simd(V::Compress(vs[0], vs[1])),
            Expand => {
                let expanded = self.value(Op::Simd(V::Expand(vs[0], vs[1])));
                Op::Ternary(Ternary::Select, vs[1], expanded, vs[2])
            }
            Swizzle => {
                let zero = self.splat_bits(Lane::I32, 0);
                Op::Simd(V::Shuffle(vs[0], zero, vs[1]))
            }
            AlignmentPrefix(b) => {
                let (base, _) = self.slice(b);
                let ptr = self.value(Op::PtrOffset(base, vs[0], Scale::Four));
                let addr = self.value(Op::Bitcast(Type::U64, ptr));
                let low = self.value(Op::Cast(Type::I32, addr));
                let neg = self.value(Op::Unary(Unary::Neg, low));
                let mask = self.value(Op::Const((self.vector_bytes - 1) as i32));
                let bytes = self.value(Op::Bin(Binary::And, neg, mask));
                let two = self.value(Op::Const(2));
                let prefix = self.value(Op::Bin(Binary::Shr, bytes, two));
                return Ok(Some(self.extend_index(prefix)));
            }
            ExtractWrapping | ReplaceWrapping => {
                let mask = self.value(Op::Const(e.lanes(self.vector_bytes) as i32 - 1));
                let i = self.value(Op::Bin(Binary::And, vs[1], mask));
                if inst.op == ExtractWrapping {
                    Op::Simd(V::ExtractLane(vs[0], i))
                } else {
                    Op::Simd(V::InsertLane(vs[0], i, vs[2]))
                }
            }
            Last => {
                let last = self.value(Op::Const(e.lanes(self.vector_bytes) as i32 - 1));
                Op::Simd(V::ExtractLane(vs[0], last))
            }
            PrefixSum => {
                let indices = self.value(Op::Simd(V::Iota(Lane::I32)));
                let mut sums = vs[0];
                let mut shift = 1;
                while shift < e.lanes(self.vector_bytes) {
                    let offset = self.splat_bits(Lane::I32, shift as u64);
                    let previous = self.value(Op::Bin(Binary::Sub, indices, offset));
                    // Negative indices select zero in a two-table Lancy shuffle.
                    let low = self.value(Op::Simd(V::Shuffle(sums, sums, previous)));
                    sums = self.value(Op::Bin(Binary::Add, sums, low));
                    shift *= 2;
                }
                return Ok(Some(sums));
            }
            Any => Op::Simd(V::AnyTrue(vs[0])),
            All => Op::Simd(V::AllTrue(vs[0])),
            Not | MaskNot => Op::Unary(Unary::Not, vs[0]),
            AddMasked => {
                let sum = self.value(Op::Bin(Binary::Add, vs[0], vs[2]));
                Op::Ternary(Ternary::Select, vs[1], sum, vs[3])
            }
            Select => Op::Ternary(Ternary::Select, vs[0], vs[1], vs[2]),
            Load(_) | Store(_) | LoadSelect(_) | StoreSelect(_) | Gather(_) | GatherSelect(_)
            | Scatter(_) | ScatterSelect(_) | CompressStore(_) => {
                unreachable!("slice accesses are lowered by splice_core::hir::memory")
            }
            op => Op::Bin(
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
        Ok(Some(self.value(op)))
    }
    pub(super) fn bind_simd(&mut self, dst: Option<Reg>, value: Option<Value>) {
        if let Some(value) = value {
            self.blocks[self.current.unwrap()]
                .defs
                .insert(dst.expect("SIMD destination"), value);
        }
    }

    /// The zero of `ty` in the open block.
    pub(super) fn zero(&mut self, ty: ValueType) -> Value {
        self.zero_in(self.current.expect("open block"), ty)
    }
    /// The zero of `ty` at the end of `block`.
    pub(super) fn zero_in(&mut self, block: Block, ty: ValueType) -> Value {
        let mut append = |op| self.graph.append(block, op).expect("a value");
        match native_type(ty) {
            Type::Vector(lane) => {
                let zero = append(Op::literal(lane, 0));
                append(Op::Simd(V::Splat(zero)))
            }
            ty => append(Op::literal(ty.lane().expect("a scalar lane"), 0)),
        }
    }
    /// A nonnegative `i32` count as an `Index`.
    pub(super) fn extend_index(&mut self, count: Value) -> Value {
        let index = self.value(Op::Cast(Type::Usize, count));
        self.indices.insert(
            index,
            Narrow::Extends {
                value: count,
                nonnegative: true,
            },
        );
        index
    }
    fn splat_bits(&mut self, ty: Lane, bits: u64) -> Value {
        let x = self.value(Op::literal(ty, bits.into()));
        self.value(Op::Simd(V::Splat(x)))
    }
    fn splat_index(&mut self, e: Element, x: Value) -> Value {
        let ty = lane(e.index()).scalar();
        let x = if ty == Type::I32 {
            x
        } else {
            self.value(Op::Cast(ty, x))
        };
        self.value(Op::Simd(V::Splat(x)))
    }
    fn mask_count(&mut self, mask: Value) -> Value {
        self.value(Op::Simd(V::MaskCount(mask)))
    }
    /// The predicate of the first `count` lanes, an `i32`.
    fn first_lanes(&mut self, e: Element, count: Value) -> Value {
        self.value(Op::Simd(V::FirstLanes(lane(e.index()), count)))
    }
}

impl memory::Primitives for Builder<'_> {
    type Scalar = Value;
    type Vector = Value;
    type Error = splice_core::BuildError;

    fn read_offset(&mut self, r: Reg) -> splice_core::Result<Value> {
        // A memory offset, in bounds only below a slice length.
        let offset = self.read(r);
        Ok(self.index32(offset))
    }
    fn read_vector(&mut self, r: Reg) -> splice_core::Result<Value> {
        Ok(self.read(r))
    }
    fn require_full(
        &mut self,
        s: Slice,
        offset: Value,
        before_trap: BeforeTrap<'_, Self>,
    ) -> splice_core::Result<()> {
        // `BoundsCheck` is the form Lancy's bounds analysis removes. An
        // offset outside the slice leaves no lane to write before the trap.
        let (_, len) = self.slice(s.buffer);
        self.effect(Op::BoundsCheck(offset, len));
        let count = self.value(Op::Simd(V::LaneCount(lane(s.element))));
        let remaining = self.value(Op::Bin(Binary::Sub, len, offset));
        if before_trap.is_none() {
            let one = self.value(Op::Const(1));
            let last = self.value(Op::Bin(Binary::Sub, count, one));
            self.effect(Op::BoundsCheck(last, remaining));
            return Ok(());
        }
        let short = self.value(Op::Bin(Binary::Lt, remaining, count));
        self.trap_if(short, before_trap)
    }
    fn room(&mut self, s: Slice, offset: Value) -> splice_core::Result<Value> {
        let (_, len) = self.slice(s.buffer);
        let zero = self.value(Op::Const(0));
        let nonnegative = self.value(Op::Bin(Binary::Ge, offset, zero));
        let left = self.value(Op::Bin(Binary::Sub, len, offset));
        let left = self.value(Op::Bin(Binary::And, left, nonnegative));
        let left = self.value(Op::Bin(Binary::Max, left, zero));
        let width = self.value(Op::Simd(V::LaneCount(lane(s.element))));
        Ok(self.value(Op::Bin(Binary::Min, left, width)))
    }
    fn first_lanes(&mut self, e: Element, count: Value) -> splice_core::Result<Value> {
        Ok(self.first_lanes(e, count))
    }
    fn outside(&mut self, s: Slice, indices: Value) -> splice_core::Result<Value> {
        let e = s.element;
        let (_, len) = self.slice(s.buffer);
        let idx = lane(e.index());
        let limit = self.splat_index(e, len);
        if idx == Lane::U16 {
            return Ok(self.value(Op::Bin(Binary::Ge, indices, limit)));
        }
        // Buffer validation caps every length at i32::MAX. Reinterpreting
        // negative indices as unsigned puts them above every valid length.
        let unsigned = Type::Vector(if idx.bytes() == 4 {
            Lane::U32
        } else {
            Lane::U64
        });
        let offsets = self.value(Op::Bitcast(unsigned, indices));
        let limit = self.value(Op::Bitcast(unsigned, limit));
        let outside = self.value(Op::Bin(Binary::Ge, offsets, limit));
        Ok(self.value(Op::Bitcast(Type::Vector(idx), outside)))
    }
    fn length_at_most(&mut self, buffer: Buffer, n: u64) -> splice_core::Result<Value> {
        let (_, len) = self.slice(buffer);
        let n = self.value(Op::Const(n.min(i32::MAX as u64) as i32));
        Ok(self.value(Op::Bin(Binary::Le, len, n)))
    }
    fn and(&mut self, a: Value, b: Value) -> splice_core::Result<Value> {
        Ok(self.value(Op::Bin(Binary::And, a, b)))
    }
    fn mask_and(&mut self, a: Value, b: Value) -> splice_core::Result<Value> {
        Ok(self.value(Op::Bin(Binary::And, a, b)))
    }
    fn mask_xor(&mut self, a: Value, b: Value) -> splice_core::Result<Value> {
        Ok(self.value(Op::Bin(Binary::Xor, a, b)))
    }
    fn any(&mut self, mask: Value) -> splice_core::Result<Value> {
        Ok(self.value(Op::Simd(V::AnyTrue(mask))))
    }
    fn before_first(&mut self, e: Element, mask: Value) -> splice_core::Result<Value> {
        // The first lane of the mask, or the lane count if it has none.
        let idx = lane(e.index());
        let iota = self.value(Op::Simd(V::Iota(idx)));
        let count = self.value(Op::Simd(V::LaneCount(idx)));
        let count = self.splat_index(e, count);
        let lanes = self.value(Op::Ternary(Ternary::Select, mask, iota, count));
        let first = self.value(Op::Simd(V::Reduce(Binary::Min, lanes)));
        let first = self.value(Op::Simd(V::Splat(first)));
        Ok(self.value(Op::Bin(Binary::Lt, iota, first)))
    }
    fn exceeds(&mut self, mask: Value, count: Value) -> splice_core::Result<Value> {
        let lanes = self.mask_count(mask);
        Ok(self.value(Op::Bin(Binary::Gt, lanes, count)))
    }
    fn trap_if(
        &mut self,
        condition: Value,
        before_trap: BeforeTrap<'_, Self>,
    ) -> splice_core::Result<()> {
        let Some(before_trap) = before_trap else {
            self.effect(Op::TrapIf(condition, Trap::Bounds));
            return Ok(());
        };
        let (fail, next) = (self.block(), self.block());
        self.end(super::Exit::Branch(condition, fail, next));
        self.current = Some(fail);
        before_trap(self)?;
        self.effect(Op::Trap(Trap::Bounds));
        self.begin(next);
        Ok(())
    }
    fn load(
        &mut self,
        s: Slice,
        at: Address<Value, Value>,
        masked: Option<Masked<Value>>,
    ) -> splice_core::Result<Value> {
        let e = s.element;
        let address = self.vector_address(s, at);
        Ok(match (address.indices, masked) {
            (None, None) => self.value(Op::Load(Type::Vector(lane(e)), address.base)),
            (_, masked) => {
                let (mask, passthrough) = match masked {
                    Some(Masked { mask, passthrough }) => (mask, passthrough),
                    None => (
                        self.splat_bits(lane(e.index()), u64::MAX),
                        self.splat_bits(lane(e), 0),
                    ),
                };
                self.value(Op::Simd(V::MaskedLoad {
                    address,
                    mask,
                    passthrough,
                }))
            }
        })
    }
    fn store(
        &mut self,
        s: Slice,
        at: Address<Value, Value>,
        mask: Option<Value>,
        value: Value,
    ) -> splice_core::Result<()> {
        let address = self.vector_address(s, at);
        match (address.indices, mask) {
            (None, None) => self.effect(Op::Store(address.base, value)),
            (_, mask) => {
                let mask =
                    mask.unwrap_or_else(|| self.splat_bits(lane(s.element.index()), u64::MAX));
                self.effect(Op::Simd(V::MaskedStore {
                    address,
                    mask,
                    value,
                }));
            }
        }
        Ok(())
    }
    fn compress_store(
        &mut self,
        s: Slice,
        offset: Value,
        mask: Value,
        value: Value,
        room: Value,
    ) -> splice_core::Result<()> {
        let (base, _) = self.slice(s.buffer);
        let base = self.value(Op::PtrOffset(base, offset, stride(s.element)));
        self.effect(Op::Simd(V::CompressStore {
            base,
            mask,
            value,
            limit: room,
        }));
        Ok(())
    }
}

impl Builder<'_> {
    /// The address of the lanes at `at` in the slice.
    fn vector_address(&mut self, s: Slice, at: Address<Value, Value>) -> VectorAddress {
        let e = s.element;
        let (base, _) = self.slice(s.buffer);
        match at {
            Address::Offset(offset) => VectorAddress {
                base: self.value(Op::PtrOffset(base, offset, stride(e))),
                indices: None,
            },
            // Lancy's gather takes signed indices: rebase unsigned u16
            // offsets around 32768.
            Address::Indices(indices) if e.index() == Element::U16 => {
                let half = self.value(Op::Const(1 << 15));
                let base = self.value(Op::PtrOffset(base, half, Scale::Two));
                let sign = self.splat_bits(Lane::U16, 1 << 15);
                let biased = self.value(Op::Bin(Binary::Xor, indices, sign));
                VectorAddress {
                    base,
                    indices: Some(self.value(Op::Bitcast(Type::Vector(Lane::I16), biased))),
                }
            }
            Address::Indices(indices) => VectorAddress {
                base,
                indices: Some(if e.index() == Element::U32 {
                    self.value(Op::Bitcast(Type::Vector(Lane::I32), indices))
                } else {
                    indices
                }),
            },
        }
    }
}
