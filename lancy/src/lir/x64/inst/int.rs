//! Scalar integers in general registers: operand sizes, condition codes and
//! the sequences of integer operations that have no single x86 instruction.
//! A narrow value keeps only its low `bytes` defined (see [`super::Inst`]),
//! so a sequence that reads more first extends its register in place.
use crate::hir::{Binary, Type, Unary};
use crate::lir::x64::emit::{R8, R16, R32, R64};
use iced_x86::{IcedError, code_asm::*};

type Result = std::result::Result<(), IcedError>;

/// A scalar integer type in a general register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lir::x64) struct Int {
    pub bytes: u8,
    pub signed: bool,
}

impl Int {
    pub(crate) const I32: Self = Self {
        bytes: 4,
        signed: true,
    };

    /// The integer view of a scalar type that lives in a general register.
    /// Pointers are unsigned words; 16-bit floats and 128-bit integers have none.
    pub(crate) fn of(ty: Type) -> Option<Self> {
        if ty == Type::Ptr {
            return Some(Self {
                bytes: 8,
                signed: false,
            });
        }
        let lane = ty.lane().filter(|_| !matches!(ty, Type::Vector(_)))?;
        (!lane.float() && lane.bytes() <= 8).then(|| Self {
            bytes: lane.bytes() as u8,
            signed: lane.signed(),
        })
    }

    pub(crate) fn bits(self) -> i32 {
        i32::from(self.bytes) * 8
    }

    /// Operations on narrow values use 32-bit registers.
    pub(crate) fn wide(self) -> bool {
        self.bytes == 8
    }

    /// The least and greatest values, sign-extended into a word.
    pub(crate) fn range(self) -> (i64, i64) {
        let bits = self.bits() as u32;
        if self.signed {
            (i64::MIN >> (64 - bits), i64::MAX >> (64 - bits))
        } else {
            (0, (u64::MAX >> (64 - bits)) as i64)
        }
    }
}

/// An x86 condition code over a preceding integer comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lir::x64) enum Cond {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Below,
    BelowEq,
    Above,
    AboveEq,
}

impl Cond {
    pub(crate) fn from_binary(op: Binary, signed: bool) -> Option<Self> {
        Some(match (op, signed) {
            (Binary::Eq, _) => Self::Eq,
            (Binary::Ne, _) => Self::Ne,
            (Binary::Lt, true) => Self::Lt,
            (Binary::Le, true) => Self::Le,
            (Binary::Gt, true) => Self::Gt,
            (Binary::Ge, true) => Self::Ge,
            (Binary::Lt, false) => Self::Below,
            (Binary::Le, false) => Self::BelowEq,
            (Binary::Gt, false) => Self::Above,
            (Binary::Ge, false) => Self::AboveEq,
            _ => return None,
        })
    }

    pub(crate) fn inverse(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Ge => Self::Lt,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
            Self::Below => Self::AboveEq,
            Self::AboveEq => Self::Below,
            Self::Above => Self::BelowEq,
            Self::BelowEq => Self::Above,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Below => "ult",
            Self::BelowEq => "ule",
            Self::Above => "ugt",
            Self::AboveEq => "uge",
        }
    }

    pub(crate) fn set(self, a: &mut CodeAssembler, dst: AsmRegister8) -> Result {
        match self {
            Self::Eq => a.sete(dst),
            Self::Ne => a.setne(dst),
            Self::Lt => a.setl(dst),
            Self::Le => a.setle(dst),
            Self::Gt => a.setg(dst),
            Self::Ge => a.setge(dst),
            Self::Below => a.setb(dst),
            Self::BelowEq => a.setbe(dst),
            Self::Above => a.seta(dst),
            Self::AboveEq => a.setae(dst),
        }
    }

    pub(crate) fn jump(self, a: &mut CodeAssembler, label: CodeLabel) -> Result {
        match self {
            Self::Eq => a.je(label),
            Self::Ne => a.jne(label),
            Self::Lt => a.jl(label),
            Self::Le => a.jle(label),
            Self::Gt => a.jg(label),
            Self::Ge => a.jge(label),
            Self::Below => a.jb(label),
            Self::BelowEq => a.jbe(label),
            Self::Above => a.ja(label),
            Self::AboveEq => a.jae(label),
        }
    }

    /// Move `src` into `dst` when the condition holds, at `wide` or 32 bits.
    pub(crate) fn select(self, a: &mut CodeAssembler, wide: bool, dst: usize, src: Src) -> Result {
        macro_rules! cmov {
            ($op:ident) => {
                match (wide, src) {
                    (true, Src::Reg(s)) => a.$op(R64[dst], R64[s]),
                    (true, Src::Mem(m)) => a.$op(R64[dst], qword_ptr(m)),
                    (false, Src::Reg(s)) => a.$op(R32[dst], R32[s]),
                    (false, Src::Mem(m)) => a.$op(R32[dst], dword_ptr(m)),
                    (_, Src::Imm(_)) => unreachable!("cmov reads a register or memory"),
                }
            };
        }
        match self {
            Self::Eq => cmov!(cmove),
            Self::Ne => cmov!(cmovne),
            Self::Lt => cmov!(cmovl),
            Self::Le => cmov!(cmovle),
            Self::Gt => cmov!(cmovg),
            Self::Ge => cmov!(cmovge),
            Self::Below => cmov!(cmovb),
            Self::BelowEq => cmov!(cmovbe),
            Self::Above => cmov!(cmova),
            Self::AboveEq => cmov!(cmovae),
        }
    }
}

/// The allocated second operand of an integer instruction.
#[derive(Clone, Copy)]
pub(in crate::lir::x64) enum Src {
    Reg(usize),
    Mem(AsmMemoryOperand),
    Imm(i32),
}

/// `$op dst, src` at operand size `$bytes`, for instructions with register,
/// memory and immediate forms.
macro_rules! sized {
    ($a:expr, $op:ident, $bytes:expr, $dst:expr, $src:expr) => {
        match ($bytes, $src) {
            (8, Src::Reg(s)) => $a.$op(R64[$dst], R64[s]),
            (8, Src::Mem(m)) => $a.$op(R64[$dst], qword_ptr(m)),
            (8, Src::Imm(n)) => $a.$op(R64[$dst], n),
            (4, Src::Reg(s)) => $a.$op(R32[$dst], R32[s]),
            (4, Src::Mem(m)) => $a.$op(R32[$dst], dword_ptr(m)),
            (4, Src::Imm(n)) => $a.$op(R32[$dst], n),
            (2, Src::Reg(s)) => $a.$op(R16[$dst], R16[s]),
            (2, Src::Mem(m)) => $a.$op(R16[$dst], word_ptr(m)),
            (2, Src::Imm(n)) => $a.$op(R16[$dst], n),
            (_, Src::Reg(s)) => $a.$op(R8[$dst], R8[s]),
            (_, Src::Mem(m)) => $a.$op(R8[$dst], byte_ptr(m)),
            (_, Src::Imm(n)) => $a.$op(R8[$dst], n),
        }
    };
}

/// Two-address integer arithmetic; `dst` holds the left operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lir::x64) enum Alu {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Min,
    Max,
}

impl Alu {
    pub(crate) fn from_binary(op: Binary) -> Option<Self> {
        Some(match op {
            Binary::Add => Self::Add,
            Binary::Sub => Self::Sub,
            Binary::Mul => Self::Mul,
            Binary::And => Self::And,
            Binary::Or => Self::Or,
            Binary::Xor => Self::Xor,
            Binary::Min => Self::Min,
            Binary::Max => Self::Max,
            _ => return None,
        })
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Min => "min",
            Self::Max => "max",
        }
    }

    /// Whether the operation reads its right operand only from a register.
    pub(crate) fn register_only(self) -> bool {
        matches!(self, Self::Min | Self::Max)
    }

    pub(crate) fn emit(self, a: &mut CodeAssembler, int: Int, dst: usize, src: Src) -> Result {
        // Carries out of a narrow value land in its undefined upper bits.
        let size = if int.wide() { 8 } else { 4 };
        match self {
            Self::Add => sized!(a, add, size, dst, src),
            Self::Sub => sized!(a, sub, size, dst, src),
            Self::And => sized!(a, and, size, dst, src),
            Self::Or => sized!(a, or, size, dst, src),
            Self::Xor => sized!(a, xor, size, dst, src),
            Self::Mul => match (int.wide(), src) {
                (true, Src::Reg(s)) => a.imul_2(R64[dst], R64[s]),
                (true, Src::Mem(m)) => a.imul_2(R64[dst], qword_ptr(m)),
                (true, Src::Imm(n)) => a.imul_3(R64[dst], R64[dst], n),
                (false, Src::Reg(s)) => a.imul_2(R32[dst], R32[s]),
                (false, Src::Mem(m)) => a.imul_2(R32[dst], dword_ptr(m)),
                (false, Src::Imm(n)) => a.imul_3(R32[dst], R32[dst], n),
            },
            Self::Min | Self::Max => {
                sized!(a, cmp, int.bytes, dst, src)?;
                let worse = match (self, int.signed) {
                    (Self::Min, true) => Cond::Gt,
                    (Self::Min, false) => Cond::Above,
                    (_, true) => Cond::Lt,
                    (_, false) => Cond::Below,
                };
                worse.select(a, int.wide(), dst, src)
            }
        }
    }
}

/// Compare `lhs` with `src` at the integer's size.
pub(in crate::lir::x64) fn compare(
    a: &mut CodeAssembler,
    bytes: u8,
    lhs: usize,
    src: Src,
) -> Result {
    sized!(a, cmp, bytes, lhs, src)
}

/// Sign- or zero-extend the low `int.bytes` of `src` into all of `dst`.
pub(in crate::lir::x64) fn extend(
    a: &mut CodeAssembler,
    int: Int,
    dst: usize,
    src: usize,
) -> Result {
    match (int.bytes, int.signed) {
        (1, true) => a.movsx(R64[dst], R8[src]),
        (2, true) => a.movsx(R64[dst], R16[src]),
        (4, true) => a.movsxd(R64[dst], R32[src]),
        (1, false) => a.movzx(R32[dst], R8[src]),
        (2, false) => a.movzx(R32[dst], R16[src]),
        (4, false) => a.mov(R32[dst], R32[src]),
        _ if dst != src => a.mov(R64[dst], R64[src]),
        _ => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lir::x64) enum Shift {
    Left,
    Right,
    RotateLeft,
    RotateRight,
}

impl Shift {
    pub(crate) fn from_binary(op: Binary) -> Option<Self> {
        Some(match op {
            Binary::Shl => Self::Left,
            Binary::Shr => Self::Right,
            Binary::RotateLeft => Self::RotateLeft,
            Binary::RotateRight => Self::RotateRight,
            _ => return None,
        })
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Left => "shl",
            Self::Right => "shr",
            Self::RotateLeft => "rotl",
            Self::RotateRight => "rotr",
        }
    }

    /// Shift `dst` by CL, or by an immediate already reduced modulo the width.
    /// x86 reduces a register count modulo 32 or 64 and rotates modulo the
    /// operand size, which is the operation's definition except for shifts
    /// of 8- and 16-bit values: those reduce CL first and restore it.
    pub(crate) fn emit(
        self,
        a: &mut CodeAssembler,
        int: Int,
        dst: usize,
        count: Option<i32>,
    ) -> Result {
        let narrow_shift =
            int.bytes < 4 && matches!(self, Self::Left | Self::Right) && count.is_none();
        // R10 keeps the count, or computes the result when the value is the
        // count in RCX; either way RCX gets R10 back.
        let dst = if narrow_shift {
            a.mov(r10, rcx)?;
            a.and(ecx, int.bits() - 1)?;
            if dst == 1 { 10 } else { dst }
        } else {
            dst
        };
        macro_rules! by {
            ($op:ident) => {
                match (int.bytes, count) {
                    (8, None) => a.$op(R64[dst], cl),
                    (8, Some(n)) => a.$op(R64[dst], n),
                    (4, None) => a.$op(R32[dst], cl),
                    (4, Some(n)) => a.$op(R32[dst], n),
                    (2, None) => a.$op(R16[dst], cl),
                    (2, Some(n)) => a.$op(R16[dst], n),
                    (_, None) => a.$op(R8[dst], cl),
                    (_, Some(n)) => a.$op(R8[dst], n),
                }
            };
        }
        match self {
            Self::Left => by!(shl)?,
            Self::Right if int.signed => by!(sar)?,
            Self::Right => by!(shr)?,
            Self::RotateLeft => by!(rol)?,
            Self::RotateRight => by!(ror)?,
        }
        if narrow_shift {
            a.mov(rcx, r10)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lir::x64) enum IntUnary {
    Neg,
    Not,
    Abs,
    Popcnt,
    LeadingZeros,
    TrailingZeros,
}

impl IntUnary {
    pub(crate) fn from_unary(op: Unary) -> Option<Self> {
        Some(match op {
            Unary::Neg => Self::Neg,
            Unary::Not => Self::Not,
            Unary::Abs => Self::Abs,
            Unary::Popcnt => Self::Popcnt,
            Unary::LeadingZeros => Self::LeadingZeros,
            Unary::TrailingZeros => Self::TrailingZeros,
            _ => return None,
        })
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Not => "not",
            Self::Abs => "abs",
            Self::Popcnt => "popcnt",
            Self::LeadingZeros => "clz",
            Self::TrailingZeros => "ctz",
        }
    }

    /// Whether the sequence needs a scratch register besides R10.
    pub(crate) fn needs_temp(self, isa: splice_target::X86Features) -> bool {
        self == Self::Popcnt && !isa.popcnt
    }

    /// Compute in place on `dst`, which holds the operand.
    pub(crate) fn emit(
        self,
        a: &mut CodeAssembler,
        isa: splice_target::X86Features,
        int: Int,
        dst: usize,
        temp: Option<usize>,
    ) -> Result {
        let (d, q) = (R32[dst], R64[dst]);
        let bits = int.bits();
        let unsigned = Int {
            signed: false,
            ..int
        };
        match self {
            Self::Neg if int.wide() => a.neg(q),
            Self::Neg => a.neg(d),
            Self::Not if int.wide() => a.not(q),
            Self::Not => a.not(d),
            Self::Abs if !int.signed => Ok(()),
            Self::Abs => {
                extend(a, int, dst, dst)?;
                // -x when it is not negative; i.e. x unless x < 0. MIN stays MIN.
                a.mov(r10, q)?;
                a.neg(r10)?;
                if int.wide() {
                    a.cmovns(q, r10)
                } else {
                    a.cmovns(d, r10d)
                }
            }
            Self::Popcnt => {
                extend(a, unsigned, dst, dst)?;
                if isa.popcnt {
                    return a.popcnt(q, q);
                }
                // SSE2 targets need not have POPCNT: count bits in parallel.
                let t = R64[temp.expect("popcnt temporary")];
                a.mov(t, q)?;
                a.shr(t, 1)?;
                a.mov(r10, 0x5555_5555_5555_5555u64)?;
                a.and(t, r10)?;
                a.sub(q, t)?;
                a.mov(r10, 0x3333_3333_3333_3333u64)?;
                a.mov(t, q)?;
                a.and(q, r10)?;
                a.shr(t, 2)?;
                a.and(t, r10)?;
                a.add(q, t)?;
                a.mov(t, q)?;
                a.shr(t, 4)?;
                a.add(q, t)?;
                a.mov(r10, 0x0f0f_0f0f_0f0f_0f0fu64)?;
                a.and(q, r10)?;
                a.mov(r10, 0x0101_0101_0101_0101u64)?;
                a.imul_2(q, r10)?;
                a.shr(q, 56)
            }
            Self::LeadingZeros => {
                extend(a, unsigned, dst, dst)?;
                if isa.lzcnt {
                    a.lzcnt(q, q)?;
                } else {
                    // 63 - index of the highest set bit; 64 for zero.
                    a.bsr(q, q)?;
                    a.mov(r10, -1i64)?;
                    a.cmovz(q, r10)?;
                    a.neg(q)?;
                    a.add(q, 63)?;
                }
                if bits < 64 {
                    a.sub(d, 64 - bits)
                } else {
                    Ok(())
                }
            }
            Self::TrailingZeros => {
                if bits < 64 {
                    // A stop bit above the value makes zero count to `bits`.
                    extend(a, unsigned, dst, dst)?;
                    a.bts(q, bits)?;
                    return a.bsf(q, q);
                }
                if isa.bmi1 {
                    a.tzcnt(q, q)
                } else {
                    a.bsf(q, q)?;
                    a.mov(r10d, 64)?;
                    a.cmovz(q, r10)
                }
            }
        }
    }
}

/// Saturating addition or subtraction of `rhs` into `dst`. R10 is scratch.
/// Only upper bits of `dst` change before `rhs` is read: the two may share
/// a register.
pub(in crate::lir::x64) fn saturating(
    a: &mut CodeAssembler,
    sub: bool,
    int: Int,
    dst: usize,
    rhs: usize,
) -> Result {
    let (d, q) = (R32[dst], R64[dst]);
    if int.bytes < 4 {
        // Exact in 32 bits, then clamped to the type's range.
        extend(a, int, dst, dst)?;
        extend(a, int, 10, rhs)?;
        if sub {
            a.sub(d, r10d)?;
        } else {
            a.add(d, r10d)?;
        }
        let (min, max) = int.range();
        a.mov(r10d, max as i32)?;
        a.cmp(d, r10d)?;
        a.cmovg(d, r10d)?;
        a.mov(r10d, min as i32)?;
        a.cmp(d, r10d)?;
        return a.cmovl(d, r10d);
    }
    let wide = int.wide();
    if int.signed {
        // Overflow saturates toward the sign of the left operand:
        // !(sign ^ MIN) is MIN for a negative operand and MAX otherwise.
        let top = int.bits() - 1;
        if wide {
            a.mov(r10, q)?;
            a.sar(r10, top)?;
            a.btc(r10, top)?;
            a.not(r10)?;
        } else {
            a.mov(r10d, d)?;
            a.sar(r10d, top)?;
            a.btc(r10d, top)?;
            a.not(r10d)?;
        }
    } else if sub {
        a.xor(r10d, r10d)?;
    } else {
        a.mov(r10, -1i64)?;
    }
    let src = Src::Reg(rhs);
    if sub {
        sized!(a, sub, int.bytes, dst, src)?;
    } else {
        sized!(a, add, int.bytes, dst, src)?;
    }
    match (int.signed, wide) {
        (true, true) => a.cmovo(q, r10),
        (true, false) => a.cmovo(d, r10d),
        (false, true) => a.cmovb(q, r10),
        (false, false) => a.cmovb(d, r10d),
    }
}

/// Classify the 16-bit float whose bits `dst` holds into an `i32` mask:
/// `IsNan`, `IsInfinite`, `IsFinite`, `IsNormal` or `IsSubnormal`.
/// Shifting out the sign left-aligns the magnitude, so each class is one
/// unsigned comparison with a left-aligned constant. R10 is scratch.
pub(in crate::lir::x64) fn classify_half(
    a: &mut CodeAssembler,
    op: Unary,
    fraction: u32,
    dst: usize,
) -> Result {
    let q = R64[dst];
    let align = |bits: u64| bits << 49;
    // Every exponent bit set: bits 10..15 for f16, 7..15 for bf16.
    let infinity = align(0x7fff & !((1 << fraction) - 1));
    let normal = align(1 << fraction);
    a.shl(q, 49)?;
    let cc = match op {
        Unary::IsNan | Unary::IsInfinite | Unary::IsFinite => {
            a.mov(r10, infinity)?;
            a.cmp(q, r10)?;
            match op {
                Unary::IsNan => Cond::Above,
                Unary::IsInfinite => Cond::Eq,
                _ => Cond::Below,
            }
        }
        Unary::IsNormal => {
            a.mov(r10, normal)?;
            a.sub(q, r10)?;
            a.mov(r10, infinity - normal)?;
            a.cmp(q, r10)?;
            Cond::Below
        }
        Unary::IsSubnormal => {
            a.sub(q, 1)?;
            a.mov(r10, normal - 1)?;
            a.cmp(q, r10)?;
            Cond::Below
        }
        _ => unreachable!("float classification"),
    };
    cc.set(a, R8[dst])?;
    a.movzx(R32[dst], R8[dst])?;
    a.neg(R32[dst])
}
