//! Scalar numeric casts that involve a float, with Rust `as` semantics:
//! float-to-integer saturates and maps NaN to zero, other casts round to
//! nearest even. `f32` and `f64` use SSE2 conversions with fix-ups; 16-bit
//! floats, which SSE2 cannot convert, go through the integer significand.
use super::int::{Int, extend};
use crate::hir::Lane;
use crate::lir::x64::emit::{R16, R32, R64, bind};
use crate::lir::x64::inst::{SCRATCH, simd::XMM};
use iced_x86::{IcedError, code_asm::*};
use regalloc2::RegClass;

type Result = std::result::Result<(), IcedError>;

fn half(lane: Lane) -> bool {
    matches!(lane, Lane::F16 | Lane::BF16)
}

/// Whether the cast decodes and encodes significands in RAX, RCX, RDX, R8
/// and R9, with the result in RAX for a general register.
pub(in crate::lir::x64) fn soft(from: Lane, to: Lane) -> bool {
    half(from) || half(to)
}

/// The class of the scratch register the cast needs, if any.
pub(in crate::lir::x64) fn temp(from: Lane, to: Lane) -> Option<RegClass> {
    let wide_unsigned = |lane| matches!(lane, Lane::U64 | Lane::Usize);
    (!soft(from, to) && wide_unsigned(from) && to.float()).then_some(RegClass::Int)
}

/// Convert `src` to `dst`; for a soft cast to an integer, `dst` is RAX.
pub(in crate::lir::x64) fn emit(
    a: &mut CodeAssembler,
    from: Lane,
    to: Lane,
    dst: usize,
    src: usize,
    temp: Option<usize>,
) -> Result {
    if soft(from, to) {
        match from {
            Lane::F32 => a.movd(eax, XMM[src])?,
            Lane::F64 => a.movq(rax, XMM[src])?,
            _ if from.float() => a.movzx(eax, R16[src])?,
            _ => extend(a, int(from), 0, src)?,
        }
        if to.float() {
            to_float(a, from, to)?;
        } else {
            to_integer(a, from, to)?;
        }
        return match to {
            Lane::F32 => a.movd(XMM[dst], eax),
            Lane::F64 => a.movq(XMM[dst], rax),
            _ => Ok(()),
        };
    }
    let single = |lane| lane == Lane::F32;
    match (from.float(), to.float()) {
        (true, true) if single(from) => a.cvtss2sd(XMM[dst], XMM[src]),
        (true, true) => a.cvtsd2ss(XMM[dst], XMM[src]),
        (false, _) => from_integer(a, int(from), single(to), dst, src, temp),
        (true, false) => to_saturated(a, single(from), int(to), dst, src),
    }
}

fn int(lane: Lane) -> Int {
    Int::of(lane.scalar()).expect("integer lane")
}

fn from_integer(
    a: &mut CodeAssembler,
    int: Int,
    single: bool,
    dst: usize,
    src: usize,
    temp: Option<usize>,
) -> Result {
    let d = XMM[dst];
    let convert = |a: &mut CodeAssembler, from: AsmRegister64| {
        if single {
            a.cvtsi2ss(d, from)
        } else {
            a.cvtsi2sd(d, from)
        }
    };
    // CVTSI2S* writes only lane 0: clear the rest to break the dependency.
    a.xorps(d, d)?;
    if int.signed || !int.wide() {
        extend(a, int, 10, src)?;
        return convert(a, r10);
    }
    // An unsigned word with the top bit set halves, keeping the lost bit
    // sticky so that doubling rounds once.
    let big = a.create_label();
    let done = a.create_label();
    a.mov(r10, R64[src])?;
    a.test(r10, r10)?;
    a.js(big)?;
    convert(a, r10)?;
    a.jmp(done)?;
    bind(a, big)?;
    let t = R64[temp.expect("unsigned conversion temporary")];
    a.mov(t, r10)?;
    a.shr(t, 1)?;
    a.and(r10d, 1)?;
    a.or(t, r10)?;
    convert(a, t)?;
    if single {
        a.addss(d, d)?;
    } else {
        a.addsd(d, d)?;
    }
    bind(a, done)
}

/// Truncate toward zero, saturating at the type's range; NaN is zero.
fn to_saturated(a: &mut CodeAssembler, single: bool, int: Int, dst: usize, src: usize) -> Result {
    let (q, x, s) = (R64[dst], XMM[src], XMM[SCRATCH]);
    let truncate = |a: &mut CodeAssembler, from: AsmRegisterXmm| {
        if single {
            a.cvttss2si(q, from)
        } else {
            a.cvttsd2si(q, from)
        }
    };
    let compare = |a: &mut CodeAssembler, with: AsmRegisterXmm| {
        if single {
            a.ucomiss(x, with)
        } else {
            a.ucomisd(x, with)
        }
    };
    let done = a.create_label();
    if int.wide() && !int.signed {
        // At or above 2^63, convert 2^63 - x and flip the result's top bit.
        let big = a.create_label();
        let saturate = a.create_label();
        if single {
            a.mov(r10d, 0x5f00_0000)?;
            a.movd(s, r10d)?;
        } else {
            a.mov(r10, 0x43e0_0000_0000_0000u64)?;
            a.movq(s, r10)?;
        }
        compare(a, s)?;
        a.jae(big)?;
        // Below 2^63 or NaN: a negative or invalid result is zero.
        truncate(a, x)?;
        a.xor(r10d, r10d)?;
        a.test(q, q)?;
        a.cmovs(q, r10)?;
        a.jmp(done)?;
        bind(a, big)?;
        if single {
            a.subss(s, x)?;
        } else {
            a.subsd(s, x)?;
        }
        truncate(a, s)?;
        // From 2^64 on, -(x - 2^63) is invalid and negates to itself.
        a.neg(q)?;
        a.jo(saturate)?;
        a.btc(q, 63)?;
        a.jmp(done)?;
        bind(a, saturate)?;
        a.mov(q, -1i64)?;
        return bind(a, done);
    }
    // The 64-bit conversion is exact for every value narrower types keep.
    // Its invalid result is i64::MIN, which is also a real result.
    let clamp = a.create_label();
    let nan = a.create_label();
    truncate(a, x)?;
    a.cmp(q, 1)?;
    a.jno(clamp)?;
    compare(a, x)?;
    a.jp(nan)?;
    a.xorps(s, s)?;
    compare(a, s)?;
    a.jbe(clamp)?;
    a.mov(q, i64::MAX)?;
    a.jmp(clamp)?;
    bind(a, nan)?;
    a.xor(R32[dst], R32[dst])?;
    a.jmp(done)?;
    bind(a, clamp)?;
    if !int.wide() {
        let (min, max) = int.range();
        a.mov(r10, max)?;
        a.cmp(q, r10)?;
        a.cmovg(q, r10)?;
        a.mov(r10, min)?;
        a.cmp(q, r10)?;
        a.cmovl(q, r10)?;
    }
    bind(a, done)
}

#[derive(Clone, Copy)]
struct Float {
    sign: u64,
    fraction: i32,
    bias: i32,
}

impl Float {
    fn new(lane: Lane) -> Self {
        let (fraction, bias) = match lane {
            Lane::F16 => (10, 15),
            Lane::BF16 => (7, 127),
            Lane::F32 => (23, 127),
            Lane::F64 => (52, 1023),
            _ => unreachable!("IEEE format"),
        };
        Self {
            sign: 1 << (lane.bytes() * 8 - 1),
            fraction,
            bias,
        }
    }
    fn infinity(self) -> u64 {
        self.sign - (1 << self.fraction)
    }
    fn fraction_mask(self) -> u64 {
        (1 << self.fraction) - 1
    }
}

fn neg_pair(a: &mut CodeAssembler) -> Result {
    a.neg(rax)?;
    a.adc(rdx, 0)?;
    a.neg(rdx)
}

// Decode the zero-extended bits in RAX as x = (-1)^sign * significand *
// 2^exponent: RAX becomes the significand, R8D the exponent and R9 the
// output sign bit. The exceptional branches leave R9 intact. A subnormal
// has the minimum normal exponent and no hidden bit.
fn unpack(a: &mut CodeAssembler, from: Lane, sign: u64, inf: CodeLabel, nan: CodeLabel) -> Result {
    let f = Float::new(from);
    a.mov(r9, rax)?;
    a.shr(r9, (from.bytes() * 8 - 1) as i32)?;
    a.shl(r9, sign.trailing_zeros() as i32)?;
    a.mov(r8, rax)?;
    a.shr(r8, f.fraction)?;
    let exponents = 2 * f.bias + 1;
    a.and(r8d, exponents)?;
    a.mov(r10, f.fraction_mask())?;
    a.and(rax, r10)?;
    let finite = a.create_label();
    a.cmp(r8d, exponents)?;
    a.jne(finite)?;
    a.test(rax, rax)?;
    a.je(inf)?;
    a.jmp(nan)?;
    bind(a, finite)?;
    let normal = a.create_label();
    let decoded = a.create_label();
    a.test(r8d, r8d)?;
    a.jne(normal)?;
    a.mov(r8d, 1)?;
    a.jmp(decoded)?;
    bind(a, normal)?;
    a.bts(rax, f.fraction)?;
    bind(a, decoded)?;
    a.sub(r8d, f.bias + f.fraction)?;
    a.xor(edx, edx)
}

/// Convert the float bits in RAX to the integer `to` in RAX.
fn to_integer(a: &mut CodeAssembler, from: Lane, to: Lane) -> Result {
    let f = Float::new(from);
    let size = to.bytes();
    let signed = to.signed();
    let inf = a.create_label();
    let zero = a.create_label();
    let sat = a.create_label();
    let done = a.create_label();
    unpack(a, from, 1, inf, zero)?;
    if !signed {
        a.test(r9d, r9d)?;
        a.jne(zero)?;
    }
    // Subnormals also truncate to zero. For normals, this is floor(log2(abs(x))).
    a.mov(ecx, r8d)?;
    a.add(ecx, f.fraction)?;
    a.js(zero)?;
    a.cmp(ecx, (size * 8) as i32 - i32::from(signed))?;
    a.jge(sat)?;
    let left = a.create_label();
    let magnitude = a.create_label();
    a.mov(ecx, r8d)?;
    a.test(ecx, ecx)?;
    a.jns(left)?;
    a.neg(ecx)?;
    a.shr(rax, cl)?;
    a.jmp(magnitude)?;
    bind(a, left)?;
    a.shl(rax, cl)?;
    bind(a, magnitude)?;
    if signed {
        a.test(r9d, r9d)?;
        a.je(done)?;
        a.neg(rax)?;
    }
    a.jmp(done)?;
    bind(a, inf)?;
    if !signed {
        a.test(r9d, r9d)?;
        a.jne(zero)?;
    }
    bind(a, sat)?;
    let max = (u64::MAX >> (64 - size * 8)) >> usize::from(signed);
    a.mov(rax, max)?;
    if signed {
        a.test(r9d, r9d)?;
        a.je(done)?;
        a.not(rax)?;
    }
    a.jmp(done)?;
    bind(a, zero)?;
    a.xor(eax, eax)?;
    bind(a, done)
}

// Normalize a nonzero 128-bit magnitude in RDX:RAX to put its leading one
// at bit 127. R8D becomes the exponent of that leading bit.
fn normalize(a: &mut CodeAssembler, zero: CodeLabel) -> Result {
    let upper = a.create_label();
    let shift = a.create_label();
    a.test(rdx, rdx)?;
    a.jne(upper)?;
    a.test(rax, rax)?;
    a.jz(zero)?;
    a.mov(rdx, rax)?;
    a.xor(eax, eax)?;
    a.jmp(shift)?;
    bind(a, upper)?;
    a.add(r8d, 64)?;
    bind(a, shift)?;
    a.bsr(rcx, rdx)?;
    a.add(r8d, ecx)?;
    a.xor(ecx, 63)?;
    a.shld(rdx, rax, cl)?;
    a.shl(rax, cl)
}

/// Convert RAX, an integer extended by its signedness or float bits, to
/// the float `to` in RAX, rounding to nearest even.
fn to_float(a: &mut CodeAssembler, from: Lane, to: Lane) -> Result {
    let f = Float::new(to);
    let inf = a.create_label();
    let nan = a.create_label();
    let zero = a.create_label();
    let done = a.create_label();
    if from.float() {
        unpack(a, from, f.sign, inf, nan)?;
    } else {
        a.xor(r9d, r9d)?;
        if from.signed() {
            let positive = a.create_label();
            a.cqo()?;
            a.test(rdx, rdx)?;
            a.jns(positive)?;
            neg_pair(a)?;
            a.mov(r9, f.sign)?;
            bind(a, positive)?;
        } else {
            a.xor(edx, edx)?;
        }
        a.xor(r8d, r8d)?;
    }
    normalize(a, zero)?;
    a.cmp(r8d, f.bias)?;
    a.jg(inf)?;
    let normal = a.create_label();
    let round = a.create_label();
    a.mov(ecx, 63 - f.fraction)?;
    a.cmp(r8d, 1 - f.bias)?;
    a.jge(normal)?;
    a.add(ecx, 1 - f.bias)?;
    a.sub(ecx, r8d)?;
    a.xor(r8d, r8d)?;
    a.jmp(round)?;
    bind(a, normal)?;
    a.add(r8d, f.bias - 1)?;
    a.shl(r8, f.fraction)?;
    bind(a, round)?;
    let full = a.create_label();
    let up = a.create_label();
    let keep = a.create_label();
    let tie = a.create_label();
    a.cmp(ecx, 64)?;
    a.jg(zero)?;
    a.je(full)?;
    // Left-align the bits the shift removes from the high word: shifting out
    // their top bit sets CF for at least a half and ZF when exactly a half.
    a.mov(r10, rdx)?;
    a.shr(rdx, cl)?;
    a.neg(ecx)?;
    a.shl(r10, cl)?;
    a.shl(r10, 1)?;
    a.jae(keep)?;
    a.jne(up)?;
    a.jmp(tie)?;
    bind(a, full)?;
    // The whole high word goes; its top bit is the normalized leading one.
    a.mov(r10, rdx)?;
    a.xor(edx, edx)?;
    a.shl(r10, 1)?;
    a.jne(up)?;
    bind(a, tie)?;
    a.test(rax, rax)?;
    a.jne(up)?;
    a.test(dl, 1)?;
    a.je(keep)?;
    bind(a, up)?;
    a.inc(rdx)?;
    bind(a, keep)?;
    a.lea(rax, ptr(r8 + rdx))?;
    a.jmp(done)?;
    bind(a, inf)?;
    a.mov(rax, f.infinity())?;
    a.jmp(done)?;
    bind(a, nan)?;
    a.mov(rax, f.infinity() | (1 << (f.fraction - 1)))?;
    a.jmp(done)?;
    bind(a, zero)?;
    a.xor(eax, eax)?;
    bind(a, done)?;
    a.or(rax, r9)
}
