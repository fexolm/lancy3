//! Numerical casts emitted as x64 instructions. Conversions that x86 does not
//! provide use the integer significand directly, including its low 64 bits.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Float {
    pub sign: u64,
    pub fraction: i32,
    pub bias: i32,
}
impl Float {
    pub fn new(lane: Lane) -> Self {
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
    pub fn infinity(self) -> u64 {
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

// Decode x = (-1)^sign * significand * 2^exponent. RAX is the significand,
// R8D the exponent and R9 the output sign bit. The exceptional branches leave
// R9 intact. A subnormal has the minimum normal exponent and no hidden bit.
fn unpack(
    a: &mut CodeAssembler,
    from: Lane,
    src: i32,
    sign: u64,
    inf: CodeLabel,
    nan: CodeLabel,
) -> Result {
    let f = Float::new(from);
    word(a, 0, src, from.bytes(), false)?;
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

fn to_integer(a: &mut CodeAssembler, from: Lane, to: Lane, src: i32, out: i32) -> Result {
    let f = Float::new(from);
    let size = to.bytes();
    let signed = to.signed();
    let inf = a.create_label();
    let zero = a.create_label();
    let sat = a.create_label();
    let done = a.create_label();
    unpack(a, from, src, 1, inf, zero)?;
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
    let small = a.create_label();
    a.test(ecx, 64)?;
    a.jz(small)?;
    a.mov(rdx, rax)?;
    a.xor(eax, eax)?;
    bind(a, small)?;
    a.shld(rdx, rax, cl)?;
    a.shl(rax, cl)?;
    bind(a, magnitude)?;
    if signed {
        a.test(r9d, r9d)?;
        a.je(done)?;
        neg_pair(a)?;
    }
    a.jmp(done)?;
    bind(a, inf)?;
    if !signed {
        a.test(r9d, r9d)?;
        a.jne(zero)?;
    }
    bind(a, sat)?;
    let max = (u128::MAX >> (128 - size * 8)) >> usize::from(signed);
    a.mov(rax, max as u64)?;
    a.mov(rdx, (max >> 64) as u64)?;
    if signed {
        a.test(r9d, r9d)?;
        a.je(done)?;
        a.not(rax)?;
        a.not(rdx)?;
    }
    a.jmp(done)?;
    bind(a, zero)?;
    a.xor(eax, eax)?;
    a.xor(edx, edx)?;
    bind(a, done)?;
    put(a, out, size)
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

// At a midpoint the residual of an exact half FMA can lie on either side of
// the rounded f64 sum. Its sign relative to the sum determines the direction.
fn midpoint(
    a: &mut CodeAssembler,
    residual: Option<i32>,
    sign: u64,
    up: CodeLabel,
    keep: CodeLabel,
) -> Result {
    if let Some(at) = residual {
        let even = a.create_label();
        a.mov(r10, qword_ptr(rsp + at))?;
        a.mov(rcx, r10)?;
        a.shl(rcx, 1)?;
        a.je(even)?;
        a.shr(r10, 63)?;
        a.shl(r10, sign.trailing_zeros() as i32)?;
        a.cmp(r9, r10)?;
        a.je(up)?;
        a.jmp(keep)?;
        bind(a, even)?;
    }
    a.test(dl, 1)?;
    a.jne(up)?;
    a.jmp(keep)
}

pub(super) fn to_float(
    a: &mut CodeAssembler,
    from: Lane,
    to: Lane,
    src: i32,
    out: i32,
    residual: Option<i32>,
) -> Result {
    let f = Float::new(to);
    let inf = a.create_label();
    let nan = a.create_label();
    let zero = a.create_label();
    let done = a.create_label();
    if from.float() {
        unpack(a, from, src, f.sign, inf, nan)?;
    } else {
        let size = from.bytes();
        a.xor(r9d, r9d)?;
        word(a, 0, src, size, from.signed())?;
        if from.signed() {
            a.cqo()?;
        } else {
            a.xor(edx, edx)?;
        }
        if from.signed() {
            let positive = a.create_label();
            a.test(rdx, rdx)?;
            a.jns(positive)?;
            neg_pair(a)?;
            a.mov(r9, f.sign)?;
            bind(a, positive)?;
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
    // Removed high-word bits are left-aligned so comparison to bit 63 tests
    // the guard bit and the complete high-word remainder in one operation.
    a.mov(r11, rdx)?;
    a.shr(rdx, cl)?;
    a.neg(ecx)?;
    a.shl(r11, cl)?;
    a.mov(r10, 1u64 << 63)?;
    a.cmp(r11, r10)?;
    a.ja(up)?;
    a.jb(keep)?;
    a.jmp(tie)?;
    bind(a, full)?;
    a.mov(r11, rdx)?;
    a.xor(edx, edx)?;
    a.mov(r10, 1u64 << 63)?;
    a.cmp(r11, r10)?;
    a.ja(up)?;
    bind(a, tie)?;
    a.test(rax, rax)?;
    a.jne(up)?;
    midpoint(a, residual, f.sign, up, keep)?;
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
    a.or(rax, r9)?;
    put(a, out, to.bytes())
}

pub(super) fn emit(a: &mut CodeAssembler, from: Lane, to: Lane, src: i32, dst: i32) -> Result {
    let n = from.bytes();
    let d = to.bytes();
    if from == to || (!from.float() && !to.float() && n >= d) {
        return copy(a, ptr(rsp + dst), ptr(rsp + src), d);
    }
    if !from.float() && !to.float() {
        word(a, 0, src, n, from.signed())?;
        return put(a, dst, d);
    }
    match (from, to) {
        (Lane::F32, Lane::F64) => {
            a.cvtss2sd(xmm0, dword_ptr(rsp + src))?;
            a.movq(qword_ptr(rsp + dst), xmm0)
        }
        (Lane::F64, Lane::F32) => {
            a.cvtsd2ss(xmm0, qword_ptr(rsp + src))?;
            a.movss(dword_ptr(rsp + dst), xmm0)
        }
        (_, Lane::F32 | Lane::F64) if !from.float() && (n < 8 || n == 8 && from.signed()) => {
            word(a, 0, src, n, from.signed())?;
            if to == Lane::F32 {
                a.cvtsi2ss(xmm0, rax)?;
                a.movss(dword_ptr(rsp + dst), xmm0)
            } else {
                a.cvtsi2sd(xmm0, rax)?;
                a.movq(qword_ptr(rsp + dst), xmm0)
            }
        }
        _ if to.float() => to_float(a, from, to, src, dst, None),
        _ => to_integer(a, from, to, src, dst),
    }
}
