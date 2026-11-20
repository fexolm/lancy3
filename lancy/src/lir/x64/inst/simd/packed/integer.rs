use super::*;

#[derive(Clone, Copy, Debug)]
pub(in crate::lir::x64) enum Rule {
    Neg,
    Abs,
    Mul,
    Cmp(B),
    MinMax(B),
    Shift(B),
    Ctz,
    BitSelect,
    Saturate(B),
}

impl Rule {
    pub(super) fn temps(self, input: Type) -> usize {
        match self {
            Self::Neg => 0,
            Self::Abs if !input.lane().unwrap().signed() => 0,
            Self::Ctz | Self::BitSelect => 1,
            Self::Shift(_) | Self::Mul => 2,
            _ => 3,
        }
    }
    pub(super) fn emit(self, v: &mut Vector, input: Type, r: &[usize], t: &[usize]) -> Result {
        let d = *r.last().unwrap();
        let x = r[0];
        let y = r[1];
        let lane = input.lane().unwrap();
        let n = lane.bytes();
        match self {
            Self::Neg => {
                v.zero(d)?;
                sub(v, n, d, d, x)
            }
            Self::Abs if !lane.signed() => v.mov(d, x),
            Self::Abs => {
                v.zero(d)?;
                compare(v, lane, B::Gt, d, d, x, t)?;
                v.xor(t[0], x, d)?;
                sub(v, n, d, t[0], d)
            }
            Self::Mul => mul(v, n, d, x, y, t),
            Self::Cmp(op) => compare(v, lane, op, d, x, y, t),
            Self::MinMax(op) => {
                compare(v, lane, B::Gt, d, x, y, t)?;
                let (a, b) = if op == B::Max { (x, y) } else { (y, x) };
                select_bits(v, d, d, a, b, t[0])
            }
            Self::Shift(op) => shift(v, lane, op, d, x, y, t),
            Self::Ctz => {
                v.zero(d)?;
                sub(v, n, d, d, x)?;
                v.and(d, d, x)?;
                v.ones(t[0])?;
                add(v, n, d, d, t[0])?;
                match n {
                    1 => v.a.vpopcntb(ZMM[d], ZMM[d]),
                    2 => v.a.vpopcntw(ZMM[d], ZMM[d]),
                    4 => v.a.vpopcntd(ZMM[d], ZMM[d]),
                    _ => v.a.vpopcntq(ZMM[d], ZMM[d]),
                }
            }
            Self::BitSelect => select_bits(v, d, x, y, r[2], t[0]),
            Self::Saturate(op) => {
                // For signed addition the sum overflows when it differs in sign from both inputs.
                let plus = op == B::SaturatingAdd;
                if plus {
                    add(v, n, d, x, y)?;
                } else {
                    sub(v, n, d, x, y)?;
                }
                if lane.signed() {
                    v.xor(t[0], x, if plus { d } else { y })?;
                    v.xor(t[1], if plus { y } else { x }, d)?;
                    v.and(t[0], t[0], t[1])?;
                    sign(v, n, t[0], t[0])?;
                    sign(v, n, t[1], x)?;
                    v.splat(t[2], n, (1u64 << (n * 8 - 1)) - 1)?;
                    v.xor(t[1], t[1], t[2])?;
                    select_bits(v, d, t[0], t[1], d, t[2])
                } else {
                    // Carry/borrow in the sign bit, computed without an unsigned comparison.
                    if plus {
                        v.and(t[0], x, y)?;
                        v.or(t[1], x, y)?;
                        v.andnot(t[1], d, t[1])?;
                    } else {
                        v.andnot(t[0], x, y)?;
                        v.xor(t[1], x, y)?;
                        v.andnot(t[1], t[1], d)?;
                    }
                    v.or(t[0], t[0], t[1])?;
                    sign(v, n, t[0], t[0])?;
                    if plus {
                        v.or(d, d, t[0])
                    } else {
                        v.andnot(d, t[0], d)
                    }
                }
            }
        }
    }
}

pub(super) fn select_bits(
    v: &mut Vector,
    d: usize,
    mask: usize,
    yes: usize,
    no: usize,
    t: usize,
) -> Result {
    if v.width == 64 && d != no && d != yes {
        v.mov(d, mask)?;
        return v.a.vpternlogd(ZMM[d], ZMM[yes], ZMM[no], 0xca);
    }
    v.and(t, mask, yes)?;
    v.andnot(d, mask, no)?;
    v.or(d, d, t)
}
pub(super) fn add(v: &mut Vector, n: usize, d: usize, x: usize, y: usize) -> Result {
    match n {
        1 => binary!(v, d, x, y, paddb, vpaddb, vpaddb),
        2 => binary!(v, d, x, y, paddw, vpaddw, vpaddw),
        4 => binary!(v, d, x, y, paddd, vpaddd, vpaddd),
        _ => binary!(v, d, x, y, paddq, vpaddq, vpaddq),
    }
}
fn sub(v: &mut Vector, n: usize, d: usize, x: usize, y: usize) -> Result {
    match n {
        1 => binary!(v, d, x, y, psubb, vpsubb, vpsubb),
        2 => binary!(v, d, x, y, psubw, vpsubw, vpsubw),
        4 => binary!(v, d, x, y, psubd, vpsubd, vpsubd),
        _ => binary!(v, d, x, y, psubq, vpsubq, vpsubq),
    }
}
fn shuffle32(v: &mut Vector, d: usize, x: usize, imm: i32) -> Result {
    match v.width {
        16 => v.a.pshufd(XMM[d], XMM[x], imm),
        32 => v.a.vpshufd(YMM[d], YMM[x], imm),
        _ => v.a.vpshufd(ZMM[d], ZMM[x], imm),
    }
}
fn sign(v: &mut Vector, n: usize, d: usize, x: usize) -> Result {
    if n == 4 {
        return immediate!(v, d, x, 31, psrad, vpsrad);
    }
    if v.width == 64 {
        return v.a.vpsraq(ZMM[d], ZMM[x], 63);
    }
    shuffle32(v, d, x, 0xf5)?;
    immediate!(v, d, d, 31, psrad, vpsrad)
}
macro_rules! cmp_short {
    ($v:expr,$d:expr,$x:expr,$y:expr,$s:ident,$p:ident) => {{
        if $v.width == 32 {
            $v.a.$p(YMM[$d], YMM[$x], YMM[$y])
        } else {
            let dest = if $d == $y && $x != $y {
                crate::lir::x64::inst::SCRATCH
            } else {
                $d
            };
            $v.mov(dest, $x)?;
            $v.a.$s(XMM[dest], XMM[$y])?;
            $v.mov($d, dest)
        }
    }};
}
fn compare(
    v: &mut Vector,
    lane: Lane,
    op: B,
    d: usize,
    mut x: usize,
    mut y: usize,
    t: &[usize],
) -> Result {
    let n = lane.bytes();
    if matches!(op, B::Eq | B::Ne) {
        match n {
            1 => cmp_short!(v, d, x, y, pcmpeqb, vpcmpeqb),
            2 => cmp_short!(v, d, x, y, pcmpeqw, vpcmpeqw),
            4 => cmp_short!(v, d, x, y, pcmpeqd, vpcmpeqd),
            _ if v.cpu.sse41 || v.width == 32 => {
                if v.width == 32 {
                    v.a.vpcmpeqq(YMM[d], YMM[x], YMM[y])
                } else {
                    v.mov(d, x)?;
                    v.a.pcmpeqq(XMM[d], XMM[y])
                }
            }
            _ => {
                cmp_short!(v, d, x, y, pcmpeqd, vpcmpeqd)?;
                shuffle32(v, t[0], d, 0xb1)?;
                v.and(d, d, t[0])
            }
        }?;
        if op == B::Ne {
            v.invert(d, d)?;
        }
        return Ok(());
    }
    if matches!(op, B::Lt | B::Ge) {
        std::mem::swap(&mut x, &mut y);
    }
    if n == 8 && v.width == 16 && !v.cpu.sse42 {
        // Signed high dword, unsigned low dword. Equal high halves defer to the low half.
        let bias = if lane.signed() {
            0x0000_0000_8000_0000
        } else {
            0x8000_0000_8000_0000
        };
        v.splat(t[0], 8, bias)?;
        v.xor(t[1], x, t[0])?;
        v.xor(t[2], y, t[0])?;
        cmp_short!(v, t[1], t[1], t[2], pcmpgtd, vpcmpgtd)?;
        v.mov(t[2], x)?;
        v.a.pcmpeqd(XMM[t[2]], XMM[y])?;
        shuffle32(v, d, t[1], 0xf5)?;
        shuffle32(v, t[0], t[1], 0xa0)?;
        shuffle32(v, t[2], t[2], 0xf5)?;
        v.and(t[0], t[0], t[2])?;
        v.or(d, d, t[0])?;
    } else {
        if !lane.signed() {
            v.splat(t[0], n, 1u64 << (n * 8 - 1))?;
            v.xor(t[1], x, t[0])?;
            v.xor(t[2], y, t[0])?;
            x = t[1];
            y = t[2];
        }
        match n {
            1 => cmp_short!(v, d, x, y, pcmpgtb, vpcmpgtb),
            2 => cmp_short!(v, d, x, y, pcmpgtw, vpcmpgtw),
            4 => cmp_short!(v, d, x, y, pcmpgtd, vpcmpgtd),
            _ => {
                if v.width == 32 {
                    v.a.vpcmpgtq(YMM[d], YMM[x], YMM[y])
                } else {
                    v.mov(d, x)?;
                    v.a.pcmpgtq(XMM[d], XMM[y])
                }
            }
        }?;
    }
    if matches!(op, B::Le | B::Ge) {
        v.invert(d, d)?;
    }
    Ok(())
}
fn mul(v: &mut Vector, n: usize, d: usize, x: usize, y: usize, t: &[usize]) -> Result {
    let (a, b) = (t[0], t[1]);
    match n {
        1 => {
            immediate!(v, a, x, 8, psrlw, vpsrlw)?;
            immediate!(v, b, y, 8, psrlw, vpsrlw)?;
            binary!(v, a, a, b, pmullw, vpmullw, vpmullw)?;
            immediate!(v, a, a, 8, psllw, vpsllw)?;
            binary!(v, d, x, y, pmullw, vpmullw, vpmullw)?;
            v.splat(b, 2, 255)?;
            v.and(d, d, b)?;
            v.or(d, d, a)
        }
        4 => {
            v.a.pshufd(XMM[a], XMM[x], 0xb1)?;
            v.a.pshufd(XMM[b], XMM[y], 0xb1)?;
            v.a.pmuludq(XMM[a], XMM[b])?;
            v.mov(d, x)?;
            v.a.pmuludq(XMM[d], XMM[y])?;
            v.a.pshufd(XMM[a], XMM[a], 0x88)?;
            v.a.pshufd(XMM[d], XMM[d], 0x88)?;
            v.a.punpckldq(XMM[d], XMM[a])
        }
        _ => {
            immediate!(v, a, x, 32, psrlq, vpsrlq)?;
            immediate!(v, b, y, 32, psrlq, vpsrlq)?;
            binary!(v, a, a, y, pmuludq, vpmuludq, vpmuludq)?;
            binary!(v, b, b, x, pmuludq, vpmuludq, vpmuludq)?;
            add(v, 8, a, a, b)?;
            immediate!(v, a, a, 32, psllq, vpsllq)?;
            binary!(v, d, x, y, pmuludq, vpmuludq, vpmuludq)?;
            add(v, 8, d, d, a)
        }
    }
}
fn shift(v: &mut Vector, lane: Lane, op: B, d: usize, x: usize, y: usize, t: &[usize]) -> Result {
    let n = lane.bytes();
    let bits = (n * 8) as u64;
    let rot = matches!(op, B::RotateLeft | B::RotateRight);
    if rot && v.width == 64 && n >= 4 {
        return match (n, op) {
            (4, B::RotateLeft) => v.a.vprolvd(ZMM[d], ZMM[x], ZMM[y]),
            (4, _) => v.a.vprorvd(ZMM[d], ZMM[x], ZMM[y]),
            (_, B::RotateLeft) => v.a.vprolvq(ZMM[d], ZMM[x], ZMM[y]),
            _ => v.a.vprorvq(ZMM[d], ZMM[x], ZMM[y]),
        };
    }
    let (a, b) = (t[0], t[1]);
    v.splat(a, n, bits - 1)?;
    v.and(b, y, a)?;
    let left = matches!(op, B::RotateLeft | B::Shl);
    if n == 8 && v.width == 32 && lane.signed() && op == B::Shr {
        sign(v, 8, a, x)?;
        v.xor(d, x, a)?;
        v.a.vpsrlvq(YMM[d], YMM[d], YMM[b])?;
        return v.xor(d, d, a);
    }
    variable(v, n, left, lane.signed() && op == B::Shr, d, x, b)?;
    if rot {
        v.zero(b)?;
        sub(v, n, b, b, y)?;
        v.and(b, b, a)?;
        variable(v, n, !left, false, b, x, b)?;
        v.or(d, d, b)?;
    }
    Ok(())
}
fn variable(
    v: &mut Vector,
    n: usize,
    left: bool,
    signed: bool,
    d: usize,
    x: usize,
    y: usize,
) -> Result {
    macro_rules! var {
        ($op:ident) => {
            if v.width == 32 {
                v.a.$op(YMM[d], YMM[x], YMM[y])
            } else {
                v.a.$op(ZMM[d], ZMM[x], ZMM[y])
            }
        };
    }
    match (n, left, signed) {
        (2, true, _) => v.a.vpsllvw(ZMM[d], ZMM[x], ZMM[y]),
        (2, false, true) => v.a.vpsravw(ZMM[d], ZMM[x], ZMM[y]),
        (2, false, false) => v.a.vpsrlvw(ZMM[d], ZMM[x], ZMM[y]),
        (4, true, _) => var!(vpsllvd),
        (4, false, true) => var!(vpsravd),
        (4, false, false) => var!(vpsrlvd),
        (_, true, _) => var!(vpsllvq),
        (_, false, true) => v.a.vpsravq(ZMM[d], ZMM[x], ZMM[y]),
        _ => var!(vpsrlvq),
    }
}
