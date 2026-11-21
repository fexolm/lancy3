use super::*;
use crate::lir::x64::inst::SCRATCH;

#[derive(Clone, Copy, Debug)]
pub(in crate::lir::x64) enum PairedShuffle {
    Interleave(bool),
    Deinterleave(bool),
}

impl PairedShuffle {
    pub fn index(self, i: usize, lanes: usize) -> usize {
        match self {
            Self::Interleave(high) => i / 2 + i % 2 * lanes + usize::from(high) * lanes / 2,
            Self::Deinterleave(odd) => 2 * i + usize::from(odd),
        }
    }

    pub fn emit(
        self,
        a: &mut CodeAssembler,
        pool: &mut emit::Constants,
        width: usize,
        [dst, lhs, rhs]: [usize; 3],
    ) -> Result {
        match width {
            16 => {
                let target = if dst == rhs && lhs != rhs {
                    SCRATCH
                } else {
                    dst
                };
                if target != lhs {
                    a.movaps(XMM[target], XMM[lhs])?;
                }
                match self {
                    Self::Interleave(false) => a.punpckldq(XMM[target], XMM[rhs])?,
                    Self::Interleave(true) => a.punpckhdq(XMM[target], XMM[rhs])?,
                    Self::Deinterleave(odd) => {
                        a.shufps(XMM[target], XMM[rhs], if odd { 0xdd } else { 0x88 })?;
                    }
                }
                if target != dst {
                    a.movaps(XMM[dst], XMM[target])?;
                }
                Ok(())
            }
            32 => match self {
                Self::Interleave(high) => {
                    let low = YMM[SCRATCH];
                    let (d, x, y) = (YMM[dst], YMM[lhs], YMM[rhs]);
                    a.vpunpckldq(low, x, y)?;
                    a.vpunpckhdq(d, x, y)?;
                    a.vperm2i128(d, low, d, if high { 0x31 } else { 0x20 })
                }
                Self::Deinterleave(odd) => {
                    let d = YMM[dst];
                    a.vshufps(d, YMM[lhs], YMM[rhs], if odd { 0xdd } else { 0x88 })?;
                    a.vpermq(d, d, 0xd8)
                }
            },
            _ => {
                let label = a.create_label();
                let mut bits = [0u8; 64];
                for (i, word) in bits.chunks_exact_mut(4).enumerate() {
                    word.copy_from_slice(&(self.index(i, 16) as u32).to_le_bytes());
                }
                pool.push((label, bits, 64));
                load(a, SCRATCH, ptr(label), 64)?;
                if dst == rhs && lhs != rhs {
                    a.vpermi2d(ZMM[SCRATCH], ZMM[lhs], ZMM[rhs])?;
                    copy(a, dst, SCRATCH, 64)
                } else {
                    if dst != lhs {
                        copy(a, dst, lhs, 64)?;
                    }
                    a.vpermt2d(ZMM[dst], ZMM[SCRATCH], ZMM[rhs])
                }
            }
        }
    }
}

pub(in crate::lir::x64) fn extract_dword(
    a: &mut CodeAssembler,
    width: usize,
    lane: Lane,
    [dst, src, index]: [usize; 3],
) -> Result {
    use emit::R32;
    let x = XMM[SCRATCH];
    a.vmovd(x, R32[index])?;
    if width == 32 {
        a.vpermd(YMM[SCRATCH], YMM[SCRATCH], YMM[src])?;
    } else {
        a.vpermd(ZMM[SCRATCH], ZMM[SCRATCH], ZMM[src])?;
    }
    if lane.float() {
        a.vmovaps(XMM[dst], x)
    } else {
        a.vmovd(R32[dst], x)
    }
}

pub(in crate::lir::x64) fn insert_dword(
    a: &mut CodeAssembler,
    pool: &mut emit::Constants,
    width: usize,
    lane: Lane,
    [dst, src, index]: [usize; 3],
    replacement: usize,
    temp: Option<usize>,
) -> Result {
    use emit::R32;
    let label = a.create_label();
    pool.push((
        label,
        crate::lir::constants::iota_bytes(Lane::I32, width),
        width,
    ));
    if width == 32 {
        let scratch = YMM[SCRATCH];
        let temp = temp.unwrap();
        a.vmovd(XMM[SCRATCH], R32[index])?;
        a.vpbroadcastd(scratch, XMM[SCRATCH])?;
        a.vpcmpeqd(scratch, scratch, ymmword_ptr(label))?;
        let scalar = if lane.float() {
            XMM[replacement]
        } else {
            a.vmovd(XMM[temp], R32[replacement])?;
            XMM[temp]
        };
        a.vpbroadcastd(YMM[temp], scalar)?;
        a.vblendvps(YMM[dst], YMM[src], YMM[temp], scratch)
    } else {
        let scratch = ZMM[SCRATCH];
        a.vpbroadcastd(scratch, R32[index])?;
        a.vpcmpd(k1, scratch, zmmword_ptr(label), 0)?;
        if lane.float() && dst == replacement && dst != src {
            a.vpbroadcastd(scratch, XMM[replacement])?;
            return a.vpblendmd(ZMM[dst].k1(), ZMM[src], scratch);
        }
        if dst != src {
            copy(a, dst, src, width)?;
        }
        if lane.float() {
            a.vpbroadcastd(ZMM[dst].k1(), XMM[replacement])
        } else {
            a.vpbroadcastd(ZMM[dst].k1(), R32[replacement])
        }
    }
}

pub(in crate::lir::x64) fn shift_dwords(
    a: &mut CodeAssembler,
    width: usize,
    dst: usize,
    src: usize,
    count: u8,
) -> Result {
    let bytes = 4 * i32::from(count);
    match width {
        16 => {
            if dst != src {
                a.movaps(XMM[dst], XMM[src])?;
            }
            a.pslldq(XMM[dst], bytes)
        }
        32 => {
            let (dst, src, low) = (YMM[dst], YMM[src], YMM[SCRATCH]);
            if bytes == 16 {
                return a.vperm2i128(dst, src, src, 0x08);
            }
            a.vperm2i128(low, src, src, 0x08)?;
            if bytes < 16 {
                a.vpalignr(dst, src, low, 16 - bytes)
            } else {
                a.vpslldq(dst, low, bytes - 16)
            }
        }
        64 => {
            let zero = ZMM[SCRATCH];
            a.vpxord(zero, zero, zero)?;
            a.valignd(ZMM[dst], ZMM[src], zero, 16 - i32::from(count))
        }
        _ => unreachable!(),
    }
}

pub(in crate::lir::x64) fn extract_lane(
    a: &mut CodeAssembler,
    cpu: Features,
    lane: Lane,
    dst: usize,
    src: usize,
    index: u8,
) -> Result {
    use emit::{R8, R32, R64};
    let size = lane.bytes();
    let byte = usize::from(index) * size;
    let chunk = (byte / 16) as i32;
    let vector_result = matches!(lane, Lane::F32 | Lane::F64 | Lane::I128 | Lane::U128);
    let wide = cpu.bytes > 16;
    let source = if chunk == 0 {
        XMM[src]
    } else {
        // GPR destinations can have the same number as an unrelated live XMM.
        let target = XMM[if vector_result { dst } else { SCRATCH }];
        if cpu.bytes == 32 {
            a.vextractf128(target, YMM[src], chunk)?;
        } else {
            a.vextracti32x4(target, ZMM[src], chunk)?;
        }
        target
    };
    let offset = (byte % 16) as i32;
    macro_rules! encoding {
        ($sse:ident, $vex:ident, $($args:expr),+) => {
            if wide { a.$vex($($args),+) } else { a.$sse($($args),+) }
        };
    }
    if vector_result {
        let target = XMM[dst];
        return if offset != 0 {
            encoding!(
                pshufd,
                vpshufd,
                target,
                source,
                if size == 8 { 0x4e } else { offset / 4 }
            )
        } else if target != source {
            encoding!(movaps, vmovaps, target, source)
        } else {
            Ok(())
        };
    }
    let dword = R32[dst];
    match size {
        1 if wide || cpu.sse41 => encoding!(pextrb, vpextrb, dword, source, offset),
        1 => {
            a.pextrw(dword, source, offset / 2)?;
            if offset % 2 == 0 {
                a.movzx(dword, R8[dst])
            } else {
                a.shr(dword, 8)
            }
        }
        2 => encoding!(pextrw, vpextrw, dword, source, offset / 2),
        4 if offset == 0 => encoding!(movd, vmovd, dword, source),
        8 if offset == 0 => encoding!(movq, vmovq, R64[dst], source),
        4 if wide || cpu.sse41 => encoding!(pextrd, vpextrd, dword, source, offset / 4),
        8 if wide || cpu.sse41 => encoding!(pextrq, vpextrq, R64[dst], source, offset / 8),
        _ => {
            let shifted = XMM[SCRATCH];
            a.pshufd(shifted, source, if size == 8 { 0x4e } else { offset / 4 })?;
            if size == 4 {
                a.movd(dword, shifted)
            } else {
                a.movq(R64[dst], shifted)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::lir::x64) enum Rule {
    FloatWidth,
    Shuffle,
    Widen(bool),
    Narrow,
    Reduce(B),
}
impl Rule {
    pub(super) fn temps(self, action: Action, input: Type, c: Features) -> usize {
        match self {
            Self::FloatWidth => match (action, input.lane().unwrap()) {
                (Action::Widen(high), Lane::F32) => usize::from(high),
                (Action::Widen(_), _) => 4,
                (_, Lane::F64) => 2,
                _ => 5,
            },
            Self::Shuffle if c.bytes == 32 && !evex(c) => 3,
            Self::Widen(false) if c.bytes != 16 || c.sse41 => 0,
            Self::Shuffle | Self::Widen(_) => 1,
            Self::Narrow if c.bytes == 64 => 1,
            Self::Narrow | Self::Reduce(_) => 2,
        }
    }
    pub(super) fn emit(
        self,
        v: &mut Vector,
        action: Action,
        input: Type,
        output: Type,
        r: &[usize],
        t: &[usize],
    ) -> Result {
        let (x, y, d) = (r[0], r[1], *r.last().unwrap());
        let e = input.lane().unwrap();
        let n = e.bytes();
        match self {
            Self::FloatWidth => float_width(v, action, input, output, r, t),
            Self::Shuffle if evex(v.cpu) => {
                let i = r[2];
                v.splat(t[0], n, (2 * v.width / n) as u64)?;
                macro_rules! shuffle {
                    ($r:ident, $cmp:ident, $permute:ident) => {{
                        v.a.$cmp(k1, $r[i], $r[t[0]], 1)?;
                        v.mov(d, x)?;
                        v.a.$permute($r[d].k1().z(), $r[i], $r[y])
                    }};
                }
                macro_rules! width {
                    ($cmp:ident, $permute:ident) => {
                        match v.width {
                            16 => shuffle!(XMM, $cmp, $permute),
                            32 => shuffle!(YMM, $cmp, $permute),
                            _ => shuffle!(ZMM, $cmp, $permute),
                        }
                    };
                }
                match n {
                    1 => width!(vpcmpub, vpermt2b),
                    2 => width!(vpcmpuw, vpermt2w),
                    4 => width!(vpcmpud, vpermt2d),
                    _ => width!(vpcmpuq, vpermt2q),
                }
            }
            Self::Shuffle => {
                let i = r[2];
                let (a, b, c) = (t[0], t[1], t[2]);
                v.a.vpermd(YMM[d], YMM[i], YMM[x])?;
                v.a.vpermd(YMM[a], YMM[i], YMM[y])?;
                v.a.vpslld(YMM[b], YMM[i], 28)?;
                v.a.vblendvps(YMM[d], YMM[d], YMM[a], YMM[b])?;
                v.a.vpsrld(YMM[b], YMM[i], 4)?;
                v.zero(c)?;
                v.a.vpcmpeqd(YMM[b], YMM[b], YMM[c])?;
                v.and(d, d, b)
            }
            Self::Widen(high) if v.width == 16 && !v.cpu.sse41 => {
                let a = t[0];
                v.zero(a)?;
                if e.signed() {
                    match n {
                        1 => v.a.pcmpgtb(XMM[a], XMM[x]),
                        2 => v.a.pcmpgtw(XMM[a], XMM[x]),
                        _ => v.a.pcmpgtd(XMM[a], XMM[x]),
                    }?;
                }
                v.mov(d, x)?;
                match (n, high) {
                    (1, false) => v.a.punpcklbw(XMM[d], XMM[a]),
                    (1, true) => v.a.punpckhbw(XMM[d], XMM[a]),
                    (2, false) => v.a.punpcklwd(XMM[d], XMM[a]),
                    (2, true) => v.a.punpckhwd(XMM[d], XMM[a]),
                    (_, false) => v.a.punpckldq(XMM[d], XMM[a]),
                    (_, true) => v.a.punpckhdq(XMM[d], XMM[a]),
                }
            }
            Self::Widen(high) => {
                let s = if high {
                    let a = t[0];
                    match v.width {
                        16 => {
                            v.mov(a, x)?;
                            v.a.psrldq(XMM[a], 8)?;
                        }
                        32 => v.a.vextractf128(XMM[a], YMM[x], 1)?,
                        _ => v.a.vextracti64x4(YMM[a], ZMM[x], 1)?,
                    }
                    a
                } else {
                    x
                };
                macro_rules! widen {
                    ($s:ident,$p:ident) => {
                        match v.width {
                            16 => v.a.$s(XMM[d], XMM[s]),
                            32 => v.a.$p(YMM[d], XMM[s]),
                            _ => v.a.$p(ZMM[d], YMM[s]),
                        }
                    };
                }
                match (n, e.signed()) {
                    (1, true) => widen!(pmovsxbw, vpmovsxbw),
                    (1, false) => widen!(pmovzxbw, vpmovzxbw),
                    (2, true) => widen!(pmovsxwd, vpmovsxwd),
                    (2, false) => widen!(pmovzxwd, vpmovzxwd),
                    (_, true) => widen!(pmovsxdq, vpmovsxdq),
                    (_, false) => widen!(pmovzxdq, vpmovzxdq),
                }
            }
            Self::Narrow if v.width == 64 => {
                let a = t[0];
                macro_rules! pack {
                    ($p:ident) => {{
                        v.a.$p(YMM[d], ZMM[x])?;
                        v.a.$p(YMM[a], ZMM[y])?;
                    }};
                }
                match n {
                    2 => pack!(vpmovwb),
                    4 => pack!(vpmovdw),
                    _ => pack!(vpmovqd),
                }
                v.a.vinserti64x4(ZMM[d], ZMM[d], YMM[a], 1)
            }
            Self::Narrow => {
                let a = t[0];
                let b = t[1];
                match n {
                    2 => {
                        v.splat(b, 2, 255)?;
                        v.and(d, x, b)?;
                        v.and(a, y, b)?;
                        binary!(v, d, d, a, packuswb, vpackuswb, vpackuswb)?;
                    }
                    4 => {
                        immediate!(v, d, x, 16, pslld, vpslld)?;
                        immediate!(v, d, d, 16, psrad, vpsrad)?;
                        immediate!(v, a, y, 16, pslld, vpslld)?;
                        immediate!(v, a, a, 16, psrad, vpsrad)?;
                        binary!(v, d, d, a, packssdw, vpackssdw, vpackssdw)?;
                    }
                    _ => {
                        if v.width == 32 {
                            v.a.vshufps(YMM[d], YMM[x], YMM[y], 0x88)?;
                        } else {
                            v.mov(d, x)?;
                            v.a.shufps(XMM[d], XMM[y], 0x88)?;
                        }
                    }
                }
                if v.width == 32 {
                    v.a.vpermq(YMM[d], YMM[d], 0xd8)?;
                }
                Ok(())
            }
            Self::Reduce(op) => {
                // These operations are associative modulo the lane width. Strict FP folds use the ordered emitter.
                let a = t[0];
                let b = t[1];
                let mut src = x;
                let combine = |v: &mut Vector, d, x, y| match op {
                    B::Add => integer::add(v, n, d, x, y),
                    B::And => v.and(d, x, y),
                    B::Or => v.or(d, x, y),
                    B::Xor => v.xor(d, x, y),
                    B::Mul if n == 2 => binary!(v, d, x, y, pmullw, vpmullw, vpmullw),
                    B::Mul => binary!(v, d, x, y, pmulld, vpmulld, vpmulld),
                    _ => match (n, e.signed(), op == B::Min) {
                        (1, false, true) => binary!(v, d, x, y, pminub, vpminub, vpminub),
                        (1, false, false) => binary!(v, d, x, y, pmaxub, vpmaxub, vpmaxub),
                        (1, true, true) => binary!(v, d, x, y, pminsb, vpminsb, vpminsb),
                        (1, true, false) => binary!(v, d, x, y, pmaxsb, vpmaxsb, vpmaxsb),
                        (2, false, true) => binary!(v, d, x, y, pminuw, vpminuw, vpminuw),
                        (2, false, false) => binary!(v, d, x, y, pmaxuw, vpmaxuw, vpmaxuw),
                        (2, true, true) => binary!(v, d, x, y, pminsw, vpminsw, vpminsw),
                        (2, true, false) => binary!(v, d, x, y, pmaxsw, vpmaxsw, vpmaxsw),
                        (_, false, true) => binary!(v, d, x, y, pminud, vpminud, vpminud),
                        (_, false, false) => binary!(v, d, x, y, pmaxud, vpmaxud, vpmaxud),
                        (_, true, true) => binary!(v, d, x, y, pminsd, vpminsd, vpminsd),
                        (_, true, false) => binary!(v, d, x, y, pmaxsd, vpmaxsd, vpmaxsd),
                    },
                };
                if v.width == 64 {
                    v.a.vextracti64x4(YMM[b], ZMM[x], 1)?;
                    v.width = 32;
                    combine(v, a, x, b)?;
                    src = a;
                }
                if v.width == 32 {
                    v.a.vextractf128(XMM[b], YMM[src], 1)?;
                    v.width = 16;
                    combine(v, a, src, b)?;
                } else {
                    v.mov(a, src)?;
                }
                let mut shift = 8;
                while shift >= n {
                    v.mov(b, a)?;
                    v.a.psrldq(XMM[b], shift as i32)?;
                    combine(v, a, a, b)?;
                    shift /= 2;
                }
                if n == 8 {
                    v.a.movq(emit::R64[d], XMM[a])
                } else {
                    v.a.movd(emit::R32[d], XMM[a])?;
                    if n < 4 {
                        v.a.and(emit::R32[d], (1i32 << (n * 8)) - 1)?;
                    }
                    Ok(())
                }
            }
        }
    }
}
fn upper(v: &mut Vector, d: usize, x: usize) -> Result {
    match v.width {
        16 => {
            v.mov(d, x)?;
            v.a.psrldq(XMM[d], 8)
        }
        32 => v.a.vextractf128(XMM[d], YMM[x], 1),
        _ => v.a.vextracti64x4(YMM[d], ZMM[x], 1),
    }
}
fn join(v: &mut Vector, d: usize, low: usize, high: usize) -> Result {
    match v.width {
        16 => {
            v.mov(d, low)?;
            v.a.movlhps(XMM[d], XMM[high])
        }
        32 => v.a.vinsertf128(YMM[d], YMM[low], XMM[high], 1),
        _ => v.a.vinserti64x4(ZMM[d], ZMM[low], YMM[high], 1),
    }
}
// Half conversions have a signed canonical NaN in the IR. Hardware conversions can retain the payload.
fn canonical_f32(v: &mut Vector, d: usize, x: usize, tmp: &[usize]) -> Result {
    let (m, c, b) = (tmp[0], tmp[1], tmp[2]);
    float::cmp(v, 4, m, x, x, 3)?;
    v.splat(c, 4, 0x8000_0000)?;
    v.and(c, c, x)?;
    v.splat(b, 4, 0x7fc0_0000)?;
    v.or(c, c, b)?;
    integer::select_bits(v, d, m, c, x, b)
}
fn float_width(
    v: &mut Vector,
    action: Action,
    input: Type,
    output: Type,
    r: &[usize],
    t: &[usize],
) -> Result {
    let x = r[0];
    let d = *r.last().unwrap();
    let from = input.lane().unwrap();
    let to = output.lane().unwrap();
    if let Action::Widen(high) = action {
        let s = if high {
            upper(v, t[0], x)?;
            t[0]
        } else {
            x
        };
        match (from, v.width) {
            (Lane::F32, 16) => return v.a.cvtps2pd(XMM[d], XMM[s]),
            (Lane::F32, 32) => return v.a.vcvtps2pd(YMM[d], XMM[s]),
            (Lane::F32, _) => return v.a.vcvtps2pd(ZMM[d], YMM[s]),
            (Lane::F16, 16) => v.a.vcvtph2ps(XMM[d], XMM[s])?,
            (Lane::F16, 32) => v.a.vcvtph2ps(YMM[d], XMM[s])?,
            (Lane::F16, _) => v.a.vcvtph2ps(ZMM[d], YMM[s])?,
            (_, 16) => {
                v.zero(d)?;
                v.a.punpcklwd(XMM[d], XMM[s])?;
            }
            (_, 32) => {
                v.a.vpmovzxwd(YMM[d], XMM[s])?;
                v.a.vpslld(YMM[d], YMM[d], 16)?;
            }
            (_, _) => {
                v.a.vpmovzxwd(ZMM[d], YMM[s])?;
                v.a.vpslld(ZMM[d], ZMM[d], 16)?;
            }
        }
        return canonical_f32(v, d, d, &t[1..4]);
    }
    let y = r[1];
    let (a, b) = (t[0], t[1]);
    for (src, out) in [(x, a), (y, b)] {
        match to {
            Lane::F32 => match v.width {
                16 => v.a.cvtpd2ps(XMM[out], XMM[src]),
                32 => v.a.vcvtpd2ps(XMM[out], YMM[src]),
                _ => v.a.vcvtpd2ps(YMM[out], ZMM[src]),
            }?,
            Lane::F16 => {
                canonical_f32(v, out, src, &t[2..5])?;
                match v.width {
                    16 => v.a.vcvtps2ph(XMM[out], XMM[out], 0),
                    32 => v.a.vcvtps2ph(XMM[out], YMM[out], 0),
                    _ => v.a.vcvtps2ph(YMM[out], ZMM[out], 0),
                }?;
            }
            _ => {
                // Round the 16 discarded bits, including subnormals. VCVTNEPS2BF16 flushes them.
                canonical_f32(v, out, src, &t[2..5])?;
                let (bias, lsb) = (t[2], t[3]);
                immediate!(v, lsb, out, 16, psrld, vpsrld)?;
                v.splat(bias, 4, 1)?;
                v.and(lsb, lsb, bias)?;
                v.splat(bias, 4, 0x7fff)?;
                integer::add(v, 4, out, out, bias)?;
                integer::add(v, 4, out, out, lsb)?;
                immediate!(v, out, out, 16, psrld, vpsrld)?;
            }
        }
    }
    if to == Lane::BF16 {
        Rule::Narrow.emit(
            v,
            Action::Narrow,
            Type::Vector(Lane::U32),
            Type::Vector(Lane::U16),
            &[a, b, d],
            &t[2..4],
        )
    } else {
        join(v, d, a, b)
    }
}
