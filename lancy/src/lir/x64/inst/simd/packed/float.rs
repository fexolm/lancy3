use super::*;
#[derive(Clone, Copy, Debug)]
pub(in crate::lir::x64) enum Rule {
    Sign(Unary),
    CopySign,
    Class(Unary),
    Compare(i32, bool),
    Convert,
    Recip,
    Fract,
    Round,
    MinMax(B),
}

impl Rule {
    pub(super) fn temps(self, input: Type) -> usize {
        match self {
            Self::Sign(_) | Self::Recip | Self::Fract | Self::Compare(..) => 0,
            Self::CopySign => 1,
            Self::MinMax(_) => 2,
            Self::Convert if !input.lane().unwrap().float() => 0,
            Self::Class(_) | Self::Convert => 2,
            Self::Round => 3,
        }
    }
    pub(super) fn emit(
        self,
        v: &mut Vector,
        input: Type,
        output: Type,
        r: &[usize],
        t: &[usize],
    ) -> Result {
        let (x, y, d) = (r[0], r[1], *r.last().unwrap());
        let e = input.lane().unwrap();
        let n = e.bytes();
        let sign = 1u64 << (8 * n - 1);
        let fp = |a: f64| {
            if n == 4 {
                u64::from((a as f32).to_bits())
            } else {
                a.to_bits()
            }
        };
        macro_rules! evex_convert {
            ($op:ident) => {
                match v.width {
                    16 => v.a.$op(XMM[d], XMM[x]),
                    32 => v.a.$op(YMM[d], YMM[x]),
                    _ => v.a.$op(ZMM[d], ZMM[x]),
                }
            };
        }
        match self {
            Self::Sign(u) => {
                v.splat(d, n, sign)?;
                if u == Unary::Neg {
                    v.xor(d, d, x)
                } else {
                    v.andnot(d, d, x)
                }
            }
            Self::CopySign => {
                v.splat(d, n, sign)?;
                integer::select_bits(v, d, d, y, x, t[0])
            }
            Self::Compare(p, swap) => {
                let (x, y) = if swap { (y, x) } else { (x, y) };
                cmp(v, n, d, x, y, p)
            }
            Self::Class(op) => {
                if op == Unary::SignBit {
                    return match v.width {
                        64 => {
                            v.sign_mask(n, x)?;
                            v.materialize_mask(d, n)
                        }
                        32 => {
                            v.zero(d)?;
                            v.ones(t[0])?;
                            if n == 4 {
                                v.a.vblendvps(YMM[d], YMM[d], YMM[t[0]], YMM[x])
                            } else {
                                v.a.vblendvpd(YMM[d], YMM[d], YMM[t[0]], YMM[x])
                            }
                        }
                        _ => {
                            if n == 4 {
                                v.mov(d, x)?;
                            } else {
                                v.a.pshufd(XMM[d], XMM[x], 0xf5)?;
                            }
                            v.a.psrad(XMM[d], 31)
                        }
                    };
                }
                if v.width == 64 && v.cpu.isa.dq {
                    let (imm, invert) = match op {
                        Unary::IsNan => (0x81, false),
                        Unary::IsInfinite => (0x18, false),
                        Unary::IsSubnormal => (0x20, false),
                        Unary::IsFinite => (0x99, true),
                        _ => (0xbf, true),
                    };
                    if n == 4 {
                        v.a.vfpclassps(k1, ZMM[x], imm)?;
                    } else {
                        v.a.vfpclasspd(k1, ZMM[x], imm)?;
                    }
                    if invert {
                        v.a.knotw(k1, k1)?;
                    }
                    return v.materialize_mask(d, n);
                }
                if op == Unary::IsNan {
                    return cmp(v, n, d, x, x, 3);
                }
                let a = t[0];
                let b = t[1];
                v.splat(a, n, sign)?;
                v.andnot(a, a, x)?;
                let threshold = if n == 4 {
                    f32::MIN_POSITIVE as f64
                } else {
                    f64::MIN_POSITIVE
                };
                if op == Unary::IsInfinite || op == Unary::IsFinite {
                    v.splat(b, n, fp(f64::INFINITY))?;
                    return cmp(v, n, d, a, b, if op == Unary::IsFinite { 1 } else { 0 });
                }
                v.splat(b, n, fp(threshold))?;
                if op == Unary::IsNormal {
                    cmp(v, n, d, b, a, 2)?;
                    v.splat(b, n, fp(f64::INFINITY))?;
                    cmp(v, n, b, a, b, 1)?;
                } else {
                    cmp(v, n, d, a, b, 1)?;
                    v.zero(b)?;
                    cmp(v, n, b, b, a, 1)?;
                }
                v.and(d, d, b)
            }
            Self::Convert if !e.float() => match (n, e.signed(), v.width) {
                (4, true, 16) => v.a.cvtdq2ps(XMM[d], XMM[x]),
                (4, true, 32) => v.a.vcvtdq2ps(YMM[d], YMM[x]),
                (4, true, _) => v.a.vcvtdq2ps(ZMM[d], ZMM[x]),
                (4, false, _) => evex_convert!(vcvtudq2ps),
                (_, true, _) => evex_convert!(vcvtqq2pd),
                (_, false, _) => evex_convert!(vcvtuqq2pd),
            },
            Self::Convert => {
                let signed = output.lane().unwrap().signed();
                match (n, signed, v.width) {
                    (4, true, 16) => v.a.cvttps2dq(XMM[d], XMM[x]),
                    (4, true, 32) => v.a.vcvttps2dq(YMM[d], YMM[x]),
                    (4, true, _) => v.a.vcvttps2dq(ZMM[d], ZMM[x]),
                    (4, false, _) => evex_convert!(vcvttps2udq),
                    (_, true, _) => evex_convert!(vcvttpd2qq),
                    (_, false, _) => evex_convert!(vcvttpd2uqq),
                }?;
                if signed {
                    // The indefinite signed result is MIN, which needs flipping on positive overflow.
                    v.splat(t[0], n, fp((1u64 << (8 * n - 1)) as f64))?;
                    cmp(v, n, t[0], t[0], x, 2)?;
                    v.xor(d, d, t[0])?;
                    cmp(v, n, t[1], x, x, 7)?;
                } else {
                    // Unsigned hardware returns MAX for NaN and negative overflow as well.
                    v.zero(t[1])?;
                    cmp(v, n, t[1], t[1], x, 2)?;
                }
                v.and(d, d, t[1])
            }
            Self::Recip => {
                v.splat(d, n, fp(1.0))?;
                if n == 4 {
                    binary!(v, d, d, x, divps, vdivps, vdivps)
                } else {
                    binary!(v, d, d, x, divpd, vdivpd, vdivpd)
                }
            }
            Self::Fract => {
                trunc(v, n, d, x)?;
                subtract(v, n, d, x, d)
            }
            Self::Round => {
                // Comparing the actual fractional distance avoids rounding abs(x)+0.5
                // up at the representable number immediately below a half-integer.
                let (a, b, c) = (t[0], t[1], t[2]);
                trunc(v, n, d, x)?;
                subtract(v, n, a, x, d)?;
                v.splat(b, n, sign)?;
                v.andnot(a, b, a)?;
                v.splat(c, n, fp(0.5))?;
                cmp(v, n, a, c, a, 2)?;
                v.and(b, b, x)?;
                v.splat(c, n, fp(1.0))?;
                v.or(b, b, c)?;
                if n == 4 {
                    binary!(v, b, d, b, addps, vaddps, vaddps)?;
                } else {
                    binary!(v, b, d, b, addpd, vaddpd, vaddpd)?;
                }
                integer::select_bits(v, d, a, b, d, c)
            }
            Self::MinMax(op) => {
                // MINPS/MAXPS return the second operand for NaN or a tie.
                // Equal operands combine their sign bits, so min(-0, +0) is
                // -0 and max(-0, +0) is +0 in either order; then a NaN second
                // operand is replaced by the first.
                let (t, t1) = (t[0], t[1]);
                match (n, op) {
                    (4, B::Min) => binary!(v, d, x, y, minps, vminps, vminps),
                    (4, _) => binary!(v, d, x, y, maxps, vmaxps, vmaxps),
                    (_, B::Min) => binary!(v, d, x, y, minpd, vminpd, vminpd),
                    _ => binary!(v, d, x, y, maxpd, vmaxpd, vmaxpd),
                }?;
                if op == B::Min {
                    cmp(v, n, t, x, y, 0)?; // EQ_OQ
                    v.and(t, t, x)?;
                    v.or(d, d, t)?;
                } else {
                    cmp(v, n, t, x, y, 4)?; // NEQ_UQ
                    v.or(t, t, x)?;
                    v.and(d, d, t)?;
                }
                cmp(v, n, t, y, y, 3)?;
                integer::select_bits(v, d, t, x, d, t1)
            }
        }
    }
}
fn subtract(v: &mut Vector, n: usize, d: usize, x: usize, y: usize) -> Result {
    if n == 4 {
        binary!(v, d, x, y, subps, vsubps, vsubps)
    } else {
        binary!(v, d, x, y, subpd, vsubpd, vsubpd)
    }
}
fn trunc(v: &mut Vector, n: usize, d: usize, x: usize) -> Result {
    match (v.width, n) {
        (16, 4) => v.a.roundps(XMM[d], XMM[x], 3),
        (16, _) => v.a.roundpd(XMM[d], XMM[x], 3),
        (32, 4) => v.a.vroundps(YMM[d], YMM[x], 3),
        (32, _) => v.a.vroundpd(YMM[d], YMM[x], 3),
        (_, 4) => v.a.vrndscaleps(ZMM[d], ZMM[x], 3),
        _ => v.a.vrndscalepd(ZMM[d], ZMM[x], 3),
    }
}
pub(super) fn cmp(v: &mut Vector, n: usize, d: usize, x: usize, y: usize, p: i32) -> Result {
    match (v.width, n) {
        (16, _) => {
            let s = if d == y && x != y {
                crate::lir::x64::inst::SCRATCH
            } else {
                d
            };
            v.mov(s, x)?;
            if n == 4 {
                v.a.cmpps(XMM[s], XMM[y], p)?;
            } else {
                v.a.cmppd(XMM[s], XMM[y], p)?;
            }
            v.mov(d, s)
        }
        (32, 4) => v.a.vcmpps(YMM[d], YMM[x], YMM[y], p),
        (32, _) => v.a.vcmppd(YMM[d], YMM[x], YMM[y], p),
        (_, 4) => {
            v.a.vcmpps(k1, ZMM[x], ZMM[y], p)?;
            v.materialize_mask(d, 4)
        }
        _ => {
            v.a.vcmppd(k1, ZMM[x], ZMM[y], p)?;
            v.materialize_mask(d, 8)
        }
    }
}
