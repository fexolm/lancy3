//! Native arithmetic and unaligned moves for each physical SIMD width.
pub(in crate::lir::x64) mod lanewise;
pub(in crate::lir::x64) mod opmask;
pub(in crate::lir::x64) mod packed;
use super::VectorMask;
use crate::lir::x64::{Cpu, emit};
use crate::{
    hir::{Binary as B, Lane, Ternary, Type, Unary},
    support::numeric::Action,
};
use iced_x86::{IcedError, code_asm::*};

pub(in crate::lir::x64) const XMM: [AsmRegisterXmm; 16] = [
    xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14,
    xmm15,
];
pub(in crate::lir::x64) const YMM: [AsmRegisterYmm; 16] = [
    ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14,
    ymm15,
];
pub(in crate::lir::x64) const ZMM: [AsmRegisterZmm; 16] = [
    zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14,
    zmm15,
];
pub(in crate::lir::x64) const K: [AsmRegisterK; 8] = [k0, k1, k2, k3, k4, k5, k6, k7];

macro_rules! predicated {
    ($operand:expr, $mask:expr) => {
        match $mask {
            1 => $operand.k1(),
            2 => $operand.k2(),
            3 => $operand.k3(),
            4 => $operand.k4(),
            5 => $operand.k5(),
            6 => $operand.k6(),
            7 => $operand.k7(),
            _ => unreachable!("EVEX needs a nonzero mask register"),
        }
    };
}

pub(in crate::lir::x64) fn evex(cpu: Cpu) -> bool {
    cpu.bytes == 64 || cpu.isa.avx10 >= 1
}

pub(in crate::lir::x64) fn mask_lanes(cpu: Cpu, lane: Lane) -> bool {
    match lane.bytes() {
        1 | 2 => cpu.isa.bw,
        4 | 8 => true,
        _ => false,
    }
}

pub(in crate::lir::x64) fn compact_lanes(cpu: Cpu, lane: Lane) -> bool {
    evex(cpu) && mask_lanes(cpu, lane) && (lane.bytes() >= 4 || cpu.isa.vbmi2)
}

#[derive(Clone, Copy, Debug)]
pub(in crate::lir::x64) struct Native {
    pub action: Action,
    pub input: Type,
    pub output: Type,
    pub(in crate::lir::x64) bytes: usize,
    pub(in crate::lir::x64) cpu: Cpu,
    pub(in crate::lir::x64) recipe: Option<packed::Recipe>,
}

impl Native {
    pub fn temporaries(self) -> usize {
        self.recipe.map_or(0, |r| {
            r.temporaries(self.action, self.input, self.output, self.cpu)
        })
    }
    pub fn separate_output(self) -> bool {
        self.recipe.is_some()
    }
    pub fn tied(self) -> bool {
        if self.recipe.is_some() {
            return false;
        }
        matches!(self.action, Action::Ternary(Ternary::MulAdd))
            || (self.bytes <= 16
                && self.input != Type::Vector(Lane::F16)
                && matches!(self.action, Action::Binary(_)))
    }

    pub fn emit_allocated(
        self,
        a: &mut CodeAssembler,
        regs: &[usize],
        temp: &[usize],
        pool: &mut emit::Constants,
    ) -> Result<(), IcedError> {
        if let Some(recipe) = self.recipe {
            return recipe.emit(a, self, regs, temp, pool);
        }

        let src = regs[0];
        let rhs = regs[1];
        let dst = *regs.last().unwrap();
        let lane = self.input.lane().unwrap();
        let scratch = super::SCRATCH;
        match self.action {
            Action::AnyTrue => {
                test_any(a, self.cpu, src)?;
                a.mov(emit::R32[dst], 0)?;
                a.setne(emit::R8[dst])?;
                return a.neg(emit::R32[dst]);
            }
            Action::AllTrue => {
                test_all(a, self.cpu, lane, src)?;
                let g = emit::R32[dst];
                a.mov(g, 0)?;
                a.sete(emit::R8[dst])?;
                return a.neg(g);
            }
            Action::Unary(Unary::Not) => {
                return match self.bytes {
                    64 => a.vpternlogd(ZMM[dst], ZMM[src], ZMM[src], 0x33),
                    32 => {
                        a.vpcmpeqd(YMM[scratch], YMM[scratch], YMM[scratch])?;
                        a.vpxor(YMM[dst], YMM[src], YMM[scratch])
                    }
                    _ => {
                        a.pcmpeqd(XMM[scratch], XMM[scratch])?;
                        if dst != src {
                            a.movaps(XMM[dst], XMM[src])?;
                        }
                        a.pxor(XMM[dst], XMM[scratch])
                    }
                };
            }
            Action::Ternary(Ternary::Select) => {
                let no = regs[2];
                if self.bytes == 16 {
                    // Canonicalize the high bit of each SSE2 lane into a full mask.
                    a.pxor(XMM[scratch], XMM[scratch])?;
                    match lane.bytes() {
                        1 => a.pcmpgtb(XMM[scratch], XMM[src])?,
                        2 => a.pcmpgtw(XMM[scratch], XMM[src])?,
                        4 => a.pcmpgtd(XMM[scratch], XMM[src])?,
                        _ => {
                            a.pcmpgtd(XMM[scratch], XMM[src])?;
                            a.pshufd(XMM[scratch], XMM[scratch], 0xf5)?;
                        }
                    }
                    if rhs == no {
                        return a.movaps(XMM[dst], XMM[no]);
                    }
                    if dst == no {
                        a.pxor(XMM[dst], XMM[rhs])?;
                        a.pandn(XMM[scratch], XMM[dst])?;
                        a.movaps(XMM[dst], XMM[scratch])?;
                        return a.pxor(XMM[dst], XMM[rhs]);
                    }
                    if dst != rhs {
                        a.movaps(XMM[dst], XMM[rhs])?;
                    }
                    a.pxor(XMM[dst], XMM[no])?;
                    a.pand(XMM[dst], XMM[scratch])?;
                    return a.pxor(XMM[dst], XMM[no]);
                }
                if self.bytes == 32 {
                    let (d, n, y, m) = (YMM[dst], YMM[no], YMM[rhs], YMM[src]);
                    return match lane.bytes() {
                        1 => a.vpblendvb(d, n, y, m),
                        2 => {
                            a.vpsraw(YMM[scratch], m, 15)?;
                            a.vpblendvb(d, n, y, YMM[scratch])
                        }
                        4 => a.vblendvps(d, n, y, m),
                        _ => a.vblendvpd(d, n, y, m),
                    };
                }
                mask_from_signs(a, self.cpu, lane.bytes(), src)?;
                return blend(a, self.bytes, lane, [dst, rhs, no], 1);
            }
            _ => {}
        }
        if lane == Lane::F16 {
            macro_rules! ph {
                ($op:ident,$($r:ident),+) => {
                    match self.bytes {
                        16 => a.$op($(XMM[$r]),+),
                        32 => a.$op($(YMM[$r]),+),
                        _ => a.$op($(ZMM[$r]),+),
                    }
                };
            }
            return match self.action {
                Action::Binary(B::Add) => ph!(vaddph, dst, src, rhs),
                Action::Binary(B::Sub) => ph!(vsubph, dst, src, rhs),
                Action::Binary(B::Mul) => ph!(vmulph, dst, src, rhs),
                Action::Binary(B::Div) => ph!(vdivph, dst, src, rhs),
                Action::Unary(Unary::Sqrt) => ph!(vsqrtph, dst, src),
                Action::Ternary(Ternary::MulAdd) => {
                    let c = regs[2];
                    ph!(vfmadd213ph, dst, rhs, c)
                }
                _ => unreachable!("selected FP16 operation"),
            };
        }
        if let Action::Binary(op) = self.action
            && let Some(form) = lanewise::Lanewise::of(op, lane)
        {
            return form.emit(a, self.bytes, [dst, src, rhs]);
        }
        match self.action {
            Action::Splat => {
                // The scalar lives either in a GPR or in the low lane of an XMM.
                if !lane.float() && self.bytes == 64 {
                    return match lane.bytes() {
                        1 => a.vpbroadcastb(ZMM[dst], emit::R32[src]),
                        2 => a.vpbroadcastw(ZMM[dst], emit::R32[src]),
                        4 => a.vpbroadcastd(ZMM[dst], emit::R32[src]),
                        _ => a.vpbroadcastq(ZMM[dst], emit::R64[src]),
                    };
                }
                if !lane.float() {
                    if lane.bytes() == 8 {
                        a.movq(XMM[dst], emit::R64[src])?;
                    } else {
                        a.movd(XMM[dst], emit::R32[src])?;
                    }
                }
                let x = if lane.float() { XMM[src] } else { XMM[dst] };
                match (self.bytes, lane.bytes()) {
                    (16, 1) => {
                        a.punpcklbw(XMM[dst], x)?;
                        a.pshuflw(XMM[dst], XMM[dst], 0)?;
                        a.pshufd(XMM[dst], XMM[dst], 0)
                    }
                    (16, 2) => {
                        a.pshuflw(XMM[dst], x, 0)?;
                        a.pshufd(XMM[dst], XMM[dst], 0)
                    }
                    (16, 4) => a.pshufd(XMM[dst], x, 0),
                    (16, 8) => a.pshufd(XMM[dst], x, 0x44),
                    (32, 1) => a.vpbroadcastb(YMM[dst], x),
                    (32, 2) => a.vpbroadcastw(YMM[dst], x),
                    (32, 4) => a.vpbroadcastd(YMM[dst], x),
                    (32, 8) => a.vpbroadcastq(YMM[dst], x),
                    (64, 1) => a.vpbroadcastb(ZMM[dst], x),
                    (64, 2) => a.vpbroadcastw(ZMM[dst], x),
                    (64, 4) => a.vpbroadcastd(ZMM[dst], x),
                    (64, 8) => a.vpbroadcastq(ZMM[dst], x),
                    _ => unreachable!(),
                }
            }
            Action::Compress | Action::Expand => {
                mask_from_signs(a, self.cpu, lane.bytes(), rhs)?;
                compact(
                    a,
                    self.bytes,
                    lane,
                    matches!(self.action, Action::Expand),
                    dst,
                    src,
                    1,
                )
            }
            Action::Unary(Unary::Abs) => {
                macro_rules! abs {
                    ($sse:ident,$avx:ident) => {
                        match self.bytes {
                            16 => a.$sse(XMM[dst], XMM[src]),
                            32 => a.$avx(YMM[dst], YMM[src]),
                            _ => a.$avx(ZMM[dst], ZMM[src]),
                        }
                    };
                }
                match lane.bytes() {
                    1 => abs!(pabsb, vpabsb),
                    2 => abs!(pabsw, vpabsw),
                    4 => abs!(pabsd, vpabsd),
                    _ => a.vpabsq(ZMM[dst], ZMM[src]),
                }
            }
            Action::Binary(op @ (B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge))
                if !lane.float() =>
            {
                if self.bytes == 64 {
                    compare_mask(
                        a,
                        64,
                        k1,
                        lane,
                        src,
                        rhs,
                        comparison_predicate(lane, op).unwrap(),
                    )?;
                    return match lane.bytes() {
                        1 => a.vpmovm2b(ZMM[dst], k1),
                        2 => a.vpmovm2w(ZMM[dst], k1),
                        4 => a.vpternlogd(ZMM[dst].k1().z(), ZMM[dst], ZMM[dst], 255),
                        _ => a.vpternlogq(ZMM[dst].k1().z(), ZMM[dst], ZMM[dst], 255),
                    };
                }
                macro_rules! compare {
                    ($sse:ident,$avx:ident) => {
                        if self.bytes == 16 {
                            a.$sse(XMM[dst], XMM[rhs])
                        } else {
                            a.$avx(YMM[dst], YMM[src], YMM[rhs])
                        }
                    };
                }
                match (op, lane.bytes()) {
                    (B::Eq, 1) => compare!(pcmpeqb, vpcmpeqb),
                    (B::Eq, 2) => compare!(pcmpeqw, vpcmpeqw),
                    (B::Eq, 4) => compare!(pcmpeqd, vpcmpeqd),
                    (B::Eq, 8) => compare!(pcmpeqq, vpcmpeqq),
                    (_, 1) => compare!(pcmpgtb, vpcmpgtb),
                    (_, 2) => compare!(pcmpgtw, vpcmpgtw),
                    (_, 4) => compare!(pcmpgtd, vpcmpgtd),
                    _ => compare!(pcmpgtq, vpcmpgtq),
                }
            }
            Action::Unary(Unary::Popcnt) => match lane.bytes() {
                1 => a.vpopcntb(ZMM[dst], ZMM[src]),
                2 => a.vpopcntw(ZMM[dst], ZMM[src]),
                4 => a.vpopcntd(ZMM[dst], ZMM[src]),
                _ => a.vpopcntq(ZMM[dst], ZMM[src]),
            },
            Action::Unary(Unary::LeadingZeros) => {
                if lane.bytes() == 4 {
                    a.vplzcntd(ZMM[dst], ZMM[src])
                } else {
                    a.vplzcntq(ZMM[dst], ZMM[src])
                }
            }
            Action::Unary(op @ (Unary::Floor | Unary::Ceil | Unary::Trunc | Unary::Nearest)) => {
                let mode = match op {
                    Unary::Floor => 1,
                    Unary::Ceil => 2,
                    Unary::Trunc => 3,
                    _ => 0,
                };
                match (lane, self.bytes) {
                    (Lane::F32, 4) => a.roundss(XMM[dst], XMM[src], mode),
                    (_, 8) => a.roundsd(XMM[dst], XMM[src], mode),
                    (Lane::F32, 16) => a.roundps(XMM[dst], XMM[src], mode),
                    (_, 16) => a.roundpd(XMM[dst], XMM[src], mode),
                    (Lane::F32, 32) => a.vroundps(YMM[dst], YMM[src], mode),
                    (_, 32) => a.vroundpd(YMM[dst], YMM[src], mode),
                    (Lane::F32, 64) => a.vrndscaleps(ZMM[dst], ZMM[src], mode),
                    _ => a.vrndscalepd(ZMM[dst], ZMM[src], mode),
                }
            }
            Action::Binary(op @ (B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge)) => {
                let cc = comparison_predicate(lane, op).unwrap();
                match (lane, self.bytes) {
                    (Lane::F32, 16) => a.cmpps(XMM[dst], XMM[rhs], cc),
                    (_, 16) => a.cmppd(XMM[dst], XMM[rhs], cc),
                    (Lane::F32, 32) => a.vcmpps(YMM[dst], YMM[src], YMM[rhs], cc),
                    (_, 32) => a.vcmppd(YMM[dst], YMM[src], YMM[rhs], cc),
                    (Lane::F32, 64) => {
                        a.vcmpps(k1, ZMM[src], ZMM[rhs], cc)?;
                        a.vpternlogd(ZMM[dst].k1().z(), ZMM[dst], ZMM[dst], 255)
                    }
                    (_, 64) => {
                        a.vcmppd(k1, ZMM[src], ZMM[rhs], cc)?;
                        a.vpternlogq(ZMM[dst].k1().z(), ZMM[dst], ZMM[dst], 255)
                    }
                    _ => unreachable!(),
                }
            }
            Action::Cast => {
                if !matches!(self.input, Type::Vector(_)) {
                    let from = lane.bytes();
                    let to = self.output.lane().unwrap().bytes();
                    let (q, d) = (emit::R64[dst], emit::R32[dst]);
                    return match (from.min(to), to > from && lane.signed()) {
                        (1, true) => a.movsx(q, emit::R8[src]),
                        (2, true) => a.movsx(q, emit::R16[src]),
                        (4, true) => a.movsxd(q, emit::R32[src]),
                        (1, false) => a.movzx(d, emit::R8[src]),
                        (2, false) => a.movzx(d, emit::R16[src]),
                        (4, false) => a.mov(d, emit::R32[src]),
                        _ => a.mov(q, emit::R64[src]),
                    };
                }
                // Intel AVX10.2, VCVTT{PS,PD}2{U,}{D,Q}QS. iced 1.22 predates MAP5.
                // Allocatable SIMD registers are 0..15, so R' and X' stay inverted.
                let p0 = 0xf5 ^ (((dst >> 3) as u8) << 7) ^ (((src >> 3) as u8) << 5);
                let p1 = if lane == Lane::F32 { 0x7c } else { 0xfd };
                let p2 = match self.bytes {
                    16 => 0x08,
                    32 => 0x28,
                    _ => 0x48,
                };
                let opcode = if self.output.lane().unwrap().signed() {
                    0x6d
                } else {
                    0x6c
                };
                a.db(&[
                    0x62,
                    p0,
                    p1,
                    p2,
                    opcode,
                    0xc0 | (dst as u8 & 7) << 3 | (src as u8 & 7),
                ])
            }
            Action::Unary(Unary::Sqrt) => match (lane, self.bytes) {
                (Lane::F32, 4) => a.sqrtss(XMM[dst], XMM[src]),
                (Lane::F64, 8) => a.sqrtsd(XMM[dst], XMM[src]),
                (Lane::F32, 16) => a.sqrtps(XMM[dst], XMM[src]),
                (_, 16) => a.sqrtpd(XMM[dst], XMM[src]),
                (Lane::F32, 32) => a.vsqrtps(YMM[dst], YMM[src]),
                (_, 32) => a.vsqrtpd(YMM[dst], YMM[src]),
                (Lane::F32, 64) => a.vsqrtps(ZMM[dst], ZMM[src]),
                _ => a.vsqrtpd(ZMM[dst], ZMM[src]),
            },
            Action::Ternary(Ternary::MulAdd) => {
                let c = regs[2];
                match (lane, self.bytes) {
                    (Lane::F32, 4) => a.vfmadd213ss(XMM[dst], XMM[rhs], XMM[c]),
                    (Lane::F64, 8) => a.vfmadd213sd(XMM[dst], XMM[rhs], XMM[c]),
                    (Lane::F32, 16) => a.vfmadd213ps(XMM[dst], XMM[rhs], XMM[c]),
                    (_, 16) => a.vfmadd213pd(XMM[dst], XMM[rhs], XMM[c]),
                    (Lane::F32, 32) => a.vfmadd213ps(YMM[dst], YMM[rhs], YMM[c]),
                    (_, 32) => a.vfmadd213pd(YMM[dst], YMM[rhs], YMM[c]),
                    (Lane::F32, 64) => a.vfmadd213ps(ZMM[dst], ZMM[rhs], ZMM[c]),
                    _ => a.vfmadd213pd(ZMM[dst], ZMM[rhs], ZMM[c]),
                }
            }
            _ => unreachable!("selected native operation"),
        }
    }
}

pub(in crate::lir::x64) fn load(
    a: &mut CodeAssembler,
    dst: usize,
    mem: AsmMemoryOperand,
    bytes: usize,
) -> Result<(), IcedError> {
    match bytes {
        4 => a.movd(XMM[dst], dword_ptr(mem)),
        8 => a.movq(XMM[dst], qword_ptr(mem)),
        16 => a.movdqu(XMM[dst], xmmword_ptr(mem)),
        32 => a.vmovdqu(YMM[dst], ymmword_ptr(mem)),
        64 => a.vmovdqu64(ZMM[dst], zmmword_ptr(mem)),
        _ => unreachable!(),
    }
}
pub(in crate::lir::x64) fn store(
    a: &mut CodeAssembler,
    mem: AsmMemoryOperand,
    src: usize,
    bytes: usize,
) -> Result<(), IcedError> {
    match bytes {
        4 => a.movd(dword_ptr(mem), XMM[src]),
        8 => a.movq(qword_ptr(mem), XMM[src]),
        16 => a.movdqu(xmmword_ptr(mem), XMM[src]),
        32 => a.vmovdqu(ymmword_ptr(mem), YMM[src]),
        64 => a.vmovdqu64(zmmword_ptr(mem), ZMM[src]),
        _ => unreachable!(),
    }
}
pub(in crate::lir::x64) fn copy(
    a: &mut CodeAssembler,
    d: usize,
    s: usize,
    bytes: usize,
) -> Result<(), IcedError> {
    match bytes {
        16 => a.movaps(XMM[d], XMM[s]),
        32 => a.vmovaps(YMM[d], YMM[s]),
        64 => a.vmovaps(ZMM[d], ZMM[s]),
        _ => unreachable!(),
    }
}

/// Sets ZF when every lane is nonzero.
pub(in crate::lir::x64) fn test_all(
    a: &mut CodeAssembler,
    cpu: Cpu,
    lane: Lane,
    src: usize,
) -> Result<(), IcedError> {
    match cpu.bytes {
        16 => {
            let t = XMM[super::SCRATCH];
            a.pxor(t, t)?;
            match lane.bytes() {
                1 => a.pcmpeqb(t, XMM[src])?,
                2 => a.pcmpeqw(t, XMM[src])?,
                4 => a.pcmpeqd(t, XMM[src])?,
                8 => a.pcmpeqq(t, XMM[src])?,
                _ => unreachable!(),
            }
            a.pmovmskb(r10d, t)?;
            a.test(r10d, r10d)
        }
        32 => {
            let t = YMM[super::SCRATCH];
            a.vpxor(t, t, t)?;
            match lane.bytes() {
                1 => a.vpcmpeqb(t, t, YMM[src])?,
                2 => a.vpcmpeqw(t, t, YMM[src])?,
                4 => a.vpcmpeqd(t, t, YMM[src])?,
                8 => a.vpcmpeqq(t, t, YMM[src])?,
                _ => unreachable!(),
            }
            a.vptest(t, t)
        }
        64 => match lane.bytes() {
            1 => {
                a.vptestmb(k1, ZMM[src], ZMM[src])?;
                a.knotq(k1, k1)?;
                a.kortestq(k1, k1)
            }
            2 => {
                a.vptestmw(k1, ZMM[src], ZMM[src])?;
                a.knotd(k1, k1)?;
                a.kortestd(k1, k1)
            }
            4 => {
                a.vptestmd(k1, ZMM[src], ZMM[src])?;
                a.knotw(k1, k1)?;
                a.kortestw(k1, k1)
            }
            8 => {
                a.vptestmq(k1, ZMM[src], ZMM[src])?;
                if cpu.isa.dq {
                    a.knotb(k1, k1)?;
                    a.kortestb(k1, k1)
                } else {
                    a.kmovw(r10d, k1)?;
                    a.cmp(r10d, 255)
                }
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}

/// Sets ZF when every bit of the vector is zero.
pub(in crate::lir::x64) fn test_any(
    a: &mut CodeAssembler,
    cpu: Cpu,
    src: usize,
) -> Result<(), IcedError> {
    match cpu.bytes {
        64 => {
            a.vptestmd(k1, ZMM[src], ZMM[src])?;
            a.kortestw(k1, k1)
        }
        32 => a.vptest(YMM[src], YMM[src]),
        _ if cpu.isa.sse41 => a.ptest(XMM[src], XMM[src]),
        _ => {
            let t = XMM[super::SCRATCH];
            a.pxor(t, t)?;
            a.pcmpeqb(t, XMM[src])?;
            a.pmovmskb(r10d, t)?;
            a.cmp(r10d, 0xffff)
        }
    }
}

// K1 is reserved inside masked pseudo-instructions and never holds a live SSA value.
fn mask_from_signs(
    a: &mut CodeAssembler,
    cpu: Cpu,
    size: usize,
    mask: usize,
) -> Result<(), IcedError> {
    if !cpu.isa.dq && size >= 4 {
        let z = ZMM[super::SCRATCH];
        a.vpxord(z, z, z)?;
        return if size == 4 {
            a.vpcmpd(k1, ZMM[mask], z, 1)
        } else {
            a.vpcmpq(k1, ZMM[mask], z, 1)
        };
    }
    macro_rules! signs {
        ($r:ident) => {
            match size {
                1 => a.vpmovb2m(k1, $r[mask]),
                2 => a.vpmovw2m(k1, $r[mask]),
                4 => a.vpmovd2m(k1, $r[mask]),
                _ => a.vpmovq2m(k1, $r[mask]),
            }
        };
    }
    match cpu.bytes {
        16 => signs!(XMM),
        32 => signs!(YMM),
        _ => signs!(ZMM),
    }
}

pub(in crate::lir::x64) fn comparison_predicate(lane: Lane, op: B) -> Option<i32> {
    Some(match op {
        B::Eq => 0,
        B::Lt => 1,
        B::Le => 2,
        B::Unordered if lane.float() => 3,
        B::Ne => 4,
        B::Ge => {
            if lane.float() {
                29
            } else {
                5
            }
        }
        B::Gt => {
            if lane.float() {
                30
            } else {
                6
            }
        }
        B::Ordered if lane.float() => 7,
        _ => return None,
    })
}

pub(in crate::lir::x64) fn compare_mask(
    a: &mut CodeAssembler,
    width: usize,
    dst: AsmRegisterK,
    lane: Lane,
    lhs: usize,
    rhs: usize,
    cc: i32,
) -> Result<(), IcedError> {
    macro_rules! cmp {
        ($op:ident) => {
            match width {
                16 => a.$op(dst, XMM[lhs], XMM[rhs], cc),
                32 => a.$op(dst, YMM[lhs], YMM[rhs], cc),
                _ => a.$op(dst, ZMM[lhs], ZMM[rhs], cc),
            }
        };
    }
    match (lane, lane.bytes(), lane.signed()) {
        (Lane::F32, ..) => cmp!(vcmpps),
        (Lane::F64, ..) => cmp!(vcmppd),
        (_, 1, true) => cmp!(vpcmpb),
        (_, 1, false) => cmp!(vpcmpub),
        (_, 2, true) => cmp!(vpcmpw),
        (_, 2, false) => cmp!(vpcmpuw),
        (_, 4, true) => cmp!(vpcmpd),
        (_, 4, false) => cmp!(vpcmpud),
        (_, 8, true) => cmp!(vpcmpq),
        (_, 8, false) => cmp!(vpcmpuq),
        _ => unreachable!(),
    }
}

enum MaskBits {
    Sign,
    Nonzero,
}

fn emit_opmask(
    a: &mut CodeAssembler,
    cpu: Cpu,
    size: usize,
    mask: VectorMask<usize>,
    bits: MaskBits,
) -> Result<usize, IcedError> {
    match mask {
        VectorMask::Lanes(reg) => mask_from_signs(a, cpu, size, reg),
        VectorMask::Register(_, reg) => return Ok(reg),
        VectorMask::Compare {
            lhs,
            rhs,
            lane,
            predicate,
            active,
        } => {
            let dst = if let Some(active) = active {
                match bits {
                    MaskBits::Sign => mask_from_signs(a, cpu, size, active)?,
                    MaskBits::Nonzero => {
                        let v = ZMM[active];
                        match size {
                            1 => a.vptestmb(k1, v, v),
                            2 => a.vptestmw(k1, v, v),
                            4 => a.vptestmd(k1, v, v),
                            _ => a.vptestmq(k1, v, v),
                        }?;
                    }
                }
                k1.k1()
            } else {
                k1
            };
            compare_mask(a, cpu.bytes, dst, lane, lhs, rhs, predicate)
        }
    }?;
    Ok(1)
}

pub(in crate::lir::x64) fn test_vector_mask(
    a: &mut CodeAssembler,
    cpu: Cpu,
    mask: VectorMask<usize>,
) -> Result<(), IcedError> {
    let lane = match mask {
        VectorMask::Lanes(reg) => return test_any(a, cpu, reg),
        VectorMask::Register(lane, _) | VectorMask::Compare { lane, .. } => lane,
    };
    let size = lane.bytes();
    let k = K[emit_opmask(a, cpu, size, mask, MaskBits::Nonzero)?];
    match cpu.bytes / size {
        64 => a.kortestq(k, k),
        32 => a.kortestd(k, k),
        _ => a.kortestw(k, k),
    }
}

fn blend(
    a: &mut CodeAssembler,
    width: usize,
    lane: Lane,
    [dst, yes, no]: [usize; 3],
    mask: usize,
) -> Result<(), IcedError> {
    macro_rules! merge {
        ($op:ident) => {
            match width {
                16 => a.$op(predicated!(XMM[dst], mask), XMM[no], XMM[yes]),
                32 => a.$op(predicated!(YMM[dst], mask), YMM[no], YMM[yes]),
                _ => a.$op(predicated!(ZMM[dst], mask), ZMM[no], ZMM[yes]),
            }
        };
    }
    match lane.bytes() {
        1 => merge!(vpblendmb),
        2 => merge!(vpblendmw),
        4 => merge!(vpblendmd),
        _ => merge!(vpblendmq),
    }
}

pub(in crate::lir::x64) fn blend_comparison(
    a: &mut CodeAssembler,
    cpu: Cpu,
    lane: Lane,
    mask: VectorMask<usize>,
    dst: usize,
    yes: usize,
    no: usize,
) -> Result<(), IcedError> {
    let mask = emit_opmask(a, cpu, lane.bytes(), mask, MaskBits::Sign)?;
    blend(a, cpu.bytes, lane, [dst, yes, no], mask)
}

pub(in crate::lir::x64) fn compact(
    a: &mut CodeAssembler,
    width: usize,
    lane: Lane,
    expand: bool,
    dst: usize,
    src: usize,
    mask: usize,
) -> Result<(), IcedError> {
    macro_rules! permute {
        ($op:ident) => {
            match width {
                16 => a.$op(predicated!(XMM[dst], mask).z(), XMM[src]),
                32 => a.$op(predicated!(YMM[dst], mask).z(), YMM[src]),
                _ => a.$op(predicated!(ZMM[dst], mask).z(), ZMM[src]),
            }
        };
    }
    match (expand, lane.bytes()) {
        (false, 1) => permute!(vpcompressb),
        (false, 2) => permute!(vpcompressw),
        (false, 4) => permute!(vpcompressd),
        (false, _) => permute!(vpcompressq),
        (true, 1) => permute!(vpexpandb),
        (true, 2) => permute!(vpexpandw),
        (true, 4) => permute!(vpexpandd),
        (true, _) => permute!(vpexpandq),
    }
}

pub(in crate::lir::x64) fn evex_masked_memory(cpu: Cpu, size: usize, indexed: bool) -> bool {
    evex(cpu) && (matches!(size, 4 | 8) || !indexed && size <= 2 && cpu.isa.bw)
}

/// A masked contiguous or indexed access through `base`: a load when it
/// has `dst` (`value` fills the inactive lanes), otherwise a store of `value`.
#[derive(Clone, Copy, Debug)]
pub(in crate::lir::x64) struct MaskedMemory<R = usize> {
    pub ty: Type,
    pub base: R,
    pub index: Option<R>,
    pub mask: VectorMask<R>,
    pub value: Option<R>,
    pub dst: Option<R>,
    pub temp: Option<R>,
}
impl<R> MaskedMemory<R> {
    pub(in crate::lir::x64) fn map<T>(self, mut reg: impl FnMut(R) -> T) -> MaskedMemory<T> {
        MaskedMemory {
            ty: self.ty,
            base: reg(self.base),
            index: self.index.map(&mut reg),
            mask: self.mask.map(&mut reg),
            value: self.value.map(&mut reg),
            dst: self.dst.map(&mut reg),
            temp: self.temp.map(reg),
        }
    }
}
impl MaskedMemory {
    pub(in crate::lir::x64) fn emit(
        self,
        a: &mut CodeAssembler,
        cpu: Cpu,
    ) -> Result<(), IcedError> {
        use crate::lir::x64::emit::R64;
        let Self {
            ty,
            base,
            index,
            mask,
            value,
            dst,
            temp,
        } = self;
        let width = cpu.bytes;
        let size = ty.lane().unwrap().bytes();
        let scratch = super::SCRATCH;
        // AVX10 also exposes EVEX memory instructions at its supported vector length.
        if evex_masked_memory(cpu, size, index.is_some()) {
            let mask = emit_opmask(a, cpu, size, mask, MaskBits::Sign)?;
            if let Some(i) = index {
                // VSIB clears its mask. K1 is reserved for destructive operands.
                if mask != 1 {
                    a.kmovw(k1, K[mask])?;
                }
                macro_rules! indexed {
                    ($r:ident) => {{
                        let address = ptr(R64[base] + $r[i] * size as i32);
                        if let Some(d) = dst {
                            let target = if d == i { scratch } else { d };
                            init_gather(a, target, value, width)?;
                            if size == 4 {
                                a.vpgatherdd($r[target].k1(), dword_ptr(address))?;
                            } else {
                                a.vpgatherqq($r[target].k1(), qword_ptr(address))?;
                            }
                            if target != d {
                                copy(a, d, target, width)?;
                            }
                            Ok(())
                        // VPSCATTER orders overlapping element writes from low to high lanes.
                        } else if size == 4 {
                            a.vpscatterdd(dword_ptr(address).k1(), $r[value.unwrap()])
                        } else {
                            a.vpscatterqq(qword_ptr(address).k1(), $r[value.unwrap()])
                        }
                    }};
                }
                return match width {
                    16 => indexed!(XMM),
                    32 => indexed!(YMM),
                    _ => indexed!(ZMM),
                };
            }
            if let (Some(d), Some(value)) = (dst, value)
                && d != value
            {
                copy(a, d, value, width)?;
            }
            macro_rules! access {
                ($r:ident,$mem:ident,$op:ident) => {{
                    let mem = $mem(R64[base]);
                    if let Some(d) = dst {
                        let dst = predicated!($r[d], mask);
                        a.$op(if value.is_some() { dst } else { dst.z() }, mem)
                    } else {
                        a.$op(predicated!(mem, mask), $r[value.unwrap()])
                    }
                }};
            }
            macro_rules! contiguous {
                ($r:ident,$mem:ident) => {
                    match size {
                        1 => access!($r, $mem, vmovdqu8),
                        2 => access!($r, $mem, vmovdqu16),
                        4 => access!($r, $mem, vmovdqu32),
                        _ => access!($r, $mem, vmovdqu64),
                    }
                };
            }
            return match width {
                16 => contiguous!(XMM, xmmword_ptr),
                32 => contiguous!(YMM, ymmword_ptr),
                _ => contiguous!(ZMM, zmmword_ptr),
            };
        }
        let VectorMask::Lanes(mask) = mask else {
            unreachable!("comparison masks require 512-bit EVEX memory instructions")
        };
        if width == 32 && matches!(size, 4 | 8) {
            if let (Some(i), Some(d)) = (index, dst) {
                // AVX2 clears enabled mask lanes as the gather completes. Regalloc
                // provides a distinct temporary for that destructive operand.
                let m = temp.expect("AVX2 gather mask temporary");
                let target = if d == i { scratch } else { d };
                a.vmovaps(YMM[m], YMM[mask])?;
                init_gather(a, target, value, width)?;
                let address = ptr(R64[base] + YMM[i] * size as i32);
                if size == 4 {
                    a.vpgatherdd_3(YMM[target], dword_ptr(address), YMM[m])?;
                } else {
                    a.vpgatherqq_3(YMM[target], qword_ptr(address), YMM[m])?;
                }
                return if target == d {
                    Ok(())
                } else {
                    a.vmovaps(YMM[d], YMM[target])
                };
            }
            if index.is_none() {
                let mem = ymmword_ptr(R64[base]);
                if let Some(d) = dst {
                    let Some(value) = value else {
                        return if size == 4 {
                            a.vmaskmovps(YMM[d], YMM[mask], mem)
                        } else {
                            a.vmaskmovpd(YMM[d], YMM[mask], mem)
                        };
                    };
                    if size == 4 {
                        a.vmaskmovps(YMM[scratch], YMM[mask], mem)?;
                        return a.vblendvps(YMM[d], YMM[value], YMM[scratch], YMM[mask]);
                    }
                    a.vmaskmovpd(YMM[scratch], YMM[mask], mem)?;
                    return a.vblendvpd(YMM[d], YMM[value], YMM[scratch], YMM[mask]);
                }
                return if size == 4 {
                    a.vmaskmovps(mem, YMM[mask], YMM[value.unwrap()])
                } else {
                    a.vmaskmovpd(mem, YMM[mask], YMM[value.unwrap()])
                };
            }
        }
        // R10/R11 are reserved. Save the workspace pointer while using R11 as the
        // lane address. No external data access takes place until its mask bit passes.
        let data = width as i32;
        let indices = 2 * data;
        let saved = indices + if index.is_some() { data } else { 0 };
        a.sub(rsp, saved + 8)?;
        a.mov(qword_ptr(rsp + saved), r11)?;
        store(a, ptr(rsp), mask, width)?;
        store(
            a,
            ptr(rsp + data),
            value.expect("scalar masked memory needs a vector"),
            width,
        )?;
        if let Some(index) = index {
            store(a, ptr(rsp + indices), index, width)?;
        }
        for offset in (0..width as i32).step_by(size) {
            let skip = a.create_label();
            a.test(byte_ptr(rsp + offset + size as i32 - 1), 0x80)?;
            a.jz(skip)?;
            if index.is_some() {
                let from = ptr(rsp + indices + offset);
                match size {
                    1 => a.movsx(r10, byte_ptr(from)),
                    2 => a.movsx(r10, word_ptr(from)),
                    4 => a.movsxd(r10, dword_ptr(from)),
                    _ => a.mov(r10, qword_ptr(from)),
                }?;
                a.shl(r10, size.trailing_zeros() as i32)?;
                a.lea(r11, ptr(R64[base] + r10))?;
            } else {
                a.lea(r11, ptr(R64[base] + offset))?;
            }
            let mut transfer = |off: i32, chunk: usize| -> Result<(), IcedError> {
                let external = ptr(r11 + off);
                let local = ptr(rsp + data + offset + off);
                let (from, to) = if dst.is_some() {
                    (external, local)
                } else {
                    (local, external)
                };
                match chunk {
                    1 => {
                        a.mov(r10b, byte_ptr(from))?;
                        a.mov(byte_ptr(to), r10b)
                    }
                    2 => {
                        a.mov(r10w, word_ptr(from))?;
                        a.mov(word_ptr(to), r10w)
                    }
                    4 => {
                        a.mov(r10d, dword_ptr(from))?;
                        a.mov(dword_ptr(to), r10d)
                    }
                    _ => {
                        a.mov(r10, qword_ptr(from))?;
                        a.mov(qword_ptr(to), r10)
                    }
                }
            };
            transfer(0, size.min(8))?;
            if size == 16 {
                transfer(8, 8)?;
            }
            emit::bind(a, skip)?;
        }
        if let Some(dst) = dst {
            load(a, dst, ptr(rsp + data), width)?;
        }
        a.mov(r11, qword_ptr(rsp + saved))?;
        a.add(rsp, saved + 8)
    }
}

fn init_gather(
    a: &mut CodeAssembler,
    target: usize,
    value: Option<usize>,
    width: usize,
) -> Result<(), IcedError> {
    match value {
        Some(value) if value != target => copy(a, target, value, width),
        Some(_) => Ok(()),
        None => match width {
            16 => a.vxorps(XMM[target], XMM[target], XMM[target]),
            32 => a.vxorps(YMM[target], YMM[target], YMM[target]),
            _ => a.vpxord(ZMM[target], ZMM[target], ZMM[target]),
        },
    }
}
