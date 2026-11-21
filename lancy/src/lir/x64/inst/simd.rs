//! Native arithmetic and unaligned moves for each physical SIMD width.
pub(in crate::lir::x64) mod packed;
use super::VectorMask;
use crate::lir::x64::{Features, emit};
use crate::{
    hir::operation::Action,
    hir::{Binary as B, Lane, Ternary, Type, Unary},
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

#[derive(Clone, Copy, Debug)]
pub(in crate::lir::x64) struct Native {
    pub action: Action,
    pub input: Type,
    pub output: Type,
    pub(in crate::lir::x64) bytes: usize,
    pub(in crate::lir::x64) cpu: Features,
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
            || (self.bytes <= 16 && matches!(self.action, Action::Binary(_)))
    }

    #[cfg(test)]
    pub fn emit(self, a: &mut CodeAssembler, regs: &[usize]) -> Result<(), IcedError> {
        let mut free = (0..16).filter(|r| *r != super::SCRATCH && !regs.contains(r));
        let mut args = regs.to_vec();
        let last = args.len() - 1;
        // Standalone instruction tests can request an aliased output. Production
        // allocation reserves a separate result throughout compound sequences.
        let alias = self.separate_output() && regs[..last].contains(&regs[last]);
        if alias {
            args[last] = free.next().unwrap();
        }
        let temp: Vec<_> = free.take(self.temporaries()).collect();
        self.emit_allocated(a, &args, &temp, None)?;
        if alias {
            if matches!(self.output, Type::Vector(_)) {
                copy(a, regs[last], args[last], self.bytes)?;
            } else if super::class(self.output) == regalloc2::RegClass::Vector {
                a.movaps(XMM[regs[last]], XMM[args[last]])?;
            } else {
                a.mov(emit::R64[regs[last]], emit::R64[args[last]])?;
            }
        }
        Ok(())
    }

    pub fn emit_allocated(
        self,
        a: &mut CodeAssembler,
        regs: &[usize],
        temp: &[usize],
        pool: Option<&mut emit::Constants>,
    ) -> Result<(), IcedError> {
        if let Some(recipe) = self.recipe {
            return recipe.emit(a, self, regs, temp, pool);
        }

        let src = regs[0];
        let rhs = regs[1];
        let dst = *regs.last().unwrap();
        macro_rules! packed {
            ($sse:ident,$avx:ident,$evex:ident) => {
                match self.bytes {
                    16 => a.$sse(XMM[dst], XMM[rhs]),
                    32 => a.$avx(YMM[dst], YMM[src], YMM[rhs]),
                    64 => a.$evex(ZMM[dst], ZMM[src], ZMM[rhs]),
                    _ => unreachable!(),
                }
            };
        }
        macro_rules! float_bin {
            ($ss:ident,$sd:ident,$ps:ident,$pd:ident,$vps:ident,$vpd:ident) => {
                match self.input {
                    Type::F32 => a.$ss(XMM[dst], XMM[rhs]),
                    Type::F64 => a.$sd(XMM[dst], XMM[rhs]),
                    Type::Vector(Lane::F32) => packed!($ps, $vps, $vps),
                    _ => packed!($pd, $vpd, $vpd),
                }
            };
        }
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
                        a.pcmpeqd(XMM[scratch], XMM[scratch])?;
                        a.vinsertf128(YMM[scratch], YMM[scratch], XMM[scratch], 1)?;
                        a.vxorps(YMM[dst], YMM[src], YMM[scratch])
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
                return blend(a, lane, dst, rhs, no);
            }
            _ => {}
        }
        if lane == Lane::F16 {
            macro_rules! ph {($op:ident,$($r:ident),+) => {if self.bytes==64 {a.$op($(ZMM[$r]),+)}else{a.$op($(YMM[$r]),+)}}; }
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
                let d = ZMM[dst].k1().z();
                match (self.action, lane.bytes()) {
                    (Action::Compress, 1) => a.vpcompressb(d, ZMM[src]),
                    (Action::Compress, 2) => a.vpcompressw(d, ZMM[src]),
                    (Action::Compress, 4) => a.vpcompressd(d, ZMM[src]),
                    (Action::Compress, 8) => a.vpcompressq(d, ZMM[src]),
                    (_, 1) => a.vpexpandb(d, ZMM[src]),
                    (_, 2) => a.vpexpandw(d, ZMM[src]),
                    (_, 4) => a.vpexpandd(d, ZMM[src]),
                    _ => a.vpexpandq(d, ZMM[src]),
                }
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
            Action::Binary(B::SaturatingAdd) => match (lane.bytes(), lane.signed()) {
                (1, false) => packed!(paddusb, vpaddusb, vpaddusb),
                (1, true) => packed!(paddsb, vpaddsb, vpaddsb),
                (2, false) => packed!(paddusw, vpaddusw, vpaddusw),
                (2, true) => packed!(paddsw, vpaddsw, vpaddsw),
                _ => unreachable!(),
            },
            Action::Binary(B::SaturatingSub) => match (lane.bytes(), lane.signed()) {
                (1, false) => packed!(psubusb, vpsubusb, vpsubusb),
                (1, true) => packed!(psubsb, vpsubsb, vpsubsb),
                (2, false) => packed!(psubusw, vpsubusw, vpsubusw),
                (2, true) => packed!(psubsw, vpsubsw, vpsubsw),
                _ => unreachable!(),
            },
            Action::Binary(B::Min) => match (lane.bytes(), lane.signed()) {
                (1, false) => packed!(pminub, vpminub, vpminub),
                (1, true) => packed!(pminsb, vpminsb, vpminsb),
                (2, false) => packed!(pminuw, vpminuw, vpminuw),
                (2, true) => packed!(pminsw, vpminsw, vpminsw),
                (4, false) => packed!(pminud, vpminud, vpminud),
                (4, true) => packed!(pminsd, vpminsd, vpminsd),
                (8, false) => a.vpminuq(ZMM[dst], ZMM[src], ZMM[rhs]),
                (8, true) => a.vpminsq(ZMM[dst], ZMM[src], ZMM[rhs]),
                _ => unreachable!(),
            },
            Action::Binary(B::Max) => match (lane.bytes(), lane.signed()) {
                (1, false) => packed!(pmaxub, vpmaxub, vpmaxub),
                (1, true) => packed!(pmaxsb, vpmaxsb, vpmaxsb),
                (2, false) => packed!(pmaxuw, vpmaxuw, vpmaxuw),
                (2, true) => packed!(pmaxsw, vpmaxsw, vpmaxsw),
                (4, false) => packed!(pmaxud, vpmaxud, vpmaxud),
                (4, true) => packed!(pmaxsd, vpmaxsd, vpmaxsd),
                (8, false) => a.vpmaxuq(ZMM[dst], ZMM[src], ZMM[rhs]),
                (8, true) => a.vpmaxsq(ZMM[dst], ZMM[src], ZMM[rhs]),
                _ => unreachable!(),
            },
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
            Action::Binary(B::Add) if lane.float() => {
                float_bin!(addss, addsd, addps, addpd, vaddps, vaddpd)
            }
            Action::Binary(B::Sub) if lane.float() => {
                float_bin!(subss, subsd, subps, subpd, vsubps, vsubpd)
            }
            Action::Binary(B::Mul) if lane.float() => {
                float_bin!(mulss, mulsd, mulps, mulpd, vmulps, vmulpd)
            }
            Action::Binary(B::Div) => float_bin!(divss, divsd, divps, divpd, vdivps, vdivpd),
            Action::Binary(B::Add) => match lane.bytes() {
                1 => packed!(paddb, vpaddb, vpaddb),
                2 => packed!(paddw, vpaddw, vpaddw),
                4 => packed!(paddd, vpaddd, vpaddd),
                _ => packed!(paddq, vpaddq, vpaddq),
            },
            Action::Binary(B::Sub) => match lane.bytes() {
                1 => packed!(psubb, vpsubb, vpsubb),
                2 => packed!(psubw, vpsubw, vpsubw),
                4 => packed!(psubd, vpsubd, vpsubd),
                _ => packed!(psubq, vpsubq, vpsubq),
            },
            Action::Binary(B::Mul) => match lane.bytes() {
                2 => packed!(pmullw, vpmullw, vpmullw),
                4 => packed!(pmulld, vpmulld, vpmulld),
                _ => a.vpmullq(ZMM[dst], ZMM[src], ZMM[rhs]),
            },
            Action::Binary(B::And) => packed!(andps, vandps, vpandd),
            Action::Binary(B::Or) => packed!(orps, vorps, vpord),
            Action::Binary(B::Xor) => packed!(xorps, vxorps, vpxord),
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

/// Sets ZF exactly when every bit of the vector is zero.
/// Set ZF if every lane is nonzero.
pub(in crate::lir::x64) fn test_all(
    a: &mut CodeAssembler,
    cpu: Features,
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
                if cpu.dq {
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

pub(in crate::lir::x64) fn test_any(
    a: &mut CodeAssembler,
    cpu: Features,
    src: usize,
) -> Result<(), IcedError> {
    match cpu.bytes {
        64 => {
            a.vptestmd(k1, ZMM[src], ZMM[src])?;
            a.kortestw(k1, k1)
        }
        32 => a.vptest(YMM[src], YMM[src]),
        _ if cpu.sse41 => a.ptest(XMM[src], XMM[src]),
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
    cpu: Features,
    size: usize,
    mask: usize,
) -> Result<(), IcedError> {
    if !cpu.dq && size >= 4 {
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

fn compare_mask(
    a: &mut CodeAssembler,
    dst: AsmRegisterK,
    lane: Lane,
    lhs: usize,
    rhs: usize,
    cc: i32,
) -> Result<(), IcedError> {
    let (lhs, rhs) = (ZMM[lhs], ZMM[rhs]);
    match (lane, lane.bytes(), lane.signed()) {
        (Lane::F32, ..) => a.vcmpps(dst, lhs, rhs, cc),
        (Lane::F64, ..) => a.vcmppd(dst, lhs, rhs, cc),
        (_, 1, true) => a.vpcmpb(dst, lhs, rhs, cc),
        (_, 1, false) => a.vpcmpub(dst, lhs, rhs, cc),
        (_, 2, true) => a.vpcmpw(dst, lhs, rhs, cc),
        (_, 2, false) => a.vpcmpuw(dst, lhs, rhs, cc),
        (_, 4, true) => a.vpcmpd(dst, lhs, rhs, cc),
        (_, 4, false) => a.vpcmpud(dst, lhs, rhs, cc),
        (_, 8, true) => a.vpcmpq(dst, lhs, rhs, cc),
        (_, 8, false) => a.vpcmpuq(dst, lhs, rhs, cc),
        _ => unreachable!(),
    }
}

enum MaskBits {
    Sign,
    Nonzero,
}

fn emit_opmask(
    a: &mut CodeAssembler,
    cpu: Features,
    size: usize,
    mask: VectorMask<usize>,
    bits: MaskBits,
) -> Result<(), IcedError> {
    match mask {
        VectorMask::Lanes(reg) => mask_from_signs(a, cpu, size, reg),
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
            compare_mask(a, dst, lane, lhs, rhs, predicate)
        }
    }
}

pub(in crate::lir::x64) fn test_vector_mask(
    a: &mut CodeAssembler,
    cpu: Features,
    mask: VectorMask<usize>,
) -> Result<(), IcedError> {
    let lane = match mask {
        VectorMask::Lanes(reg) => return test_any(a, cpu, reg),
        VectorMask::Compare { lane, .. } => lane,
    };
    let size = lane.bytes();
    emit_opmask(a, cpu, size, mask, MaskBits::Nonzero)?;
    match size {
        1 => a.kortestq(k1, k1),
        2 => a.kortestd(k1, k1),
        _ => a.kortestw(k1, k1),
    }
}

fn blend(
    a: &mut CodeAssembler,
    lane: Lane,
    dst: usize,
    yes: usize,
    no: usize,
) -> Result<(), IcedError> {
    let (d, no, yes) = (ZMM[dst].k1(), ZMM[no], ZMM[yes]);
    match lane.bytes() {
        1 => a.vpblendmb(d, no, yes),
        2 => a.vpblendmw(d, no, yes),
        4 => a.vpblendmd(d, no, yes),
        _ => a.vpblendmq(d, no, yes),
    }
}

pub(in crate::lir::x64) fn blend_comparison(
    a: &mut CodeAssembler,
    cpu: Features,
    lane: Lane,
    mask: VectorMask<usize>,
    dst: usize,
    yes: usize,
    no: usize,
) -> Result<(), IcedError> {
    emit_opmask(a, cpu, lane.bytes(), mask, MaskBits::Sign)?;
    blend(a, lane, dst, yes, no)
}

pub(in crate::lir::x64) fn evex_masked_memory(cpu: Features, size: usize, indexed: bool) -> bool {
    (cpu.bytes == 64 || cpu.avx10 >= 1)
        && (matches!(size, 4 | 8) || !indexed && size <= 2 && cpu.bw)
}

pub(in crate::lir::x64) struct MaskedMemory {
    pub ty: Type,
    pub base: usize,
    pub index: Option<usize>,
    pub mask: VectorMask<usize>,
    pub value: usize,
    pub dst: Option<usize>,
    pub temp: Option<usize>,
}
impl MaskedMemory {
    pub(in crate::lir::x64) fn emit(
        self,
        a: &mut CodeAssembler,
        cpu: Features,
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
            emit_opmask(a, cpu, size, mask, MaskBits::Sign)?;
            if let Some(i) = index {
                macro_rules! indexed {
                    ($r:ident) => {{
                        let address = ptr(R64[base] + $r[i] * size as i32);
                        if let Some(d) = dst {
                            let target = if d == i { scratch } else { d };
                            if target != value {
                                copy(a, target, value, width)?;
                            }
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
                            a.vpscatterdd(dword_ptr(address).k1(), $r[value])
                        } else {
                            a.vpscatterqq(qword_ptr(address).k1(), $r[value])
                        }
                    }};
                }
                return match width {
                    16 => indexed!(XMM),
                    32 => indexed!(YMM),
                    _ => indexed!(ZMM),
                };
            }
            if let Some(d) = dst
                && d != value
            {
                copy(a, d, value, width)?;
            }
            macro_rules! access {
                ($r:ident,$mem:ident,$op:ident) => {{
                    let mem = $mem(R64[base]);
                    if let Some(d) = dst {
                        a.$op($r[d].k1(), mem)
                    } else {
                        a.$op(mem.k1(), $r[value])
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
            if let (Some(i), Some(d)) = (index, dst)
                && cpu.avx2
            {
                // AVX2 clears enabled mask lanes as the gather completes. Regalloc
                // provides a distinct temporary for that destructive operand.
                let m = temp.expect("AVX2 gather mask temporary");
                let target = if d == i { scratch } else { d };
                a.vmovaps(YMM[m], YMM[mask])?;
                if target != value {
                    a.vmovaps(YMM[target], YMM[value])?;
                }
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
                    if size == 4 {
                        a.vmaskmovps(YMM[scratch], YMM[mask], mem)?;
                        return a.vblendvps(YMM[d], YMM[value], YMM[scratch], YMM[mask]);
                    }
                    a.vmaskmovpd(YMM[scratch], YMM[mask], mem)?;
                    return a.vblendvpd(YMM[d], YMM[value], YMM[scratch], YMM[mask]);
                }
                return if size == 4 {
                    a.vmaskmovps(mem, YMM[mask], YMM[value])
                } else {
                    a.vmaskmovpd(mem, YMM[mask], YMM[value])
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
        store(a, ptr(rsp + data), value, width)?;
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
