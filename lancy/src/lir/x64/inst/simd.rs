//! Native arithmetic and unaligned moves for each physical SIMD width.
use crate::lir::x64::{Features, emit};
use crate::{
    hir::{Binary as B, Lane, Ternary, Type, Unary},
    numbers::{Action, Math},
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
    pub math: Math,
    bytes: usize,
}

impl Native {
    pub fn select(math: Math, cpu: Features) -> Option<Self> {
        let lane = math.input.lane()?;
        let packed = matches!(math.input, Type::Vector(_));
        let bytes = if packed || matches!(math.action, Action::Splat) {
            cpu.bytes
        } else {
            lane.bytes()
        };
        if matches!(lane, Lane::F16 | Lane::BF16) {
            let supported = lane == Lane::F16
                && packed
                && cpu.fp16
                && (bytes == 64 || (bytes == 32 && cpu.avx10 >= 1))
                && matches!(
                    math.action,
                    Action::Binary(B::Add | B::Sub | B::Mul | B::Div)
                        | Action::Unary(Unary::Sqrt)
                        | Action::Ternary(Ternary::MulAdd)
                );
            return supported.then_some(Self { math, bytes });
        }
        let int =
            packed && (bytes != 32 || cpu.avx2) && (bytes != 64 || lane.bytes() >= 4 || cpu.bw);
        let supported = match math.action {
            Action::Splat => {
                lane.bytes() <= 8
                    && match bytes {
                        16 => true,
                        32 => cpu.avx2,
                        64 => lane.bytes() >= 4 || cpu.bw,
                        _ => false,
                    }
            }
            Action::Compress | Action::Expand => {
                bytes == 64
                    && packed
                    && match lane.bytes() {
                        1 | 2 => cpu.vbmi2 && cpu.bw,
                        4 | 8 => cpu.dq,
                        _ => false,
                    }
            }
            Action::Unary(Unary::Abs) if lane.signed() => {
                int && match lane.bytes() {
                    1 | 2 | 4 => bytes != 16 || cpu.ssse3,
                    8 => bytes == 64,
                    _ => false,
                }
            }
            Action::Binary(op @ (B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge))
                if !lane.float() =>
            {
                int && lane.bytes() <= 8
                    && (bytes == 64
                        || match op {
                            B::Eq => lane.bytes() != 8 || bytes == 32 || cpu.sse41,
                            B::Gt => {
                                lane.signed() && (lane.bytes() != 8 || bytes == 32 || cpu.sse42)
                            }
                            _ => false,
                        })
            }
            Action::Unary(Unary::Popcnt) => {
                packed
                    && bytes == 64
                    && match lane.bytes() {
                        1 | 2 => cpu.bitalg,
                        4 | 8 => cpu.vpopcnt,
                        _ => false,
                    }
            }
            Action::Unary(Unary::LeadingZeros) => {
                packed && bytes == 64 && cpu.cd && matches!(lane.bytes(), 4 | 8)
            }
            Action::Unary(Unary::Floor | Unary::Ceil | Unary::Trunc | Unary::Nearest) => {
                lane.float() && (cpu.sse41 || bytes >= 32)
            }
            Action::Binary(B::SaturatingAdd | B::SaturatingSub) => int && lane.bytes() <= 2,
            Action::Binary(B::Min | B::Max) => {
                int && !lane.float()
                    && match lane.bytes() {
                        1 => !lane.signed() || bytes != 16 || cpu.sse41,
                        2 => lane.signed() || bytes != 16 || cpu.sse41,
                        4 => bytes != 16 || cpu.sse41,
                        8 => bytes == 64,
                        _ => false,
                    }
            }
            Action::Binary(B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge)
                if packed && lane.float() =>
            {
                bytes != 16 || !matches!(math.action, Action::Binary(B::Gt | B::Ge))
            }
            Action::Cast => {
                cpu.avx10 >= 2
                    && matches!(
                        (math.input, math.output),
                        (Type::Vector(Lane::F32), Type::Vector(Lane::I32 | Lane::U32))
                            | (
                                Type::Vector(Lane::F64),
                                Type::Vector(Lane::I64 | Lane::U64 | Lane::Isize | Lane::Usize)
                            )
                    )
            }
            Action::Binary(B::Add | B::Sub) => lane.float() || (int && lane.bytes() <= 8),
            Action::Binary(B::Mul) => {
                lane.float()
                    || (int
                        && match lane.bytes() {
                            2 => true,
                            4 => bytes != 16 || cpu.sse41,
                            8 => bytes == 64 && cpu.dq,
                            _ => false,
                        })
            }
            Action::Binary(B::Div) | Action::Unary(Unary::Sqrt) => lane.float(),
            Action::Binary(B::And | B::Or | B::Xor) => packed,
            Action::Ternary(Ternary::MulAdd) => lane.float() && (cpu.fma || bytes == 64),
            _ => false,
        };
        supported.then_some(Self { math, bytes })
    }

    pub fn tied(self) -> bool {
        matches!(self.math.action, Action::Ternary(_))
            || (self.bytes <= 16 && matches!(self.math.action, Action::Binary(_)))
    }

    pub fn emit(self, a: &mut CodeAssembler, regs: &[usize]) -> Result<(), IcedError> {
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
                match self.math.input {
                    Type::F32 => a.$ss(XMM[dst], XMM[rhs]),
                    Type::F64 => a.$sd(XMM[dst], XMM[rhs]),
                    Type::Vector(Lane::F32) => packed!($ps, $vps, $vps),
                    _ => packed!($pd, $vpd, $vpd),
                }
            };
        }
        let lane = self.math.input.lane().unwrap();
        if lane == Lane::F16 {
            macro_rules! ph {($op:ident,$($r:ident),+) => {if self.bytes==64 {a.$op($(ZMM[$r]),+)}else{a.$op($(YMM[$r]),+)}}; }
            return match self.math.action {
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
        match self.math.action {
            Action::Splat => {
                // The scalar lives either in a GPR or in the low lane of an XMM.
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
                mask_from_signs(a, lane.bytes(), rhs)?;
                let d = ZMM[dst].k1().z();
                match (self.math.action, lane.bytes()) {
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
                    let cc = match op {
                        B::Eq => 0,
                        B::Lt => 1,
                        B::Le => 2,
                        B::Ne => 4,
                        B::Ge => 5,
                        _ => 6,
                    };
                    macro_rules! cmp {
                        ($op:ident) => {
                            a.$op(k1, ZMM[src], ZMM[rhs], cc)
                        };
                    }
                    match (lane.bytes(), lane.signed()) {
                        (1, true) => cmp!(vpcmpb),
                        (1, false) => cmp!(vpcmpub),
                        (2, true) => cmp!(vpcmpw),
                        (2, false) => cmp!(vpcmpuw),
                        (4, true) => cmp!(vpcmpd),
                        (4, false) => cmp!(vpcmpud),
                        (8, true) => cmp!(vpcmpq),
                        _ => cmp!(vpcmpuq),
                    }?;
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
                let cc = match op {
                    B::Eq => 0,
                    B::Ne => 4,
                    B::Lt => 1,
                    B::Le => 2,
                    B::Gt => 30,
                    _ => 29,
                };
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
                // Intel AVX10.2, VCVTT{PS,PD}2{U,}{D,Q}QS. iced 1.22 predates MAP5.
                // Allocatable SIMD registers are 0..15, so R' and X' stay inverted.
                let p0 = 0xf5 ^ (((dst >> 3) as u8) << 7) ^ (((src >> 3) as u8) << 5);
                let p1 = if lane == Lane::F32 { 0x7c } else { 0xfd };
                let p2 = match self.bytes {
                    16 => 0x08,
                    32 => 0x28,
                    _ => 0x48,
                };
                let opcode = if self.math.output.lane().unwrap().signed() {
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

// K1 is reserved inside masked pseudo-instructions and never holds a live SSA value.
fn mask_from_signs(a: &mut CodeAssembler, size: usize, mask: usize) -> Result<(), IcedError> {
    match size {
        1 => a.vpmovb2m(k1, ZMM[mask]),
        2 => a.vpmovw2m(k1, ZMM[mask]),
        4 => a.vpmovd2m(k1, ZMM[mask]),
        _ => a.vpmovq2m(k1, ZMM[mask]),
    }
}

pub(in crate::lir::x64) fn masked(
    a: &mut CodeAssembler,
    ty: Type,
    base: usize,
    index: Option<usize>,
    mask: usize,
    value: usize,
    dst: Option<usize>,
) -> Result<(), IcedError> {
    use crate::lir::x64::emit::R64;
    let cpu = Features::host();
    let width = cpu.bytes;
    let size = ty.lane().unwrap().bytes();
    if width == 64
        && index.is_none()
        && match size {
            1 | 2 => cpu.bw,
            4 | 8 => cpu.dq,
            _ => false,
        }
    {
        mask_from_signs(a, size, mask)?;
        let mem = zmmword_ptr(R64[base]);
        macro_rules! access {
            ($op:ident) => {
                if let Some(d) = dst {
                    if d != value {
                        a.vmovaps(ZMM[d], ZMM[value])?;
                    }
                    a.$op(ZMM[d].k1(), mem)
                } else {
                    a.$op(mem.k1(), ZMM[value])
                }
            };
        }
        return match size {
            1 => access!(vmovdqu8),
            2 => access!(vmovdqu16),
            4 => access!(vmovdqu32),
            _ => access!(vmovdqu64),
        };
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
