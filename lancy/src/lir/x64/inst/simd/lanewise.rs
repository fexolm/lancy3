//! Lane-wise binary operations with one instruction per vector width: the
//! single description both selection and emission read. A form exists for
//! a width exactly when this table names an instruction for it and the
//! target has the extension that instruction needs.
use super::{XMM, YMM, ZMM};
use crate::hir::{Binary as B, Lane};
use crate::lir::x64::Cpu;
use iced_x86::{IcedError, code_asm::*};

type Two<R> = fn(&mut CodeAssembler, R, R) -> Result<(), IcedError>;
type Three<R> = fn(&mut CodeAssembler, R, R, R) -> Result<(), IcedError>;

/// The extension an encoding needs beyond what its width implies (SSE2 at
/// 16 bytes, AVX2 at 32, AVX-512F at 64).
#[derive(Clone, Copy, Debug)]
enum Needs {
    Width,
    Sse41,
    Bw,
    Dq,
}

impl Needs {
    fn met(self, cpu: Cpu) -> bool {
        match self {
            Self::Width => true,
            Self::Sse41 => cpu.isa.sse41,
            Self::Bw => cpu.isa.bw,
            Self::Dq => cpu.isa.dq,
        }
    }
}

/// The instructions of one operation on one lane type. SSE forms are
/// destructive: the output is tied to the first input.
#[derive(Clone, Copy)]
pub(in crate::lir::x64) struct Lanewise {
    /// `ss`/`sd` on the low lane of a scalar float.
    scalar: Option<Two<AsmRegisterXmm>>,
    sse: Option<(Two<AsmRegisterXmm>, Needs)>,
    vex: Option<Three<AsmRegisterYmm>>,
    evex: Option<(Three<AsmRegisterZmm>, Needs)>,
}

/// `$sse` (needing `$s`), `$vex` and `$evex` (needing `$e`); `-` for none.
macro_rules! form {
    (@sse -) => { None };
    (@sse $op:ident $needs:ident) => { Some((CodeAssembler::$op as Two<AsmRegisterXmm>, Needs::$needs)) };
    (@vex -) => { None };
    (@vex $op:ident) => { Some(CodeAssembler::$op as Three<AsmRegisterYmm>) };
    (@evex -) => { None };
    (@evex $op:ident $needs:ident) => { Some((CodeAssembler::$op as Three<AsmRegisterZmm>, Needs::$needs)) };
    ($sse:tt $($s:ident)?, $vex:tt, $evex:tt $($e:ident)?) => {
        Lanewise {
            scalar: None,
            sse: form!(@sse $sse $($s)?),
            vex: form!(@vex $vex),
            evex: form!(@evex $evex $($e)?),
        }
    };
}

/// A float operation: scalar `ss`/`sd`, packed `ps`/`pd`.
macro_rules! float {
    ($ss:ident, $sd:ident, $ps:ident, $pd:ident, $vps:ident, $vpd:ident, $f64:expr) => {
        if $f64 {
            Lanewise {
                scalar: Some(CodeAssembler::$sd as Two<AsmRegisterXmm>),
                ..form!($pd Width, $vpd, $vpd Width)
            }
        } else {
            Lanewise {
                scalar: Some(CodeAssembler::$ss as Two<AsmRegisterXmm>),
                ..form!($ps Width, $vps, $vps Width)
            }
        }
    };
}

impl Lanewise {
    /// The instructions of `op` on `lane` lanes, if the backend has them.
    pub(crate) fn of(op: B, lane: Lane) -> Option<Self> {
        let (size, signed) = (lane.bytes(), lane.signed());
        if lane.float() {
            let f64 = match lane {
                Lane::F32 => false,
                Lane::F64 => true,
                _ => return None,
            };
            return Some(match op {
                B::Add => float!(addss, addsd, addps, addpd, vaddps, vaddpd, f64),
                B::Sub => float!(subss, subsd, subps, subpd, vsubps, vsubpd, f64),
                B::Mul => float!(mulss, mulsd, mulps, mulpd, vmulps, vmulpd, f64),
                B::Div => float!(divss, divsd, divps, divpd, vdivps, vdivpd, f64),
                B::And => form!(andps Width, vandps, vpandd Width),
                B::Or => form!(orps Width, vorps, vpord Width),
                B::Xor => form!(xorps Width, vxorps, vpxord Width),
                _ => return None,
            });
        }
        Some(match (op, size, signed) {
            (B::And, ..) => form!(andps Width, vandps, vpandd Width),
            (B::Or, ..) => form!(orps Width, vorps, vpord Width),
            (B::Xor, ..) => form!(xorps Width, vxorps, vpxord Width),
            (B::Add, 1, _) => form!(paddb Width, vpaddb, vpaddb Bw),
            (B::Add, 2, _) => form!(paddw Width, vpaddw, vpaddw Bw),
            (B::Add, 4, _) => form!(paddd Width, vpaddd, vpaddd Width),
            (B::Add, 8, _) => form!(paddq Width, vpaddq, vpaddq Width),
            (B::Sub, 1, _) => form!(psubb Width, vpsubb, vpsubb Bw),
            (B::Sub, 2, _) => form!(psubw Width, vpsubw, vpsubw Bw),
            (B::Sub, 4, _) => form!(psubd Width, vpsubd, vpsubd Width),
            (B::Sub, 8, _) => form!(psubq Width, vpsubq, vpsubq Width),
            (B::Mul, 2, _) => form!(pmullw Width, vpmullw, vpmullw Bw),
            (B::Mul, 4, _) => form!(pmulld Sse41, vpmulld, vpmulld Width),
            (B::Mul, 8, _) => form!(-, -, vpmullq Dq),
            (B::SaturatingAdd, 1, false) => form!(paddusb Width, vpaddusb, vpaddusb Bw),
            (B::SaturatingAdd, 1, true) => form!(paddsb Width, vpaddsb, vpaddsb Bw),
            (B::SaturatingAdd, 2, false) => form!(paddusw Width, vpaddusw, vpaddusw Bw),
            (B::SaturatingAdd, 2, true) => form!(paddsw Width, vpaddsw, vpaddsw Bw),
            (B::SaturatingSub, 1, false) => form!(psubusb Width, vpsubusb, vpsubusb Bw),
            (B::SaturatingSub, 1, true) => form!(psubsb Width, vpsubsb, vpsubsb Bw),
            (B::SaturatingSub, 2, false) => form!(psubusw Width, vpsubusw, vpsubusw Bw),
            (B::SaturatingSub, 2, true) => form!(psubsw Width, vpsubsw, vpsubsw Bw),
            (B::Min, 1, false) => form!(pminub Width, vpminub, vpminub Bw),
            (B::Min, 1, true) => form!(pminsb Sse41, vpminsb, vpminsb Bw),
            (B::Min, 2, false) => form!(pminuw Sse41, vpminuw, vpminuw Bw),
            (B::Min, 2, true) => form!(pminsw Width, vpminsw, vpminsw Bw),
            (B::Min, 4, false) => form!(pminud Sse41, vpminud, vpminud Width),
            (B::Min, 4, true) => form!(pminsd Sse41, vpminsd, vpminsd Width),
            (B::Min, 8, false) => form!(-, -, vpminuq Width),
            (B::Min, 8, true) => form!(-, -, vpminsq Width),
            (B::Max, 1, false) => form!(pmaxub Width, vpmaxub, vpmaxub Bw),
            (B::Max, 1, true) => form!(pmaxsb Sse41, vpmaxsb, vpmaxsb Bw),
            (B::Max, 2, false) => form!(pmaxuw Sse41, vpmaxuw, vpmaxuw Bw),
            (B::Max, 2, true) => form!(pmaxsw Width, vpmaxsw, vpmaxsw Bw),
            (B::Max, 4, false) => form!(pmaxud Sse41, vpmaxud, vpmaxud Width),
            (B::Max, 4, true) => form!(pmaxsd Sse41, vpmaxsd, vpmaxsd Width),
            (B::Max, 8, false) => form!(-, -, vpmaxuq Width),
            (B::Max, 8, true) => form!(-, -, vpmaxsq Width),
            _ => return None,
        })
    }

    /// The operation has an instruction for `bytes` (a scalar float's lane
    /// size, or the vector width) on `cpu`.
    pub(crate) fn supports(self, bytes: usize, cpu: Cpu) -> bool {
        match bytes {
            4 | 8 => self.scalar.is_some(),
            16 => self.sse.is_some_and(|(_, needs)| needs.met(cpu)),
            32 => self.vex.is_some(),
            64 => self.evex.is_some_and(|(_, needs)| needs.met(cpu)),
            _ => false,
        }
    }

    pub(crate) fn emit(
        self,
        a: &mut CodeAssembler,
        bytes: usize,
        [dst, lhs, rhs]: [usize; 3],
    ) -> Result<(), IcedError> {
        const CHECKED: &str = "selection checked the form with `supports`";
        // The SSE and scalar forms overwrite their first input: `Native::tied`.
        match bytes {
            4 | 8 => self.scalar.expect(CHECKED)(a, XMM[dst], XMM[rhs]),
            16 => self.sse.expect(CHECKED).0(a, XMM[dst], XMM[rhs]),
            32 => self.vex.expect(CHECKED)(a, YMM[dst], YMM[lhs], YMM[rhs]),
            _ => self.evex.expect(CHECKED).0(a, ZMM[dst], ZMM[lhs], ZMM[rhs]),
        }
    }
}
