use super::{B, K, Lane};
use crate::lir::x64::Cpu;
use iced_x86::{IcedError, code_asm::*};

pub(in crate::lir::x64) fn immediate(
    a: &mut CodeAssembler,
    dst: usize,
    bits: u64,
) -> Result<(), IcedError> {
    let d = K[dst];
    match bits {
        0 => a.kxorw(d, d, d),
        0xffff => a.kxnorw(d, d, d),
        0xffff_ffff => a.kxnord(d, d, d),
        u64::MAX => a.kxnorq(d, d, d),
        _ => {
            a.mov(r10d, bits as u32)?;
            a.kmovw(d, r10d)
        }
    }
}

pub(in crate::lir::x64) fn test_all(
    a: &mut CodeAssembler,
    cpu: Cpu,
    lane: Lane,
    src: usize,
) -> Result<(), IcedError> {
    let k = K[src];
    match cpu.bytes / lane.bytes() {
        64 => {
            a.knotq(k1, k)?;
            a.kortestq(k1, k1)
        }
        32 => {
            a.knotd(k1, k)?;
            a.kortestd(k1, k1)
        }
        16 => {
            a.knotw(k1, k)?;
            a.kortestw(k1, k1)
        }
        8 if cpu.isa.dq => {
            a.knotb(k1, k)?;
            a.kortestb(k1, k1)
        }
        lanes => {
            a.kmovw(r10d, k)?;
            a.cmp(r10d, (1 << lanes) - 1)
        }
    }
}

pub(in crate::lir::x64) fn logic(
    a: &mut CodeAssembler,
    lanes: usize,
    op: B,
    dst: usize,
    lhs: usize,
    rhs: usize,
) -> Result<(), IcedError> {
    let (d, x, y) = (K[dst], K[lhs], K[rhs]);
    macro_rules! logic {
        ($w:ident, $d:ident, $q:ident) => {
            match lanes {
                64 => a.$q(d, x, y),
                32 => a.$d(d, x, y),
                _ => a.$w(d, x, y),
            }
        };
    }
    match op {
        B::And => logic!(kandw, kandd, kandq),
        B::Or => logic!(korw, kord, korq),
        B::Xor => logic!(kxorw, kxord, kxorq),
        _ => unreachable!(),
    }
}

pub(in crate::lir::x64) fn not(
    a: &mut CodeAssembler,
    cpu: Cpu,
    lane: Lane,
    dst: usize,
    src: usize,
) -> Result<(), IcedError> {
    let (d, x) = (K[dst], K[src]);
    match cpu.bytes / lane.bytes() {
        64 => a.knotq(d, x),
        32 => a.knotd(d, x),
        16 => a.knotw(d, x),
        8 if cpu.isa.dq => a.knotb(d, x),
        lanes => {
            // Bits above the logical lane count must stay zero for KORTEST.
            a.mov(r10d, (1i32 << lanes) - 1)?;
            a.kmovw(k1, r10d)?;
            a.kxorw(d, x, k1)
        }
    }
}
