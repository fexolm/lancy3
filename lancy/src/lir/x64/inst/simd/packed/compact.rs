use super::{XMM, YMM};
use crate::lir::x64::Cpu;
use crate::lir::x64::{
    emit::{R32, R64},
    inst::SCRATCH,
};
use iced_x86::{IcedError, code_asm::*};

const fn indices<const LANES: usize, const ENTRIES: usize>(expand: bool) -> [[i8; LANES]; ENTRIES] {
    let mut table = [[-1; LANES]; ENTRIES];
    let mut mask = 0;
    while mask < ENTRIES {
        let mut active = 0;
        let mut i = 0;
        while i < LANES {
            if mask & (1 << i) != 0 {
                let (from, to) = if expand { (active, i) } else { (i, active) };
                table[mask][to] = from as i8;
                active += 1;
            }
            i += 1;
        }
        mask += 1;
    }
    table
}

#[repr(align(16))]
struct ByteControls([[u8; 16]; 16]);

const fn bytes(expand: bool) -> ByteControls {
    let indices = indices::<4, 16>(expand);
    let mut out = [[0x80; 16]; 16];
    let mut mask = 0;
    while mask < 16 {
        let mut lane = 0;
        while lane < 4 {
            let source = indices[mask][lane];
            if source >= 0 {
                let mut i = 0;
                while i < 4 {
                    out[mask][4 * lane + i] = 4 * source as u8 + i as u8;
                    i += 1;
                }
            }
            lane += 1;
        }
        mask += 1;
    }
    ByteControls(out)
}

static COMPRESS_YMM: [[i8; 8]; 256] = indices(false);
static EXPAND_YMM: [[i8; 8]; 256] = indices(true);
static COMPRESS_XMM: ByteControls = bytes(false);
static EXPAND_XMM: ByteControls = bytes(true);

#[repr(align(16))]
struct DwordControls([[[u32; 4]; 4]; 16]);

const fn dwords(expand: bool) -> DwordControls {
    let indices = indices::<4, 16>(expand);
    let mut out = [[[0; 4]; 4]; 16];
    let mut mask = 0;
    while mask < 16 {
        let mut target = 0;
        while target < 4 {
            let source = indices[mask][target];
            if source >= 0 {
                out[mask][source as usize][target] = u32::MAX;
            }
            target += 1;
        }
        mask += 1;
    }
    DwordControls(out)
}

static COMPRESS_SSE2: DwordControls = dwords(false);
static EXPAND_SSE2: DwordControls = dwords(true);

pub(in crate::lir::x64) fn emit(
    a: &mut CodeAssembler,
    cpu: Cpu,
    expand: bool,
    regs: [usize; 4],
) -> Result<(), IcedError> {
    let [dst, value, mask, index] = regs;
    // JIT code lives in this process. Each immutable table outlives the JIT
    // and MOVMSK selects a row entirely within it.
    if cpu.bytes == 32 {
        let table = if expand { &EXPAND_YMM } else { &COMPRESS_YMM };
        a.vmovmskps(R32[index], YMM[mask])?;
        a.mov(r10, table.as_ptr() as u64)?;
        let control = YMM[SCRATCH];
        a.vpmovsxbd(control, qword_ptr(r10 + R64[index] * 8))?;
        a.vpermd(YMM[dst], control, YMM[value])?;
        a.vpsrad(control, control, 31)?;
        a.vpandn(YMM[dst], control, YMM[dst])
    } else if cpu.isa.ssse3 {
        let table = if expand { &EXPAND_XMM } else { &COMPRESS_XMM };
        a.movmskps(R32[index], XMM[mask])?;
        a.mov(r10, table.0.as_ptr() as u64)?;
        a.add(R32[index], R32[index])?;
        if dst != value {
            a.movaps(XMM[dst], XMM[value])?;
        }
        a.pshufb(XMM[dst], xmmword_ptr(r10 + R64[index] * 8))
    } else {
        let table = if expand { &EXPAND_SSE2 } else { &COMPRESS_SSE2 };
        a.movmskps(R32[index], XMM[mask])?;
        a.mov(r10, table.0.as_ptr() as u64)?;
        a.shl(R32[index], 6)?;
        let row = xmmword_ptr(r10 + R64[index]);
        let (d, x, scratch) = (XMM[dst], XMM[value], XMM[SCRATCH]);
        a.pshufd(d, x, 0)?;
        a.pand(d, row)?;
        for i in 1..4 {
            a.pshufd(scratch, x, i * 0x55)?;
            a.pand(scratch, row + i * 16)?;
            a.por(d, scratch)?;
        }
        Ok(())
    }
}
