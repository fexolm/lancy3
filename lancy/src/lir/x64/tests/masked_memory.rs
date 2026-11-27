//! Exercise the physical registers of the production masked-memory emitter at
//! narrower widths and on AVX-512F without its optional mask-conversion subset.
use super::packed_code::{host_supports, profiles, supports};
use crate::hir::{Lane, Type};
use crate::jit::packed_test;
use crate::lir::x64::{Cpu, inst::simd};
use iced_x86::{Decoder, DecoderOptions, Mnemonic, Register, code_asm::*};
use splice_target::X86Features;

fn word(slot: &[u8; 64], lane: usize, size: usize) -> u64 {
    let mut bits = [0; 8];
    bits[..size].copy_from_slice(&slot[lane * size..(lane + 1) * size]);
    u64::from_le_bytes(bits)
}

fn set(slot: &mut [u8; 64], lane: usize, size: usize, value: u64) {
    slot[lane * size..(lane + 1) * size].copy_from_slice(&value.to_le_bytes()[..size]);
}

// Slots hold the memory region, indices, mask, value/passthrough, and output.
// Copy live inputs back to expose accidental destruction by the gather.
fn wrap(
    body: &[u8],
    cpu: Cpu,
    index: Option<usize>,
    mask: usize,
    value: usize,
    dst: Option<usize>,
) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    a.endbr64().unwrap();
    a.mov(r9, if cfg!(windows) { rcx } else { rdi }).unwrap();
    a.lea(
        r8,
        ptr(r9
            + if index.is_some() {
                cpu.bytes as i32 / 2
            } else {
                0
            }),
    )
    .unwrap();
    for (i, r) in [index, Some(mask), Some(value)].into_iter().enumerate() {
        if let Some(r) = r {
            simd::load(&mut a, r, ptr(r9 + 64 * (i + 1) as i32), cpu.bytes).unwrap();
        }
    }
    a.db(body).unwrap();
    for (i, r) in [index, Some(mask), Some(value), dst]
        .into_iter()
        .enumerate()
    {
        if let Some(r) = r {
            simd::store(&mut a, ptr(r9 + 64 * (i + 1) as i32), r, cpu.bytes).unwrap();
        }
    }
    a.ret().unwrap();
    a.assemble(0).unwrap()
}

fn exercise(
    cpu: Cpu,
    lane: Lane,
    index: Option<usize>,
    mask: usize,
    value: usize,
    dst: Option<usize>,
) {
    let size = lane.bytes();
    let count = cpu.bytes / size;
    let gather = index.is_some() && dst.is_some();
    let evex = cpu.bytes == 64 || cpu.isa.avx10 >= 1;
    let mut a = CodeAssembler::new(64).unwrap();
    simd::MaskedMemory {
        ty: Type::Vector(lane),
        base: 8,
        index,
        mask: super::super::inst::VectorMask::Lanes(mask),
        value: Some(value),
        dst,
        temp: (!evex && cpu.isa.avx2 && gather).then_some(4),
    }
    .emit(&mut a, cpu)
    .unwrap();
    let bytes = a.assemble(0).unwrap();
    let mut host_ok = cpu.bytes <= Cpu::host().bytes;
    let expected = match (index.is_some(), dst.is_some(), size, evex) {
        (true, true, 4, _) if evex || cpu.isa.avx2 => Some(Mnemonic::Vpgatherdd),
        (true, true, 8, _) if evex || cpu.isa.avx2 => Some(Mnemonic::Vpgatherqq),
        (true, false, 4, true) => Some(Mnemonic::Vpscatterdd),
        (true, false, 8, true) => Some(Mnemonic::Vpscatterqq),
        (false, _, 1, true) if cpu.isa.bw => Some(Mnemonic::Vmovdqu8),
        (false, _, 2, true) if cpu.isa.bw => Some(Mnemonic::Vmovdqu16),
        (false, _, 4, true) => Some(Mnemonic::Vmovdqu32),
        (false, _, 8, true) => Some(Mnemonic::Vmovdqu64),
        (false, _, 4, false) if cpu.bytes == 32 => Some(Mnemonic::Vmaskmovps),
        (false, _, 8, false) if cpu.bytes == 32 => Some(Mnemonic::Vmaskmovpd),
        _ => None,
    };
    let mut accesses = 0;
    for inst in Decoder::new(64, &bytes, DecoderOptions::NONE) {
        assert!(!inst.is_invalid());
        for &feature in inst.cpuid_features() {
            assert!(
                supports(cpu, feature),
                "{cpu:?} unexpectedly requires {feature:?} for {:?}",
                inst.code()
            );
            host_ok &= host_supports(feature);
        }
        if expected.is_some() {
            assert_ne!(
                inst.memory_base(),
                Register::RSP,
                "the mask temporary must not spill to the host stack"
            );
            assert!(!(0..inst.op_count()).any(|i| inst.op_register(i) == Register::RSP));
            assert!(!matches!(inst.mnemonic(), Mnemonic::Push | Mnemonic::Pop));
        }
        for reg in (0..inst.op_count())
            .map(|i| inst.op_register(i))
            .chain([inst.memory_index()])
        {
            assert!(
                !reg.is_zmm() || cpu.bytes == 64,
                "{cpu:?} {:?}",
                inst.code()
            );
            assert!(
                !reg.is_ymm() || cpu.bytes >= 32,
                "{cpu:?} {:?}",
                inst.code()
            );
        }
        if Some(inst.mnemonic()) == expected && inst.memory_base() == Register::R8 {
            accesses += 1;
            assert_eq!(
                inst.has_op_mask(),
                evex,
                "expected width-appropriate encoding of {expected:?}"
            );
            if gather {
                assert_ne!(inst.op0_register(), inst.memory_index());
                if !evex {
                    // VEX gather writes both destination and mask. Its index
                    // must differ from both registers even when SSA values alias.
                    assert_ne!(inst.op2_register(), inst.op0_register());
                    assert_ne!(inst.op2_register(), inst.memory_index());
                }
            }
        }
    }
    assert_eq!(
        accesses,
        usize::from(expected.is_some()),
        "{lane:?} on {cpu:?}"
    );
    if !host_ok {
        return;
    }

    let code = wrap(&bytes, cpu, index, mask, value, dst);
    let sign = 1u64 << (size * 8 - 1);
    let mut slots = [[0u8; 64]; 5];
    for i in 0..count {
        let ix = [-1i64, 1, -2, -1][i % 4] as u64;
        let bits = if index == Some(mask) {
            ix
        } else {
            [sign, 1, sign | 17, u64::MAX][i % 4]
        };
        let item = if index == Some(value) {
            ix
        } else if value == mask {
            bits
        } else {
            0xb900_13a7 + i as u64
        };
        set(&mut slots[0], i, size, 0xe07f_7e00 + i as u64);
        set(&mut slots[1], i, size, ix);
        set(&mut slots[2], i, size, bits);
        set(&mut slots[3], i, size, item);
    }
    let mut expected = slots;
    expected[4] = slots[3];
    for i in 0..count {
        if word(&slots[2], i, size) & sign == 0 {
            continue;
        }
        let j = if index.is_some() {
            let shift = 64 - 8 * size;
            let ix = ((word(&slots[1], i, size) << shift) as i64) >> shift;
            (count as i64 / 2 + ix) as usize
        } else {
            i
        };
        if dst.is_some() {
            set(&mut expected[4], i, size, word(&slots[0], j, size));
        } else {
            set(&mut expected[0], j, size, word(&slots[3], i, size));
        }
    }
    // SAFETY: even inactive indices here address the live first slot, and
    // the decoded body uses only CPU-supported instructions. The test wrapper
    // uses volatile GPRs and vector registers (including the reserved
    // scratch) on both x64 platforms. The fallback releases its local frame.
    unsafe {
        packed_test::with_kernel(&code, |call| {
            call(&mut slots);
            assert_eq!(slots[0], expected[0]);
            if dst.is_some() {
                assert_eq!(slots[4], expected[4]);
            }
            for (i, r) in [index, Some(mask), Some(value)].into_iter().enumerate() {
                if r.is_some() && r != dst {
                    assert_eq!(slots[i + 1], expected[i + 1], "clobbered input {i}");
                }
            }
        })
    }
}

#[test]
fn masked_memory_uses_signed_lane_bits_and_only_available_isa() {
    let [sse2, avx2, avx512f, avx512bw] = profiles();
    let avx10 = Cpu {
        isa: X86Features {
            avx10: 1,
            bw: true,
            dq: true,
            ..avx2.isa
        },
        ..avx2
    };
    for cpu in [
        sse2,
        avx2,
        avx512f,
        avx512bw,
        Cpu { bytes: 16, ..avx10 },
        avx10,
    ] {
        for lane in [Lane::I32, Lane::I64] {
            for dst in [Some(0), Some(1), Some(2)] {
                exercise(cpu, lane, Some(0), 1, 2, dst);
                exercise(cpu, lane, None, 1, 2, dst);
            }
            exercise(cpu, lane, Some(0), 0, 0, Some(3));
            exercise(cpu, lane, Some(0), 1, 0, Some(1));
            exercise(cpu, lane, None, 1, 2, None);
            exercise(cpu, lane, Some(0), 1, 2, None);
            exercise(cpu, lane, Some(0), 0, 1, None);
        }
        for lane in [Lane::I8, Lane::I16] {
            exercise(cpu, lane, None, 1, 2, Some(1));
            exercise(cpu, lane, None, 1, 2, None);
        }
    }
}
