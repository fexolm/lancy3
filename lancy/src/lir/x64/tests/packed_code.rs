use crate::lir::x64::{Cpu, emit::bind, inst::simd};
use iced_x86::{CpuidFeature as Isa, code_asm::*};
use splice_target::X86Features;

pub(super) fn profiles() -> [Cpu; 4] {
    let sse2 = Cpu {
        bytes: 16,
        isa: X86Features {
            ssse3: false,
            sse41: false,
            sse42: false,
            avx2: false,
            avx10: 0,
            bw: false,
            dq: false,
            cd: false,
            bitalg: false,
            vpopcnt: false,
            vbmi2: false,
            vbmi: false,
            fp16: false,
            f16c: false,
            bf16: false,
            fma: false,
            ..X86Features::default()
        },
    };
    let avx2 = Cpu {
        bytes: 32,
        isa: X86Features {
            avx2: true,
            ssse3: true,
            sse41: true,
            sse42: true,
            ..sse2.isa
        },
    };
    let foundation = Cpu { bytes: 64, ..avx2 };
    let extended = Cpu {
        isa: X86Features {
            bw: true,
            dq: true,
            cd: true,
            vpopcnt: true,
            bitalg: true,
            ..foundation.isa
        },
        ..foundation
    };
    [sse2, avx2, foundation, extended]
}

pub(super) fn supports(cpu: Cpu, feature: Isa) -> bool {
    match feature {
        Isa::INTEL8086
        | Isa::INTEL186
        | Isa::INTEL286
        | Isa::INTEL386
        | Isa::INTEL486
        | Isa::X64
        | Isa::CMOV
        | Isa::SSE
        | Isa::SSE2 => true,
        Isa::SSE3 | Isa::SSSE3 => cpu.isa.ssse3,
        Isa::SSE4_1 => cpu.isa.sse41,
        Isa::SSE4_2 => cpu.isa.sse42,
        Isa::AVX => cpu.bytes >= 32 || cpu.isa.avx10 >= 1,
        Isa::AVX2 => cpu.isa.avx2,
        Isa::AVX512F => cpu.bytes == 64 || cpu.isa.avx10 >= 1,
        Isa::AVX512VL => cpu.isa.avx10 >= 1,
        Isa::AVX512DQ => cpu.isa.dq,
        Isa::AVX512BW => cpu.isa.bw,
        Isa::AVX512CD => cpu.isa.cd,
        Isa::AVX512_VBMI => cpu.isa.vbmi,
        Isa::AVX512_VBMI2 => cpu.isa.vbmi2,
        Isa::AVX512_BF16 => cpu.isa.bf16,
        Isa::AVX512_FP16 => cpu.isa.fp16,
        Isa::F16C => cpu.isa.f16c,
        Isa::FMA => cpu.isa.fma,
        Isa::AVX512_BITALG => cpu.isa.bitalg,
        Isa::AVX512_VPOPCNTDQ => cpu.isa.vpopcnt,
        _ => false,
    }
}

pub(super) fn host_supports(feature: Isa) -> bool {
    // A real AVX-512 host can have VL without implementing AVX10. A synthetic
    // foundation-only target must not gain VL from the machine running the test.
    supports(Cpu::host(), feature)
        || feature == Isa::AVX512VL && is_x86_feature_detected!("avx512vl")
}

// Use the production emitter and a relocatable constant pool. The jump around
// that pool lands at the wrapper's first store. Only the prefix is executable.
pub(super) fn body(op: simd::Native, regs: &[usize]) -> (Vec<u8>, usize) {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut pool = Vec::new();
    let temps: Vec<_> = (6..15)
        .filter(|r| !regs.contains(r))
        .take(op.temporaries())
        .collect();
    assert_eq!(temps.len(), op.temporaries());
    op.emit_allocated(&mut a, regs, &temps, &mut pool).unwrap();
    let data_len: usize = pool.iter().map(|(_, _, n)| n).sum();
    if !pool.is_empty() {
        let done = a.create_label();
        a.jmp(done).unwrap();
        for (label, bits, len) in pool {
            bind(&mut a, label).unwrap();
            a.db(&bits[..len]).unwrap();
        }
        bind(&mut a, done).unwrap();
    }
    let bytes = a.assemble(0).unwrap();
    let text_len = bytes.len() - data_len;
    (bytes, text_len)
}

// Slots 0 and 1 hold the inputs to unary/binary operations, 2 holds their result.
// A ternary operation uses slot 2 for its last input and 3 for its result.
// Store the input registers back as well,
// so a sequence which silently destroys a live non-result operand is observable.
pub(super) fn executable(body: &[u8], width: usize, regs: &[usize]) -> Vec<u8> {
    assert!((2..=4).contains(&regs.len()));
    let (dst, inputs) = regs.split_last().unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.endbr64().unwrap();
    let data = if cfg!(windows) { rcx } else { rdi };
    let saved = if cfg!(windows) {
        &simd::XMM[6..16]
    } else {
        &[][..]
    };
    let frame = 8 + 16 * saved.len() as i32;
    a.sub(rsp, frame).unwrap();
    a.mov(qword_ptr(rsp), data).unwrap();
    for (i, &r) in saved.iter().enumerate() {
        a.movdqu(xmmword_ptr(rsp + 8 + 16 * i as i32), r).unwrap();
    }
    for (i, &reg) in inputs.iter().enumerate() {
        if !inputs[..i].contains(&reg) {
            simd::load(&mut a, reg, ptr(data + 64 * i as i32), width).unwrap();
        }
    }
    a.db(body).unwrap();
    a.mov(r10, qword_ptr(rsp)).unwrap();
    simd::store(
        &mut a,
        ptr(r10 + 64 * inputs.len().max(2) as i32),
        *dst,
        width,
    )
    .unwrap();
    for (i, &reg) in inputs.iter().enumerate() {
        simd::store(&mut a, ptr(r10 + 64 * i as i32), reg, width).unwrap();
    }
    for (i, &r) in saved.iter().enumerate() {
        a.movdqu(r, xmmword_ptr(rsp + 8 + 16 * i as i32)).unwrap();
    }
    a.add(rsp, frame).unwrap();
    a.ret().unwrap();
    a.assemble(0).unwrap()
}
