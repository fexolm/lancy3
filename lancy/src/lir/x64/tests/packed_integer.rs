use super::packed_code::{body, executable, host_supports, profiles, supports};
use crate::{
    hir::{Binary as B, Lane, Type, Unary as U},
    hir::{eval::evaluate, operation::Action},
    jit::packed_test,
    lir::x64::{Cpu, inst::simd::Native},
};
use iced_x86::{Decoder, DecoderOptions, Mnemonic};
use splice_target::X86Features;

const INTS: [Lane; 8] = [
    Lane::I8,
    Lane::U8,
    Lane::I16,
    Lane::U16,
    Lane::I32,
    Lane::U32,
    Lane::I64,
    Lane::U64,
];

fn operands(action: Action, lane: Lane, width: usize, seed: usize) -> [[u8; 64]; 3] {
    let mut slots = [[0u8; 64]; 3];
    let size = lane.bytes();
    let bits = 8 * size;
    let mask = u64::MAX >> (64 - bits);
    let sign = 1u64 << (bits - 1);
    let xs = [
        0,
        sign,
        sign - 1,
        mask,
        1,
        0x8017_7bad_eca1_f7fe,
        0xfa41_7231_0bee_3e4e,
        mask - 1,
        0x3527_0319_8000_0000,
        0x3527_0319_7fff_ffff,
    ];
    let ys = [
        mask,
        sign - 1,
        sign,
        0,
        0x3000_0013_cabe_2d51,
        0xd531_fa11_1f37_fbad,
        1,
        mask - 1,
        0x3527_0319_7fff_ffff,
        0x3527_0319_8000_0000,
    ];
    let counts = [
        0,
        bits as u64,
        bits as u64 + 1,
        2 * bits as u64 - 1,
        mask,
        1,
        bits as u64 - 1,
        sign,
    ];
    let shift = matches!(
        action,
        Action::Binary(B::Shl | B::Shr | B::RotateLeft | B::RotateRight)
    );
    let saturating = matches!(action, Action::Binary(B::SaturatingAdd | B::SaturatingSub));
    for i in 0..width / size {
        let j = (seed + i) % xs.len();
        let (x, y) = if saturating {
            (
                [sign - 1, sign, sign - 1, sign, mask, 0, 3, mask - 2][j % 8],
                [1, mask, mask, 1, 1, 1, 4, 5][j % 8],
            )
        } else {
            (
                xs[j],
                if shift {
                    counts[j % counts.len()]
                } else {
                    ys[j]
                },
            )
        };
        let p = i * size..(i + 1) * size;
        slots[0][p.clone()].copy_from_slice(&x.to_le_bytes()[..size]);
        slots[1][p].copy_from_slice(&y.to_le_bytes()[..size]);
    }
    slots
}

fn exercise(cpu: Cpu, lane: Lane, action: Action) {
    let ty = Type::Vector(lane);
    let output = match action {
        Action::Binary(op) => op.result_type(ty),
        _ => ty,
    };
    let op = Native::select(action, ty, output, cpu)
        .unwrap_or_else(|| panic!("missing packed {action:?} for {cpu:?}"));
    let binary = matches!(action, Action::Binary(_));
    let unary: &[&[usize]] = &[&[1, 1], &[1, 3]];
    let distinct: &[&[usize]] = &[&[1, 2, 1], &[1, 2, 2], &[1, 2, 3], &[1, 1, 1], &[1, 1, 3]];
    for &regs in if binary { distinct } else { unary } {
        let dst = *regs.last().unwrap();
        if op.tied() && dst != regs[0]
            || op.separate_output() && regs[..regs.len() - 1].contains(&dst)
        {
            continue;
        }
        let (bytes, text_len) = body(op, regs);
        let host = Cpu::host();
        let mut can_run = cpu.bytes <= host.bytes;
        let mut packed_operation = false;
        for inst in Decoder::new(64, &bytes[..text_len], DecoderOptions::NONE) {
            assert!(!inst.is_invalid(), "{action:?} {regs:?}");
            assert_ne!(inst.mnemonic(), Mnemonic::Call, "{action:?} {regs:?}");
            let copies = matches!(
                inst.mnemonic(),
                Mnemonic::Movaps
                    | Mnemonic::Vmovaps
                    | Mnemonic::Movdqa
                    | Mnemonic::Vmovdqa
                    | Mnemonic::Movups
                    | Mnemonic::Vmovups
            );
            if !copies {
                packed_operation |= (0..inst.op_count())
                    .filter(|&i| inst.op_register(i).is_vector_register())
                    .count()
                    >= 2;
            }
            for &f in inst.cpuid_features() {
                assert!(
                    supports(cpu, f),
                    "{action:?} for {cpu:?} emitted {:?} requiring {f:?}",
                    inst.code()
                );
                can_run &= host_supports(f);
            }
        }
        // With equal inputs a comparison or min/max may be a copy or constant.
        if (!binary || regs[0] != regs[1])
            && !matches!(action,Action::Unary(U::Abs) if !lane.signed())
        {
            assert!(
                packed_operation,
                "{action:?} width {} contains only moves or scalar operations",
                cpu.bytes
            );
        }
        if !can_run {
            continue;
        }
        let code = executable(&bytes, cpu.bytes, regs);
        // SAFETY: the wrapper preserves the host ABI and touches only these
        // live slots. The decoder checked each arithmetic instruction's ISA.
        unsafe {
            packed_test::with_kernel(&code, |call| {
                for seed in 0..10 {
                    let mut slots = operands(action, lane, cpu.bytes, seed);
                    if binary && regs[0] == regs[1] {
                        slots[1] = slots[0];
                    }
                    let before = slots;
                    let expected = evaluate(action, ty, output, &before, cpu.bytes).unwrap();
                    call(&mut slots);
                    assert_eq!(
                        slots[2], expected,
                        "{action:?} width {} {regs:?} seed {seed}",
                        cpu.bytes
                    );
                    for (i, &reg) in regs[..regs.len() - 1].iter().enumerate() {
                        if reg != dst {
                            assert_eq!(
                                slots[i], before[i],
                                "{action:?} clobbered source {i}, regs {regs:?}"
                            );
                        }
                    }
                }
            })
        }
    }
}

fn arithmetic_and_ordering(cpu: Cpu, lane: Lane) {
    for op in [
        B::Add,
        B::Sub,
        B::Mul,
        B::SaturatingAdd,
        B::SaturatingSub,
        B::Eq,
        B::Ne,
        B::Lt,
        B::Le,
        B::Gt,
        B::Ge,
        B::Min,
        B::Max,
    ] {
        exercise(cpu, lane, Action::Binary(op));
    }
    exercise(cpu, lane, Action::Unary(U::Neg));
    exercise(cpu, lane, Action::Unary(U::Abs));
}

#[test]
fn packed_arithmetic_and_ordering_obey_each_isa_and_operand_alias() {
    let [sse2, avx2, foundation, extended] = profiles();
    // SSE4.1 has PCMPEQQ and PMULLD but signed PCMPGTQ requires SSE4.2.
    let sse41 = Cpu {
        isa: X86Features {
            sse41: true,
            ssse3: true,
            ..sse2.isa
        },
        ..sse2
    };
    let sse42 = Cpu {
        isa: X86Features {
            sse42: true,
            ..sse41.isa
        },
        ..sse41
    };
    for cpu in [sse2, sse41, sse42, avx2, foundation, extended] {
        for lane in INTS {
            if cpu.bytes != 64 || lane.bytes() >= 4 || cpu.isa.bw {
                arithmetic_and_ordering(cpu, lane);
            }
        }
    }
}

#[test]
fn avx512f_without_bw_computes_byte_and_word_lanes_in_both_halves() {
    let foundation = profiles()[2];
    for lane in [Lane::I8, Lane::U8, Lane::I16, Lane::U16] {
        arithmetic_and_ordering(foundation, lane);
    }
}

#[test]
fn variable_shifts_and_rotates_wrap_each_count_to_its_lane() {
    for cpu in profiles().into_iter().filter(|p| p.bytes >= 32) {
        for lane in [
            Lane::I16,
            Lane::U16,
            Lane::I32,
            Lane::U32,
            Lane::I64,
            Lane::U64,
        ] {
            if lane.bytes() == 2 && !cpu.isa.bw {
                continue;
            }
            for op in [B::Shl, B::Shr, B::RotateLeft, B::RotateRight] {
                exercise(cpu, lane, Action::Binary(op));
            }
        }
    }
}

#[test]
fn packed_bit_counts_handle_zero_and_the_top_bit() {
    let [sse2, avx2, foundation, extended] = profiles();
    let cd = Cpu {
        isa: X86Features {
            cd: true,
            ..foundation.isa
        },
        ..foundation
    };
    let vpopcnt = Cpu {
        isa: X86Features {
            vpopcnt: true,
            ..foundation.isa
        },
        ..foundation
    };
    let bw = Cpu {
        isa: X86Features {
            bw: true,
            ..foundation.isa
        },
        ..foundation
    };
    let bitalg = Cpu {
        isa: X86Features {
            bitalg: true,
            ..bw.isa
        },
        ..bw
    };
    for cpu in [sse2, avx2, foundation, cd, vpopcnt, bw, bitalg, extended] {
        for lane in [Lane::U8, Lane::U16, Lane::U32, Lane::U64] {
            let ty = Type::Vector(lane);
            for op in [U::LeadingZeros, U::TrailingZeros, U::Popcnt] {
                let action = Action::Unary(op);
                let required = match op {
                    U::LeadingZeros => lane.bytes() >= 4 && cpu.isa.cd,
                    _ if lane.bytes() >= 4 => cpu.isa.vpopcnt,
                    _ => cpu.isa.bitalg && cpu.isa.bw,
                };
                // An alternative packed expansion is valid on a lower ISA;
                // decode it there too to catch accidental AVX-512 instructions.
                if required || Native::select(action, ty, ty, cpu).is_some() {
                    exercise(cpu, lane, action);
                }
            }
        }
    }
}

// Force regalloc to keep packed inputs live beyond multi-instruction recipes.
// A temporary that aliases an unrelated value can corrupt an otherwise correct
// instruction sequence only when it is installed as part of a larger kernel.
#[test]
fn packed_integer_temporaries_preserve_live_inputs_under_pressure() {
    use crate::{
        ScalarKernel,
        hir::{Function, Op, Terminator, Ternary},
    };

    const ARITY: usize = 18;
    let v = Type::Vector(Lane::U64);
    let mut f = Function::new(&[v; ARITY], v);
    let block = f.entry();
    let args = f.params(block).to_vec();
    let mut acc = args[0];
    for pair in args.windows(2) {
        let [x, y] = [pair[0], pair[1]];
        let mut emit = |op| f.append(block, op).unwrap();
        let product = emit(Op::Bin(B::Mul, x, y));
        let rotated = emit(Op::Bin(B::RotateLeft, product, y));
        let neg = emit(Op::Unary(U::Neg, y));
        let lt = emit(Op::Bin(B::Lt, x, y));
        let term = emit(Op::Ternary(Ternary::Select, lt, rotated, neg));
        acc = emit(Op::Bin(B::Xor, acc, term));
    }
    for &x in &args {
        acc = f.append(block, Op::Bin(B::Add, acc, x)).unwrap();
    }
    f.set_terminator(block, Terminator::Return(acc));
    let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
    let lanes = crate::test_target::width() / 8;
    let inputs: Vec<u64> = (0..ARITY * lanes)
        .map(|i| {
            (i as u64 + 1)
                .wrapping_mul(0x79d9_91f2_c53a_df1d)
                .rotate_right((i % 61) as u32)
        })
        .collect();
    let mut actual = vec![0; lanes];
    k.call_words(&inputs, &mut actual).unwrap();
    for lane in 0..lanes {
        let args: Vec<_> = inputs[lane..].iter().step_by(lanes).copied().collect();
        let mut expected = args[0];
        for pair in args.windows(2) {
            let [x, y] = [pair[0], pair[1]];
            expected ^= if x < y {
                x.wrapping_mul(y).rotate_left(y as u32)
            } else {
                y.wrapping_neg()
            };
        }
        expected = args.into_iter().fold(expected, u64::wrapping_add);
        assert_eq!(actual[lane], expected);
    }
}
