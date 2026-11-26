//! VEX re-encoding of legacy SSE instructions (X64-02).
//!
//! Emitters write 16-byte and scalar forms once, in SSE2 spelling. With AVX a
//! legacy SSE instruction after a 256- or 512-bit one pays a state transition
//! or a false dependency on the upper lanes, so the assembler re-encodes every
//! such instruction in its VEX form before encoding. A VEX form reads the
//! legacy destination through `vvvv` and the implicit `XMM0` blend mask through
//! its `is4` operand, so the operation is the same. It also zeroes the lanes
//! above 128 bits, which no emitter relies on, and lifts the 16-byte alignment
//! requirement of legacy memory operands.
use iced_x86::{
    Code, EncodingKind, Instruction, Mnemonic, OpCodeOperandKind as K, OpKind, Register,
};
use std::{collections::HashMap, sync::OnceLock};

/// The VEX form of `instruction`, or `None` when it is not legacy SSE.
pub(super) fn encode(instruction: &Instruction) -> Option<Instruction> {
    forms()[instruction.code() as usize]
        .iter()
        .find_map(|&code| convert(instruction, code))
}

/// The 128-bit VEX codes with the `V`-prefixed mnemonic of each legacy SSE
/// code, indexed by `Code`.
fn forms() -> &'static [Box<[Code]>] {
    static FORMS: OnceLock<Box<[Box<[Code]>]>> = OnceLock::new();
    FORMS.get_or_init(|| {
        let mut vex: HashMap<String, Vec<Code>> = HashMap::new();
        for code in Code::values() {
            let op = code.op_code();
            if op.encoding() == EncodingKind::VEX && (op.is_lig() || op.l() == 0) {
                vex.entry(name(code.mnemonic())).or_default().push(code);
            }
        }
        Code::values()
            .map(|code| {
                let op = code.op_code();
                let sse = op.encoding() == EncodingKind::Legacy
                    && op
                        .op_kinds()
                        .iter()
                        .any(|k| matches!(k, K::xmm_reg | K::xmm_rm | K::xmm_or_mem));
                match vex.get(&format!("v{}", name(code.mnemonic()))) {
                    Some(codes) if sse => codes.clone().into_boxed_slice(),
                    _ => Box::default(),
                }
            })
            .collect()
    })
}

fn name(mnemonic: Mnemonic) -> String {
    format!("{mnemonic:?}").to_ascii_lowercase()
}

/// `instruction` as `code`, if the operands line up: the VEX form inserts a
/// `vvvv` copy of the destination and may name the implicit blend mask.
fn convert(instruction: &Instruction, code: Code) -> Option<Instruction> {
    let legacy = instruction.code().op_code().op_kinds();
    let kinds = code.op_code().op_kinds();
    let mut out = *instruction;
    out.set_code(code);
    let mut next = 0;
    for (i, &kind) in kinds.iter().enumerate() {
        let i = i as u32;
        let register = match kind {
            K::xmm_vvvv => Some(instruction.op0_register()),
            K::xmm_is4 if next == legacy.len() => Some(Register::XMM0),
            _ => {
                let from = *legacy.get(next)?;
                let operand = instruction.op_kind(next as u32);
                let memory = operand == OpKind::Memory;
                let fits = from == kind
                    || (from == K::xmm_or_mem && kind == if memory { K::mem } else { K::xmm_rm });
                if !fits {
                    return None;
                }
                next += 1;
                out.set_op_kind(i, operand);
                (operand == OpKind::Register).then(|| instruction.op_register(next as u32 - 1))
            }
        };
        if let Some(register) = register {
            out.set_op_kind(i, OpKind::Register);
            out.set_op_register(i, register);
        }
    }
    (next == legacy.len()).then_some(out)
}
