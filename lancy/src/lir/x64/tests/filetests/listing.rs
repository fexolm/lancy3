use crate::filetests::listing::list;
use crate::lir::{
    self,
    x64::inst::{Address, Inst, Rhs},
};
use crate::module::FunctionId;
use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};
use regalloc2::{Allocation, Edit, Function as _, InstOrEdit, Output};
use std::fmt::Write;

fn opcode(inst: Inst) -> String {
    let rhs = |arg| match arg {
        Rhs::Imm(i) => format!(" imm={i}"),
        Rhs::Reg(_) => String::new(),
    };
    let addr = |a: Address| {
        format!(
            "scale={} disp={}",
            a.index.map_or(0, |(_, s)| 1 << s as u8),
            a.disp
        )
    };
    match inst {
        Inst::Argv(_) => "argv".into(),
        Inst::LoadArg { index, .. } => format!("arg {index}"),
        Inst::Native(n, ..) => format!("native.{:?}.{:?}", n.input, n.action),
        Inst::Lanes { input, action, .. } => format!("lanes.{input:?}.{action:?}"),
        Inst::Literal(l, n, _) => format!("literal.{l:?} {n:#x}"),
        Inst::Iota(l, _) => format!("iota.{l:?}"),
        Inst::BitMove(from, to, ..) => format!("bitcast.{from:?}.{to:?}"),
        Inst::IntBinary(op, ty, ..) => format!("alu.{ty:?}.{op:?}"),
        Inst::Imm(_, n) => format!("imm {n}"),
        Inst::Copy(..) => "copy".into(),
        Inst::Alu(op, _, _, arg) => format!("alu.{op:?}{}", rhs(arg)),
        Inst::Shift { left, count, .. } => {
            format!("{}{}", if left { "shl" } else { "sar" }, rhs(count))
        }
        Inst::Compare(cc, _, _, arg) => format!("cmp.{cc:?}{}", rhs(arg)),
        Inst::SelectI32 { .. } => "select.i32".into(),
        Inst::BrCmp(cc, _, arg) => format!("brcmp.{cc:?}{}", rhs(arg)),
        Inst::CheckMask(_) => "bounds.mask".into(),
        Inst::BranchAny(_) => "branch.any".into(),
        Inst::BranchAll(lane, _) => format!("branch.all.{lane:?}"),
        Inst::CheckBounds(..) => "bounds".into(),
        Inst::Cdq { .. } => "cdq".into(),
        Inst::IDiv { remainder, .. } => if remainder { "irem" } else { "idiv" }.into(),
        Inst::DivImm {
            remainder,
            divisor,
            quotient,
            ..
        } => format!(
            "{}.imm={divisor} {quotient:?}",
            if remainder { "irem" } else { "idiv" },
        ),
        Inst::Movsxd(..) => "movsxd".into(),
        Inst::Lea(_, a) => format!("lea {}", addr(a)),
        Inst::Masked { ty, dst, .. } => format!(
            "masked{}.{ty:?}",
            if dst.is_some() { "load" } else { "store" }
        ),
        Inst::Load(t, _, a) => format!("load.{t:?} {}", addr(a)),
        Inst::PackedMemory(kind, _, _, _, a) => format!("packed.{kind:?}.mem {}", addr(a)),
        Inst::BroadcastMemory(lane, _, a) => format!("broadcast.{lane:?}.mem {}", addr(a)),
        Inst::BroadcastLane(_, _, lane) => format!("broadcast.lane {lane}"),
        Inst::Store(t, a, _) => format!("store.{t:?} {}", addr(a)),
        Inst::StoreArg(i, ..) => format!("store_arg {i}"),
        Inst::Call {
            target, clobbers, ..
        } => format!("call f{} clobbers={clobbers:?}", target.index()),
        Inst::Jump => "jump".into(),
        Inst::Branch(_) => "branch".into(),
        Inst::Ret(_) => "ret".into(),
    }
}

fn location(a: Allocation) -> String {
    const REGS: [&str; 16] = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15",
    ];
    match a.as_reg() {
        Some(r) if r.class() == regalloc2::RegClass::Vector => format!(
            "{}{}",
            match crate::ScalarKernel::vector_bytes() {
                16 => "xmm",
                32 => "ymm",
                _ => "zmm",
            },
            r.hw_enc()
        ),
        Some(r) => REGS[r.hw_enc()].into(),
        None => format!("spill[{}]", a.as_stack().unwrap().index()),
    }
}

pub(super) fn lir(
    out: &mut String,
    id: FunctionId,
    f: &lir::Function<Inst>,
    allocation: Option<&Output>,
) {
    writeln!(out, "fn f{}", id.index()).unwrap();
    if let Some(a) = allocation {
        writeln!(out, "spill_slots={}", a.num_spillslots).unwrap();
    }
    for i in 0..f.num_blocks() {
        let b = regalloc2::Block::new(i);
        writeln!(out, "b{i}({})", list(f.block_params(b))).unwrap();
        let inst = |i: regalloc2::Inst| opcode(f.insts()[i.index()]);
        if let Some(a) = allocation {
            for item in a.block_insts_and_edits(f, b) {
                match item {
                    InstOrEdit::Edit(Edit::Move { from, to }) => {
                        writeln!(out, "  move {} -> {}", location(*from), location(*to)).unwrap()
                    }
                    InstOrEdit::Inst(i) => writeln!(
                        out,
                        "  {} | {}",
                        inst(i),
                        list(a.inst_allocs(i).iter().copied().map(location))
                    )
                    .unwrap(),
                }
            }
        } else {
            for i in f.block_insns(b).iter() {
                writeln!(out, "  {} | {}", inst(i), list(f.inst_operands(i))).unwrap();
            }
        }
        for (n, target) in f.block_succs(b).iter().enumerate() {
            writeln!(
                out,
                "  -> b{}({})",
                target.index(),
                list(f.branch_blockparams(b, f.block_insns(b).last(), n))
            )
            .unwrap();
        }
    }
}

pub(super) fn asm(bytes: &[u8]) -> Result<String, String> {
    let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    let mut out = String::new();
    while decoder.can_decode() {
        let start = decoder.position();
        // iced 1.22 has no AVX10.2 MAP5 conversion entries. Recognize exactly the
        // unmasked register forms emitted by this backend; leave all other validation to iced.
        if let [0x62, p0, p1, p2, op, modrm, ..] = bytes[start..]
            && p0 & 0x5f == 0x55
            && matches!(p1, 0x7c | 0xfd)
            && matches!(p2, 0x08 | 0x28 | 0x48)
            && matches!(op, 0x6c | 0x6d)
            && modrm >> 6 == 3
        {
            let size = match p2 {
                0x08 => "xmm",
                0x28 => "ymm",
                _ => "zmm",
            };
            let from = if p1 == 0x7c { "ps" } else { "pd" };
            let to = if p1 == 0x7c { "dq" } else { "qq" };
            let unsigned = if op == 0x6c { "u" } else { "" };
            let d = (!p0 >> 4 & 8) | (modrm >> 3 & 7);
            let r = (!p0 >> 2 & 8) | (modrm & 7);
            let encoding = list(bytes[start..start + 6].iter().map(|b| format!("{b:02x}")));
            writeln!(
                out,
                "{start:04x}  {encoding:<32} vcvtt{from}2{unsigned}{to}s {size}{d},{size}{r}"
            )
            .unwrap();
            decoder.set_position(start + 6).map_err(|e| e.to_string())?;
            decoder.set_ip((start + 6) as u64);
            continue;
        }
        let inst = decoder.decode();
        if inst.is_invalid() {
            return Err(format!("invalid x64 instruction at byte {}", inst.ip()));
        }
        let start = inst.ip() as usize;
        let encoding = list(
            bytes[start..start + inst.len()]
                .iter()
                .map(|b| format!("{b:02x}")),
        );
        write!(out, "{start:04x}  {encoding:<32} ").unwrap();
        formatter.format(&inst, &mut out);
        out.push('\n');
    }
    Ok(out)
}
