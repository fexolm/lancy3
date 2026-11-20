use crate::hir::{
    self, Function, Op, Simd, Terminator, Value, ssa::VerifiedBody, transforms::DomTree,
};
use crate::lir::{
    self,
    x64::inst::{Address, Inst, Rhs},
};
use crate::module::{FunctionId, FunctionIds, Module};
use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};
use regalloc2::{Allocation, Edit, Function as _, InstOrEdit, Output};
use std::fmt::{Display, Write};

fn list(items: impl IntoIterator<Item = impl Display>) -> String {
    items
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn v(value: Value) -> String {
    format!("v{}", value.index())
}

pub(super) fn hir(module: &Module<VerifiedBody>, ids: &FunctionIds<Function>) -> String {
    let mut out = String::new();
    for (id, src) in module.functions.iter() {
        let f = src.function;
        writeln!(out, "fn f{} -> {:?}", id.index(), f.return_type()).unwrap();
        let edge = |e| {
            let (b, args) = f.edge(e);
            format!("b{}({})", b.index(), list(args.iter().copied().map(v)))
        };
        for b in f.blocks() {
            let params = f
                .params(b)
                .iter()
                .map(|&x| format!("{}:{:?}", v(x), f.value_type(x)));
            writeln!(out, "b{}({})", b.index(), list(params)).unwrap();
            for i in f.instructions(b) {
                out.push_str("  ");
                if let Some(x) = f.result(i) {
                    write!(out, "{} = ", v(x)).unwrap();
                }
                let op = match *f.op(i) {
                    Op::Const(n) => format!("const {n}"),
                    Op::Copy(x) => format!("copy {}", v(x)),
                    Op::Bin(op, x, y) => format!("{op:?} {} {}", v(x), v(y)),
                    Op::Simd(Simd::LaneCount(lane)) => format!("simd.lanes {lane:?}"),
                    Op::Simd(Simd::Load(l, p)) => format!("simd.load {l:?} {}", v(p)),
                    Op::Simd(Simd::Store(p, x)) => format!("simd.store {} {}", v(p), v(x)),
                    Op::PtrOffset(p, i, s) => format!("offset {} {} {}", v(p), v(i), 1 << s as u8),
                    Op::BoundsCheck(i, n) => format!("bounds {} {}", v(i), v(n)),
                    Op::Simd(Simd::MaskedLoad {
                        address,
                        mask,
                        passthrough: x,
                    })
                    | Op::Simd(Simd::MaskedStore {
                        address,
                        mask,
                        value: x,
                    }) => {
                        let op = if matches!(f.op(i), Op::Simd(Simd::MaskedLoad { .. })) {
                            "maskedload"
                        } else {
                            "maskedstore"
                        };
                        format!(
                            "simd.{op} {} {} {} {}",
                            v(address.base),
                            address.indices.map(v).unwrap_or_else(|| "-".into()),
                            v(mask),
                            v(x)
                        )
                    }
                    Op::Literal(l, n) => format!("Literal({n}) {:?} ", l.scalar()),
                    op @ (Op::Simd(_)
                    | Op::Unary(..)
                    | Op::Ternary(..)
                    | Op::Cast(..)
                    | Op::Bitcast(..)) => {
                        let prefix = if matches!(op, Op::Simd(_)) {
                            "simd."
                        } else {
                            ""
                        };
                        let (m, args) =
                            crate::numbers::Math::from_op(op, |v| f.value_type(v)).unwrap();
                        format!(
                            "{prefix}{:?} {:?} {}",
                            m.action,
                            m.output,
                            list(args.into_iter().flatten().map(v))
                        )
                    }
                    Op::Load(t, p) => format!("load {t:?} {}", v(p)),
                    Op::Store(p, x) => format!("store {} {}", v(p), v(x)),
                    Op::Call(site) => {
                        let (callee, args) = f.call(site);
                        let id = ids[&std::sync::Arc::as_ptr(callee)];
                        format!(
                            "call f{}({})",
                            id.index(),
                            list(args.iter().copied().map(v))
                        )
                    }
                };
                writeln!(out, "{op}").unwrap();
            }
            let term = match *f.terminator(b) {
                Terminator::Return(x) => format!("ret {}", v(x)),
                Terminator::Jump(e) => format!("jump {}", edge(e)),
                Terminator::Branch { cond, yes, no } => {
                    format!("br {} {} | {}", v(cond), edge(yes), edge(no))
                }
            };
            writeln!(out, "  {term}").unwrap();
        }
    }
    out
}

pub(super) fn cfg(module: &Module<VerifiedBody>) -> String {
    let mut out = String::new();
    for (id, src) in module.functions.iter() {
        let cfg = &src.cfg;
        let dom = DomTree::compute(cfg);
        let block = |b: hir::Block| format!("b{}", b.index());
        writeln!(out, "fn f{}", id.index()).unwrap();
        for &b in cfg.reverse_postorder() {
            let preds = cfg.predecessors(b).iter().copied().map(block);
            let successors = cfg.successors(b).map(block);
            let dominators = dom
                .preorder()
                .iter()
                .copied()
                .filter(|&d| dom.dominates(d, b))
                .map(block);
            writeln!(
                out,
                "{} preds=[{}] succs=[{}] dom=[{}]",
                block(b),
                list(preds),
                list(successors),
                list(dominators)
            )
            .unwrap();
        }
    }
    out
}

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
        Inst::Native(n, ..) => format!("native.{:?}.{:?}", n.math.input, n.math.action),
        Inst::Lanes(m, ..) => format!("lanes.{:?}.{:?}", m.input, m.action),
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
        Inst::BrCmp(cc, _, arg) => format!("brcmp.{cc:?}{}", rhs(arg)),
        Inst::CheckMask(_) => "bounds.mask".into(),
        Inst::BranchAny(_) => "branch.any".into(),
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
