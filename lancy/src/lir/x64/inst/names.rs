//! How x64 instructions read in listings: HIR spellings for types,
//! lanes and operations, assembler names for registers.
use super::{Address, Details, Inst, Int, Quotient, Rhs, VectorMask};
use crate::hir::text::{TypeName, binary_name, lane_name, trap_name, unary_name};
use crate::lir::{printer::Named, x64::abi::Machine};
use regalloc2::{PReg, RegClass};

const GPRS: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

fn int(t: Int) -> String {
    format!("{}{}", if t.signed { "i" } else { "u" }, t.bits())
}

fn rhs(arg: Rhs) -> String {
    match arg {
        Rhs::Imm(i) => format!(" imm={i}"),
        Rhs::Reg(_) => String::new(),
    }
}

fn address(a: Address) -> String {
    let scale = a.index.map_or(0, |(_, s)| 1 << s as u8);
    format!("scale={scale} disp={}", a.disp)
}

fn mask_kind<R>(mask: VectorMask<R>) -> &'static str {
    match mask {
        VectorMask::Compare {
            active: Some(_), ..
        } => ".compare.and",
        VectorMask::Compare { .. } => ".compare",
        VectorMask::Register(..) => ".register",
        VectorMask::Lanes(_) => "",
    }
}

fn which(yes: bool, then: &'static str, otherwise: &'static str) -> &'static str {
    if yes { then } else { otherwise }
}

impl Named for Inst {
    fn mnemonic(&self, machine: &Machine, details: &Details) -> String {
        let ty = TypeName;
        match *self {
            Self::Argv(_) => "argv".into(),
            Self::LoadArg { index, .. } => format!("arg {index}"),
            Self::Native(id) => {
                let n = details.natives[id].native;
                format!("native.{}.{}", ty(n.input), n.action)
            }
            Self::Fallback(id) => {
                let e = details.expansions[id];
                format!("fallback.{}.{}", ty(e.input), e.action)
            }
            Self::Literal(lane, n, _) => format!("literal.{} {n:#x}", lane_name(lane)),
            Self::Iota(lane, _) => format!("iota.{}", lane_name(lane)),
            Self::SplatLiteral(lane, n, _) => format!("splat.literal.{} {n:#x}", lane_name(lane)),
            Self::BitMove(from, to, ..) => format!("bitcast.{}.{}", ty(from), ty(to)),
            Self::Imm(_, n) => format!("imm {n}"),
            Self::Copy(..) => "copy".into(),
            Self::Alu(op, t, _, _, arg) => format!("alu.{}.{}{}", int(t), op.name(), rhs(arg)),
            Self::Shift {
                op, int: t, count, ..
            } => format!("{}.{}{}", op.name(), int(t), rhs(count)),
            Self::IntUnary(op, t, ..) => format!("{}.{}", op.name(), int(t)),
            Self::Saturating { sub, int: t, .. } => {
                format!("{}.{}", which(sub, "subsat", "addsat"), int(t))
            }
            Self::Convert { from, to, .. } => {
                format!("convert.{}.{}", lane_name(from), lane_name(to))
            }
            Self::Classify(op, lane, ..) => {
                format!("classify.{} {}", lane_name(lane), unary_name(op))
            }
            Self::Extend(t, ..) => format!("extend.{}", int(t)),
            Self::Compare(cc, bytes, _, _, arg) => {
                format!("cmp.{}.{}{}", cc.name(), bytes * 8, rhs(arg))
            }
            Self::SelectVector { .. } => "select.vector".into(),
            Self::Select { wide, .. } => which(wide, "select.64", "select.32").into(),
            Self::BrCmp(cc, bytes, _, arg) => {
                format!("brcmp.{}.{}{}", cc.name(), bytes * 8, rhs(arg))
            }
            Self::CheckMask(mask) => format!("bounds.mask{}", mask_kind(mask)),
            Self::BranchAny(mask) => format!("branch.any{}", mask_kind(mask)),
            Self::AnyMask(_, mask) => format!("any.mask{}", mask_kind(mask)),
            Self::AllMask(lane, ..) => format!("all.mask.{}", lane_name(lane)),
            Self::SelectMask(id) => format!("select.mask{}", mask_kind(details.selects[id].mask)),
            Self::MaskCompare(lane, cc, ..) => format!("mask.cmp.{} {cc}", lane_name(lane)),
            Self::MaskLogic(lane, op, ..) => {
                format!("mask.{}.{}", binary_name(op), lane_name(lane))
            }
            Self::MaskNot(lane, ..) => format!("mask.not.{}", lane_name(lane)),
            Self::MaskImmediate(_, bits) => format!("mask.imm {bits:#x}"),
            Self::MaskPrefix(..) => "mask.prefix".into(),
            Self::MaskCount(..) => "mask.count".into(),
            Self::CompactStore { .. } => "compress.store".into(),
            Self::CompactLookup(expand, ..) => {
                format!("{}.lookup", which(expand, "expand", "compress"))
            }
            Self::MaskCompact(lane, expand, ..) => format!(
                "mask.{}.{}",
                which(expand, "expand", "compress"),
                lane_name(lane)
            ),
            Self::BranchAll(lane, _) => format!("branch.all.{}", lane_name(lane)),
            Self::CheckBounds(..) => "bounds".into(),
            Self::TrapIf(_, trap) => format!("trap_if {}", trap_name(trap)),
            Self::Trap(trap) => format!("trap {}", trap_name(trap)),
            Self::RustCall(function, ..) => format!("rust_call {}", function.name),
            Self::CallScratch(offset, bytes, _) => format!("call_scratch {offset} {bytes}"),
            Self::DivideHigh { int: t, .. } => format!("divide.high.{}", int(t)),
            Self::Divide {
                remainder, int: t, ..
            } => format!("{}.{}", which(remainder, "rem", "div"), int(t)),
            Self::DivImm {
                remainder,
                divisor,
                quotient,
                ..
            } => {
                let quotient = match quotient {
                    Quotient::PowerOfTwo(shift) => format!("pow2 shift={shift}"),
                    Quotient::Reciprocal { multiplier, shift } => {
                        format!("reciprocal={multiplier:#x} shift={shift}")
                    }
                };
                format!(
                    "{}.imm={divisor} {quotient}",
                    which(remainder, "irem", "idiv")
                )
            }
            Self::Lea(_, a) => format!("lea {}", address(a)),
            Self::Masked(id) => {
                let access = details.masked[id];
                format!(
                    "masked{}.{}{}",
                    which(access.dst.is_some(), "load", "store"),
                    ty(access.ty),
                    mask_kind(access.mask)
                )
            }
            Self::Load(t, _, a) => format!("load.{} {}", ty(t), address(a)),
            Self::PackedMemory(op, ..) => format!("packed.{}.mem", binary_name(op)),
            Self::BroadcastMemory(lane, _, a) => {
                format!("broadcast.{}.mem {}", lane_name(lane), address(a))
            }
            Self::BroadcastLane(_, _, lane) => format!("broadcast.lane {lane}"),
            Self::ExtractLane(lane, _, _, index) => format!("extract.{} {index}", lane_name(lane)),
            Self::ShiftDwords(_, _, count) => format!("shift.dwords {count}"),
            Self::PairedShuffle(form, ..) => {
                let (name, high) = form.parts();
                format!("shuffle.{name}.{}", which(high, "hi", "lo"))
            }
            Self::DynamicLane {
                lane,
                checked,
                replacement,
                ..
            } => format!(
                "{}.{}{}",
                which(replacement.is_some(), "insert", "extract"),
                lane_name(lane),
                which(checked, ".checked", "")
            ),
            Self::VectorShiftImm(op, lane, _, _, count) => {
                format!("packed.{}.{} imm={count}", lane_name(lane), binary_name(op))
            }
            Self::Store(t, a, _) => format!("store.{} {}", ty(t), address(a)),
            Self::DebugStore(t, offset, _) => format!("debug_store.{} {offset}", ty(t)),
            Self::DebugPoint(id) => format!("debug_point {id}"),
            Self::StoreArg(i, ..) => format!("store_arg {i}"),
            Self::Call {
                target, clobbers, ..
            } => {
                let regs = details.clobbers[clobbers]
                    .into_iter()
                    .map(|reg| Self::register(machine, reg));
                format!(
                    "call f{} clobbers=[{}]",
                    target.index(),
                    crate::lir::printer::list(regs)
                )
            }
            Self::Jump => "jump".into(),
            Self::Branch(_) => "branch".into(),
            Self::Ret(_) => "ret".into(),
        }
    }

    fn register(machine: &Machine, reg: PReg) -> String {
        let n = reg.hw_enc();
        match reg.class() {
            RegClass::Int => GPRS[n].into(),
            RegClass::Float => format!("k{n}"),
            RegClass::Vector => {
                let name = match machine.cpu.bytes {
                    16 => "xmm",
                    32 => "ymm",
                    _ => "zmm",
                };
                format!("{name}{n}")
            }
        }
    }
}
