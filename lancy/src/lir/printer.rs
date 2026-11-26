//! The text of selected and allocated LIR: what the filetest stages read.
//! Blocks, operands and register moves print the same on every backend;
//! each backend names its instructions and registers.
use super::{Function, TargetInst};
use crate::module::FunctionId;
use regalloc2::{Allocation, Block, Edit, Function as _, InstOrEdit, Output, PReg};
use std::fmt::{Display, Write};

/// How a backend spells its instructions in a listing.
pub(super) trait Named: TargetInst {
    /// The instruction's stable name and its attributes that are not
    /// operands: `alu.sub`, `load.i32 scale=4 disp=0`.
    fn mnemonic(&self, machine: &Self::Machine, details: &Self::Details) -> String;
    /// The assembler name of a register.
    fn register(machine: &Self::Machine, reg: PReg) -> String;
}

pub(super) fn list(items: impl IntoIterator<Item = impl Display>) -> String {
    items
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// `f` as function `f{id}`: each instruction with its operands, or with
/// their locations and the moves between them once `allocation` is known.
pub(super) fn function<I: Named>(
    id: FunctionId,
    f: &Function<I>,
    allocation: Option<&Output>,
) -> String {
    let mut out = String::new();
    let location = |a: Allocation| match (a.as_reg(), a.as_stack()) {
        (Some(reg), _) => I::register(f.machine(), reg),
        (_, Some(slot)) => format!("spill[{}]", slot.index()),
        _ => "none".into(),
    };
    let mnemonic = |i: regalloc2::Inst| f.insts()[i.index()].mnemonic(f.machine(), f.details());
    writeln!(out, "fn f{}", id.index()).unwrap();
    if let Some(a) = allocation {
        writeln!(out, "spill_slots={}", a.num_spillslots).unwrap();
    }
    for b in (0..f.num_blocks()).map(Block::new) {
        writeln!(out, "b{}({})", b.index(), list(f.block_params(b))).unwrap();
        if let Some(a) = allocation {
            for item in a.block_insts_and_edits(f, b) {
                match item {
                    InstOrEdit::Edit(Edit::Move { from, to }) => {
                        writeln!(out, "  move {} -> {}", location(*from), location(*to)).unwrap()
                    }
                    InstOrEdit::Inst(i) => {
                        let allocs = a.inst_allocs(i).iter().copied().map(location);
                        writeln!(out, "  {} | {}", mnemonic(i), list(allocs)).unwrap()
                    }
                }
            }
        } else {
            for i in f.block_insns(b).iter() {
                writeln!(out, "  {} | {}", mnemonic(i), list(f.inst_operands(i))).unwrap();
            }
        }
        for (n, target) in f.block_succs(b).iter().enumerate() {
            let args = f.branch_blockparams(b, f.block_insns(b).last(), n);
            writeln!(out, "  -> b{}({})", target.index(), list(args)).unwrap();
        }
    }
    out
}
