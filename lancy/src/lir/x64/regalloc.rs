//! Conservative linear intervals with backward CFG liveness, including loop
//! backedges. Definitions and uses occupy distinct positions in an instruction.
//! Four volatile registers are assigned by linear scan; remaining values spill.
mod liveness;

use super::frame::{Frame, Home, LINK_SLOTS};
use crate::hir::{Function, Value};
use crate::lir::ssa::SsaBody;
use crate::support::slotmap::DenseMap;

pub(super) struct Allocation {
    pub homes: DenseMap<Value, Home>,
    pub frame: Frame,
}

pub(super) fn allocate(f: &Function, body: &SsaBody, root: bool) -> Allocation {
    let lives = liveness::intervals(body);
    let mut order: Vec<_> = lives
        .iter()
        .filter_map(|(id, life)| life.def_block.map(|_| id))
        .collect();
    order.sort_unstable_by_key(|&id| (lives[id].first, id));
    let status_slot = if root { 0 } else { LINK_SLOTS + f.arity() };
    let mut slots = status_slot + 1;
    let mut occupied_until = [None; 4];
    let mut homes = DenseMap::filled(lives.len(), Home::Slot(0));
    for id in order {
        let live = &lives[id];
        homes[id] = if let Some(r) = occupied_until
            .iter()
            .position(|end| end.is_none_or(|p| p < live.first))
        {
            occupied_until[r] = Some(live.last);
            Home::Register(8 + r as u8)
        } else {
            let slot = slots;
            slots += 1;
            Home::Slot(slot as u32)
        };
    }
    let save_base = slots;
    if !body.calls.is_empty() {
        slots += occupied_until.len();
    }
    let args = body.params[body.entry].iter().map(|v| homes[*v]).collect();
    Allocation {
        homes,
        frame: Frame {
            args,
            slots,
            status_slot,
            save_base,
        },
    }
}
