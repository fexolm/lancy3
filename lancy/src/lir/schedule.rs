//! Bounded pre-allocation list scheduling on the selected SSA instructions.
use super::{Function, TargetInst};
use regalloc2::{Block, Function as _, Inst, Operand, OperandConstraint, OperandKind, OperandPos};
use std::{cmp::Reverse, ops::Range};

mod liveness;
#[cfg(test)]
mod tests;
use liveness::Liveness;

const WINDOW: usize = 64;
const MIN_REGION: usize = 8;
const READY_LIMIT: usize = 12;
// The timing model only ranks orders within one region. Four issue slots per
// modeled cycle let independent operations cover a producer's latency.
const ISSUE: u16 = 4;
type Pressure = [u16; 3];

#[derive(Clone, Copy)]
pub(super) enum Scheduling {
    Compute(u8),
    /// Memory accesses and checks share one order, independent of SSA uses.
    Ordered(u8),
    Barrier,
}

pub(super) trait Schedulable {
    fn scheduling(&self) -> Scheduling;
}

fn members(mut mask: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        let i = (mask != 0).then(|| mask.trailing_zeros() as usize)?;
        mask &= mask - 1;
        Some(i)
    })
}

#[derive(Clone, Copy, Default)]
struct Value {
    generation: usize,
    uses: u64,
    def: u64,
    live_out: bool,
}

#[derive(Clone, Copy, Default)]
struct Node {
    data_pred: u64,
    ordered_pred: u64,
    successors: u64,
    latency: u16,
    height: u16,
}

struct Region<'a, I: TargetInst> {
    f: &'a Function<I>,
    start: usize,
    nodes: &'a [Node],
    values: &'a [Value],
    live_in: Pressure,
}

#[derive(Default)]
struct Score {
    peak: Pressure,
    area: u32,
    copies: usize,
    completion: u16,
}

struct Step {
    pressure: Pressure,
    peak: Pressure,
    copies: usize,
    issue_at: u16,
}

fn excess(pressure: Pressure, limit: Pressure) -> u16 {
    (0..3).map(|c| pressure[c].saturating_sub(limit[c])).sum()
}

impl<I: TargetInst> Region<'_, I> {
    fn step(&self, node: usize, state: &State) -> Step {
        let ops = self.f.inst_operands(Inst::new(self.start + node));
        let mut pressure = state.pressure;
        let mut early_defs = [0; 3];
        let mut dead_late_defs = [0; 3];
        let mut copies = 0;
        let bit = 1 << node;
        for (i, op) in ops.iter().enumerate() {
            let reg = op.vreg();
            let value = self.values[reg.vreg()];
            let c = reg.class() as usize;
            if op.kind() == OperandKind::Use {
                if !value.live_out
                    && value.uses & state.pending == bit
                    && !ops[..i]
                        .iter()
                        .any(|o| o.vreg() == reg && o.kind() == OperandKind::Use)
                {
                    pressure[c] -= 1;
                }
                continue;
            }
            let result_live = value.live_out || value.uses & !bit != 0;
            pressure[c] += u16::from(result_live);
            if op.pos() == OperandPos::Early {
                early_defs[c] += 1;
            } else if !result_live {
                dead_late_defs[c] += 1;
            }
            if let OperandConstraint::Reuse(input) = op.constraint() {
                let input = self.values[ops[input].vreg().vreg()];
                copies += usize::from(input.live_out || input.uses & state.pending & !bit != 0);
            }
        }
        let peak = std::array::from_fn(|c| {
            (state.pressure[c] + early_defs[c]).max(pressure[c] + dead_late_defs[c])
        });
        Step {
            pressure,
            peak,
            copies,
            issue_at: members(self.nodes[node].data_pred)
                .map(|i| state.finished[i])
                .fold(state.slot, u16::max),
        }
    }

    fn schedule(&self, hard_limit: Pressure, order: &mut Vec<usize>) -> bool {
        let soft_limit = hard_limit.map(|n| n.saturating_sub(2));
        let mut state = State::new(self.nodes.len(), self.live_in);
        let mut ready = 0u64;
        for (n, node) in self.nodes.iter().enumerate() {
            if node.data_pred | node.ordered_pred == 0 {
                ready |= 1 << n;
            }
        }
        order.clear();
        let mut changed = false;
        while ready != 0 {
            let (n, step) = if ready.is_power_of_two() {
                let n = ready.trailing_zeros() as usize;
                (n, self.step(n, &state))
            } else {
                members(ready)
                    .take(READY_LIMIT)
                    .map(|n| (n, self.step(n, &state)))
                    .min_by_key(|(n, step)| {
                        (
                            excess(step.peak, hard_limit),
                            excess(step.pressure, soft_limit),
                            step.issue_at,
                            Reverse(self.nodes[*n].height),
                            step.copies,
                            step.pressure.iter().sum::<u16>(),
                            *n,
                        )
                    })
                    .unwrap()
            };
            let node = &self.nodes[n];
            state.emit(n, node, step);
            ready &= !(1 << n);
            changed |= n != order.len();
            order.push(n);
            for next in members(node.successors) {
                let next_node = &self.nodes[next];
                if (next_node.data_pred | next_node.ordered_pred) & state.pending == 0 {
                    ready |= 1 << next;
                }
            }
        }

        debug_assert_eq!(order.len(), self.nodes.len());
        if !changed {
            return false;
        }
        let mut original = State::new(self.nodes.len(), self.live_in);
        for (n, node) in self.nodes.iter().enumerate() {
            let step = self.step(n, &original);
            original.emit(n, node, step);
        }
        let old = original.score;
        let new = state.score;
        // Unknown live-through values add a constant within the window. Never
        // trade a higher peak in one register class for gains in another.
        new.peak.iter().zip(old.peak).all(|(&a, b)| a <= b)
            && new.copies <= old.copies
            && (excess(new.peak, hard_limit) < excess(old.peak, hard_limit)
                || (new.completion <= old.completion && new.area < old.area)
                || (new.completion + ISSUE <= old.completion && new.area <= old.area))
    }
}

struct State {
    pending: u64,
    pressure: Pressure,
    slot: u16,
    finished: [u16; WINDOW],
    score: Score,
}

impl State {
    fn new(len: usize, pressure: Pressure) -> Self {
        Self {
            pending: u64::MAX >> (WINDOW - len),
            pressure,
            slot: 0,
            finished: [0; WINDOW],
            score: Score {
                peak: pressure,
                ..Score::default()
            },
        }
    }

    fn emit(&mut self, n: usize, node: &Node, step: Step) {
        self.pending &= !(1 << n);
        self.pressure = step.pressure;
        self.slot = step.issue_at + 1;
        self.finished[n] = step.issue_at + node.latency * ISSUE;
        self.score.completion = self.score.completion.max(self.finished[n]);
        self.score.copies += step.copies;
        self.score.area += self.pressure.iter().map(|&n| u32::from(n)).sum::<u32>();
        for (high, peak) in self.score.peak.iter_mut().zip(step.peak) {
            *high = (*high).max(peak);
        }
    }
}

struct Workspace {
    values: Vec<Value>,
    registers: Vec<regalloc2::VReg>,
    nodes: [Node; WINDOW],
    order: Vec<usize>,
    liveness: Option<Liveness>,
    limit: Pressure,
}

impl Workspace {
    fn reorder<I: TargetInst + Schedulable>(
        &mut self,
        f: &Function<I>,
        range: Range<usize>,
        block: Block,
    ) -> bool {
        let nodes = &mut self.nodes[..range.len()];
        nodes.fill(Node::default());
        self.registers.clear();
        let generation = range.start + 1;
        let mut ordered: Option<usize> = None;
        for (n, i) in range.clone().enumerate() {
            let bit = 1 << n;
            nodes[n].latency = u16::from(match f.insts()[i].scheduling() {
                Scheduling::Ordered(latency) => {
                    if let Some(p) = ordered {
                        nodes[n].ordered_pred = 1 << p;
                        nodes[p].successors |= bit;
                    }
                    ordered = Some(n);
                    latency
                }
                Scheduling::Compute(latency) => latency,
                Scheduling::Barrier => unreachable!(),
            });
            for op in f.inst_operands(Inst::new(i)) {
                let reg = op.vreg();
                let value = &mut self.values[reg.vreg()];
                if value.generation != generation {
                    *value = Value {
                        generation,
                        ..Value::default()
                    };
                    self.registers.push(reg);
                }
                if op.kind() == OperandKind::Def {
                    value.def = bit;
                } else {
                    value.uses |= bit;
                    nodes[n].data_pred |= value.def & !bit;
                    for p in members(value.def & !bit) {
                        nodes[p].successors |= bit;
                    }
                }
            }
        }
        if (1..nodes.len())
            .all(|n| (nodes[n].data_pred | nodes[n].ordered_pred) & (1 << (n - 1)) != 0)
        {
            return false;
        }
        // A region with a fixed dependency order does not need pressure
        // estimates. Build live-out information for the first movable region.
        let liveness = self.liveness.get_or_insert_with(|| Liveness::new(f));
        for n in (0..nodes.len()).rev() {
            nodes[n].height = nodes[n].latency
                + members(nodes[n].successors)
                    .map(|i| nodes[i].height)
                    .max()
                    .unwrap_or(0);
        }
        let mut live_in = [0; 3];
        for &reg in &self.registers {
            let uses = liveness.range(f, block, reg);
            let value = &mut self.values[reg.vreg()];
            value.live_out = uses.live_out || uses.last.is_some_and(|i| i.index() >= range.end);
            if value.def == 0 {
                live_in[reg.class() as usize] += 1;
            }
        }
        Region {
            f,
            start: range.start,
            nodes,
            values: &self.values,
            live_in,
        }
        .schedule(self.limit, &mut self.order)
    }
}

pub(super) fn run<I: TargetInst + Schedulable>(f: &mut Function<I>) {
    let env = I::machine_env(f.machine());
    let limit = std::array::from_fn(|c| {
        (env.preferred_regs_by_class[c] | env.non_preferred_regs_by_class[c]).len() as u16
    });
    let mut workspace = None;
    let mut scratch = Vec::new();
    let mut operands: Vec<Operand> = Vec::new();
    for b in (0..f.num_blocks()).map(Block::new) {
        let insts = f.block_insns(b);
        let first = insts.first().index();
        let end = insts.last().index() + 1;
        if end - first < MIN_REGION {
            continue;
        }
        let mut start = first;
        let mut total = 0;
        for i in first..=end {
            let kind = if i == end {
                Scheduling::Barrier
            } else {
                f.insts()[i].scheduling()
            };
            if matches!(kind, Scheduling::Barrier) || i - start == WINDOW {
                if i - start >= MIN_REGION && total >= 4 {
                    let workspace = workspace.get_or_insert_with(|| Workspace {
                        values: vec![Value::default(); f.num_vregs()],
                        registers: Vec::new(),
                        nodes: [Node::default(); WINDOW],
                        order: Vec::new(),
                        liveness: None,
                        limit,
                    });
                    if workspace.reorder(f, start..i, b) {
                        f.reorder(start, &workspace.order, &mut scratch, &mut operands);
                    }
                }
                start = i + usize::from(matches!(kind, Scheduling::Barrier));
                total = 0;
            }
            total += usize::from(matches!(kind, Scheduling::Compute(_)));
        }
    }
}
