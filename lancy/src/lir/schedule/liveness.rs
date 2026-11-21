//! SSA live-out information for registers appearing in a scheduling region.
use crate::lir::{Function, TargetInst};
use regalloc2::{Block, Function as _, Inst, OperandKind, VReg};

const NONE: usize = usize::MAX;
const LATEST: usize = usize::MAX - 1;

pub(super) struct UseRange {
    pub(super) last: Option<Inst>,
    pub(super) live_out: bool,
}

#[derive(Clone, Copy)]
struct Site {
    block: Block,
    last: Inst,
    live_out: bool,
}

struct Earlier {
    site: Site,
    previous: usize,
}

#[derive(Clone, Copy)]
struct Register {
    definition: Block,
    latest: Site,
    earlier: usize,
    solved: bool,
}

pub(super) struct Liveness {
    registers: Vec<Register>,
    earlier: Vec<Earlier>,
    acyclic: bool,
    visited: Vec<usize>,
    local: Vec<usize>,
    pending: Vec<Block>,
}

impl Liveness {
    pub(super) fn new<I: TargetInst>(f: &Function<I>) -> Self {
        let mut registers = vec![
            Register {
                definition: Block::invalid(),
                latest: Site {
                    block: Block::invalid(),
                    last: Inst::invalid(),
                    live_out: false
                },
                earlier: NONE,
                solved: false,
            };
            f.num_vregs()
        ];
        let mut earlier = Vec::new();
        let mut acyclic = true;
        let mut record = |b: Block, v: VReg, use_at: Option<Inst>| {
            let register = &mut registers[v.vreg()];
            let Some(last) = use_at else {
                register.definition = b;
                return;
            };
            if register.latest.block == b {
                register.latest.last = last;
            } else {
                if register.latest.block != Block::invalid() {
                    earlier.push(Earlier {
                        site: register.latest,
                        previous: register.earlier,
                    });
                    register.earlier = earlier.len() - 1;
                }
                register.latest = Site {
                    block: b,
                    last,
                    live_out: false,
                };
            }
        };
        for b in (0..f.num_blocks()).map(Block::new) {
            for &v in f.block_params(b) {
                record(b, v, None);
            }
            let insts = f.block_insns(b);
            for i in insts.iter() {
                for op in f.inst_operands(i) {
                    record(b, op.vreg(), (op.kind() == OperandKind::Use).then_some(i));
                }
            }
            for (edge, &target) in f.block_succs(b).iter().enumerate() {
                acyclic &= target > b;
                let end = insts.last();
                for &v in f.branch_blockparams(b, end, edge) {
                    record(b, v, Some(end));
                }
            }
        }

        Self {
            registers,
            earlier,
            acyclic,
            visited: Vec::new(),
            local: Vec::new(),
            pending: Vec::new(),
        }
    }

    pub(super) fn range<I: TargetInst>(
        &mut self,
        f: &Function<I>,
        block: Block,
        v: VReg,
    ) -> UseRange {
        let index = v.vreg();
        let register = self.registers[index];
        let latest = register.latest.block == block;
        let earlier = if latest {
            None
        } else {
            sites(&self.earlier, register.earlier)
                .take_while(|&i| self.earlier[i].site.block >= block)
                .find(|&i| self.earlier[i].site.block == block)
        };
        let last = if latest {
            Some(register.latest.last)
        } else {
            earlier.map(|i| self.earlier[i].site.last)
        };

        let live_out = if register.definition == block {
            // Every use outside the defining block is reached through it.
            register.earlier != NONE || (!latest && register.latest.block != Block::invalid())
        } else if f.block_succs(block).is_empty() || self.acyclic && latest {
            // A path in the DAG cannot reach a use at or before this block.
            false
        } else {
            if !register.solved {
                self.solve(f, index);
            }
            if latest {
                self.registers[index].latest.live_out
            } else {
                earlier.is_some_and(|i| self.earlier[i].site.live_out)
            }
        };
        UseRange { last, live_out }
    }

    fn solve<I: TargetInst>(&mut self, f: &Function<I>, reg: usize) {
        let register = self.registers[reg];
        let def = register.definition;
        self.visited.resize(f.num_blocks(), NONE);
        self.local.resize(f.num_blocks(), NONE);
        self.local[register.latest.block.index()] = LATEST;
        if register.latest.block != def {
            self.pending.push(register.latest.block);
        }
        for i in sites(&self.earlier, register.earlier) {
            let block = self.earlier[i].site.block;
            self.local[block.index()] = i;
            if block != def {
                self.pending.push(block);
            }
        }
        // Only uses outside the defining block can extend a live range
        // across an edge. A return to the definition starts a new range.
        while let Some(block) = self.pending.pop() {
            if std::mem::replace(&mut self.visited[block.index()], reg) == reg {
                continue;
            }
            for &pred in f.block_preds(block) {
                if pred == def {
                    continue;
                }
                match self.local[pred.index()] {
                    NONE => {}
                    LATEST => self.registers[reg].latest.live_out = true,
                    i => self.earlier[i].site.live_out = true,
                }
                if self.visited[pred.index()] != reg {
                    self.pending.push(pred);
                }
            }
        }
        self.local[register.latest.block.index()] = NONE;
        for i in sites(&self.earlier, register.earlier) {
            self.local[self.earlier[i].site.block.index()] = NONE;
        }
        self.registers[reg].solved = true;
    }
}

fn sites(earlier: &[Earlier], mut current: usize) -> impl Iterator<Item = usize> + '_ {
    std::iter::from_fn(move || {
        let index = current;
        current = earlier.get(current)?.previous;
        Some(index)
    })
}
