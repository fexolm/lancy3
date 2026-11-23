//! Randomized checks of the scheduler's legality and of `Liveness` on a
//! stand-in instruction set, independent of any target.
use super::{Liveness, Schedulable, Scheduling, run};
use crate::lir::{Function, TargetInst};
use regalloc2::{
    Block, Function as _, MachineEnv, Operand, OperandKind, PReg, PRegSet, RegClass, VReg,
};
use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Fake {
    id: usize,
    def: Option<VReg>,
    uses: [Option<VReg>; 2],
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Compute,
    Ordered,
    Barrier,
    Branch,
    Ret,
}

impl TargetInst for Fake {
    type Machine = ();
    fn operands(&self, _: &(), out: &mut Vec<Operand>) {
        out.extend(self.uses.iter().flatten().map(|&v| Operand::reg_use(v)));
        out.extend(self.def.map(Operand::reg_def));
    }
    fn clobbers(&self, _: &()) -> PRegSet {
        PRegSet::empty()
    }
    fn is_branch(&self) -> bool {
        self.kind == Kind::Branch
    }
    fn is_ret(&self) -> bool {
        self.kind == Kind::Ret
    }
    fn machine_env(_: &()) -> &'static MachineEnv {
        static ENV: OnceLock<MachineEnv> = OnceLock::new();
        ENV.get_or_init(|| {
            let mut regs = PRegSet::empty();
            for i in 0..6 {
                regs.add(PReg::new(i, RegClass::Int));
            }
            MachineEnv {
                preferred_regs_by_class: [regs, PRegSet::empty(), PRegSet::empty()],
                non_preferred_regs_by_class: [PRegSet::empty(); 3],
                scratch_by_class: [None; 3],
                fixed_stack_slots: vec![],
            }
        })
    }
    fn spillslot_size(_: &(), _: RegClass) -> usize {
        1
    }
}

impl Schedulable for Fake {
    fn scheduling(&self) -> Scheduling {
        match self.kind {
            Kind::Compute => Scheduling::Compute(1 + (self.id % 4) as u8),
            Kind::Ordered => Scheduling::Ordered(3),
            Kind::Barrier | Kind::Branch | Kind::Ret => Scheduling::Barrier,
        }
    }
}

struct Random(u64);

impl Random {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

#[test]
fn scheduling_keeps_data_ordered_and_barrier_dependencies() {
    let mut random = Random(0x853c_49e6_748f_ea9b);
    let mut reordered = 0;
    for _ in 0..300 {
        let len = 8 + random.below(90);
        let mut f = Function::<Fake>::new((), 1, len).unwrap();
        let mut defined = Vec::new();
        for id in 0..len {
            let kind = match random.below(10) {
                0..=5 => Kind::Compute,
                6..=8 => Kind::Ordered,
                _ => Kind::Barrier,
            };
            let mut operand = || {
                (!defined.is_empty() && random.below(3) != 0)
                    .then(|| defined[random.below(defined.len())])
            };
            let uses = [operand(), operand()];
            let def = (kind != Kind::Barrier || random.below(2) == 0)
                .then(|| VReg::new(id, RegClass::Int));
            defined.extend(def);
            f.push(Fake {
                id,
                def,
                uses,
                kind,
            });
        }
        f.push(Fake {
            id: len,
            def: None,
            uses: [defined.last().copied(), None],
            kind: Kind::Ret,
        });
        f.end_block([], 0).unwrap();
        let before = f.insts().to_vec();
        run(&mut f);
        let after = f.insts();
        reordered += usize::from(after != before.as_slice());

        let mut ids: Vec<_> = after.iter().map(|i| i.id).collect();
        ids.sort_unstable();
        assert!(ids.iter().copied().eq(0..=len), "a permutation");
        let position: Vec<_> = {
            let mut position = vec![0; len + 1];
            for (at, inst) in after.iter().enumerate() {
                position[inst.id] = at;
            }
            position
        };
        for (at, inst) in after.iter().enumerate() {
            for v in inst.uses.iter().flatten() {
                assert!(
                    position[v.vreg()] < at,
                    "{v:?} is used before its definition"
                );
            }
        }
        let fixed: Vec<_> = before
            .iter()
            .filter(|i| matches!(i.kind, Kind::Barrier | Kind::Ret))
            .map(|i| i.id)
            .collect();
        for id in &fixed {
            assert_eq!(position[*id], *id, "barriers stay in place");
        }
        let ordered: Vec<_> = before
            .iter()
            .filter(|i| matches!(i.kind, Kind::Ordered | Kind::Barrier | Kind::Ret))
            .map(|i| position[i.id])
            .collect();
        assert!(
            ordered.windows(2).all(|w| w[0] < w[1]),
            "memory accesses and checks keep their order and stay between barriers"
        );
    }
    assert!(
        reordered > 100,
        "the scheduler moved only {reordered} of 300 bodies"
    );
}

/// Random reducible and irreducible CFGs with SSA values: each use is
/// dominated by its definition. `Liveness` must agree with textbook dataflow.
#[test]
fn liveness_matches_iterative_dataflow() {
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    for _ in 0..300 {
        let blocks = 2 + random.below(10);
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); blocks];
        for b in 1..blocks {
            succs[random.below(b)].push(b);
        }
        for targets in &mut succs {
            while targets.len() < 2 && random.below(3) == 0 {
                let target = 1 + random.below(blocks - 1);
                if !targets.contains(&target) {
                    targets.push(target);
                }
            }
        }
        let mut preds = vec![Vec::new(); blocks];
        for (b, targets) in succs.iter().enumerate() {
            for &t in targets {
                preds[t].push(b);
            }
        }
        // Iterative dominator sets; block 0 reaches every block.
        let all: HashSet<usize> = (0..blocks).collect();
        let mut dom = vec![all; blocks];
        dom[0] = HashSet::from([0]);
        let mut changed = true;
        while changed {
            changed = false;
            for b in 1..blocks {
                let mut set = preds[b]
                    .iter()
                    .map(|&p| dom[p].clone())
                    .reduce(|a, b| &a & &b)
                    .unwrap_or_default();
                set.insert(b);
                if set != dom[b] {
                    dom[b] = set;
                    changed = true;
                }
            }
        }

        let mut owner = Vec::new();
        let mut code: Vec<Vec<Fake>> = vec![Vec::new(); blocks];
        let mut id = 0;
        for b in 0..blocks {
            for _ in 0..1 + random.below(5) {
                let visible: Vec<_> = (0..owner.len())
                    .filter(|&v| {
                        let d = owner[v];
                        d == b
                            && code[b]
                                .iter()
                                .any(|i: &Fake| i.def == Some(VReg::new(v, RegClass::Int)))
                            || d != b && dom[b].contains(&d)
                    })
                    .collect();
                let mut operand = || {
                    (!visible.is_empty() && random.below(2) == 0)
                        .then(|| VReg::new(visible[random.below(visible.len())], RegClass::Int))
                };
                let uses = [operand(), operand()];
                let def = VReg::new(owner.len(), RegClass::Int);
                owner.push(b);
                code[b].push(Fake {
                    id,
                    def: Some(def),
                    uses,
                    kind: Kind::Compute,
                });
                id += 1;
            }
        }
        let mut f = Function::<Fake>::new((), blocks, owner.len()).unwrap();
        for b in 0..blocks {
            for &inst in &code[b] {
                f.push(inst);
            }
            let kind = if succs[b].is_empty() {
                Kind::Ret
            } else {
                Kind::Branch
            };
            f.push(Fake {
                id,
                def: None,
                uses: [None; 2],
                kind,
            });
            id += 1;
            for &t in &succs[b] {
                f.edge(Block::new(t), [], t <= b);
            }
            f.end_block([], 0).unwrap();
        }

        // Reference: live_in(b) = upward-exposed uses ∪ (live_out(b) − defs).
        let mut live_in = vec![HashSet::new(); blocks];
        let mut live_out = vec![HashSet::new(); blocks];
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..blocks).rev() {
                let out: HashSet<usize> = succs[b]
                    .iter()
                    .flat_map(|&s| live_in[s].iter().copied())
                    .collect();
                let mut inside = out.clone();
                for inst in code[b].iter().rev() {
                    if let Some(d) = inst.def {
                        inside.remove(&d.vreg());
                    }
                    inside.extend(inst.uses.iter().flatten().map(|v| v.vreg()));
                }
                if out != live_out[b] || inside != live_in[b] {
                    live_out[b] = out;
                    live_in[b] = inside;
                    changed = true;
                }
            }
        }

        let mut liveness = Liveness::new(&f);
        for (b, live_out) in live_out.iter().enumerate() {
            let block = Block::new(b);
            for i in f.block_insns(block).iter() {
                for op in f.inst_operands(i) {
                    let v = op.vreg();
                    let range = liveness.range(&f, block, v);
                    assert_eq!(
                        range.live_out,
                        live_out.contains(&v.vreg()),
                        "{v:?} out of block {b}, edges {succs:?}"
                    );
                    let last = f.block_insns(block).iter().rev().find(|&i| {
                        f.inst_operands(i)
                            .iter()
                            .any(|o| o.vreg() == v && o.kind() == OperandKind::Use)
                    });
                    assert_eq!(range.last, last, "last use of {v:?} in block {b}");
                }
            }
        }
    }
}
