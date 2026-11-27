use crate::hir::transforms::ControlFlowGraph;
use crate::hir::{Binary, Block, Edge, Function, Op, Simd, Terminator, Ternary, Type, Value};
use crate::support::slotmap::DenseMap;
use std::collections::HashMap;

#[derive(Clone, Copy)]
enum Path {
    New,
    Visiting,
    Cycle,
    End { block: Block, distance: usize },
}

struct ForwardingPaths {
    next: DenseMap<Block, Option<Edge>>,
    paths: DenseMap<Block, Path>,
}

impl ForwardingPaths {
    fn new(f: &Function, cfg: &ControlFlowGraph) -> Self {
        let mut next = DenseMap::filled(f.block_capacity(), None);
        let mut paths = DenseMap::filled(f.block_capacity(), Path::New);
        for &block in cfg.reverse_postorder() {
            if f.instructions(block).next().is_none()
                && let Terminator::Jump(edge) = *f.terminator(block)
            {
                next[block] = Some(edge);
            }
        }
        let mut stack = Vec::new();
        for &start in cfg.reverse_postorder() {
            let mut block = start;
            let mut path = loop {
                match paths[block] {
                    Path::New => {
                        let Some(edge) = next[block] else {
                            paths[block] = Path::End { block, distance: 0 };
                            break paths[block];
                        };
                        paths[block] = Path::Visiting;
                        stack.push(block);
                        block = f.edge(edge).0;
                    }
                    Path::Visiting => break Path::Cycle,
                    known => break known,
                }
            };
            while let Some(block) = stack.pop() {
                if let Path::End { distance, .. } = &mut path {
                    *distance += 1;
                }
                paths[block] = path;
            }
        }
        Self { next, paths }
    }

    fn merge(&self, f: &Function, mut a: Block, mut b: Block) -> Option<Block> {
        if a == b {
            return Some(a);
        }
        let (
            Path::End {
                block: end_a,
                distance: mut da,
            },
            Path::End {
                block: end_b,
                distance: mut db,
            },
        ) = (self.paths[a], self.paths[b])
        else {
            return None;
        };
        if end_a != end_b {
            return None;
        }
        while a != b {
            if da >= db {
                a = f.edge(self.next[a].unwrap()).0;
                da -= 1;
            } else {
                b = f.edge(self.next[b].unwrap()).0;
                db -= 1;
            }
        }
        Some(a)
    }

    fn arguments(&self, f: &Function, mut edge: Edge, merge: Block) -> Vec<Value> {
        let mut bindings = HashMap::new();
        let mut args = Vec::new();
        loop {
            let (block, values) = f.edge(edge);
            args.clear();
            args.extend(
                values
                    .iter()
                    .map(|v| bindings.get(v).copied().unwrap_or(*v)),
            );
            if block == merge {
                return args;
            }
            // Downstream jumps can use any parameter defined along the path.
            bindings.extend(f.params(block).iter().copied().zip(args.iter().copied()));
            edge = self.next[block].unwrap();
        }
    }
}

/// Replace a branch joining through empty jump blocks with a numeric select.
/// Other predecessors of a forwarding block can keep using that block.
pub(super) fn convert_branches(f: &mut Function, cfg: &ControlFlowGraph) -> bool {
    if !cfg
        .reverse_postorder()
        .iter()
        .any(|&b| matches!(f.terminator(b), Terminator::Branch { .. }))
    {
        return false;
    }
    let paths = ForwardingPaths::new(f, cfg);
    let mut changed = false;
    for &block in cfg.reverse_postorder() {
        let Terminator::Branch { cond, yes, no } = *f.terminator(block) else {
            continue;
        };
        let yes_target = f.edge(yes).0;
        let Some(target) = paths.merge(f, yes_target, f.edge(no).0) else {
            continue;
        };
        if target == block {
            continue;
        }
        let mut args = paths.arguments(f, yes, target);
        let right_args = paths.arguments(f, no, target);
        let (first, second) = {
            let mut differing = args
                .iter()
                .copied()
                .zip(right_args)
                .enumerate()
                .filter(|(_, (x, y))| x != y);
            (differing.next(), differing.next())
        };
        let Some((index, (x, y))) = first else {
            if yes_target == target {
                f.set_terminator(block, Terminator::Jump(yes));
            } else {
                f.jump(block, target, &args);
            }
            changed = true;
            continue;
        };
        let ty = f.value_type(x);
        if ty == Type::Ptr || second.is_some() {
            continue;
        }
        let cond = if matches!(ty, Type::Vector(_)) {
            // Branches test nonzero; SIMD selects test the sign bit per lane.
            let zero = f.append(block, Op::Const(0)).unwrap();
            let truth = f.append(block, Op::Bin(Binary::Ne, cond, zero)).unwrap();
            let mask_type = ty.mask_type().lane().unwrap().scalar();
            let mask = if mask_type == Type::I32 {
                truth
            } else {
                f.append(block, Op::Cast(mask_type, truth)).unwrap()
            };
            f.append(block, Op::Simd(Simd::Splat(mask))).unwrap()
        } else {
            cond
        };
        args[index] = f
            .append(block, Op::Ternary(Ternary::Select, cond, x, y))
            .unwrap();
        f.jump(block, target, &args);
        changed = true;
    }
    changed
}
