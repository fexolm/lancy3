//! If-convert forward scalar control flow into a lane-wise HIR projection.
use super::{Binary, BuildValue, Builder, Function, Instr, Program, Reg};

impl Function {
    /// Lower independent invocations to a lane-wise program for the SIMD or
    /// adaptive batch backend. Unsupported control flow yields `Ok(None)`.
    pub fn vectorize(&self) -> Result<Option<Program>, String> {
        Ok(vector_program(&self.simplified()?))
    }
}

#[derive(Clone, Copy)]
struct Lane {
    value: BuildValue,
    is_mask: bool,
}

#[derive(Clone, Copy)]
struct Active {
    full: BuildValue,
    edge: Option<(usize, bool)>,
}

struct Fork {
    parent: Active,
    condition: BuildValue,
}

struct Blend {
    condition: BuildValue,
    new_on_true: bool,
}

impl Blend {
    fn pick(&self, old: Lane, new: Lane, b: &mut Builder) -> Lane {
        let value = if old.value == new.value {
            old.value
        } else if self.new_on_true {
            b.select(self.condition, new.value, old.value)
        } else {
            b.select(self.condition, old.value, new.value)
        };
        Lane {
            value,
            is_mask: old.is_mask && new.is_mask,
        }
    }
}

// When both arms of the same fork meet, their union reconstructs exactly
// the parent. The local condition suffices to choose an SSA value on that
// parent. For an unstructured join, keep the full masks of incoming paths.
fn join(old: Active, new: Active, forks: &[Fork], b: &mut Builder) -> (Active, Blend) {
    if let (Some((a, old_side)), Some((c, new_side))) = (old.edge, new.edge)
        && a == c
        && old_side != new_side
    {
        return (
            forks[a].parent,
            Blend {
                condition: forks[a].condition,
                new_on_true: new_side,
            },
        );
    }
    (
        Active {
            full: b.or(old.full, new.full),
            edge: None,
        },
        Blend {
            condition: new.full,
            new_on_true: true,
        },
    )
}

#[derive(Clone)]
struct PathState {
    active: Active,
    regs: Vec<Option<Lane>>,
}

impl PathState {
    fn get(&self, r: Reg) -> Lane {
        self.regs[r.0 as usize].expect("verified scalar register")
    }
}

fn merge(slot: &mut Option<PathState>, next: PathState, forks: &[Fork], b: &mut Builder) {
    if let Some(prev) = slot {
        let (active, blend) = join(prev.active, next.active, forks, b);
        for (a, v) in prev.regs.iter_mut().zip(next.regs) {
            *a = match (*a, v) {
                (Some(a), Some(v)) => Some(blend.pick(a, v, b)),
                _ => None,
            };
        }
        prev.active = active;
    } else {
        *slot = Some(next);
    }
}

pub(crate) fn vector_program(f: &Function) -> Option<Program> {
    if !f.buffer_kinds().is_empty() || f.arity() > usize::from(u16::MAX) + 1 {
        return None;
    }
    let mut b = Builder::new(f.arity());
    let args: Vec<_> = (0..f.arity())
        .map(|i| Lane {
            value: b.input(i as u16),
            is_mask: false,
        })
        .collect();
    // Bound native expansion for very large or heavily shared call graphs.
    let mut budget = 8192usize;
    let result = predicated(f, &args, &mut b, &mut budget, 0)?;
    b.finish(result.value).ok()
}

fn predicated(
    f: &Function,
    args: &[Lane],
    b: &mut Builder,
    budget: &mut usize,
    depth: usize,
) -> Option<Lane> {
    *budget = budget.checked_sub(f.code.len())?;
    if depth >= 64 {
        return None;
    }
    let all = b.constant(-1);
    let zero = b.constant(0);
    let mut initial = PathState {
        active: Active {
            full: all,
            edge: None,
        },
        regs: vec![None; f.registers],
    };
    for (reg, arg) in initial.regs.iter_mut().zip(args) {
        *reg = Some(*arg);
    }
    let mut incoming = std::collections::BTreeMap::<usize, Option<PathState>>::new();
    let mut current = Some(initial);
    let mut result: Option<(Active, Lane)> = None;
    let mut forks = Vec::new();
    for (pc, op) in f.code.iter().enumerate() {
        if let Some(Some(arrived)) = incoming.remove(&pc) {
            merge(&mut current, arrived, &forks, b);
        }
        let Some(mut state) = current.take() else {
            continue;
        };
        match op {
            Instr::Const(d, x) => {
                state.regs[d.0 as usize] = Some(Lane {
                    value: b.constant(*x),
                    is_mask: matches!(x, 0 | -1),
                })
            }
            Instr::Copy(d, s) => state.regs[d.0 as usize] = Some(state.get(*s)),
            Instr::Bin(d, op, a, c) => {
                let (lhs, rhs) = (state.get(*a), state.get(*c));
                let (x, y) = (lhs.value, rhs.value);
                let v = match op {
                    Binary::Add => b.add(x, y),
                    Binary::Sub => b.sub(x, y),
                    Binary::Mul => b.mul(x, y),
                    Binary::And => b.and(x, y),
                    Binary::Or => b.or(x, y),
                    Binary::Xor => b.xor(x, y),
                    Binary::Lt => b.lt(x, y),
                    Binary::Gt => b.lt(y, x),
                    Binary::Eq => b.eq(x, y),
                    Binary::Min | Binary::Max => {
                        let smaller = b.lt(x, y);
                        if *op == Binary::Min {
                            b.select(smaller, x, y)
                        } else {
                            b.select(smaller, y, x)
                        }
                    }
                    Binary::Le => {
                        let m = b.lt(y, x);
                        b.xor(m, all)
                    }
                    Binary::Ge => {
                        let m = b.lt(x, y);
                        b.xor(m, all)
                    }
                    Binary::Ne => {
                        let m = b.eq(x, y);
                        b.xor(m, all)
                    }
                    Binary::Shl | Binary::Shr | Binary::Div | Binary::Rem => return None,
                };
                let is_mask = match op {
                    Binary::Lt | Binary::Gt | Binary::Eq | Binary::Le | Binary::Ge | Binary::Ne => {
                        true
                    }
                    Binary::And | Binary::Or | Binary::Xor => lhs.is_mask && rhs.is_mask,
                    _ => false,
                };
                state.regs[d.0 as usize] = Some(Lane { value: v, is_mask });
            }
            Instr::Label(_) => {}
            Instr::BufferLen { .. } | Instr::Load { .. } | Instr::Store { .. } => return None,
            Instr::Jump(l) => {
                let target = f.targets[l.0 as usize];
                if target <= pc {
                    return None;
                }
                merge(incoming.entry(target).or_default(), state, &forks, b);
                continue;
            }
            Instr::Branch { cond, yes, no } => {
                let (yes, no) = (f.targets[yes.0 as usize], f.targets[no.0 as usize]);
                if yes <= pc || no <= pc {
                    return None;
                }
                // Normalize a raw nonzero i32 to a full boolean lane mask.
                let cond = state.get(*cond);
                let (true_mask, false_mask) = if cond.is_mask {
                    (cond.value, b.xor(cond.value, all))
                } else {
                    let no = b.eq(cond.value, zero);
                    (b.xor(no, all), no)
                };
                let fork = forks.len();
                forks.push(Fork {
                    parent: state.active,
                    condition: true_mask,
                });
                let mut other = state.clone();
                other.active = Active {
                    full: b.and(other.active.full, false_mask),
                    edge: Some((fork, false)),
                };
                state.active = Active {
                    full: b.and(state.active.full, true_mask),
                    edge: Some((fork, true)),
                };
                merge(incoming.entry(yes).or_default(), state, &forks, b);
                merge(incoming.entry(no).or_default(), other, &forks, b);
                continue;
            }
            Instr::Call {
                dst,
                function,
                args,
                ..
            } => {
                let actual: Vec<_> = args.iter().map(|r| state.get(*r)).collect();
                state.regs[dst.0 as usize] =
                    Some(predicated(function, &actual, b, budget, depth + 1)?);
            }
            Instr::Return(r) => {
                let value = state.get(*r);
                result = Some(match result {
                    Some((old, previous)) => {
                        let (active, blend) = join(old, state.active, &forks, b);
                        (active, blend.pick(previous, value, b))
                    }
                    None => (state.active, value),
                });
                continue;
            }
        }
        current = Some(state);
    }
    result.map(|(_, value)| value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{FunctionBuilder, Op};

    #[test]
    fn nested_diamonds_use_local_conditions_for_their_blends() {
        let mut b = FunctionBuilder::new(2);
        let zero = b.constant(0);
        let result = b.local();
        let [outer_yes, outer_no, inner_yes, inner_no, inner_exit, exit] =
            std::array::from_fn(|_| b.label());
        let negative_x = b.binary(Binary::Lt, b.arg(0), zero);
        b.branch(negative_x, outer_yes, outer_no);
        b.mark(outer_yes);
        let negative_y = b.binary(Binary::Lt, b.arg(1), zero);
        b.branch(negative_y, inner_yes, inner_no);
        b.mark(inner_yes);
        let seven = b.constant(7);
        b.assign(result, seven);
        b.jump(inner_exit);
        b.mark(inner_no);
        b.assign(result, b.arg(0));
        b.mark(inner_exit);
        b.jump(exit);
        b.mark(outer_no);
        b.assign(result, b.arg(1));
        b.mark(exit);
        b.ret(result);
        let f = b.finish().unwrap();
        let p = crate::hir::projection::optimize(&vector_program(&f).unwrap(), &[]);
        assert_eq!(
            p.ops()
                .iter()
                .filter(|op| matches!(op, Op::Select(..)))
                .count(),
            2
        );
        assert!(p.ops().iter().all(|op| matches!(
            op,
            Op::Input(_) | Op::Const(_) | Op::Lt(..) | Op::Select(..)
        )));
        for x in [-7, 0, 2] {
            for y in [-3, 0, 4] {
                assert_eq!(p.eval_row(&[&[x], &[y]], 0), f.interpret(&[x, y]).unwrap());
            }
        }
    }

    #[test]
    fn crossed_edges_keep_full_path_masks_and_raw_nonzero_semantics() {
        // The two exits each receive paths from separate nested conditionals.
        let mut b = FunctionBuilder::new(3);
        let [left, right, yes, no] = std::array::from_fn(|_| b.label());
        b.branch(b.arg(0), left, right);
        b.mark(left);
        b.branch(b.arg(1), yes, no);
        b.mark(right);
        b.branch(b.arg(2), yes, no);
        b.mark(yes);
        b.ret(b.arg(1));
        b.mark(no);
        b.ret(b.arg(2));
        let f = b.finish().unwrap();
        let p = crate::hir::projection::optimize(&vector_program(&f).unwrap(), &[]);
        for a in [-1, 0, 7] {
            for c in [-3, 0, 2] {
                for d in [0, i32::MIN, i32::MAX] {
                    assert_eq!(
                        p.eval_row(&[&[a], &[c], &[d]], 0),
                        f.interpret(&[a, c, d]).unwrap()
                    );
                }
            }
        }
    }
}
