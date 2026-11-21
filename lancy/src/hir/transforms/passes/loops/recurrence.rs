//! Close small additive recurrences over the i32 wrapping ring.
use super::super::facts::flatten;
use super::{CountedLoop, Facts, NaturalLoops};
use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Binary, Block, Function, Op, Terminator, Type, Value, ssa::Edge};
use crate::support::slotmap::DenseMap;

const VALUE_LIMIT: usize = 32;
const EXPRESSION_LIMIT: usize = 64;

type Coefficient = Option<usize>; // None is the zero coefficient.

/// Coefficients in the binomial basis: a + b*n + c*choose(n, 2).
#[derive(Clone, Copy)]
struct Series([Coefficient; 3]);

impl Series {
    fn invariant(value: Coefficient) -> Self {
        Self([value, None, None])
    }

    fn varies(self) -> bool {
        self.0[1].is_some() || self.0[2].is_some()
    }
}

#[derive(Clone, Copy)]
enum Expr {
    Available(Value),
    Constant(i32),
    Binary(Binary, usize, usize),
}

struct Analysis<'a> {
    f: &'a Function,
    dom: &'a DomTree,
    facts: &'a Facts,
    info: CountedLoop,
    initial: &'a [Value],
    next: &'a [Value],
    expressions: Vec<Expr>,
    seen: Vec<(Value, Option<Series>)>,
}

impl Analysis<'_> {
    fn term(&mut self, expr: Expr) -> Option<Coefficient> {
        if self.expressions.len() == EXPRESSION_LIMIT {
            return None;
        }
        let id = self.expressions.len();
        self.expressions.push(expr);
        Some(Some(id))
    }

    fn constant(&mut self, n: i32) -> Option<Coefficient> {
        if n == 0 {
            Some(None)
        } else {
            self.term(Expr::Constant(n))
        }
    }

    fn binary(&mut self, op: Binary, a: Coefficient, b: Coefficient) -> Option<Coefficient> {
        match (op, a, b) {
            (Binary::Mul, None, _) | (Binary::Mul, _, None) | (_, None, None) => Some(None),
            (Binary::Add, None, _) => Some(b),
            (Binary::Add | Binary::Sub, _, None) => Some(a),
            (Binary::Sub, None, Some(b)) => {
                let zero = self.term(Expr::Constant(0))??;
                self.term(Expr::Binary(op, zero, b))
            }
            (_, Some(a), Some(b)) => self.term(Expr::Binary(op, a, b)),
            _ => unreachable!("only add, sub, and mul form coefficients"),
        }
    }

    fn combine(&mut self, op: Binary, a: Series, b: Series) -> Option<Series> {
        Some(Series([
            self.binary(op, a.0[0], b.0[0])?,
            self.binary(op, a.0[1], b.0[1])?,
            self.binary(op, a.0[2], b.0[2])?,
        ]))
    }

    fn scale(&mut self, value: Series, factor: Coefficient) -> Option<Series> {
        Some(Series([
            self.binary(Binary::Mul, value.0[0], factor)?,
            self.binary(Binary::Mul, value.0[1], factor)?,
            self.binary(Binary::Mul, value.0[2], factor)?,
        ]))
    }

    fn series(&mut self, value: Value) -> Option<Series> {
        let value = self.facts.root(value);
        if self.f.value_type(value) != Type::I32 {
            return None;
        }
        if let Some((_, known)) = self.seen.iter().find(|&&(v, _)| v == value) {
            return *known;
        }
        if self.seen.len() == VALUE_LIMIT {
            return None;
        }
        // An unresolved recursive use rejects coupled recurrences.
        let id = self.seen.len();
        self.seen.push((value, None));
        let result = self.infer(value);
        self.seen[id].1 = result;
        result
    }

    fn infer(&mut self, value: Value) -> Option<Series> {
        if let Some(n) = self.facts.constant(self.f, value) {
            return Some(Series::invariant(self.constant(n)?));
        }
        if let Some(value) = self.facts.available(value, self.info.preheader, self.dom) {
            return Some(Series::invariant(self.term(Expr::Available(value))?));
        }
        if let Ok(index) = self.f.params(self.info.header).binary_search(&value) {
            let initial = self.series(self.initial[index])?;
            if initial.varies() {
                return None;
            }
            let next = self.facts.root(self.next[index]);
            if next == value {
                return Some(initial);
            }
            let Op::Bin(op, a, b) = self.facts.op(self.f, next)? else {
                return None;
            };
            let a = self.facts.root(a);
            let b = self.facts.root(b);
            let (increment, negate) = match op {
                Binary::Add if a == value => (b, false),
                Binary::Add if b == value => (a, false),
                Binary::Sub if a == value => (b, true),
                _ => return None,
            };
            let mut step = self.series(increment)?;
            if negate {
                step = self.combine(Binary::Sub, Series::invariant(None), step)?;
            }
            if step.0[2].is_some() {
                return None; // Integrating this step would require degree three.
            }
            return Some(Series([initial.0[0], step.0[0], step.0[1]]));
        }

        let Op::Bin(op @ (Binary::Add | Binary::Sub | Binary::Mul), a, b) =
            self.facts.op(self.f, value)?
        else {
            return None;
        };
        let a = self.series(a)?;
        let b = self.series(b)?;
        match op {
            Binary::Add | Binary::Sub => self.combine(op, a, b),
            Binary::Mul if !a.varies() => self.scale(b, a.0[0]),
            Binary::Mul if !b.varies() => self.scale(a, b.0[0]),
            _ => None,
        }
    }
}

struct Candidate {
    info: CountedLoop,
    backedge: Edge,
    live_out: Vec<Value>,
    valid: bool,
}

#[derive(Clone, Copy)]
enum FinalValue {
    Available(Value),
    Induction,
    Series(Series),
}

struct Plan {
    info: CountedLoop,
    exits: Vec<(Value, FinalValue)>,
    expressions: Vec<Expr>,
    start: Option<i32>,
}

fn plan(f: &Function, dom: &DomTree, facts: &Facts, candidate: &Candidate) -> Option<Plan> {
    let info = candidate.info;
    let Terminator::Jump(enter) = *f.terminator(info.preheader) else {
        return None;
    };
    let mut analysis = Analysis {
        f,
        dom,
        facts,
        info,
        initial: f.edge(enter).1,
        next: f.edge(candidate.backedge).1,
        expressions: Vec::new(),
        seen: Vec::new(),
    };
    let mut exits = Vec::with_capacity(candidate.live_out.len());
    let mut found_recurrence = false;
    for &value in &candidate.live_out {
        let root = facts.root(value);
        let replacement = if let Some(v) = facts.available(root, info.preheader, dom) {
            FinalValue::Available(v)
        } else if root == info.iv {
            found_recurrence = true;
            FinalValue::Induction
        } else {
            // A single exit from the header can expose only definitions
            // available at that header (including provably forwarded values).
            if facts.defs[root].block != info.header {
                return None;
            }
            let series = analysis.series(root)?;
            found_recurrence |= series.varies();
            FinalValue::Series(series)
        };
        exits.push((value, replacement));
    }
    // The separate dead-loop pass handles candidates with no evolving live-out.
    found_recurrence.then_some(Plan {
        info,
        exits,
        expressions: analysis.expressions,
        start: facts.constant(f, info.start),
    })
}

fn emit(f: &mut Function, plan: Plan, replacements: &mut Vec<(Value, Value)>) {
    let preheader = plan.info.preheader;
    let bin = |f: &mut Function, op, a, b| f.append(preheader, Op::Bin(op, a, b)).unwrap();
    let constant = |f: &mut Function, n| f.append(preheader, Op::Const(n)).unwrap();

    let mut values = Vec::with_capacity(plan.expressions.len());
    for &expr in &plan.expressions {
        let value = match expr {
            Expr::Available(value) => value,
            Expr::Constant(n) => constant(f, n),
            Expr::Binary(op, a, b) => bin(f, op, values[a], values[b]),
        };
        values.push(value);
    }

    let [start, bound] = [plan.info.start, plan.info.bound];
    let max = bin(f, Binary::Max, start, bound);
    let trip = if plan.start == Some(0) {
        max
    } else {
        bin(f, Binary::Sub, max, start)
    };

    // A signed < loop may have up to u32::MAX iterations. Treat trip
    // as unsigned for halving. Divide an even factor before multiplying
    // so the modulo 2^32 result retains the bit that the divide needs.
    let mut choose_two = None;
    for (original, replacement) in plan.exits {
        let replacement = match replacement {
            FinalValue::Available(value) => value,
            FinalValue::Induction => max,
            FinalValue::Series(Series([base, linear, quadratic])) => {
                let mut sum = base.map(|v| values[v]);
                for (coefficient, degree) in [(linear, 1), (quadratic, 2)] {
                    let Some(coefficient) = coefficient else {
                        continue;
                    };
                    let factor = if degree == 1 {
                        trip
                    } else {
                        *choose_two.get_or_insert_with(|| {
                            let one = constant(f, 1);
                            let half = bin(f, Binary::Shr, trip, one);
                            let half = if plan.start.is_some_and(|n| n >= 0) {
                                half // The iteration count fits in signed i32.
                            } else {
                                let mask = constant(f, i32::MAX);
                                bin(f, Binary::And, half, mask)
                            };
                            let predecessor = bin(f, Binary::Sub, trip, one);
                            let odd = bin(f, Binary::Or, predecessor, one);
                            bin(f, Binary::Mul, half, odd)
                        })
                    };
                    let term = if matches!(plan.expressions[coefficient], Expr::Constant(1)) {
                        factor
                    } else {
                        bin(f, Binary::Mul, values[coefficient], factor)
                    };
                    sum = Some(sum.map_or(term, |sum| bin(f, Binary::Add, sum, term)));
                }
                sum.unwrap_or_else(|| constant(f, 0))
            }
        };
        replacements.push((original, replacement));
    }

    f.set_terminator(preheader, Terminator::Jump(plan.info.exit));
}

pub(super) fn collapse(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    loops: &NaturalLoops,
    facts: &Facts,
    counted: &DenseMap<Block, Option<CountedLoop>>,
) -> bool {
    if counted.values().all(Option::is_none) {
        return false;
    }
    let mut has_child = DenseMap::filled(f.block_capacity(), false);
    for &header in &loops.headers {
        let parent = loops.parent[header];
        if parent != Block::INVALID {
            has_child[parent] = true;
        }
    }
    let mut candidates = Vec::new();
    let mut indices = DenseMap::filled(f.block_capacity(), None);
    for &header in &loops.headers {
        let Some(info) = counted[header] else {
            continue;
        };
        if has_child[header] || !info.strict_unit_step() {
            continue;
        }
        let mut latches = cfg
            .predecessors(header)
            .iter()
            .copied()
            .filter(|&b| b != info.preheader);
        let Some(latch) = latches.next() else {
            continue;
        };
        if latches.next().is_some() {
            continue;
        }
        let mut edges = f
            .terminator(latch)
            .edges()
            .filter(|&e| f.edge(e).0 == header);
        let Some(backedge) = edges.next() else {
            continue;
        };
        if edges.next().is_some() {
            continue;
        }
        // A loop with only its induction parameter and no loop-dependent exit
        // operands can still expose that parameter directly, so keep its case.
        indices[header] = Some(candidates.len());
        candidates.push(Candidate {
            info,
            backedge,
            live_out: Vec::new(),
            valid: true,
        });
    }
    if candidates.is_empty() {
        return false;
    }

    let mut rank = DenseMap::filled(f.block_capacity(), 0);
    for (i, &block) in cfg.reverse_postorder().iter().enumerate() {
        rank[block] = i;
    }
    for &block in cfg.reverse_postorder() {
        let header = loops.owner[block];
        let Some(index) = indices.get(header).copied().flatten() else {
            continue;
        };
        let candidate = &mut candidates[index];
        if !candidate.valid {
            continue;
        }
        let term = *f.terminator(block);
        // The false edge from the header is the only exit. Removing the
        // latch edge leaves a DAG, so no other cycle can keep it from exiting.
        candidate.valid = !matches!(term, Terminator::Return(_))
            && term.edges().all(|e| {
                let target = f.edge(e).0;
                if block == header && e == candidate.info.exit {
                    return true;
                }
                loops.owner[target] == header
                    && (e == candidate.backedge && target == header || rank[target] > rank[block])
            })
            && f.instructions(block).all(|i| facts.is_total(f, i));
    }
    if candidates.iter().all(|candidate| !candidate.valid) {
        return false;
    }

    let mut seen = DenseMap::filled(f.value_count(), false);
    for &block in cfg.reverse_postorder() {
        let loop_exit = indices
            .get(loops.owner[block])
            .copied()
            .flatten()
            .map(|index| candidates[index].info.exit);
        let mut note = |value: Value, target: Block| {
            let header = loops.owner[facts.defs[value].block];
            if header != loops.owner[target]
                && let Some(index) = indices.get(header).copied().flatten()
                && candidates[index].valid
                && !std::mem::replace(&mut seen[value], true)
            {
                let candidate = &mut candidates[index];
                if candidate.live_out.len() == VALUE_LIMIT {
                    candidate.valid = false;
                } else {
                    candidate.live_out.push(value);
                }
            }
        };
        for i in f.instructions(block) {
            f.visit_operands(i, |v| note(v, block));
        }
        let term = *f.terminator(block);
        if let Terminator::Return(v) | Terminator::Branch { cond: v, .. } = term {
            note(v, block);
        }
        for edge in term.edges() {
            let (target, values) = f.edge(edge);
            for &value in values {
                // Edge operands are used at the source except that a loop's
                // exit edge must also survive when retargeted to its preheader.
                note(value, block);
                if Some(edge) == loop_exit {
                    note(value, target);
                }
            }
        }
    }

    let plans: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.valid)
        .filter_map(|candidate| plan(f, dom, facts, candidate))
        .collect();
    if plans.is_empty() {
        return false;
    }
    let mut replacements = Vec::new();
    for plan in plans {
        emit(f, plan, &mut replacements);
    }
    let mut aliases = DenseMap::from_fn(f.value_count(), |value| value);
    for (original, replacement) in replacements {
        aliases[original] = replacement;
    }
    flatten(&mut aliases);
    f.substitute_values(&aliases);
    true
}
