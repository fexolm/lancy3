//! Specialize reachable static callees and share copies with identical known arguments.

mod cleanup;
mod const_eval;
mod dataflow;

use super::{Function, Instr, Reg};
use cleanup::{remove_dead_values, remove_implicit_jumps};
use const_eval::ConstEvaluator;
use std::collections::HashMap;
use std::sync::Arc;

const VARIANTS_PER_CALLEE: usize = 4;
const VARIANTS_PER_COMPILATION: usize = 32;
const VARIANT_WORK_BUDGET: usize = 16 * 1024;

// Indices are formal parameters of the callee, in increasing order.
type KnownArgs = Vec<(Reg, i32)>;

#[derive(PartialEq, Eq, Hash)]
struct CallKey {
    callee: *const Function,
    known: KnownArgs,
}

struct VariantBudget {
    per_callee: HashMap<*const Function, usize>,
    count: usize,
    work: usize,
}

impl VariantBudget {
    fn key(
        &mut self,
        function: &Function,
        known: KnownArgs,
        memo: &HashMap<CallKey, Arc<Function>>,
    ) -> CallKey {
        let mut key = CallKey {
            callee: std::ptr::from_ref(function),
            known,
        };
        if key.known.is_empty() || memo.contains_key(&key) {
            return key;
        }
        let cost = function.code.len().saturating_add(function.registers);
        if self.count < VARIANTS_PER_COMPILATION && cost <= self.work {
            let count = self.per_callee.entry(key.callee).or_default();
            if *count < VARIANTS_PER_CALLEE {
                *count += 1;
                self.count += 1;
                self.work -= cost;
                return key;
            }
        }
        // Empty facts identify the shared generic callee.
        key.known.clear();
        key
    }
}

struct Job {
    parent: Option<(usize, CallKey)>,
    simplified: Function,
    cursor: usize,
    call_facts: std::vec::IntoIter<KnownArgs>,
}

fn simplify_one(
    f: &Function,
    evaluator: &mut ConstEvaluator,
    parent: Option<(usize, CallKey)>,
) -> Result<Job, String> {
    let known = parent
        .as_ref()
        .map_or(&[][..], |(_, key)| key.known.as_slice());
    let (mut code, call_facts) = dataflow::fold_constants(f, evaluator, known);
    remove_dead_values(&mut code, f.registers);
    remove_implicit_jumps(&mut code);
    // Both cleanup passes preserve the order and number of surviving calls.
    Ok(Job {
        parent,
        simplified: f.rebuild(code)?,
        cursor: 0,
        call_facts: call_facts.into_iter(),
    })
}

impl Function {
    /// Fold values known on every reachable incoming path, eliminate dead
    /// branches, and evaluate small pure calls when their arguments are known.
    /// Mixed calls can share a callee specialized for statically known arguments.
    /// Calls beyond the optional specialization budget use a generic version.
    pub fn simplified(&self) -> Result<Function, String> {
        let mut evaluator = ConstEvaluator::new();
        let mut jobs = vec![simplify_one(self, &mut evaluator, None)?];
        let mut memo = HashMap::<CallKey, Arc<Function>>::new();
        let mut budget = VariantBudget {
            per_callee: HashMap::new(),
            count: 0,
            work: VARIANT_WORK_BUDGET,
        };
        // Simplify only surviving callees. An application can share a helper
        // at arbitrarily many sites; equivalent contexts retain one shared Arc.
        // The original root borrows the entire immutable call DAG throughout.
        loop {
            let job = jobs.last_mut().unwrap();
            match job.simplified.code.get_mut(job.cursor) {
                Some(Instr::Call { function, .. }) => {
                    let known = job.call_facts.next().expect("missing facts for call");
                    let key = budget.key(function, known, &memo);
                    let call_index = job.cursor;
                    job.cursor += 1;
                    if let Some(cached) = memo.get(&key) {
                        *function = Arc::clone(cached);
                    } else {
                        let child =
                            simplify_one(function, &mut evaluator, Some((call_index, key)))?;
                        jobs.push(child);
                    }
                }
                Some(_) => job.cursor += 1,
                None => {
                    let mut job = jobs.pop().unwrap();
                    // Specialized callees can eliminate the last fallible operation.
                    if job.simplified.needs_status() {
                        job.simplified.refresh_status();
                    }
                    match job.parent {
                        Some((call_index, key)) => {
                            let parent = jobs.last_mut().unwrap();
                            let Instr::Call { function, .. } =
                                &mut parent.simplified.code[call_index]
                            else {
                                unreachable!("pending callee lost its caller")
                            };
                            *function = Arc::new(job.simplified);
                            memo.insert(key, Arc::clone(function));
                        }
                        None => return Ok(job.simplified),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
