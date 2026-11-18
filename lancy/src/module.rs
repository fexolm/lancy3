//! A module owns the function bodies for one compilation stage. Function IDs
//! are stable across lowering and numbered in callee-first order.
use crate::{CompileError, call_graph::CallGraph, support::slotmap::DenseMap};
use std::collections::HashMap;

crate::slot_key!(pub(crate) FunctionId);
pub(crate) type FunctionIds<T> = HashMap<*const T, FunctionId>;

pub(crate) struct Module<T> {
    pub functions: DenseMap<FunctionId, T>,
    pub call_graph: CallGraph,
}

impl<T> Module<T> {
    /// Inspect each distinct source body once. The callback supplies its
    /// direct callees and the payload to retain at this stage of compilation.
    pub fn discover<'a, N>(
        root: &'a N,
        mut inspect: impl FnMut(&'a N, &mut Vec<&'a N>) -> Result<T, CompileError>,
    ) -> Result<(Self, FunctionIds<N>), CompileError> {
        enum Visit<'a, N, T> {
            Enter(&'a N),
            Finish(&'a N, T, Vec<&'a N>),
        }
        let mut ids = FunctionIds::new();
        let mut functions = DenseMap::new();
        let mut callees = DenseMap::new();
        let mut pending = vec![Visit::Enter(root)];
        while let Some(visit) = pending.pop() {
            match visit {
                Visit::Enter(node) => {
                    let key = std::ptr::from_ref(node);
                    if let Some(&id) = ids.get(&key) {
                        if id == FunctionId::INVALID {
                            return Err(CompileError::RecursiveCall);
                        }
                        continue;
                    }
                    ids.insert(key, FunctionId::INVALID);
                    let mut children = Vec::new();
                    let body = inspect(node, &mut children)?;
                    let first_child = pending.len();
                    pending.extend(children.iter().rev().copied().map(Visit::Enter));
                    pending.insert(first_child, Visit::Finish(node, body, children));
                }
                Visit::Finish(node, body, children) => {
                    let id = functions.insert(body);
                    callees.insert(
                        children
                            .into_iter()
                            .map(|c| ids[&std::ptr::from_ref(c)])
                            .collect(),
                    );
                    ids.insert(std::ptr::from_ref(node), id);
                }
            }
        }
        let root = ids[&std::ptr::from_ref(root)];
        Ok((
            Self {
                functions,
                call_graph: CallGraph::new(root, callees),
            },
            ids,
        ))
    }

    /// Translate the bodies while retaining the same call graph. Each
    /// callee's translated body is already available to its caller.
    pub fn try_map_functions<U>(
        self,
        mut lower: impl FnMut(
            FunctionId,
            &T,
            &[FunctionId],
            &DenseMap<FunctionId, U>,
        ) -> Result<U, CompileError>,
    ) -> Result<Module<U>, CompileError> {
        let mut functions = DenseMap::with_capacity(self.functions.len());
        for (id, source) in self.functions.iter() {
            functions.insert(lower(id, source, self.call_graph.callees(id), &functions)?);
        }
        Ok(Module {
            functions,
            call_graph: self.call_graph,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_callees_keep_their_ids_after_lowering() {
        let nodes = [0, 1, 2, 3];
        let edges: [&[usize]; 4] = [&[1, 2], &[3], &[3], &[]];
        let (module, ids) = Module::discover(&nodes[0], |n, children| {
            children.extend(edges[*n].iter().map(|&i| &nodes[i]));
            Ok(*n)
        })
        .unwrap();
        assert_eq!(module.functions.len(), 4);
        let leaf = ids[&std::ptr::from_ref(&nodes[3])];
        let module = module
            .try_map_functions(|id, _, callees, counts| {
                assert!(callees.iter().all(|&c| c < id));
                Ok(1usize + callees.iter().map(|&c| counts[c]).sum::<usize>())
            })
            .unwrap();
        assert_eq!(module.functions[leaf], 1);
        assert_eq!(module.functions[module.call_graph.root()], 5);
    }

    #[test]
    fn recursion_is_reported_during_discovery() {
        let nodes = [0, 1];
        let result = Module::discover(&nodes[0], |n, children| {
            children.push(&nodes[1 - *n]);
            Ok(())
        });
        assert!(matches!(result, Err(CompileError::RecursiveCall)));
    }
}
