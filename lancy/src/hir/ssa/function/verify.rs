use super::{Block, Call, Function, Op, Terminator, Type, Value};
use crate::CompileError;
use crate::hir::{
    IrError,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::module::{FunctionId, Module};
use crate::support::slotmap::DenseMap;

pub(crate) struct VerifiedBody<'a> {
    pub function: &'a Function,
    pub cfg: ControlFlowGraph,
    pub dom: DomTree,
    pub has_call_scratch: bool,
    /// The module ID of each call site's callee; `INVALID` for a call
    /// record that no instruction uses.
    callees: DenseMap<Call, FunctionId>,
}

impl VerifiedBody<'_> {
    pub(crate) fn callee(&self, call: Call) -> FunctionId {
        self.callees[call]
    }

    /// The body of `function`, which has the blocks, terminators and calls
    /// of this one and more instructions inside its blocks (see
    /// `transforms::canonical`), so the analyses still describe it.
    pub(crate) fn with_instructions<'b>(&self, function: &'b Function) -> VerifiedBody<'b> {
        if cfg!(debug_assertions) {
            function
                .verify()
                .unwrap_or_else(|error| panic!("expanded body: {error}"));
        }
        VerifiedBody {
            function,
            cfg: self.cfg.clone(),
            dom: self.dom.clone(),
            has_call_scratch: self.has_call_scratch,
            callees: self.callees.clone(),
        }
    }
}

impl Function {
    pub(in crate::hir) fn checked_successors(
        &self,
        block: Block,
    ) -> Result<[Option<Block>; 2], IrError> {
        let term = self.checked_terminator(block)?;
        let mut successors = [None; 2];
        for (i, edge) in term.edges().enumerate() {
            let target = self.checked_target(block, edge)?;
            let (_, args) = self.edge(edge);
            let target_block = &self.blocks[target];
            if target == self.entry() {
                return Err(IrError::EntryBackedge { from: block });
            }
            if target_block.params.len() != args.len() {
                return Err(IrError::EdgeArity {
                    from: block,
                    edge,
                    target,
                    expected: target_block.params.len(),
                    actual: args.len(),
                });
            }
            successors[i] = Some(target);
        }
        Ok(successors)
    }

    /// Verify this body. Callee signatures participate in the check; code
    /// generation verifies every distinct callee body in the reachable graph.
    pub fn verify(&self) -> Result<(), IrError> {
        self.verified_body().map(drop)
    }

    fn verified_body(&self) -> Result<VerifiedBody<'_>, IrError> {
        if self.params(self.entry()).len() != self.arity {
            return Err(IrError::EntryArity {
                expected: self.arity,
                actual: self.params(self.entry()).len(),
            });
        }
        let cfg = ControlFlowGraph::compute(self)?;
        let dom = DomTree::compute(&cfg);
        let mut has_call_scratch = false;
        let mut defs = DenseMap::filled(self.value_count(), None);
        for block in self.blocks() {
            for &param in self.params(block) {
                defs[param] = Some((block, 0));
            }
            for (i, inst) in self.instructions(block).enumerate() {
                if let Some(value) = self.result(inst) {
                    defs[value] = Some((block, i + 1));
                }
            }
        }
        for block in self.blocks() {
            let used = |val: Value, index: usize| -> Result<(), IrError> {
                let &(def, position) = defs
                    .get(val)
                    .and_then(Option::as_ref)
                    .ok_or(IrError::UndefinedValue { value: val, block })?;
                if (block == def && position >= index) || !dom.dominates(def, block) {
                    return Err(IrError::DoesNotDominate {
                        value: val,
                        definition: def,
                        block,
                    });
                }
                Ok(())
            };
            for (i, inst) in self.instructions(block).enumerate() {
                let operand = |v: Value| {
                    used(v, i + 1)?;
                    Ok::<_, IrError>(self.value_type(v))
                };
                let op = *self.op(inst);
                has_call_scratch |= matches!(op, Op::CallScratch { .. });
                let expected = op.result_type(self, inst, operand)?;
                match (expected, self.result(inst)) {
                    (None, None) => {}
                    (Some(expected), Some(result)) => {
                        let actual = self.value_type(result);
                        if actual != expected {
                            return Err(IrError::ResultType {
                                inst,
                                expected,
                                actual,
                            });
                        }
                    }
                    _ => return Err(IrError::ReplacementResultCount { inst }),
                }
            }
            let term = *self.terminator(block);
            match term {
                Terminator::Branch { cond, .. } => {
                    used(cond, usize::MAX)?;
                    if self.value_type(cond) != Type::I32 {
                        return Err(IrError::BranchType {
                            block,
                            actual: self.value_type(cond),
                        });
                    }
                }
                Terminator::Return(value) => {
                    used(value, usize::MAX)?;
                    if self.value_type(value) != self.return_type {
                        return Err(IrError::ReturnType {
                            block,
                            expected: self.return_type,
                            actual: self.value_type(value),
                        });
                    }
                }
                Terminator::Jump(_) => {}
            }
            for edge in term.edges() {
                let (target, args) = self.edge(edge);
                for (&v, &param) in args.iter().zip(self.params(target)) {
                    used(v, usize::MAX)?;
                    if self.value_type(v) != self.value_type(param) {
                        return Err(IrError::EdgeArgumentType {
                            edge,
                            value: v,
                            expected: self.value_type(param),
                            actual: self.value_type(v),
                        });
                    }
                }
            }
        }
        Ok(VerifiedBody {
            function: self,
            cfg,
            dom,
            has_call_scratch,
            callees: DenseMap::new(),
        })
    }
}

impl Function {
    /// Verify this function and every distinct callee, numbered callee first.
    pub(crate) fn verified_module(&self) -> Result<Module<VerifiedBody<'_>>, CompileError> {
        let _span = tracing::debug_span!(target: "lancy::compile", "verify_lancy_hir").entered();
        let (mut module, ids) = Module::discover(self, |function, callees| {
            let body = function.verified_body()?;
            for inst in function.insts.values() {
                if let Op::Call(site) = inst.op {
                    callees.push(function.calls[site].callee.as_ref());
                }
            }
            Ok(body)
        })?;
        for body in module.functions.values_mut() {
            let calls = &body.function.calls;
            body.callees = DenseMap::from_fn(calls.len(), |call| {
                let callee = std::sync::Arc::as_ptr(&calls[call].callee);
                ids.get(&callee).copied().unwrap_or(FunctionId::INVALID)
            });
        }
        Ok(module)
    }
}
