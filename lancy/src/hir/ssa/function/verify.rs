use super::{Block, Function, Op, Terminator, Type, Value};
use crate::CompileError;
use crate::hir::{
    IrError, Ternary,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::module::{FunctionIds, Module};
use crate::support::slotmap::DenseMap;

pub(crate) struct VerifiedBody<'a> {
    pub function: &'a Function,
    pub cfg: ControlFlowGraph,
    pub dom: DomTree,
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
                let expect = |v, ty| {
                    let actual = operand(v)?;
                    if actual != ty {
                        return Err(IrError::OperandType {
                            inst,
                            value: v,
                            expected: ty,
                            actual,
                        });
                    }
                    Ok(())
                };
                let ty = match *self.op(inst) {
                    Op::Const(_) => Type::I32,
                    Op::Simd(op) => {
                        op.verify(inst, operand)?;
                        let Some(ty) = op.result_type(|v| self.value_type(v)) else {
                            continue;
                        };
                        ty
                    }
                    Op::Literal(lane, _) => lane.scalar(),
                    Op::Copy(v) => operand(v)?,
                    Op::Bin(op, a, b) => {
                        let ty = operand(a)?;
                        expect(b, ty)?;
                        if matches!(ty, Type::Vector(_)) || !op.supports(ty) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        op.result_type(ty)
                    }
                    Op::Unary(op, v) => {
                        let ty = operand(v)?;
                        if matches!(ty, Type::Vector(_)) || !op.supports(ty) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        op.result_type(ty)
                    }
                    Op::Ternary(op, a, b, c) => {
                        let ty = operand(b)?;
                        if matches!(ty, Type::Vector(_)) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        expect(
                            a,
                            if op == Ternary::Select {
                                ty.mask_type()
                            } else {
                                ty
                            },
                        )?;
                        expect(c, ty)?;
                        if ty
                            .lane()
                            .is_none_or(|l| op == Ternary::MulAdd && !l.float())
                        {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        ty
                    }
                    Op::Cast(to, v) | Op::Bitcast(to, v) => {
                        let ty = operand(v)?;
                        let valid = match (ty, to) {
                            // Expose the address bits for alignment arithmetic.
                            (Type::Ptr, Type::U64) => matches!(self.op(inst), Op::Bitcast(..)),
                            (Type::Ptr, _) | (_, Type::Ptr) => false,
                            (Type::Vector(_), _) | (_, Type::Vector(_)) => false,
                            _ => {
                                matches!(self.op(inst), Op::Cast(..))
                                    || ty.bytes(16) == to.bytes(16)
                            }
                        };
                        if !valid {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        to
                    }
                    Op::BoundsCheck(index, len) => {
                        expect(index, Type::I32)?;
                        expect(len, Type::I32)?;
                        continue;
                    }
                    Op::PtrOffset(base, index, _) => {
                        expect(base, Type::Ptr)?;
                        expect(index, Type::I32)?;
                        Type::Ptr
                    }
                    Op::Load(ty, addr) => {
                        if matches!(ty, Type::Vector(_)) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        expect(addr, Type::Ptr)?;
                        ty
                    }
                    Op::Store(addr, value) => {
                        expect(addr, Type::Ptr)?;
                        let ty = operand(value)?;
                        if matches!(ty, Type::Vector(_)) {
                            return Err(IrError::InvalidNumeric { inst, ty });
                        }
                        continue;
                    }
                    Op::DebugStore(_, value) => {
                        operand(value)?;
                        continue;
                    }
                    Op::DebugPoint(_) => continue,
                    Op::Call(id) => {
                        let site = self
                            .calls
                            .get(id)
                            .ok_or(IrError::UnknownCall { inst, call: id })?;
                        let args = &self.operands[site.args.range()];
                        if args.len() != site.callee.arity() {
                            return Err(IrError::CallArity {
                                call: id,
                                expected: site.callee.arity(),
                                actual: args.len(),
                            });
                        }
                        for (&arg, &ty) in args.iter().zip(site.callee.param_types()) {
                            expect(arg, ty)?;
                        }
                        site.callee.return_type()
                    }
                };
                let actual = self.value_type(self.result(inst).unwrap());
                if actual != ty {
                    return Err(IrError::ResultType {
                        inst,
                        expected: ty,
                        actual,
                    });
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
        })
    }
}

impl Function {
    pub(crate) fn verified_module(
        &self,
    ) -> Result<(Module<VerifiedBody<'_>>, FunctionIds<Self>), CompileError> {
        let _span = tracing::debug_span!(target: "splice::compile", "verify_lancy_hir").entered();
        Module::discover(self, |function, callees| {
            let body = function.verified_body()?;
            for inst in function.insts.values() {
                if let Op::Call(site) = inst.op {
                    callees.push(function.calls[site].callee.as_ref());
                }
            }
            Ok(body)
        })
    }
}
