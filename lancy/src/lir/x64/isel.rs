use super::inst::{Address, Alu, Cond, Inst, Rhs};
use crate::hir::{Binary, Function, Op, Scale, Terminator, Value, transforms::ControlFlowGraph};
use crate::lir::isel::{self, Context, Root};
use crate::{CompileError, support::slotmap::DenseMap};
use crate::{lir, module::FunctionId};
use regalloc2::{Block, PRegSet, RegClass, VReg};

fn reg(v: Value) -> VReg {
    VReg::new(v.index(), RegClass::Int)
}

pub(super) fn select(
    f: &Function,
    cfg: &ControlFlowGraph,
    root: bool,
    callee_info: impl Fn(&Function) -> (FunctionId, PRegSet),
) -> Result<lir::Function<Inst>, CompileError> {
    let mut mapping = DenseMap::filled(f.block_capacity(), Block::invalid());
    let mut count = 0;
    let split = |source, target| {
        matches!(f.terminator(source), Terminator::Branch { .. })
            && cfg.predecessors(target).len() > 1
    };
    for &b in cfg.reverse_postorder() {
        mapping[b] = Block::new(count);
        count += 1 + cfg.successors(b).filter(|&target| split(b, target)).count();
    }
    let mut body = lir::Function::new(count, f.value_count())?;
    let mut blocks = isel::select(f, cfg, |ctx, root, out| {
        let mut rules = Rules {
            ctx,
            body: &mut body,
            out,
        };
        match root {
            Root::Term(term) => rules.branch(term),
            Root::Inst(i) => {
                let dst = || reg(f.result(i).expect("value-producing instruction"));
                let op = match *f.op(i) {
                    Op::Const(n) => Inst::Imm(dst(), n),
                    Op::Copy(v) => Inst::Copy(dst(), rules.input(v)),
                    Op::Bin(kind, a, b) => rules.binary(kind, dst(), a, b)?,
                    Op::BoundsCheck(i, n) => Inst::CheckBounds(rules.input(i), rules.input(n)),
                    Op::PtrOffset(base, i, scale) => {
                        Inst::Lea(dst(), rules.offset(base, i, scale)?)
                    }
                    Op::Load(ty, p) => Inst::Load(ty, dst(), rules.address(p)?),
                    Op::Store(p, v) => {
                        Inst::Store(f.value_type(v), rules.address(p)?, rules.input(v))
                    }
                    Op::Call(site) => {
                        let (callee, args) = f.call(site);
                        for (index, &arg) in args.iter().enumerate() {
                            let value = rules.input(arg);
                            rules.out.push(Inst::StoreArg(index, value));
                        }
                        let (target, written) = callee_info(callee);
                        Inst::call(target, dst(), written)
                    }
                };
                rules.out.push(op);
            }
        }
        Ok(())
    })?;
    for &b in cfg.reverse_postorder() {
        let entry = b == f.entry();
        if entry {
            let argv = if root && f.arity() != 0 {
                let addr = body.temp(RegClass::Int)?;
                body.push(Inst::Argv(addr));
                Some(addr)
            } else {
                None
            };
            for (index, &arg) in f.params(b).iter().enumerate() {
                body.push(Inst::LoadArg {
                    dst: reg(arg),
                    argv,
                    index,
                });
            }
        }
        for op in blocks[b].drain(..) {
            body.push(op);
        }
        let term = *f.terminator(b);
        // Each critical edge gets its own block, including two arms going
        // to the same join with different arguments. Keep it near its source
        // so that forward edges stay forward in the allocator's loop estimate.
        let mut bridge = mapping[b].index() + 1;
        for e in term.edges() {
            let (target, args) = f.edge(e);
            if split(b, target) {
                body.edge(Block::new(bridge), []);
                bridge += 1;
            } else {
                body.edge(mapping[target], args.iter().copied().map(reg));
            }
        }
        let params = if entry { &[] } else { f.params(b) };
        body.end_block(params.iter().copied().map(reg))?;
        for e in term.edges() {
            let (target, args) = f.edge(e);
            if split(b, target) {
                body.push(Inst::Jump);
                body.edge(mapping[target], args.iter().copied().map(reg));
                body.end_block([])?;
            }
        }
    }
    Ok(body)
}

struct Rules<'a> {
    ctx: &'a mut Context,
    body: &'a mut lir::Function<Inst>,
    out: &'a mut Vec<Inst>,
}

impl Rules<'_> {
    fn input(&mut self, v: Value) -> VReg {
        self.ctx.require(v);
        reg(v)
    }

    fn rhs(&mut self, v: Value) -> Rhs {
        match self.ctx.constant(v) {
            Some(n) => Rhs::Imm(n),
            None => Rhs::Reg(self.input(v)),
        }
    }

    fn branch(&mut self, term: Terminator) {
        let inst = match term {
            Terminator::Return(v) => Inst::Ret(self.input(v)),
            Terminator::Jump(_) => Inst::Jump,
            Terminator::Branch { cond, .. } => {
                if let Some(Op::Bin(kind, a, b)) = self.ctx.single_use(cond)
                    && let Some(cc) = Cond::from_binary(kind)
                {
                    Inst::BrCmp(cc, self.input(a), self.rhs(b))
                } else {
                    Inst::Branch(self.input(cond))
                }
            }
        };
        self.out.push(inst);
    }

    fn address(&mut self, v: Value) -> Result<Address, CompileError> {
        match self.ctx.single_use(v) {
            Some(Op::PtrOffset(base, index, scale)) => self.offset(base, index, scale),
            _ => Ok(Address {
                base: self.input(v),
                index: None,
                disp: 0,
            }),
        }
    }

    fn offset(&mut self, base: Value, index: Value, scale: Scale) -> Result<Address, CompileError> {
        let base = self.input(base);
        if let Some(disp) = self
            .ctx
            .constant(index)
            .and_then(|i| i.checked_mul(1 << scale as u8))
        {
            return Ok(Address {
                base,
                index: None,
                disp,
            });
        }
        let wide = self.body.temp(RegClass::Int)?;
        let index = self.input(index);
        self.out.push(Inst::Movsxd(wide, index));
        Ok(Address {
            base,
            index: Some((wide, scale)),
            disp: 0,
        })
    }

    fn binary(
        &mut self,
        kind: Binary,
        dst: VReg,
        mut a: Value,
        mut b: Value,
    ) -> Result<Inst, CompileError> {
        if let Some(cc) = Cond::from_binary(kind) {
            return Ok(Inst::Compare(cc, dst, self.input(a), self.rhs(b)));
        }
        if matches!(
            kind,
            Binary::Add | Binary::Mul | Binary::And | Binary::Or | Binary::Xor
        ) && self.ctx.constant(a).is_some()
            && self.ctx.constant(b).is_none()
        {
            std::mem::swap(&mut a, &mut b);
        }
        let a = self.input(a);
        Ok(match kind {
            Binary::Div | Binary::Rem => {
                let hi = self.body.temp(RegClass::Int)?;
                self.out.push(Inst::Cdq { lo: a, hi });
                Inst::IDiv {
                    remainder: kind == Binary::Rem,
                    dst,
                    lo: a,
                    hi,
                    divisor: self.input(b),
                }
            }
            Binary::Shl | Binary::Shr => Inst::Shift {
                left: kind == Binary::Shl,
                dst,
                value: a,
                count: self.rhs(b),
            },
            Binary::Add => Inst::Alu(Alu::Add, dst, a, self.rhs(b)),
            Binary::Sub => Inst::Alu(Alu::Sub, dst, a, self.rhs(b)),
            Binary::Mul => Inst::Alu(Alu::Mul, dst, a, self.rhs(b)),
            Binary::And => Inst::Alu(Alu::And, dst, a, self.rhs(b)),
            Binary::Or => Inst::Alu(Alu::Or, dst, a, self.rhs(b)),
            Binary::Xor => Inst::Alu(Alu::Xor, dst, a, self.rhs(b)),
            Binary::Min => Inst::Alu(Alu::Min, dst, a, Rhs::Reg(self.input(b))),
            Binary::Max => Inst::Alu(Alu::Max, dst, a, Rhs::Reg(self.input(b))),
            _ => unreachable!("comparisons handled above"),
        })
    }
}
