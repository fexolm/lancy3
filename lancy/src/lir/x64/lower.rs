use super::inst::{Alu, Cond, Inst};
use crate::hir::{Binary, Function, Op, Terminator, Value, transforms::ControlFlowGraph};
use crate::{CompileError, support::slotmap::DenseMap};
use crate::{lir, module::FunctionId};
use regalloc2::{Block, PRegSet, RegClass, VReg};

fn reg(v: Value) -> VReg {
    VReg::new(v.index(), RegClass::Int)
}

pub(super) fn lower(
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
        for i in f.instructions(b) {
            let dst = || reg(f.result(i).expect("verified value-producing op"));
            let inst = match *f.op(i) {
                Op::Const(n) => Inst::Imm(dst(), n),
                Op::Copy(v) => Inst::Copy(dst(), reg(v)),
                Op::Bin(kind, a, b) => binary(&mut body, kind, dst(), reg(a), reg(b))?,
                Op::BoundsCheck(i, n) => Inst::CheckBounds(reg(i), reg(n)),
                Op::PtrOffset(base, index, scale) => {
                    let wide = body.temp(RegClass::Int)?;
                    body.push(Inst::Movsxd(wide, reg(index)));
                    Inst::Lea(dst(), reg(base), wide, scale)
                }
                Op::Load(ty, a) => Inst::Load(ty, dst(), reg(a)),
                Op::Store(a, v) => Inst::Store(f.value_type(v), reg(a), reg(v)),
                Op::Call(c) => {
                    let (callee, args) = f.call(c);
                    // Separate stores keep arbitrary arities out of a single
                    // instruction's set of simultaneous register constraints.
                    for (index, &arg) in args.iter().enumerate() {
                        body.push(Inst::StoreArg(index, reg(arg)));
                    }
                    let (target, written) = callee_info(callee);
                    Inst::call(target, dst(), written)
                }
            };
            body.push(inst);
        }
        let term = *f.terminator(b);
        body.push(match term {
            Terminator::Return(v) => Inst::Ret(reg(v)),
            Terminator::Jump(_) => Inst::Jump,
            Terminator::Branch { cond, .. } => Inst::Branch(reg(cond)),
        });
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

fn binary(
    body: &mut lir::Function<Inst>,
    kind: Binary,
    dst: VReg,
    a: VReg,
    b: VReg,
) -> Result<Inst, CompileError> {
    Ok(match kind {
        Binary::Div | Binary::Rem => {
            let hi = body.temp(RegClass::Int)?;
            body.push(Inst::Cdq { lo: a, hi });
            Inst::IDiv {
                remainder: kind == Binary::Rem,
                dst,
                lo: a,
                hi,
                divisor: b,
            }
        }
        Binary::Shl | Binary::Shr => Inst::Shift {
            left: kind == Binary::Shl,
            dst,
            value: a,
            count: b,
        },
        Binary::Eq => Inst::Compare(Cond::Eq, dst, a, b),
        Binary::Ne => Inst::Compare(Cond::Ne, dst, a, b),
        Binary::Lt => Inst::Compare(Cond::Lt, dst, a, b),
        Binary::Le => Inst::Compare(Cond::Le, dst, a, b),
        Binary::Gt => Inst::Compare(Cond::Gt, dst, a, b),
        Binary::Ge => Inst::Compare(Cond::Ge, dst, a, b),
        Binary::Add => Inst::Alu(Alu::Add, dst, a, b),
        Binary::Sub => Inst::Alu(Alu::Sub, dst, a, b),
        Binary::Mul => Inst::Alu(Alu::Mul, dst, a, b),
        Binary::And => Inst::Alu(Alu::And, dst, a, b),
        Binary::Or => Inst::Alu(Alu::Or, dst, a, b),
        Binary::Xor => Inst::Alu(Alu::Xor, dst, a, b),
        Binary::Min => Inst::Alu(Alu::Min, dst, a, b),
        Binary::Max => Inst::Alu(Alu::Max, dst, a, b),
    })
}
