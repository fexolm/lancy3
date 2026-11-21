use super::Rules;
use crate::{
    hir::{Call, Function, Op, Simd, Terminator, Type},
    lir::x64::inst::{Cond, Inst},
    module::FunctionId,
};
use regalloc2::PRegSet;

impl Rules<'_> {
    pub(super) fn terminator(&mut self, term: Terminator) -> Inst {
        match term {
            Terminator::Return(v) => Inst::Ret(self.input(v)),
            Terminator::Jump(_) => Inst::Jump,
            Terminator::Branch { cond, .. } => match self.ctx.single_use(cond) {
                Some(Op::Simd(Simd::AnyTrue(v))) => Inst::BranchAny(self.vector_mask(v)),
                Some(Op::Simd(Simd::AllTrue(v))) if cfg!(feature = "experimental-simd") => self
                    .experimental_all_true(v)
                    .unwrap_or_else(|| Inst::Branch(self.input(cond))),
                Some(Op::Bin(kind, a, b)) if self.ty(a) == Type::I32 => {
                    if let Some(cc) = Cond::from_binary(kind) {
                        Inst::BrCmp(cc, self.input(a), self.rhs(b))
                    } else {
                        Inst::Branch(self.input(cond))
                    }
                }
                _ => Inst::Branch(self.input(cond)),
            },
        }
    }

    pub(super) fn call(
        &mut self,
        site: Call,
        callee: impl Fn(&Function) -> (FunctionId, PRegSet),
    ) -> Inst {
        let (function, args) = self.f.call(site);
        let mut index = 0;
        for &arg in args {
            let ty = self.ty(arg);
            let value = self.input(arg);
            self.out.push(Inst::StoreArg(index, ty, value));
            index += ty.bytes(self.cpu.bytes).div_ceil(8);
        }
        let (target, written) = callee(function);
        Inst::call(target, self.dst(), written)
    }
}
