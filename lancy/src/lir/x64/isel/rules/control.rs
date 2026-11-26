use super::Rules;
use crate::lir::frame::argument_words;
use crate::{
    hir::{Binary, Call, Op, Simd, Terminator, Value},
    lir::x64::inst::{Cond, Inst, Int},
    module::FunctionId,
};
use regalloc2::PRegSet;

impl Rules<'_> {
    pub(super) fn terminator(&mut self, term: Terminator) -> Inst {
        match term {
            Terminator::Return(v) => Inst::Ret(self.input(v)),
            Terminator::Jump(_) => Inst::Jump,
            Terminator::Branch { cond, .. } => match self.ctx.single_use(cond) {
                Some(Op::Bin(kind, a, b)) if self.integer_condition(kind, a).is_some() => {
                    let (cc, int) = self.integer_condition(kind, a).unwrap();
                    Inst::BrCmp(cc, int.bytes, self.input(a), self.rhs(b))
                }
                Some(Op::Simd(Simd::AnyTrue(v))) => Inst::BranchAny(self.vector_mask(v)),
                Some(Op::Simd(Simd::AllTrue(v))) if self.masks.contains(v) => {
                    Inst::BranchAll(self.ty(v).lane().unwrap(), self.input(v))
                }
                Some(Op::Simd(Simd::AllTrue(v))) if cfg!(feature = "experimental-simd") => self
                    .experimental_all_true(v)
                    .unwrap_or_else(|| Inst::Branch(self.input(cond))),
                _ => Inst::Branch(self.input(cond)),
            },
        }
    }

    /// The condition code of an integer comparison.
    pub(super) fn integer_condition(&self, kind: Binary, lhs: Value) -> Option<(Cond, Int)> {
        let int = Int::of(self.ty(lhs))?;
        Some((Cond::from_binary(kind, int.signed)?, int))
    }

    pub(super) fn call(
        &mut self,
        site: Call,
        callee: impl Fn(Call) -> (FunctionId, PRegSet),
    ) -> Inst {
        let (_, args) = self.f.call(site);
        let types = args.iter().map(|&v| self.ty(v));
        let words: Vec<_> = argument_words(types, self.cpu.bytes).collect();
        for (&arg, (ty, index)) in args.iter().zip(words) {
            let value = self.input(arg);
            self.out.push(Inst::StoreArg(index, ty, value));
        }
        let (target, written) = callee(site);
        let dst = self.dst();
        self.body.details_mut().call(target, dst, written)
    }
}
