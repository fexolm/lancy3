use super::Rules;
use crate::lir::frame::argument_words;
use crate::{
    hir::{Call, Terminator},
    lir::aarch64::backend::inst::Inst,
    module::FunctionId,
};
use regalloc2::PRegSet;

impl Rules<'_> {
    pub(super) fn terminator(&mut self, term: Terminator, entry: bool) -> Inst {
        match term {
            Terminator::Return(v) => Inst::Return(self.ty(v), self.input(v), entry),
            Terminator::Jump(_) => Inst::Jump,
            Terminator::Branch { cond, .. } => Inst::Branch(self.input(cond)),
        }
    }

    pub(super) fn call(
        &mut self,
        site: Call,
        callee: impl Fn(Call) -> (FunctionId, PRegSet),
    ) -> Inst {
        let (_, args) = self.f.call(site);
        let types = args.iter().map(|&v| self.ty(v));
        let words: Vec<_> = argument_words(types, self.config.bytes).collect();
        for (&arg, (ty, word)) in args.iter().zip(words) {
            let input = self.input(arg);
            self.out.push(Inst::StoreArg(ty, word * 8, input));
        }
        let (target, writes) = callee(site);
        Inst::Call(target, writes, self.ty(self.result()), self.dst())
    }
}
