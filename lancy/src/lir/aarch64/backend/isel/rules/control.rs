use super::Rules;
use crate::{
    hir::{Call, Function, Terminator},
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
        callee: impl Fn(&Function) -> (FunctionId, PRegSet),
    ) -> Inst {
        let (function, args) = self.f.call(site);
        let mut offset = 0;
        for &arg in args {
            let ty = self.ty(arg);
            let input = self.input(arg);
            self.out.push(Inst::StoreArg(ty, offset, input));
            offset += ty.bytes(self.config.bytes).next_multiple_of(8);
        }
        let (target, writes) = callee(function);
        Inst::Call(target, writes, self.ty(self.result()), self.dst())
    }
}
