use super::Rules;
use crate::{
    hir::Value,
    lir::aarch64::backend::{
        arrangements,
        inst::{Expand, Inst, Shuffle},
        sve,
    },
};
use splice_target::Profile;

impl Rules<'_> {
    pub(super) fn compress(&mut self, value: Value, mask: Value) -> Inst {
        Inst::Compact {
            size: self.ty(value).lane().unwrap().bytes(),
            scalable: sve(self.config.profile),
            dst: self.dst(),
            value: self.input(value),
            mask: self.input(mask),
        }
    }

    pub(super) fn expand(&mut self, value: Value, mask: Value) -> Inst {
        let size = self.ty(value).lane().unwrap().bytes();
        let form = if size == 16 {
            Expand::Wide(self.config.profile)
        } else {
            let (neon, scalable) = arrangements(size);
            match self.config.profile {
                Profile::Sve2 if size >= 4 => Expand::Sve2(scalable),
                Profile::Sve | Profile::Sve2 => Expand::Sve(scalable),
                _ => Expand::Neon(neon, size),
            }
        };
        Inst::Expand(form, self.dst(), self.input(value), self.input(mask))
    }

    pub(super) fn shuffle(&mut self, a: Value, b: Value, indices: Value) -> Inst {
        let size = self.ty(a).lane().unwrap().bytes();
        let form = if size == 16 {
            Shuffle::Wide(sve(self.config.profile))
        } else if sve(self.config.profile) {
            let (_, scalable) = arrangements(size);
            Shuffle::Sve(scalable)
        } else {
            let (neon, _) = arrangements(size);
            Shuffle::Neon(neon, size)
        };
        Inst::Shuffle(
            form,
            self.dst(),
            [self.input(a), self.input(b), self.input(indices)],
        )
    }
}
