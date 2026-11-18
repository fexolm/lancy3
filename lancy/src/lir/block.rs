use std::ops::Range;

crate::slot_key!(pub(crate) BlockId);
crate::slot_key!(pub(crate) EdgeId);

#[derive(Clone, Copy)]
pub(super) enum Exit<R, Target> {
    Return(R),
    Jump(Target),
    Branch(R, Target, Target),
}
impl<R, Target> Exit<R, Target> {
    pub(super) fn successors(self) -> impl Iterator<Item = Target> {
        match self {
            Self::Return(_) => [None, None],
            Self::Jump(edge) => [Some(edge), None],
            Self::Branch(_, a, b) => [Some(a), Some(b)],
        }
        .into_iter()
        .flatten()
    }
}
pub(super) struct Block<R, Target> {
    pub ops: Range<usize>,
    pub exit: Exit<R, Target>,
}
