//! Lowering state per HIR block, created alongside the blocks.
use lancy::hir::Block;
use std::ops::{Index, IndexMut};

/// A side table with one entry for each block of the function being built.
#[derive(Debug)]
pub(super) struct BlockMap<T>(Vec<T>);

impl<T> BlockMap<T> {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }

    pub(super) fn filled(len: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self(vec![value; len])
    }

    pub(super) fn from_fn(len: usize, make: impl FnMut(usize) -> T) -> Self {
        Self((0..len).map(make).collect())
    }

    /// Add the entry of the next block and return that block.
    pub(super) fn insert(&mut self, value: T) -> Block {
        self.0.push(value);
        Block::from_index(self.0.len() - 1)
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (Block, &T)> {
        self.0
            .iter()
            .enumerate()
            .map(|(i, value)| (Block::from_index(i), value))
    }

    pub(super) fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<T> Index<Block> for BlockMap<T> {
    type Output = T;
    fn index(&self, block: Block) -> &T {
        &self.0[block.index()]
    }
}

impl<T> IndexMut<Block> for BlockMap<T> {
    fn index_mut(&mut self, block: Block) -> &mut T {
        &mut self.0[block.index()]
    }
}
