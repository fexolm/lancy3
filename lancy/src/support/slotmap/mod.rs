//! Typed indices for IR entities and the side tables keyed by them.

mod dense;
mod key;
mod slots;

pub(crate) use dense::DenseMap;
pub(crate) use key::{SlotKey, slot_key};
pub(crate) use slots::SlotMap;

#[cfg(test)]
mod tests;
