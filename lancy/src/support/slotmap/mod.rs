//! Typed indices for IR entities and the side tables keyed by them.

mod dense;
mod key;
mod slots;

pub use dense::DenseMap;
pub use key::SlotKey;
pub use slots::SlotMap;
