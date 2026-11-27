//! AArch64 code generation and instruction encoding.
//!
//! The emitter selects an exact instruction form ([`Code`]) and its operands.
//! PC-relative operands carry target addresses, so a branch is encoded once
//! its destination is known.
mod backend;
mod instruction;
mod operand;

pub(crate) use backend::emit;
#[cfg(test)]
pub(crate) use backend::encoding;

pub(crate) use instruction::{Code, EncodeError, Feature, FeatureSet};
pub(crate) use operand::{Arrangement, ConditionCode, Operand, PredQual, Register, ShiftKind};

#[cfg(test)]
mod tests;
