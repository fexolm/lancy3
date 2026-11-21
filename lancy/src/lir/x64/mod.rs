mod compile;
mod emit;
mod frame;
pub(crate) use splice_target::Features;
mod inst;
mod isel;
mod schedule;
#[cfg(test)]
mod tests;

pub(crate) use emit::{emit, emit_debug};
#[cfg(test)]
pub(crate) use tests::filetests::check_machine;
