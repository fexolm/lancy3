mod compile;
mod emit;
mod frame;
pub(crate) use splice_target::Features;
mod inst;
mod isel;
#[cfg(test)]
mod tests;

pub(crate) use emit::emit;
