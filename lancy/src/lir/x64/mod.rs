mod compile;
mod emit;
mod frame;
pub(crate) use splice_target::Features;
mod inst;
mod isel;
#[cfg(test)]
mod tests;

pub(crate) use emit::emit;
#[cfg(test)]
pub(crate) use tests::filetests::check_machine;
