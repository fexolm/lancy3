mod cfg;
mod domtree;
mod natural_loops;
pub(crate) mod passes;
#[cfg(test)]
mod tests;

pub use cfg::ControlFlowGraph;
pub use domtree::DomTree;
pub(crate) use natural_loops::NaturalLoops;
