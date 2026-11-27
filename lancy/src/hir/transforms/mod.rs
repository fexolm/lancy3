pub(crate) mod canonical;
mod cfg;
mod domtree;
mod natural_loops;
pub(crate) mod passes;
#[cfg(test)]
mod tests;

pub(crate) use cfg::ControlFlowGraph;
pub(crate) use domtree::DomTree;
pub(crate) use natural_loops::NaturalLoops;
