//! The reachable call edges at a compilation stage. HIR transformations that
//! change call sites can produce a new graph. Lowering transfers this graph
//! intact to the LIR module, which only queries it.
use crate::{module::FunctionId, support::slotmap::DenseMap};

pub(crate) struct CallGraph {
    root: FunctionId,
    callees: DenseMap<FunctionId, Vec<FunctionId>>,
}

impl CallGraph {
    pub fn new(root: FunctionId, callees: DenseMap<FunctionId, Vec<FunctionId>>) -> Self {
        Self { root, callees }
    }
    pub fn root(&self) -> FunctionId {
        self.root
    }
    pub fn callees(&self, caller: FunctionId) -> &[FunctionId] {
        &self.callees[caller]
    }
}
