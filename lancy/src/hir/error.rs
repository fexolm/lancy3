use super::{Block, Call, Edge, Inst, Lane, Type, Value, ssa::SliceParam};
use thiserror::Error;

/// A structural, typing or dominance error in an SSA function or an attempted edit.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum IrError {
    #[error("unknown SSA instruction {inst:?}")]
    UnknownInstruction { inst: Inst },
    #[error("retired SSA instruction {inst:?}")]
    RetiredInstruction { inst: Inst },
    #[error("replacement for {inst:?} must preserve the SSA result count")]
    ReplacementResultCount { inst: Inst },
    #[error("unknown SSA block {block:?}")]
    UnknownBlock { block: Block },
    #[error("cannot merge {block:?} with itself")]
    MergeSameBlock { block: Block },
    #[error("SSA predecessor {block:?} must end in an unconditional jump")]
    ExpectedJump { block: Block },
    #[error("SSA jump from {from:?} targets {actual:?}, expected {expected:?}")]
    WrongJumpTarget {
        from: Block,
        expected: Block,
        actual: Block,
    },
    #[error("SSA successor {block:?} requires one incoming edge, found {incoming}")]
    MergePredecessors { block: Block, incoming: usize },
    #[error("unterminated {block:?}")]
    UnterminatedBlock { block: Block },
    #[error("unknown SSA edge {edge:?} from {from:?}")]
    UnknownEdge { from: Block, edge: Edge },
    #[error("unknown target {target:?} of {edge:?} from {from:?}")]
    UnknownTarget {
        from: Block,
        edge: Edge,
        target: Block,
    },
    #[error("branch from {from:?} into SSA ABI entry block")]
    EntryBackedge { from: Block },
    #[error(
        "edge {edge:?} from {from:?} to {target:?} has wrong arity: expected {expected}, got {actual}"
    )]
    EdgeArity {
        from: Block,
        edge: Edge,
        target: Block,
        expected: usize,
        actual: usize,
    },
    #[error("unreachable SSA block {block:?}")]
    UnreachableBlock { block: Block },
    #[error("SSA entry parameter count {actual} disagrees with function arity {expected}")]
    EntryArity { expected: usize, actual: usize },
    #[error("undefined {value:?} used in {block:?}")]
    UndefinedValue { value: Value, block: Block },
    #[error("{value:?} defined in {definition:?} does not dominate its use in {block:?}")]
    DoesNotDominate {
        value: Value,
        definition: Block,
        block: Block,
    },
    #[error("expected {expected:?} operand in {inst:?}, got {actual:?} for {value:?}")]
    OperandType {
        inst: Inst,
        value: Value,
        expected: Type,
        actual: Type,
    },
    /// An `i32` literal is `Op::Const`; other literals hold only the bits of
    /// their lane. See `Op::literal`.
    #[error("non-canonical literal in {inst:?}: {bits:#x} as {lane:?}")]
    NonCanonicalLiteral { inst: Inst, lane: Lane, bits: u128 },
    /// See `Function::set_slice_params`.
    #[error("invalid slice parameter {slice:?}")]
    InvalidSliceParam { slice: SliceParam },
    #[error("{inst:?} stops with {trap:?}, which only a Rust entry reports")]
    InvalidTrap { inst: Inst, trap: crate::Trap },
    #[error("{inst:?} defines no value")]
    NoResult { inst: Inst },
    #[error("invalid numeric operation in {inst:?} for {ty:?}")]
    InvalidNumeric { inst: Inst, ty: Type },
    /// Pointers convert only by a bitcast to `U64`, which exposes the
    /// address bits for alignment arithmetic. Every other conversion would
    /// let integer arithmetic forge an address.
    #[error("{inst:?} converts {from:?} to {to:?}; a pointer only bitcasts to U64")]
    PointerCast { inst: Inst, from: Type, to: Type },
    #[error("unknown SSA call {call:?} in {inst:?}")]
    UnknownCall { inst: Inst, call: Call },
    #[error("invalid Rust import packet in {inst:?}")]
    InvalidRustPacket { inst: Inst },
    #[error("scratch memory in {inst:?} is unaligned or exceeds 4 GiB")]
    InvalidScratch { inst: Inst },
    #[error("wrong scalar arity at SSA call {call:?}: expected {expected}, got {actual}")]
    CallArity {
        call: Call,
        expected: usize,
        actual: usize,
    },
    #[error("result type mismatch in {inst:?}: expected {expected:?}, got {actual:?}")]
    ResultType {
        inst: Inst,
        expected: Type,
        actual: Type,
    },
    #[error("expected i32 branch condition in {block:?}, got {actual:?}")]
    BranchType { block: Block, actual: Type },
    #[error("wrong SSA return type in {block:?}: expected {expected:?}, got {actual:?}")]
    ReturnType {
        block: Block,
        expected: Type,
        actual: Type,
    },
    #[error(
        "wrong SSA edge argument type for {value:?} on {edge:?}: expected {expected:?}, got {actual:?}"
    )]
    EdgeArgumentType {
        edge: Edge,
        value: Value,
        expected: Type,
        actual: Type,
    },
}
