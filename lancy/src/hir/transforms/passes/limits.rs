//! Budgets that keep each pass linear in the size of a function. They were
//! set by hand when each pass was written; no recorded measurement backs the
//! exact values and they are the same for every target. A budget only stops
//! an optimization early: it never changes what a program computes. Change
//! one together with `cargo xtask bench` (build and call time).

/// Vector-valued definitions (and scalar floats or 128-bit integers, which
/// also live in vector registers) in a linear loop body up to which loop
/// invariant code motion hoists more than `i32` arithmetic. Past it, values
/// hoisted across the backedge would compete with the loop for the 16
/// vector registers of x86-64 and spill.
pub(super) const HOIST_VECTOR_VALUES: usize = 12;

/// Memory accesses remembered per block by load forwarding and dead store
/// removal. The oldest is forgotten first.
pub(super) const KNOWN_ACCESSES: usize = 16;

/// Accesses of one loop body that loop fusion compares pairwise.
pub(super) const FUSION_ACCESSES: usize = 8;

/// Blocks and instructions that unswitching clones for one loop.
pub(super) const UNSWITCH_BLOCKS: usize = 16;
pub(super) const UNSWITCH_COST: usize = 128;

/// Values and affine expressions that closing one recurrence tracks.
pub(super) const RECURRENCE_VALUES: usize = 32;
pub(super) const RECURRENCE_EXPRESSIONS: usize = 64;
