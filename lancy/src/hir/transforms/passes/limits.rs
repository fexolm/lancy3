//! Budgets that keep each pass linear in the size of a function. They were
//! set by hand when each pass was written; no recorded measurement backs the
//! exact values and only `hoist_vector_values` depends on the target. A budget only stops
//! an optimization early: it never changes what a program computes. Change
//! one together with `cargo xtask bench` (build and call time).

use splice_target::Arch;

/// Vector-valued definitions (and scalar floats or 128-bit integers, which
/// also live in vector registers) in a linear loop body up to which loop
/// invariant code motion hoists more than `i32` arithmetic. Past it, values
/// hoisted across the backedge would compete with the loop for the vector
/// registers the allocator has: 15 on x86-64 (`xmm0..15` without the move
/// scratch, also with AVX-512) and 23 on AArch64 (the caller-saved
/// `v0..7`, `v17..31`). Each budget leaves three for the loop's
/// temporaries. Literals and their splats are not counted: a spilled one
/// reloads with one load, which is what rematerializing it costs.
pub(super) fn hoist_vector_values(arch: Arch) -> usize {
    match arch {
        Arch::X86_64 => 12,
        Arch::Aarch64 => 20,
    }
}

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
