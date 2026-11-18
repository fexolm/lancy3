//! Installation and invocation of column projection kernels.
mod adaptive;
mod profile;

use super::Executable;
use crate::hir::projection::optimize;
use crate::hir::{Assumption, Program};
use crate::lir::emit_batch;
pub(super) use adaptive::BatchJob;
pub use adaptive::{AdaptiveKernel, KernelStats};
use std::mem::MaybeUninit;
use std::ptr;

type EntryPoint =
    unsafe extern "C" fn(*const *const i32, *mut i32, usize, *mut MaybeUninit<u128>) -> usize;

const _: () = assert!(align_of::<u128>() == 16);

/// Immutable machine code. A specialized kernel can return a partial result
/// when a guard fails. Use `AdaptiveKernel` for automatic recovery.
pub struct NativeKernel {
    memory: Executable,
    input_count: usize,
    spill_slots: usize,
}

impl NativeKernel {
    /// Compile the program directly into a scalar loop.
    pub fn baseline(program: &Program) -> Result<Self, String> {
        Self::compile(program, &[], false)
    }

    /// Fold constants under the given assumptions and emit a checked SIMD loop.
    /// Empty assumptions produce an unconditional SIMD kernel.
    pub fn specialize(program: &Program, assumptions: &[Assumption]) -> Result<Self, String> {
        let mut guards = assumptions.to_vec();
        guards.sort_unstable_by_key(|guard| guard.column);
        if guards
            .last()
            .is_some_and(|g| usize::from(g.column) >= program.input_count())
        {
            return Err("guard column is outside the program schema".into());
        }
        guards.dedup();
        if guards
            .windows(2)
            .any(|pair| pair[0].column == pair[1].column)
        {
            return Err("conflicting constants for the same column".into());
        }
        let simplified = optimize(program, &guards);
        Self::compile(&simplified, &guards, true)
    }

    fn compile(program: &Program, guards: &[Assumption], vectorize: bool) -> Result<Self, String> {
        let (code, spill_slots) = emit_batch(program, guards, vectorize)?;
        Ok(Self {
            memory: Executable::new(&code)?,
            input_count: program.input_count(),
            spill_slots,
        })
    }

    /// Size of the generated function in bytes, before OS page rounding.
    pub fn code_size(&self) -> usize {
        self.memory.len
    }

    /// Write a contiguous output prefix and return its length. A guard miss
    /// leaves the entire suffix untouched, including the failing SIMD block.
    pub fn run(&self, columns: &[&[i32]], out: &mut [i32]) -> Result<usize, String> {
        if columns.iter().any(|c| c.len() != out.len()) {
            return Err(format!(
                "expected {} columns of {} rows",
                self.input_count,
                out.len()
            ));
        }
        self.run_at(columns, 0, out)
    }

    /// Run over an output window using column pointers based at `start`.
    pub(crate) fn run_at(
        &self,
        columns: &[&[i32]],
        start: usize,
        out: &mut [i32],
    ) -> Result<usize, String> {
        let end = start.checked_add(out.len()).ok_or("row index overflow")?;
        if columns.len() != self.input_count || columns.iter().any(|c| c.len() < end) {
            return Err("native kernel window is outside the input batch".into());
        }
        if out.is_empty() {
            return Ok(0);
        }

        // Rust slices are fat pointers. Build the native array of thin pointers
        // on the stack for the common narrow-query case.
        let mut local = [ptr::null(); 16];
        let wide;
        let pointers = if columns.len() <= local.len() {
            for (slot, column) in local.iter_mut().zip(columns) {
                *slot = column[start..].as_ptr();
            }
            local.as_ptr()
        } else {
            wide = columns
                .iter()
                .map(|c| c[start..].as_ptr())
                .collect::<Vec<_>>();
            wide.as_ptr()
        };
        // Spills belong to this invocation. Generated functions never move the
        // stack pointer and are ordinary leaf frames for host stack walkers.
        let mut short_scratch = [MaybeUninit::<u128>::uninit(); 8];
        let mut long_scratch = Vec::<u128>::new();
        let scratch = if self.spill_slots <= short_scratch.len() {
            short_scratch.as_mut_ptr()
        } else {
            long_scratch
                .try_reserve_exact(self.spill_slots)
                .map_err(|e| format!("cannot allocate JIT spill area: {e}"))?;
            long_scratch.spare_capacity_mut().as_mut_ptr()
        };
        // SAFETY: Only our emitter creates this function, with the native C
        // ABI. The RX mapping and all slices outlive the call. Builder and guard
        // validation bound column indices, and the shape check covers every row.
        let processed = unsafe {
            let entry: EntryPoint = std::mem::transmute(self.memory.address.as_ptr());
            entry(pointers, out.as_mut_ptr(), out.len(), scratch)
        };
        if processed > out.len() {
            return Err("native kernel returned an invalid row count".into());
        }
        Ok(processed)
    }
}
