//! `filetests` only lists code and runs on every host; the other tests
//! execute x64 code.
#[cfg(target_arch = "x86_64")]
mod arithmetic;
#[cfg(target_arch = "x86_64")]
mod calls;
pub(super) mod filetests;
#[cfg(target_arch = "x86_64")]
mod isel;
#[cfg(target_arch = "x86_64")]
mod masked_memory;
#[cfg(target_arch = "x86_64")]
mod numeric;
#[cfg(target_arch = "x86_64")]
mod packed_code;
#[cfg(target_arch = "x86_64")]
mod packed_fp;
#[cfg(target_arch = "x86_64")]
mod packed_integer;
#[cfg(target_arch = "x86_64")]
mod vector_patterns;
