//! Publish native code to debuggers through the GDB JIT interface.
#![deny(unsafe_op_in_unsafe_fn)]

mod dwarf;
mod elf;
#[cfg(target_os = "linux")]
mod perf_map;
mod registry;
#[cfg(test)]
mod tests;

pub use gimli;
use gimli::write::FrameTable;
use splice_core::debug::{Location, Type};

/// A source statement and the machine range with its saved values.
pub struct Statement {
    pub address: u64,
    pub source: Location,
    pub register: u16,
    pub variables: Vec<(&'static str, Type, VariableLocation)>,
}

pub enum VariableLocation {
    Memory(i64),
    Constant(u64),
}

/// Installed machine code for one native function.
pub struct Function {
    pub name: String,
    pub address: u64,
    pub size: u64,
    pub statements: Vec<Statement>,
}

impl Function {
    pub fn statement(
        &mut self,
        function: &splice_core::Function,
        point: &splice_core::debug::Point,
        address: u64,
        register: u16,
        base: i64,
        vector_bytes: usize,
    ) {
        self.statements.push(Statement {
            address,
            source: point.location,
            register,
            variables: point
                .slots(function, vector_bytes)
                .map(|(var, ty, offset)| {
                    (var.name, ty, VariableLocation::Memory(base + offset as i64))
                })
                .chain(point.variables.iter().filter_map(|var| match var.value {
                    splice_core::debug::Value::Constant(Some(value)) => {
                        let (ty, bits) = value.parts();
                        Some((
                            var.name,
                            Type::Constant(ty),
                            VariableLocation::Constant(bits),
                        ))
                    }
                    _ => None,
                }))
                .collect(),
        });
    }
}

/// Owns a symbol file until the kernel is dropped. Store this before the
/// executable memory owner so the debugger sees removal before unmapping.
pub struct Registration {
    _entry: registry::Registration,
}

impl Registration {
    pub fn new(functions: &[Function], frames: &FrameTable) -> splice_core::Result<Self> {
        let sections = dwarf::sections(functions, frames)
            .map_err(|e| splice_core::BuildError::backend("JIT debug info", e))?;
        let file = elf::symbols(functions, &sections)?;
        tracing::debug!(target: "splice_core::compile", count = functions.len(), "register_jit_symbols");
        Ok(Self {
            _entry: registry::Registration::new(file),
        })
    }
}

/// Publish names to Linux perf on request, including for optimized code.
pub fn perf_enabled() -> bool {
    cfg!(target_os = "linux") && std::env::var_os("SPLICE_PERF_MAP").is_some_and(|v| v == "1")
}

/// Reserve the map before a native backend that can write it on its own.
pub fn prepare_perf_map() {
    #[cfg(target_os = "linux")]
    if perf_enabled() {
        perf_map::with_file(|_| Ok(()));
    }
}

pub fn profile(functions: &[Function]) {
    #[cfg(target_os = "linux")]
    if perf_enabled() && !functions.is_empty() {
        use std::fmt::Write as _;
        use std::io::Write as _;
        let mut lines = String::new();
        for function in functions {
            let name = function.name.replace(['\n', '\r'], "_");
            writeln!(lines, "{:x} {:x} {name}", function.address, function.size).unwrap();
        }
        perf_map::with_file(|file| file.write_all(lines.as_bytes()));
    }
    #[cfg(not(target_os = "linux"))]
    let _ = functions;
}
