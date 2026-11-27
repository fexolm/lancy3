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
    /// A source point whose runtime variables were saved in `slots`, at
    /// `base` bytes from `register`.
    pub fn statement(
        &mut self,
        point: &splice_core::debug::Point,
        slots: impl IntoIterator<Item = splice_core::debug::Slot>,
        address: u64,
        register: u16,
        base: i64,
    ) {
        self.statements.push(Statement {
            address,
            source: point.location,
            register,
            variables: slots
                .into_iter()
                .map(|slot| {
                    (
                        slot.variable.name,
                        slot.ty,
                        VariableLocation::Memory(base + slot.offset as i64),
                    )
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

/// The symbol name of kernel `name` in debuggers and profilers.
pub fn symbol_name(name: &str) -> String {
    format!("splice_jit::{name}")
}

/// The symbols of one compiled kernel, collected while its code is
/// installed and published together by [`Publish::finish`].
pub struct Publish {
    debug: bool,
    functions: Vec<Function>,
    frames: FrameTable,
}

impl Publish {
    /// In a `debug` build the symbols also go to debuggers.
    pub fn new(debug: bool) -> Self {
        Self {
            debug,
            functions: Vec::new(),
            frames: FrameTable::default(),
        }
    }

    /// Installed code of the kernel function `name` (its source name), to
    /// which the caller adds source statements.
    pub fn function(&mut self, name: &str, address: u64, size: u64) -> &mut Function {
        self.functions.push(Function {
            name: symbol_name(name),
            address,
            size,
            statements: Vec::new(),
        });
        self.functions.last_mut().expect("just pushed")
    }

    /// Unwind tables for the published functions.
    pub fn frames(&mut self) -> &mut FrameTable {
        &mut self.frames
    }

    /// Give the names to perf when `profile` (a backend whose own writer
    /// is active passes `false`), register the symbols with debuggers in a
    /// debug build, and keep them until `memory`, the owner of the code,
    /// is dropped.
    pub fn finish<M>(self, profile: bool, memory: M) -> splice_core::Result<Published<M>> {
        if profile {
            self::profile(&self.functions);
        }
        let registration = self
            .debug
            .then(|| Registration::new(&self.functions, &self.frames))
            .transpose()?;
        Ok(Published {
            registration,
            memory,
        })
    }
}

/// Executable memory `M` whose symbols stay registered while it is mapped:
/// dropping it removes the symbols before the code.
pub struct Published<M> {
    registration: Option<Registration>,
    memory: M,
}

impl<M> Published<M> {
    pub fn memory(&self) -> &M {
        &self.memory
    }
}

impl<M> Drop for Published<M> {
    fn drop(&mut self) {
        // The debugger must see the removal before the code is unmapped,
        // which happens when `memory` drops after this.
        drop(self.registration.take());
    }
}

/// A symbol file registered with the GDB JIT interface.
struct Registration {
    _entry: registry::Registration,
}

impl Registration {
    fn new(functions: &[Function], frames: &FrameTable) -> splice_core::Result<Self> {
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

fn profile(functions: &[Function]) {
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
