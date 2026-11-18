//! Compile Splice functions with LLVM's host-target O3 pipeline and ORC LLJIT.
//! The frontend emits instructions directly into an Inkwell module.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(all(target_arch = "x86_64", any(windows, target_os = "linux"))))]
compile_error!("splice-llvm currently supports x86-64 Windows and Linux");

mod jit;
mod lower;

use inkwell::{context::Context, module::Module, targets::TargetMachine};
use splice::{BufferArg, BufferKind, Codegen, Executable, Function};

/// In-process LLVM code generator.
pub struct Llvm;

impl Llvm {
    /// Report the linked LLVM version after checking it matches the bindings.
    pub fn version() -> Result<(u32, u32, u32), String> {
        #[cfg(windows)]
        const EXPECTED: u32 = 19;
        #[cfg(target_os = "linux")]
        const EXPECTED: u32 = 21;
        let version @ (major, minor, patch) = inkwell::support::get_llvm_version();
        if major != EXPECTED {
            return Err(format!(
                "splice-llvm was built for LLVM {EXPECTED}, but the process loaded LLVM {major}.{minor}.{patch}"
            ));
        }
        Ok(version)
    }
}

fn build_module<'ctx>(
    context: &'ctx Context,
    function: &Function,
    target: &TargetMachine,
) -> Result<Module<'ctx>, String> {
    let module = context.create_module("splice");
    module.set_triple(&target.get_triple());
    module.set_data_layout(&target.get_target_data().get_data_layout());
    lower::populate(context, &module, function)?;
    module.verify().map_err(|error| error.to_string())?;
    Ok(module)
}

/// Print LLVM's verified module before host-target optimization.
pub fn to_llvm_ir(function: &Function) -> Result<String, String> {
    Llvm::version()?;
    let context = Context::create();
    Ok(build_module(&context, function, &jit::host_machine()?)?
        .print_to_string()
        .to_string())
}

#[repr(C)]
struct Descriptor {
    data: *mut i32,
    len: usize,
}

const _: () = {
    assert!(std::mem::offset_of!(Descriptor, len) == 8);
    assert!(std::mem::size_of::<Descriptor>() == 16);
};

type Entry = unsafe extern "C" fn(*const i32, *const Descriptor, *mut i32) -> i32;

/// Owns the LLVM session and the executable kernel it compiled.
pub struct LlvmKernel {
    _owner: jit::Jit,
    entry: Entry,
    arity: usize,
    buffers: Box<[BufferKind]>,
}

impl Codegen for Llvm {
    type Kernel = LlvmKernel;

    fn compile(function: &Function) -> Result<Self::Kernel, String> {
        Self::version()?;
        let (owner, entry) = jit::Jit::compile(function)?;
        Ok(LlvmKernel {
            _owner: owner,
            entry,
            arity: function.arity(),
            buffers: function.buffer_kinds().into(),
        })
    }
}

impl Executable for LlvmKernel {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32, String> {
        if self.arity != args.len() {
            return Err(format!(
                "expected {} scalar arguments, got {}",
                self.arity,
                args.len()
            ));
        }
        splice::validate_buffers(&self.buffers, buffers)?;
        let descriptors: Vec<_> = buffers
            .iter_mut()
            .map(|buffer| match buffer {
                BufferArg::Read(s) => Descriptor {
                    data: s.as_ptr().cast_mut(),
                    len: s.len(),
                },
                BufferArg::Write(s) => Descriptor {
                    data: s.as_mut_ptr(),
                    len: s.len(),
                },
            })
            .collect();
        let mut status = 0;
        // Signature and permissions were checked above. All descriptor pointers
        // refer to live borrows, and the generated code checks every index.
        let result = unsafe { (self.entry)(args.as_ptr(), descriptors.as_ptr(), &mut status) };
        match status {
            0 => Ok(result),
            1 => Err("kernel slice index out of bounds".into()),
            2 => Err("kernel integer division by zero".into()),
            3 => Err("kernel signed division overflow".into()),
            code => Err(format!("unexpected LLVM kernel status {code}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splice::kernel;

    #[kernel]
    fn update(#[comptime] amount: i32, data: &mut [i32]) -> i32 {
        data[0] += amount;
        data[0]
    }

    #[test]
    fn concurrent_compilation_and_calls_have_independent_symbol_ownership() {
        let base = update::build::<Llvm>(1).unwrap();
        std::thread::scope(|scope| {
            for delta in [7, 11, 13] {
                let base = &base;
                scope.spawn(move || {
                    let mut value = [0];
                    let specialized = update::build::<Llvm>(delta).unwrap();
                    assert_eq!(
                        specialized
                            .call(&[], &mut [BufferArg::Write(&mut value)])
                            .unwrap(),
                        delta
                    );
                    drop(specialized);
                    assert_eq!(
                        base.call(&[], &mut [BufferArg::Write(&mut value)]).unwrap(),
                        delta + 1
                    );
                });
            }
        });
    }
}
