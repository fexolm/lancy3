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
    target: &TargetMachine,
    populate: impl FnOnce(&'ctx Context, &Module<'ctx>) -> Result<(), String>,
) -> Result<Module<'ctx>, String> {
    let module = context.create_module("splice");
    module.set_triple(&target.get_triple());
    module.set_data_layout(&target.get_target_data().get_data_layout());
    populate(context, &module)?;
    module.verify().map_err(|error| error.to_string())?;
    Ok(module)
}

/// Print LLVM's verified module before host-target optimization.
pub fn to_llvm_ir(function: &Function) -> Result<String, String> {
    Llvm::version()?;
    let context = Context::create();
    Ok(build_module(&context, &jit::host_machine()?, |ctx, m| {
        lower::populate(ctx, m, function)
    })?
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
    type Output = LlvmKernel;

    fn compile(function: &Function) -> splice::Result<Self::Output> {
        Self::version()?;
        let (owner, entry) = jit::Jit::compile(|ctx, m| lower::populate(ctx, m, function))?;
        Ok(LlvmKernel {
            _owner: owner,
            entry,
            arity: function.arity(),
            buffers: function.buffer_kinds().into(),
        })
    }

    fn build(
        arity: usize,
        buffers: &[BufferKind],
        body: splice::frontend::Body<'_>,
    ) -> splice::Result<Self::Output> {
        Self::version()?;
        let (owner, entry) =
            jit::Jit::compile(move |ctx, m| lower::populate_direct(ctx, m, arity, buffers, body))?;
        Ok(LlvmKernel {
            _owner: owner,
            entry,
            arity,
            buffers: buffers.into(),
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

    #[test]
    fn both_lowering_paths_offer_the_entire_kernel_to_o3() {
        use splice::{frontend::KernelBuilder, hir::Binary};
        fn staged(b: &mut dyn KernelBuilder) -> splice::Result<()> {
            let value = b.call_kernel(
                1,
                &[],
                &[b.arg(0)],
                &[],
                Box::new(|child| {
                    let value = child.call_kernel(
                        1,
                        &[],
                        &[child.arg(0)],
                        &[],
                        Box::new(|leaf| {
                            let one = leaf.constant(1);
                            let value = leaf.binary(Binary::Add, leaf.arg(0), one);
                            leaf.ret(value);
                            Ok(())
                        }),
                    )?;
                    child.ret(value);
                    Ok(())
                }),
            )?;
            b.ret(value);
            Ok(())
        }
        let ir = splice::frontend::materialize(1, &[], Box::new(staged)).unwrap();
        let target = jit::host_machine().unwrap();
        let ctx = Context::create();
        for direct in [false, true] {
            let m = build_module(&ctx, &target, |ctx, m| {
                if direct {
                    lower::populate_direct(ctx, m, 1, &[], Box::new(staged))
                } else {
                    lower::populate(ctx, m, &ir)
                }
            })
            .unwrap();
            for id in 0..3 {
                let helper = m.get_function(&format!("kernel{id}")).unwrap();
                assert_eq!(helper.get_linkage(), inkwell::module::Linkage::Internal);
                assert!(helper.get_first_basic_block().is_some());
            }
            jit::optimize(&m, &target).unwrap();
            let definitions: Vec<_> = m
                .get_functions()
                .filter(|f| f.get_first_basic_block().is_some())
                .collect();
            assert_eq!(definitions, [m.get_function("splice_entry").unwrap()]);
        }
        for native in [
            Llvm::build(1, &[], Box::new(staged)).unwrap(),
            Llvm::compile(&ir).unwrap(),
        ] {
            assert_eq!(native.call(&[i32::MAX], &mut []), Ok(i32::MIN));
        }
    }

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
