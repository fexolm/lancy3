//! Compile Splice functions with LLVM's host-target O3 pipeline and ORC LLJIT.
//! The frontend emits instructions directly into an Inkwell module.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(any(
    all(target_arch = "x86_64", any(windows, target_os = "linux")),
    all(target_arch = "aarch64", target_os = "linux")
)))]
compile_error!("splice-llvm supports x86-64 Windows and Linux, and AArch64 Linux");

mod jit;
mod lower;

#[cfg(all(test, target_arch = "aarch64"))]
mod aarch64_tests;

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
    #[cfg(target_arch = "aarch64")]
    {
        use inkwell::attributes::{Attribute, AttributeLoc};
        use splice_target::Profile;
        let simd = splice_target::simd()?;
        if matches!(simd.profile, Profile::Sve | Profile::Sve2) {
            // LLVM encodes the minimum vscale in the high 32 bits and the
            // maximum in the low 32. SVE has at most 16 architectural quads.
            let range = ((simd.bytes as u64 / 16) << 32) | 16;
            let attr = context
                .create_enum_attribute(Attribute::get_named_enum_kind_id("vscale_range"), range);
            for function in module.get_functions() {
                if !function.as_global_value().is_declaration() {
                    function.add_attribute(AttributeLoc::Function, attr);
                }
            }
        }
    }
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
    data: *mut u8,
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
        #[cfg(target_arch = "aarch64")]
        splice_target::simd()?.check_execution()?;
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
            .map(|buffer| Descriptor {
                data: buffer.data(),
                len: buffer.len(),
            })
            .collect();
        let mut status = 0;
        // SAFETY: the target emits the host C ABI (Windows x64, SysV x64 or
        // AAPCS64). ORC owns the code for this borrow. Descriptors refer to
        // live slices, permissions were checked, and the kernel checks indices.
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

    #[test]
    fn full_vector_accesses_use_element_alignment_and_short_stores_preserve_the_prefix() {
        use splice::{
            frontend::{self, KernelBuilder},
            hir::{Buffer, Simd, SimdOp},
            simd::Element,
        };
        let kinds = [
            BufferKind::new(Element::U16, false),
            BufferKind::new(Element::U16, true),
        ];
        fn stage(b: &mut dyn KernelBuilder) -> splice::Result<()> {
            let offset = b.arg(0);
            let op = |op| Simd {
                element: Element::U16,
                op,
            };
            let value = b.simd(op(SimdOp::Load(Buffer(0))), &[offset])?.unwrap();
            let one = b.simd(op(SimdOp::Splat(1)), &[])?.unwrap();
            let sum = b.simd(op(SimdOp::Add), &[one, value])?.unwrap();
            b.simd(op(SimdOp::Store(Buffer(1))), &[offset, sum])?;
            b.ret(offset);
            Ok(())
        }
        let ir = frontend::materialize(1, &kinds, Box::new(stage)).unwrap();
        let lanes = Element::U16.lanes();
        let ty = format!("<{lanes} x i16>");
        let target = jit::host_machine().unwrap();
        let ctx = Context::create();
        for direct in [true, false] {
            let m = build_module(&ctx, &target, |ctx, m| {
                if direct {
                    lower::populate_direct(ctx, m, 1, &kinds, Box::new(stage))
                } else {
                    lower::populate(ctx, m, &ir)
                }
            })
            .unwrap();
            let text = m.print_to_string().to_string();
            for prefix in ["load", "store"] {
                assert!(
                    text.lines()
                        .any(|line| line.contains(&format!("{prefix} {ty}"))
                            && line.ends_with("align 2")),
                    "{text}"
                );
            }
            assert!(text.contains(&format!("add {ty}")));
            let native = if direct {
                Llvm::build(1, &kinds, Box::new(stage))
            } else {
                Llvm::compile(&ir)
            }
            .unwrap();
            let mut src: Vec<u16> = (0..=lanes as u16).collect();
            src[lanes] = u16::MAX;
            for len in [0, 1, 2, lanes, lanes + 1] {
                let mut dst = vec![333; len + 1];
                let result = native.call(
                    &[1],
                    &mut [
                        BufferArg::from(src.as_slice()),
                        BufferArg::from(&mut dst[..len]),
                    ],
                );
                let mut expected = vec![333; len + 1];
                for i in 1..len {
                    expected[i] = src[i].wrapping_add(1);
                }
                assert_eq!(dst, expected);
                assert_eq!(result.is_ok(), len == lanes + 1);
            }
        }
    }

    #[test]
    fn masked_memory_emits_intrinsics_and_preserves_ordered_errors() {
        use splice::{
            frontend::{self, KernelBuilder},
            hir::{Buffer, Simd, SimdOp},
            simd::Element,
        };
        let kinds = [
            BufferKind::Read,
            BufferKind::Read,
            BufferKind::Read,
            BufferKind::Write,
            BufferKind::Write,
        ];
        fn stage(b: &mut dyn KernelBuilder) -> splice::Result<()> {
            let op = |op| Simd {
                element: Element::I32,
                op,
            };
            let base = b.arg(0);
            let zero = b.constant(0);
            let indices = b.simd(op(SimdOp::Load(Buffer(0))), &[zero])?.unwrap();
            let flags = b.simd(op(SimdOp::Load(Buffer(1))), &[zero])?.unwrap();
            let zeros = b.simd(op(SimdOp::Splat(0)), &[])?.unwrap();
            let mask = b.simd(op(SimdOp::Ne), &[flags, zeros])?.unwrap();
            let lanes = b.simd(op(SimdOp::Iota), &[])?.unwrap();
            let contiguous = b
                .simd(op(SimdOp::LoadSelect(Buffer(2))), &[base, mask, lanes])?
                .unwrap();
            b.simd(
                op(SimdOp::StoreSelect(Buffer(3))),
                &[base, mask, contiguous],
            )?;
            let gathered = b
                .simd(
                    op(SimdOp::GatherSelect(Buffer(2))),
                    &[indices, mask, contiguous],
                )?
                .unwrap();
            let distinct = b.simd(op(SimdOp::Add), &[gathered, lanes])?.unwrap();
            b.simd(
                op(SimdOp::ScatterSelect(Buffer(4))),
                &[indices, mask, distinct],
            )?;
            b.ret(zero);
            Ok(())
        }
        fn reference(
            base: i32,
            indices: &[i32],
            flags: &[i32],
            src: &[i32],
            mid: &mut [i32],
            dst: &mut [i32],
        ) -> Option<i32> {
            let active = || (0..flags.len()).filter(|&i| flags[i] != 0);
            let mut lanes = vec![0; flags.len()];
            for i in active() {
                lanes[i] = *src.get(base as usize + i)?;
            }
            for i in active() {
                *mid.get_mut(base as usize + i)? = lanes[i];
            }
            for i in active() {
                lanes[i] = src.get(indices[i] as usize)?.wrapping_add(i as i32);
            }
            for i in active() {
                *dst.get_mut(indices[i] as usize)? = lanes[i];
            }
            Some(0)
        }

        let ir = frontend::materialize(1, &kinds, Box::new(stage)).unwrap();
        let target = jit::host_machine().unwrap();
        let ctx = Context::create();
        let n = Element::I32.lanes();
        for direct in [false, true] {
            let m = build_module(&ctx, &target, |ctx, m| {
                if direct {
                    lower::populate_direct(ctx, m, 1, &kinds, Box::new(stage))
                } else {
                    lower::populate(ctx, m, &ir)
                }
            })
            .unwrap();
            let text = m.print_to_string().to_string();
            for (name, ptr) in [
                ("load", "p0".to_owned()),
                ("store", "p0".to_owned()),
                ("gather", format!("v{n}p0")),
                ("scatter", format!("v{n}p0")),
            ] {
                let intrinsic = format!("@llvm.masked.{name}.v{n}i32.{ptr}(");
                assert!(
                    text.lines()
                        .any(|line| line.contains("call ") && line.contains(&intrinsic)),
                    "missing call to {intrinsic}"
                );
            }

            let native = if direct {
                Llvm::build(1, &kinds, Box::new(stage))
            } else {
                Llvm::compile(&ir)
            }
            .unwrap();
            for (base, mask, first_indices, src_len, mid_len, dst_len) in [
                // Repeated successful writes, sparse masks, and disabled out-of-bounds addresses.
                (1, 0b0011, [1, 1, i32::MIN, i32::MAX], 3, 3, 3),
                (1, 0b0101, [1, i32::MIN, 1, i32::MAX], 4, 4, 3),
                (-1, 0, [i32::MAX; 4], 0, 0, 0),
                // The invalid scatter lane occurs after two writes to the same address.
                (0, 0b1111, [1, 1, 6, 0], 8, 8, 3),
                // Gather faults after the contiguous writes, before any scatter.
                (0, 0b1111, [1, 1, -1, 0], 8, 8, 3),
                // The third active contiguous store faults; the first two are visible.
                (1, 0b0111, [1, 1, 1, 1], 8, 3, 8),
                // A failing masked load stops execution before the first store.
                (1, 0b0111, [0; 4], 3, 8, 8),
            ] {
                let mut indices = vec![i32::MAX; n];
                indices[..4].copy_from_slice(&first_indices);
                let flags: Vec<_> = (0..n)
                    .map(|i| i32::from(mask & (1_u32 << i) != 0))
                    .collect();
                let src: Vec<_> = (30..30 + src_len).collect();
                let mut mid = vec![-777; mid_len + 2];
                let mut dst = vec![-333; dst_len + 2];
                let (mut want_mid, mut want_dst) = (mid.clone(), dst.clone());
                let expected = reference(
                    base,
                    &indices,
                    &flags,
                    &src,
                    &mut want_mid[1..=mid_len],
                    &mut want_dst[1..=dst_len],
                );
                let actual = native.call(
                    &[base],
                    &mut [
                        BufferArg::Read(&indices),
                        BufferArg::Read(&flags),
                        BufferArg::Read(&src),
                        BufferArg::Write(&mut mid[1..=mid_len]),
                        BufferArg::Write(&mut dst[1..=dst_len]),
                    ],
                );
                assert_eq!(
                    actual,
                    expected.ok_or_else(|| "kernel slice index out of bounds".to_owned())
                );
                assert_eq!(mid, want_mid);
                assert_eq!(dst, want_dst);
            }
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
