//! Compile verified Splice kernels to native code with Cranelift.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(any(
    all(target_arch = "x86_64", any(windows, target_os = "linux")),
    all(target_arch = "aarch64", target_os = "linux")
)))]
compile_error!("splice-cranelift supports x86-64 Windows and Linux, and AArch64 Linux");

mod debug_info;
mod intrinsic;
mod lower;
mod module;

use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, default_libcall_names};
use splice_core::call::{Entry, Interface, SliceDescriptor, enter_native};
use splice_core::{Backend, BuildOptions, CallError, Executable, Function, Result};

/// Compile a specialized kernel or a portable Splice function with Cranelift.
pub struct Cranelift;

/// Cranelift settings for the build target. Every ISA extension comes from
/// the target, so a profile selects the same instructions on every machine.
fn builder(target: &splice_core::Target, debug: bool) -> Result<JITBuilder> {
    use cranelift_codegen::settings::{self, Configurable};

    target.check_execution()?;
    target.check_native_convention()?;
    let mut flags = settings::builder();
    for (name, value) in [
        ("opt_level", if debug { "none" } else { "speed" }),
        (
            "preserve_frame_pointers",
            if debug { "true" } else { "false" },
        ),
        ("enable_probestack", "true"),
        ("probestack_strategy", "inline"),
        // Definitions may be farther apart than a direct AArch64 call.
        ("use_colocated_libcalls", "false"),
        ("is_pic", "false"),
    ] {
        flags
            .set(name, value)
            .map_err(|e| splice_core::BuildError::backend("cranelift", e))?;
    }
    let mut isa = cranelift_native::builder_with_options(false)
        .map_err(splice_core::BuildError::unsupported)?;
    for (flag, enabled) in isa_flags(target) {
        if enabled {
            isa.enable(flag)
                .map_err(|e| splice_core::BuildError::backend("cranelift", e))?;
        }
    }
    let isa = isa
        .finish(settings::Flags::new(flags))
        .map_err(|e| splice_core::BuildError::backend("cranelift", e))?;
    let builder = JITBuilder::with_isa(isa, default_libcall_names());
    // Without SSSE3, Cranelift lowers byte shuffles to this libcall.
    #[cfg(target_arch = "x86_64")]
    let builder = {
        let mut builder = builder;
        builder.symbol("__cranelift_x86_pshufb", x86_pshufb as *const u8);
        builder
    };
    Ok(builder)
}

/// `pshufb` for SSE2 targets: a control byte with its high bit set selects
/// zero, otherwise its low four bits select a source byte. Cranelift passes
/// 128-bit vectors like the C convention of this OS.
#[cfg(target_arch = "x86_64")]
#[expect(
    improper_ctypes_definitions,
    reason = "Cranelift passes I8X16 libcall operands as the C ABI passes __m128i"
)]
extern "C" fn x86_pshufb(
    bytes: std::arch::x86_64::__m128i,
    control: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m128i {
    // SAFETY: both types are 16 plain bytes.
    let (bytes, control) = unsafe {
        (
            std::mem::transmute::<std::arch::x86_64::__m128i, [u8; 16]>(bytes),
            std::mem::transmute::<std::arch::x86_64::__m128i, [u8; 16]>(control),
        )
    };
    let shuffled = control.map(|c| {
        if c & 0x80 == 0 {
            bytes[usize::from(c & 15)]
        } else {
            0
        }
    });
    // SAFETY: as above.
    unsafe { std::mem::transmute::<[u8; 16], std::arch::x86_64::__m128i>(shuffled) }
}

/// The Cranelift ISA settings for a target. Cranelift has no AVX10 or SVE
/// settings; it uses 128-bit vectors on AArch64.
fn isa_flags(target: &splice_core::Target) -> Vec<(&'static str, bool)> {
    let Some(isa) = target.x86() else {
        return Vec::new();
    };
    vec![
        ("has_sse3", isa.sse3),
        ("has_ssse3", isa.ssse3),
        ("has_sse41", isa.sse41),
        ("has_sse42", isa.sse42),
        ("has_popcnt", isa.popcnt),
        ("has_lzcnt", isa.lzcnt),
        ("has_bmi1", isa.bmi1),
        ("has_bmi2", isa.bmi2),
        ("has_avx", isa.avx),
        ("has_avx2", isa.avx2),
        ("has_fma", isa.fma),
        ("has_avx512f", isa.avx512f),
        ("has_avx512vl", isa.vl),
        ("has_avx512dq", isa.dq),
        ("has_avx512bitalg", isa.bitalg),
        ("has_avx512vbmi", isa.vbmi),
    ]
}

struct CodeMemory(Option<JITModule>);

// SAFETY: a shared `CodeMemory` is never used: the module is only touched
// while compiling, before the kernel exists, and by `Drop`, which has
// exclusive access. Calls run the finalized code without the module.
unsafe impl Sync for CodeMemory {}

impl Drop for CodeMemory {
    fn drop(&mut self) {
        if let Some(module) = self.0.take() {
            // SAFETY: No entry pointer escapes its kernel. Drop has exclusive
            // access after the last call, or compilation has failed.
            unsafe { module.free_memory() };
        }
    }
}

/// Owns the executable code of the entry point and all its callees.
pub struct CraneliftKernel {
    // Concurrent calls use the immutable machine code, never the module.
    _memory: splice_jit::Published<CodeMemory>,
    entry: Entry,
    interface: Interface,
}

impl Backend for Cranelift {
    type Output = CraneliftKernel;

    fn compile_function(function: &Function, options: BuildOptions) -> Result<CraneliftKernel> {
        let _span = tracing::debug_span!(target: "splice_core::compile", "compile", backend = "cranelift", input = "portable").entered();
        let debug = options.debug();
        let target = options.target;
        splice_jit::prepare_perf_map();
        let mut memory = CodeMemory(Some(JITModule::new(builder(&target, debug)?)));
        let module = memory.0.as_mut().unwrap();
        let mut kernel = module::KernelModule::lower(module, function, &target, debug)?;
        let id = kernel.entry;
        let sources = std::mem::take(&mut kernel.functions);
        let mut symbols = Vec::new();
        kernel.emit(&mut module.make_context(), debug, |id, ctx| {
            tracing::debug_span!(target: "splice_core::compile", "emit_cranelift_code").in_scope(
                || {
                    module
                        .define_function(id, ctx)
                        .map_err(|e| splice_core::BuildError::backend("cranelift", e))
                },
            )?;
            if let Some(source) = sources.get(&id) {
                symbols.push((id, debug_info::Pending::new(module, ctx, source, debug)?));
            }
            Ok(())
        })?;
        tracing::debug_span!(target: "splice_core::compile", "finalize_cranelift_code").in_scope(
            || {
                module
                    .finalize_definitions()
                    .map_err(|e| splice_core::BuildError::backend("cranelift", e))
            },
        )?;
        let address = module.get_finalized_function(id);
        let publish = debug_info::describe(module, symbols, target.vector_bytes(), debug);
        // SAFETY: KernelModule declares the entry with the host C calling
        // convention, three pointer parameters and an i64 status return
        // (`splice_core::call::Entry`). The kernel owns its module.
        let entry = unsafe { std::mem::transmute::<*const u8, Entry>(address) };
        Ok(CraneliftKernel {
            _memory: publish.finish(debug_info::profile_here(), memory)?,
            entry,
            interface: Interface::new(function.signature()),
        })
    }
}

// SAFETY: `entry` is the code compiled for `interface`, alive as long as
// `_memory`. Every function returns a status (see `Trap::status`), so a
// callee's status forwards through any number of calls.
unsafe impl Executable for CraneliftKernel {
    fn interface(&self) -> &Interface {
        &self.interface
    }

    unsafe fn enter(
        &self,
        words: &[u64],
        slices: &[SliceDescriptor],
    ) -> std::result::Result<u64, CallError> {
        // SAFETY: the caller's contract covers both arrays; the entry writes
        // at most the result word.
        unsafe { enter_native(self.entry, words, slices) }
    }
}

#[cfg(test)]
mod tests {
    use super::isa_flags;
    use splice_core::{Target, target::Profile};

    fn enabled(profile: Profile) -> Vec<&'static str> {
        let target = Target::from_profile(profile).unwrap();
        isa_flags(&target)
            .into_iter()
            .filter_map(|(flag, on)| on.then_some(flag))
            .collect()
    }

    #[test]
    fn fixed_profiles_select_the_same_cranelift_isa_on_every_machine() {
        assert!(enabled(Profile::Sse).is_empty());
        let v3 = [
            "has_sse3",
            "has_ssse3",
            "has_sse41",
            "has_sse42",
            "has_popcnt",
            "has_lzcnt",
            "has_bmi1",
            "has_bmi2",
            "has_avx",
            "has_avx2",
            "has_fma",
        ];
        assert_eq!(enabled(Profile::Avx2), v3);
        let mut v4 = v3.to_vec();
        v4.extend(["has_avx512f", "has_avx512vl", "has_avx512dq"]);
        assert_eq!(enabled(Profile::Avx512), v4);
        assert!(enabled(Profile::Neon).is_empty());
    }
}
