//! LLVM target selection shared by IR lowering and native code generation.
use inkwell::OptimizationLevel;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use splice_target::Profile;
use std::sync::Once;

pub(crate) fn host_machine() -> Result<TargetMachine, String> {
    host_machine_with_debug(false)
}

pub(crate) fn host_machine_with_debug(debug: bool) -> Result<TargetMachine, String> {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let config = InitializationConfig {
            asm_parser: false,
            disassembler: false,
            ..Default::default()
        };
        #[cfg(target_arch = "x86_64")]
        Target::initialize_x86(&config);
        #[cfg(target_arch = "aarch64")]
        Target::initialize_aarch64(&config);
    });
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let simd = splice_target::simd()?;
    // Explicit profiles start from a baseline CPU: an inferred CPU model
    // could re-enable extensions that the selected profile excludes.
    let (cpu, features) = match simd.profile {
        Profile::Auto => (
            TargetMachine::get_host_cpu_name()
                .to_string_lossy()
                .into_owned(),
            TargetMachine::get_host_cpu_features()
                .to_string_lossy()
                .into_owned(),
        ),
        profile => {
            let features = match profile {
                Profile::Sse => "+sse2",
                Profile::Avx2 => "+avx2",
                Profile::Avx512 => "+avx512f",
                Profile::Avx10 => match simd.bytes {
                    32 => "+avx10.1-256",
                    64 => "+avx10.1-512",
                    _ => return Err("LLVM requires at least 256-bit AVX10 support".into()),
                },
                Profile::Neon => "+neon",
                Profile::Sve => "+neon,+sve",
                Profile::Sve2 => "+neon,+sve,+sve2",
                Profile::Auto => unreachable!(),
            };
            let cpu = if cfg!(target_arch = "x86_64") {
                "x86-64"
            } else {
                "generic"
            };
            (cpu.to_owned(), features.to_owned())
        }
    };
    target
        .create_target_machine(
            &triple,
            &cpu,
            &features,
            if debug {
                OptimizationLevel::None
            } else {
                OptimizationLevel::Aggressive
            },
            RelocMode::PIC,
            CodeModel::JITDefault,
        )
        .ok_or_else(|| "LLVM failed to create the host target machine".into())
}

/// Keep target-specific intrinsics in sync with the actual codegen profile.
pub(crate) fn x86_compress(lanes: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let simd = splice_target::simd().expect("JIT validates the SIMD profile before lowering");
        match simd.profile {
            Profile::Auto => {
                is_x86_feature_detected!("avx512f")
                    && (lanes == 16 || is_x86_feature_detected!("avx512vl"))
            }
            Profile::Avx512 | Profile::Avx10 => true,
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = lanes;
        false
    }
}
