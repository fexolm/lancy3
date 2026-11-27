//! LLVM target selection shared by IR lowering and native code generation.
//! Every ISA extension comes from the build target, so a profile selects the
//! same instructions on every machine.
use inkwell::OptimizationLevel;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, TargetMachine};
use splice_core::Target;
use std::sync::Once;

/// What lowering needs from the build target.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Machine {
    pub(crate) vector_bytes: usize,
    /// AVX-512 compress and expand instructions at this vector width.
    pub(crate) x86_compress: bool,
    /// See [`scalable`].
    pub(crate) scalable: bool,
    /// A fused multiply-add instruction: AArch64, or x86 with FMA.
    pub(crate) fma: bool,
}

impl Machine {
    pub(crate) fn new(target: &Target) -> Self {
        let bytes = target.vector_bytes();
        Self {
            vector_bytes: bytes,
            x86_compress: target
                .x86()
                .is_some_and(|isa| isa.avx10 > 0 || isa.avx512f && (bytes == 64 || isa.vl)),
            scalable: scalable(target),
            fma: target.x86().is_none_or(|isa| isa.fma),
        }
    }
}

/// Whether SVE code generation is sound on the calling thread. LLVM's
/// AArch64 backend assumes vscale is a power of two, so a thread with, say,
/// a 48-byte vector length gets fixed-length NEON code for the target width.
pub(crate) fn scalable(target: &Target) -> bool {
    target.sve() && splice_target::thread_sve_bytes().is_some_and(usize::is_power_of_two)
}

/// The LLVM CPU and feature string. A baseline CPU keeps an inferred model
/// from enabling extensions that the target excludes.
pub(crate) fn cpu_features(target: &Target) -> (&'static str, String) {
    let mut features = Vec::new();
    if let Some(isa) = target.x86() {
        features.push("+sse2".to_owned());
        for (name, enabled) in isa.flags() {
            // LLVM names BMI1 `bmi`.
            let name = if name == "bmi1" { "bmi" } else { name };
            features.push(format!("{}{name}", if enabled { '+' } else { '-' }));
        }
        if isa.avx10 > 0 {
            // LLVM 19 knows AVX10.1; later versions include its instructions.
            features.push("+avx10.1-256".to_owned());
            if target.vector_bytes() == 64 {
                features.push("+avx10.1-512".to_owned());
            }
        }
        return ("x86-64", features.join(","));
    }
    let isa = target.arm64().expect("a target is x86-64 or AArch64");
    features.push("+neon".to_owned());
    if scalable(target) {
        features.push("+sve".to_owned());
        if isa.sve2 {
            features.push("+sve2".to_owned());
        }
    }
    ("generic", features.join(","))
}

/// A target machine for `target` on this machine's architecture. Listing
/// the IR needs no CPU support; [`crate::jit`] checks execution.
pub(crate) fn machine(target: &Target, debug: bool) -> splice_core::Result<TargetMachine> {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let config = InitializationConfig {
            asm_parser: false,
            disassembler: false,
            ..Default::default()
        };
        #[cfg(target_arch = "x86_64")]
        inkwell::targets::Target::initialize_x86(&config);
        #[cfg(target_arch = "aarch64")]
        inkwell::targets::Target::initialize_aarch64(&config);
    });
    if splice_target::Arch::host() != Some(target.arch()) {
        return Err(splice_core::TargetError::Foreign(target.arch()).into());
    }
    let triple = TargetMachine::get_default_triple();
    let llvm = inkwell::targets::Target::from_triple(&triple).map_err(crate::error)?;
    let (cpu, features) = cpu_features(target);
    llvm.create_target_machine(
        &triple,
        cpu,
        &features,
        if debug {
            OptimizationLevel::None
        } else {
            OptimizationLevel::Aggressive
        },
        RelocMode::PIC,
        CodeModel::JITDefault,
    )
    .ok_or_else(|| crate::error("LLVM failed to create the target machine"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use splice_target::Profile;

    #[test]
    fn fixed_profiles_select_the_same_llvm_features_on_every_machine() {
        let sse = Target::from_profile(Profile::Sse).unwrap();
        assert_eq!(
            cpu_features(&sse),
            (
                "x86-64",
                "+sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-lzcnt,-bmi,-bmi2,-avx,-avx2,\
                 -fma,-f16c,-avx512f,-avx512bw,-avx512dq,-avx512vl,-avx512cd,-avx512vbmi,\
                 -avx512vbmi2,-avx512bf16,-avx512fp16,-avx512vpopcntdq,-avx512bitalg"
                    .to_owned()
            )
        );
        let avx2 = Target::from_profile(Profile::Avx2).unwrap();
        assert_eq!(
            cpu_features(&avx2).1,
            "+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+lzcnt,+bmi,+bmi2,+avx,+avx2,\
             +fma,+f16c,-avx512f,-avx512bw,-avx512dq,-avx512vl,-avx512cd,-avx512vbmi,\
             -avx512vbmi2,-avx512bf16,-avx512fp16,-avx512vpopcntdq,-avx512bitalg"
        );
        let avx512 = Target::from_profile(Profile::Avx512).unwrap();
        assert_eq!(
            cpu_features(&avx512).1,
            "+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+lzcnt,+bmi,+bmi2,+avx,+avx2,\
             +fma,+f16c,+avx512f,+avx512bw,+avx512dq,+avx512vl,+avx512cd,-avx512vbmi,\
             -avx512vbmi2,-avx512bf16,-avx512fp16,-avx512vpopcntdq,-avx512bitalg"
        );
        let neon = Target::from_profile(Profile::Neon).unwrap();
        assert_eq!(cpu_features(&neon), ("generic", "+neon".to_owned()));
        assert!(!Machine::new(&avx2).x86_compress);
        assert!(Machine::new(&avx512).x86_compress);
    }
}
