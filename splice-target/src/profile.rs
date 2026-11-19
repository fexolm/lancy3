use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Auto,
    Sse,
    Avx2,
    Avx512,
    Avx10,
    Neon,
    Sve,
    Sve2,
}

#[derive(Clone, Copy, Debug)]
pub struct Simd {
    pub profile: Profile,
    pub bytes: usize,
}

/// Read SPLICE_SIMD once so that staging, listings and executable kernels
/// agree on the logical vector width throughout the process.
pub fn simd() -> Result<Simd, String> {
    static CONFIG: OnceLock<Result<Simd, String>> = OnceLock::new();
    CONFIG
        .get_or_init(select)
        .as_ref()
        .copied()
        .map_err(Clone::clone)
}

fn select() -> Result<Simd, String> {
    use Profile::*;
    let name = match std::env::var("SPLICE_SIMD") {
        Ok(name) => name,
        Err(std::env::VarError::NotPresent) => String::from("auto"),
        Err(error) => return Err(format!("invalid SPLICE_SIMD: {error}")),
    };
    let profile = match name.as_str() {
        "auto" => Auto,
        "sse" => Sse,
        "avx2" => Avx2,
        "avx512" => Avx512,
        "avx10" => Avx10,
        "neon" => Neon,
        "sve" => Sve,
        "sve2" => Sve2,
        _ => return Err(format!("unknown SPLICE_SIMD profile {name:?}")),
    };
    // On Linux AArch64 Auto picks the best ISA exposed to userspace.
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    let profile = if profile == Auto {
        if width(Sve2).is_some() {
            Sve2
        } else if width(Sve).is_some() {
            Sve
        } else {
            Neon
        }
    } else {
        profile
    };
    let bytes = width(profile)
        .ok_or_else(|| format!("SPLICE_SIMD={name} is unavailable on this CPU/OS"))?;
    Ok(Simd { profile, bytes })
}

fn width(profile: Profile) -> Option<usize> {
    use Profile::*;
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = super::Features::detected();
        match profile {
            Auto => Some(cpu.bytes),
            Sse => Some(16),
            Avx2 if is_x86_feature_detected!("avx2") => Some(32),
            Avx512 if is_x86_feature_detected!("avx512f") => Some(64),
            Avx10 if cpu.avx10 != 0 => Some(cpu.bytes),
            _ => None,
        }
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    match profile {
        Neon if std::arch::is_aarch64_feature_detected!("neon") => Some(16),
        Sve if std::arch::is_aarch64_feature_detected!("sve") => sve_bytes(),
        Sve2 if std::arch::is_aarch64_feature_detected!("sve2") => sve_bytes(),
        _ => None,
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_os = "linux")
    )))]
    {
        (profile == Auto).then_some(16)
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn thread_sve_bytes() -> Option<usize> {
    // Linux uapi/linux/prctl.h.
    const PR_SVE_GET_VL: libc::c_int = 51;
    const PR_SVE_VL_LEN_MASK: libc::c_int = 0xffff;
    // SAFETY: PR_SVE_GET_VL takes no pointers and the four unused arguments
    // are supplied as unsigned long, as required by the Linux prctl ABI.
    let vl = unsafe { libc::prctl(PR_SVE_GET_VL, 0usize, 0usize, 0usize, 0usize) };
    (vl >= 0).then_some((vl & PR_SVE_VL_LEN_MASK) as usize)
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn sve_bytes() -> Option<usize> {
    let vl = thread_sve_bytes()?;
    // Lane storage and alignment support power-of-two widths through 64 bytes.
    [64, 32, 16].into_iter().find(|&bytes| bytes <= vl)
}

impl Simd {
    /// Verify the current thread can execute code generated for this width.
    /// Linux lets each thread change its SVE length independently.
    pub fn check_execution(self) -> Result<(), String> {
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        if matches!(self.profile, Profile::Sve | Profile::Sve2)
            && thread_sve_bytes().is_none_or(|vl| vl < self.bytes)
        {
            return Err(format!(
                "kernel requires an SVE vector length of at least {} bytes",
                self.bytes
            ));
        }
        Ok(())
    }
}
