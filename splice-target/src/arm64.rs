//! AArch64 vector extensions and the calling thread's SVE vector length.

/// AArch64 extensions beyond the mandatory NEON that code generators use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Arm64Features {
    pub sve: bool,
    /// SVE2. Implies SVE.
    pub sve2: bool,
}

impl Arm64Features {
    pub(crate) fn missing_from(self, available: Self) -> Option<&'static str> {
        if self.sve2 && !available.sve2 {
            Some("sve2")
        } else if self.sve && !available.sve {
            Some("sve")
        } else {
            None
        }
    }

    /// The extensions this CPU and OS provide. Empty off AArch64 Linux.
    pub fn detected() -> Self {
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        {
            Self {
                sve: std::arch::is_aarch64_feature_detected!("sve"),
                sve2: std::arch::is_aarch64_feature_detected!("sve2"),
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
        {
            Self::default()
        }
    }
}

/// The calling thread's SVE vector length in bytes, when the CPU and OS
/// provide SVE. Linux lets each thread change it independently.
pub fn thread_sve_bytes() -> Option<usize> {
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        // SAFETY: Linux HWCAP guarantees SVE in userspace.
        Arm64Features::detected()
            .sve
            .then(|| unsafe { current_sve_bytes() })
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
    {
        None
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
#[target_feature(enable = "sve")]
unsafe fn current_sve_bytes() -> usize {
    let bytes;
    // SAFETY: the caller checks SVE. CNTB reads the calling thread's current
    // vector length and writes only its destination register.
    unsafe {
        std::arch::asm!("cntb {bytes}", bytes = out(reg) bytes, options(nomem, nostack, preserves_flags));
    }
    bytes
}
