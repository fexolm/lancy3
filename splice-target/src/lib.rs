//! SIMD profiles and native vector widths shared by the frontend and JITs.
pub mod abi;
mod profile;

pub use profile::{Profile, Simd, TargetError, simd};

#[cfg(target_arch = "x86_64")]
use std::{
    arch::x86_64::{__cpuid, __cpuid_count, _xgetbv},
    sync::OnceLock,
};

#[derive(Copy, Clone, Debug, Default)]
pub struct Features {
    pub bytes: usize,
    pub avx2: bool,
    pub avx10: u8,
    pub bw: bool,
    pub dq: bool,
    pub sse41: bool,
    pub ssse3: bool,
    pub sse42: bool,
    pub vbmi2: bool,
    pub vbmi: bool,
    pub f16c: bool,
    pub bf16: bool,
    pub fp16: bool,
    pub vpopcnt: bool,
    pub bitalg: bool,
    pub cd: bool,
    pub fma: bool,
}

#[cfg(any(target_arch = "x86_64", test))]
/// Native vector bytes. 32-byte vectors need AVX2: AVX alone has no 256-bit
/// integer operations, which every backend would have to split.
fn width_for_features(avx2: bool, avx512: bool, avx10: u32) -> usize {
    if avx10 as u8 != 0 {
        if avx10 & (1 << 18) != 0 {
            64
        } else if avx10 & (1 << 17) != 0 {
            32
        } else {
            16
        }
    } else if avx512 {
        64
    } else if avx2 {
        32
    } else {
        16
    }
}

impl Features {
    #[cfg(target_arch = "x86_64")]
    pub fn host() -> Self {
        static SELECTED: OnceLock<Features> = OnceLock::new();
        *SELECTED.get_or_init(Self::selected)
    }

    #[cfg(target_arch = "x86_64")]
    fn selected() -> Self {
        let selected = simd().expect("invalid SIMD configuration");
        match selected.profile {
            Profile::Sse => Self {
                bytes: 16,
                ..Self::default()
            },
            Profile::Avx2 => {
                let cpu = Self::detected();
                Self {
                    bytes: 32,
                    avx2: true,
                    ssse3: cpu.ssse3,
                    sse41: cpu.sse41,
                    sse42: cpu.sse42,
                    f16c: cpu.f16c,
                    fma: cpu.fma,
                    ..Self::default()
                }
            }
            Profile::Avx512 => Self::x86_features(0),
            _ => Self::detected(),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detected() -> Self {
        static HOST: OnceLock<Features> = OnceLock::new();
        *HOST.get_or_init(|| {
            let max = __cpuid(0).eax;
            let avx10 = if max >= 0x24
                && __cpuid_count(7, 0).eax >= 1
                && __cpuid_count(7, 1).edx & (1 << 19) != 0
                && __cpuid(1).ecx & (1 << 27) != 0
            {
                // SAFETY: OSXSAVE above permits XGETBV. EVEX also needs the OS to save opmask and ZMM state.
                if unsafe { _xgetbv(0) } & 0xe6 == 0xe6 {
                    __cpuid_count(0x24, 0).ebx
                } else {
                    0
                }
            } else {
                0
            };
            Self::x86_features(avx10)
        })
    }

    #[cfg(target_arch = "x86_64")]
    fn x86_features(avx10: u32) -> Self {
        let converged = avx10 & 0xff >= 1;
        Self {
            bytes: width_for_features(
                is_x86_feature_detected!("avx2"),
                is_x86_feature_detected!("avx512f"),
                avx10,
            ),
            avx10: avx10 as u8,
            avx2: is_x86_feature_detected!("avx2") || converged,
            bw: is_x86_feature_detected!("avx512bw") || converged,
            dq: is_x86_feature_detected!("avx512dq") || converged,
            fp16: is_x86_feature_detected!("avx512fp16") || converged,
            ssse3: is_x86_feature_detected!("ssse3"),
            sse42: is_x86_feature_detected!("sse4.2"),
            vbmi: is_x86_feature_detected!("avx512vbmi") || converged,
            f16c: is_x86_feature_detected!("f16c"),
            bf16: is_x86_feature_detected!("avx512bf16") || converged,
            vbmi2: is_x86_feature_detected!("avx512vbmi2") || converged,
            sse41: is_x86_feature_detected!("sse4.1"),
            vpopcnt: is_x86_feature_detected!("avx512vpopcntdq") || converged,
            bitalg: is_x86_feature_detected!("avx512bitalg") || converged,
            cd: is_x86_feature_detected!("avx512cd") || converged,
            fma: is_x86_feature_detected!("fma"),
        }
    }
}

/// The portable frontend and the ordinary Rust implementation can also run on non-x64 hosts.
pub fn vector_bytes() -> usize {
    simd().expect("invalid SIMD configuration").bytes
}

#[cfg(test)]
#[test]
fn avx10_width_is_independent_of_version() {
    assert_eq!(width_for_features(false, false, 0), 16);
    assert_eq!(width_for_features(true, true, (1 << 16) | 1), 16);
    assert_eq!(width_for_features(true, false, 0), 32);
    assert_eq!(width_for_features(true, true, 0), 64);
    assert_eq!(width_for_features(true, true, (1 << 17) | 2), 32);
    assert_eq!(width_for_features(true, false, (1 << 18) | 1), 64);
}
