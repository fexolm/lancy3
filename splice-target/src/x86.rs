//! x86-64 instruction set extensions and their detection on this machine.

/// x86-64 extensions beyond the SSE2 baseline that code generators use.
/// Every backend translates this one description; none queries the CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct X86Features {
    pub sse3: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub popcnt: bool,
    pub lzcnt: bool,
    pub bmi1: bool,
    pub bmi2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avx512f: bool,
    /// AVX-512 byte and word instructions.
    pub bw: bool,
    /// AVX-512 doubleword and quadword instructions.
    pub dq: bool,
    /// AVX-512 on 128- and 256-bit registers.
    pub vl: bool,
    /// AVX-512 conflict detection.
    pub cd: bool,
    pub vbmi: bool,
    pub vbmi2: bool,
    pub bf16: bool,
    pub fp16: bool,
    /// AVX-512 VPOPCNTD/Q.
    pub vpopcnt: bool,
    /// AVX-512 VPOPCNTB/W and VPSHUFBITQMB.
    pub bitalg: bool,
    /// The AVX10 version, or 0. AVX10 includes every AVX-512 extension above
    /// at each vector width the CPU reports.
    pub avx10: u8,
}

impl X86Features {
    /// x86-64-v3: the extensions every AVX2 CPU has.
    pub const V3: Self = Self {
        sse3: true,
        ssse3: true,
        sse41: true,
        sse42: true,
        popcnt: true,
        lzcnt: true,
        bmi1: true,
        bmi2: true,
        avx: true,
        avx2: true,
        fma: true,
        f16c: true,
        avx512f: false,
        bw: false,
        dq: false,
        vl: false,
        cd: false,
        vbmi: false,
        vbmi2: false,
        bf16: false,
        fp16: false,
        vpopcnt: false,
        bitalg: false,
        avx10: 0,
    };

    /// x86-64-v4: x86-64-v3 with AVX-512 F, BW, DQ, VL and CD.
    pub const V4: Self = Self {
        avx512f: true,
        bw: true,
        dq: true,
        vl: true,
        cd: true,
        ..Self::V3
    };

    /// Every extension, by name, in a fixed order.
    pub fn flags(self) -> [(&'static str, bool); 23] {
        [
            ("sse3", self.sse3),
            ("ssse3", self.ssse3),
            ("sse4.1", self.sse41),
            ("sse4.2", self.sse42),
            ("popcnt", self.popcnt),
            ("lzcnt", self.lzcnt),
            ("bmi1", self.bmi1),
            ("bmi2", self.bmi2),
            ("avx", self.avx),
            ("avx2", self.avx2),
            ("fma", self.fma),
            ("f16c", self.f16c),
            ("avx512f", self.avx512f),
            ("avx512bw", self.bw),
            ("avx512dq", self.dq),
            ("avx512vl", self.vl),
            ("avx512cd", self.cd),
            ("avx512vbmi", self.vbmi),
            ("avx512vbmi2", self.vbmi2),
            ("avx512bf16", self.bf16),
            ("avx512fp16", self.fp16),
            ("avx512vpopcntdq", self.vpopcnt),
            ("avx512bitalg", self.bitalg),
        ]
    }

    /// The first extension `self` uses that `available` lacks.
    pub(crate) fn missing_from(self, available: Self) -> Option<&'static str> {
        if self.avx10 > available.avx10 {
            return Some("avx10");
        }
        let have = available.flags();
        self.flags()
            .into_iter()
            .zip(have)
            .find_map(|((name, used), (_, present))| (used && !present).then_some(name))
    }

    /// The extensions of this CPU that its OS enables. Empty off x86-64.
    pub fn detected() -> Self {
        detected().0
    }
}

#[cfg(any(target_arch = "x86_64", test))]
/// Native vector bytes for this CPU. 32-byte vectors need AVX2: AVX alone has
/// no 256-bit integer operations, which every backend would have to split.
/// `avx10` is CPUID leaf 0x24 EBX: the version and the supported widths.
fn native_bytes(avx2: bool, avx512: bool, avx10: u32) -> usize {
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

/// The host's extensions and widest native vector, read once.
pub(crate) fn detected() -> (X86Features, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        static HOST: std::sync::OnceLock<(X86Features, usize)> = std::sync::OnceLock::new();
        *HOST.get_or_init(detect)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (X86Features::default(), 16)
    }
}

#[cfg(target_arch = "x86_64")]
fn detect() -> (X86Features, usize) {
    use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};
    let avx10 = if __cpuid(0).eax >= 0x24
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
    let converged = avx10 & 0xff >= 1;
    let bytes = native_bytes(
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("avx512f"),
        avx10,
    );
    let features = X86Features {
        sse3: is_x86_feature_detected!("sse3"),
        ssse3: is_x86_feature_detected!("ssse3"),
        sse41: is_x86_feature_detected!("sse4.1"),
        sse42: is_x86_feature_detected!("sse4.2"),
        popcnt: is_x86_feature_detected!("popcnt"),
        lzcnt: is_x86_feature_detected!("lzcnt"),
        bmi1: is_x86_feature_detected!("bmi1"),
        bmi2: is_x86_feature_detected!("bmi2"),
        avx: is_x86_feature_detected!("avx"),
        avx2: is_x86_feature_detected!("avx2") || converged,
        fma: is_x86_feature_detected!("fma"),
        f16c: is_x86_feature_detected!("f16c"),
        avx512f: is_x86_feature_detected!("avx512f") || bytes == 64,
        bw: is_x86_feature_detected!("avx512bw") || converged,
        dq: is_x86_feature_detected!("avx512dq") || converged,
        vl: is_x86_feature_detected!("avx512vl") || converged,
        cd: is_x86_feature_detected!("avx512cd") || converged,
        vbmi: is_x86_feature_detected!("avx512vbmi") || converged,
        vbmi2: is_x86_feature_detected!("avx512vbmi2") || converged,
        bf16: is_x86_feature_detected!("avx512bf16") || converged,
        fp16: is_x86_feature_detected!("avx512fp16") || converged,
        vpopcnt: is_x86_feature_detected!("avx512vpopcntdq") || converged,
        bitalg: is_x86_feature_detected!("avx512bitalg") || converged,
        avx10: avx10 as u8,
    };
    (features, bytes)
}

/// The widest native vector of the AVX10 implementation on this CPU.
pub(crate) fn avx10_bytes() -> Option<usize> {
    let (features, bytes) = detected();
    (features.avx10 != 0).then_some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx10_width_is_independent_of_version() {
        assert_eq!(native_bytes(false, false, 0), 16);
        assert_eq!(native_bytes(true, true, (1 << 16) | 1), 16);
        assert_eq!(native_bytes(true, false, 0), 32);
        assert_eq!(native_bytes(true, true, 0), 64);
        assert_eq!(native_bytes(true, true, (1 << 17) | 2), 32);
        assert_eq!(native_bytes(true, false, (1 << 18) | 1), 64);
    }

    #[test]
    fn missing_extensions_are_named() {
        assert_eq!(X86Features::V3.missing_from(X86Features::V4), None);
        assert_eq!(
            X86Features::V4.missing_from(X86Features::V3),
            Some("avx512f")
        );
        assert_eq!(
            X86Features::V3.missing_from(X86Features::default()),
            Some("sse3")
        );
        let avx10 = X86Features {
            avx10: 1,
            ..X86Features::default()
        };
        assert_eq!(avx10.missing_from(X86Features::V4), Some("avx10"));
    }
}
