//! Code generation targets as values: instruction set extensions, vector
//! width and calling convention. Every backend translates a [`Target`]; none
//! queries the CPU or the environment itself.
pub mod abi;
mod arm64;
mod profile;
mod x86;

pub use arm64::{Arm64Features, thread_sve_bytes};
pub use profile::{Profile, TargetError};
pub use x86::X86Features;

/// The narrowest vector any target uses.
pub const MIN_VECTOR_BYTES: usize = 16;
/// The widest vector any target uses. Lane storage and masks are sized for it.
pub const MAX_VECTOR_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    X86_64,
    Aarch64,
}

impl Arch {
    /// The architecture of this process, if code generation supports it.
    pub const fn host() -> Option<Self> {
        if cfg!(target_arch = "x86_64") {
            Some(Self::X86_64)
        } else if cfg!(target_arch = "aarch64") {
            Some(Self::Aarch64)
        } else {
            None
        }
    }
}

/// How generated code receives arguments and which registers it preserves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallingConvention {
    /// x86-64 System V (Linux).
    SysV,
    /// x86-64 Windows.
    Win64,
    /// AArch64 procedure call standard.
    Aapcs64,
}

impl CallingConvention {
    pub const fn arch(self) -> Arch {
        match self {
            Self::SysV | Self::Win64 => Arch::X86_64,
            Self::Aapcs64 => Arch::Aarch64,
        }
    }

    /// The convention of `extern "C"` on this OS for `arch`.
    pub const fn native(arch: Arch) -> Self {
        match arch {
            Arch::X86_64 if cfg!(windows) => Self::Win64,
            Arch::X86_64 => Self::SysV,
            Arch::Aarch64 => Self::Aapcs64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Features {
    X86(X86Features),
    Arm64(Arm64Features),
}

/// What generated code may assume about the machine that runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    vector_bytes: usize,
    features: Features,
    convention: CallingConvention,
}

impl Target {
    /// x86-64 code using `features` on `vector_bytes`-byte vectors: 16, 32
    /// with AVX2, or 64 with AVX-512.
    pub fn x86_64(features: X86Features, vector_bytes: usize) -> Result<Self, TargetError> {
        let valid = match vector_bytes {
            16 => true,
            32 => features.avx2,
            64 => features.avx512f,
            _ => false,
        };
        if !valid {
            return Err(TargetError::VectorBytes(vector_bytes));
        }
        Ok(Self {
            vector_bytes,
            features: Features::X86(features),
            convention: CallingConvention::native(Arch::X86_64),
        })
    }

    /// AArch64 code on `vector_bytes`-byte vectors: 16 for NEON, or a power
    /// of two up to [`MAX_VECTOR_BYTES`] for SVE.
    pub fn aarch64(features: Arm64Features, vector_bytes: usize) -> Result<Self, TargetError> {
        let features = Arm64Features {
            sve: features.sve || features.sve2,
            ..features
        };
        let valid = match vector_bytes {
            16 => true,
            32 | 64 => features.sve,
            _ => false,
        };
        if !valid {
            return Err(TargetError::VectorBytes(vector_bytes));
        }
        Ok(Self {
            vector_bytes,
            features: Features::Arm64(features),
            convention: CallingConvention::Aapcs64,
        })
    }

    /// The target `profile` names. `Sse`, `Avx2`, `Avx512` and `Neon` are
    /// the same on every machine. `Auto` and `Avx10` describe this CPU, and
    /// the SVE profiles use the calling thread's vector length.
    pub fn from_profile(profile: Profile) -> Result<Self, TargetError> {
        let unavailable = || TargetError::Unavailable(profile);
        match profile {
            Profile::Sse => Self::x86_64(X86Features::default(), 16),
            Profile::Avx2 => Self::x86_64(X86Features::V3, 32),
            Profile::Avx512 => Self::x86_64(X86Features::V4, 64),
            Profile::Neon => Self::aarch64(Arm64Features::default(), 16),
            Profile::Avx10 => {
                let bytes = x86::avx10_bytes().ok_or_else(unavailable)?;
                Self::x86_64(X86Features::detected(), bytes)
            }
            Profile::Sve | Profile::Sve2 => {
                let host = Arm64Features::detected();
                let sve2 = profile == Profile::Sve2;
                if !host.sve || sve2 && !host.sve2 {
                    return Err(unavailable());
                }
                let length = thread_sve_bytes().ok_or_else(unavailable)?;
                // Lane storage and alignment support power-of-two widths.
                let bytes = [64, 32, 16]
                    .into_iter()
                    .find(|&bytes| bytes <= length)
                    .ok_or_else(unavailable)?;
                Self::aarch64(Arm64Features { sve: true, sve2 }, bytes)
            }
            Profile::Auto => match Arch::host().ok_or_else(unavailable)? {
                Arch::X86_64 => {
                    let (features, bytes) = x86::detected();
                    Self::x86_64(features, bytes)
                }
                Arch::Aarch64 => [Profile::Sve2, Profile::Sve]
                    .into_iter()
                    .find_map(|profile| Self::from_profile(profile).ok())
                    .map_or_else(|| Self::from_profile(Profile::Neon), Ok),
            },
        }
    }

    /// The default target: the profile `SPLICE_SIMD` names, or `auto`, if
    /// this machine runs it. The variable is read once per process; pass a
    /// target explicitly to build for any other.
    pub fn host() -> Result<Self, TargetError> {
        static HOST: std::sync::OnceLock<Result<Target, TargetError>> = std::sync::OnceLock::new();
        HOST.get_or_init(|| {
            let target = Self::from_profile(Profile::from_env()?)?;
            target.check_execution()?;
            Ok(target)
        })
        .clone()
    }

    /// The same code with another calling convention of its architecture.
    pub fn with_calling_convention(
        self,
        convention: CallingConvention,
    ) -> Result<Self, TargetError> {
        if convention.arch() != self.arch() {
            return Err(TargetError::CallingConvention(convention));
        }
        Ok(Self { convention, ..self })
    }

    pub fn arch(&self) -> Arch {
        match self.features {
            Features::X86(_) => Arch::X86_64,
            Features::Arm64(_) => Arch::Aarch64,
        }
    }

    /// Bytes in one vector register value.
    pub fn vector_bytes(&self) -> usize {
        self.vector_bytes
    }

    pub fn features(&self) -> Features {
        self.features
    }

    pub fn x86(&self) -> Option<X86Features> {
        match self.features {
            Features::X86(features) => Some(features),
            Features::Arm64(_) => None,
        }
    }

    pub fn arm64(&self) -> Option<Arm64Features> {
        match self.features {
            Features::Arm64(features) => Some(features),
            Features::X86(_) => None,
        }
    }

    pub fn calling_convention(&self) -> CallingConvention {
        self.convention
    }

    /// Whether this AArch64 target uses SVE.
    pub fn sve(&self) -> bool {
        self.arm64().is_some_and(|features| features.sve)
    }

    /// Verify the calling thread can execute code generated for this
    /// target. Both x86-64 calling conventions are callable on either OS.
    /// Linux lets each thread change its SVE length independently.
    pub fn check_execution(&self) -> Result<(), TargetError> {
        if Arch::host() != Some(self.arch()) {
            return Err(TargetError::Foreign(self.arch()));
        }
        let missing = match self.features {
            Features::X86(features) => features.missing_from(X86Features::detected()),
            Features::Arm64(features) => features.missing_from(Arm64Features::detected()),
        };
        if let Some(feature) = missing {
            return Err(TargetError::Missing(feature));
        }
        if self.sve() {
            let available = thread_sve_bytes().ok_or(TargetError::Missing("sve"))?;
            if available < self.vector_bytes {
                return Err(TargetError::VectorLength {
                    required: self.vector_bytes,
                    available,
                });
            }
        }
        Ok(())
    }
}

/// Ф3 bridge for crates that still read the process profile; removed once
/// the splice backends take their target from `BuildOptions`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct Simd {
    pub profile: Profile,
    pub bytes: usize,
}

#[doc(hidden)]
pub fn simd() -> Result<Simd, TargetError> {
    let target = Target::host()?;
    let profile = match (Profile::from_env()?, target.arm64()) {
        (Profile::Auto, Some(features)) if features.sve2 => Profile::Sve2,
        (Profile::Auto, Some(features)) if features.sve => Profile::Sve,
        (Profile::Auto, Some(_)) => Profile::Neon,
        (profile, _) => profile,
    };
    Ok(Simd {
        profile,
        bytes: target.vector_bytes(),
    })
}

#[doc(hidden)]
pub fn vector_bytes() -> usize {
    Target::host()
        .expect("invalid SIMD configuration")
        .vector_bytes()
}

#[doc(hidden)]
impl Simd {
    pub fn sve_vector_bytes(self) -> Option<usize> {
        matches!(self.profile, Profile::Sve | Profile::Sve2)
            .then(thread_sve_bytes)
            .flatten()
    }
    pub fn check_execution(self) -> Result<(), TargetError> {
        Target::host()?.check_execution()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_profiles_describe_the_same_target_on_every_machine() {
        for (profile, bytes, features) in [
            (Profile::Sse, 16, X86Features::default()),
            (Profile::Avx2, 32, X86Features::V3),
            (Profile::Avx512, 64, X86Features::V4),
        ] {
            let target = Target::from_profile(profile).unwrap();
            assert_eq!(target.vector_bytes(), bytes, "{profile:?}");
            assert_eq!(target.x86(), Some(features), "{profile:?}");
            assert_eq!(
                target.calling_convention(),
                CallingConvention::native(Arch::X86_64)
            );
        }
        let neon = Target::from_profile(Profile::Neon).unwrap();
        assert_eq!(neon.vector_bytes(), 16);
        assert_eq!(neon.arm64(), Some(Arm64Features::default()));
        assert_eq!(neon.calling_convention(), CallingConvention::Aapcs64);
    }

    #[test]
    fn every_profile_is_a_valid_target_or_unavailable_here() {
        for profile in Profile::ALL {
            match Target::from_profile(profile) {
                Ok(target) => {
                    let bytes = target.vector_bytes();
                    assert!(
                        (MIN_VECTOR_BYTES..=MAX_VECTOR_BYTES).contains(&bytes)
                            && bytes.is_power_of_two(),
                        "{profile:?}: {bytes}"
                    );
                    if matches!(profile, Profile::Auto | Profile::Avx10) {
                        target.check_execution().unwrap();
                    }
                }
                Err(error) => assert_eq!(error, TargetError::Unavailable(profile)),
            }
            assert_eq!(Profile::from_name(profile.name()), Ok(profile));
        }
        assert_eq!(
            Profile::from_name("mmx"),
            Err(TargetError::UnknownProfile("mmx".into()))
        );
    }

    #[test]
    fn vector_width_needs_the_extensions_that_operate_on_it() {
        let avx = X86Features {
            avx: true,
            ..X86Features::default()
        };
        assert_eq!(Target::x86_64(avx, 32), Err(TargetError::VectorBytes(32)));
        assert_eq!(
            Target::x86_64(X86Features::V3, 64),
            Err(TargetError::VectorBytes(64))
        );
        assert_eq!(
            Target::x86_64(X86Features::V4, 8),
            Err(TargetError::VectorBytes(8))
        );
        assert_eq!(
            Target::aarch64(Arm64Features::default(), 32),
            Err(TargetError::VectorBytes(32))
        );
        let sve2 = Arm64Features {
            sve: false,
            sve2: true,
        };
        assert!(Target::aarch64(sve2, 64).unwrap().sve());
    }

    #[test]
    fn calling_conventions_belong_to_one_architecture() {
        let sse = Target::from_profile(Profile::Sse).unwrap();
        let win64 = sse
            .with_calling_convention(CallingConvention::Win64)
            .unwrap();
        assert_eq!(win64.calling_convention(), CallingConvention::Win64);
        assert_eq!(
            sse.with_calling_convention(CallingConvention::Aapcs64),
            Err(TargetError::CallingConvention(CallingConvention::Aapcs64))
        );
    }

    #[test]
    fn foreign_code_and_missing_extensions_do_not_run() {
        let (native, foreign) = if cfg!(target_arch = "aarch64") {
            (Profile::Neon, Profile::Sse)
        } else {
            (Profile::Sse, Profile::Neon)
        };
        Target::from_profile(native)
            .unwrap()
            .check_execution()
            .unwrap();
        let foreign = Target::from_profile(foreign).unwrap();
        assert_eq!(
            foreign.check_execution(),
            Err(TargetError::Foreign(foreign.arch()))
        );
        #[cfg(target_arch = "x86_64")]
        if X86Features::V4
            .missing_from(X86Features::detected())
            .is_some()
        {
            let avx512 = Target::from_profile(Profile::Avx512).unwrap();
            assert!(matches!(
                avx512.check_execution(),
                Err(TargetError::Missing(_))
            ));
        }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    #[test]
    fn sve_targets_need_the_thread_vector_length() {
        let sve = Target::aarch64(
            Arm64Features {
                sve: true,
                sve2: false,
            },
            16,
        )
        .unwrap();
        assert_eq!(sve.check_execution().is_ok(), Arm64Features::detected().sve);
    }
}
