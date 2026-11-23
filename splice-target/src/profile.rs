use crate::{Arch, CallingConvention};
use std::fmt;

/// A named instruction set level. [`crate::Target::from_profile`] turns it
/// into a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// The best vector extensions of this machine.
    Auto,
    /// x86-64 SSE2 baseline, 16-byte vectors.
    Sse,
    /// x86-64-v3, 32-byte vectors.
    Avx2,
    /// x86-64-v4, 64-byte vectors.
    Avx512,
    /// This CPU's AVX10 at its widest vector.
    Avx10,
    /// AArch64 NEON, 16-byte vectors.
    Neon,
    /// SVE at the calling thread's vector length.
    Sve,
    /// SVE2 at the calling thread's vector length.
    Sve2,
}

impl Profile {
    pub const ALL: [Self; 8] = [
        Self::Auto,
        Self::Sse,
        Self::Avx2,
        Self::Avx512,
        Self::Avx10,
        Self::Neon,
        Self::Sve,
        Self::Sve2,
    ];

    /// The `SPLICE_SIMD` spelling.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Sse => "sse",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Avx10 => "avx10",
            Self::Neon => "neon",
            Self::Sve => "sve",
            Self::Sve2 => "sve2",
        }
    }

    pub fn from_name(name: &str) -> Result<Self, TargetError> {
        Self::ALL
            .into_iter()
            .find(|profile| profile.name() == name)
            .ok_or_else(|| TargetError::UnknownProfile(name.to_owned()))
    }

    /// The profile `SPLICE_SIMD` names, or `Auto` when it is unset.
    pub fn from_env() -> Result<Self, TargetError> {
        match std::env::var("SPLICE_SIMD") {
            Ok(name) => Self::from_name(&name),
            Err(std::env::VarError::NotPresent) => Ok(Self::Auto),
            Err(std::env::VarError::NotUnicode(_)) => Err(TargetError::InvalidVariable),
        }
    }
}

/// A target cannot be described or cannot run here.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetError {
    /// `SPLICE_SIMD` is not valid Unicode.
    InvalidVariable,
    /// `SPLICE_SIMD` names no profile.
    UnknownProfile(String),
    /// The CPU or OS does not provide this profile.
    Unavailable(Profile),
    /// The CPU or OS lacks an extension the target uses.
    Missing(&'static str),
    /// Code for this architecture cannot run on this machine.
    Foreign(Arch),
    /// The extensions cannot operate on vectors of this many bytes.
    VectorBytes(usize),
    /// The architecture has no such calling convention.
    CallingConvention(CallingConvention),
    /// Code with this calling convention cannot call Rust functions of this
    /// process, which use the OS's C convention.
    HostCalls(CallingConvention),
    /// The calling thread's SVE vector length is shorter than the code needs.
    VectorLength { required: usize, available: usize },
    /// The code was generated for scalable vectors, which need a power-of-two
    /// SVE vector length on the calling thread.
    VectorLengthNotPowerOfTwo,
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVariable => f.write_str("SPLICE_SIMD is not valid Unicode"),
            Self::UnknownProfile(name) => write!(f, "unknown SPLICE_SIMD profile {name:?}"),
            Self::Unavailable(profile) => write!(f, "{profile:?} is unavailable on this CPU/OS"),
            Self::Missing(feature) => write!(f, "this CPU/OS lacks {feature}"),
            Self::Foreign(arch) => write!(f, "{arch:?} code cannot run on this machine"),
            Self::VectorBytes(bytes) => {
                write!(f, "the target's extensions have no {bytes}-byte vectors")
            }
            Self::CallingConvention(convention) => {
                write!(
                    f,
                    "{convention:?} is not a calling convention of this architecture"
                )
            }
            Self::HostCalls(convention) => write!(
                f,
                "{convention:?} code cannot call this process's Rust functions"
            ),
            Self::VectorLength {
                required,
                available,
            } => write!(
                f,
                "kernel requires an SVE vector length of at least {required} bytes, the thread has {available}"
            ),
            Self::VectorLengthNotPowerOfTwo => {
                f.write_str("scalable code requires a power-of-two SVE vector length")
            }
        }
    }
}

impl std::error::Error for TargetError {}
