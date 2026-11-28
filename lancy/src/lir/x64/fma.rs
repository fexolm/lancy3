//! Fused four-lane f32 arithmetic for x86 targets without native FMA.

type Vector = [f32; 4];
pub(super) const SLOT_BYTES: usize = size_of::<Vector>();

pub(super) fn entry(win64: bool) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if win64 {
            win64_fma as *const () as u64
        } else {
            sysv_fma as *const () as u64
        }
    }
    // Cross emission can list the instructions but cannot enter x86 code.
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = win64;
        compute as *const () as u64
    }
}

#[cfg(target_arch = "x86_64")]
unsafe extern "win64" fn win64_fma(area: *mut Vector) {
    // SAFETY: the generated caller supplies the slots described by compute.
    unsafe { compute(area) }
}

#[cfg(target_arch = "x86_64")]
unsafe extern "sysv64" fn sysv_fma(area: *mut Vector) {
    // SAFETY: the generated caller supplies the slots described by compute.
    unsafe { compute(area) }
}

/// The emitter reserves four slots in the caller's live frame, initializes
/// the first three with the vector arguments and reads the fourth afterwards.
unsafe fn compute(area: *mut Vector) {
    // SAFETY: the generated caller provides three initialized readable slots
    // followed by an exclusive writable result slot, each SLOT_BYTES wide.
    unsafe {
        let a = area.read_unaligned();
        let b = area.add(1).read_unaligned();
        let c = area.add(2).read_unaligned();
        area.add(3)
            .write_unaligned(std::array::from_fn(|i| fused(a[i], b[i], c[i])));
    }
}

#[inline]
fn fused(a: f32, b: f32, c: f32) -> f32 {
    // The product of two finite f32 operands fits exactly in f64. A
    // subsequent f64 addition can change a normally rounded f32 result
    // only when it lands exactly on a midpoint between adjacent f32s.
    let wide = f64::from(a) * f64::from(b) + f64::from(c);
    let magnitude = wide.to_bits() & (u64::MAX >> 1);
    const MIN: u64 = (f32::MIN_POSITIVE as f64).to_bits();
    const MAX: u64 = (f32::MAX as f64).to_bits();
    const EXTRA: u32 = f64::MANTISSA_DIGITS - f32::MANTISSA_DIGITS;
    if (MIN..=MAX).contains(&magnitude) && magnitude & ((1 << EXTRA) - 1) != 1 << (EXTRA - 1) {
        wide as f32
    } else {
        a.mul_add(b, c)
    }
}
