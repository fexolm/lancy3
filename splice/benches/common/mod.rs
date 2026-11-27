//! Shared benchmark inputs.

/// Marsaglia's xorshift32: a fixed seed gives every backend the same data.
pub(crate) fn xorshift(mut seed: u32) -> impl Iterator<Item = u32> {
    std::iter::repeat_with(move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    })
}
