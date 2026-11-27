// The declarations of `splice::simd::Lane` and `Mask`, the single source of
// their kernel operations. `splice_core::simd` expands this file (the
// `__simd_api!` macro) and the kernel macro types every SIMD call from its
// signatures (`scalar::lower::simd::declared`).
//
// Each method is an intrinsic without a body: a kernel stages it as
// portable SIMD operations (`splice_core::frontend::methods::simd` lowers
// it by name), which every backend implements. Outside a kernel a method
// panics. Write each declaration out: the kernel macro does not expand
// macros in this file.

/// One native-width block of elements, a kernel value: the build target
/// decides the number of lanes. Use `lanes()` to advance a slice cursor.
#[crate::intrinsic(simd)]
#[derive(Clone, Copy)]
pub struct Lane<T: LaneElement>;
/// One predicate per lane of the corresponding data vector.
#[crate::intrinsic(simd)]
#[derive(Clone, Copy)]
pub struct Mask<T: LaneElement>;

#[crate::intrinsic(simd)]
impl<T: LaneElement> Mask<T> {
    /// Whether any lane is active.
    pub fn any(self) -> bool;
    /// Whether every lane is active.
    pub fn all(self) -> bool;
    /// `yes` in the active lanes, `no` in the others.
    pub fn select<U: LaneElement<Index = T::Index>>(self, yes: Lane<U>, no: Lane<U>) -> Lane<U>;
    /// Number of active lanes.
    pub fn count(self) -> i32;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement> Not for Mask<T> {
    type Output = Self;
    fn not(self) -> Self;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement> BitAnd for Mask<T> {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement> BitOr for Mask<T> {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement> BitXor for Mask<T> {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self;
}

#[crate::intrinsic(simd)]
impl<T: LaneElement> Lane<T> {
    /// Lanes of the build target's vector.
    pub fn lanes() -> usize;
    /// Activate the first min(count, lanes()) lanes. Useful for the last partial block.
    pub fn first_n(count: usize) -> Mask<T>;
    /// `v` in every lane: a known value or a runtime scalar of the lane type.
    pub fn splat(v: T) -> Self;
    /// Lane `i` holds `i`.
    pub fn iota() -> Self;
    /// `src[offset..offset + lanes()]`; out of bounds traps.
    pub fn load(src: &[T], offset: usize) -> Self;
    /// Store the lanes at `dst[offset..]`; out of bounds traps.
    pub fn store(self, dst: &mut [T], offset: usize);
    /// The active lanes from `src[offset + i]`, `fallback` in the others;
    /// only active lanes are bounds-checked.
    pub fn load_select<M: LaneElement<Index = T::Index>>(
        src: &[T],
        offset: usize,
        mask: Mask<M>,
        fallback: Self,
    ) -> Self;
    /// Store the active lanes at `dst[offset + i]`.
    pub fn store_select<M: LaneElement<Index = T::Index>>(
        self,
        dst: &mut [T],
        offset: usize,
        mask: Mask<M>,
    );
    /// Lane `i` from `src[indices[i]]`.
    pub fn gather(src: &[T], indices: Lane<T::Index>) -> Self;
    /// The active lanes from `src[indices[i]]`, `fallback` in the others.
    pub fn gather_select<M: LaneElement<Index = T::Index>>(
        src: &[T],
        mask: Mask<M>,
        indices: Lane<T::Index>,
        fallback: Self,
    ) -> Self;
    /// Write lane `i` to `dst[indices[i]]` in lane order.
    pub fn scatter(self, dst: &mut [T], indices: Lane<T::Index>);
    /// Write the active lanes to `dst[indices[i]]` in lane order.
    pub fn scatter_select<M: LaneElement<Index = T::Index>>(
        self,
        dst: &mut [T],
        mask: Mask<M>,
        indices: Lane<T::Index>,
    );
    /// `self + rhs` in the active lanes, `fallback` in the others.
    pub fn add_masked<M: LaneElement<Index = T::Index>>(
        self,
        mask: Mask<M>,
        rhs: Self,
        fallback: Self,
    ) -> Self;
    pub fn simd_eq(self, rhs: Self) -> Mask<T>;
    pub fn simd_ne(self, rhs: Self) -> Mask<T>;
    pub fn simd_lt(self, rhs: Self) -> Mask<T>;
    pub fn simd_le(self, rhs: Self) -> Mask<T>;
    pub fn simd_gt(self, rhs: Self) -> Mask<T>;
    pub fn simd_ge(self, rhs: Self) -> Mask<T>;
    /// Convert each lane to `U` of the same size (`as`).
    pub fn cast<U: LaneElement>(self) -> Lane<U>
    where
        T: LaneCast<U>;
    /// Convert the low half of the lanes to `U` of twice the size.
    pub fn widen_low<U: LaneElement>(self) -> Lane<U>
    where
        T: LaneCast<U>;
    /// Convert the high half of the lanes to `U` of twice the size.
    pub fn widen_high<U: LaneElement>(self) -> Lane<U>
    where
        T: LaneCast<U>;
    /// Convert the lanes of `self`, then of `high`, to `U` of half the size.
    pub fn narrow<U: LaneElement>(self, high: Self) -> Lane<U>
    where
        T: LaneCast<U>;
    /// Lane `index` modulo `lanes()`.
    pub fn extract_wrapping(self, index: i32) -> T;
    /// `self` with lane `index` modulo `lanes()` set to `value`.
    pub fn replace_wrapping(self, index: i32, value: T) -> Self;
    /// The last lane.
    pub fn last(self) -> T;
}

#[crate::intrinsic(simd)]
impl<T: LaneInteger> Lane<T> {
    /// The wrapping sum of the lanes.
    pub fn reduce_sum(self) -> T;
    /// The smaller of each pair of lanes.
    pub fn simd_min(self, rhs: Self) -> Self;
    /// The larger of each pair of lanes.
    pub fn simd_max(self, rhs: Self) -> Self;
    /// Add each pair of lanes, clamping to the lane type's range.
    pub fn saturating_add(self, rhs: Self) -> Self;
    /// Subtract each pair of lanes, clamping to the lane type's range.
    pub fn saturating_sub(self, rhs: Self) -> Self;
}

#[crate::intrinsic(simd)]
impl Lane<u32> {
    /// Rotate each lane left by a known count modulo 32.
    pub fn rotate_left(self, count: u32) -> Self;
}

#[crate::intrinsic(simd)]
impl Lane<i32> {
    /// Rotate each lane's bits left by a known count modulo 32.
    pub fn rotate_left(self, count: u32) -> Self;
    /// Elements from `offset` until `src[offset..]` is vector-aligned.
    pub fn alignment_prefix(src: &[i32], offset: usize) -> usize;
    /// The active lanes in order, zero in the remaining lanes.
    pub fn compress(self, mask: Mask<i32>) -> Self;
    /// The lanes in order into the active lanes, `fallback` in the others.
    pub fn expand(self, mask: Mask<i32>, fallback: Self) -> Self;
    /// Lane `i` from lane `indices[i]`, zero for an out-of-range index.
    pub fn swizzle(self, indices: Self) -> Self;
    /// Store the active lanes in order at `dst[offset..]`: the stores that
    /// fit happen, then an out-of-bounds lane traps.
    pub fn compress_store(self, dst: &mut [i32], offset: usize, mask: Mask<i32>);
    pub fn interleave_low(self, rhs: Self) -> Self;
    pub fn interleave_high(self, rhs: Self) -> Self;
    pub fn deinterleave_even(self, rhs: Self) -> Self;
    pub fn deinterleave_odd(self, rhs: Self) -> Self;
    /// Inclusive wrapping prefix sums of the lanes.
    pub fn prefix_sum(self) -> Self;
}

#[crate::intrinsic(simd)]
impl Lane<f32> {
    pub fn sqrt(self) -> Self;
    /// `self * multiplier + addend`, rounded once.
    pub fn mul_add(self, multiplier: Self, addend: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl Lane<f64> {
    pub fn sqrt(self) -> Self;
    /// `self * multiplier + addend`, rounded once.
    pub fn mul_add(self, multiplier: Self, addend: Self) -> Self;
}

#[crate::intrinsic(simd)]
impl<T: LaneElement> Add for Lane<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement> Sub for Lane<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement> Mul for Lane<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl Div for Lane<f32> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl Div for Lane<f64> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement + BitAnd<Output = T>> BitAnd for Lane<T> {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement + BitOr<Output = T>> BitOr for Lane<T> {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement + BitXor<Output = T>> BitXor for Lane<T> {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self;
}
#[crate::intrinsic(simd)]
impl<T: LaneElement + Not<Output = T>> Not for Lane<T> {
    type Output = Self;
    fn not(self) -> Self;
}
#[crate::intrinsic(simd)]
impl Shr<u32> for Lane<i32> {
    type Output = Self;
    /// Arithmetic shift by a known count modulo 32.
    fn shr(self, count: u32) -> Self;
}
#[crate::intrinsic(simd)]
impl Shr<u32> for Lane<u32> {
    type Output = Self;
    /// Logical shift by a known count modulo 32.
    fn shr(self, count: u32) -> Self;
}

// `x op= y` is `x = x op y`: a kernel stages the operator's intrinsic.
#[crate::intrinsic(simd)]
impl<T: LaneElement> AddAssign for Lane<T> {
    fn add_assign(&mut self, rhs: Self);
}
#[crate::intrinsic(simd)]
impl<T: LaneElement> SubAssign for Lane<T> {
    fn sub_assign(&mut self, rhs: Self);
}
#[crate::intrinsic(simd)]
impl<T: LaneElement> MulAssign for Lane<T> {
    fn mul_assign(&mut self, rhs: Self);
}
#[crate::intrinsic(simd)]
impl DivAssign for Lane<f32> {
    fn div_assign(&mut self, rhs: Self);
}
#[crate::intrinsic(simd)]
impl DivAssign for Lane<f64> {
    fn div_assign(&mut self, rhs: Self);
}
#[crate::intrinsic(simd)]
impl<T: LaneElement + BitAnd<Output = T>> BitAndAssign for Lane<T> {
    fn bitand_assign(&mut self, rhs: Self);
}
#[crate::intrinsic(simd)]
impl<T: LaneElement + BitOr<Output = T>> BitOrAssign for Lane<T> {
    fn bitor_assign(&mut self, rhs: Self);
}
#[crate::intrinsic(simd)]
impl<T: LaneElement + BitXor<Output = T>> BitXorAssign for Lane<T> {
    fn bitxor_assign(&mut self, rhs: Self);
}
