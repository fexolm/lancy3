//! Slice memory operations, lowered to the checks and unchecked accesses
//! of [`Primitives`]. What an access that leaves its slice does is decided
//! here, once for every backend:
//!
//! - an inactive lane is neither checked nor accessed;
//! - a lane is outside when its element is not below the slice length; a
//!   `u16` index lies inside every slice longer than `u16::MAX`;
//! - a read with an active lane outside traps before reading anything;
//! - a write with an active lane outside first writes, in lane order, its
//!   active lanes before the first such lane, then traps;
//! - a scatter writes its lanes in lane order, so the last of the lanes
//!   that repeat an index wins;
//! - a compress store writes the compressed lanes that fit, then traps if
//!   any did not.
use super::ScalarType;
use super::{Access, Reg, Simd, SimdOp, Slice};

/// The slice an operation accesses and the element of its vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub slice: Slice,
    pub element: ScalarType,
}

/// Where the lanes of an access are: consecutive elements from an offset,
/// or one element per lane of an index vector.
#[derive(Clone, Copy, Debug)]
pub enum Address<S, V> {
    Offset(S),
    Indices(V),
}

/// Work a backend does on the path that traps, before it reports the trap.
pub type BeforeTrap<'a, P> =
    Option<&'a mut dyn FnMut(&mut P) -> Result<(), <P as Primitives>::Error>>;

/// What a backend implements to lower slice memory operations with
/// [`lower`]. A scalar is a backend value of any scalar type; a vector is
/// a data, index or mask vector. Masks of an element `e` have the lanes of
/// `e.index()`.
pub trait Primitives {
    type Scalar: Copy;
    type Vector: Copy;
    type Error;

    /// The value of an [`Offset`](super::Slot::Offset) operand.
    fn read_offset(&mut self, r: Reg) -> Result<Self::Scalar, Self::Error>;
    /// The value of a vector, index or mask operand.
    fn read_vector(&mut self, r: Reg) -> Result<Self::Vector, Self::Error>;

    /// Trap unless the whole vector at `offset` lies in the slice, after
    /// running `before_trap`. When `offset` itself is outside, no lane is
    /// inside, and the backend may trap without running it.
    fn require_full(
        &mut self,
        region: Region,
        offset: Self::Scalar,
        before_trap: BeforeTrap<'_, Self>,
    ) -> Result<(), Self::Error>;
    /// How many lanes of the vector at `offset` lie in the slice: a lane
    /// count, which may exceed the number of lanes when all of them do.
    fn room(&mut self, region: Region, offset: Self::Scalar) -> Result<Self::Scalar, Self::Error>;
    /// The mask of the first `count` lanes, for a [`room`](Self::room).
    fn first_lanes(
        &mut self,
        e: ScalarType,
        count: Self::Scalar,
    ) -> Result<Self::Vector, Self::Error>;
    /// The lanes whose index, with the signedness of the index lanes, is
    /// negative or not below the slice length. This is exact for every
    /// length but, with narrow unsigned index lanes, a length above their
    /// largest value, where the result is ignored (no such index leaves the
    /// slice).
    fn outside(
        &mut self,
        region: Region,
        indices: Self::Vector,
    ) -> Result<Self::Vector, Self::Error>;
    /// Whether `slice` holds at most `n` elements: a `bool`.
    fn length_at_most(&mut self, slice: Slice, n: u64) -> Result<Self::Scalar, Self::Error>;
    fn and(&mut self, a: Self::Scalar, b: Self::Scalar) -> Result<Self::Scalar, Self::Error>;
    fn mask_and(&mut self, a: Self::Vector, b: Self::Vector) -> Result<Self::Vector, Self::Error>;
    fn mask_xor(&mut self, a: Self::Vector, b: Self::Vector) -> Result<Self::Vector, Self::Error>;
    /// Whether `mask` has a lane: a `bool`.
    fn any(&mut self, mask: Self::Vector) -> Result<Self::Scalar, Self::Error>;
    /// The lanes before the first lane of `mask`, or all lanes if it has none.
    fn before_first(
        &mut self,
        e: ScalarType,
        mask: Self::Vector,
    ) -> Result<Self::Vector, Self::Error>;
    /// Whether `mask` has more lanes than `count`, a lane count: a `bool`.
    fn exceeds(
        &mut self,
        mask: Self::Vector,
        count: Self::Scalar,
    ) -> Result<Self::Scalar, Self::Error>;
    /// Trap with [`crate::Trap::Bounds`] when `condition` holds, after
    /// running `before_trap`.
    fn trap_if(
        &mut self,
        condition: Self::Scalar,
        before_trap: BeforeTrap<'_, Self>,
    ) -> Result<(), Self::Error>;

    /// Read the lanes at `at`, which are in bounds. A masked read reads
    /// only the lanes of `mask` and takes the others from `passthrough`.
    fn load(
        &mut self,
        region: Region,
        at: Address<Self::Scalar, Self::Vector>,
        masked: Option<Masked<Self::Vector>>,
    ) -> Result<Self::Vector, Self::Error>;
    /// Write the lanes of `value` at `at`, which are in bounds, in lane
    /// order: all of them, or only those of `mask`.
    fn store(
        &mut self,
        region: Region,
        at: Address<Self::Scalar, Self::Vector>,
        mask: Option<Self::Vector>,
        value: Self::Vector,
    ) -> Result<(), Self::Error>;
    /// Write the lanes of `value` that `mask` selects to consecutive
    /// elements from `offset`, at most `room` of them: the
    /// [`room`](Self::room) at `offset`.
    fn compress_store(
        &mut self,
        region: Region,
        offset: Self::Scalar,
        mask: Self::Vector,
        value: Self::Vector,
        room: Self::Scalar,
    ) -> Result<(), Self::Error>;
}

/// The lanes a masked read reads, and the value of the others.
#[derive(Clone, Copy, Debug)]
pub struct Masked<V> {
    pub mask: V,
    pub passthrough: V,
}

/// Lower a SIMD operation that reads or writes a slice.
///
/// # Panics
/// If `inst` does not access the elements of a slice.
pub fn lower<P: Primitives>(
    p: &mut P,
    inst: Simd,
    args: &[Reg],
) -> Result<Option<P::Vector>, P::Error> {
    let memory = inst
        .op
        .info()
        .memory
        .filter(|memory| memory.access != Access::Address)
        .expect("a slice memory operation");
    let e = inst.element;
    let region = Region {
        slice: memory.slice,
        element: e,
    };
    let write = memory.access == Access::Write;
    if let SimdOp::CompressStore(_) = inst.op {
        let offset = p.read_offset(args[0])?;
        let (mask, value) = (p.read_vector(args[1])?, p.read_vector(args[2])?);
        let room = p.room(region, offset)?;
        p.compress_store(region, offset, mask, value, room)?;
        let over = p.exceeds(mask, room)?;
        return p.trap_if(over, None).map(|()| None);
    }
    let at = if memory.indexed {
        Address::Indices(p.read_vector(args[0])?)
    } else {
        Address::Offset(p.read_offset(args[0])?)
    };
    let active = if memory.masked {
        Some(p.read_vector(args[1])?)
    } else {
        None
    };
    // The value written, or the passthrough of a masked read.
    let last = if write || memory.masked {
        Some(p.read_vector(*args.last().expect("an operand"))?)
    } else {
        None
    };
    let value = move || last.expect("a value operand");
    let masked = |mask| Masked {
        mask,
        passthrough: value(),
    };
    match (at, active) {
        (Address::Offset(offset), None) if write => {
            p.require_full(
                region,
                offset,
                Some(&mut |p| {
                    let room = p.room(region, offset)?;
                    let inside = p.first_lanes(e, room)?;
                    p.store(region, at, Some(inside), value())
                }),
            )?;
            p.store(region, at, None, value()).map(|()| None)
        }
        (Address::Offset(offset), None) => {
            p.require_full(region, offset, None)?;
            p.load(region, at, None).map(Some)
        }
        (Address::Offset(offset), Some(active)) => {
            // The lanes outside are a suffix: the active lanes inside are
            // exactly those before the first active lane outside.
            let room = p.room(region, offset)?;
            let inside = p.first_lanes(e, room)?;
            let enabled = p.mask_and(active, inside)?;
            let bad = p.mask_xor(active, enabled)?;
            let fail = p.any(bad)?;
            if write {
                p.store(region, at, Some(enabled), value())?;
                return p.trap_if(fail, None).map(|()| None);
            }
            p.trap_if(fail, None)?;
            p.load(region, at, Some(masked(enabled))).map(Some)
        }
        (Address::Indices(indices), active) => {
            let outside = p.outside(region, indices)?;
            let bad = match active {
                Some(active) => p.mask_and(active, outside)?,
                None => outside,
            };
            let mut fail = p.any(bad)?;
            // Narrow unsigned indices cannot leave a longer slice; narrow
            // signed ones still can, by being negative.
            let index = e.index();
            if !index.is_signed() && index.bits() < 32 {
                let largest = u64::MAX >> (64 - index.bits());
                let short = p.length_at_most(memory.slice, largest)?;
                fail = p.and(fail, short)?;
            }
            if !write {
                p.trap_if(fail, None)?;
                return p.load(region, at, active.map(masked)).map(Some);
            }
            p.trap_if(
                fail,
                Some(&mut |p| {
                    let before = p.before_first(e, bad)?;
                    let enabled = match active {
                        Some(active) => p.mask_and(active, before)?,
                        None => before,
                    };
                    p.store(region, at, Some(enabled), value())
                }),
            )?;
            p.store(region, at, active, value()).map(|()| None)
        }
    }
}
