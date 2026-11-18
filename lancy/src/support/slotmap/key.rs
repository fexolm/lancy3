/// A compact typed index. The maximum `u32` is reserved as a sentinel.
/// For each representable index `i`, `Self::from_index(i).index()` must equal `i`.
pub trait SlotKey: Copy {
    fn from_index(index: usize) -> Self;
    fn index(self) -> usize;
}

/// Define a distinct four-byte key type for a family of IR entities.
#[macro_export]
macro_rules! slot_key {
    ($vis:vis $name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        #[repr(transparent)]
        $vis struct $name(pub u32);

        impl $name {
            pub const INVALID: Self = Self(u32::MAX);

            pub const fn from_index(index: usize) -> Self {
                assert!(index < Self::INVALID.0 as usize, "slot key exceeds 32-bit range");
                Self(index as u32)
            }

            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl $crate::support::slotmap::SlotKey for $name {
            fn from_index(index: usize) -> Self {
                Self::from_index(index)
            }

            fn index(self) -> usize {
                self.index()
            }
        }
    };
}
