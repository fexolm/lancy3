/// A compact typed index. The maximum `u32` is reserved as a sentinel.
/// For each representable index `i`, `Self::from_index(i).index()` must equal `i`.
pub(crate) trait SlotKey: Copy {
    fn from_index(index: usize) -> Self;
    fn index(self) -> usize;
}

/// Define a distinct four-byte key type for a family of IR entities.
macro_rules! slot_key {
    ($(#[$meta:meta])* $vis:vis $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        #[repr(transparent)]
        $vis struct $name(pub(crate) u32);

        impl $name {
            /// The reserved sentinel; never names a live entity.
            $vis const INVALID: Self = Self(u32::MAX);

            /// The key with position `index`.
            ///
            /// # Panics
            ///
            /// If `index` is not below `u32::MAX`.
            $vis const fn from_index(index: usize) -> Self {
                assert!(index < Self::INVALID.0 as usize, "slot key exceeds 32-bit range");
                Self(index as u32)
            }

            /// The position of this key.
            $vis const fn index(self) -> usize {
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
pub(crate) use slot_key;
