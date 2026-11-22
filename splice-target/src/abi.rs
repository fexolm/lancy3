//! Rust imports using the host system calling convention.

/// Unused words are zero. On success the upper return word is zero and
/// the lower word holds the result. An error sets the upper word.
pub type ScalarEntry = extern "C" fn(i32, i32) -> i64;

/// Static linker descriptor shared by the portable and native IR. A resolver
/// returns the Rust adapter for this process and monomorphization.
#[derive(Debug)]
pub struct RustSymbol {
    pub name: &'static str,
    pub resolve: fn() -> ScalarEntry,
}

impl PartialEq for RustSymbol {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && std::ptr::fn_addr_eq(self.resolve, other.resolve)
    }
}
impl Eq for RustSymbol {}

/// Rust-side adapter for a fixed-arity import with more than two scalar words.
pub type PacketEntry = fn(&[i32]) -> i32;

#[derive(Debug)]
pub struct RustPacketSymbol {
    pub name: &'static str,
    pub arity: u32,
    pub resolve: fn() -> PacketEntry,
}

impl RustPacketSymbol {
    pub fn packet_bytes(&self) -> Option<u32> {
        (3..=i32::MAX as u32 / 4)
            .contains(&self.arity)
            .then(|| self.arity * 4)
    }

    pub fn invoke(&self, words: &[i32]) -> i64 {
        const FAILURE: i64 = 5_i64 << 32;
        if words.len() != self.arity as usize || self.packet_bytes().is_none() {
            return FAILURE;
        }
        match std::panic::catch_unwind(|| (self.resolve)()(words)) {
            Ok(value) => i64::from(value as u32),
            Err(payload) => {
                if let Err(secondary) =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(payload)))
                {
                    std::mem::forget(secondary);
                }
                FAILURE
            }
        }
    }
}

impl PartialEq for RustPacketSymbol {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.arity == other.arity
            && std::ptr::fn_addr_eq(self.resolve, other.resolve)
    }
}
impl Eq for RustPacketSymbol {}

/// # Safety
/// `symbol` points to a live immutable linker descriptor. `words` points to
/// `arity` initialized, aligned `i32` values for this synchronous call.
pub unsafe extern "C" fn invoke_packet_import(
    symbol: *const RustPacketSymbol,
    words: *const i32,
    arity: u32,
) -> i64 {
    // SAFETY: native lowering embeds the static descriptor's address.
    let symbol = unsafe { &*symbol };
    if symbol.arity != arity || symbol.packet_bytes().is_none() {
        return 5_i64 << 32;
    }
    // SAFETY: the verified call initializes each word in its aligned scratch
    // area before handing control to the Rust adapter.
    symbol.invoke(unsafe { std::slice::from_raw_parts(words, arity as usize) })
}
