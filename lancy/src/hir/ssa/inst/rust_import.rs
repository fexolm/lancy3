//! Lancy's Rust imports: a function of at most two `i32` words called with
//! the host convention, or a fixed number of words passed in memory.
use splice_target::abi::{Trap, rust_entry};

/// A Rust import with at most two scalar words. Unused words are zero. It
/// returns a status (see [`Trap::status`]).
pub type ScalarEntry = extern "C" fn(i32, i32) -> i64;

/// Static linker descriptor shared by the portable and native IR. A resolver
/// returns the Rust adapter for this process and monomorphization.
#[derive(Debug)]
pub struct RustImport {
    pub name: &'static str,
    pub resolve: fn() -> ScalarEntry,
}

impl PartialEq for RustImport {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && std::ptr::fn_addr_eq(self.resolve, other.resolve)
    }
}
impl Eq for RustImport {}

/// Rust-side adapter for a fixed-arity import with more than two scalar words.
pub type PacketEntry = fn(&[i32]) -> i32;

#[derive(Debug)]
pub struct RustPacketImport {
    pub name: &'static str,
    pub arity: u32,
    pub resolve: fn() -> PacketEntry,
}

impl RustPacketImport {
    pub fn packet_bytes(&self) -> Option<u32> {
        (3..=i32::MAX as u32 / 4)
            .contains(&self.arity)
            .then(|| self.arity * 4)
    }

    /// Call the adapter and return its status.
    pub fn invoke(&self, words: &[i32]) -> i64 {
        if words.len() != self.arity as usize || self.packet_bytes().is_none() {
            return Trap::Abi.status();
        }
        rust_entry(|| (self.resolve)()(words))
    }
}

impl PartialEq for RustPacketImport {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.arity == other.arity
            && std::ptr::fn_addr_eq(self.resolve, other.resolve)
    }
}
impl Eq for RustPacketImport {}

/// # Safety
/// `symbol` points to a live immutable linker descriptor. `words` points to
/// `arity` initialized, aligned `i32` values for this synchronous call.
pub unsafe extern "C" fn invoke_packet_import(
    symbol: *const RustPacketImport,
    words: *const i32,
    arity: u32,
) -> i64 {
    // SAFETY: native lowering embeds the static descriptor's address.
    let symbol = unsafe { &*symbol };
    if symbol.arity != arity || symbol.packet_bytes().is_none() {
        return Trap::Abi.status();
    }
    // SAFETY: the verified call initializes each word in its aligned scratch
    // area before handing control to the Rust adapter.
    symbol.invoke(unsafe { std::slice::from_raw_parts(words, arity as usize) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use splice_target::abi::{decode_status, take_panic_message};

    #[test]
    fn a_mismatched_packet_is_an_abi_trap_not_a_panic() {
        static SYMBOL: RustPacketImport = RustPacketImport {
            name: "sum",
            arity: 3,
            resolve: || |words| words.iter().sum(),
        };
        assert_eq!(decode_status(SYMBOL.invoke(&[1, 2, 3])), Ok(6));
        assert_eq!(decode_status(SYMBOL.invoke(&[1, 2])), Err(Trap::Abi));
        // SAFETY: the descriptor is static and the words outlive the call.
        let status = unsafe { invoke_packet_import(&SYMBOL, [1, 2].as_ptr(), 2) };
        assert_eq!(decode_status(status), Err(Trap::Abi));
        assert_eq!(take_panic_message(), None);
    }
}
