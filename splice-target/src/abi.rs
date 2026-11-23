//! The native kernel ABI shared by every backend: trap codes, the status a
//! Rust entry returns, and Rust imports using the host calling convention.
use std::{
    any::Any,
    cell::RefCell,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
};

/// A runtime error that stops a kernel. Every backend, the interpreter and
/// every Rust entry report it with the same code. Stores that completed
/// before the trap remain visible to the caller.
#[repr(u32)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Trap {
    /// A slice index at or beyond the slice length.
    Bounds = 1,
    /// Integer `/` or `%` by zero.
    DivisionByZero = 2,
    /// Signed `MIN / -1` or `MIN % -1`.
    DivisionOverflow = 3,
    /// Checked integer arithmetic or shift overflow.
    Overflow = 4,
    /// A Rust import panicked. [`take_panic_message`] returns its message.
    RustPanic = 5,
    /// A native caller entered Rust with a call shape the entry was not
    /// built for. This is a backend defect, not an error in the kernel.
    Abi = 6,
}

impl Trap {
    pub const ALL: [Self; 6] = [
        Self::Bounds,
        Self::DivisionByZero,
        Self::DivisionOverflow,
        Self::Overflow,
        Self::RustPanic,
        Self::Abi,
    ];

    /// The nonzero code native code stores or returns for this trap.
    pub const fn code(self) -> u32 {
        self as u32
    }

    pub const fn from_code(code: u32) -> Option<Self> {
        Some(match code {
            1 => Self::Bounds,
            2 => Self::DivisionByZero,
            3 => Self::DivisionOverflow,
            4 => Self::Overflow,
            5 => Self::RustPanic,
            6 => Self::Abi,
            _ => return None,
        })
    }

    /// The `i64` status for this trap: its code in the upper word. A
    /// successful status holds its zero-extended `u32` result instead, so
    /// any number of calls can forward a status without losing a result.
    pub const fn status(self) -> i64 {
        (self as i64) << 32
    }

    const fn message(self) -> &'static str {
        match self {
            Self::Bounds => "kernel slice index out of bounds",
            Self::DivisionByZero => "kernel integer division by zero",
            Self::DivisionOverflow => "kernel signed division overflow",
            Self::Overflow => "kernel integer arithmetic overflow",
            Self::RustPanic => "Rust import panicked",
            Self::Abi => "native code entered Rust with an unexpected call shape",
        }
    }
}

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Trap {}

/// The result word of a successful status, or its trap. An unknown code
/// can only come from a broken native caller and reads as [`Trap::Abi`].
pub const fn decode_status(status: i64) -> Result<u32, Trap> {
    let code = (status as u64 >> 32) as u32;
    if code == 0 {
        return Ok(status as u32);
    }
    match Trap::from_code(code) {
        Some(trap) => Err(trap),
        None => Err(Trap::Abi),
    }
}

thread_local! {
    static PANIC_MESSAGE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Run the Rust side of a native call. A panic stops at this boundary, its
/// message waits for [`take_panic_message`] on this thread, and the native
/// caller receives [`Trap::RustPanic`]. Native code never observes Rust
/// state after an unwind, so the closure need not be unwind safe.
pub fn rust_entry(call: impl FnOnce() -> i32) -> i64 {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(word) => i64::from(word as u32),
        Err(payload) => {
            let message = panic_message(payload);
            PANIC_MESSAGE.with(|slot| *slot.borrow_mut() = Some(message));
            Trap::RustPanic.status()
        }
    }
}

/// The message of the latest Rust panic that [`rust_entry`] caught on this
/// thread. The kernel call that reports [`Trap::RustPanic`] takes it.
pub fn take_panic_message() -> Option<String> {
    PANIC_MESSAGE.with(|slot| slot.borrow_mut().take())
}

/// The text of a panic payload. The payload is dropped inside Rust, even
/// when its own destructor panics.
pub fn panic_message(payload: Box<dyn Any + Send>) -> String {
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "Rust code panicked".to_owned());
    if let Err(secondary) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
        std::mem::forget(secondary);
    }
    message
}

/// A Rust import with at most two scalar words. Unused words are zero. It
/// returns a status (see [`Trap::status`]).
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

    /// Call the adapter and return its status.
    pub fn invoke(&self, words: &[i32]) -> i64 {
        if words.len() != self.arity as usize || self.packet_bytes().is_none() {
            return Trap::Abi.status();
        }
        rust_entry(|| (self.resolve)()(words))
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
        return Trap::Abi.status();
    }
    // SAFETY: the verified call initializes each word in its aligned scratch
    // area before handing control to the Rust adapter.
    symbol.invoke(unsafe { std::slice::from_raw_parts(words, arity as usize) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_round_trip_every_trap_and_result_word() {
        for trap in Trap::ALL {
            assert_eq!(Trap::from_code(trap.code()), Some(trap));
            assert_eq!(decode_status(trap.status()), Err(trap));
        }
        for word in [0, 1, u32::MAX] {
            assert_eq!(decode_status(i64::from(word)), Ok(word));
        }
        assert_eq!(Trap::from_code(0), None);
        assert_eq!(decode_status(99 << 32), Err(Trap::Abi));
        assert_eq!(decode_status(-1), Err(Trap::Abi));
    }

    #[test]
    fn a_caught_panic_keeps_its_message_for_the_caller() {
        let status = rust_entry(|| panic!("import failed with {}", 7));
        assert_eq!(decode_status(status), Err(Trap::RustPanic));
        assert_eq!(
            take_panic_message().as_deref(),
            Some("import failed with 7")
        );
        assert_eq!(take_panic_message(), None);
        assert_eq!(decode_status(rust_entry(|| -1)), Ok(u32::MAX));
    }

    #[test]
    fn a_mismatched_packet_is_an_abi_trap_not_a_panic() {
        static SYMBOL: RustPacketSymbol = RustPacketSymbol {
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
