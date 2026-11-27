//! The numeric fallback of both backends: operations without a native form
//! call [`entry`], which evaluates [`crate::support::numeric`] over an area
//! of the caller's frame.
//!
//! The caller stores each operand at `index * SLOT_BYTES` of the area, calls
//! `entry(site, area)` with the platform C convention, and on a zero return
//! reads the result from `RESULT`. A nonzero return is a [`Trap`] code.
use crate::Trap;
use crate::hir::Type;
use crate::support::numeric::{Action, Bits, evaluate};

/// Bytes of one operand or result slot: the widest vector.
pub(crate) const SLOT_BYTES: usize = 64;
/// Offset of the result slot, after three operand slots.
pub(crate) const RESULT: usize = 3 * SLOT_BYTES;
/// Bytes of the area the caller reserves.
pub(crate) const AREA_BYTES: usize = 4 * SLOT_BYTES;
const _: () = assert!(crate::hir::MAX_VECTOR_BYTES <= SLOT_BYTES);

/// One call site: what to evaluate and how many bytes each operand has.
#[derive(Debug)]
pub(crate) struct Site {
    action: Action,
    input: Type,
    output: Type,
    sizes: [usize; 3],
    width: usize,
}

impl Site {
    pub(crate) fn new(
        action: Action,
        input: Type,
        output: Type,
        args: [Option<Type>; 3],
        width: usize,
    ) -> Self {
        Self {
            action,
            input,
            output,
            sizes: args.map(|arg| arg.map_or(0, |ty| ty.bytes(width))),
            width,
        }
    }
}

/// The sites of one compiled module. Each site is boxed, so the address
/// [`Sites::add`] returns stays valid while the code that embeds it lives:
/// the kernel keeps this table next to its executable memory.
#[derive(Debug, Default)]
#[expect(
    clippy::vec_box,
    reason = "each site keeps its address while the table grows"
)]
pub(crate) struct Sites(Vec<Box<Site>>);

impl Sites {
    /// Keep `site` and return its address for the generated code.
    pub(crate) fn add(&mut self, site: Site) -> u64 {
        let site = Box::new(site);
        let address = std::ptr::from_ref::<Site>(&site) as u64;
        self.0.push(site);
        address
    }
}

/// The address generated code calls.
pub(crate) fn entry() -> u64 {
    primitive as *const () as u64
}

/// Evaluate `site` over `area` and return 0 or a trap code. A panic cannot
/// unwind into native code: it stops here and reports [`Trap::Abi`], as
/// would any other helper defect.
///
/// # Safety
///
/// `site` comes from a live [`Sites`] table, and `area` points to
/// [`AREA_BYTES`] writable bytes whose operand slots hold the operands that
/// `site` describes. Both backends emit exactly that before the call.
unsafe extern "C" fn primitive(site: &Site, area: *mut u8) -> u32 {
    let run = || -> Result<(), Trap> {
        let args = std::array::from_fn(|i| {
            let mut data: Bits = [0; SLOT_BYTES];
            // SAFETY: slot `i` of the area holds `sizes[i]` operand bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    area.add(i * SLOT_BYTES),
                    data.as_mut_ptr(),
                    site.sizes[i],
                )
            };
            data
        });
        let value = evaluate(site.action, site.input, site.output, &args, site.width)?;
        // SAFETY: the result slot has SLOT_BYTES bytes, enough for any type.
        unsafe {
            std::ptr::copy_nonoverlapping(
                value.as_ptr(),
                area.add(RESULT),
                site.output.bytes(site.width),
            )
        };
        Ok(())
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(())) => 0,
        Ok(Err(trap)) => trap.code(),
        Err(payload) => {
            // Only the code reaches the caller: this is a backend defect.
            drop(splice_target::abi::panic_message(payload));
            Trap::Abi.code()
        }
    }
}
