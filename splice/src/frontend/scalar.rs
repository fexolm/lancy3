//! Staging support used by the scalar kernel attribute during HIR construction.
use super::KernelBuilder;
use crate::hir::Reg;
use std::{cell::Cell, marker::PhantomData, rc::Rc};

#[doc(hidden)]
pub struct Host<T>(pub T);
#[doc(hidden)]
pub trait ScalarConstant {
    fn bits(self) -> i32;
}
impl ScalarConstant for i32 {
    fn bits(self) -> i32 {
        self
    }
}
impl ScalarConstant for bool {
    fn bits(self) -> i32 {
        -i32::from(self)
    }
}
impl<T: ScalarConstant + Copy> ScalarConstant for &T {
    fn bits(self) -> i32 {
        (*self).bits()
    }
}
#[doc(hidden)]
#[diagnostic::on_unimplemented(message = "a scalar kernel parameter requires an i32 value")]
pub trait RunArg {
    fn reg(self, builder: &mut dyn KernelBuilder) -> Reg;
}
impl RunArg for Reg {
    fn reg(self, _: &mut dyn KernelBuilder) -> Reg {
        self
    }
}
impl RunArg for Host<i32> {
    fn reg(self, b: &mut dyn KernelBuilder) -> Reg {
        b.constant(self.0)
    }
}

// Method lookup on &operands selects this implementation for ordinary i32
// division (including borrowed operands). Other host types retain their own
// Rust operator through the extra autoref below.
trait HostI32 {
    fn integer(&self) -> i32;
}
impl HostI32 for i32 {
    fn integer(&self) -> i32 {
        *self
    }
}
impl HostI32 for &i32 {
    fn integer(&self) -> i32 {
        **self
    }
}
impl HostI32 for &mut i32 {
    fn integer(&self) -> i32 {
        **self
    }
}

#[doc(hidden)]
pub trait DivisionGuard {
    fn faulting_i32_division(self) -> Option<(i32, i32)>;
}
impl<L: HostI32, R: HostI32> DivisionGuard for &(L, R) {
    fn faulting_i32_division(self) -> Option<(i32, i32)> {
        let (lhs, rhs) = (self.0.integer(), self.1.integer());
        (rhs == 0 || (lhs == i32::MIN && rhs == -1)).then_some((lhs, rhs))
    }
}
impl<L, R> DivisionGuard for &&(L, R) {
    fn faulting_i32_division(self) -> Option<(i32, i32)> {
        None
    }
}

thread_local! {
    static EXPANSION_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Bounds non-terminating specialization, including mutually recursive factories.
#[doc(hidden)]
pub struct ExpansionGuard(PhantomData<Rc<()>>);

#[doc(hidden)]
pub fn enter_expansion() -> Result<ExpansionGuard, String> {
    EXPANSION_DEPTH.with(|depth| {
        if depth.get() >= 64 {
            return Err("kernel specialization exceeded 64 nested expansions".into());
        }
        depth.set(depth.get() + 1);
        Ok(ExpansionGuard(PhantomData))
    })
}
impl Drop for ExpansionGuard {
    fn drop(&mut self) {
        EXPANSION_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}
/// Resolve a read-only column against the layout supplied at specialization.
#[doc(hidden)]
pub fn column(
    (kinds, base): (&[crate::BufferKind], crate::hir::Buffer),
    slot: usize,
    element: crate::simd::Element,
) -> crate::Result<crate::hir::Buffer> {
    match kinds.get(slot) {
        Some(kind) if kind.element() == element => {
            let index = base
                .index()
                .checked_add(slot)
                .and_then(|i| u32::try_from(i).ok())
                .ok_or_else(|| format!("column {slot} is not representable"))?;
            Ok(crate::hir::Buffer(index))
        }
        Some(_) => Err(format!("column {slot} expects {} elements", element.name())),
        None => Err(format!("unknown column {slot}")),
    }
}
