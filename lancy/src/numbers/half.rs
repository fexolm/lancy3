//! Binary16 and bfloat16 use the same sign/exponent/fraction layout with
//! different exponent biases. Widening either format to binary64 is exact.
#[cfg(any(test, target_arch = "aarch64"))]
use super::{Action, Binary, Ternary, Unary, reference::float64};
use crate::hir::Lane;

pub(crate) struct Format {
    fraction: u32,
    bias: i32,
}

impl Format {
    pub const fn for_lane(lane: Lane) -> Self {
        match lane {
            Lane::F16 => Self {
                fraction: 10,
                bias: 15,
            },
            Lane::BF16 => Self {
                fraction: 7,
                bias: 127,
            },
            _ => unreachable!(),
        }
    }
    fn infinity(&self) -> u16 {
        0x7fff & !((1 << self.fraction) - 1)
    }
    #[cfg(any(test, target_arch = "aarch64"))]
    pub fn decode(&self, raw: u16) -> f64 {
        let magnitude = raw & 0x7fff;
        let frac = magnitude & ((1 << self.fraction) - 1);
        let exponent = i32::from(magnitude >> self.fraction);
        let n = if magnitude >= self.infinity() {
            if frac == 0 { f64::INFINITY } else { f64::NAN }
        } else if exponent == 0 {
            f64::from(frac) * 2f64.powi(1 - self.bias - self.fraction as i32)
        } else {
            f64::from((1 << self.fraction) + frac)
                * 2f64.powi(exponent - self.bias - self.fraction as i32)
        };
        if raw >> 15 != 0 { -n } else { n }
    }
    /// Round once. A residual breaks a midpoint tie when binary64 rounded an
    /// exact product-plus-addend or wide integer onto that midpoint.
    pub fn encode(&self, n: f64, residual: f64) -> u16 {
        let sign = ((n.to_bits() >> 48) as u16) & 0x8000;
        let abs = n.abs();
        if n.is_nan() {
            return sign | self.infinity() | (1 << (self.fraction - 1));
        }
        if abs == 0. {
            return sign;
        }
        let exponent = ((abs.to_bits() >> 52) as i32) - 1023;
        if exponent > self.bias {
            return sign | self.infinity();
        }
        let e = exponent.max(1 - self.bias);
        let units = abs * 2f64.powi(self.fraction as i32 - e);
        let rounded = if residual != 0. && units.fract() == 0.5 {
            if n.is_sign_negative() == residual.is_sign_negative() {
                units.ceil()
            } else {
                units.floor()
            }
        } else {
            units.round_ties_even()
        };
        let base = ((e + self.bias - 1) as u16) << self.fraction;
        sign | (base + rounded as u16)
    }

    #[cfg(any(test, target_arch = "aarch64"))]
    pub fn eval(&self, action: Action, x: u16, y: u16, z: u16) -> u128 {
        let a = self.decode(x);
        let b = self.decode(y);
        let c = self.decode(z);
        let mag = x & 0x7fff;
        match action {
            Action::Unary(Unary::Neg) => return u128::from(x ^ 0x8000),
            Action::Unary(Unary::Abs) => return mag.into(),
            Action::Binary(Binary::CopySign) => return u128::from(mag | y & 0x8000),
            Action::Unary(Unary::IsNormal) => {
                return if mag >= 1 << self.fraction && mag < self.infinity() {
                    u128::MAX
                } else {
                    0
                };
            }
            Action::Unary(Unary::IsSubnormal) => {
                return if mag != 0 && mag < 1 << self.fraction {
                    u128::MAX
                } else {
                    0
                };
            }
            Action::Unary(op @ (Unary::NextUp | Unary::NextDown)) => {
                let up = op == Unary::NextUp;
                let result = if a.is_nan() || (a.is_infinite() && up == a.is_sign_positive()) {
                    x
                } else if mag == 0 {
                    if up { 1 } else { 0x8001 }
                } else if up == a.is_sign_positive() {
                    x + 1
                } else {
                    x - 1
                };
                return result.into();
            }
            Action::Ternary(Ternary::MulAdd) => {
                let product = a * b;
                let sum = product + c;
                // The significand product is exact in f64. FastTwoSum recovers the rounding residual.
                let residual = if product.abs() >= c.abs() {
                    (product - sum) + c
                } else {
                    (c - sum) + product
                };
                return self.encode(sum, residual).into();
            }
            _ => {}
        }
        let bits = float64(action, a.to_bits(), b.to_bits(), c.to_bits());
        match action {
            Action::Unary(op) if op.result_type(crate::hir::Type::F64) == crate::hir::Type::I32 => {
                bits
            }
            Action::Binary(op)
                if op.result_type(crate::hir::Type::F64) == crate::hir::Type::I32 =>
            {
                bits
            }
            _ => self.encode(f64::from_bits(bits as u64), 0.).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_finite_16_bit_pattern_round_trips_including_signed_zero() {
        for lane in [Lane::F16, Lane::BF16] {
            let format = Format::for_lane(lane);
            for raw in 0..=u16::MAX {
                let n = format.decode(raw);
                if !n.is_nan() {
                    assert_eq!(format.encode(n, 0.), raw);
                }
            }
            let midpoint = 1. + 2f64.powi(-1 - format.fraction as i32);
            let one = format.encode(1., 0.);
            assert_eq!(format.encode(midpoint, 0.), one);
            assert_eq!(format.encode(midpoint.next_up(), 0.), one + 1);
            assert_eq!(format.encode(-midpoint.next_up(), 0.), (one + 1) | 0x8000);
        }
    }
}
