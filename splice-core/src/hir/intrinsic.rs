use super::{BufferKind, ScalarType, Simd, ValueType};

/// A built-in operation. Buffer identities and build-time immediates live
/// in the operation, while runtime operands are registers of the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    Simd(Simd),
    Math { op: MathOp, ty: ScalarType },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathOp {
    CountOnes,
    LeadingZeros,
    TrailingZeros,
    RotateLeft,
}

impl MathOp {
    pub fn result(self, ty: ScalarType) -> ScalarType {
        match self {
            Self::RotateLeft => ty,
            _ => ScalarType::U32,
        }
    }
}

impl Intrinsic {
    pub fn output(self) -> Option<ValueType> {
        match self {
            Self::Simd(simd) => simd.output(),
            Self::Math { op, ty } => Some(ValueType::Scalar(op.result(ty))),
        }
    }

    pub fn validate(self, args: &[ValueType], buffers: &[BufferKind]) -> crate::Result<()> {
        match self {
            Self::Simd(simd) => simd.validate(args, buffers),
            Self::Math { op, ty } => {
                let operands = [ValueType::Scalar(ty), ValueType::Scalar(ScalarType::U32)];
                let arity = if op == MathOp::RotateLeft { 2 } else { 1 };
                if ty.is_integer() && ty != ScalarType::Index && args == &operands[..arity] {
                    Ok(())
                } else {
                    Err(crate::BuildError::verify(format!(
                        "invalid operands for {ty}::{op:?}: {args:?}"
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_intrinsics_require_the_declared_receiver_count_and_result_types() {
        use ScalarType::{F32, I32, U32, U64};
        let scalar = ValueType::Scalar;
        let rotate = Intrinsic::Math {
            op: MathOp::RotateLeft,
            ty: U64,
        };
        assert_eq!(rotate.output(), Some(scalar(U64)));
        assert!(rotate.validate(&[scalar(U64), scalar(U32)], &[]).is_ok());
        for args in [
            vec![scalar(U64)],
            vec![scalar(U64), scalar(I32)],
            vec![scalar(U32), scalar(U32)],
        ] {
            assert!(rotate.validate(&args, &[]).is_err());
        }
        let ones = Intrinsic::Math {
            op: MathOp::CountOnes,
            ty: I32,
        };
        assert_eq!(ones.output(), Some(scalar(U32)));
        assert!(ones.validate(&[scalar(I32), scalar(U32)], &[]).is_err());
        let float = Intrinsic::Math {
            op: MathOp::CountOnes,
            ty: F32,
        };
        assert!(float.validate(&[scalar(F32)], &[]).is_err());
    }
}
