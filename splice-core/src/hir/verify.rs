//! Verify a portable function by replaying it through the one verifier.

use super::{Function, Instr};

impl Function {
    pub(super) fn verify(mut self) -> crate::Result<Self> {
        if self.register_count() > u32::MAX as usize {
            return Err(crate::BuildError::verify("too many registers"));
        }
        let mut sink = crate::validate::Sink::new(self.signature(), self.options.debug());
        let mut validating = crate::validate::Validating::new(&mut sink, self.signature())?;
        super::replay(&self, &mut validating)?;
        validating.finish()?;
        self.needs_status = self.code.iter().any(Instr::may_trap);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        Binary, BufferKind, Constant, Intrinsic, Label, Reg, ScalarType, Simd, SimdOp, ValueType,
    };
    use super::*;
    use crate::debug::{Location, Point, Value, Variable};
    use crate::hir::Slice;
    use std::sync::Arc;

    const R0: Reg = Reg(0);
    const R1: Reg = Reg(1);
    const L0: Label = Label(0);

    const I32: ValueType = ValueType::Scalar(ScalarType::I32);

    /// A function of `arity` `i32` parameters with an `i32` result.
    fn function(
        arity: usize,
        buffer_kinds: Vec<BufferKind>,
        locals: Vec<ValueType>,
        code: Vec<Instr>,
    ) -> Function {
        typed(vec![ScalarType::I32; arity], buffer_kinds, locals, code)
    }

    fn typed(
        params: Vec<ScalarType>,
        buffer_kinds: Vec<BufferKind>,
        locals: Vec<ValueType>,
        code: Vec<Instr>,
    ) -> Function {
        let labels = code
            .iter()
            .filter_map(|op| match op {
                Instr::Label(l) | Instr::Jump(l) => Some(l.0 as usize + 1),
                Instr::Branch { yes, no, .. } => Some(yes.0.max(no.0) as usize + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        Function {
            source: None,
            options: crate::BuildOptions::host(crate::Mode::Debug).unwrap(),
            params,
            result: ScalarType::I32,
            buffer_kinds,
            needs_status: false,
            locals,
            code,
            labels,
        }
    }

    fn error(
        arity: usize,
        kinds: Vec<BufferKind>,
        locals: Vec<ValueType>,
        code: Vec<Instr>,
    ) -> String {
        reason(function(arity, kinds, locals, code).verify().unwrap_err())
    }

    fn typed_error(
        params: Vec<ScalarType>,
        kinds: Vec<BufferKind>,
        locals: Vec<ValueType>,
        code: Vec<Instr>,
    ) -> String {
        reason(typed(params, kinds, locals, code).verify().unwrap_err())
    }

    /// The verifier's reason, with the instruction it names.
    fn reason(error: crate::BuildError) -> String {
        match error {
            crate::BuildError::Verify {
                instr: Some(i),
                reason,
            } => format!("{reason} at instruction {i}"),
            crate::BuildError::Verify {
                instr: None,
                reason,
            } => reason,
            other => panic!("expected a verifier error, got {other:?}"),
        }
    }

    fn debug(location: Location, variables: Vec<Variable>) -> Instr {
        Instr::Debug(Point {
            location,
            variables: variables.into(),
        })
    }

    const HERE: Location = Location {
        file: "kernel.rs",
        directory: "/src",
        line: 1,
        column: 1,
    };

    #[test]
    fn a_well_formed_function_verifies_and_records_traps() {
        let f = typed(
            vec![ScalarType::Index],
            vec![BufferKind::READ_I32],
            vec![I32],
            vec![
                Instr::Load {
                    dst: Reg(3),
                    slice: Slice::param(1, 0),
                    index: R0,
                },
                Instr::Return(Reg(3)),
            ],
        )
        .verify()
        .unwrap();
        assert!(f.needs_status);
    }

    #[test]
    fn structural_errors() {
        assert_eq!(
            error(0, vec![], vec![], vec![]),
            "reachable path falls off the end of the function"
        );
        let mut bad_label = function(1, vec![], vec![], vec![Instr::Label(L0), Instr::Return(R0)]);
        bad_label.labels = 0;
        assert_eq!(
            reason(bad_label.verify().unwrap_err()),
            "invalid label index at instruction 0"
        );
        assert_eq!(
            error(
                1,
                vec![],
                vec![],
                vec![Instr::Label(L0), Instr::Label(L0), Instr::Return(R0)]
            ),
            "duplicate label Label(0) at instruction 1"
        );
        assert_eq!(
            error(
                1,
                vec![],
                vec![],
                vec![Instr::Jump(Label(1)), Instr::Label(L0)]
            ),
            "undefined label Label(1) at instruction 0"
        );
        assert_eq!(
            error(1, vec![], vec![], vec![Instr::Const(R0, Constant::from(1))]),
            "reachable path falls off the end of the function"
        );
    }

    #[test]
    fn register_errors() {
        assert_eq!(
            error(1, vec![], vec![], vec![Instr::Return(R1)]),
            "invalid register at instruction 0"
        );
        let vector = ValueType::Vector(ScalarType::I32);
        assert_eq!(
            error(
                1,
                vec![],
                vec![vector],
                vec![Instr::Copy(R1, R0), Instr::Return(R0)]
            ),
            "register copy type mismatch at instruction 0"
        );
        let index = ValueType::Scalar(ScalarType::Index);
        assert_eq!(
            typed_error(
                vec![ScalarType::Index],
                vec![BufferKind::READ_I32],
                vec![index],
                vec![
                    Instr::Copy(Reg(3), R0),
                    Instr::Copy(Reg(2), Reg(3)),
                    Instr::Return(R0)
                ]
            ),
            "a buffer parameter's registers are read-only at instruction 1"
        );
        assert_eq!(
            error(1, vec![], vec![vector], vec![Instr::Return(R1)]),
            "expected a scalar register at instruction 0"
        );
        assert_eq!(
            error(
                1,
                vec![],
                vec![vector],
                vec![
                    Instr::CallIntrinsic {
                        dst: Some(R1),
                        intrinsic: Intrinsic::Simd(Simd {
                            element: ScalarType::I32,
                            op: SimdOp::ElementWidth,
                        }),
                        args: vec![],
                    },
                    Instr::Return(R0),
                ]
            ),
            "result register type mismatch at instruction 0"
        );
        assert_eq!(
            error(
                1,
                vec![],
                vec![I32],
                vec![Instr::Bin(R0, Binary::Add, R0, R1), Instr::Return(R0)]
            ),
            "uninitialized Reg(1) at instruction 0"
        );
    }

    #[test]
    fn a_register_must_be_initialized_on_every_path_to_a_join() {
        let code = vec![
            Instr::Branch {
                cond: R0,
                yes: L0,
                no: Label(1),
            },
            Instr::Label(L0),
            Instr::Const(R1, Constant::from(1)),
            Instr::Jump(Label(2)),
            Instr::Label(Label(1)),
            Instr::Jump(Label(2)),
            Instr::Label(Label(2)),
            Instr::Return(R1),
        ];
        assert_eq!(
            typed_error(vec![ScalarType::Bool], vec![], vec![I32], code),
            "uninitialized Reg(1) at instruction 7"
        );
    }

    #[test]
    fn call_errors() {
        let callee = Arc::new(
            function(1, vec![], vec![], vec![Instr::Return(R0)])
                .verify()
                .unwrap(),
        );
        let call = |dst, args| Instr::Call {
            dst,
            function: callee.clone(),
            args,
            buffers: vec![],
        };
        assert_eq!(
            error(1, vec![], vec![], vec![call(R0, vec![]), Instr::Return(R0)]),
            "expected 1 scalar arguments, got 0 at instruction 0"
        );
        assert_eq!(
            error(
                1,
                vec![],
                vec![],
                vec![call(R1, vec![R0]), Instr::Return(R0)]
            ),
            "invalid result register at instruction 0"
        );
    }

    #[test]
    fn debug_errors() {
        for location in [
            Location { file: "", ..HERE },
            Location {
                directory: "",
                ..HERE
            },
            Location {
                file: "a\0b",
                ..HERE
            },
            Location {
                directory: "a\0b",
                ..HERE
            },
            Location { line: 0, ..HERE },
        ] {
            assert_eq!(
                error(
                    1,
                    vec![],
                    vec![],
                    vec![debug(location, vec![]), Instr::Return(R0)]
                ),
                "invalid debug source location at instruction 0"
            );
        }
        let slice = vec![Variable {
            name: "xs",
            value: Value::Slice(Slice::param(1, 0)),
        }];
        assert_eq!(
            error(
                1,
                vec![],
                vec![],
                vec![debug(HERE, slice), Instr::Return(R0)]
            ),
            "invalid debug buffer at instruction 0"
        );
    }

    #[test]
    fn buffer_errors() {
        let load = Instr::Load {
            dst: Reg(3),
            slice: Slice::param(1, 0),
            index: R0,
        };
        let store = Instr::Store {
            slice: Slice::param(1, 0),
            index: R0,
            value: R0,
        };
        let index = || vec![ScalarType::Index];
        assert_eq!(
            typed_error(
                index(),
                vec![],
                vec![I32; 3],
                vec![load.clone(), Instr::Return(Reg(3))]
            ),
            "invalid buffer at instruction 0"
        );
        assert_eq!(
            typed_error(
                index(),
                vec![BufferKind::new(ScalarType::U16, true)],
                vec![I32],
                vec![load.clone(), Instr::Return(Reg(3))]
            ),
            "result register type mismatch at instruction 0",
            "a u16 element loads into a u16 register"
        );
        assert_eq!(
            error(
                1,
                vec![BufferKind::READ_I32],
                vec![I32],
                vec![load, Instr::Return(Reg(3))]
            ),
            "expected a register of type usize, got i32 at instruction 0"
        );
        assert_eq!(
            typed_error(
                index(),
                vec![BufferKind::READ_I32],
                vec![],
                vec![store.clone(), Instr::Return(R0)]
            ),
            "store to read-only buffer at instruction 0"
        );
        assert_eq!(
            typed_error(
                index(),
                vec![BufferKind::WRITE_I32],
                vec![],
                vec![store, Instr::Return(R0)]
            ),
            "expected a register of type i32, got usize at instruction 0"
        );
    }

    /// Portable IR passes the same pairing check as a staged body.
    #[test]
    fn slice_operands_are_recorded_pairs() {
        let index = || vec![ScalarType::Index];
        let param = Slice::param(1, 0);
        let derived = Slice {
            ptr: Reg(3),
            len: Reg(4),
        };
        let range = |dst| Instr::SliceRange {
            dst,
            slice: param,
            start: R0,
            end: R0,
        };
        let load = |slice| Instr::Load {
            dst: Reg(5),
            slice,
            index: R0,
        };
        let locals = || {
            vec![
                ValueType::Ptr(BufferKind::READ_I32),
                ValueType::Scalar(ScalarType::Index),
                I32,
            ]
        };
        let forged = Slice {
            ptr: param.ptr,
            len: R0,
        };
        assert_eq!(
            typed_error(
                index(),
                vec![BufferKind::READ_I32],
                vec![I32],
                vec![
                    Instr::Load {
                        dst: Reg(3),
                        slice: forged,
                        index: R0
                    },
                    Instr::Return(Reg(3))
                ]
            ),
            "a slice operand must be a buffer parameter or the result of one slice operation \
             at instruction 0"
        );
        let paired = typed(
            index(),
            vec![BufferKind::READ_I32],
            locals(),
            vec![range(derived), load(derived), Instr::Return(Reg(5))],
        );
        assert!(paired.verify().is_ok());
        let mixed = Slice {
            ptr: derived.ptr,
            len: param.len,
        };
        assert_eq!(
            typed_error(
                index(),
                vec![BufferKind::READ_I32],
                locals(),
                vec![range(derived), load(mixed), Instr::Return(Reg(5))]
            ),
            "a slice operand must be a buffer parameter or the result of one slice operation \
             at instruction 1"
        );
        let reversed = Slice {
            ptr: Reg(4),
            len: Reg(3),
        };
        assert_eq!(
            typed_error(
                index(),
                vec![BufferKind::READ_I32],
                vec![
                    ValueType::Scalar(ScalarType::Index),
                    ValueType::Ptr(BufferKind::READ_I32),
                    I32
                ],
                vec![range(reversed), Instr::Return(R0)]
            ),
            "a slice range defines a fresh register pair at instruction 0"
        );
    }

    #[test]
    fn type_errors() {
        let ty = ValueType::Scalar;
        let (bool, i64) = (ScalarType::Bool, ScalarType::I64);
        assert_eq!(
            typed_error(
                vec![ScalarType::I32, i64],
                vec![],
                vec![I32],
                vec![Instr::Bin(Reg(2), Binary::Add, R0, R1), Instr::Return(R0)]
            ),
            "expected a register of type i32, got i64 at instruction 0"
        );
        assert_eq!(
            typed_error(
                vec![bool, bool],
                vec![],
                vec![ty(bool)],
                vec![Instr::Bin(Reg(2), Binary::Lt, R0, R1), Instr::Return(R0)]
            ),
            "Lt does not apply to bool at instruction 0"
        );
        assert_eq!(
            typed_error(
                vec![ScalarType::I32],
                vec![],
                vec![ty(bool)],
                vec![Instr::Cast { dst: R1, src: R0 }, Instr::Return(R0)]
            ),
            "Rust has no cast from i32 to bool at instruction 0"
        );
        assert_eq!(
            error(
                1,
                vec![],
                vec![],
                vec![
                    Instr::Branch {
                        cond: R0,
                        yes: L0,
                        no: L0
                    },
                    Instr::Label(L0),
                    Instr::Return(R0)
                ]
            ),
            "expected a register of type bool, got i32 at instruction 0"
        );
        assert_eq!(
            typed_error(vec![i64], vec![], vec![], vec![Instr::Return(R0)]),
            "expected a register of type i32, got i64 at instruction 0"
        );
        // A comparison produces a bool; a cast of a bool is an integer.
        typed(
            vec![i64, i64],
            vec![],
            vec![ty(bool), I32],
            vec![
                Instr::Bin(Reg(2), Binary::Lt, R0, R1),
                Instr::Cast {
                    dst: Reg(3),
                    src: Reg(2),
                },
                Instr::Return(Reg(3)),
            ],
        )
        .verify()
        .unwrap();
    }
}
