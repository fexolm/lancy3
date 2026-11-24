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
    use super::super::{Binary, Buffer, BufferKind, Label, Reg, Simd, SimdOp, ValueType};
    use super::*;
    use crate::debug::{Location, Point, Value, Variable};
    use crate::simd::Element;
    use std::sync::Arc;

    const R0: Reg = Reg(0);
    const R1: Reg = Reg(1);
    const L0: Label = Label(0);

    fn function(
        arity: usize,
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
            arity,
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
        let f = function(
            1,
            vec![BufferKind::Read],
            vec![ValueType::Scalar],
            vec![
                Instr::Load {
                    dst: R1,
                    buffer: Buffer(0),
                    index: R0,
                },
                Instr::Return(R1),
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
            error(1, vec![], vec![], vec![Instr::Const(R0, 1)]),
            "reachable path falls off the end of the function"
        );
    }

    #[test]
    fn register_errors() {
        assert_eq!(
            error(1, vec![], vec![], vec![Instr::Return(R1)]),
            "invalid register at instruction 0"
        );
        let vector = ValueType::Vector(Element::I32);
        assert_eq!(
            error(
                1,
                vec![],
                vec![vector],
                vec![Instr::Copy(R1, R0), Instr::Return(R0)]
            ),
            "register copy type mismatch at instruction 0"
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
                    Instr::Simd {
                        dst: Some(R1),
                        inst: Simd {
                            element: Element::I32,
                            op: SimdOp::Lanes,
                        },
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
                vec![ValueType::Scalar],
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
            Instr::Const(R1, 1),
            Instr::Jump(Label(2)),
            Instr::Label(Label(1)),
            Instr::Jump(Label(2)),
            Instr::Label(Label(2)),
            Instr::Return(R1),
        ];
        assert_eq!(
            error(1, vec![], vec![ValueType::Scalar], code),
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
            value: Value::Slice(Buffer(0)),
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
            dst: R0,
            buffer: Buffer(0),
            index: R0,
        };
        let store = Instr::Store {
            buffer: Buffer(0),
            index: R0,
            value: R0,
        };
        assert_eq!(
            error(1, vec![], vec![], vec![load.clone(), Instr::Return(R0)]),
            "invalid buffer at instruction 0"
        );
        assert_eq!(
            error(
                1,
                vec![BufferKind::new(Element::F32, true)],
                vec![],
                vec![load, Instr::Return(R0)]
            ),
            "scalar element access requires an i32 buffer at instruction 0"
        );
        assert_eq!(
            error(
                1,
                vec![BufferKind::Read],
                vec![],
                vec![store, Instr::Return(R0)]
            ),
            "store to read-only buffer at instruction 0"
        );
    }
}
