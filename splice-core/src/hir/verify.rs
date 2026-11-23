//! Resolve labels, check calls and register indices, and verify definite initialization.

use super::{Function, Instr, Label, ValueType};
use crate::simd::Element;
use std::collections::VecDeque;

impl Function {
    pub(super) fn verify(mut self) -> crate::Result<Self> {
        if self.register_count() > u32::MAX as usize {
            return Err(crate::BuildError::verify("too many registers"));
        }
        if self.buffer_kinds.len() > u32::MAX as usize {
            return Err(crate::BuildError::verify("too many buffer parameters"));
        }
        if self.code.is_empty() {
            return Err(crate::BuildError::verify("empty scalar function"));
        }
        let mut targets = vec![None; self.labels];
        let mut starts = Vec::new();
        let mut new_block = true;
        for (i, op) in self.code.iter().enumerate() {
            if new_block || matches!(op, Instr::Label(_)) {
                starts.push(i);
            }
            new_block = op.terminates();
            if let Instr::Label(l) = op {
                let slot = targets
                    .get_mut(l.0 as usize)
                    .ok_or_else(|| crate::BuildError::verify("invalid label index"))?;
                if slot.replace(starts.len() - 1).is_some() {
                    return Err(crate::BuildError::verify(format!("duplicate label {l:?}")));
                }
            }
            if let Instr::Call {
                dst,
                function,
                args,
                buffers,
            } = op
            {
                super::check_call(
                    function.arity(),
                    function.buffer_kinds(),
                    args.iter().map(|&r| self.checked_reg_type(r)),
                    &self.buffer_kinds,
                    buffers,
                )
                .map_err(|e| e.at(i))?;
                if self.checked_reg_type(*dst) != Some(ValueType::Scalar) {
                    return Err(crate::BuildError::verify_at(
                        i,
                        "invalid result register for call",
                    ));
                }
                continue;
            }
            if op
                .reads()
                .chain(op.written())
                .any(|r| r.0 as usize >= self.register_count())
            {
                return Err(crate::BuildError::verify_at(i, "invalid register"));
            }
            match op {
                Instr::RustContextCall {
                    dst,
                    target,
                    args,
                    slices,
                } => {
                    super::check_rust_context_call(
                        &self.buffer_kinds,
                        *target,
                        args.iter().map(|&r| self.checked_reg_type(r)),
                        slices,
                    )?;
                    if self.reg_type(*dst) != ValueType::Scalar {
                        return Err(crate::BuildError::verify(
                            "invalid Rust context operation result type",
                        ));
                    }
                }
                Instr::RustCall { dst, import, args } => {
                    import.validate(args.iter().map(|&reg| self.checked_reg_type(reg)))?;
                    if self.reg_type(*dst) != ValueType::Scalar {
                        return Err(crate::BuildError::verify("invalid Rust import result type"));
                    }
                }
                Instr::RustPacketCall { dst, import, args } => {
                    super::check_rust_packet_call(
                        import,
                        args.iter().map(|&reg| self.checked_reg_type(reg)),
                    )?;
                    if self.reg_type(*dst) != ValueType::Scalar {
                        return Err(crate::BuildError::verify("invalid Rust import result type"));
                    }
                }
                Instr::Debug(point) => {
                    let location = point.location;
                    if location.file.is_empty()
                        || location.directory.is_empty()
                        || location.file.contains('\0')
                        || location.directory.contains('\0')
                        || location.line == 0
                    {
                        return Err(crate::BuildError::verify_at(
                            i,
                            "invalid debug source location",
                        ));
                    }
                    if point.variables.iter().any(|var| {
                        matches!(var.value, crate::debug::Value::Slice(buffer)
                            if self.buffer_kinds.get(buffer.index()).is_none_or(|kind| kind.context_signature().is_some()))
                    }) {
                        return Err(crate::BuildError::verify_at(i, "invalid debug buffer"));
                    }
                }
                Instr::Simd { dst, inst, args } => {
                    let types: Vec<_> = args.iter().map(|&r| self.reg_type(r)).collect();
                    inst.validate(&types, &self.buffer_kinds)?;
                    if dst.map(|r| self.reg_type(r)) != inst.output() {
                        return Err(crate::BuildError::verify("invalid SIMD result type"));
                    }
                }
                Instr::Copy(d, s) if self.reg_type(*d) != self.reg_type(*s) => {
                    return Err(crate::BuildError::verify("register copy type mismatch"));
                }
                Instr::Copy(..) => {}
                _ if op
                    .reads()
                    .chain(op.written())
                    .any(|r| self.reg_type(r) != ValueType::Scalar) =>
                {
                    return Err(crate::BuildError::verify("expected a scalar register"));
                }
                _ => {}
            }
            if let Instr::BufferLen { buffer, .. }
            | Instr::Load { buffer, .. }
            | Instr::Store { buffer, .. } = op
            {
                let kind = self
                    .buffer_kinds
                    .get(buffer.index())
                    .ok_or_else(|| crate::BuildError::verify_at(i, "invalid buffer"))?;
                if kind.context_signature().is_some() {
                    return Err(crate::BuildError::verify(
                        "slice memory operation cannot use Rust context",
                    ));
                }
                if !matches!(op, Instr::BufferLen { .. }) && kind.element() != Element::I32 {
                    return Err(crate::BuildError::verify(
                        "scalar element access requires an i32 buffer",
                    ));
                }
                if matches!(op, Instr::Store { .. }) && !kind.writable() {
                    return Err(crate::BuildError::verify_at(i, "store to read-only buffer"));
                }
            }
        }
        starts.push(self.code.len());
        check_initialization(&self, &starts, &targets)?;
        self.needs_status = self.code.iter().any(Instr::may_trap);
        Ok(self)
    }
}

/// Forward must-analysis. A register at a join must have been assigned on
/// every reachable incoming edge, including edges from loop back to header.
fn check_initialization(
    f: &Function,
    starts: &[usize],
    targets: &[Option<usize>],
) -> crate::Result<()> {
    let target = |l: Label| {
        targets
            .get(l.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| crate::BuildError::verify(format!("undefined label {l:?}")))
    };
    // Resolve every edge, including jumps in unreachable blocks. Fallthrough
    // past the last block is an error only if execution can reach it.
    let mut edges = Vec::with_capacity(starts.len() - 1);
    for (block, &end) in starts.iter().skip(1).enumerate() {
        edges.push(match f.code[end - 1] {
            Instr::Return(_) | Instr::TrapBounds => [None, None],
            Instr::Branch { yes, no, .. } => [Some(target(yes)?), Some(target(no)?)],
            Instr::Jump(l) => [Some(target(l)?), None],
            _ => [Some(block + 1), None],
        });
    }
    let mut entry = vec![0u64; f.register_count().div_ceil(64)];
    for r in 0..f.arity {
        entry[r / 64] |= 1 << (r % 64);
    }
    let mut states = vec![None; edges.len()];
    states[0] = Some(entry);
    let mut queue = VecDeque::from([0]);
    let mut bits = Vec::new();
    while let Some(block) = queue.pop_front() {
        bits.clone_from(states[block].as_ref().unwrap());
        for pc in starts[block]..starts[block + 1] {
            let op = &f.code[pc];
            if let Some(r) = op
                .operands()
                .find(|r| bits[r.0 as usize / 64] & (1 << (r.0 % 64)) == 0)
            {
                return Err(crate::BuildError::verify_at(
                    pc,
                    format!("uninitialized {r:?}"),
                ));
            }
            if let Some(r) = op.written() {
                bits[r.0 as usize / 64] |= 1 << (r.0 % 64);
            }
        }
        for next in edges[block].into_iter().flatten() {
            if next == edges.len() {
                return Err(crate::BuildError::verify(
                    "reachable path falls off the end of the function",
                ));
            }
            if let Some(existing) = &mut states[next] {
                let mut changed = false;
                for (a, b) in existing.iter_mut().zip(&bits) {
                    let common = *a & b;
                    changed |= *a != common;
                    *a = common;
                }
                if changed {
                    queue.push_back(next);
                }
            } else {
                states[next] = Some(bits.clone());
                queue.push_back(next);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{Binary, Buffer, BufferKind, Reg, Simd, SimdOp};
    use super::*;
    use crate::debug::{Location, Point, Value, Variable};
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
            options: crate::BuildOptions::default(),
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
        assert_eq!(error(0, vec![], vec![], vec![]), "empty scalar function");
        let mut bad_label = function(1, vec![], vec![], vec![Instr::Label(L0), Instr::Return(R0)]);
        bad_label.labels = 0;
        assert_eq!(
            reason(bad_label.verify().unwrap_err()),
            "invalid label index"
        );
        assert_eq!(
            error(
                1,
                vec![],
                vec![],
                vec![Instr::Label(L0), Instr::Label(L0), Instr::Return(R0)]
            ),
            "duplicate label Label(0)"
        );
        assert_eq!(
            error(
                1,
                vec![],
                vec![],
                vec![Instr::Jump(Label(1)), Instr::Label(L0)]
            ),
            "undefined label Label(1)"
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
            "register copy type mismatch"
        );
        assert_eq!(
            error(1, vec![], vec![vector], vec![Instr::Return(R1)]),
            "expected a scalar register"
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
            "invalid SIMD result type"
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
            "invalid result register for call at instruction 0"
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
            "scalar element access requires an i32 buffer"
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
