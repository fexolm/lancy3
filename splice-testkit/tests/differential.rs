//! Operation × element × boundary data on every backend, build route and
//! build mode, checked against each other and against Rust.
use splice::frontend::{KernelBuilder, Signature};
use splice::hir::{Binary, Buffer, Reg, Simd, SimdOp};
use splice::hir::{Constant, ScalarType};
use splice::simd::Element;
use splice::{BufferArg, BufferKind, CallError, KernelScalar, Trap};
use splice_testkit::{Stage, agree};
use std::rc::Rc;

/// Vector width of the target `agree` builds for.
fn bytes() -> usize {
    splice::Target::host().unwrap().vector_bytes()
}

fn simd(b: &mut dyn KernelBuilder, element: Element, op: SimdOp, args: &[Reg]) -> Reg {
    b.simd(Simd { element, op }, args).unwrap().unwrap()
}

fn store(b: &mut dyn KernelBuilder, element: Element, buffer: u32, at: Reg, value: Reg) {
    b.simd(
        Simd {
            element,
            op: SimdOp::Store(Buffer(buffer)),
        },
        &[at, value],
    )
    .unwrap();
}

fn load(b: &mut dyn KernelBuilder, element: Element, buffer: u32, at: Reg) -> Reg {
    simd(b, element, SimdOp::Load(Buffer(buffer)), &[at])
}

/// Boundary values of each scalar type.
fn edges(ty: ScalarType) -> Vec<Constant> {
    fn all<T: Into<Constant>>(values: impl IntoIterator<Item = T>) -> Vec<Constant> {
        values.into_iter().map(Into::into).collect()
    }
    match ty {
        ScalarType::Bool => all([false, true]),
        ScalarType::I32 => all([0, 1, -1, 31, 32, -32, i32::MIN, i32::MAX]),
        ScalarType::U32 => all([0, 1, 31, 32, 33, 1 << 31, (1 << 31) - 1, u32::MAX]),
        ScalarType::I64 => all([0, 1, -1, 63, 64, 1 << 32, i64::MIN, i64::MAX]),
        ScalarType::U64 => all([0, 1, 63, 64, 1 << 32, 1 << 63, (1 << 63) - 1, u64::MAX]),
        ScalarType::Index => all([0, 1, 63, 64, 1 << 32, 1 << 63, (1 << 63) - 1, usize::MAX]),
        ScalarType::F32 => all([
            0.0,
            -0.0,
            1.5,
            -2.0,
            3e9,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ]),
        ScalarType::F64 => all([
            0.0,
            -0.0,
            1.5,
            -2.0,
            1e19,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ]),
    }
}

/// The buffer element that holds a scalar type: a `bool` is an `i32`
/// word, an unsigned 64-bit integer the bits of an `i64`.
fn storage(ty: ScalarType) -> Element {
    match ty {
        ScalarType::Bool => Element::I32,
        ScalarType::U64 | ScalarType::Index => Element::I64,
        ty => ty.element().expect("a buffer element"),
    }
}

/// Typed slices for `values`, stored as `storage(ty)`.
enum Column {
    I32(Vec<i32>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Column {
    fn new(ty: ScalarType, values: impl IntoIterator<Item = Constant>) -> Self {
        let bits = values.into_iter().map(Constant::bits);
        match storage(ty) {
            Element::I32 => Self::I32(bits.map(|b| b as u32 as i32).collect()),
            Element::U32 => Self::U32(bits.map(|b| b as u32).collect()),
            Element::I64 => Self::I64(bits.map(|b| b as i64).collect()),
            Element::F32 => Self::F32(bits.map(|b| f32::from_bits(b as u32)).collect()),
            _ => Self::F64(bits.map(f64::from_bits).collect()),
        }
    }
    fn read(&self) -> BufferArg<'_> {
        match self {
            Self::I32(v) => BufferArg::read(v),
            Self::U32(v) => BufferArg::read(v),
            Self::I64(v) => BufferArg::read(v),
            Self::F32(v) => BufferArg::read(v),
            Self::F64(v) => BufferArg::read(v),
        }
    }
    fn write(&mut self) -> BufferArg<'_> {
        match self {
            Self::I32(v) => BufferArg::write(v),
            Self::U32(v) => BufferArg::write(v),
            Self::I64(v) => BufferArg::write(v),
            Self::F32(v) => BufferArg::write(v),
            Self::F64(v) => BufferArg::write(v),
        }
    }
    /// The first value as `ty`. Every NaN prints alike: the reference
    /// does not define NaN payloads.
    fn first(&self, ty: ScalarType) -> String {
        let bits = match self {
            Self::I32(v) => u64::from(v[0] as u32),
            Self::U32(v) => u64::from(v[0]),
            Self::I64(v) => v[0] as u64,
            Self::F32(v) if v[0].is_nan() => return "NaN".into(),
            Self::F32(v) => u64::from(v[0].to_bits()),
            Self::F64(v) if v[0].is_nan() => return "NaN".into(),
            Self::F64(v) => v[0].to_bits(),
        };
        show(Constant::new(ty, bits))
    }
}

fn show(value: Constant) -> String {
    let nan = match value.ty() {
        ScalarType::F32 => f32::from_bits(value.bits() as u32).is_nan(),
        ScalarType::F64 => f64::from_bits(value.bits()).is_nan(),
        _ => false,
    };
    if nan { "NaN".into() } else { value.to_string() }
}

/// Read `inputs` elements of type `ty` from buffer 0 at
/// `inputs * arg(0) + i`, as registers of type `ty`.
fn read_operands(
    b: &mut dyn KernelBuilder,
    ty: ScalarType,
    inputs: usize,
) -> splice::Result<Vec<Reg>> {
    let case = b.arg(0)?;
    let case = b.cast(case, ScalarType::Index)?;
    let width = b.constant(Constant::from(inputs))?;
    let first = b.binary(Binary::Mul, case, width)?;
    (0..inputs)
        .map(|i| {
            let offset = b.constant(Constant::from(i))?;
            let at = b.binary(Binary::Add, first, offset)?;
            let word = b.load(Buffer(0), at)?;
            match ty {
                ScalarType::Bool => {
                    let zero = b.constant(Constant::from(0))?;
                    b.binary(Binary::Ne, word, zero)
                }
                ScalarType::U64 | ScalarType::Index => b.cast(word, ty),
                _ => Ok(word),
            }
        })
        .collect()
}

/// Store `value`, a register of type `ty`, at element 0 of buffer 1 and return 0.
fn write_result(b: &mut dyn KernelBuilder, ty: ScalarType, value: Reg) -> splice::Result<()> {
    let value = match ty {
        ScalarType::Bool => b.cast(value, ScalarType::I32)?,
        ScalarType::U64 | ScalarType::Index => b.cast(value, ScalarType::I64)?,
        _ => value,
    };
    let zero = b.constant(Constant::from(0usize))?;
    b.store(Buffer(1), zero, value)?;
    let zero = b.constant(Constant::from(0))?;
    b.ret(zero)
}

/// Run `stage` on every row of `rows` (`inputs` operands of type `from`)
/// and compare each result of type `to`, or its trap, with `expected`.
fn check_rows(
    name: &str,
    stage: &Stage,
    (from, to): (ScalarType, ScalarType),
    rows: &[Vec<Constant>],
    expected: impl Fn(&[Constant]) -> Result<Constant, Trap>,
) {
    let kinds = [
        BufferKind::new(storage(from), false),
        BufferKind::new(storage(to), true),
    ];
    let signature = Signature::new(&[ScalarType::I32], ScalarType::I32, &kinds);
    let data = Column::new(from, rows.iter().flatten().copied());
    let seen = agree(name, signature, stage, |call| {
        (0..rows.len())
            .map(|case| {
                let mut out = Column::new(to, [Constant::new(to, 0)]);
                match call(&[case as u64], &mut [data.read(), out.write()]) {
                    Ok(_) => out.first(to),
                    Err(error) => format!("{error:?}"),
                }
            })
            .collect()
    });
    let want: Vec<_> = rows
        .iter()
        .map(|row| match expected(row) {
            Ok(value) => show(value),
            Err(trap) => format!("{:?}", CallError::from(trap)),
        })
        .collect();
    for ((row, seen), want) in rows.iter().zip(&seen).zip(&want) {
        assert_eq!(seen, want, "{name} {row:?}");
    }
}

/// The kernel code for one row's operands.
type RowBody<'a> = dyn Fn(&mut dyn KernelBuilder, &[Reg]) -> splice::Result<()> + 'a;

/// Stage `body` once per row of `rows`, selected by `arg(0)`, on the row's
/// operands: loaded as `read_operands` does, except where `known` is set,
/// where the operand is the row's value as a constant, for the backends to
/// fold. Each case returns through `body`.
fn each_row(
    b: &mut dyn KernelBuilder,
    ty: ScalarType,
    rows: &[Vec<Constant>],
    known: &[bool],
    body: &RowBody<'_>,
) -> splice::Result<()> {
    let loaded = read_operands(b, ty, known.len())?;
    let case = b.arg(0)?;
    for (i, row) in rows.iter().enumerate() {
        let (here, next) = (b.label()?, b.label()?);
        let index = b.constant(Constant::from(i as i32))?;
        let selected = b.binary(Binary::Eq, case, index)?;
        b.branch(selected, here, next)?;
        b.mark(here)?;
        let operands = row
            .iter()
            .zip(&loaded)
            .zip(known)
            .map(|((&value, &load), &known)| if known { b.constant(value) } else { Ok(load) })
            .collect::<splice::Result<Vec<_>>>()?;
        body(b, &operands)?;
        b.mark(next)?;
    }
    b.trap_bounds()
}

/// The result word of a call, or its error. A word with bits above its
/// value is shown as such: the entry must zero them.
fn show_word(ty: ScalarType, result: Result<u64, CallError>) -> String {
    match result {
        Ok(word) if Constant::new(ty, word).bits() == word => show(Constant::new(ty, word)),
        Ok(word) => format!("unnormalized {ty} word {word:#x}"),
        Err(error) => format!("{error:?}"),
    }
}

/// Every scalar type crosses the entry as one word at its own position:
/// with one parameter of each type, the kernel returns parameter `k` (or,
/// with a slice after the scalars, also through the descriptor path).
#[test]
fn every_scalar_type_passes_into_and_out_of_the_entry_at_its_position() {
    let types = ScalarType::ALL;
    let row = |i: usize| -> Vec<Constant> {
        types
            .iter()
            .enumerate()
            .map(|(k, &ty)| {
                let values = edges(ty);
                values[(i + k) % values.len()]
            })
            .collect()
    };
    let rows: Vec<_> = (0..8).map(row).collect();
    let slices = [BufferKind::new(Element::I32, false)];
    for buffers in [&[][..], &slices[..]] {
        for (k, &ty) in types.iter().enumerate() {
            let stage: Stage = Rc::new(move |b| b.ret(b.arg(k)?));
            let seen = agree(
                &format!("pass {ty} with {} slices", buffers.len()),
                Signature::new(&types, ty, buffers),
                &stage,
                |call| {
                    rows.iter()
                        .map(|row| {
                            let words: Vec<u64> = row.iter().map(|c| c.bits()).collect();
                            let mut slice: Vec<_> = buffers
                                .iter()
                                .map(|_| BufferArg::read::<i32>(&[1, 2]))
                                .collect();
                            show_word(ty, call(&words, &mut slice))
                        })
                        .collect()
                },
            );
            let want: Vec<_> = rows.iter().map(|row| show(row[k])).collect();
            assert_eq!(seen, want, "{ty}");
        }
    }
}

/// The shared call path describes up to 8 slices on the stack and more on
/// the heap, and Lancy packs up to 32 words on the stack: each slice and
/// the scalar must arrive at its position on every side of both limits.
#[test]
fn every_slice_count_reaches_the_kernel_at_its_position() {
    for count in [0, 1, 8, 9, 15, 16, 17] {
        let kinds = vec![BufferKind::new(Element::I32, count % 2 == 0); count];
        let kinds: &'static [BufferKind] = Box::leak(kinds.into_boxed_slice());
        let stage: Stage = Rc::new(move |b| {
            // `arg(0) + Σ (i + 1)·xs_i[i]`, stored in the last writable slice.
            let mut total = b.arg(0)?;
            for i in 0..count {
                let at = b.constant(Constant::from(i))?;
                let x = b.load(Buffer(i as u32), at)?;
                let weight = b.constant(Constant::from(i as i32 + 1))?;
                let x = b.binary(Binary::Mul, x, weight)?;
                total = b.binary(Binary::Add, total, x)?;
            }
            if let Some(last) = (0..count).rev().find(|i| kinds[*i].writable()) {
                let at = b.constant(Constant::from(0usize))?;
                b.store(Buffer(last as u32), at, total)?;
            }
            b.ret(total)
        });
        let data: Vec<Vec<i32>> = (0..count)
            .map(|i| (0..count as i32).map(|k| 10 * i as i32 + k).collect())
            .collect();
        let seen = agree(
            &format!("{count} slices"),
            Signature::new(&[ScalarType::I32], ScalarType::I32, kinds),
            &stage,
            |call| {
                let mut data = data.clone();
                let mut buffers: Vec<BufferArg<'_>> = data
                    .iter_mut()
                    .zip(kinds)
                    .map(|(xs, kind)| {
                        if kind.writable() {
                            BufferArg::write(xs.as_mut_slice())
                        } else {
                            BufferArg::read(xs.as_slice())
                        }
                    })
                    .collect();
                let result = show_word(ScalarType::I32, call(&[7], &mut buffers));
                drop(buffers);
                vec![result, format!("{data:?}")]
            },
        );
        let total = (0..count).fold(7i32, |t, i| t + (i as i32 + 1) * data[i][i]);
        assert_eq!(seen[0], show(Constant::from(total)), "{count} slices");
    }
}

/// Two parameters of one type combine natively into a result of that type
/// (`Add`, or `Xor` for `bool`), against `checked_eval`.
#[test]
fn every_scalar_type_computes_on_entry_words_and_returns_one() {
    for ty in ScalarType::ALL {
        let op = if ty == ScalarType::Bool {
            Binary::Xor
        } else {
            Binary::Add
        };
        let stage: Stage = Rc::new(move |b| {
            let r = b.binary(op, b.arg(0)?, b.arg(1)?)?;
            b.ret(r)
        });
        let values = edges(ty);
        let pairs: Vec<_> = values
            .iter()
            .flat_map(|&a| values.iter().map(move |&z| (a, z)))
            .collect();
        let seen = agree(
            &format!("{ty} {op:?} of entry words"),
            Signature::new(&[ty; 2], ty, &[]),
            &stage,
            |call| {
                pairs
                    .iter()
                    .map(|(a, z)| show_word(ty, call(&[a.bits(), z.bits()], &mut [])))
                    .collect()
            },
        );
        let want: Vec<_> = pairs
            .iter()
            .map(|&(a, z)| show(op.checked_eval(a, z).unwrap()))
            .collect();
        assert_eq!(seen, want, "{ty}");
    }
}

/// A word with bits above its type's value is rejected before the call on
/// every backend, as is a wrong number of words.
#[test]
fn entry_words_must_hold_exactly_their_value() {
    use splice::SignatureError;
    let types = [ScalarType::Bool, ScalarType::I32, ScalarType::F32];
    let stage: Stage = Rc::new(|b| {
        let r = b.constant(Constant::from(7))?;
        b.ret(r)
    });
    let calls: [&[u64]; 5] = [
        &[1, u64::from(u32::MAX), u64::from(1f32.to_bits())],
        &[2, 0, 0],
        &[0, 1 << 32, 0],
        &[0, 0, u64::MAX],
        &[0, 0],
    ];
    let seen = agree(
        "entry words",
        Signature::new(&types, ScalarType::I32, &[]),
        &stage,
        |call| {
            calls
                .iter()
                .map(|words| format!("{:?}", call(words, &mut [])))
                .collect()
        },
    );
    let word = |index: usize| {
        CallError::from(SignatureError::Word {
            index,
            ty: types[index],
        })
    };
    let want = [
        Ok(7),
        Err(word(0)),
        Err(word(1)),
        Err(word(2)),
        Err(CallError::from(SignatureError::Scalars {
            expected: 3,
            actual: 2,
        })),
    ];
    assert_eq!(seen, want.map(|r| format!("{r:?}")));
}

#[test]
fn every_scalar_binary_matches_checked_eval_at_every_type() {
    for ty in ScalarType::ALL {
        let values = edges(ty);
        let rows: Vec<_> = values
            .iter()
            .flat_map(|&a| values.iter().map(move |&z| vec![a, z]))
            .collect();
        for op in Binary::ALL {
            let Some(output) = op.output(ty) else {
                continue;
            };
            let stage: Stage = Rc::new(move |b| {
                let operands = read_operands(b, ty, 2)?;
                let r = b.binary(op, operands[0], operands[1])?;
                write_result(b, output, r)
            });
            check_rows(
                &format!("{ty} {op:?}"),
                &stage,
                (ty, output),
                &rows,
                |row| op.checked_eval(row[0], row[1]),
            );
        }
    }
}

#[test]
fn every_scalar_cast_matches_rust_as() {
    for from in ScalarType::ALL {
        let rows: Vec<_> = edges(from).into_iter().map(|v| vec![v]).collect();
        for to in ScalarType::ALL.into_iter().filter(|&to| from.casts_to(to)) {
            let stage: Stage = Rc::new(move |b| {
                let operand = read_operands(b, from, 1)?[0];
                let r = b.cast(operand, to)?;
                write_result(b, to, r)
            });
            check_rows(
                &format!("{from} as {to}"),
                &stage,
                (from, to),
                &rows,
                |row| Ok(row[0].cast(to).expect("a Rust cast")),
            );
        }
    }
}

/// `every_scalar_binary_matches_checked_eval_at_every_type` with constant
/// operands, which each backend may fold at build time: both constant, or
/// one constant and one loaded.
#[test]
fn every_scalar_binary_folds_constant_operands_like_checked_eval() {
    for ty in ScalarType::ALL {
        let values = edges(ty);
        let rows: Rc<Vec<_>> = Rc::new(
            values
                .iter()
                .flat_map(|&a| values.iter().map(move |&z| vec![a, z]))
                .collect(),
        );
        for op in Binary::ALL {
            let Some(output) = op.output(ty) else {
                continue;
            };
            for known in [[true, true], [false, true], [true, false]] {
                let cases = Rc::clone(&rows);
                let stage: Stage = Rc::new(move |b| {
                    each_row(b, ty, &cases, &known, &|b, operands| {
                        let r = b.binary(op, operands[0], operands[1])?;
                        write_result(b, output, r)
                    })
                });
                check_rows(
                    &format!("{ty} {op:?} with known {known:?}"),
                    &stage,
                    (ty, output),
                    &rows,
                    |row| op.checked_eval(row[0], row[1]),
                );
            }
        }
    }
}

/// `every_scalar_cast_matches_rust_as` with a constant source.
#[test]
fn every_scalar_cast_folds_a_constant_like_rust_as() {
    for from in ScalarType::ALL {
        let rows: Rc<Vec<_>> = Rc::new(edges(from).into_iter().map(|v| vec![v]).collect());
        for to in ScalarType::ALL.into_iter().filter(|&to| from.casts_to(to)) {
            let cases = Rc::clone(&rows);
            let stage: Stage = Rc::new(move |b| {
                each_row(b, from, &cases, &[true], &|b, operands| {
                    let r = b.cast(operands[0], to)?;
                    write_result(b, to, r)
                })
            });
            check_rows(
                &format!("constant {from} as {to}"),
                &stage,
                (from, to),
                &rows,
                |row| Ok(row[0].cast(to).expect("a Rust cast")),
            );
        }
    }
}

/// A Rust import that returns `value` for `fail == 0` and panics otherwise.
/// A panic is the only trap Rust code reports: every call goes through the
/// packet entry, which owns the status.
fn import(args: splice::__private::ContextArguments<'_>) -> u64 {
    let (value, fail) = (args.word(0), args.word(1));
    assert!(fail == 0, "import failed on {}", value as i32);
    value
}

static IMPORT: splice::__private::StaticRustSymbol =
    splice::__private::StaticRustSymbol::new("import", 2, 0, import);

#[test]
fn every_backend_reports_the_same_typed_trap() {
    // trap(kind, value): 0 bounds, 1 division by zero, 2 division overflow,
    // 3 overflow, 4 Rust panic; otherwise value through the import.
    let stage: Stage = Rc::new(|b| {
        let kind = b.arg(0)?;
        let value = b.arg(1)?;
        let exits = (0..5)
            .map(|_| b.label())
            .collect::<splice::Result<Vec<_>>>()?;
        for (case, &exit) in exits.iter().enumerate() {
            let next = b.label()?;
            let k = b.constant(Constant::from(case as i32))?;
            let hit = b.binary(Binary::Eq, kind, k)?;
            b.branch(hit, exit, next)?;
            b.mark(next)?;
        }
        let import = splice::hir::RustTarget::Static(&IMPORT);
        let zero = b.constant(Constant::from(0))?;
        let passed = b.rust_call(import, &[value, zero], &[], ScalarType::I32)?;
        b.ret(passed)?;
        let operations = [
            (Binary::Div, 1, 0),
            (Binary::Rem, i32::MIN, -1),
            (Binary::CheckedMul, i32::MAX, 2),
        ];
        b.mark(exits[0])?;
        let at = b.cast(value, ScalarType::Index)?;
        let read = b.load(Buffer(0), at)?;
        b.ret(read)?;
        for (&exit, (op, x, y)) in exits[1..4].iter().zip(&operations) {
            b.mark(exit)?;
            let (x, y) = (
                b.constant(Constant::from(*x))?,
                b.constant(Constant::from(*y))?,
            );
            let r = b.binary(*op, x, y)?;
            b.ret(r)?;
        }
        b.mark(exits[4])?;
        let hundred = b.constant(Constant::from(100))?;
        let fail = b.binary(Binary::Sub, value, hundred)?;
        let r = b.rust_call(import, &[value, fail], &[], ScalarType::I32)?;
        b.ret(r)?;
        Ok(())
    });
    let cases = [
        (0, 3),
        (0, 4),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 105),
        (4, 100),
        (9, 7),
    ];
    let kinds = [BufferKind::new(Element::I32, false)];
    let seen = agree(
        "traps",
        Signature::new(&[ScalarType::I32; 2], ScalarType::I32, &kinds),
        &stage,
        |call| {
            cases
                .iter()
                .map(|&(kind, value)| {
                    format!(
                        "{:?}",
                        call(&[kind, value], &mut [BufferArg::from(&[5, 6, 7, 8][..])])
                    )
                })
                .collect()
        },
    );
    let expected = [
        Ok(8),
        Err(CallError::Trap(Trap::Bounds)),
        Err(CallError::Trap(Trap::DivisionByZero)),
        Err(CallError::Trap(Trap::DivisionOverflow)),
        Err(CallError::Trap(Trap::Overflow)),
        Err(CallError::RustPanic("import failed on 105".into())),
        Ok(100),
        Ok(7),
    ]
    .map(|result: Result<i32, CallError>| format!("{result:?}"));
    assert_eq!(seen, expected);
}

/// Return 0 from the kernel.
fn done(b: &mut dyn KernelBuilder) -> splice::Result<()> {
    let zero = b.constant(Constant::from(0))?;
    b.ret(zero)
}

/// Scalar argument `i` as a slice index, as Rust's `i as usize`.
fn index(b: &mut dyn KernelBuilder, i: usize) -> splice::Result<Reg> {
    let word = b.arg(i)?;
    b.cast(word, ScalarType::Index)
}

/// Load `extra + 1` vectors of `from` (at 0 and at each argument), apply
/// `op`, store the result as `to`.
fn unary(from: Element, to: Element, op: SimdOp, extra: usize) -> Stage {
    Rc::new(move |b| {
        let zero = b.constant(Constant::from(0usize))?;
        let mut args = vec![load(b, from, 0, zero)];
        for i in 0..extra {
            let at = index(b, i)?;
            args.push(load(b, from, 0, at));
        }
        let r = simd(b, from, op, &args);
        store(b, to, 1, zero, r);
        done(b)?;
        Ok(())
    })
}

fn kinds(from: Element, to: Element) -> [BufferKind; 2] {
    [BufferKind::new(from, false), BufferKind::new(to, true)]
}

#[test]
fn float_to_integer_casts_saturate_like_rust() {
    let n = Element::F32.lanes(bytes());
    let src: Vec<f32> = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        3e9,
        -3e9,
        -1.5,
        2.5,
        -0.0,
        4294967040.0,
        1e-40,
        2147483520.0,
        2147483648.0,
        -2147483904.0,
        0.5,
        65536.0,
        -1.0,
    ]
    .into_iter()
    .cycle()
    .take(n)
    .collect();
    let stage = unary(Element::F32, Element::I32, SimdOp::Cast(Element::I32), 0);
    let seen = agree(
        "f32 -> i32",
        Signature::new(
            &[ScalarType::I32; 0],
            ScalarType::I32,
            &kinds(Element::F32, Element::I32),
        ),
        &stage,
        |call| {
            let mut out = vec![0i32; n];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {out:?}")]
        },
    );
    let expected: Vec<i32> = src.iter().map(|&x| x as i32).collect();
    assert_eq!(seen, [format!("Ok(0) {expected:?}")]);

    let stage = unary(Element::F32, Element::U32, SimdOp::Cast(Element::U32), 0);
    let seen = agree(
        "f32 -> u32",
        Signature::new(
            &[ScalarType::I32; 0],
            ScalarType::I32,
            &kinds(Element::F32, Element::U32),
        ),
        &stage,
        |call| {
            let mut out = vec![0u32; n];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {out:?}")]
        },
    );
    let expected: Vec<u32> = src.iter().map(|&x| x as u32).collect();
    assert_eq!(seen, [format!("Ok(0) {expected:?}")]);

    let m = Element::F64.lanes(bytes());
    let src: Vec<f64> = [
        f64::NAN,
        1e19,
        -1e19,
        -1.5,
        9.3e18,
        2.5,
        f64::INFINITY,
        -0.0,
    ]
    .into_iter()
    .cycle()
    .take(m)
    .collect();
    let stage = unary(Element::F64, Element::I64, SimdOp::Cast(Element::I64), 0);
    let seen = agree(
        "f64 -> i64",
        Signature::new(
            &[ScalarType::I32; 0],
            ScalarType::I32,
            &kinds(Element::F64, Element::I64),
        ),
        &stage,
        |call| {
            let mut out = vec![0i64; m];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {out:?}")]
        },
    );
    let expected: Vec<i64> = src.iter().map(|&x| x as i64).collect();
    assert_eq!(seen, [format!("Ok(0) {expected:?}")]);
}

#[test]
fn integer_to_float_casts_round_like_rust() {
    let m = Element::I64.lanes(bytes());
    let src: Vec<i64> = [
        i64::MAX,
        i64::MIN,
        -1,
        1 << 53 | 1,
        3,
        -(1 << 60) - 1,
        0,
        12345,
    ]
    .into_iter()
    .cycle()
    .take(m)
    .collect();
    let stage = unary(Element::I64, Element::F64, SimdOp::Cast(Element::F64), 0);
    let seen = agree(
        "i64 -> f64",
        Signature::new(
            &[ScalarType::I32; 0],
            ScalarType::I32,
            &kinds(Element::I64, Element::F64),
        ),
        &stage,
        |call| {
            let mut out = vec![0f64; m];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {:?}", out.map_bits())]
        },
    );
    let expected: Vec<f64> = src.iter().map(|&x| x as f64).collect();
    assert_eq!(seen, [format!("Ok(0) {:?}", expected.map_bits())]);

    let n = Element::U32.lanes(bytes());
    let src: Vec<u32> = [
        u32::MAX,
        0x8000_0001,
        16777217,
        0,
        1,
        0xFFFF_FF7F,
        3_000_000_000,
        7,
    ]
    .into_iter()
    .cycle()
    .take(n)
    .collect();
    let stage = unary(Element::U32, Element::F32, SimdOp::Cast(Element::F32), 0);
    let seen = agree(
        "u32 -> f32",
        Signature::new(
            &[ScalarType::I32; 0],
            ScalarType::I32,
            &kinds(Element::U32, Element::F32),
        ),
        &stage,
        |call| {
            let mut out = vec![0f32; n];
            let r = call(
                &[],
                &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
            );
            vec![format!("{r:?} {:?}", out.map_bits())]
        },
    );
    let expected: Vec<f32> = src.iter().map(|&x| x as f32).collect();
    assert_eq!(seen, [format!("Ok(0) {:?}", expected.map_bits())]);
}

trait Bits {
    fn map_bits(&self) -> Vec<u64>;
}
impl Bits for Vec<f32> {
    fn map_bits(&self) -> Vec<u64> {
        self.iter().map(|x| x.to_bits().into()).collect()
    }
}
impl Bits for Vec<f64> {
    fn map_bits(&self) -> Vec<u64> {
        self.iter().map(|x| x.to_bits()).collect()
    }
}

#[test]
fn float_arithmetic_is_correctly_rounded_on_special_values() {
    let n = Element::F32.lanes(bytes());
    let src: Vec<f32> = [0.1f32, -4.0, 1e30, -0.0, 3.0, 16777217.0, 1.0000001, 7.0]
        .into_iter()
        .cycle()
        .take(n + 2)
        .collect();
    let reference = |op, i: usize| -> f32 {
        let (x, y, z) = (src[i], src[i + 1], src[i + 2]);
        match op {
            SimdOp::Sqrt => x.sqrt(),
            SimdOp::Div => x / y,
            _ => x.mul_add(y, z),
        }
    };
    for (op, extra) in [(SimdOp::Sqrt, 0), (SimdOp::Div, 1), (SimdOp::MulAdd, 2)] {
        let stage = unary(Element::F32, Element::F32, op, extra);
        let seen = agree(
            &format!("{op:?}"),
            Signature::new(
                &vec![ScalarType::I32; extra],
                ScalarType::I32,
                &kinds(Element::F32, Element::F32),
            ),
            &stage,
            |call| {
                let mut out = vec![0f32; n];
                let args: Vec<u64> = (1..=extra as u64).collect();
                let r = call(
                    &args,
                    &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
                );
                vec![format!("{r:?} {:?}", out.map_bits())]
            },
        );
        let expected: Vec<f32> = (0..n).map(|i| reference(op, i)).collect();
        assert_eq!(seen, [format!("Ok(0) {:?}", expected.map_bits())], "{op:?}");
    }
}

#[test]
fn float_comparisons_follow_ieee_unordered_rules() {
    let n = Element::F32.lanes(bytes());
    let src: Vec<f32> = [f32::NAN, 1.0, f32::NAN, 2.0, 1.0, -0.0, 0.0, f32::INFINITY]
        .into_iter()
        .cycle()
        .take(n + 1)
        .collect();
    use SimdOp::*;
    for (op, rust) in [
        (Eq, f32::eq as fn(&f32, &f32) -> bool),
        (Ne, f32::ne),
        (Lt, f32::lt),
        (Le, f32::le),
        (Gt, f32::gt),
        (Ge, f32::ge),
    ] {
        let stage: Stage = Rc::new(move |b| {
            let (zero, one) = (
                b.constant(Constant::from(0usize))?,
                b.constant(Constant::from(1usize))?,
            );
            let x = load(b, Element::F32, 0, zero);
            let y = load(b, Element::F32, 0, one);
            let m = simd(b, Element::F32, op, &[x, y]);
            let yes = simd(b, Element::F32, Splat(1f32.to_bits().into()), &[]);
            let no = simd(b, Element::F32, Splat(0), &[]);
            let s = simd(b, Element::F32, Select, &[m, yes, no]);
            store(b, Element::F32, 1, zero, s);
            done(b)?;
            Ok(())
        });
        let seen = agree(
            &format!("{op:?}"),
            Signature::new(
                &[ScalarType::I32; 0],
                ScalarType::I32,
                &kinds(Element::F32, Element::F32),
            ),
            &stage,
            |call| {
                let mut out = vec![9f32; n];
                let r = call(
                    &[],
                    &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
                );
                vec![format!("{r:?} {out:?}")]
            },
        );
        let expected: Vec<f32> = (0..n)
            .map(|i| f32::from(u8::from(rust(&src[i], &src[i + 1]))))
            .collect();
        assert_eq!(seen, [format!("Ok(0) {expected:?}")], "{op:?}");
    }
}

#[test]
fn integer_lane_operations_agree_on_boundary_values() {
    let n = Element::I32.lanes(bytes());
    let src: Vec<i32> = (0..2 * n as i32)
        .map(|i| {
            [
                i32::MIN,
                -1,
                0,
                1,
                31,
                32,
                -33,
                i32::MAX,
                100,
                -100,
                7,
                15,
                16,
                17,
                64,
                -64,
            ][i as usize % 16]
                .wrapping_add(i / 16)
        })
        .collect();
    use SimdOp::*;
    for element in [Element::I32, Element::U32] {
        for (op, extra) in [
            (Shr(5), 0),
            (Shr(37), 0),
            (RotateLeft(7), 0),
            (RotateLeft(33), 0),
            (Min, 1),
            (Max, 1),
            (Swizzle, 1),
            (PrefixSum, 0),
            (Mul, 1),
            (Not, 0),
        ] {
            if matches!(op, Swizzle | PrefixSum) && element != Element::I32 {
                continue;
            }
            let stage = unary(element, element, op, extra);
            agree(
                &format!("{element:?} {op:?}"),
                Signature::new(
                    &vec![ScalarType::I32; extra],
                    ScalarType::I32,
                    &kinds(element, element),
                ),
                &stage,
                |call| {
                    [1, 3, n as i32 - 1]
                        .into_iter()
                        .map(|offset| {
                            let args = if extra == 1 {
                                vec![offset.to_word()]
                            } else {
                                vec![]
                            };
                            let mut out = vec![0i32; n];
                            let r = if element == Element::I32 {
                                call(
                                    &args,
                                    &mut [BufferArg::from(&src[..]), BufferArg::from(&mut out[..])],
                                )
                            } else {
                                let src: Vec<u32> = src.iter().map(|&x| x as u32).collect();
                                let mut wide = vec![0u32; n];
                                let r = call(
                                    &args,
                                    &mut [
                                        BufferArg::from(&src[..]),
                                        BufferArg::from(&mut wide[..]),
                                    ],
                                );
                                out = wide.iter().map(|&x| x as i32).collect();
                                r
                            };
                            format!("{r:?} {out:?}")
                        })
                        .collect()
                },
            );
        }
    }
}

#[test]
fn first_n_masks_saturate_at_both_ends() {
    use SimdOp::*;
    let stage: Stage = Rc::new(|b| {
        let k = index(b, 0)?;
        let m = simd(b, Element::U16, FirstN, &[k]);
        let ones = simd(b, Element::U16, Splat(1), &[]);
        let zeros = simd(b, Element::U16, Splat(0), &[]);
        let s = simd(b, Element::U16, Select, &[m, ones, zeros]);
        let zero = b.constant(Constant::from(0usize))?;
        store(b, Element::U16, 0, zero, s);
        let m = simd(b, Element::I64, FirstN, &[k]);
        let any = simd(b, Element::I64, Any, &[m]);
        let any = b.cast(any, ScalarType::I32)?;
        let all = simd(b, Element::I64, All, &[m]);
        let all = b.cast(all, ScalarType::I32)?;
        let r = b.binary(Binary::Add, any, all)?;
        let m = simd(b, Element::I32, FirstN, &[k]);
        let count = simd(b, Element::I32, Count, &[m]);
        let ten = b.constant(Constant::from(10))?;
        let count = b.binary(Binary::Mul, count, ten)?;
        let r = b.binary(Binary::Add, r, count)?;
        b.ret(r)?;
        Ok(())
    });
    let n = Element::U16.lanes(bytes());
    let seen = agree(
        "first_n",
        Signature::new(
            &[ScalarType::I32; 1],
            ScalarType::I32,
            &[BufferKind::new(Element::U16, true)],
        ),
        &stage,
        |call| {
            [
                0,
                1,
                3,
                8,
                31,
                32,
                33,
                65535,
                65536,
                65537,
                -1,
                i32::MIN,
                i32::MAX,
            ]
            .into_iter()
            .map(|k| {
                let mut out = vec![7u16; n];
                let r = call(&[k.to_word()], &mut [BufferArg::from(&mut out[..])]);
                format!("{r:?} {out:?}")
            })
            .collect()
        },
    );
    let expected: Vec<_> = [
        0,
        1,
        3,
        8,
        31,
        32,
        33,
        65535,
        65536,
        65537,
        -1,
        i32::MIN,
        i32::MAX,
    ]
    .into_iter()
    .map(|k: i32| {
        // The count is `k as usize`.
        let active = |lanes: usize| (k as usize).min(lanes);
        let i64s = Element::I64.lanes(bytes());
        let any = i32::from(active(i64s) > 0);
        let all = i32::from(active(i64s) == i64s);
        let count = active(Element::I32.lanes(bytes())) as i32;
        let out: Vec<u16> = (0..n).map(|i| u16::from(i < active(n))).collect();
        format!("Ok({}) {out:?}", any + all + 10 * count)
    })
    .collect();
    assert_eq!(seen, expected);
}

#[test]
fn compress_expand_and_compress_store_agree_at_buffer_edges() {
    let n = Element::I32.lanes(bytes());
    for op in [SimdOp::Compress, SimdOp::Expand] {
        let stage: Stage = Rc::new(move |b| {
            let (zero, one) = (
                b.constant(Constant::from(0usize))?,
                b.constant(Constant::from(1usize))?,
            );
            let e = Element::I32;
            let x = load(b, e, 0, zero);
            let flags = load(b, e, 1, zero);
            let z = simd(b, e, SimdOp::Splat(0), &[]);
            let m = simd(b, e, SimdOp::Ne, &[flags, z]);
            let fallback = load(b, e, 0, one);
            let r = if op == SimdOp::Compress {
                simd(b, e, op, &[x, m])
            } else {
                simd(b, e, op, &[x, m, fallback])
            };
            store(b, e, 2, zero, r);
            let at = index(b, 0)?;
            b.simd(
                Simd {
                    element: e,
                    op: SimdOp::CompressStore(Buffer(3)),
                },
                &[at, m, x],
            )?;
            done(b)?;
            Ok(())
        });
        let x: Vec<i32> = (100..100 + n as i32 + 1).collect();
        let kinds = [
            BufferKind::READ_I32,
            BufferKind::READ_I32,
            BufferKind::WRITE_I32,
            BufferKind::WRITE_I32,
        ];
        agree(
            &format!("{op:?}"),
            Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds),
            &stage,
            |call| {
                [
                    (0b1011u32, 0, 8),
                    (0xffff, 2, 5),
                    (0, 100, 0),
                    (0x8001, 6, 7),
                    (0b111, -1, 4),
                ]
                .into_iter()
                .map(|(bits, at, len)| {
                    let flags: Vec<i32> = (0..n).map(|i| ((bits >> (i % 32)) & 1) as i32).collect();
                    let mut out = vec![-9; n];
                    let mut dst = vec![-5; len];
                    let r = call(
                        &[at.to_word()],
                        &mut [
                            BufferArg::from(&x[..]),
                            BufferArg::from(&flags[..]),
                            BufferArg::from(&mut out[..]),
                            BufferArg::from(&mut dst[..]),
                        ],
                    );
                    format!("{r:?} {out:?} {dst:?}")
                })
                .collect()
            },
        );
    }
}

#[test]
fn u16_gather_and_scatter_check_indices_at_16_bit_boundaries() {
    let stage: Stage = Rc::new(|b| {
        let zero = b.constant(Constant::from(0usize))?;
        let e = Element::U16;
        let index = load(b, e, 0, zero);
        let g = simd(b, e, SimdOp::Gather(Buffer(1)), &[index]);
        store(b, e, 2, zero, g);
        b.simd(
            Simd {
                element: e,
                op: SimdOp::Scatter(Buffer(2)),
            },
            &[index, g],
        )?;
        done(b)?;
        Ok(())
    });
    let n = Element::U16.lanes(bytes());
    let kinds = [
        BufferKind::new(Element::U16, false),
        BufferKind::new(Element::U16, false),
        BufferKind::new(Element::U16, true),
    ];
    let cases = [
        (70000usize, 65535u16),
        (65535, 65534),
        (65535, 65535),
        (40, 39),
        (40, 40),
        (32768, 32767),
        (32769, 32768),
    ];
    let seen = agree(
        "u16 gather",
        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
        &stage,
        |call| {
            cases
                .into_iter()
                .map(|(len, special)| {
                    let src: Vec<u16> = (0..len).map(|i| (i * 7) as u16).collect();
                    let mut index: Vec<u16> = (0..n).map(|i| i as u16 % 30).collect();
                    index[3] = special;
                    let mut out = vec![1u16; n + 40];
                    let r = call(
                        &[],
                        &mut [
                            BufferArg::from(&index[..]),
                            BufferArg::from(&src[..]),
                            BufferArg::from(&mut out[..]),
                        ],
                    );
                    format!("{r:?} {:?}", &out[..n])
                })
                .collect()
        },
    );
    for ((len, special), seen) in cases.into_iter().zip(&seen) {
        // The gather reads `src[special]`; the scatter then writes `out[special]`.
        let gather_ok = (special as usize) < len;
        let scatter_ok = (special as usize) < n + 40;
        assert_eq!(
            seen.starts_with("Ok"),
            gather_ok && scatter_ok,
            "{len} {special}: {seen}"
        );
    }
}

#[test]
fn partial_vector_accesses_agree_at_negative_and_far_offsets() {
    let stage: Stage = Rc::new(|b| {
        let e = Element::I64;
        let zero = b.constant(Constant::from(0usize))?;
        let x = simd(b, e, SimdOp::Iota, &[]);
        let k = index(b, 1)?;
        let m = simd(b, e, SimdOp::FirstN, &[k]);
        let at = index(b, 0)?;
        let l = simd(b, e, SimdOp::LoadSelect(Buffer(1)), &[at, m, x]);
        store(b, e, 0, zero, l);
        store(b, e, 0, at, x);
        done(b)?;
        Ok(())
    });
    let n = Element::I64.lanes(bytes());
    let kinds = [
        BufferKind::new(Element::I64, true),
        BufferKind::new(Element::I64, false),
    ];
    agree(
        "i64 partial accesses",
        Signature::new(&[ScalarType::I32; 2], ScalarType::I32, &kinds),
        &stage,
        |call| {
            let src: Vec<i64> = (0..n as i64 + 3).map(|i| i * 1000).collect();
            let mut seen = Vec::new();
            for at in [0, 1, 3, -1, i32::MIN, i32::MAX, n as i32, 2] {
                for k in [0, 1, n as i32, -1] {
                    let mut out = vec![-1i64; n + 2];
                    let r = call(
                        &[at, k].map(i32::to_word),
                        &mut [BufferArg::from(&mut out[..]), BufferArg::from(&src[..])],
                    );
                    seen.push(format!("at={at} k={k} {r:?} {out:?}"));
                }
            }
            seen
        },
    );
}

#[test]
fn narrowing_and_widening_agree_on_extreme_lanes() {
    let stage: Stage = Rc::new(|b| {
        let (zero, one) = (
            b.constant(Constant::from(0usize))?,
            b.constant(Constant::from(1usize))?,
        );
        let x = load(b, Element::I64, 0, zero);
        let y = load(b, Element::I64, 0, one);
        let narrow = simd(b, Element::I64, SimdOp::Narrow(Element::I32), &[x, y]);
        store(b, Element::I32, 1, zero, narrow);
        let w = load(b, Element::U16, 2, zero);
        let high = SimdOp::Widen {
            to: Element::U32,
            high: true,
        };
        let hi = simd(b, Element::U16, high, &[w]);
        store(b, Element::U32, 3, zero, hi);
        let high = SimdOp::Widen {
            to: Element::I64,
            high: true,
        };
        let hi = simd(b, Element::I32, high, &[narrow]);
        store(b, Element::I64, 4, zero, hi);
        done(b)?;
        Ok(())
    });
    let (n, m, h) = (
        Element::I32.lanes(bytes()),
        Element::I64.lanes(bytes()),
        Element::U16.lanes(bytes()),
    );
    let kinds = [
        BufferKind::new(Element::I64, false),
        BufferKind::new(Element::I32, true),
        BufferKind::new(Element::U16, false),
        BufferKind::new(Element::U32, true),
        BufferKind::new(Element::I64, true),
    ];
    agree(
        "narrow/widen",
        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
        &stage,
        |call| {
            let x: Vec<i64> = (0..m + 1)
                .map(|i| [i64::MAX, i64::MIN, -1, 0x1_0000_0005, -0x1_0000_0005, 3][i % 6])
                .collect();
            let w: Vec<u16> = (0..h).map(|i| (i as u16).wrapping_mul(40000)).collect();
            let (mut a, mut c, mut d) = (vec![0i32; n], vec![0u32; n], vec![0i64; m]);
            let r = call(
                &[],
                &mut [
                    BufferArg::from(&x[..]),
                    BufferArg::from(&mut a[..]),
                    BufferArg::from(&w[..]),
                    BufferArg::from(&mut c[..]),
                    BufferArg::from(&mut d[..]),
                ],
            );
            vec![format!("{r:?} {a:?} {c:?} {d:?}")]
        },
    );
}

/// Each body breaks one verifier rule. Staging it directly on any backend
/// reports the same `BuildError::Verify`, with the same instruction, as
/// materializing portable IR: both routes run the one streaming verifier.
#[test]
fn every_build_route_reports_the_same_verify_error() {
    use splice::frontend::{self, Signature};
    use splice::hir::{Label, ValueType};
    use splice::{BuildError, BuildOptions, Codegen, Mode};
    type Body = fn(&mut dyn KernelBuilder) -> splice::Result<()>;
    const KINDS: [BufferKind; 1] = [BufferKind::READ_I32];
    let bodies: [(&str, Body); 10] = [
        ("uninitialized", |b| {
            let value = b.local(ValueType::Scalar(ScalarType::I32))?;
            b.ret(value)
        }),
        ("uninitialized on one path", |b| {
            let value = b.local(ValueType::Scalar(ScalarType::I32))?;
            let (set, join) = (b.label()?, b.label()?);
            let zero = b.constant(Constant::from(0))?;
            let taken = b.binary(Binary::Ne, b.arg(0)?, zero)?;
            b.branch(taken, set, join)?;
            b.mark(set)?;
            b.assign(value, b.arg(0)?)?;
            b.jump(join)?;
            b.mark(join)?;
            b.ret(value)
        }),
        ("undefined label", |b| {
            let label = b.label()?;
            b.jump(label)
        }),
        ("invalid label", |b| b.jump(Label(7))),
        ("duplicate label", |b| {
            let label = b.label()?;
            b.mark(label)?;
            b.mark(label)?;
            b.ret(b.arg(0)?)
        }),
        ("invalid register", |b| b.ret(Reg(99))),
        ("invalid parameter", |b| b.ret(b.arg(3)?)),
        ("read-only store", |b| {
            let first = b.constant(Constant::from(0usize))?;
            let zero = b.constant(Constant::from(0))?;
            b.store(Buffer(0), first, zero)?;
            b.ret(zero)
        }),
        ("invalid buffer", |b| {
            let len = b.buffer_len(Buffer(1))?;
            b.ret(len)
        }),
        ("falls off the end", |b| {
            b.constant(Constant::from(1))?;
            Ok(())
        }),
    ];
    let options = BuildOptions::host(Mode::Release).unwrap();
    let signature = Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &KINDS);
    for (name, body) in bodies {
        let expected = frontend::materialize(signature, Box::new(body), options).unwrap_err();
        assert!(
            matches!(expected, BuildError::Verify { .. }),
            "{name}: {expected}"
        );
        splice_testkit::each_backend!(B => {
            let error = B::build_with_options(signature, Box::new(body), options)
                .err()
                .unwrap();
            assert!(matches!(error, BuildError::Verify { .. }), "{name}: {error}");
            assert_eq!(
                error.to_string(),
                expected.to_string(),
                "{name} on {}",
                std::any::type_name::<B>()
            );
        });
    }
}

/// Writes that trap: a contiguous store that crosses the end, masked stores
/// and scatters with a bad active lane, and scatters that repeat an index.
/// Every backend leaves the same buffer behind; a scatter writes its lanes
/// in order, so the last of repeated lanes wins and the lanes before the
/// first bad active one are written.
#[test]
fn trapping_and_repeated_writes_leave_the_same_buffer_on_every_backend() {
    let e = Element::I32;
    let n = e.lanes(bytes());
    let stage: Stage = Rc::new(move |b| {
        let at = index(b, 0)?;
        let limit = index(b, 1)?;
        let x = simd(b, e, SimdOp::Iota, &[]);
        let hundred = simd(b, e, SimdOp::Splat(100), &[]);
        let value = simd(b, e, SimdOp::Add, &[x, hundred]);
        let mask = simd(b, e, SimdOp::FirstN, &[limit]);
        let zero = b.constant(Constant::from(0usize))?;
        let indices = load(b, e, 0, zero);
        let op = b.arg(2)?;
        let ops = [
            SimdOp::Store(Buffer(1)),
            SimdOp::StoreSelect(Buffer(1)),
            SimdOp::Scatter(Buffer(1)),
            SimdOp::ScatterSelect(Buffer(1)),
        ];
        let next = (0..ops.len())
            .map(|_| b.label())
            .collect::<splice::Result<Vec<_>>>()?;
        for (case, (&simd_op, &label)) in ops.iter().zip(&next).enumerate() {
            let here = b.label()?;
            let k = b.constant(Constant::from(case as i32))?;
            let hit = b.binary(Binary::Eq, op, k)?;
            b.branch(hit, here, label)?;
            b.mark(here)?;
            let args = match simd_op {
                SimdOp::Store(_) => vec![at, value],
                SimdOp::StoreSelect(_) => vec![at, mask, value],
                SimdOp::Scatter(_) => vec![indices, value],
                _ => vec![indices, mask, value],
            };
            b.simd(
                Simd {
                    element: e,
                    op: simd_op,
                },
                &args,
            )?;
            done(b)?;
            b.mark(label)?;
        }
        done(b)?;
        Ok(())
    });
    let kinds = [
        BufferKind::new(Element::I32, false),
        BufferKind::new(Element::I32, true),
    ];
    let len = n + 2;
    // Scatter indices: lane 1 repeats lane 0, lane `bad` is out of bounds.
    let scatter = |bad: Option<usize>| -> Vec<i32> {
        (0..n)
            .map(|lane| match lane {
                _ if Some(lane) == bad => len as i32 + 5,
                0 | 1 => 1,
                lane => (lane as i32 * 3) % len as i32,
            })
            .collect()
    };
    let mut cases = Vec::new();
    for at in [0, 2, 3, len as i32] {
        for limit in [0, 1, n as i32] {
            for op in [0, 1] {
                cases.push((vec![0; n], at, limit, op));
            }
        }
    }
    for bad in [None, Some(0), Some(n / 2), Some(n - 1)] {
        for limit in [0, n as i32 / 2, n as i32] {
            for op in [2, 3] {
                cases.push((scatter(bad), 0, limit, op));
            }
        }
    }
    let seen = agree(
        "trapping writes",
        Signature::new(&[ScalarType::I32; 3], ScalarType::I32, &kinds),
        &stage,
        |call| {
            cases
                .iter()
                .map(|(indices, at, limit, op)| {
                    let mut out = vec![-1; len];
                    let r = call(
                        &[*at, *limit, *op].map(i32::to_word),
                        &mut [BufferArg::from(&indices[..]), BufferArg::from(&mut out[..])],
                    );
                    format!("{r:?} {out:?}")
                })
                .collect()
        },
    );
    // A scatter writes its active lanes in order up to the first bad one.
    for ((indices, _, limit, op), seen) in cases.iter().zip(&seen) {
        if *op < 2 {
            continue;
        }
        let active = |lane: usize| *op == 2 || (lane as i32) < *limit;
        let mut out = vec![-1; len];
        let mut status = "Ok(0)";
        for lane in (0..n).filter(|&lane| active(lane)) {
            let Some(slot) = out.get_mut(indices[lane] as usize) else {
                status = "Err(Trap(Bounds))";
                break;
            };
            *slot = 100 + lane as i32;
        }
        assert_eq!(
            seen,
            &format!("{status} {out:?}"),
            "{indices:?} op {op} limit {limit}"
        );
    }
}

/// Every slice access against one model of its lanes: the active lanes, in
/// lane order, up to the first one outside the slice. A read with such a
/// lane reads nothing; a write keeps the lanes before it, and the last of
/// repeated indices wins. Inactive lanes are never checked.
#[test]
fn every_slice_access_matches_the_lane_order_model() {
    check_lane_order::<u16>(Element::U16);
    check_lane_order::<i32>(Element::I32);
    check_lane_order::<i64>(Element::I64);
}

fn check_lane_order<T>(e: Element)
where
    T: splice::BufferElement + Copy + std::fmt::Debug + TryFrom<i64>,
{
    let t = |x: i64| T::try_from(x).ok().expect("a value of the element");
    let n = e.lanes(bytes());
    let ops = [
        SimdOp::Load(Buffer(1)),
        SimdOp::LoadSelect(Buffer(1)),
        SimdOp::Gather(Buffer(1)),
        SimdOp::GatherSelect(Buffer(1)),
        SimdOp::Store(Buffer(1)),
        SimdOp::StoreSelect(Buffer(1)),
        SimdOp::Scatter(Buffer(1)),
        SimdOp::ScatterSelect(Buffer(1)),
    ];
    let stage: Stage = Rc::new(move |b| {
        let at = index(b, 0)?;
        let limit = index(b, 1)?;
        let op = b.arg(2)?;
        let zero = b.constant(Constant::from(0usize))?;
        let indices = load(b, e, 0, zero);
        let iota = simd(b, e, SimdOp::Iota, &[]);
        let hundred = simd(b, e, SimdOp::Splat(100), &[]);
        let value = simd(b, e, SimdOp::Add, &[iota, hundred]);
        let passthrough = simd(b, e, SimdOp::Splat(7), &[]);
        let mask = simd(b, e, SimdOp::FirstN, &[limit]);
        for (case, &simd_op) in ops.iter().enumerate() {
            let (here, next) = (b.label()?, b.label()?);
            let k = b.constant(Constant::from(case as i32))?;
            let hit = b.binary(Binary::Eq, op, k)?;
            b.branch(hit, here, next)?;
            b.mark(here)?;
            let memory = simd_op.info().memory.unwrap();
            let mut args = vec![if memory.indexed { indices } else { at }];
            if memory.masked {
                args.push(mask);
            }
            match memory.access {
                splice::hir::Access::Write => args.push(value),
                _ if memory.masked => args.push(passthrough),
                _ => {}
            }
            let inst = Simd {
                element: e,
                op: simd_op,
            };
            if let Some(read) = b.simd(inst, &args)? {
                store(b, e, 2, zero, read);
            }
            done(b)?;
            b.mark(next)?;
        }
        done(b)?;
        Ok(())
    });
    let mut cases: Vec<(usize, usize, i32, usize, Vec<i64>)> = Vec::new();
    for len in [0, 3, n + 2] {
        let lanes: Vec<i64> = (0..n as i64).collect();
        let mut patterns: Vec<Vec<i64>> = vec![
            lanes
                .iter()
                .map(|&lane| lane * 3 % len.max(1) as i64)
                .collect(),
        ];
        for bad in [0, n / 2, n - 1] {
            for value in [len as i64 + 5, -1] {
                if T::try_from(value).is_ok() {
                    let mut indices = patterns[0].clone();
                    indices[1] = indices[0];
                    indices[bad] = value;
                    patterns.push(indices);
                }
            }
        }
        for (op, simd_op) in ops.iter().enumerate() {
            let memory = simd_op.info().memory.unwrap();
            for limit in [0, 1, n / 2, n] {
                if memory.indexed {
                    for indices in &patterns {
                        cases.push((op, len, 0, limit, indices.clone()));
                    }
                } else {
                    for at in [0, 1, 2, len as i32, -1] {
                        cases.push((op, len, at, limit, patterns[0].clone()));
                    }
                }
            }
        }
    }
    let kinds = [
        BufferKind::new(e, false),
        BufferKind::new(e, true),
        BufferKind::new(e, true),
    ];
    let initial = |len: usize| (0..len as i64).map(|i| t(i * 3 + 1)).collect::<Vec<T>>();
    let seen = agree(
        &format!("{e:?} lane order"),
        Signature::new(&[ScalarType::I32; 3], ScalarType::I32, &kinds),
        &stage,
        |call| {
            cases
                .iter()
                .map(|(op, len, at, limit, indices)| {
                    let indices: Vec<T> = indices.iter().map(|&i| t(i)).collect();
                    let mut data = initial(*len);
                    let mut out = vec![t(9); n];
                    let r = call(
                        &[*at, *limit as i32, *op as i32].map(i32::to_word),
                        &mut [
                            BufferArg::from(&indices[..]),
                            BufferArg::from(&mut data[..]),
                            BufferArg::from(&mut out[..]),
                        ],
                    );
                    format!("{r:?} {data:?} {out:?}")
                })
                .collect()
        },
    );
    for ((op, len, at, limit, indices), seen) in cases.iter().zip(&seen) {
        let memory = ops[*op].info().memory.unwrap();
        let active = |lane: usize| !memory.masked || lane < *limit;
        let address = |lane: usize| {
            if memory.indexed {
                indices[lane]
            } else {
                i64::from(*at) + lane as i64
            }
        };
        let slot = |lane: usize| usize::try_from(address(lane)).ok().filter(|&a| a < *len);
        let lanes = (0..n).filter(|&lane| active(lane));
        let fails = lanes.clone().any(|lane| slot(lane).is_none());
        let mut data = initial(*len);
        let mut out = vec![t(9); n];
        let write = memory.access == splice::hir::Access::Write;
        if write {
            for lane in lanes {
                let Some(slot) = slot(lane) else { break };
                data[slot] = t(100 + lane as i64);
            }
        } else if !fails {
            for (lane, out) in out.iter_mut().enumerate() {
                *out = match slot(lane) {
                    Some(slot) if active(lane) => data[slot],
                    _ => t(7),
                };
            }
        }
        let status = if fails { "Err(Trap(Bounds))" } else { "Ok(0)" };
        assert_eq!(
            seen,
            &format!("{status} {data:?} {out:?}"),
            "{e:?} {:?} len {len} at {at} limit {limit} indices {indices:?}",
            ops[*op]
        );
    }
}
