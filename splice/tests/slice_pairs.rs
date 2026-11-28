//! A slice operand is a pointer and the length it was produced with: a
//! buffer parameter's pair or the pair of one `slice_range`. A body that
//! pairs a pointer with another length (which would let the backend check
//! an index against the wrong length) fails to build on every route:
//! `materialize`, each backend's direct build, in release and debug.
use splice::frontend::{self, KernelBuilder, Signature};
use splice::hir::{Constant, ScalarType, Simd, SimdOp, Slice, ValueType};
use splice::{BufferArg, BufferKind, BuildOptions, Codegen, Executable, Mode};

const KINDS: [BufferKind; 2] = [
    BufferKind::new(ScalarType::I32, true),
    BufferKind::new(ScalarType::I32, false),
];

type Body = fn(&mut dyn KernelBuilder) -> splice::Result<()>;

const PAIRED: &str =
    "a slice operand must be a buffer parameter or the result of one slice operation";

/// Build `body` on every route and expect each to fail with `what`.
fn rejected_everywhere(body: Body, what: &str) {
    for mode in [Mode::Release, Mode::Debug] {
        let options = BuildOptions::host(mode).unwrap();
        let signature = || Signature::new(&[ScalarType::I32], ScalarType::I32, &KINDS);
        let error = frontend::materialize(signature(), Box::new(body), options).unwrap_err();
        assert!(
            error.to_string().contains(what),
            "materialize {mode:?}: {error}"
        );
        splice_testkit::each_backend!(B => {
            let name = std::any::type_name::<B>();
            let error = B::build_with_options(signature(), Box::new(body), options)
                .map(|_| ())
                .unwrap_err();
            assert!(error.to_string().contains(what), "{name} {mode:?}: {error}");
        });
    }
}

/// `xs.ptr` with the length 2 of no slice: `out[1] = 7` would write past a
/// one-element `out`.
fn forged(b: &mut dyn KernelBuilder) -> splice::Result<Slice> {
    let out = b.buffer(0)?;
    let len = b.constant(Constant::index(2))?;
    Ok(Slice { ptr: out.ptr, len })
}

fn finish(b: &mut dyn KernelBuilder) -> splice::Result<()> {
    let zero = b.constant(Constant::from(0))?;
    b.ret(zero)
}

#[test]
fn a_pointer_with_an_unrelated_length_is_rejected() {
    fn store(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let fake = forged(b)?;
        let (one, seven) = (
            b.constant(Constant::index(1))?,
            b.constant(Constant::from(7))?,
        );
        b.store(fake, one, seven)?;
        finish(b)
    }
    fn load(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let fake = forged(b)?;
        let one = b.constant(Constant::index(1))?;
        let value = b.load(fake, one)?;
        b.ret(value)
    }
    fn simd(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let fake = forged(b)?;
        let zero = b.constant(Constant::index(0))?;
        let op = SimdOp::Load(fake);
        b.simd(
            Simd {
                element: ScalarType::I32,
                op,
            },
            &[zero],
        )?;
        finish(b)
    }
    fn reslice(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let fake = forged(b)?;
        let zero = b.constant(Constant::index(0))?;
        b.slice_range(fake, zero, fake.len)?;
        finish(b)
    }
    fn swapped_lengths(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        // The pointer of `out` with the length of the other parameter.
        let (out, xs) = (b.buffer(0)?, b.buffer(1)?);
        let zero = b.constant(Constant::index(0))?;
        b.load(
            Slice {
                ptr: out.ptr,
                len: xs.len,
            },
            zero,
        )?;
        finish(b)
    }
    fn mixed_ranges(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        // The pointer of one subslice with the length of another.
        let out = b.buffer(0)?;
        let (zero, one) = (
            b.constant(Constant::index(0))?,
            b.constant(Constant::index(1))?,
        );
        let short = b.slice_range(out, zero, one)?;
        let long = b.slice_range(out, one, out.len)?;
        b.load(
            Slice {
                ptr: long.ptr,
                len: out.len,
            },
            zero,
        )?;
        b.load(
            Slice {
                ptr: short.ptr,
                len: long.len,
            },
            zero,
        )?;
        finish(b)
    }
    for body in [store, load, simd, reslice, swapped_lengths, mixed_ranges] {
        rejected_everywhere(body, PAIRED);
    }
}

#[test]
fn the_registers_of_a_slice_are_not_reassigned() {
    fn length(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let out = b.buffer(0)?;
        let zero = b.constant(Constant::index(0))?;
        let part = b.slice_range(out, zero, out.len)?;
        let two = b.constant(Constant::index(2))?;
        b.assign(part.len, two)?;
        finish(b)
    }
    fn pointer_local(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let kind = BufferKind::new(ScalarType::I32, true);
        b.local(ValueType::Ptr(kind))?;
        finish(b)
    }
    rejected_everywhere(
        length,
        "a slice's registers are written only by its slice operation",
    );
    rejected_everywhere(pointer_local, "a slice pointer is produced with its length");
}

/// `slice_range` clamps: whatever its operands, the slice stays inside.
#[test]
fn a_slice_range_stays_within_its_slice() {
    fn body(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let (out, start) = (b.buffer(0)?, b.arg(0)?);
        let start = b.cast(start, ScalarType::Index)?;
        let big = b.constant(Constant::index(usize::MAX))?;
        // `out[min(start, len)..len]`, then write its length to `out[0]`.
        let part = b.slice_range(out, start, big)?;
        let len = b.cast(part.len, ScalarType::I32)?;
        let zero = b.constant(Constant::index(0))?;
        b.store(out, zero, len)?;
        b.ret(len)
    }
    fn run<B: Codegen>(code: &B::Output, start: i32) -> (i32, [i32; 3])
    where
        B::Output: Executable,
    {
        let (mut out, xs) = ([9; 3], [0; 1]);
        let word = start as u32 as u64;
        let result = code
            .call(
                &[word],
                &mut [BufferArg::write(&mut out), BufferArg::read(&xs)],
            )
            .unwrap();
        (result as i32, out)
    }
    for mode in [Mode::Release, Mode::Debug] {
        let options = BuildOptions::host(mode).unwrap();
        let signature = || Signature::new(&[ScalarType::I32], ScalarType::I32, &KINDS);
        let ir = frontend::materialize(signature(), Box::new(body), options).unwrap();
        splice_testkit::each_backend!(B => {
            let direct = B::build_with_options(signature(), Box::new(body), options).unwrap();
            for code in [direct, B::compile(&ir).unwrap()] {
                for (start, len) in [(0, 3), (2, 1), (3, 0), (4, 0), (i32::MAX, 0)] {
                    let (result, out) = run::<B>(&code, start);
                    assert_eq!((result, out), (len, [len, 9, 9]), "start {start}");
                }
            }
        });
    }
}
