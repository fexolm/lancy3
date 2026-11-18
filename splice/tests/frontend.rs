use splice::hir::{Binary, BufferKind, FunctionBuilder, Instr};
use splice::{BufferArg, Codegen, Executable, Function, kernel};

// A backend used only to inspect the IR handed off by build().
struct Inspect(Function);

impl Codegen for Inspect {
    type Kernel = Self;

    fn compile(f: &Function) -> Result<Self, String> {
        Ok(Self(f.clone()))
    }
}

impl Executable for Inspect {
    fn call(&self, _: &[i32], _: &mut [BufferArg<'_>]) -> Result<i32, String> {
        Err("inspection only".into())
    }
}

trait Factor {
    fn factor(&self) -> i32;
}

impl Factor for i32 {
    fn factor(&self) -> i32 {
        *self
    }
}

#[kernel]
fn scaled<T, const OFFSET: i32>(#[comptime] settings: &T, row: i32) -> i32
where
    T: Factor,
{
    row.wrapping_mul(settings.factor()).wrapping_add(OFFSET)
}

#[kernel]
fn adjusted(#[comptime] positive: bool, row: i32) -> i32 {
    if positive {
        scaled::<_, 3>(&2, row)
    } else {
        row
    }
}

#[kernel(batch)]
fn column(#[comptime] positive: bool, row: i32) -> i32 {
    adjusted(positive, row)
}

#[test]
fn generic_functions_and_nested_calls_stage_before_codegen() {
    let compiled = scaled::build::<Inspect, _, 3>(&2).unwrap();
    assert_eq!(scaled::<_, 3>(&2, 7), 17);
    assert_eq!(compiled.0.arity(), 1);
    assert!(
        compiled
            .0
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Bin(_, Binary::Mul, ..)))
    );
    let positive = adjusted::ir(true).unwrap();
    assert!(
        positive
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Call { .. }))
    );
    let negative = adjusted::build::<Inspect>(false).unwrap();
    assert!(
        !negative
            .0
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Call { .. }))
    );
}

#[test]
fn batch_compiles_the_complete_column_loop() {
    let batch = column::build::<Inspect>(true).unwrap().0;
    assert_eq!(batch.arity(), 0);
    assert_eq!(batch.buffer_kinds(), [BufferKind::Read, BufferKind::Write]);
    assert!(
        batch
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Store { .. }))
    );
    let input = [1, 2];
    assert!(
        splice::validate_buffers(
            batch.buffer_kinds(),
            &[BufferArg::Read(&input), BufferArg::Read(&input)]
        )
        .is_err()
    );
}

#[test]
fn verification_rejects_a_branch_with_an_uninitialized_result() {
    let mut b = FunctionBuilder::new(1);
    let result = b.local();
    let assign = b.label();
    let done = b.label();
    b.branch(b.arg(0), assign, done);
    b.mark(assign);
    b.assign(result, b.arg(0));
    b.mark(done);
    b.ret(result);
    assert!(b.finish().unwrap_err().contains("uninitialized"));
}
