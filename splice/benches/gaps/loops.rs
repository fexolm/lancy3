use super::{Builders, Workload};
use criterion::Criterion;
use splice::{BufferArg, Executable, kernel};
use splice_cranelift::Cranelift;
use splice_lancy::Lancy;
use splice_llvm::LlvmManualSimd;
use std::hint::black_box;

// Transform one stream of samples, with a different format for negative values.
// The scaling coefficients depend on the runtime dimensions of the output grid.
#[kernel]
fn conditional_scale(src: &[i32], rows: &[i32], cols: &[i32], out: &mut [i32]) -> i32 {
    let n = src.len() as i32;
    if n > out.len() as i32 {
        return -1;
    }
    for i in 0..n {
        let x = src[i as usize];
        out[i as usize] = if x >= 0 {
            let area = (rows.len() as i32).wrapping_mul(cols.len() as i32);
            let scale = area
                .wrapping_mul(rows.len() as i32)
                .wrapping_add(src.len() as i32);
            x.wrapping_mul(scale).wrapping_add(x ^ area)
        } else {
            x
        };
    }
    n
}

// Specialization can make the checksum unobservable while leaving its loop in SSA.
#[kernel]
fn optional_checksum(
    #[comptime] include_checksum: bool,
    src: &[i32],
    rows: &[i32],
    cols: &[i32],
    out: &mut [i32],
) -> i32 {
    let n = src.len() as i32;
    if n > out.len() as i32 {
        return -1;
    }
    for i in 0..n {
        let x = src[i as usize];
        let mut checksum = x ^ rows.len() as i32;
        let mut round = 0;
        while round < cols.len() as i32 {
            checksum = checksum.wrapping_mul(31) ^ round;
            round += 1;
        }
        out[i as usize] = if include_checksum { checksum } else { x };
    }
    n
}

fn compare_kernel<S>(
    c: &mut Criterion,
    name: &'static str,
    inputs: [&[i32]; 3],
    expected: &[i32],
    stage: impl Fn() -> Result<splice::TypedIr<S>, splice::BuildError>,
    builds: Builders<'_, S>,
) {
    Workload {
        name,
        elements: inputs[0].len(),
        workspace_words: expected.len(),
        expected: &|| expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            let [src, rows, cols] = inputs;
            let count = kernel.call(
                &[],
                &mut [
                    BufferArg::Read(src),
                    BufferArg::Read(rows),
                    BufferArg::Read(cols),
                    BufferArg::Write(out),
                ],
            )?;
            usize::try_from(count).map_err(|_| "invalid loop benchmark buffers".into())
        },
    }
    .compare(c, stage, builds);
}

pub(super) fn compare(c: &mut Criterion) {
    let mut state = 0xd73a_fb45u32;
    let src: Vec<i32> = (0..65_539)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as i32
        })
        .collect();
    let rows = [0; 37];
    let cols = [0; 59];
    let area = (rows.len() * cols.len()) as i64;
    let scale = area * rows.len() as i64 + src.len() as i64;
    let expected: Vec<_> = src
        .iter()
        .map(|&x| {
            if x < 0 {
                x
            } else {
                (scale * i64::from(x) + (i64::from(x) ^ area)) as i32
            }
        })
        .collect();
    let inputs = [&src[..], &rows[..], &cols[..]];
    compare_kernel(
        c,
        "gap_loop_conditional_scale",
        inputs,
        &expected,
        conditional_scale::ir,
        Builders {
            lancy: &|| conditional_scale::build::<Lancy>(conditional_scale::BuildArgs {}),
            cranelift: &|| conditional_scale::build::<Cranelift>(conditional_scale::BuildArgs {}),
            llvm: &|| conditional_scale::build::<LlvmManualSimd>(conditional_scale::BuildArgs {}),
        },
    );
    compare_kernel(
        c,
        "gap_loop_dead_checksum",
        inputs,
        &src,
        || optional_checksum::ir(black_box(false)),
        Builders {
            lancy: &|| {
                optional_checksum::build::<Lancy>(optional_checksum::BuildArgs {
                    include_checksum: black_box(false),
                })
            },
            cranelift: &|| {
                optional_checksum::build::<Cranelift>(optional_checksum::BuildArgs {
                    include_checksum: black_box(false),
                })
            },
            llvm: &|| {
                optional_checksum::build::<LlvmManualSimd>(optional_checksum::BuildArgs {
                    include_checksum: black_box(false),
                })
            },
        },
    );
}
