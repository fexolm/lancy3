//! A struct-of-fields loop against the same loop over separate slices:
//! `x + vx * dt` and `y + vy * dt` for each particle, written as positions.
//! The struct kernel reads each field at `i * 16 + offset` and stores a
//! whole `Position`; the other reads four `f32` slices.
use super::{N, floats, length};
use crate::Workload;
use criterion::Criterion;
use splice::{BufferArg, Executable, KernelScalar, KernelType, kernel};

#[derive(Clone, Copy, KernelType)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

/// The layout of two `f32`s, so the benchmark workspace holds positions.
#[derive(Clone, Copy, KernelType)]
#[repr(C)]
struct Position {
    x: f32,
    y: f32,
}

#[kernel]
fn advance_structs(particles: &[Particle], out: &mut [Position], dt: f32) -> i32 {
    let n = particles.len();
    if out.len() < n {
        return -1;
    }
    for i in 0..n {
        let p = particles[i];
        out[i] = Position {
            x: p.x + p.vx * dt,
            y: p.y + p.vy * dt,
        };
    }
    2 * n as i32
}

#[kernel]
fn advance_slices(x: &[f32], y: &[f32], vx: &[f32], vy: &[f32], out: &mut [f32], dt: f32) -> i32 {
    let n = x.len();
    if y.len() != n || vx.len() != n || vy.len() != n || out.len() < 2 * n {
        return -1;
    }
    for i in 0..n {
        out[2 * i] = x[i] + vx[i] * dt;
        out[2 * i + 1] = y[i] + vy[i] * dt;
    }
    2 * n as i32
}

fn positions(words: &mut [i32]) -> &mut [Position] {
    let words = floats(words);
    // SAFETY: `Position` is `repr(C)` with two `f32` fields, so it has the
    // size and alignment of `[f32; 2]`; the result borrows `words`
    // exclusively and covers whole pairs only.
    unsafe {
        std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<Position>(), words.len() / 2)
    }
}

pub(super) fn compare(c: &mut Criterion) {
    const DT: f32 = 0.125;
    let particles: Vec<Particle> = (0..N)
        .map(|i| {
            let v = (i % 997) as f32;
            Particle {
                x: v,
                y: -v * 0.5,
                vx: 1.0 - v / 64.0,
                vy: v / 32.0,
            }
        })
        .collect();
    let column = |f: fn(&Particle) -> f32| particles.iter().map(f).collect::<Vec<_>>();
    let (x, y, vx, vy) = (
        column(|p| p.x),
        column(|p| p.y),
        column(|p| p.vx),
        column(|p| p.vy),
    );
    let expected: Vec<i32> = particles
        .iter()
        .flat_map(|p| [p.x + p.vx * DT, p.y + p.vy * DT])
        .map(|v| v.to_bits() as i32)
        .collect();
    let dt = [DT.to_word()];
    Workload {
        name: "micro_struct_particles",
        elements: N,
        workspace_words: 2 * N,
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            length(kernel.call(
                &dt,
                &mut [
                    BufferArg::records(&particles),
                    BufferArg::records_mut(positions(out)),
                ],
            )?)
        },
    }
    .compare(
        c,
        advance_structs::ir,
        builders!(B => advance_structs::build::<B>(advance_structs::BuildArgs {})),
    );
    Workload {
        name: "micro_struct_particles_slices",
        elements: N,
        workspace_words: 2 * N,
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, out: &mut [i32]| {
            length(kernel.call(
                &dt,
                &mut [
                    BufferArg::read(&x),
                    BufferArg::read(&y),
                    BufferArg::read(&vx),
                    BufferArg::read(&vy),
                    BufferArg::write(floats(out)),
                ],
            )?)
        },
    }
    .compare(
        c,
        advance_slices::ir,
        builders!(B => advance_slices::build::<B>(advance_slices::BuildArgs {})),
    );
}
