//! User structs as kernel data: by value, by reference and in slices, read
//! and written field by field or whole, with the layout rustc gives them
//! (`repr(Rust)` reorders fields; padding, narrow fields and `bool`s).
//! Every case runs on each backend, staged directly and from portable IR,
//! in release and debug, against the kernel's own Rust function; an index
//! outside traps where Rust panics, after the writes before it.
#![allow(clippy::needless_range_loop, clippy::assign_op_pattern)]
use splice::{BuildOptions, CallError, Codegen, Executable, KernelType, Mode, Trap, kernel};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Clone, Copy, Debug, PartialEq, KernelType)]
struct Point {
    x: f32,
    y: f32,
}

/// Narrow fields between wide ones: rustc reorders `repr(Rust)` fields and
/// pads the struct.
#[derive(Clone, Copy, Debug, PartialEq, KernelType)]
struct Particle {
    alive: bool,
    pos: Point,
    tag: u8,
    mass: f64,
    kind: i16,
    id: char,
}

/// A `repr(C)` struct keeps declaration order, with padding.
#[derive(Clone, Copy, Debug, PartialEq, KernelType)]
#[repr(C)]
struct Sample {
    flag: u8,
    value: i64,
    small: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, KernelType)]
#[repr(transparent)]
struct Id(u32);

#[kernel]
fn norm1(p: Point) -> f32 {
    p.x * p.x - p.y.min(1.0)
}

#[kernel]
fn by_reference(p: &Point, scale: f32) -> f32 {
    let q = *p;
    q.x * scale - p.y
}

/// Moves every live particle; `xs[k]` is read last, so a bad `k` traps after
/// every write.
#[kernel]
fn step(xs: &mut [Particle], velocity: &[Point], dt: f32, k: i32) -> i32 {
    let mut moved = 0_i32;
    for i in 0..xs.len() {
        if xs[i].alive {
            xs[i].pos.x += velocity[i].x * dt;
            xs[i].pos.y = xs[i].pos.y + velocity[i].y * dt;
            xs[i].tag = xs[i].tag.wrapping_add(1);
            xs[i].kind -= 1;
            moved += 1;
        }
    }
    moved * 100 + xs[k as usize].tag as i32
}

/// Whole elements: a load, a change through a mutable local and a store,
/// struct literals with nested structs and literals.
#[kernel]
fn swap_pairs(xs: &mut [Particle], out: &mut [Point]) -> i32 {
    let n = xs.len();
    for i in 0..n / 2 {
        let a = xs[i];
        let mut b = xs[n - 1 - i];
        b.alive = !b.alive;
        b.pos.x = b.pos.x * 2.0;
        xs[i] = b;
        xs[n - 1 - i] = a;
    }
    for i in 0..out.len() {
        let p = xs[i % n.max(1)].pos;
        out[i] = Point { x: p.y, y: 1.5 };
    }
    if n > 0 {
        xs[0] = Particle {
            alive: true,
            pos: Point {
                x: -1.0,
                y: xs[0].pos.x,
            },
            tag: 7,
            mass: 0.25,
            kind: -3,
            id: 'z',
        };
    }
    n as i32
}

#[kernel]
fn samples(xs: &[Sample]) -> i64 {
    let mut sum = 0_i64;
    for i in 0..xs.len() {
        let s = xs[i];
        if s.flag != 0 {
            sum += s.value * s.small as i64;
        }
    }
    sum
}

/// A transparent newtype is its scalar: compared, ordered and stored.
#[kernel]
fn largest(ids: &[Id], out: &mut [Id], floor: Id) -> i32 {
    let mut best = floor;
    let mut index = -1_i32;
    for i in 0..ids.len() {
        if ids[i] > best {
            best = ids[i];
            index = i as i32;
        }
    }
    if !out.is_empty() {
        out[0] = best;
        out[out.len() - 1].0 = floor.0 + 1;
        if out.len() > 1 {
            out[1] = Id(7);
        }
    }
    index
}

#[kernel]
fn centroid_x(points: &[Point]) -> f32 {
    let mut sum = 0.0_f32;
    for p in points {
        sum += p.x;
    }
    for &p in points.iter() {
        sum += p.y * 0.5;
    }
    sum
}

#[kernel]
fn scale_all(points: &mut [Point], by: f32) -> i32 {
    for i in 0..points.len() {
        points[i].x *= by;
    }
    points.len() as i32
}

/// Struct slices are resliced, split and passed to nested kernels; `at`
/// is read last.
#[kernel]
fn regions(points: &mut [Point], cut: i32, at: i32) -> f32 {
    let tail = &points[cut as usize..];
    let total = centroid_x(tail);
    let (head, rest) = points.split_at_mut(1);
    scale_all(head, 3.0);
    let moved = scale_all(&mut rest[1..], -1.0);
    let mut chunks = 0.0_f32;
    for pair in points.chunks_exact(2) {
        chunks += pair[1].x - pair[0].y;
    }
    total * 100.0 + moved as f32 + chunks + points[at as usize].y
}

macro_rules! builds {
    ($kernel:ident, $b:ident) => {{
        let mut codes = Vec::new();
        for mode in [Mode::Release, Mode::Debug] {
            let options = BuildOptions::host(mode).unwrap();
            codes.push($kernel::build_with_options::<$b>(options, $kernel::BuildArgs {}).unwrap());
            codes.push($b::compile(&$kernel::ir_with_options(options).unwrap()).unwrap());
        }
        codes
    }};
}

fn rust<T>(f: impl FnOnce() -> T) -> Result<T, CallError> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|_| CallError::Trap(Trap::Bounds))
}

fn particles() -> Vec<Particle> {
    (0..5)
        .map(|i| Particle {
            alive: i % 3 != 1,
            pos: Point {
                x: i as f32,
                y: -(i as f32) * 0.5,
            },
            tag: 250 + i as u8,
            mass: i as f64 * 1.5,
            kind: i as i16 - 2,
            id: char::from(b'a' + i as u8),
        })
        .collect()
}

fn check<B: Codegen<Output: Executable>>() {
    let points = [Point { x: 1.5, y: -2.0 }, Point { x: -0.0, y: 3.25 }];
    for code in builds!(norm1, B) {
        for p in points {
            assert_eq!(code.call(norm1::CallArgs { p }), Ok(norm1(p)));
        }
    }
    for code in builds!(by_reference, B) {
        for p in &points {
            let call = by_reference::CallArgs { p, scale: 3.0 };
            assert_eq!(code.call(call), Ok(by_reference(p, 3.0)));
        }
    }
    let velocity: Vec<Point> = (0..6)
        .map(|i| Point {
            x: i as f32 * 0.25,
            y: 1.0 - i as f32,
        })
        .collect();
    for code in builds!(step, B) {
        for n in 0..=5 {
            for k in [0, 2, n as i32, 9] {
                let (mut actual, mut expected) = (particles(), particles());
                let call = step::CallArgs {
                    xs: &mut actual[..n],
                    velocity: &velocity,
                    dt: 0.5,
                    k,
                };
                let result = code.call(call);
                let wanted = rust(|| step(&mut expected[..n], &velocity, 0.5, k));
                assert_eq!((result, actual), (wanted, expected), "{n} particles, k {k}");
            }
        }
    }
    for code in builds!(swap_pairs, B) {
        for n in 0..=5 {
            for m in [0, 3] {
                let (mut actual, mut expected) = (particles(), particles());
                let (mut out, mut wanted_out) = (
                    vec![Point { x: 0.0, y: 0.0 }; m],
                    vec![Point { x: 0.0, y: 0.0 }; m],
                );
                let call = swap_pairs::CallArgs {
                    xs: &mut actual[..n],
                    out: &mut out,
                };
                let result = code.call(call);
                let wanted = rust(|| swap_pairs(&mut expected[..n], &mut wanted_out));
                assert_eq!(
                    (result, actual, out),
                    (wanted, expected, wanted_out),
                    "{n} particles, {m} points"
                );
            }
        }
    }
    let rows: Vec<Sample> = (0..4)
        .map(|i| Sample {
            flag: (i % 2) as u8,
            value: -(1 << 40) + i,
            small: 60000 + i as u16,
        })
        .collect();
    for code in builds!(samples, B) {
        assert_eq!(
            code.call(samples::CallArgs { xs: &rows }),
            Ok(samples(&rows))
        );
    }
    let ids = [Id(4), Id(9), Id(2), Id(9), Id(11)];
    for code in builds!(largest, B) {
        for n in 0..=ids.len() {
            for floor in [Id(0), Id(10), Id(20)] {
                let (mut out, mut wanted_out) = ([Id(0); 2], [Id(0); 2]);
                let call = largest::CallArgs {
                    ids: &ids[..n],
                    out: &mut out,
                    floor,
                };
                let result = code.call(call);
                let wanted = largest(&ids[..n], &mut wanted_out, floor);
                assert_eq!(
                    (result, out),
                    (Ok(wanted), wanted_out),
                    "{n} ids, floor {floor:?}"
                );
            }
        }
    }
    let cloud: Vec<Point> = (0..7)
        .map(|i| Point {
            x: i as f32 * 1.25 - 2.0,
            y: 3.0 - i as f32,
        })
        .collect();
    for code in builds!(centroid_x, B) {
        for n in 0..=cloud.len() {
            let call = centroid_x::CallArgs {
                points: &cloud[..n],
            };
            assert_eq!(code.call(call), Ok(centroid_x(&cloud[..n])), "{n} points");
        }
    }
    for code in builds!(regions, B) {
        for n in 2..=cloud.len() {
            for (cut, at) in [
                (0, 0),
                (2, 1),
                (n as i32, 0),
                (n as i32 + 1, 0),
                (1, n as i32),
            ] {
                let (mut actual, mut expected) = (cloud[..n].to_vec(), cloud[..n].to_vec());
                let call = regions::CallArgs {
                    points: &mut actual,
                    cut,
                    at,
                };
                let result = code.call(call);
                let wanted = rust(|| regions(&mut expected, cut, at));
                assert_eq!(
                    (result, actual),
                    (wanted, expected),
                    "{n} points, cut {cut}, at {at}"
                );
            }
        }
    }
}

#[test]
fn structs_match_rust_on_every_backend() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !info.to_string().contains("out of bounds") {
            default(info);
        }
    }));
    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn same(xs: &[Point]) -> bool {
    xs[0] == xs[1]
}

/// A whole struct is not a scalar: comparing two is reported when the
/// kernel is built, on every backend.
#[test]
fn struct_values_are_not_compared_yet() {
    fn check<B: Codegen<Output: Executable>>() {
        let options = BuildOptions::host(Mode::Release).unwrap();
        let error = same::build_with_options::<B>(options, same::BuildArgs {})
            .err()
            .expect("comparing structs is rejected");
        assert!(
            error.to_string().contains("kernels do not compare"),
            "{error}"
        );
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// The derived layout is the one rustc chose, reordered fields and padding
/// included.
#[test]
fn layouts_are_rustc_layouts() {
    use splice::KernelData;
    use splice::hir::{DataType, ScalarType};
    use std::mem::{align_of, offset_of, size_of};
    let DataType::Struct(particle) = Particle::TYPE else {
        panic!("a struct layout")
    };
    assert_eq!(
        (particle.size, particle.align),
        (size_of::<Particle>(), align_of::<Particle>())
    );
    let offsets: Vec<_> = particle.fields.iter().map(|f| (f.name, f.offset)).collect();
    assert_eq!(
        offsets,
        [
            ("alive", offset_of!(Particle, alive)),
            ("pos", offset_of!(Particle, pos)),
            ("tag", offset_of!(Particle, tag)),
            ("mass", offset_of!(Particle, mass)),
            ("kind", offset_of!(Particle, kind)),
            ("id", offset_of!(Particle, id)),
        ]
    );
    assert_eq!(
        Particle::TYPE.leaf(2),
        Some((
            offset_of!(Particle, pos) + offset_of!(Point, y),
            ScalarType::F32
        ))
    );
    assert_eq!(Id::TYPE, DataType::Scalar(ScalarType::U32));
    assert_eq!(Particle::TYPE.check(), Ok(()));
}
