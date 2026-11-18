use lancy_jit::{Assumption, Builder, NativeKernel, Program};

struct Rng(u64);

impl Rng {
    fn word(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

fn expression(rng: &mut Rng, steps: usize) -> Program {
    let mut b = Builder::new(3);
    let mut values = vec![b.input(0), b.input(1), b.input(2)];
    let mut acc = values[0];
    for _ in 0..steps {
        let rhs = values[rng.word() as usize % values.len()];
        let next = match rng.word() % 11 {
            0 => b.add(acc, rhs),
            1 => b.sub(rhs, acc),
            2 => b.mul(acc, rhs),
            3 => b.or(acc, rhs),
            4 => b.and(rhs, acc),
            5 => b.xor(rhs, acc),
            6 => b.eq(acc, rhs),
            7 => b.lt(rhs, acc),
            8 => b.select(acc, rhs, values[1]),
            9 => b.select(values[2], acc, rhs),
            _ => {
                let k = b.constant(rng.word() as i32);
                b.add(acc, k)
            }
        };
        values.push(next);
        acc = next;
    }
    b.finish(acc).unwrap()
}

#[test]
fn scalar_and_vector_agree_with_interpreter_for_generated_graphs() {
    let mut rng = Rng(0x9fc3_014d_5a7d);
    let mut data: [Vec<i32>; 3] =
        std::array::from_fn(|_| (0..29).map(|_| rng.word() as i32).collect());
    data[0][1..5].copy_from_slice(&[i32::MIN, i32::MAX, 0, -1]);
    data[1][1..5].copy_from_slice(&[i32::MAX, i32::MIN, -1, 0]);
    for trial in 0..120 {
        let p = expression(&mut rng, 1 + trial % 64);
        let scalar = NativeKernel::baseline(&p).unwrap();
        let vector = NativeKernel::specialize(&p, &[]).unwrap();
        let len = trial % 28;
        let cols = data.each_ref().map(|c| &c[1..1 + len]);
        let expected: Vec<_> = (0..len).map(|i| p.eval_row(&cols, i)).collect();
        for kernel in [&scalar, &vector] {
            let mut storage = vec![0x5050_5050; len + 2];
            assert_eq!(kernel.run(&cols, &mut storage[1..len + 1]).unwrap(), len);
            assert_eq!(&storage[1..len + 1], expected, "trial {trial}");
            assert_eq!(storage[0], 0x5050_5050);
            assert_eq!(storage[len + 1], 0x5050_5050);
        }
    }
}

#[test]
fn guard_exit_preserves_entire_uncommitted_suffix() {
    let mut b = Builder::new(3);
    let a = b.input(0);
    let b1 = b.input(1);
    let c = b.input(2);
    let product = b.mul(a, b1);
    let expression = b.sub(product, c);
    let p = b.finish(expression).unwrap();
    let guards = [
        Assumption {
            column: 1,
            value: -3,
        },
        Assumption {
            column: 2,
            value: 11,
        },
    ];
    let kernel = NativeKernel::specialize(&p, &guards).unwrap();
    for len in 1..18 {
        let x: Vec<_> = (0..len as i32).collect();
        for miss in 0..=len {
            for column in [1, 2] {
                let mut y = vec![-3; len];
                let mut z = vec![11; len];
                if miss < len {
                    if column == 1 {
                        y[miss] = -7;
                    } else {
                        z[miss] = 42;
                    }
                }
                let mut out = vec![999; len];
                let done = kernel.run(&[&x, &y, &z], &mut out).unwrap();
                // The vector guard protects four lanes; the tail commits one at a time.
                let expected_prefix = if miss < len / 4 * 4 {
                    miss / 4 * 4
                } else {
                    miss
                };
                assert_eq!(done, expected_prefix);
                assert!(
                    out[..done]
                        .iter()
                        .enumerate()
                        .all(|(i, &v)| v == x[i] * -3 - 11)
                );
                assert!(out[done..].iter().all(|&v| v == 999));
            }
        }
    }
}

#[test]
fn wide_live_ranges_spill_and_reload() {
    let mut b = Builder::new(320);
    let terms: Vec<_> = (0..320).map(|j| b.input(j)).collect();
    let result = terms.into_iter().rev().reduce(|a, x| b.sub(a, x)).unwrap();
    let p = b.finish(result).unwrap();
    let data: Vec<Vec<i32>> = (0..320i32)
        .map(|j| (0..13).map(|i| i - 5 * j).collect())
        .collect();
    let cols: Vec<_> = data.iter().map(Vec::as_slice).collect();
    let expected: Vec<_> = (0..13).map(|row| p.eval_row(&cols, row)).collect();
    for kernel in [
        NativeKernel::baseline(&p).unwrap(),
        NativeKernel::specialize(&p, &[]).unwrap(),
    ] {
        let mut out = [0; 13];
        assert_eq!(kernel.run(&cols, &mut out).unwrap(), 13);
        assert_eq!(out.as_slice(), expected);
    }
}

#[test]
fn result_can_precede_dead_nodes_and_inputs_can_be_empty() {
    let mut b = Builder::new(0);
    let result = b.constant(-19);
    let dead = b.constant(8);
    b.mul(result, dead);
    let p = b.finish(result).unwrap();
    for kernel in [
        NativeKernel::baseline(&p).unwrap(),
        NativeKernel::specialize(&p, &[]).unwrap(),
    ] {
        let mut out = [0; 11];
        assert_eq!(kernel.run(&[], &mut out).unwrap(), out.len());
        assert_eq!(out, [-19; 11]);
        assert_eq!(kernel.run(&[], &mut []).unwrap(), 0);
    }
}

#[test]
fn invalid_external_guards_and_shapes_return_errors() {
    let mut b = Builder::new(1);
    let x = b.input(0);
    let p = b.finish(x).unwrap();
    let a = Assumption {
        column: 0,
        value: 1,
    };
    assert!(NativeKernel::specialize(&p, &[a, Assumption { value: 2, ..a }]).is_err());
    assert!(NativeKernel::specialize(&p, &[Assumption { column: 1, ..a }]).is_err());
    let kernel = NativeKernel::specialize(&p, &[a, a]).unwrap();
    let mut out = [99; 5];
    assert!(kernel.run(&[], &mut out).is_err());
    assert!(kernel.run(&[&[1; 4]], &mut out).is_err());
    assert_eq!(out, [99; 5]);
}
