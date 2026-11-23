use crate::ScalarKernel;
use crate::hir::{Binary, Function, Op, Terminator};

// A missing move can affect a later use far from the instruction that clobbered
// it. Keep many independently updated values live, then consume them in order.
const WIDTH: usize = 24;

fn next(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn program(mut seed: u32) -> Function {
    use Binary::*;
    let ops = [
        Add, Sub, Mul, Div, Rem, And, Or, Xor, Eq, Ne, Lt, Le, Gt, Ge, Shl, Shr, Min, Max,
    ];
    let mut f = Function::scalar(WIDTH);
    let b = f.entry();
    let mut live = f.params(b).to_vec();
    let mask = f.append(b, Op::Const(31)).unwrap();
    let one = f.append(b, Op::Const(1)).unwrap();
    for i in 0..(WIDTH * 4) {
        let dst = i % WIDTH;
        let op = ops[next(&mut seed) as usize % ops.len()];
        let mut rhs = live[next(&mut seed) as usize % WIDTH];
        if matches!(op, Div | Rem) {
            // Keep every path running through the final reduction.
            let low = f.append(b, Op::Bin(And, rhs, mask)).unwrap();
            rhs = f.append(b, Op::Bin(Add, low, one)).unwrap();
        }
        live[dst] = f.append(b, Op::Bin(op, live[dst], rhs)).unwrap();
    }
    let mut result = one;
    for value in live {
        let scaled = f.append(b, Op::Bin(Mul, result, mask)).unwrap();
        result = f.append(b, Op::Bin(Add, scaled, value)).unwrap();
    }
    f.set_terminator(b, Terminator::Return(result));
    f
}

#[test]
fn mixed_instruction_sequences_match_unoptimized_ssa() {
    for seed in 1..=64 {
        let f = program(seed);
        let kernel = ScalarKernel::compile(&f, &crate::test_target::options())
            .unwrap_or_else(|e| panic!("seed={seed}: {e}"));
        let mut state = seed;
        let inputs = [
            [0; WIDTH],
            [-1; WIDTH],
            std::array::from_fn(|i| if i % 2 == 0 { i32::MIN } else { i32::MAX }),
            std::array::from_fn(|_| next(&mut state) as i32),
            std::array::from_fn(|_| next(&mut state) as i32),
        ];
        for args in inputs {
            let expected = f.interpret(&args).unwrap();
            assert_eq!(
                kernel.call(&args),
                Ok(expected),
                "seed={seed}, args={args:?}"
            );
        }
    }
}
