use lancy_jit::function::Binary;
use lancy_jit::{FunctionBuilder, MappedKernel, ScalarKernel};

#[test]
fn scalar_entry_guard_deoptimizes_before_executing_a_mutable_loop() {
    use lancy_jit::{AdaptiveScalar, Tier};
    let mut b = FunctionBuilder::new(2);
    let (head, body, done) = (b.label(), b.label(), b.label());
    let x = b.arg(0);
    let remaining = b.arg(1);
    let one = b.constant(1);
    b.mark(head);
    b.branch(remaining, body, done);
    b.mark(body);
    let next = b.binary(Binary::Add, x, one);
    b.assign(x, next);
    let step = b.binary(Binary::Sub, remaining, one);
    b.assign(remaining, step);
    b.jump(head);
    b.mark(done);
    b.ret(x);
    let f = b.finish().unwrap();
    assert!(AdaptiveScalar::<2>::with_assumptions(f.clone(), &[(2, 0)]).is_err());
    assert!(AdaptiveScalar::<2>::with_assumptions(f.clone(), &[(0, 3), (0, 4)]).is_err());
    let kernel = AdaptiveScalar::<2>::with_assumptions(f, &[(0, 3), (0, 3)]).unwrap();
    assert_eq!(kernel.invoke([3, 8]), 11);
    kernel.wait_for_optimization().unwrap();
    assert_eq!(kernel.tier(), Tier::Specialized);
    assert_eq!(kernel.invoke([3, 8]), 11);
    std::thread::scope(|scope| {
        for input in [-7, 9, 3, i32::MAX] {
            let k = kernel.clone();
            scope.spawn(move || assert_eq!(k.invoke([input, 1]), input.wrapping_add(1)));
        }
    });
    assert_eq!(kernel.tier(), Tier::Generic);
    assert_eq!(kernel.invoke([5, 6]), 11);
}

#[test]
fn reject_bad_edges_uninitialized_join_wrong_call_and_fallthrough() {
    let mut f = FunctionBuilder::new(1);
    let unknown = f.label();
    f.jump(unknown);
    assert!(f.finish().unwrap_err().contains("undefined label"));
    let mut f = FunctionBuilder::new(1);
    let (yes, join) = (f.label(), f.label());
    let local = f.local();
    f.branch(f.arg(0), yes, join);
    f.mark(yes);
    f.assign(local, f.arg(0));
    f.mark(join);
    f.ret(local);
    assert!(f.finish().unwrap_err().contains("uninitialized"));
    let mut identity = FunctionBuilder::new(1);
    identity.ret(identity.arg(0));
    let mut f = FunctionBuilder::new(0);
    let value = f.call(identity.finish().unwrap(), &[]);
    f.ret(value);
    assert!(f.finish().unwrap_err().contains("expects 1 arguments"));
    let mut f = FunctionBuilder::new(0);
    f.constant(2);
    assert!(f.finish().unwrap_err().contains("falls off"));
}

#[test]
fn mapping_callee_with_multiple_returns_preserves_noncanonical_masks_and_joins() {
    let mut child = FunctionBuilder::new(2);
    let (yes, no) = (child.label(), child.label());
    child.branch(child.arg(0), yes, no);
    child.mark(yes);
    let one = child.constant(1);
    let plus = child.binary(Binary::Add, child.arg(1), one);
    child.ret(plus);
    child.mark(no);
    child.ret(child.arg(0));
    let mut f = FunctionBuilder::new(2);
    let res = f.call(child.finish().unwrap(), &[f.arg(0), f.arg(1)]);
    let (pos, neg, join) = (f.label(), f.label(), f.label());
    let out = f.local();
    let zero = f.constant(0);
    let mask = f.binary(Binary::Lt, res, zero);
    f.branch(mask, neg, pos);
    f.mark(neg);
    f.assign(out, f.arg(1));
    f.jump(join);
    f.mark(pos);
    f.assign(out, res);
    f.mark(join);
    f.ret(out);
    let ir = f.finish().unwrap();
    let map = MappedKernel::compile(&ir).unwrap();
    assert!(map.is_vectorized());
    let mask = [1, -2, 0, -1, 0x01010101, 1, 2];
    let v = [i32::MIN, i32::MAX, 10, -20, 99, 0, 3];
    let expected: Vec<_> = mask
        .iter()
        .zip(v)
        .map(|(&m, v)| ir.interpret(&[m, v]).unwrap())
        .collect();
    let mut dst = [0; 7];
    map.run(&[&mask, &v], &mut dst).unwrap();
    assert_eq!(&dst, expected.as_slice());
    assert!(map.run(&[&mask], &mut dst).is_err());
    let compiled = ScalarKernel::compile(&ir).unwrap();
    let mut caller = compiled.caller();
    for i in 0..v.len() {
        assert_eq!(caller.call(&[mask[i], v[i]]).unwrap(), expected[i]);
    }
    assert!(caller.call(&[1]).is_err());
}

#[test]
fn loops_map_with_the_scalar_backend() {
    let mut f = FunctionBuilder::new(1);
    let n = f.arg(0);
    let total = f.constant(0);
    let one = f.constant(1);
    let zero = f.constant(0);
    let (head, body, exit) = (f.label(), f.label(), f.label());
    f.mark(head);
    let positive = f.binary(Binary::Gt, n, zero);
    f.branch(positive, body, exit);
    f.mark(body);
    let sum = f.binary(Binary::Add, total, n);
    f.assign(total, sum);
    let decreased = f.binary(Binary::Sub, n, one);
    f.assign(n, decreased);
    f.jump(head);
    f.mark(exit);
    f.ret(total);
    let map = MappedKernel::compile(&f.finish().unwrap()).unwrap();
    assert!(!map.is_vectorized());
    let mut dst = [99; 5];
    map.run(&[&[0, 4, 1, 10, -1]], &mut dst).unwrap();
    assert_eq!(dst, [0, 10, 1, 55, 0]);
}

#[test]
fn scalar_backends_agree_on_forward_edges_and_shared_returns() {
    use lancy_jit::ScalarFn;
    let mut seed = 734_851u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed >> 8
    };
    for _ in 0..64 {
        let mut b = FunctionBuilder::new(3);
        let labels: Vec<_> = (0..9).map(|_| b.label()).collect();
        let value = b.local();
        b.assign(value, b.arg(1));
        for (i, &label) in labels.iter().enumerate() {
            b.mark(label);
            let tail = &labels[i + 1..];
            if tail.is_empty() {
                b.ret(value);
                break;
            }
            let other = tail[next() as usize % tail.len()];
            match next() % 5 {
                0 => {
                    let constant = b.constant(next() as i32);
                    b.assign(value, constant);
                    b.jump(other);
                }
                1 => b.ret(value),
                2 => {
                    b.assign(value, b.arg(2));
                    b.jump(other);
                }
                _ => {
                    let bit = b.constant(1 << (next() & 7));
                    let cond = b.binary(Binary::And, b.arg(0), bit);
                    b.branch(cond, tail[0], other);
                }
            }
        }
        let f = b.finish().unwrap();
        let baseline = ScalarKernel::baseline(&f).unwrap();
        let typed = ScalarFn::<3>::compile(&f).unwrap();
        for flags in [0, 1, 7, 0x55, 0xaa, -1] {
            let args = [flags, -207, 42];
            let expected = f.interpret(&args).unwrap();
            assert_eq!(baseline.call(&args).unwrap(), expected);
            assert_eq!(typed.invoke(args), expected);
        }
    }
}
