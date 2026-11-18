use super::*;
use crate::input::{Binary, FunctionBuilder};

#[test]
fn division_and_remainder_trap_with_dead_results_and_keep_prior_stores() {
    use crate::input::{Buffer, BufferArg, BufferKind};
    for op in [Binary::Div, Binary::Rem] {
        for (lhs, rhs) in [(9, 0), (i32::MIN, -1)] {
            let mut b = FunctionBuilder::with_buffers(0, vec![BufferKind::Write]);
            let a = b.constant(lhs);
            let divisor = b.constant(rhs);
            let zero = b.constant(0);
            b.store(Buffer(0), zero, a);
            b.binary(op, a, divisor);
            b.store(Buffer(0), zero, zero);
            b.ret(zero);
            let base = b.finish().unwrap();
            for f in [&base, &base.simplified().unwrap()] {
                let mut data = [30];
                let result = f.interpret_with_buffers(&[], &mut [BufferArg::Write(&mut data)]);
                assert!(result.is_err());
                assert_eq!(data, [lhs]);
            }
        }
    }
}

#[test]
fn signed_arithmetic_uses_truncation_and_refuses_speculative_vector_execution() {
    for (op, expected) in [(Binary::Div, -2), (Binary::Rem, -1)] {
        let mut b = FunctionBuilder::new(2);
        let val = b.binary(op, b.arg(0), b.arg(1));
        b.ret(val);
        let f = b.finish().unwrap();
        assert!(f.needs_status());
        assert_eq!(f.interpret(&[-7, 3]), Ok(expected));

        for (a, c, succeeds) in [(-7, 3, true), (2, 0, false), (i32::MIN, -1, false)] {
            let mut caller = FunctionBuilder::new(0);
            let lhs = caller.constant(a);
            let rhs = caller.constant(c);
            let r = caller.call(f.clone(), &[lhs, rhs]);
            caller.ret(r);
            let base = caller.finish().unwrap();
            assert!(base.needs_status());
            let opt = base.simplified().unwrap();
            assert_eq!(opt.interpret(&[]).is_ok(), succeeds);
            assert_eq!(opt.needs_status(), !succeeds);
            if succeeds {
                assert_eq!(opt.interpret(&[]), Ok(expected));
            }
        }
    }
}

#[test]
fn mixed_specialization_clears_status_when_the_only_trap_is_in_a_dead_arm() {
    let mut child = FunctionBuilder::new(2);
    let good = child.label();
    let bad = child.label();
    child.branch(child.arg(0), good, bad);
    child.mark(bad);
    let zero = child.constant(0);
    let trap = child.binary(Binary::Div, child.arg(1), zero);
    child.ret(trap);
    child.mark(good);
    child.ret(child.arg(1));
    let mut parent = FunctionBuilder::new(1);
    let flag = parent.constant(1);
    let r = parent.call(child.finish().unwrap(), &[flag, parent.arg(0)]);
    parent.ret(r);
    let p = parent.finish().unwrap();
    assert!(p.needs_status());
    let opt = p.simplified().unwrap();
    assert!(!opt.needs_status());
    assert_eq!(opt.interpret(&[-9]), Ok(-9));
}

#[test]
fn prune_unreachable_call_and_fold_constants_across_jump() {
    let mut endless = FunctionBuilder::new(0);
    let spin = endless.label();
    endless.mark(spin);
    endless.jump(spin);
    let mut b = FunctionBuilder::new(0);
    let value = b.local();
    let yes = b.label();
    let no = b.label();
    let done = b.label();
    let a = b.constant(i32::MAX);
    let one = b.constant(1);
    let wrapped = b.binary(Binary::Add, a, one);
    let cond = b.binary(Binary::Lt, wrapped, one);
    b.branch(cond, yes, no);
    b.mark(yes);
    b.assign(value, wrapped);
    b.jump(done);
    b.mark(no);
    let unused = b.call(endless.finish().unwrap(), &[]);
    b.assign(value, unused);
    b.mark(done);
    b.ret(value);
    let opt = b.finish().unwrap().simplified().unwrap();
    assert!(matches!(
        opt.instructions(),
        [Instr::Const(_, i32::MIN), Instr::Return(_)]
    ));
    assert_eq!(opt.interpret(&[]), Ok(i32::MIN));
}

#[test]
fn equal_join_values_fold_and_disagreeing_values_remain_dynamic() {
    for rhs in [15, 16] {
        let mut b = FunctionBuilder::new(1);
        let local = b.local();
        let yes = b.label();
        let no = b.label();
        let done = b.label();
        b.branch(b.arg(0), yes, no);
        b.mark(yes);
        let a = b.constant(15);
        b.assign(local, a);
        b.jump(done);
        b.mark(no);
        let c = b.constant(rhs);
        b.assign(local, c);
        b.mark(done);
        let one = b.constant(1);
        let result = b.binary(Binary::Add, local, one);
        b.ret(result);
        let f = b.finish().unwrap();
        let opt = f.simplified().unwrap();
        assert_eq!(
            opt.code.iter().any(|op| matches!(op, Instr::Bin(..))),
            rhs != 15
        );
        for params in [[0], [1], [-8]] {
            assert_eq!(opt.interpret(&params), f.interpret(&params));
        }
    }
}

fn sum_to_n() -> Function {
    let mut b = FunctionBuilder::new(1);
    let acc = b.local();
    let zero = b.constant(0);
    let one = b.constant(1);
    b.assign(acc, zero);
    let head = b.label();
    let body = b.label();
    let done = b.label();
    b.mark(head);
    let cont = b.binary(Binary::Gt, b.arg(0), zero);
    b.branch(cont, body, done);
    b.mark(body);
    let sum = b.binary(Binary::Add, acc, b.arg(0));
    b.assign(acc, sum);
    let dec = b.binary(Binary::Sub, b.arg(0), one);
    b.assign(b.arg(0), dec);
    b.jump(head);
    b.mark(done);
    b.ret(acc);
    b.finish().unwrap()
}

#[test]
fn loop_carried_values_and_nested_consteval_calls() {
    let sum = sum_to_n();
    let opt_sum = sum.simplified().unwrap();
    for n in [-1, 0, 3, 10, 33] {
        assert_eq!(opt_sum.interpret(&[n]), sum.interpret(&[n]));
    }
    let mut nested = FunctionBuilder::new(1);
    let answer = nested.call(sum, &[nested.arg(0)]);
    let one = nested.constant(1);
    let plus_one = nested.binary(Binary::Add, answer, one);
    nested.ret(plus_one);
    let mut caller = FunctionBuilder::new(0);
    let known = caller.constant(10);
    let result = caller.call(nested.finish().unwrap(), &[known]);
    caller.ret(result);
    let opt = caller.finish().unwrap().simplified().unwrap();
    assert!(matches!(
        opt.instructions(),
        [Instr::Const(_, 56), Instr::Return(_)]
    ));
}

#[test]
fn divergent_unused_call_is_retained_and_shared_callee_identity_survives() {
    let mut endless = FunctionBuilder::new(1);
    let spin = endless.label();
    endless.mark(spin);
    endless.jump(spin);
    let shared = Arc::new(endless.finish().unwrap());
    let mut b = FunctionBuilder::new(1);
    let fixed = b.constant(0);
    b.call_shared(shared.clone(), &[fixed]);
    b.call_shared(shared, &[b.arg(0)]);
    b.ret(fixed);
    let opt = b.finish().unwrap().simplified().unwrap();
    let callees: Vec<_> = opt
        .code
        .iter()
        .filter_map(|op| match op {
            Instr::Call { function, .. } => Some(function),
            _ => None,
        })
        .collect();
    assert_eq!(callees.len(), 2);
    assert!(Arc::ptr_eq(callees[0], callees[1]));
}

fn pick() -> Arc<Function> {
    let mut b = FunctionBuilder::new(2);
    let yes = b.label();
    let no = b.label();
    b.branch(b.arg(0), yes, no);
    b.mark(yes);
    b.ret(b.arg(1));
    b.mark(no);
    let zero = b.constant(0);
    let neg = b.binary(Binary::Sub, zero, b.arg(1));
    b.ret(neg);
    Arc::new(b.finish().unwrap())
}

fn callees(f: &Function) -> Vec<&Arc<Function>> {
    f.instructions()
        .iter()
        .filter_map(|op| match op {
            Instr::Call { function, .. } => Some(function),
            _ => None,
        })
        .collect()
}

fn branches(f: &Function) -> usize {
    f.instructions()
        .iter()
        .filter(|op| matches!(op, Instr::Branch { .. }))
        .count()
}

#[test]
fn mixed_calls_specialize_by_callee_identity_parameter_index_and_value() {
    let helper = pick();
    let mut b = FunctionBuilder::new(1);
    let x = b.arg(0);
    let one = b.constant(1);
    let zero = b.constant(0);
    let calls = [
        b.call_shared(helper.clone(), &[one, x]),
        b.call_shared(helper.clone(), &[one, x]),
        b.call_shared(helper.clone(), &[zero, x]),
        b.call_shared(helper, &[x, one]),
        b.call_shared(pick(), &[one, x]),
    ];
    let sum = calls
        .into_iter()
        .reduce(|a, c| b.binary(Binary::Add, a, c))
        .unwrap();
    b.ret(sum);
    let generic = b.finish().unwrap();
    let opt = generic.simplified().unwrap();
    let targets = callees(&opt);
    assert_eq!(targets.len(), 5);
    assert!(targets.iter().all(|f| f.arity() == 2));
    assert!(Arc::ptr_eq(targets[0], targets[1]));
    for distinct in [2, 3, 4] {
        assert!(!Arc::ptr_eq(targets[0], targets[distinct]));
    }
    assert_eq!(branches(targets[0]), 0);
    assert_eq!(branches(targets[2]), 0);
    assert_eq!(branches(targets[3]), 1);
    for x in [i32::MIN, -5, 0, 12, i32::MAX] {
        assert_eq!(opt.interpret(&[x]), generic.interpret(&[x]));
    }
}

#[test]
fn variant_limit_shares_generic_fallback_and_reuses_earlier_specializations() {
    let helper = pick();
    let mut b = FunctionBuilder::new(1);
    let x = b.arg(0);
    let mut total = b.constant(0);
    for n in (0..VARIANTS_PER_CALLEE + 2).chain([0]) {
        let mode = b.constant(n as i32);
        let result = b.call_shared(helper.clone(), &[mode, x]);
        total = b.binary(Binary::Add, total, result);
    }
    b.ret(total);
    let original = b.finish().unwrap();
    let opt = original.simplified().unwrap();
    let targets = callees(&opt);
    assert_eq!(branches(targets[VARIANTS_PER_CALLEE]), 1);
    assert!(Arc::ptr_eq(
        targets[VARIANTS_PER_CALLEE],
        targets[VARIANTS_PER_CALLEE + 1]
    ));
    assert!(Arc::ptr_eq(targets[0], targets[VARIANTS_PER_CALLEE + 2]));
    assert_eq!(opt.interpret(&[-9]), original.interpret(&[-9]));
}

#[test]
fn mixed_argument_fact_is_killed_on_loop_backedge() {
    let mut helper = FunctionBuilder::new(2);
    let head = helper.label();
    let body = helper.label();
    let done = helper.label();
    helper.mark(head);
    helper.branch(helper.arg(0), body, done);
    helper.mark(body);
    let zero = helper.constant(0);
    helper.assign(helper.arg(0), zero);
    helper.jump(head);
    helper.mark(done);
    helper.ret(helper.arg(1));
    let mut b = FunctionBuilder::new(1);
    let one = b.constant(1);
    let call = b.call(helper.finish().unwrap(), &[one, b.arg(0)]);
    b.ret(call);
    let opt = b.finish().unwrap().simplified().unwrap();
    assert_eq!(branches(callees(&opt)[0]), 1);
    assert_eq!(opt.interpret(&[45]), Ok(45));
}
