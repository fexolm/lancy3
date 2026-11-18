use lancy_jit::function::{Function, Instr, Label};
use lancy_jit::{ScalarFn, ScalarKernel, kernel};

#[kernel]
fn weighted(#[comptime] bias: i32, position: i32) -> i32 {
    if position == 5 {
        return -bias;
    }
    match position {
        -9..=-1 => position - bias,
        1 | 3 => bias,
        _ => position * 3 + bias,
    }
}

#[kernel]
fn scan(#[comptime] bias: i32, start: i32, end: i32, stop_at: i32) -> i32 {
    let mut position = start;
    let mut total = 0;
    while position < end {
        position += 1;
        if position >= stop_at {
            break;
        }
        if (position & 1) == 0 {
            continue;
        }
        total += weighted(bias, position);
    }
    total
}

#[kernel]
fn two_scans(#[comptime] bias: i32, start: i32, end: i32, stop_at: i32) -> i32 {
    // Both callees create and mutate their own total and position.
    let left = scan(bias, start, end, stop_at);
    let right = scan(bias, start + 1, end + 3, stop_at);
    left - right
}

fn expected_scan(bias: i32, start: i32, end: i32, stop_at: i32) -> i32 {
    (start + 1..=end)
        .take_while(|&pos| pos < stop_at)
        .filter(|pos| pos & 1 != 0)
        .map(|pos| match pos {
            5 => -bias,
            -9..=-1 => pos - bias,
            1 | 3 => bias,
            _ => pos * 3 + bias,
        })
        .sum()
}

#[test]
fn nested_calls_keep_separate_loop_state_on_each_invocation() {
    for bias in [2, -17] {
        let ir = two_scans::build(bias).unwrap();
        let jit = ScalarKernel::compile(&ir).unwrap();
        let typed = ScalarFn::<3>::compile(&ir).unwrap();
        for start in -10..5 {
            for (end, stop) in [(-5, 7), (10, 2), (4, 15), (12, -8)] {
                let args = [start, end, stop];
                let expected = expected_scan(bias, start, end, stop)
                    - expected_scan(bias, start + 1, end + 3, stop);
                assert_eq!(ir.interpret(&args).unwrap(), expected, "args = {args:?}");
                assert_eq!(jit.call(&args).unwrap(), expected, "args = {args:?}");
                assert_eq!(typed.invoke(args), expected, "args = {args:?}");
            }
        }
    }
}

#[kernel]
fn bounded_sum(length: i32) -> i32 {
    let mut i = 0;
    let mut sum = 0;
    while i < length {
        i += 1;
        sum += i;
    }
    sum
}

enum Gate {
    And,
    Or,
    Bypass,
}

#[kernel]
fn gated(#[comptime] mode: Gate, enabled: i32, length: i32) -> i32 {
    match mode {
        Gate::And => {
            if enabled != 0 && bounded_sum(length) > 3 {
                10
            } else {
                20
            }
        }
        Gate::Or => {
            if enabled != 0 || bounded_sum(length) > 3 {
                10
            } else {
                20
            }
        }
        Gate::Bypass => enabled,
    }
}

// The helper is pure, so output equality alone cannot detect eager evaluation.
// Verify that exactly one outgoing edge of the first branch can reach the call.
fn call_reachable_from(code: &[Instr], edge: Label) -> bool {
    let target = |label| {
        code.iter()
            .position(|i| matches!(i, Instr::Label(l) if *l == label))
            .unwrap()
    };
    let mut pending = vec![target(edge)];
    let mut visited = vec![false; code.len()];
    while let Some(pc) = pending.pop() {
        if visited[pc] {
            continue;
        }
        visited[pc] = true;
        match &code[pc] {
            Instr::Call { .. } => return true,
            Instr::Return(_) => {}
            Instr::Jump(label) => pending.push(target(*label)),
            Instr::Branch { yes, no, .. } => {
                pending.push(target(*yes));
                pending.push(target(*no));
            }
            _ => pending.push(pc + 1),
        }
    }
    false
}

fn assert_short_circuit_cfg(ir: &Function) {
    let code = ir.instructions();
    for instruction in code {
        match instruction {
            Instr::Call { .. } => panic!("RHS call precedes the branch on the LHS"),
            Instr::Branch { yes, no, .. } => {
                assert_ne!(
                    call_reachable_from(code, *yes),
                    call_reachable_from(code, *no)
                );
                return;
            }
            _ => {}
        }
    }
    panic!("short circuit must branch on the runtime input");
}

#[test]
fn runtime_boolean_operators_skip_the_right_hand_call() {
    for (mode, use_and) in [(Gate::And, true), (Gate::Or, false)] {
        let ir = gated::build(mode).unwrap();
        assert_short_circuit_cfg(&ir);
        let compiled = ScalarFn::<2>::compile(&ir).unwrap();
        for enabled in [0, 1] {
            for length in [0, 2, 5] {
                let enough = length * (length + 1) / 2 > 3;
                let result = if use_and {
                    enabled != 0 && enough
                } else {
                    enabled != 0 || enough
                };
                assert_eq!(
                    compiled.invoke([enabled, length]),
                    if result { 10 } else { 20 }
                );
            }
        }
    }
}

#[test]
fn comptime_match_removes_whole_call_graph_from_scalar_ir() {
    let plain = gated::build(Gate::Bypass).unwrap();
    assert!(
        !plain
            .instructions()
            .iter()
            .any(|i| matches!(i, Instr::Call { .. } | Instr::Branch { .. }))
    );
    let compiled = ScalarFn::<2>::compile(&plain).unwrap();
    assert_eq!(compiled.invoke([-21, 5]), -21);
}
