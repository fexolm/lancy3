use lancy::kernel;

const PRIORITY: i32 = 1;
const REGULATED: i32 = 2;
const DUPLICATE: i32 = 4;
const HOLD: i32 = 8;

enum Predicate {
    HasFlag(i32),
    RetryLimit(i32),
    LowDuplicate(i32),
    Expedited(i32),
}

enum Policy {
    Forward(i32),
    When(Predicate, Box<Policy>, Box<Policy>),
}

fn deployed_policy() -> Policy {
    // Earlier matches take priority. A tenant can change the ordering or
    // thresholds by installing a different tree with the same event schema.
    [
        (Predicate::HasFlag(HOLD), 4),
        (Predicate::RetryLimit(5), 3),
        (Predicate::LowDuplicate(6), 0),
        (Predicate::Expedited(6), 2),
        (Predicate::HasFlag(REGULATED), 5),
    ]
    .into_iter()
    .rev()
    .fold(Policy::Forward(1), |otherwise, (test, destination)| {
        Policy::When(
            test,
            Box::new(Policy::Forward(destination)),
            Box::new(otherwise),
        )
    })
}

#[kernel]
fn route(#[comptime] policy: &Policy, severity: i32, attempts: i32, flags: i32) -> i32 {
    match policy {
        Policy::Forward(destination) => *destination,
        Policy::When(test, matched, otherwise) => {
            let should_route = match test {
                Predicate::HasFlag(flag) => (flags & *flag) != 0,
                Predicate::RetryLimit(limit) => attempts >= *limit,
                Predicate::LowDuplicate(level) => (flags & DUPLICATE) != 0 && severity < *level,
                Predicate::Expedited(level) => {
                    severity >= *level || ((flags & PRIORITY) != 0 && attempts < 2)
                }
            };
            if should_route {
                route(matched, severity, attempts, flags)
            } else {
                route(otherwise, severity, attempts, flags)
            }
        }
    }
}

unsafe extern "C" {
    fn rules_cpp_one(severity: i32, attempts: i32, flags: i32) -> i32;
    fn rules_cpp_batch(
        severity: *const i32,
        attempts: *const i32,
        flags: *const i32,
        out: *mut i32,
        n: usize,
    );
}

// Expected values are independent of either implementation. These cases
// exercise each priority tie, the signed comparisons and every route.
const EDGE_CASES: &[(i32, i32, i32, i32)] = &[
    (6, 5, HOLD | DUPLICATE | PRIORITY, 4),
    (7, 5, PRIORITY | REGULATED, 3),
    (5, 4, DUPLICATE | PRIORITY, 0),
    (6, 4, DUPLICATE | REGULATED, 2),
    (0, 1, PRIORITY | REGULATED, 2),
    (0, 2, PRIORITY | REGULATED, 5),
    (0, 2, PRIORITY, 1),
    (i32::MIN, i32::MIN, 0, 1),
    (i32::MAX, i32::MAX, i32::MIN, 3),
];

struct Events {
    severity: Vec<i32>,
    attempts: Vec<i32>,
    flags: Vec<i32>,
}

impl Events {
    fn new(len: usize, skewed: bool) -> Self {
        let mut events = Self {
            severity: Vec::with_capacity(len),
            attempts: Vec::with_capacity(len),
            flags: Vec::with_capacity(len),
        };
        let mut seed = 0xf33f_17b5u32;
        for i in 0..len {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let (s, a, f) = if let Some(&(s, a, f, _)) = EDGE_CASES.get(i) {
                (s, a, f)
            } else if skewed && (seed >> 26) != 0 {
                (((seed >> 8) & 3) as i32, ((seed >> 10) & 1) as i32, 0)
            } else {
                (
                    ((seed >> 16) & 7) as i32,
                    ((seed >> 8) & 7) as i32,
                    (seed & 15) as i32,
                )
            };
            events.severity.push(s);
            events.attempts.push(a);
            events.flags.push(f);
        }
        events
    }

    fn columns(&self) -> [&[i32]; 3] {
        [&self.severity, &self.attempts, &self.flags]
    }

    fn cpp_batch(&self, output: &mut [i32]) {
        assert_eq!(output.len(), self.severity.len());
        // SAFETY: Each column contains the same number of items. The owned
        // output slice is disjoint from the borrowed input vectors.
        unsafe {
            rules_cpp_batch(
                self.severity.as_ptr(),
                self.attempts.as_ptr(),
                self.flags.as_ptr(),
                output.as_mut_ptr(),
                output.len(),
            )
        }
    }
}

// A slow interpreter is used only to prepare expected results for the harness.
fn interpret(policy: &Policy, severity: i32, attempts: i32, flags: i32) -> i32 {
    match policy {
        Policy::Forward(destination) => *destination,
        Policy::When(test, yes, no) => {
            let passed = match test {
                Predicate::HasFlag(bit) => flags & bit != 0,
                Predicate::RetryLimit(limit) => attempts >= *limit,
                Predicate::LowDuplicate(level) => flags & DUPLICATE != 0 && severity < *level,
                Predicate::Expedited(level) => {
                    severity >= *level || (flags & PRIORITY != 0 && attempts < 2)
                }
            };
            interpret(if passed { yes } else { no }, severity, attempts, flags)
        }
    }
}

pub fn run() -> Result<(), String> {
    let policy = deployed_policy();
    crate::build("rules/baseline + first event", || {
        let baseline = lancy::ScalarFn3::baseline(&route::build(&policy)?)?;
        let result = baseline.call(6, 5, HOLD | DUPLICATE | PRIORITY);
        if result != 4 {
            return Err("first baseline event chose the wrong route".into());
        }
        Ok(baseline)
    })?;
    let scalar = crate::build("rules/scalar", || {
        lancy::ScalarFn3::compile(&route::build(&policy)?)
    })?;
    let batch = crate::build("rules/SSE2", || {
        lancy::MappedKernel::compile(&route::build(&policy)?)
    })?;
    assert!(batch.is_vectorized());

    for skewed in [false, true] {
        let n = 65_539;
        let events = Events::new(n, skewed);
        let columns = events.columns();
        let expected: Vec<_> = (0..n)
            .map(|i| interpret(&policy, columns[0][i], columns[1][i], columns[2][i]))
            .collect();
        for (i, &(_, _, _, destination)) in EDGE_CASES.iter().enumerate() {
            assert_eq!(expected[i], destination);
        }
        let pattern = if skewed { "mostly-normal" } else { "mixed" };
        crate::compare(
            &format!("rules/{pattern}/one-event"),
            n,
            &expected,
            |out| {
                for (i, destination) in out.iter_mut().enumerate() {
                    // SAFETY: All three arguments and the result are plain i32.
                    *destination =
                        unsafe { rules_cpp_one(columns[0][i], columns[1][i], columns[2][i]) };
                }
            },
            |out| {
                for (i, destination) in out.iter_mut().enumerate() {
                    *destination = scalar.call(columns[0][i], columns[1][i], columns[2][i]);
                }
            },
        )?;
        crate::compare(
            &format!("rules/{pattern}/batch"),
            n,
            &expected,
            |out| events.cpp_batch(out),
            |out| batch.run(&columns, out).unwrap(),
        )?;
    }
    Ok(())
}
