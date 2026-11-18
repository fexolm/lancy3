use splice::{Executable, kernel};
use splice_lancy::Lancy;

#[derive(splice::ComptimeKey)]
struct RetryPolicy {
    base_ms: i32,
    max_ms: i32,
    max_attempts: i32,
}

#[kernel]
fn exponential_delay(#[comptime] policy: &RetryPolicy, attempt: i32) -> i32 {
    let mut remaining = attempt;
    let mut delay = policy.base_ms;
    while remaining > 0 {
        // Avoid overflow even if a future policy sets max_ms to i32::MAX.
        if delay >= policy.max_ms - delay {
            return policy.max_ms;
        }
        delay = delay + delay;
        remaining -= 1;
    }
    delay
}

#[kernel]
fn server_delay(#[comptime] policy: &RetryPolicy, retry_after_ms: i32) -> i32 {
    if retry_after_ms > policy.max_ms {
        return policy.max_ms;
    }
    retry_after_ms
}

/// Return -1 to stop delivery attempts, otherwise the delay in milliseconds.
/// Index 0 requests the first retry. A nonpositive Retry-After means absent.
#[kernel]
fn retry_delay(
    #[comptime] policy: &RetryPolicy,
    status: i32,
    attempt: i32,
    retry_after_ms: i32,
) -> i32 {
    if attempt < 0 || attempt >= policy.max_attempts {
        return -1;
    }
    let class = match status {
        429 | 503 => 2,
        408 | 425 | 500 | 502 | 504 => 1,
        _ => 0,
    };
    if class == 0 {
        return -1;
    }
    if class == 2 && retry_after_ms > 0 {
        return server_delay(policy, retry_after_ms);
    }
    exponential_delay(policy, attempt)
}

fn main() -> Result<(), String> {
    let policy = RetryPolicy {
        base_ms: 100,
        max_ms: 2_000,
        max_attempts: 8,
    };
    let decide = retry_delay::build::<Lancy>(&policy)?;

    // A single native scalar call handles one failed delivery.
    for (status, attempt, retry_after_ms, expected) in [
        (503, 2, 0, 400),
        (429, 0, 750, 750),
        (503, 0, 9_000, 2_000),
        (500, 6, 0, 2_000),
        (502, 1, 800, 200),
        (429, 8, 50, -1),
        (404, 0, 0, -1),
    ] {
        let delay = decide.call(&[status, attempt, retry_after_ms], &mut [])?;
        assert_eq!(delay, expected);
        println!("status {status}, attempt {attempt}, Retry-After {retry_after_ms} ms => {delay}");
    }

    Ok(())
}
