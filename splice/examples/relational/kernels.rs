use super::{Combine, Expr, Plan, Predicate};
use splice::kernel;

#[kernel]
fn eval(#[comptime] expr: &Expr, key: i32, value: i32) -> i32 {
    match expr {
        Expr::Key => key,
        Expr::Value => value,
        Expr::Literal(n) => *n,
        Expr::Add(a, b) => eval(a, key, value).wrapping_add(eval(b, key, value)),
        Expr::Mul(a, b) => eval(a, key, value).wrapping_mul(eval(b, key, value)),
        Expr::Min(a, b) => eval(a, key, value).min(eval(b, key, value)),
    }
}

#[kernel]
fn matches_row(#[comptime] predicate: &Predicate, key: i32, value: i32) -> i32 {
    match predicate {
        Predicate::AtLeast(a, b) => {
            if eval(a, key, value) >= eval(b, key, value) {
                1
            } else {
                0
            }
        }
        Predicate::Equal(a, b) => {
            if eval(a, key, value) == eval(b, key, value) {
                1
            } else {
                0
            }
        }
        Predicate::And(a, b) => {
            if matches_row(a, key, value) != 0 && matches_row(b, key, value) != 0 {
                1
            } else {
                0
            }
        }
    }
}

#[kernel]
fn scan(
    #[comptime] table: usize,
    offsets: &[i32],
    data: &[i32],
    arena: &mut [i32],
    start: i32,
) -> i32 {
    let first = offsets[table];
    let last = offsets[table.saturating_add(1)];
    if last - first > arena.len() as i32 - start {
        return -1;
    }
    let mut dst = start;
    for src in first..last {
        arena[dst as usize] = data[src as usize];
        dst += 1;
    }
    dst
}

#[kernel]
fn filter(#[comptime] pred: &Predicate, arena: &mut [i32], start: i32, end: i32) -> i32 {
    let mut src = start;
    let mut dst = start;
    while src < end {
        let key = arena[src as usize];
        let value = arena[(src + 1) as usize];
        if matches_row(pred, key, value) != 0 {
            arena[dst as usize] = key;
            arena[(dst + 1) as usize] = value;
            dst += 2;
        }
        src += 2;
    }
    dst
}

#[kernel]
fn project(
    #[comptime] key_expr: &Expr,
    #[comptime] value_expr: &Expr,
    arena: &mut [i32],
    start: i32,
    end: i32,
) -> i32 {
    let mut row = start;
    while row < end {
        let key = arena[row as usize];
        let value = arena[(row + 1) as usize];
        arena[row as usize] = eval(key_expr, key, value);
        arena[(row + 1) as usize] = eval(value_expr, key, value);
        row += 2;
    }
    end
}

#[kernel]
fn bucket(key: i32, mask: i32) -> i32 {
    (key ^ (key >> 16)) & mask
}

// The build side lives at [right, end). Bucket heads followed by chain links
// live immediately after it. Inserting in reverse preserves right input order.
#[kernel]
fn hash_build(arena: &mut [i32], right: i32, end: i32, buckets: i32) -> i32 {
    let links = end + buckets;
    for i in end..links {
        arena[i as usize] = -1;
    }
    let mut row = end;
    while row > right {
        row -= 2;
        let slot = end + bucket(arena[row as usize], buckets - 1);
        let next = arena[slot as usize];
        arena[(links + ((row - right) >> 1)) as usize] = next;
        arena[slot as usize] = row;
    }
    links
}

#[kernel]
fn join_probe(
    #[comptime] combine: Combine,
    arena: &mut [i32],
    start: i32,
    right: i32,
    end: i32,
) -> i32 {
    if right == end {
        return start;
    }
    let count = (end - right) >> 1;
    let mut buckets = 1;
    while buckets < count {
        buckets *= 2;
    }
    let capacity = arena.len() as i32;
    if buckets > capacity - end || count > capacity - end - buckets {
        return -1;
    }
    let links = hash_build(arena, right, end, buckets);
    let output = links + count;
    let mut dst = output;
    let mut left = start;
    while left < right {
        let key = arena[left as usize];
        let value = arena[(left + 1) as usize];
        let mut hit = arena[(end + bucket(key, buckets - 1)) as usize];
        while hit >= 0 {
            if arena[hit as usize] == key {
                if capacity - dst < 2 {
                    return -1;
                }
                let other = arena[(hit + 1) as usize];
                let joined = match combine {
                    Combine::Add => value.wrapping_add(other),
                    Combine::Multiply => value.wrapping_mul(other),
                };
                arena[dst as usize] = key;
                arena[(dst + 1) as usize] = joined;
                dst += 2;
            }
            hit = arena[(links + ((hit - right) >> 1)) as usize];
        }
        left += 2;
    }
    // Leftward copy is safe even when a many-to-many result overlaps its source.
    let mut result_end = start;
    for i in output..dst {
        arena[result_end as usize] = arena[i as usize];
        result_end += 1;
    }
    result_end
}

// Each subtree writes contiguous (key, value) rows starting at start and
// returns their exclusive end. All scratch above that end can be reused.
#[kernel]
fn materialize(
    #[comptime] plan: &Plan,
    offsets: &[i32],
    data: &[i32],
    arena: &mut [i32],
    start: i32,
) -> i32 {
    match plan {
        Plan::Scan(id) => scan(*id, offsets, data, arena, start),
        Plan::Filter(child, pred) => {
            let end = materialize(child, offsets, data, arena, start);
            if end < 0 {
                return -1;
            }
            filter(pred, arena, start, end)
        }
        Plan::Project(child, key, value) => {
            let end = materialize(child, offsets, data, arena, start);
            if end < 0 {
                return -1;
            }
            project(key, value, arena, start, end)
        }
        Plan::Join(left, right, combine) => {
            let middle = materialize(left, offsets, data, arena, start);
            if middle <= start {
                return middle;
            }
            let end = materialize(right, offsets, data, arena, middle);
            if end < 0 {
                return -1;
            }
            join_probe(*combine, arena, start, middle, end)
        }
    }
}

#[kernel]
pub fn query(#[comptime] plan: &Plan, offsets: &[i32], data: &[i32], workspace: &mut [i32]) -> i32 {
    let end = materialize(plan, offsets, data, workspace, 0);
    end >> 1
}
