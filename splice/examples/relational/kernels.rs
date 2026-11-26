use super::pipeline::{Program, Step, Task};
use super::{Combine, Expr, Predicate};
use splice::{
    kernel,
    simd::{Lane, Mask},
};

#[kernel]
fn eval(#[comptime] expr: &Expr, key: Lane<i32>, value: Lane<i32>) -> Lane<i32> {
    match expr {
        Expr::Key => key,
        Expr::Value => value,
        Expr::Literal(n) => Lane::<i32>::splat(*n),
        Expr::Add(a, b) => eval(a, key, value) + eval(b, key, value),
        Expr::Mul(a, b) => eval(a, key, value) * eval(b, key, value),
        Expr::Min(a, b) => {
            let x = eval(a, key, value);
            let y = eval(b, key, value);
            x.simd_lt(y).select(x, y)
        }
    }
}
#[kernel]
fn matches_rows(#[comptime] pred: &Predicate, key: Lane<i32>, value: Lane<i32>) -> Mask<i32> {
    match pred {
        Predicate::AtLeast(a, b) => {
            let x: Lane<i32> = eval(a, key, value);
            let y: Lane<i32> = eval(b, key, value);
            x.simd_ge(y)
        }
        Predicate::Equal(a, b) => {
            let x: Lane<i32> = eval(a, key, value);
            let y: Lane<i32> = eval(b, key, value);
            x.simd_eq(y)
        }
        Predicate::And(a, b) => matches_rows(a, key, value) & matches_rows(b, key, value),
    }
}

#[kernel]
fn write_rows(
    arena: &mut [i32],
    start: i32,
    count: i32,
    keys: Lane<i32>,
    values: Lane<i32>,
) -> i32 {
    let width = Lane::<i32>::lanes() as i32;
    let lo = keys.interleave_low(values);
    let hi = keys.interleave_high(values);
    if count == width {
        lo.store(arena, start as usize);
        hi.store(arena, (start + width) as usize);
    } else {
        let words = count * 2;
        lo.store_select(arena, start as usize, Lane::<i32>::first_n(words as usize));
        if words > width {
            hi.store_select(
                arena,
                (start + width) as usize,
                Lane::<i32>::first_n((words - width) as usize),
            );
        }
    }
    0
}

// Slot heads point to value words. The preceding word holds the next value
// for that exact key. Reuse the streamed key word after inserting its key.
#[kernel]
fn build_index(arena: &mut [i32], begin: i32, end: i32, base: i32, buckets: i32) -> i32 {
    let heads = base + buckets;
    let width = Lane::<i32>::lanes() as i32;
    let zero = Lane::<i32>::splat(0);
    let mut slot = 0;
    while buckets - slot >= width {
        zero.store(arena, (heads + slot) as usize);
        slot += width;
    }
    while slot < buckets {
        arena[(heads + slot) as usize] = 0;
        slot += 1;
    }
    let mut duplicates = 0;
    let mut row = begin;
    while row < end {
        let key = arena[row as usize];
        let mut probe = (key ^ (key >> 16)) & (buckets - 1);
        let mut head = arena[(heads + probe) as usize];
        while head != 0 && arena[(base + probe) as usize] != key {
            probe = (probe + 1) & (buckets - 1);
            head = arena[(heads + probe) as usize];
        }
        if head != 0 {
            duplicates = 1;
        }
        arena[(base + probe) as usize] = key;
        arena[(heads + probe) as usize] = row + 1;
        arena[row as usize] = head;
        row += 2;
    }
    duplicates
}

// Only the last few live lanes of an exhausted pipeline take the scalar
// path. All vector gathers operate on a full packet of real requests.
#[kernel]
fn read_indices(arena: &[i32], indices: Lane<i32>, count: i32) -> Lane<i32> {
    if count == Lane::<i32>::lanes() as i32 {
        Lane::<i32>::gather(arena, indices)
    } else {
        let mut result = Lane::<i32>::splat(0);
        for lane in 0..count {
            result = result.replace_wrapping(lane, arena[indices.extract_wrapping(lane) as usize]);
        }
        result
    }
}

#[kernel(print)]
pub fn query(
    #[comptime] program: &Program<'_>,
    offsets: &[i32],
    data: &[i32],
    arena: &mut [i32],
    state: &mut [i32],
) -> i32 {
    let required = program.state_words;
    if state.len() < required {
        return -1;
    }
    let bases = program.indices;
    let sizes = bases + program.indices;
    let duplicates = sizes + program.indices;
    let counts = duplicates + program.indices;
    for word in 0..counts as i32 {
        state[word as usize] = 0;
    }
    let lanes = program.lanes;
    let width = Lane::<i32>::lanes() as i32;
    let zero = Lane::<i32>::splat(0);
    if width > lanes as i32 {
        return -1;
    }
    let mut low = 0;
    let mut high = arena.len() as i32;
    for task in &program.tasks {
        match task {
            Task::Save(id, high_side) => {
                state[*id] = if *high_side { low } else { high };
            }
            Task::Run(pipe) => {
                let steps: &[Step] = &pipe.steps;
                let last = steps.len();
                // Boundary queues and retry queues reuse the same scratch between pipes.
                let keys = counts + last + 1;
                let vals = keys + (last + 1) * lanes;
                let heads = vals + (last + 1) * lanes;
                let pending = heads + (last + 1) * lanes;
                let pkeys = pending + last;
                let pvals = pkeys + last * lanes;
                let positions = pvals + last * lanes;
                let end = positions + last * lanes;
                for word in counts as i32..end as i32 {
                    state[word as usize] = 0;
                }
                let start = if pipe.high { high } else { low };
                let first = offsets[pipe.table];
                let mut rows = (offsets[pipe.table.saturating_add(1)] - first) >> 1;
                for step in steps {
                    match step {
                        Step::Find(id) => {
                            if state[sizes + *id] == 0 {
                                rows = 0;
                            }
                        }
                        Step::Filter(_) | Step::Project(..) | Step::Values(..) => {}
                    }
                }
                let mut limit = Lane::<i32>::alignment_prefix(data, first as usize) as i32;
                if limit == 0 {
                    limit = width;
                }
                let mut row = 0;
                let mut done = false;
                while !done {
                    if state[counts] == 0 && row < rows {
                        let n = (rows - row).min(limit);
                        limit = width;
                        let offset = first + row;
                        if n == width {
                            Lane::<i32>::load(data, offset as usize).store(state, keys);
                            Lane::<i32>::load(data, (offset + rows) as usize).store(state, vals);
                        } else {
                            let active = Lane::<i32>::first_n(n as usize);
                            Lane::load_select(data, offset as usize, active, zero)
                                .store(state, keys);
                            Lane::load_select(data, (offset + rows) as usize, active, zero)
                                .store(state, vals);
                        }
                        state[counts] = n;
                        row += n;
                    }
                    // An empty queue may still receive input. Only propagate EOF
                    // past a stage after both its input and its retry prefix drain.
                    let mut drained = row == rows;
                    for (i, step) in steps.iter().enumerate() {
                        let j = i + 1;
                        let n = state[counts + i];
                        if state[counts + j] == 0 {
                            match step {
                                Step::Filter(pred) => {
                                    if n != 0 {
                                        let k = Lane::<i32>::load(state, keys + i * lanes);
                                        let v = Lane::<i32>::load(state, vals + i * lanes);
                                        let keep = Lane::<i32>::first_n(n as usize)
                                            & matches_rows(pred, k, v);
                                        k.compress(keep).store(state, keys + j * lanes);
                                        v.compress(keep).store(state, vals + j * lanes);
                                        state[counts + j] = keep.count();
                                        state[counts + i] = 0;
                                    }
                                }
                                Step::Project(kexpr, vexpr) => {
                                    if n != 0 {
                                        let k = Lane::<i32>::load(state, keys + i * lanes);
                                        let v = Lane::<i32>::load(state, vals + i * lanes);
                                        eval(kexpr, k, v).store(state, keys + j * lanes);
                                        eval(vexpr, k, v).store(state, vals + j * lanes);
                                        state[counts + j] = n;
                                        state[counts + i] = 0;
                                    }
                                }
                                Step::Find(id) | Step::Values(id, _) => {
                                    let mut live = state[pending + i];
                                    let mut k = Lane::<i32>::load(state, pkeys + i * lanes);
                                    let mut v = Lane::<i32>::load(state, pvals + i * lanes);
                                    let mut pos = Lane::<i32>::load(state, positions + i * lanes);
                                    let mask = Lane::<i32>::splat(state[sizes + *id] - 1);
                                    let take = n.min(width - live);
                                    if take != 0 {
                                        let fresh = Lane::<i32>::load(state, keys + i * lanes);
                                        let addr = match step {
                                            Step::Find(_) => (fresh ^ (fresh >> 16)) & mask,
                                            _ => Lane::<i32>::load(state, heads + i * lanes),
                                        };
                                        if live == 0 {
                                            k = fresh;
                                            v = Lane::<i32>::load(state, vals + i * lanes);
                                            pos = addr;
                                        } else {
                                            let free = Lane::<i32>::iota()
                                                .simd_ge(Lane::<i32>::splat(live));
                                            k = fresh.expand(free, k);
                                            v = Lane::<i32>::load(state, vals + i * lanes)
                                                .expand(free, v);
                                            pos = addr.expand(free, pos);
                                        }
                                        state[counts + i] = n - take;
                                        if n != take {
                                            let shift =
                                                Lane::<i32>::iota() + Lane::<i32>::splat(take);
                                            fresh.swizzle(shift).store(state, keys + i * lanes);
                                            Lane::<i32>::load(state, vals + i * lanes)
                                                .swizzle(shift)
                                                .store(state, vals + i * lanes);
                                            if let Step::Values(..) = step {
                                                Lane::<i32>::load(state, heads + i * lanes)
                                                    .swizzle(shift)
                                                    .store(state, heads + i * lanes);
                                            }
                                        }
                                        live += take;
                                    }
                                    if live == width || (drained && live > 0) {
                                        match step {
                                            Step::Find(_) => {
                                                let base = state[bases + *id];
                                                let candidates: Lane<i32> = read_indices(
                                                    arena,
                                                    pos + Lane::<i32>::splat(base),
                                                    live,
                                                );
                                                let links: Lane<i32> = read_indices(
                                                    arena,
                                                    pos + Lane::<i32>::splat(
                                                        base + state[sizes + *id],
                                                    ),
                                                    live,
                                                );
                                                let occupied = links.simd_ne(zero);
                                                let equal = candidates.simd_eq(k);
                                                let hit = occupied & equal;
                                                let retry = occupied & !equal;
                                                if hit.all() {
                                                    k.store(state, keys + j * lanes);
                                                    v.store(state, vals + j * lanes);
                                                    links.store(state, heads + j * lanes);
                                                    state[counts + j] = width;
                                                    live = 0;
                                                } else {
                                                    k.compress(hit).store(state, keys + j * lanes);
                                                    v.compress(hit).store(state, vals + j * lanes);
                                                    links
                                                        .compress(hit)
                                                        .store(state, heads + j * lanes);
                                                    state[counts + j] = hit.count();
                                                    if retry.any() {
                                                        k = k.compress(retry);
                                                        v = v.compress(retry);
                                                        pos = ((pos + Lane::<i32>::splat(1))
                                                            & mask)
                                                            .compress(retry);
                                                        live = retry.count();
                                                    } else {
                                                        live = 0;
                                                    }
                                                }
                                            }
                                            Step::Values(_, combine) => {
                                                let payload: Lane<i32> =
                                                    read_indices(arena, pos, live);
                                                k.store(state, keys + j * lanes);
                                                (match combine {
                                                    Combine::Add => v + payload,
                                                    Combine::Multiply => v * payload,
                                                })
                                                .store(state, vals + j * lanes);
                                                state[counts + j] = live;
                                                if state[duplicates + *id] != 0 {
                                                    let next: Lane<i32> = read_indices(
                                                        arena,
                                                        pos - Lane::<i32>::splat(1),
                                                        live,
                                                    );
                                                    let more = next.simd_ne(zero);
                                                    k = k.compress(more);
                                                    v = v.compress(more);
                                                    pos = next.compress(more);
                                                    live = more.count();
                                                } else {
                                                    live = 0;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                    state[pending + i] = live;
                                    k.store(state, pkeys + i * lanes);
                                    v.store(state, pvals + i * lanes);
                                    pos.store(state, positions + i * lanes);
                                }
                            }
                        }
                        drained = drained && state[counts + i] == 0 && state[pending + i] == 0;
                    }
                    let n = state[counts + last];
                    if n != 0 {
                        if n > ((high - low) >> 1) {
                            return -1;
                        }
                        let addr = if pipe.high { high - 2 * n } else { low };
                        write_rows(
                            arena,
                            addr,
                            n,
                            Lane::<i32>::load(state, keys + last * lanes),
                            Lane::<i32>::load(state, vals + last * lanes),
                        );
                        if pipe.high {
                            high = addr;
                        } else {
                            low += 2 * n;
                        }
                        state[counts + last] = 0;
                    }
                    done = drained;
                }
                match pipe.index {
                    Some(id) => {
                        // Dependent indices lived at the opposite end. Reclaim them
                        // before allocating the directory for the newly built index.
                        if pipe.high {
                            low = state[id];
                        } else {
                            high = state[id];
                        }
                        let begin = if pipe.high { high } else { start };
                        let end = if pipe.high { start } else { low };
                        let entries = (end - begin) >> 1;
                        if entries > 0 {
                            let room = (high - low) >> 1;
                            let mut buckets = 1;
                            while buckets <= entries {
                                if buckets > (room >> 1) {
                                    return -1;
                                }
                                buckets *= 2;
                            }
                            // Fix the load factor independently of the available workspace.
                            if entries > buckets - (buckets >> 2) {
                                if buckets > (room >> 1) {
                                    return -1;
                                }
                                buckets *= 2;
                            }
                            let base = if pipe.high { high - 2 * buckets } else { low };
                            state[duplicates + id] = build_index(arena, begin, end, base, buckets);
                            state[sizes + id] = buckets;
                            state[bases + id] = base;
                            if pipe.high {
                                high = base;
                            } else {
                                low += 2 * buckets;
                            }
                        }
                    }
                    None => {
                        return low >> 1;
                    }
                }
            }
        }
    }
    0
}
