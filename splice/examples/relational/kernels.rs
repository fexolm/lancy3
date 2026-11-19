use super::pipeline::{Program, Step, Task};
use super::{Combine, Expr, Plan, Predicate};
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

#[kernel]
pub fn query(#[comptime] plan: &Plan, offsets: &[i32], data: &[i32], arena: &mut [i32]) -> i32 {
    let program = comptime!(Program::new(plan));
    let mut saved = vec![0; program.indices];
    let mut bases = vec![0; program.indices];
    let mut sizes = vec![0; program.indices];
    let mut duplicates = vec![0; program.indices];
    let width = Lane::<i32>::lanes() as i32;
    let zero = Lane::<i32>::splat(0);
    let mut low = 0;
    let mut high = arena.len() as i32;
    for task in &program.tasks {
        match task {
            Task::Save(id, high_side) => {
                saved[*id] = if *high_side { low } else { high };
            }
            Task::Run(pipe) => {
                let last = pipe.steps.len();
                // Each boundary holds at most one packet. The compiled plan has
                // one scalar count and SIMD locals per boundary, indexed at build time.
                let mut counts = vec![0; last + 1];
                let mut keys = vec![zero; last + 1];
                let mut vals = vec![zero; last + 1];
                let mut heads = vec![zero; last + 1];
                // A probe retains just the unresolved prefix and its next addresses.
                let mut pending = vec![0; last];
                let mut pkeys = vec![zero; last];
                let mut pvals = vec![zero; last];
                let mut positions = vec![zero; last];
                let start = if pipe.high { high } else { low };
                let first = offsets[pipe.table];
                let mut rows = (offsets[pipe.table.saturating_add(1)] - first) >> 1;
                for id in pipe.dependencies() {
                    if sizes[id] == 0 {
                        rows = 0;
                    }
                }
                let mut limit = Lane::<i32>::alignment_prefix(data, first as usize) as i32;
                if limit == 0 {
                    limit = width;
                }
                let mut row = 0;
                let mut done = false;
                while !done {
                    if counts[0] == 0 && row < rows {
                        let n = (rows - row).min(limit);
                        limit = width;
                        let offset = first + row;
                        if n == width {
                            keys[0] = Lane::<i32>::load(data, offset as usize);
                            vals[0] = Lane::<i32>::load(data, (offset + rows) as usize);
                        } else {
                            let active = Lane::<i32>::first_n(n as usize);
                            keys[0] = Lane::load_select(data, offset as usize, active, zero);
                            vals[0] =
                                Lane::load_select(data, (offset + rows) as usize, active, zero);
                        }
                        counts[0] = n;
                        row += n;
                    }
                    // An empty queue may still receive input. Only propagate EOF
                    // past a stage after both its input and its retry prefix drain.
                    let mut drained = row == rows;
                    for (i, step) in pipe.steps.iter().enumerate() {
                        let j = i + 1;
                        let n = counts[i];
                        if counts[j] == 0 {
                            match step {
                                Step::Filter(pred) => {
                                    if n != 0 {
                                        let k = keys[i];
                                        let v = vals[i];
                                        let keep = Lane::<i32>::first_n(n as usize)
                                            & matches_rows(pred, k, v);
                                        keys[j] = k.compress(keep);
                                        vals[j] = v.compress(keep);
                                        counts[j] = keep.count();
                                        counts[i] = 0;
                                    }
                                }
                                Step::Project(kexpr, vexpr) => {
                                    if n != 0 {
                                        let k = keys[i];
                                        let v = vals[i];
                                        keys[j] = eval(kexpr, k, v);
                                        vals[j] = eval(vexpr, k, v);
                                        counts[j] = n;
                                        counts[i] = 0;
                                    }
                                }
                                Step::Find(id) | Step::Values(id, _) => {
                                    let mut live = pending[i];
                                    let mut k = pkeys[i];
                                    let mut v = pvals[i];
                                    let mut pos = positions[i];
                                    let mask = Lane::<i32>::splat(sizes[*id] - 1);
                                    let take = n.min(width - live);
                                    if take != 0 {
                                        let fresh = keys[i];
                                        let addr = match step {
                                            Step::Find(_) => (fresh ^ (fresh >> 16)) & mask,
                                            _ => heads[i],
                                        };
                                        if live == 0 {
                                            k = fresh;
                                            v = vals[i];
                                            pos = addr;
                                        } else {
                                            let free = Lane::<i32>::iota()
                                                .simd_ge(Lane::<i32>::splat(live));
                                            k = fresh.expand(free, k);
                                            v = vals[i].expand(free, v);
                                            pos = addr.expand(free, pos);
                                        }
                                        counts[i] = n - take;
                                        if n != take {
                                            let shift =
                                                Lane::<i32>::iota() + Lane::<i32>::splat(take);
                                            keys[i] = fresh.swizzle(shift);
                                            vals[i] = vals[i].swizzle(shift);
                                            if comptime!(step.carries_heads()) {
                                                heads[i] = heads[i].swizzle(shift);
                                            }
                                        }
                                        live += take;
                                    }
                                    if live == width || (drained && live > 0) {
                                        match step {
                                            Step::Find(_) => {
                                                let base = bases[*id];
                                                let candidates: Lane<i32> = read_indices(
                                                    arena,
                                                    pos + Lane::<i32>::splat(base),
                                                    live,
                                                );
                                                let links: Lane<i32> = read_indices(
                                                    arena,
                                                    pos + Lane::<i32>::splat(base + sizes[*id]),
                                                    live,
                                                );
                                                let occupied = links.simd_ne(zero);
                                                let equal = candidates.simd_eq(k);
                                                let hit = occupied & equal;
                                                let retry = occupied & !equal;
                                                if hit.all() {
                                                    keys[j] = k;
                                                    vals[j] = v;
                                                    heads[j] = links;
                                                    counts[j] = width;
                                                    live = 0;
                                                } else {
                                                    keys[j] = k.compress(hit);
                                                    vals[j] = v.compress(hit);
                                                    heads[j] = links.compress(hit);
                                                    counts[j] = hit.count();
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
                                                keys[j] = k;
                                                vals[j] = match combine {
                                                    Combine::Add => v + payload,
                                                    Combine::Multiply => v * payload,
                                                };
                                                counts[j] = live;
                                                if duplicates[*id] != 0 {
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
                                    pending[i] = live;
                                    pkeys[i] = k;
                                    pvals[i] = v;
                                    positions[i] = pos;
                                }
                            }
                        }
                        drained = drained && counts[i] == 0 && pending[i] == 0;
                    }
                    let n = counts[last];
                    if n != 0 {
                        if n > ((high - low) >> 1) {
                            return -1;
                        }
                        let addr = if pipe.high { high - 2 * n } else { low };
                        write_rows(arena, addr, n, keys[last], vals[last]);
                        if pipe.high {
                            high = addr;
                        } else {
                            low += 2 * n;
                        }
                        counts[last] = 0;
                    }
                    done = drained;
                }
                match pipe.index {
                    Some(id) => {
                        // Dependent indices lived at the opposite end. Reclaim them
                        // before allocating the directory for the newly built index.
                        if pipe.high {
                            low = saved[id];
                        } else {
                            high = saved[id];
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
                            duplicates[id] = build_index(arena, begin, end, base, buckets);
                            sizes[id] = buckets;
                            bases[id] = base;
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
