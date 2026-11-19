//! Batched lookup in a power-of-two linear-probing table without tombstones.
use splice::{kernel, simd::Lane};
use std::collections::HashMap;

const EMPTY: i32 = 0;
const MULTIPLIER: i32 = 0x9e37_79b1_u32 as i32;

/// Writes the slot of each query, or -1 for a missing key. The table contains
/// unique keys and reserves zero for empty slots. Repeated queries return the
/// same slot. A malformed capacity or short output returns -1 up front.
#[kernel]
pub fn lookup(table: &[i32], queries: &[i32], out: &mut [i32]) -> i32 {
    let capacity = table.len() as i32;
    if capacity == 0 || (capacity & (capacity - 1)) != 0 || out.len() < queries.len() {
        return -1;
    }
    let wrap = Lane::<i32>::splat(capacity - 1);
    let empty = Lane::<i32>::splat(EMPTY);
    let miss = Lane::<i32>::splat(-1);
    let step = Lane::<i32>::splat(1);
    let width = Lane::<i32>::lanes() as i32;
    let count = queries.len() as i32;
    let mut base = 0;
    while base < count {
        let batch = (count - base).min(width);
        let valid = Lane::<i32>::first_n(batch as usize);
        let keys = Lane::load_select(queries, base as usize, valid, empty);
        let mut slots = (keys * Lane::<i32>::splat(MULTIPLIER)) & wrap;
        let mut active = valid & keys.simd_ne(empty);
        let mut result = miss;
        let mut probes = 0;
        // One full lap also settles misses when every slot is occupied.
        while probes < capacity && active.any() {
            let resident = Lane::gather_select(table, active, slots, empty);
            let hits = active & resident.simd_eq(keys);
            result = hits.select(slots, result);
            active = active & !hits & resident.simd_ne(empty);
            slots = (slots + step) & wrap;
            probes += 1;
        }
        result.store_select(out, base as usize, valid);
        base += batch;
    }
    count
}

// A permutation of u32 keeps the benchmark's generated key sets disjoint.
// Zero maps to itself, so the positive input IDs below never generate EMPTY.
fn key(id: u32) -> i32 {
    let x = (id ^ (id >> 16)).wrapping_mul(0x7feb_352d);
    let y = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    (y ^ (y >> 16)) as i32
}

/// Construct the table once, then interleave hits and misses in query order.
pub fn fixture(capacity: usize, entries: usize, query_count: usize) -> (Vec<i32>, Vec<i32>) {
    assert!(capacity.is_power_of_two() && capacity <= i32::MAX as usize);
    assert!(entries <= capacity && query_count <= i32::MAX as usize);
    let mut table = vec![EMPTY; capacity];
    let wrap = capacity - 1;
    for id in 1..=entries {
        let k = key(id as u32);
        let mut slot = k.wrapping_mul(MULTIPLIER) as usize & wrap;
        while table[slot] != EMPTY {
            slot = (slot + 1) & wrap;
        }
        table[slot] = k;
    }
    let queries = (0..query_count)
        .map(|i| {
            let sample = key(i as u32 + 1) as u32;
            let id = if i % 2 == 0 && entries != 0 {
                sample % entries as u32
            } else {
                entries as u32 + sample % capacity as u32
            };
            key(id + 1)
        })
        .collect();
    (table, queries)
}

/// Scalar oracle built from the occupied slots. It ignores the probing scheme.
pub fn reference(table: &[i32], queries: &[i32]) -> Vec<i32> {
    let slots: HashMap<_, _> = table
        .iter()
        .enumerate()
        .filter(|&(_, &key)| key != EMPTY)
        .map(|(slot, &key)| (key, slot as i32))
        .collect();
    queries
        .iter()
        .map(|key| *slots.get(key).unwrap_or(&-1))
        .collect()
}
