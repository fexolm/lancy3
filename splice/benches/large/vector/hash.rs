//! MurmurHash3 x86_32 over eight INT32 columns, with one key per SIMD lane.
use splice::{kernel, simd::Lane};

pub(crate) const COLUMNS: usize = 8;
const COLUMN_IDS: [i32; COLUMNS] = [0, 1, 2, 3, 4, 5, 6, 7];
const C1: i32 = 0xcc9e_2d51_u32 as i32;
const C2: i32 = 0x1b87_3593;
const BIAS: i32 = 0xe654_6b64_u32 as i32;
const FINAL1: i32 = 0x85eb_ca6b_u32 as i32;
const FINAL2: i32 = 0xc2b2_ae35_u32 as i32;

#[kernel]
fn logical_right(#[comptime] shift: u32, value: Lane<i32>) -> Lane<i32> {
    let mask = comptime!((u32::MAX >> shift) as i32);
    (value >> shift) & Lane::<i32>::splat(mask)
}

#[kernel]
fn rotate_left(#[comptime] shift: u32, value: Lane<i32>) -> Lane<i32> {
    let factor = comptime!(1i32 << shift);
    let low: Lane<i32> = logical_right(32 - shift, value);
    (value * Lane::<i32>::splat(factor)) | low
}

#[kernel]
fn packet(
    #[comptime] full: bool,
    count: i32,
    stride: i32,
    row: i32,
    seed: i32,
    keys: &[i32],
    out: &mut [i32],
) -> i32 {
    let valid = Lane::<i32>::first_n(count as usize);
    let zero = Lane::<i32>::splat(0);
    let mut h = Lane::<i32>::splat(seed);
    for &col in COLUMN_IDS.iter() {
        let offset = (col * stride + row) as usize;
        let value = if full {
            Lane::load(keys, offset)
        } else {
            Lane::load_select(keys, offset, valid, zero)
        };
        let rotated: Lane<i32> = rotate_left(15, value * Lane::<i32>::splat(C1));
        let mixed = h ^ (rotated * Lane::<i32>::splat(C2));
        h = rotate_left(13, mixed);
        h = h * Lane::<i32>::splat(5) + Lane::<i32>::splat(BIAS);
    }
    h ^= Lane::<i32>::splat((4 * COLUMNS) as i32);
    let upper: Lane<i32> = logical_right(16, h);
    h = (h ^ upper) * Lane::<i32>::splat(FINAL1);
    let upper: Lane<i32> = logical_right(13, h);
    h = (h ^ upper) * Lane::<i32>::splat(FINAL2);
    let upper: Lane<i32> = logical_right(16, h);
    h ^= upper;
    if full {
        h.store(out, row as usize);
    } else {
        h.store_select(out, row as usize, valid);
    }
    count
}

/// Hash each row as a 32-byte key in little-endian column order. `keys` contains
/// eight consecutive columns of equal length. Returns -1 for an invalid shape.
#[kernel]
pub(crate) fn hash_keys(seed: i32, keys: &[i32], out: &mut [i32]) -> i32 {
    let words = keys.len() as i32;
    let rows = words / COLUMNS as i32;
    if words % COLUMNS as i32 != 0 || rows > out.len() as i32 {
        return -1;
    }
    let width = Lane::<i32>::lanes() as i32;
    let mut row = 0;
    while rows - row >= width {
        packet(true, width, rows, row, seed, keys, out);
        row += width;
    }
    if row < rows {
        packet(false, rows - row, rows, row, seed, keys, out);
    }
    rows
}
