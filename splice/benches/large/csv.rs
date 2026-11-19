//! Fixed-scale CSV export for a four-column financial record.
use splice::kernel;

const DELIMITERS: [i32; 4] = [44, 44, 44, 10];
pub(super) const FIELDS: usize = DELIMITERS.len();
pub(super) const FIELD_WORDS: usize = 3;
pub(super) const LIMIT: i32 = 999_999_999;

#[kernel]
fn digits(value: i32) -> i32 {
    let hundreds = value / 100;
    let rest = value - 100 * hundreds;
    let tens = rest / 10;
    let ones = rest - 10 * tens;
    (hundreds + 48) | ((tens + 48) << 8) | ((ones + 48) << 16)
}

/// Export signed thousandths as `+000000.000` through `+999999.999`, or with
/// a minus sign. Each field and its separator occupy three little-endian words.
/// Returns -1 on insufficient space, a partial input row or an out-of-range value.
/// If a value is out of range, fields already written remain in the output.
#[kernel]
#[allow(
    clippy::manual_range_contains,
    reason = "Splice lowers scalar comparisons"
)]
pub(super) fn export(records: &[i32], out: &mut [i32]) -> i32 {
    let fields = records.len() as i32;
    if fields % FIELDS as i32 != 0 || fields > out.len() as i32 / FIELD_WORDS as i32 {
        return -1;
    }
    let mut field = 0;
    while field < fields {
        for (column, &delimiter) in DELIMITERS.iter().enumerate() {
            let index = field + column as i32;
            let value = records[index as usize];
            if value < -LIMIT || value > LIMIT {
                return -1;
            }
            let magnitude = if value < 0 { 0 - value } else { value };
            let sign = if value < 0 { 45 } else { 43 };
            let millions = magnitude / 1_000_000;
            let remainder = magnitude - millions * 1_000_000;
            let thousands = remainder / 1_000;
            let fraction = remainder - thousands * 1_000;
            let offset = index * FIELD_WORDS as i32;
            out[offset as usize] = sign | (digits(millions) << 8);
            out[(offset + 1) as usize] = digits(thousands) | (46 << 24);
            out[(offset + 2) as usize] = digits(fraction) | (delimiter << 24);
        }
        field += FIELDS as i32;
    }
    fields * FIELD_WORDS as i32
}
