use splice::{BufferArg, Executable, kernel};
use splice_lancy::Lancy;

const PAID: i32 = 2;
const MINIMUM_EXCLUSIVE: i32 = 2;

#[kernel]
fn attributed_amount(
    #[comptime] packed_validity: bool,
    #[comptime] paid: i32,
    #[comptime] minimum_exclusive: i32,
    status: i32,
    quantity: i32,
    amount: i32,
    validity: i32,
) -> i32 {
    let accepted = status == paid && quantity > minimum_exclusive;
    let selected = if packed_validity {
        accepted && validity != 0
    } else {
        accepted
    };
    if selected { amount } else { 0 }
}

unsafe extern "C" {
    fn sql_cpp_project(
        status: *const i32,
        quantity: *const i32,
        amount: *const i32,
        bitmap: *const u8,
        out: *mut i32,
        count: usize,
        clustered: bool,
    );
}

const fn masks() -> [[i32; 8]; 256] {
    let mut table = [[0_i32; 8]; 256];
    let mut byte = 0;
    while byte < table.len() {
        let mut lane = 0;
        while lane < 8 {
            table[byte][lane] = if byte & (1 << lane) != 0 { -1 } else { 0 };
            lane += 1;
        }
        byte += 1;
    }
    table
}

static BYTE_MASKS: [[i32; 8]; 256] = masks();

struct Batch {
    status: Vec<i32>,
    quantity: Vec<i32>,
    amount: Vec<i32>,
    bitmap: Option<Vec<u8>>,
}

impl Batch {
    fn make(n: usize, nullable: bool) -> Self {
        let mut state = 0x7461_9c25_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut batch = Self {
            status: Vec::with_capacity(n),
            quantity: Vec::with_capacity(n),
            amount: Vec::with_capacity(n),
            bitmap: nullable.then(|| vec![0; n.div_ceil(8)]),
        };
        for row in 0..n {
            batch.status.push(if next() & 1 == 0 { PAID } else { 1 });
            batch.quantity.push(match next() % 16 {
                0 => i32::MIN,
                1 => i32::MAX,
                k => k as i32 - 4,
            });
            batch.amount.push(next() as i32);
            let valid = next() & 7 != 0;
            if let Some(bits) = &mut batch.bitmap
                && valid
            {
                bits[row / 8] |= 1 << (row % 8);
            }
        }
        batch
    }

    fn oracle(&self) -> Vec<i32> {
        (0..self.amount.len())
            .map(|i| {
                let valid = self
                    .bitmap
                    .as_ref()
                    .is_none_or(|b| b[i / 8] & (1 << (i % 8)) != 0);
                let quantity = valid.then_some(self.quantity[i]);
                if self.status[i] == PAID && quantity.is_some_and(|q| q > MINIMUM_EXCLUSIVE) {
                    self.amount[i]
                } else {
                    0
                }
            })
            .collect()
    }

    fn cpp(&self, out: &mut [i32], clustered: bool) {
        assert_eq!(out.len(), self.amount.len());
        // Private construction guarantees matching column lengths and one bit per row.
        unsafe {
            sql_cpp_project(
                self.status.as_ptr(),
                self.quantity.as_ptr(),
                self.amount.as_ptr(),
                self.bitmap
                    .as_ref()
                    .map_or(std::ptr::null(), |b| b.as_ptr()),
                out.as_mut_ptr(),
                out.len(),
                clustered,
            );
        }
    }

    fn columns<'a>(&'a self, scratch: &'a mut [i32]) -> [&'a [i32]; 4] {
        let validity = if let Some(bits) = &self.bitmap {
            assert_eq!(scratch.len(), self.amount.len());
            for (word, byte) in scratch.chunks_mut(8).zip(bits) {
                word.copy_from_slice(&BYTE_MASKS[usize::from(*byte)][..word.len()]);
            }
            &*scratch
        } else {
            // The program schema still includes the unused last parameter.
            // Reuse an existing slice. No mask is read or allocated on this path.
            &self.quantity
        };
        [&self.status, &self.quantity, &self.amount, validity]
    }
}

#[kernel]
fn project(
    #[comptime] nullable: bool,
    status: &[i32],
    qty: &[i32],
    amount: &[i32],
    valid: &[i32],
    out: &mut [i32],
) -> i32 {
    for i in 0..out.len() as i32 {
        let validity = if nullable { valid[i as usize] } else { -1 };
        out[i as usize] = attributed_amount(
            nullable,
            PAID,
            MINIMUM_EXCLUSIVE,
            status[i as usize],
            qty[i as usize],
            amount[i as usize],
            validity,
        );
    }
    out.len() as i32
}

fn compile(nullable: bool) -> Result<impl Executable, String> {
    project::build::<Lancy>(nullable)
}

fn run_query(
    code: &impl Executable,
    columns: [&[i32]; 4],
    out: &mut [i32],
) -> Result<usize, String> {
    use BufferArg::{Read, Write};
    code.call(
        &[],
        &mut [
            Read(columns[0]),
            Read(columns[1]),
            Read(columns[2]),
            Read(columns[3]),
            Write(out),
        ],
    )
    .map(|n| n as usize)
}

fn check_edges() -> Result<(), String> {
    for nullable in [false, true] {
        let code = compile(nullable)?;
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31] {
            let mut batch = Batch::make(n, nullable);
            batch.status.fill(PAID);
            for (i, v) in batch.quantity.iter_mut().enumerate() {
                *v = [i32::MIN, 2, 3, i32::MAX][i % 4];
            }
            if let Some(bitmap) = &mut batch.bitmap {
                bitmap.fill(0b1011_0101);
            }
            let expected = batch.oracle();
            let mut cpp = vec![1; n];
            batch.cpp(&mut cpp, false);
            let mut ir = vec![2; n];
            let mut scratch = vec![0; n];
            let count = run_query(&code, batch.columns(&mut scratch), &mut ir)?;
            if count != n || cpp != expected || ir != expected {
                return Err(format!(
                    "SQL: incorrect {n}-row tail or NULL handling (nullable={nullable})"
                ));
            }
        }
    }
    Ok(())
}

fn mixed(n: usize, nullable: bool, code: &impl Executable) -> Result<(), String> {
    let batch = Batch::make(n, nullable);
    let expected = batch.oracle();
    let mut scratch = vec![0; if nullable { n } else { 0 }];
    let label = format!(
        "sql/{}/n={n}",
        if nullable {
            "packed-nullable"
        } else {
            "all-valid"
        }
    );
    crate::compare(
        &label,
        n,
        &expected,
        |out| batch.cpp(out, false),
        |out| {
            let columns = batch.columns(&mut scratch);
            assert_eq!(run_query(code, columns, out).unwrap(), n);
        },
    )
}

fn clustered(code: &impl Executable) -> Result<(), String> {
    let n = 65_539;
    let mut batch = Batch::make(n, false);
    batch.status.fill(0);
    let mut out = vec![i32::MAX; n];
    run_query(code, batch.columns(&mut []), &mut out)?;
    if out != batch.oracle() {
        return Err("SQL: incorrect result for clustered input".into());
    }
    crate::compare(
        "sql/clustered-rejected",
        n,
        &batch.oracle(),
        |out| batch.cpp(out, true),
        |out| {
            run_query(code, batch.columns(&mut []), out).unwrap();
        },
    )?;

    // Reuse the installed code after the input distribution changes.
    batch.status[1003..].fill(PAID);
    batch.quantity[1003] = 3;
    batch.amount[1003] = i32::MIN;
    run_query(code, batch.columns(&mut []), &mut out)?;
    let expected = batch.oracle();
    let mut cpp = vec![0; n];
    batch.cpp(&mut cpp, true);
    if out != expected || cpp != expected {
        return Err("SQL: wrong output after the input distribution changed".into());
    }
    Ok(())
}

pub fn run() -> Result<(), String> {
    check_edges()?;
    let nullable = crate::build("sql/nullable projection", || compile(true))?;
    let all_valid = crate::build("sql/all-valid projection", || compile(false))?;
    mixed(31, true, &nullable)?;
    mixed(65_539, true, &nullable)?;
    mixed(65_539, false, &all_valid)?;
    clustered(&all_valid)
}

#[cfg(test)]
mod tests {
    #[test]
    fn projection_checks_nulls_and_tails() {
        super::check_edges().unwrap();
    }
}
