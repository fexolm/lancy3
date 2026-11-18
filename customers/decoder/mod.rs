use lancy_jit::{NativeKernel, kernel};

const RECORD_SIZE: usize = 8;
const CHUNK: usize = 1024;
const MISSING: i32 = i32::MIN;

struct Schema {
    channel: u8,
    gain: i32,
}

const SCHEMA: Schema = Schema {
    channel: 7,
    gain: 3,
};

#[kernel]
fn project(#[comptime] schema: &Schema, header: i32, sample: i32, zero_point: i32) -> i32 {
    let tag = (i32::from(schema.channel) << 8) | 1;
    if (header & 0xff01) == tag {
        (sample - zero_point) * schema.gain
    } else {
        MISSING
    }
}

unsafe extern "C" {
    fn decoder_batch(bytes: *const u8, len: usize, out: *mut i32, count: usize) -> bool;
}

fn cpp(bytes: &[u8], out: &mut [i32]) -> bool {
    // Both pointers span exactly the lengths passed; C++ checks the shape
    // before any read or write, even for the malformed-frame test.
    unsafe { decoder_batch(bytes.as_ptr(), bytes.len(), out.as_mut_ptr(), out.len()) }
}

struct Scratch {
    headers: Vec<i32>,
    samples: Vec<i32>,
    zeros: Vec<i32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            headers: vec![0; CHUNK],
            samples: vec![0; CHUNK],
            zeros: vec![0; CHUNK],
        }
    }
}

fn decode_ir(kernel: &NativeKernel, bytes: &[u8], out: &mut [i32], scratch: &mut Scratch) -> bool {
    if !bytes.len().is_multiple_of(RECORD_SIZE) || bytes.len() / RECORD_SIZE != out.len() {
        return false;
    }
    // Reuse 12 KiB of buffers. The wire-to-column pass is part of the timed
    // invocation, since the public IR currently accepts aligned i32 columns.
    for (wire, dst) in bytes.chunks(CHUNK * RECORD_SIZE).zip(out.chunks_mut(CHUNK)) {
        let n = dst.len();
        let headers = &mut scratch.headers[..n];
        let samples = &mut scratch.samples[..n];
        let zeros = &mut scratch.zeros[..n];
        let fields = headers
            .iter_mut()
            .zip(samples.iter_mut())
            .zip(zeros.iter_mut());
        for (((header, sample), zero), record) in fields.zip(wire.chunks_exact(RECORD_SIZE)) {
            *header = i32::from(u16::from_le_bytes([record[0], record[1]]));
            *sample = i32::from(i16::from_le_bytes([record[2], record[3]]));
            *zero = i32::from(i16::from_le_bytes([record[4], record[5]]));
        }
        let written = kernel
            .run(&[headers, samples, zeros], dst)
            .expect("wire projection");
        assert_eq!(written, n, "unguarded wire projection must complete");
    }
    true
}

// A leading pad deliberately puts the wire slice on a byte boundary.
// The expected output comes from logical records before their serialization.
fn fixture(rows: usize) -> (Vec<u8>, Vec<i32>) {
    let mut wire = Vec::with_capacity(1 + rows * RECORD_SIZE);
    wire.push(0xd3);
    let mut expected = Vec::with_capacity(rows);
    let mut state = 0x7279_52e9u32;
    for i in 0..rows {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let flags = if i < 2 { 1 } else { state as u8 };
        let channel = if i >= 2 && (state >> 8).is_multiple_of(3) {
            5
        } else {
            7
        };
        let (sample, zero) = match i % 31 {
            0 => (i16::MIN, i16::MAX),
            1 => (i16::MAX, i16::MIN),
            _ => ((state >> 16) as i16, state.rotate_left(7) as i16),
        };
        wire.extend_from_slice(&[flags, channel]);
        wire.extend_from_slice(&sample.to_le_bytes());
        wire.extend_from_slice(&zero.to_le_bytes());
        wire.extend_from_slice(&(i as u16).to_le_bytes());
        expected.push(if flags & 1 != 0 && channel == 7 {
            3 * (i32::from(sample) - i32::from(zero))
        } else {
            MISSING
        });
    }
    (wire, expected)
}

fn verify_parts(kernel: &NativeKernel, scratch: &mut Scratch) {
    for count in [0, 1, 2, 3, 4, 5, 7, 8, 9, CHUNK - 1, CHUNK, CHUNK + 1] {
        let (wire, expected) = fixture(count);
        let mut output = vec![0; count];
        assert!(cpp(&wire[1..], &mut output));
        assert_eq!(output, expected);
        output.fill(0);
        assert!(decode_ir(kernel, &wire[1..], &mut output, scratch));
        assert_eq!(output, expected);
    }
    let (wire, _) = fixture(5);
    let mut output = [0x2222_2222; 5];
    for bad in [&wire[1..wire.len() - 1], &wire[1..1 + 4 * RECORD_SIZE]] {
        assert!(!cpp(bad, &mut output));
        assert_eq!(output, [0x2222_2222; 5]);
        assert!(!decode_ir(kernel, bad, &mut output, scratch));
        assert_eq!(output, [0x2222_2222; 5]);
    }
}

pub fn run() -> Result<(), String> {
    let code = crate::build("decoder / schema to SIMD", || {
        let program = project::build(&SCHEMA)?
            .vectorize()?
            .ok_or("decoder projection requires SIMD")?;
        NativeKernel::specialize(&program, &[])
    })?;
    let mut scratch = Scratch::new();
    verify_parts(&code, &mut scratch);
    for count in [1, 31, 65_539] {
        let (wire, expected) = fixture(count);
        let bytes = &wire[1..];
        crate::compare(
            &format!("decoder / wire to output / {count} records"),
            count,
            &expected,
            |out| assert!(cpp(bytes, out)),
            |out| assert!(decode_ir(&code, bytes, out, &mut scratch)),
        )?;
    }
    Ok(())
}
