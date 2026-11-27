//! The Intel-syntax text of emitted x64 code, for `@asm` filetests.
use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};
use std::fmt::Write;

fn list(bytes: &[u8]) -> String {
    let hex: Vec<_> = bytes.iter().map(|b| format!("{b:02x}")).collect();
    hex.join(" ")
}

pub(crate) fn disassemble(bytes: &[u8]) -> Result<String, String> {
    let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    let mut out = String::new();
    while decoder.can_decode() {
        let start = decoder.position();
        // iced 1.22 has no AVX10.2 MAP5 conversion entries. Recognize exactly the
        // unmasked register forms emitted by this backend; leave all other validation to iced.
        if let [0x62, p0, p1, p2, op, modrm, ..] = bytes[start..]
            && p0 & 0x5f == 0x55
            && matches!(p1, 0x7c | 0xfd)
            && matches!(p2, 0x08 | 0x28 | 0x48)
            && matches!(op, 0x6c | 0x6d)
            && modrm >> 6 == 3
        {
            let size = match p2 {
                0x08 => "xmm",
                0x28 => "ymm",
                _ => "zmm",
            };
            let from = if p1 == 0x7c { "ps" } else { "pd" };
            let to = if p1 == 0x7c { "dq" } else { "qq" };
            let unsigned = if op == 0x6c { "u" } else { "" };
            let d = (!p0 >> 4 & 8) | (modrm >> 3 & 7);
            let r = (!p0 >> 2 & 8) | (modrm & 7);
            let encoding = list(&bytes[start..start + 6]);
            writeln!(
                out,
                "{start:04x}  {encoding:<32} vcvtt{from}2{unsigned}{to}s {size}{d},{size}{r}"
            )
            .unwrap();
            decoder.set_position(start + 6).map_err(|e| e.to_string())?;
            decoder.set_ip((start + 6) as u64);
            continue;
        }
        let inst = decoder.decode();
        if inst.is_invalid() {
            return Err(format!("invalid x64 instruction at byte {}", inst.ip()));
        }
        let start = inst.ip() as usize;
        let encoding = list(&bytes[start..start + inst.len()]);
        write!(out, "{start:04x}  {encoding:<32} ").unwrap();
        formatter.format(&inst, &mut out);
        out.push('\n');
    }
    Ok(out)
}
