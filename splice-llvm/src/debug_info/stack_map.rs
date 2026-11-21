//! Decode the LLVM 19 and 21 stack map format, version 3. Each instruction
//! record refers to one source point and its static snapshot alloca.
use std::collections::HashMap;

pub(super) struct Point {
    pub index: usize,
    pub offset: u32,
    pub register: u16,
    pub displacement: i32,
}

pub(super) type Points = HashMap<usize, Vec<Point>>;

pub(super) fn read(bytes: &[u8]) -> splice::Result<Points> {
    let mut bytes = bytes;
    let header = take::<16>(&mut bytes)?;
    if header[..4] != [3, 0, 0, 0] {
        return Err("unsupported LLVM stack map version".into());
    }
    let word = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap()) as usize;
    // The record ID contains our kernel ID, so function address relocations
    // and LLVM's constant pool are unnecessary.
    skip(&mut bytes, 24 * word(4) + 8 * word(8))?;
    let mut points = Points::new();
    for _ in 0..word(12) {
        let record = take::<16>(&mut bytes)?;
        let index = u32::from_le_bytes(record[..4].try_into().unwrap()) as usize;
        let function = u32::from_le_bytes(record[4..8].try_into().unwrap()) as usize;
        let offset = u32::from_le_bytes(record[8..12].try_into().unwrap());
        if record[12..] != [0, 0, 1, 0] {
            return Err("LLVM debug record must contain one snapshot address".into());
        }
        let location = take::<12>(&mut bytes)?;
        if location[..4] != [2, 0, 8, 0] {
            return Err("LLVM did not encode the snapshot as a direct stack address".into());
        }
        let register = u16::from_le_bytes(location[4..6].try_into().unwrap());
        let displacement = i32::from_le_bytes(location[8..12].try_into().unwrap());
        skip(&mut bytes, 4)?; // Align the one 12-byte location to 8 bytes.
        let live = take::<4>(&mut bytes)?;
        let live_outs = usize::from(u16::from_le_bytes(live[2..].try_into().unwrap()));
        skip(&mut bytes, (4 + 4 * live_outs).next_multiple_of(8) - 4)?;
        points.entry(function).or_default().push(Point {
            index,
            offset,
            register,
            displacement,
        });
    }
    Ok(points)
}

fn skip(bytes: &mut &[u8], size: usize) -> splice::Result<()> {
    *bytes = bytes.get(size..).ok_or("truncated LLVM stack map")?;
    Ok(())
}

fn take<const N: usize>(bytes: &mut &[u8]) -> splice::Result<[u8; N]> {
    let (record, rest) = bytes
        .split_first_chunk::<N>()
        .ok_or("truncated LLVM stack map")?;
    *bytes = rest;
    Ok(*record)
}
