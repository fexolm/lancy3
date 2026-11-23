//! ELF64 symbol files for the in-process GDB reader. The executable bytes
//! already live in the inferior, so text sections describe their address ranges.

use crate::Function;
use gimli::LittleEndian;
use gimli::write::{EndianVec, Sections};

const HEADER_SIZE: usize = 64;
const SECTION_SIZE: u16 = 64;
const SYMBOL_SIZE: u64 = 24;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_NOBITS: u32 = 8;
const SHF_ALLOC_EXEC: u64 = 0x6;

#[derive(Default)]
struct Section {
    name: u32,
    ty: u32,
    flags: u64,
    address: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    align: u64,
    entry_size: u64,
}

fn string(table: &mut Vec<u8>, s: &str) -> splice::Result<u32> {
    let index = u32::try_from(table.len()).map_err(|_| "JIT string table is too large")?;
    table.extend_from_slice(s.as_bytes());
    table.push(0);
    Ok(index)
}

fn data(out: &mut Vec<u8>, name: u32, bytes: &[u8], ty: u32, align: u64) -> Section {
    out.resize(out.len().next_multiple_of(align as usize), 0);
    let offset = out.len() as u64;
    out.extend_from_slice(bytes);
    Section {
        name,
        ty,
        offset,
        size: bytes.len() as u64,
        align,
        ..Section::default()
    }
}

pub(super) fn symbols(
    functions: &[Function],
    debug: &Sections<EndianVec<LittleEndian>>,
) -> splice::Result<Vec<u8>> {
    let mut out = vec![0; HEADER_SIZE];
    let mut section_names = vec![0];
    let mut sections = vec![Section::default()];
    let mut symbol_names = vec![0];
    let mut symbols = vec![0; SYMBOL_SIZE as usize];
    for (i, f) in functions.iter().enumerate() {
        let index = u16::try_from(sections.len()).map_err(|_| "too many JIT text sections")?;
        sections.push(Section {
            name: string(&mut section_names, &format!(".text.{i}"))?,
            ty: SHT_NOBITS,
            flags: SHF_ALLOC_EXEC,
            address: f.address,
            size: f.size,
            align: 1,
            ..Section::default()
        });
        symbols.extend(string(&mut symbol_names, &f.name)?.to_le_bytes());
        symbols.extend([0x12, 0]); // STB_GLOBAL | STT_FUNC, STV_DEFAULT
        symbols.extend(index.to_le_bytes());
        symbols.extend(f.address.to_le_bytes());
        symbols.extend(f.size.to_le_bytes());
    }
    debug.for_each(|id, section| -> splice::Result<()> {
        if !section.slice().is_empty() {
            let name = string(&mut section_names, id.name())?;
            sections.push(data(&mut out, name, section.slice(), SHT_PROGBITS, 1));
        }
        Ok(())
    })?;
    let strtab = sections.len() as u32;
    sections.push(data(
        &mut out,
        string(&mut section_names, ".strtab")?,
        &symbol_names,
        SHT_STRTAB,
        1,
    ));
    sections.push(Section {
        link: strtab,
        info: 1, // The null symbol is the only local symbol.
        entry_size: SYMBOL_SIZE,
        ..data(
            &mut out,
            string(&mut section_names, ".symtab")?,
            &symbols,
            SHT_SYMTAB,
            8,
        )
    });
    let shstrtab = u16::try_from(sections.len()).map_err(|_| "too many JIT ELF sections")?;
    let name = string(&mut section_names, ".shstrtab")?;
    sections.push(data(&mut out, name, &section_names, SHT_STRTAB, 1));
    let count = shstrtab
        .checked_add(1)
        .filter(|n| *n < 0xff00)
        .ok_or("too many JIT ELF sections")?;
    out.resize(out.len().next_multiple_of(8), 0);
    let section_offset = out.len() as u64;
    for s in sections {
        out.extend(s.name.to_le_bytes());
        out.extend(s.ty.to_le_bytes());
        out.extend(s.flags.to_le_bytes());
        out.extend(s.address.to_le_bytes());
        out.extend(s.offset.to_le_bytes());
        out.extend(s.size.to_le_bytes());
        out.extend(s.link.to_le_bytes());
        out.extend(s.info.to_le_bytes());
        out.extend(s.align.to_le_bytes());
        out.extend(s.entry_size.to_le_bytes());
    }
    out[..8].copy_from_slice(b"\x7fELF\x02\x01\x01\0"); // ELF64, little endian, version 1.
    out[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC, symbol values are absolute.
    out[18..20].copy_from_slice(&MACHINE.to_le_bytes());
    out[20..24].copy_from_slice(&1u32.to_le_bytes());
    out[40..48].copy_from_slice(&section_offset.to_le_bytes());
    out[52..54].copy_from_slice(&(HEADER_SIZE as u16).to_le_bytes());
    out[58..60].copy_from_slice(&SECTION_SIZE.to_le_bytes());
    out[60..62].copy_from_slice(&count.to_le_bytes());
    out[62..64].copy_from_slice(&shstrtab.to_le_bytes());
    Ok(out)
}

/// ELF `e_machine` of the symbol file: the host architecture.
#[cfg(target_arch = "x86_64")]
const MACHINE: u16 = 62; // EM_X86_64
#[cfg(target_arch = "aarch64")]
const MACHINE: u16 = 183; // EM_AARCH64
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("splice-jit describes JIT code only for x86_64 and AArch64 hosts");
