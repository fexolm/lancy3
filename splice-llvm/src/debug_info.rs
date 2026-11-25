use crate::jit::Jit;
use inkwell::memory_buffer::MemoryBuffer;
use splice_jit::Publish;
use std::ffi::CString;

struct NativeFunction {
    name: CString,
    source_id: Option<usize>,
    start: u64,
    size: u64,
    section: usize,
    section_end: u64,
}

#[derive(Default)]
pub(super) struct ObjectSymbols {
    functions: Vec<NativeFunction>,
    points: stack_map::Points,
}

impl ObjectSymbols {
    pub fn read(object: &MemoryBuffer<'_>, debug: bool) -> splice_core::Result<Self> {
        let file = object.create_binary_file(None).map_err(crate::error)?;
        let mut symbols = file
            .get_symbols()
            .ok_or_else(|| crate::error("LLVM object has no symbol table"))?;
        let mut sections = file
            .get_sections()
            .ok_or_else(|| crate::error("LLVM object has no sections"))?;
        let mut result = Self::default();
        while let Some(symbol) = symbols.next_symbol() {
            let Some(name) = symbol.get_name() else {
                continue;
            };
            let Ok(text) = name.to_str() else {
                continue;
            };
            let source_id = if debug {
                text.strip_prefix("kernel")
                    .and_then(|id| id.parse::<usize>().ok())
            } else {
                None
            };
            if text == "splice_entry" || source_id.is_some() {
                sections.move_to_containing_section(&symbol);
                let section = sections
                    .next_section()
                    .ok_or_else(|| crate::error("LLVM function missing its text section"))?;
                result.functions.push(NativeFunction {
                    name: name.into(),
                    source_id,
                    start: symbol.get_address(),
                    size: if cfg!(windows) { 0 } else { symbol.get_size() },
                    // COFF can have several distinct text sections with the
                    // same name and virtual address. Use their object identity.
                    section: section.get_contents().as_ptr() as usize,
                    section_end: section.get_address() + section.get_size(),
                });
            }
        }
        // ELF records function sizes. COFF can omit them, in which case the
        // interval ends at the next function in the section or its boundary.
        for index in 0..result.functions.len() {
            let f = &result.functions[index];
            if f.size == 0 {
                let end = result
                    .functions
                    .iter()
                    .filter(|other| other.section == f.section && other.start > f.start)
                    .map(|other| other.start)
                    .min()
                    .unwrap_or(f.section_end);
                result.functions[index].size = end
                    .checked_sub(f.start)
                    .ok_or_else(|| crate::error("invalid LLVM function range"))?;
            }
        }
        if debug {
            let mut sections = file
                .get_sections()
                .ok_or_else(|| crate::error("LLVM object has no sections"))?;
            while let Some(section) = sections.next_section() {
                if section.get_name() == Some(c".llvm_stackmaps") {
                    result.points = stack_map::read(section.get_contents())?;
                    break;
                }
            }
        }
        Ok(result)
    }

    /// Add the linked functions to `publish`, with source statements in a
    /// debug build. LLVM's frames carry no unwind rows: a debug build keeps
    /// frame pointers, which debuggers unwind through.
    pub fn describe(
        self,
        publish: &mut Publish,
        jit: &Jit,
        staged: &crate::lower::Staged,
        debug: bool,
    ) -> splice_core::Result<()> {
        for native in self.functions {
            let address = jit.lookup(&native.name)?;
            let f = native.source_id.and_then(|id| staged.functions.get(id));
            let name = f
                .and_then(|f| f.source)
                .or(staged.source)
                .map_or("kernel", |src| src.name);
            let function = if f.is_some() || !debug {
                publish.function(name, address, native.size)
            } else {
                publish.function(&format!("{name}::entry"), address, native.size)
            };
            if let (Some(id), Some(f)) = (native.source_id, f)
                && let Some(points) = self.points.get(&id)
            {
                for location in points {
                    let offset = u64::from(location.offset);
                    if offset >= native.size {
                        return Err(crate::error(format!(
                            "LLVM sequence point {offset:#x} lies outside {} (size {:#x})",
                            native.name.to_string_lossy(),
                            native.size
                        )));
                    }
                    let Some((point, slots)) = f.points.get(location.index) else {
                        return Err(crate::error(
                            "LLVM label does not correspond to a Splice source point",
                        ));
                    };
                    function.statement(
                        point,
                        slots.iter().copied(),
                        address + offset,
                        location.register,
                        i64::from(location.displacement),
                    );
                }
            }
        }
        Ok(())
    }
}
mod stack_map;
