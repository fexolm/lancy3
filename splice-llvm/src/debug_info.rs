use crate::jit::Jit;
use inkwell::memory_buffer::MemoryBuffer;
use splice::{Function, hir::Instr};
use splice_jit::gimli::write::FrameTable;
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
    pub fn read(object: &MemoryBuffer<'_>, debug: bool) -> splice::Result<Self> {
        let file = object.create_binary_file(None).map_err(|e| e.to_string())?;
        let mut symbols = file
            .get_symbols()
            .ok_or("LLVM object has no symbol table")?;
        let mut sections = file.get_sections().ok_or("LLVM object has no sections")?;
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
                    .ok_or("LLVM function missing its text section")?;
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
                    .ok_or("invalid LLVM function range")?;
            }
        }
        if debug {
            let mut sections = file.get_sections().ok_or("LLVM object has no sections")?;
            while let Some(section) = sections.next_section() {
                if section.get_name() == Some(c".llvm_stackmaps") {
                    result.points = stack_map::read(section.get_contents())?;
                    break;
                }
            }
        }
        Ok(result)
    }

    pub fn publish(
        self,
        jit: &Jit,
        source: Option<&Function>,
        kernel_source: Option<&splice::frontend::KernelSource>,
        debug: bool,
    ) -> splice::Result<Option<splice_jit::Registration>> {
        let sources = source
            .filter(|_| debug)
            .map(|source| source.reachable_functions().0);
        let mut functions = Vec::with_capacity(self.functions.len());
        for native in self.functions {
            let address = jit.lookup(&native.name)?;
            let f = native
                .source_id
                .and_then(|id| sources.as_ref()?.get(id).copied());
            let name = f
                .and_then(Function::source)
                .or(kernel_source)
                .map_or("kernel", |src| src.name);
            let name = if f.is_some() || !debug {
                format!("splice_jit::{name}")
            } else {
                format!("splice_jit::{name}::entry")
            };
            let mut function = splice_jit::Function {
                name,
                address,
                size: native.size,
                statements: Vec::new(),
            };
            if let (Some(id), Some(f)) = (native.source_id, f)
                && let Some(points) = self.points.get(&id)
            {
                for location in points {
                    let offset = u64::from(location.offset);
                    if offset >= native.size {
                        return Err(format!(
                            "LLVM sequence point {offset:#x} lies outside {} (size {:#x})",
                            native.name.to_string_lossy(),
                            native.size
                        ));
                    }
                    let Some(Instr::Debug(point)) = f.instructions().get(location.index) else {
                        return Err(
                            "LLVM label does not correspond to a Splice source point".into()
                        );
                    };
                    function.statement(
                        f,
                        point,
                        address + offset,
                        location.register,
                        i64::from(location.displacement),
                    );
                }
            }
            functions.push(function);
        }
        splice_jit::profile(&functions);
        debug
            .then(|| splice_jit::Registration::new(&functions, &FrameTable::default()))
            .transpose()
    }
}
mod stack_map;
