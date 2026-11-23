use cranelift_codegen::{
    Context,
    ir::DebugTag,
    isa::unwind::{UnwindInfo, UnwindInfoKind, systemv},
};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use splice_core::hir::Instr;
use splice_jit::gimli::write::{Address, FrameTable};

pub(super) struct Pending<'a> {
    source: &'a splice_core::Function,
    size: u32,
    points: Vec<(u32, u32, u32)>,
    unwind: Option<systemv::UnwindInfo>,
}

impl<'a> Pending<'a> {
    pub fn new(
        module: &JITModule,
        ctx: &Context,
        source: &'a splice_core::Function,
        debug: bool,
    ) -> splice_core::Result<Self> {
        let code = ctx.compiled_code().expect("just defined");
        let mut points = Vec::new();
        for list in code.buffer.debug_tags() {
            if let [DebugTag::User(id), DebugTag::StackSlot(slot)] = *list.tags {
                let frame = code.buffer.frame_layout().ok_or_else(|| {
                    splice_core::BuildError::backend("cranelift", "missing Cranelift frame layout")
                })?;
                points.push((list.offset, id, frame.stackslots[slot].offset));
            }
        }
        let unwind = if debug {
            match code
                .create_unwind_info_of_kind(module.isa(), UnwindInfoKind::SystemV)
                .map_err(|e| splice_core::BuildError::backend("cranelift", e))?
            {
                Some(UnwindInfo::SystemV(info)) => Some(info),
                _ => None,
            }
        } else {
            None
        };
        Ok(Self {
            source,
            size: code.buffer.total_size(),
            points,
            unwind,
        })
    }
}

pub(super) fn publish(
    module: &JITModule,
    pending: Vec<(FuncId, Pending<'_>)>,
    debug: bool,
) -> splice_core::Result<Option<splice_jit::Registration>> {
    use splice_jit::gimli::{AArch64, X86_64};
    let sp = if cfg!(target_arch = "x86_64") {
        X86_64::RSP
    } else {
        AArch64::SP
    };
    let mut functions = Vec::with_capacity(pending.len());
    let mut frames = FrameTable::default();
    let cie = debug
        .then(|| {
            module
                .isa()
                .create_systemv_cie()
                .map(|cie| frames.add_cie(cie))
        })
        .flatten();
    for (id, info) in pending {
        let address = module.get_finalized_function(id) as u64;
        let name = info.source.source().map_or("kernel", |source| source.name);
        let mut f = splice_jit::Function {
            name: format!("splice_jit::{name}"),
            address,
            size: u64::from(info.size),
            statements: Vec::new(),
        };
        for (offset, index, base) in info.points {
            let Instr::Debug(point) = &info.source.instructions()[index as usize] else {
                unreachable!("only sequence points bear Splice debug tags")
            };
            let start = address + u64::from(offset);
            f.statement(info.source, point, start, sp.0, i64::from(base));
        }
        if let (Some(cie), Some(info)) = (cie, info.unwind) {
            frames.add_fde(cie, info.to_fde(Address::Constant(address)));
        }
        functions.push(f);
    }
    // perf record sets PERF_BUILDID_DIR, which activates Cranelift's
    // native writer. Its declarations already carry the source names.
    if !cfg!(target_os = "linux") || std::env::var_os("PERF_BUILDID_DIR").is_none() {
        splice_jit::profile(&functions);
    }
    debug
        .then(|| splice_jit::Registration::new(&functions, &frames))
        .transpose()
}
