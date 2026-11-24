use crate::lower::DebugSource;
use lancy::ScalarKernel;
use splice_jit::gimli::write::{
    Address, CallFrameInstruction as Cfi, CommonInformationEntry, Expression,
    FrameDescriptionEntry, FrameTable,
};
use splice_jit::gimli::{AArch64, Encoding, Format, X86_64};

pub(super) fn name(source: Option<&splice_core::frontend::KernelSource>) -> String {
    let name = source.map_or("kernel", |source| source.name);
    format!("splice_jit::{name}")
}

pub(super) fn publish(
    code: &ScalarKernel,
    sources: &[DebugSource],
) -> splice_core::Result<splice_jit::Registration> {
    let mut frames = FrameTable::default();
    let (sp, ra, workspace) = if cfg!(target_arch = "x86_64") {
        (X86_64::RSP, X86_64::RA, X86_64::R11)
    } else {
        (AArch64::SP, AArch64::X30, AArch64::X19)
    };
    let cie = frames.add_cie(CommonInformationEntry::new(
        Encoding {
            format: Format::Dwarf32,
            version: 4,
            address_size: 8,
        },
        1,
        -8,
        ra,
    ));
    let address = code.code_address() as u64;
    let mut functions = Vec::new();
    for body in code.debug_functions() {
        let marker = sources
            .get(body.points[0].0 as usize)
            .expect("debug body marker");
        let mut function = splice_jit::Function {
            name: name(marker.function),
            address: address + body.start,
            size: body.end - body.start,
            statements: Vec::new(),
        };
        for &(id, at) in &body.points {
            if let Some((point, slots)) = &sources[id as usize].point {
                function.statement(
                    point,
                    slots.iter().copied(),
                    address + at,
                    workspace.0,
                    body.workspace_offset as i64,
                );
            }
        }
        let frame = &body.frame;
        let mut fde = FrameDescriptionEntry::new(
            Address::Constant(address + frame.start),
            (frame.end - frame.start) as u32,
        );
        fde.add_instruction(0, Cfi::Cfa(sp, frame.cfa_from_sp));
        for &(reg, offset) in &frame.saved {
            fde.add_instruction(0, Cfi::Offset(splice_jit::gimli::Register(reg), offset));
        }
        for &(reg, offset) in &frame.workspace_saved {
            let mut expression = Expression::new();
            expression.op_breg(workspace, offset);
            fde.add_instruction(
                0,
                Cfi::Expression(splice_jit::gimli::Register(reg), expression),
            );
        }
        frames.add_fde(cie, fde);
        functions.push(function);
    }
    splice_jit::profile(&functions);
    splice_jit::Registration::new(&functions, &frames)
}
