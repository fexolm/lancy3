use crate::lower::DebugSource;
use lancy::ScalarKernel;
use splice_jit::Publish;
use splice_jit::gimli::write::{
    Address, CallFrameInstruction as Cfi, CommonInformationEntry, Expression, FrameDescriptionEntry,
};
use splice_jit::gimli::{AArch64, Encoding, Format, X86_64};

/// Add the functions, source statements and unwind rows of a debug build.
pub(super) fn describe(publish: &mut Publish, code: &ScalarKernel, sources: &[DebugSource]) {
    let (sp, ra, workspace) = if cfg!(target_arch = "x86_64") {
        (X86_64::RSP, X86_64::RA, X86_64::R11)
    } else {
        (AArch64::SP, AArch64::X30, AArch64::X19)
    };
    let cie = publish.frames().add_cie(CommonInformationEntry::new(
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
    for body in code.debug_functions() {
        let marker = sources
            .get(body.points[0].0 as usize)
            .expect("debug body marker");
        let function = publish.function(
            marker.function.map_or("kernel", |source| source.name),
            address + body.start,
            body.end - body.start,
        );
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
        publish.frames().add_fde(cie, fde);
    }
}
