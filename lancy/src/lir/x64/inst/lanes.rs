//! Lower operations without a native rule into x64 instructions in lane
//! order, through stack slots. All opcode, type and width decisions happen
//! during emission. 128-bit lanes and math library routines call the shared
//! numeric fallback (`lir::fallback`) instead; native rules (X64-01) replace
//! this module.
mod convert;
mod float;
mod integer;

use crate::hir::{Binary, Lane, Ternary, Type, Unary};
use crate::lir::x64::emit::{R32, R64, bind};
use crate::support::numeric::Action;
use iced_x86::{IcedError, code_asm::*};
type Result = std::result::Result<(), IcedError>;
/// Lane index, division by zero and division overflow exits.
type Faults = [Option<CodeLabel>; 3];

// Both entry ABIs have rsp % 16 == 8. Separate 64-byte slots hold three
// inputs and the result; the expansion uses the following four scratch words.
pub(in crate::lir::x64) const INPUT: [i32; 3] = [32, 96, 160];
pub(in crate::lir::x64) const OUTPUT: i32 = 224;
const TEMP: i32 = 288;
pub(in crate::lir::x64) const WORKSPACE: i32 = TEMP + 32;
pub(in crate::lir::x64) const FRAME: i32 = WORKSPACE + 8;

fn copy(
    a: &mut CodeAssembler,
    to: AsmMemoryOperand,
    from: AsmMemoryOperand,
    size: usize,
) -> Result {
    match size {
        1 => {
            a.mov(r10b, byte_ptr(from))?;
            a.mov(byte_ptr(to), r10b)
        }
        2 => {
            a.mov(r10w, word_ptr(from))?;
            a.mov(word_ptr(to), r10w)
        }
        4 => {
            a.mov(r10d, dword_ptr(from))?;
            a.mov(dword_ptr(to), r10d)
        }
        8 => {
            a.mov(r10, qword_ptr(from))?;
            a.mov(qword_ptr(to), r10)
        }
        _ => unreachable!(),
    }
}

// Load one integer into a full machine word, with the requested extension.
fn word(a: &mut CodeAssembler, dst: usize, off: i32, size: usize, signed: bool) -> Result {
    let mem = ptr(rsp + off);
    let q = R64[dst];
    let d = R32[dst];
    match (size, signed) {
        (1, true) => a.movsx(q, byte_ptr(mem)),
        (2, true) => a.movsx(q, word_ptr(mem)),
        (4, true) => a.movsxd(q, dword_ptr(mem)),
        (1, false) => a.movzx(d, byte_ptr(mem)),
        (2, false) => a.movzx(d, word_ptr(mem)),
        (4, false) => a.mov(d, dword_ptr(mem)),
        (8, _) => a.mov(q, qword_ptr(mem)),
        _ => unreachable!(),
    }
}
fn put(a: &mut CodeAssembler, off: i32, size: usize) -> Result {
    match size {
        1 => a.mov(byte_ptr(rsp + off), al),
        2 => a.mov(word_ptr(rsp + off), ax),
        4 => a.mov(dword_ptr(rsp + off), eax),
        8 => a.mov(qword_ptr(rsp + off), rax),
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy)]
struct Args {
    x: i32,
    y: i32,
    z: i32,
    out: i32,
}
fn elem(
    a: &mut CodeAssembler,
    action: Action,
    input: Type,
    output: Type,
    p: Args,
    faults: &mut Faults,
) -> Result {
    let Args { x, y, z, out, .. } = p;
    let lane = input.lane().unwrap();
    let size = lane.bytes();
    let dest = output.lane().unwrap();
    match action {
        Action::Cast | Action::Widen(_) | Action::Narrow => convert::emit(a, lane, dest, x, out),
        Action::Ternary(Ternary::Select) => {
            if matches!(input, Type::Vector(_)) {
                a.test(byte_ptr(rsp + x + size as i32 - 1), 0x80)?;
            } else {
                a.cmp(dword_ptr(rsp + x), 0)?;
            }
            let no = a.create_label();
            let end = a.create_label();
            a.je(no)?;
            copy(a, ptr(rsp + out), ptr(rsp + y), size)?;
            a.jmp(end)?;
            bind(a, no)?;
            copy(a, ptr(rsp + out), ptr(rsp + z), size)?;
            bind(a, end)
        }
        Action::Unary(Unary::Not)
        | Action::Binary(Binary::And | Binary::Or | Binary::Xor)
        | Action::Ternary(Ternary::BitSelect) => {
            let part = size.min(8);
            for i in (0..size as i32).step_by(part) {
                word(a, 0, x + i, part, false)?;
                match action {
                    Action::Unary(_) => a.not(rax)?,
                    Action::Binary(op) => {
                        word(a, 1, y + i, part, false)?;
                        match op {
                            Binary::And => a.and(rax, rcx),
                            Binary::Or => a.or(rax, rcx),
                            _ => a.xor(rax, rcx),
                        }?;
                    }
                    _ => {
                        word(a, 1, y + i, part, false)?;
                        word(a, 2, z + i, part, false)?;
                        a.xor(rcx, rdx)?;
                        a.and(rax, rcx)?;
                        a.xor(rax, rdx)?;
                    }
                }
                put(a, out + i, part)?;
            }
            Ok(())
        }
        _ if lane.float() => float::emit(a, action, input, output, p),
        _ => integer::emit(a, action, input, output, p, faults),
    }
}
/// Whether the inline expansion covers `action`: lanes of at most eight
/// bytes and no math library routine. The 128-bit and math operations call
/// the shared numeric fallback. Native rules replace this expansion (X64-01).
pub(in crate::lir::x64) fn inline(action: Action, input: Type, output: Type) -> bool {
    let small = |ty: Type| ty.lane().is_some_and(|lane| lane.bytes() <= 8);
    small(input)
        && small(output)
        && !input
            .lane()
            .is_some_and(|lane| float::needs_library(action, lane))
}

/// Expand `action`, which [`inline`] accepts, over `width`-byte vectors.
pub(in crate::lir::x64) fn emit(
    a: &mut CodeAssembler,
    width: usize,
    action: Action,
    input: Type,
    output: Type,
    faults: &mut Faults,
) -> Result {
    let lane = input.lane().unwrap();
    let size = lane.bytes();
    let n = size as i32;
    let dest = output.lane().unwrap();
    let d = dest.bytes() as i32;
    let [x, y, z] = INPUT;
    let out = OUTPUT;
    let count = width as i32 / n;
    match action {
        Action::AnyTrue | Action::AllTrue => {
            let all = matches!(action, Action::AllTrue);
            let stop = a.create_label();
            let end = a.create_label();
            for i in 0..count {
                word(a, 0, x + i * n, size, false)?;
                a.test(rax, rax)?;
                if all {
                    a.je(stop)?;
                } else {
                    a.jne(stop)?;
                }
            }
            a.mov(eax, -i32::from(all))?;
            a.jmp(end)?;
            bind(a, stop)?;
            a.mov(eax, -i32::from(!all))?;
            bind(a, end)?;
            put(a, out, 4)
        }
        Action::Splat => {
            for i in 0..count {
                copy(a, ptr(rsp + out + i * n), ptr(rsp + x), size)?;
            }
            Ok(())
        }
        Action::Extract | Action::Insert => {
            let fail = *faults[0].get_or_insert_with(|| a.create_label());
            a.mov(eax, dword_ptr(rsp + y))?;
            a.cmp(eax, count)?;
            a.jae(fail)?;
            a.shl(rax, size.trailing_zeros() as i32)?;
            if matches!(action, Action::Extract) {
                copy(a, ptr(rsp + out), ptr(rsp + rax + x), size)
            } else {
                for i in (0..width as i32).step_by(8) {
                    copy(a, ptr(rsp + out + i), ptr(rsp + x + i), 8)?;
                }
                copy(a, ptr(rsp + rax + out), ptr(rsp + z), size)
            }
        }
        Action::Compress | Action::Expand => {
            for i in (0..width as i32).step_by(8) {
                a.mov(qword_ptr(rsp + out + i), 0)?;
            }
            a.xor(r9d, r9d)?;
            for i in 0..count {
                let skip = a.create_label();
                a.test(byte_ptr(rsp + y + (i + 1) * n - 1), 0x80)?;
                a.jz(skip)?;
                let (to, from) = if matches!(action, Action::Compress) {
                    (ptr(rsp + r9 + out), ptr(rsp + x + i * n))
                } else {
                    (ptr(rsp + out + i * n), ptr(rsp + r9 + x))
                };
                copy(a, to, from, size)?;
                a.add(r9d, n)?;
                bind(a, skip)?;
            }
            Ok(())
        }
        Action::Shuffle => {
            for i in 0..count {
                let skip = a.create_label();
                let first = a.create_label();
                let at = out + i * n;
                a.xor(eax, eax)?;
                put(a, at, size)?;
                word(a, 0, z + i * n, size, false)?;
                a.cmp(rax, 2 * count)?;
                a.jae(skip)?;
                a.shl(rax, size.trailing_zeros() as i32)?;
                a.cmp(eax, width as i32)?;
                a.jb(first)?;
                a.add(eax, y - x - width as i32)?;
                bind(a, first)?;
                copy(a, ptr(rsp + at), ptr(rsp + rax + x), size)?;
                bind(a, skip)?;
            }
            Ok(())
        }
        Action::Reduce(op) => {
            copy(a, ptr(rsp + out), ptr(rsp + x), size)?;
            let scalar = lane.scalar();
            for i in 1..count {
                elem(
                    a,
                    Action::Binary(op),
                    scalar,
                    scalar,
                    Args {
                        x: out,
                        y: x + i * n,
                        z,
                        out,
                    },
                    faults,
                )?;
            }
            Ok(())
        }
        Action::Iota | Action::Bitcast => unreachable!("selected before compound lowering"),
        Action::FirstLanes | Action::MaskCount => {
            unreachable!("canonical::expand defines predicate operations")
        }
        _ => {
            let len = output.bytes(width) as i32 / d;
            for i in 0..len {
                let offset = i * n
                    + if matches!(action, Action::Widen(true)) {
                        width as i32 / 2
                    } else {
                        0
                    };
                let source = if offset >= width as i32 {
                    y + offset - width as i32
                } else {
                    x + offset
                };
                elem(
                    a,
                    action,
                    input,
                    output,
                    Args {
                        x: source,
                        y: y + i * n,
                        z: z + i * n,
                        out: out + i * d,
                    },
                    faults,
                )?;
            }
            Ok(())
        }
    }
}
