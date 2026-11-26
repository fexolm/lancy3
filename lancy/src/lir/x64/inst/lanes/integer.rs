use super::*;
use Binary as B;
use Unary as U;

fn set(a: &mut CodeAssembler, op: B, signed: bool) -> Result {
    match op {
        B::Eq => a.sete(al),
        B::Ne => a.setne(al),
        B::Lt if signed => a.setl(al),
        B::Le if signed => a.setle(al),
        B::Gt if signed => a.setg(al),
        B::Ge if signed => a.setge(al),
        B::Lt => a.setb(al),
        B::Le => a.setbe(al),
        B::Gt => a.seta(al),
        B::Ge => a.setae(al),
        _ => unreachable!(),
    }
}
fn mask(a: &mut CodeAssembler) -> Result {
    a.movzx(eax, al)?;
    a.neg(rax)?;
    a.mov(rdx, rax)
}
fn popcnt(a: &mut CodeAssembler) -> Result {
    // Clear successive set bits. SSE2 targets need not implement POPCNT.
    let next = a.create_label();
    let done = a.create_label();
    a.xor(r9d, r9d)?;
    a.test(rax, rax)?;
    a.je(done)?;
    bind(a, next)?;
    a.lea(r10, ptr(rax - 1))?;
    a.inc(r9d)?;
    a.and(rax, r10)?;
    a.jne(next)?;
    bind(a, done)?;
    a.mov(eax, r9d)
}
fn zeros(a: &mut CodeAssembler, leading: bool, bits: i32) -> Result {
    if leading {
        a.bsr(rax, rax)?;
        a.mov(r10, -1i64)?;
        a.cmovz(rax, r10)?;
        a.mov(r10d, bits - 1)?;
        a.sub(r10, rax)?;
        a.mov(rax, r10)
    } else {
        a.bsf(rax, rax)?;
        a.mov(r10d, bits)?;
        a.cmovz(rax, r10)
    }
}

pub(super) fn emit(
    a: &mut CodeAssembler,
    action: Action,
    input: Type,
    output: Type,
    p: Args,
    faults: &mut Faults,
) -> Result {
    let lane = input.lane().unwrap();
    let size = lane.bytes();
    let signed = lane.signed();
    let bits = (size * 8) as i32;
    let sign_extend = signed
        && !matches!(
            action,
            Action::Unary(U::Popcnt | U::LeadingZeros | U::TrailingZeros)
        );
    word(a, 0, p.x, size, sign_extend)?;
    match action {
        Action::Unary(op) => match op {
            U::Neg => a.neg(rax)?,
            U::Abs if signed => {
                a.cqo()?;
                a.xor(rax, rdx)?;
                a.sub(rax, rdx)?;
            }
            U::Abs => {}
            U::Popcnt => popcnt(a)?,
            U::LeadingZeros | U::TrailingZeros => zeros(a, op == U::LeadingZeros, bits)?,
            _ => unreachable!("verified integer unary"),
        },
        Action::Binary(op) => {
            word(a, 1, p.y, size, signed)?;
            match op {
                B::Add => a.add(rax, rcx)?,
                B::Sub => a.sub(rax, rcx)?,
                B::Mul => a.imul_2(rax, rcx)?,
                B::Div | B::Rem => {
                    let zero = *faults[1].get_or_insert_with(|| a.create_label());
                    a.test(rcx, rcx)?;
                    a.je(zero)?;
                    if signed {
                        let ok = a.create_label();
                        let overflow = *faults[2].get_or_insert_with(|| a.create_label());
                        a.cmp(rcx, -1)?;
                        a.jne(ok)?;
                        a.mov(r10, -(1i128 << (bits - 1)) as i64)?;
                        a.cmp(rax, r10)?;
                        a.je(overflow)?;
                        bind(a, ok)?;
                        a.cqo()?;
                        a.idiv(rcx)?;
                    } else {
                        a.xor(edx, edx)?;
                        a.div(rcx)?;
                    }
                    if op == B::Rem {
                        a.mov(rax, rdx)?;
                    }
                }
                B::Shl | B::Shr | B::RotateLeft | B::RotateRight => {
                    a.and(ecx, bits - 1)?;
                    match op {
                        B::Shl => a.shl(rax, cl)?,
                        B::Shr if signed => a.sar(rax, cl)?,
                        B::Shr => a.shr(rax, cl)?,
                        B::RotateLeft => match size {
                            1 => a.rol(al, cl),
                            2 => a.rol(ax, cl),
                            4 => a.rol(eax, cl),
                            _ => a.rol(rax, cl),
                        }?,
                        _ => match size {
                            1 => a.ror(al, cl),
                            2 => a.ror(ax, cl),
                            4 => a.ror(eax, cl),
                            _ => a.ror(rax, cl),
                        }?,
                    }
                }
                B::Min | B::Max => {
                    a.cmp(rax, rcx)?;
                    match (op, signed) {
                        (B::Min, true) => a.cmovg(rax, rcx),
                        (B::Min, false) => a.cmova(rax, rcx),
                        (_, true) => a.cmovl(rax, rcx),
                        _ => a.cmovb(rax, rcx),
                    }?;
                }
                B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge => {
                    a.cmp(rax, rcx)?;
                    set(a, op, signed)?;
                    mask(a)?;
                }
                B::SaturatingAdd | B::SaturatingSub => {
                    if signed && size == 8 {
                        a.mov(rdx, rax)?;
                        a.sar(rdx, 63)?;
                        a.mov(r10, i64::MAX)?;
                        a.xor(rdx, r10)?;
                        if op == B::SaturatingAdd {
                            a.add(rax, rcx)?;
                        } else {
                            a.sub(rax, rcx)?;
                        }
                        a.cmovo(rax, rdx)?;
                    } else if signed {
                        let max = (1i64 << (bits - 1)) - 1;
                        if op == B::SaturatingAdd {
                            a.add(rax, rcx)?;
                        } else {
                            a.sub(rax, rcx)?;
                        }
                        a.mov(r10, max)?;
                        a.cmp(rax, r10)?;
                        a.cmovg(rax, r10)?;
                        a.mov(r10, !max)?;
                        a.cmp(rax, r10)?;
                        a.cmovl(rax, r10)?;
                    } else if op == B::SaturatingSub {
                        a.xor(edx, edx)?;
                        a.sub(rax, rcx)?;
                        a.cmovb(rax, rdx)?;
                    } else {
                        a.mov(rdx, u64::MAX >> (64 - bits))?;
                        a.add(rax, rcx)?;
                        if size == 8 {
                            a.cmovb(rax, rdx)?;
                        } else {
                            a.cmp(rax, rdx)?;
                            a.cmova(rax, rdx)?;
                        }
                    }
                }
                _ => unreachable!("verified integer binary"),
            }
        }
        _ => unreachable!("compound integer operation"),
    }
    put(a, p.out, output.lane().unwrap().bytes())
}
