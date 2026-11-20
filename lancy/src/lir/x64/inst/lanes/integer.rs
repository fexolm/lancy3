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
    if size == 16 {
        return wide(a, action, input, output, p, faults);
    }
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

fn negate(a: &mut CodeAssembler, lo: AsmRegister64, hi: AsmRegister64) -> Result {
    a.neg(lo)?;
    a.adc(hi, 0)?;
    a.neg(hi)
}
fn abs(a: &mut CodeAssembler, lo: AsmRegister64, hi: AsmRegister64) -> Result {
    let ok = a.create_label();
    a.test(hi, hi)?;
    a.jns(ok)?;
    negate(a, lo, hi)?;
    bind(a, ok)
}
// Shift one dividend bit at a time into the remainder. The quotient replaces
// consumed dividend bits in RDX:RAX. The extra carry handles 129-bit remainders.
fn division(
    a: &mut CodeAssembler,
    p: Args,
    signed: bool,
    rem: bool,
    faults: &mut Faults,
) -> Result {
    let fail = *faults[1].get_or_insert_with(|| a.create_label());
    a.mov(r9, rcx)?;
    a.or(r9, r8)?;
    a.je(fail)?;
    if signed {
        let ok = a.create_label();
        let overflow = *faults[2].get_or_insert_with(|| a.create_label());
        a.cmp(r8, -1)?;
        a.jne(ok)?;
        a.cmp(rcx, -1)?;
        a.jne(ok)?;
        a.test(rax, rax)?;
        a.jne(ok)?;
        a.mov(r9, i64::MIN)?;
        a.cmp(rdx, r9)?;
        a.je(overflow)?;
        bind(a, ok)?;
        abs(a, rax, rdx)?;
        abs(a, rcx, r8)?;
    }
    a.xor(r9d, r9d)?;
    a.xor(r10d, r10d)?;
    a.mov(r11d, 128)?;
    let step = a.create_label();
    let subtract = a.create_label();
    let skip = a.create_label();
    bind(a, step)?;
    a.shl(rax, 1)?;
    a.rcl(rdx, 1)?;
    a.rcl(r9, 1)?;
    a.rcl(r10, 1)?;
    a.jc(subtract)?;
    a.cmp(r10, r8)?;
    a.ja(subtract)?;
    a.jb(skip)?;
    a.cmp(r9, rcx)?;
    a.jb(skip)?;
    bind(a, subtract)?;
    a.sub(r9, rcx)?;
    a.sbb(r10, r8)?;
    a.inc(rax)?;
    bind(a, skip)?;
    a.dec(r11d)?;
    a.jne(step)?;
    if rem {
        a.mov(rax, r9)?;
        a.mov(rdx, r10)?;
    }
    if signed {
        let done = a.create_label();
        a.mov(r9, qword_ptr(rsp + p.x + 8))?;
        if !rem {
            a.xor(r9, qword_ptr(rsp + p.y + 8))?;
        }
        a.test(r9, r9)?;
        a.jns(done)?;
        negate(a, rax, rdx)?;
        bind(a, done)?;
    }
    Ok(())
}
fn shift_wide(a: &mut CodeAssembler, op: B, signed: bool) -> Result {
    let small = a.create_label();
    a.test(ecx, 64)?;
    a.je(small)?;
    match op {
        B::Shl => {
            a.mov(rdx, rax)?;
            a.xor(eax, eax)?;
        }
        B::Shr => {
            a.mov(rax, rdx)?;
            if signed {
                a.sar(rdx, 63)?;
            } else {
                a.xor(edx, edx)?;
            }
        }
        _ => a.xchg(rax, rdx)?,
    }
    bind(a, small)?;
    match op {
        B::Shl => {
            a.shld(rdx, rax, cl)?;
            a.shl(rax, cl)
        }
        B::Shr => {
            a.shrd(rax, rdx, cl)?;
            if signed {
                a.sar(rdx, cl)
            } else {
                a.shr(rdx, cl)
            }
        }
        B::RotateLeft => {
            a.mov(r9, rdx)?;
            a.shld(rdx, rax, cl)?;
            a.shld(rax, r9, cl)
        }
        _ => {
            a.mov(r9, rax)?;
            a.shrd(rax, rdx, cl)?;
            a.shrd(rdx, r9, cl)
        }
    }
}

fn wide(
    a: &mut CodeAssembler,
    action: Action,
    input: Type,
    output: Type,
    p: Args,
    faults: &mut Faults,
) -> Result {
    let signed = input.lane().unwrap().signed();
    a.mov(rax, qword_ptr(rsp + p.x))?;
    a.mov(rdx, qword_ptr(rsp + p.x + 8))?;
    match action {
        Action::Unary(op) => match op {
            U::Neg => negate(a, rax, rdx)?,
            U::Abs if signed => abs(a, rax, rdx)?,
            U::Abs => {}
            U::Popcnt => {
                popcnt(a)?;
                a.mov(r8d, eax)?;
                a.mov(rax, rdx)?;
                popcnt(a)?;
                a.add(eax, r8d)?;
                a.xor(edx, edx)?;
            }
            U::LeadingZeros | U::TrailingZeros => {
                let leading = op == U::LeadingZeros;
                if leading {
                    a.xchg(rax, rdx)?;
                }
                let nonzero = a.create_label();
                let done = a.create_label();
                a.test(rax, rax)?;
                a.jne(nonzero)?;
                a.mov(rax, rdx)?;
                zeros(a, leading, 64)?;
                a.add(eax, 64)?;
                a.jmp(done)?;
                bind(a, nonzero)?;
                zeros(a, leading, 64)?;
                bind(a, done)?;
                a.xor(edx, edx)?;
            }
            _ => unreachable!(),
        },
        Action::Binary(op) => {
            a.mov(rcx, qword_ptr(rsp + p.y))?;
            a.mov(r8, qword_ptr(rsp + p.y + 8))?;
            match op {
                B::Add | B::Sub | B::SaturatingAdd | B::SaturatingSub => {
                    let add = matches!(op, B::Add | B::SaturatingAdd);
                    let sat = matches!(op, B::SaturatingAdd | B::SaturatingSub);
                    if sat && signed {
                        a.mov(r9, rdx)?;
                        a.sar(r9, 63)?;
                    }
                    if add {
                        a.add(rax, rcx)?;
                        a.adc(rdx, r8)?;
                    } else {
                        a.sub(rax, rcx)?;
                        a.sbb(rdx, r8)?;
                    }
                    if sat {
                        let end = a.create_label();
                        if signed {
                            a.jno(end)?;
                            a.mov(rdx, i64::MAX)?;
                            a.xor(rdx, r9)?;
                            a.mov(rax, r9)?;
                            a.not(rax)?;
                        } else {
                            a.jnc(end)?;
                            a.mov(rax, if add { -1i64 } else { 0 })?;
                            a.mov(rdx, rax)?;
                        }
                        bind(a, end)?;
                    }
                }
                B::Mul => {
                    a.imul_2(rdx, rcx)?;
                    a.imul_2(r8, rax)?;
                    a.add(r8, rdx)?;
                    a.mul(rcx)?;
                    a.add(rdx, r8)?;
                }
                B::Div | B::Rem => division(a, p, signed, op == B::Rem, faults)?,
                B::Shl | B::Shr | B::RotateLeft | B::RotateRight => shift_wide(a, op, signed)?,
                B::Min | B::Max | B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge => {
                    let low = a.create_label();
                    let done = a.create_label();
                    let pred = match op {
                        B::Min => B::Gt,
                        B::Max => B::Lt,
                        _ => op,
                    };
                    a.mov(r9, rax)?;
                    a.cmp(rdx, r8)?;
                    a.je(low)?;
                    set(a, pred, signed)?;
                    a.jmp(done)?;
                    bind(a, low)?;
                    a.cmp(rax, rcx)?;
                    set(a, pred, false)?;
                    bind(a, done)?;
                    if matches!(op, B::Min | B::Max) {
                        a.test(al, al)?;
                        a.mov(rax, r9)?;
                        a.cmovne(rax, rcx)?;
                        a.cmovne(rdx, r8)?;
                    } else {
                        mask(a)?;
                    }
                }
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    }
    put(a, p.out, output.lane().unwrap().bytes())
}
