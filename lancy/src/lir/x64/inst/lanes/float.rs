use super::*;
use Binary as B;
use Unary as U;

// Round an IEEE value in its own format. Working on significand bits preserves
// signed zero and implements both half-away and half-even on baseline SSE2.
fn round_bits(a: &mut CodeAssembler, lane: Lane, op: U, src: i32, out: i32) -> Result {
    let f = convert::Float::new(lane);
    let fraction = f.fraction;
    let bias = f.bias;
    let one = (bias as u64) << fraction;
    let finish = a.create_label();
    let zero = a.create_label();
    let unit = a.create_label();
    let small = a.create_label();
    let bump = a.create_label();
    word(a, 0, src, lane.bytes(), false)?;
    a.mov(r9, f.sign)?;
    a.and(r9, rax)?;
    a.mov(r10, f.sign - 1)?;
    a.and(rax, r10)?;
    a.jz(finish)?;
    a.mov(r10, ((bias + fraction) as u64) << fraction)?;
    a.cmp(rax, r10)?;
    a.jae(finish)?;
    a.mov(r10, one)?;
    a.cmp(rax, r10)?;
    a.jb(small)?;
    a.mov(rcx, rax)?;
    a.shr(rcx, fraction)?;
    a.neg(ecx)?;
    a.add(ecx, bias + fraction)?;
    a.mov(r8d, 1)?;
    a.shl(r8, cl)?;
    a.lea(rdx, ptr(r8 - 1))?;
    a.and(rdx, rax)?;
    a.sub(rax, rdx)?;
    a.test(rdx, rdx)?;
    a.je(finish)?;
    match op {
        U::Floor | U::Ceil => {
            a.test(r9, r9)?;
            if op == U::Floor {
                a.jne(bump)?;
            } else {
                a.je(bump)?;
            }
        }
        U::Round | U::Nearest => {
            a.add(rdx, rdx)?;
            a.cmp(rdx, r8)?;
            if op == U::Round {
                a.jae(bump)?;
            } else {
                a.ja(bump)?;
                a.jb(finish)?;
                a.test(rax, r8)?;
                a.jne(bump)?;
            }
        }
        U::Trunc => {}
        _ => unreachable!(),
    }
    a.jmp(finish)?;
    bind(a, bump)?;
    a.add(rax, r8)?;
    a.jmp(finish)?;
    bind(a, small)?;
    match op {
        U::Floor | U::Ceil => {
            a.test(r9, r9)?;
            if op == U::Floor {
                a.jne(unit)?;
            } else {
                a.je(unit)?;
            }
        }
        U::Round | U::Nearest => {
            a.mov(r10, ((bias - 1) as u64) << fraction)?;
            a.cmp(rax, r10)?;
            if op == U::Round {
                a.jae(unit)?;
            } else {
                a.ja(unit)?;
            }
        }
        _ => {}
    }
    bind(a, zero)?;
    a.xor(eax, eax)?;
    a.jmp(finish)?;
    bind(a, unit)?;
    a.mov(rax, one)?;
    bind(a, finish)?;
    a.or(rax, r9)?;
    put(a, out, lane.bytes())
}

fn bits(a: &mut CodeAssembler, action: Action, input: Type, output: Type, p: Args) -> Result {
    let lane = input.lane().unwrap();
    let size = lane.bytes();
    let fmt = convert::Float::new(lane);
    let (sign, inf, normal) = (fmt.sign, fmt.infinity(), 1u64 << fmt.fraction);
    word(a, 0, p.x, size, false)?;
    a.mov(r8, sign)?;
    a.mov(r9, sign - 1)?;
    match action {
        Action::Unary(U::Signum) => {
            let done = a.create_label();
            a.and(r9, rax)?;
            a.mov(r10, inf)?;
            a.cmp(r9, r10)?;
            a.ja(done)?;
            a.and(rax, r8)?;
            a.mov(r10, (fmt.bias as u64) << fmt.fraction)?;
            a.or(rax, r10)?;
            bind(a, done)?;
        }
        Action::Unary(U::Neg) => a.xor(rax, r8)?,
        Action::Unary(U::Abs) => a.and(rax, r9)?,
        Action::Binary(B::CopySign) => {
            word(a, 1, p.y, size, false)?;
            a.and(rcx, r8)?;
            a.and(rax, r9)?;
            a.or(rax, rcx)?;
        }
        Action::Unary(op @ (U::NextUp | U::NextDown)) => {
            let up = op == U::NextUp;
            let done = a.create_label();
            let step = a.create_label();
            let dec = a.create_label();
            a.and(r9, rax)?;
            a.mov(r10, inf)?;
            a.cmp(r9, r10)?;
            a.ja(done)?;
            a.mov(r10, if up { inf } else { sign | inf })?;
            a.cmp(rax, r10)?;
            a.je(done)?;
            a.test(r9, r9)?;
            a.jne(step)?;
            a.mov(rax, if up { 1 } else { sign | 1 })?;
            a.jmp(done)?;
            bind(a, step)?;
            a.test(rax, r8)?;
            if up {
                a.jnz(dec)?;
            } else {
                a.jz(dec)?;
            }
            a.inc(rax)?;
            a.jmp(done)?;
            bind(a, dec)?;
            a.dec(rax)?;
            bind(a, done)?;
        }
        Action::Unary(op) => {
            a.and(r9, rax)?;
            match op {
                U::SignBit => {
                    a.test(rax, r8)?;
                    a.setne(al)?;
                }
                U::IsNan | U::IsFinite | U::IsInfinite => {
                    a.mov(r10, inf)?;
                    a.cmp(r9, r10)?;
                    match op {
                        U::IsNan => a.seta(al),
                        U::IsFinite => a.setb(al),
                        _ => a.sete(al),
                    }?;
                }
                U::IsNormal | U::IsSubnormal => {
                    let (low, high) = if op == U::IsNormal {
                        (normal, inf)
                    } else {
                        (1, normal)
                    };
                    a.mov(r10, low)?;
                    a.sub(r9, r10)?;
                    a.mov(r10, high - low)?;
                    a.cmp(r9, r10)?;
                    a.setb(al)?;
                }
                _ => unreachable!(),
            }
            a.movzx(eax, al)?;
            a.neg(rax)?;
        }
        _ => unreachable!(),
    }
    put(a, p.out, output.lane().unwrap().bytes())
}

// Functions with no exact instruction use the platform scalar math ABI.
// The call site contains only the address of the concrete function; its
// arguments and result are plain floating point values (XMM0..2 on both ABIs).
pub(super) fn math_function(action: Action, lane: Lane) -> Option<u64> {
    if !lane.float() {
        return None;
    }
    let single = lane == Lane::F32;
    macro_rules! unary {
        ($op:ident) => {{
            extern "C" fn f(x: f32) -> f32 {
                x.$op()
            }
            extern "C" fn d(x: f64) -> f64 {
                x.$op()
            }
            if single {
                f as *const ()
            } else {
                d as *const ()
            }
        }};
    }
    macro_rules! binary {
        ($op:ident) => {{
            extern "C" fn f(x: f32, y: f32) -> f32 {
                x.$op(y)
            }
            extern "C" fn d(x: f64, y: f64) -> f64 {
                x.$op(y)
            }
            if single {
                f as *const ()
            } else {
                d as *const ()
            }
        }};
    }
    let address = match action {
        Action::Unary(op) => match op {
            U::Exp => unary!(exp),
            U::Exp2 => unary!(exp2),
            U::Ln => unary!(ln),
            U::Log2 => unary!(log2),
            U::Log10 => unary!(log10),
            U::Sin => unary!(sin),
            U::Cos => unary!(cos),
            U::Tan => unary!(tan),
            U::Asin => unary!(asin),
            U::Acos => unary!(acos),
            U::Atan => unary!(atan),
            U::Sinh => unary!(sinh),
            U::Cosh => unary!(cosh),
            U::Tanh => unary!(tanh),
            U::Asinh => unary!(asinh),
            U::Acosh => unary!(acosh),
            U::Atanh => unary!(atanh),
            U::ExpM1 => unary!(exp_m1),
            U::Ln1p => unary!(ln_1p),
            U::Cbrt => unary!(cbrt),
            _ => return None,
        },
        Action::Binary(op) => match op {
            B::Pow => binary!(powf),
            B::Atan2 => binary!(atan2),
            B::Hypot => binary!(hypot),
            B::Rem => {
                extern "C" fn f(x: f32, y: f32) -> f32 {
                    x % y
                }
                extern "C" fn d(x: f64, y: f64) -> f64 {
                    x % y
                }
                if single {
                    f as *const ()
                } else {
                    d as *const ()
                }
            }
            _ => return None,
        },
        Action::Ternary(Ternary::MulAdd) if matches!(lane, Lane::F32 | Lane::F64) => {
            extern "C" fn f(x: f32, y: f32, z: f32) -> f32 {
                x.mul_add(y, z)
            }
            extern "C" fn d(x: f64, y: f64, z: f64) -> f64 {
                x.mul_add(y, z)
            }
            if single {
                f as *const ()
            } else {
                d as *const ()
            }
        }
        _ => return None,
    };
    Some(address as u64)
}

// Two half significands multiply exactly in binary64. FastTwoSum recovers
// the residual of the addition so that conversion rounds only once.
fn half_fma(a: &mut CodeAssembler, lane: Lane, out: i32) -> Result {
    a.mulsd(xmm0, xmm1)?;
    a.movaps(xmm1, xmm0)?;
    a.addsd(xmm0, xmm2)?;
    a.movq(rax, xmm1)?;
    a.btr(rax, 63)?;
    a.movq(rcx, xmm2)?;
    a.btr(rcx, 63)?;
    let second = a.create_label();
    let done = a.create_label();
    a.cmp(rax, rcx)?;
    a.jb(second)?;
    a.subsd(xmm1, xmm0)?;
    a.addsd(xmm1, xmm2)?;
    a.jmp(done)?;
    bind(a, second)?;
    a.subsd(xmm2, xmm0)?;
    a.addsd(xmm1, xmm2)?;
    bind(a, done)?;
    a.movq(qword_ptr(rsp + TEMP), xmm0)?;
    a.movq(qword_ptr(rsp + TEMP + 8), xmm1)?;
    convert::to_float(a, Lane::F64, lane, TEMP, out, Some(TEMP + 8))
}

pub(super) fn emit(
    a: &mut CodeAssembler,
    action: Action,
    input: Type,
    output: Type,
    p: Args,
) -> Result {
    let lane = input.lane().unwrap();
    let single = lane == Lane::F32;
    if let Action::Unary(op @ (U::Floor | U::Ceil | U::Trunc | U::Round | U::Nearest)) = action {
        return round_bits(a, lane, op, p.x, p.out);
    }
    if matches!(action, Action::Unary(U::Signum)) {
        return bits(a, action, input, output, p);
    }
    if matches!(
        action,
        Action::Unary(
            U::Abs
                | U::Neg
                | U::NextUp
                | U::NextDown
                | U::SignBit
                | U::IsNan
                | U::IsInfinite
                | U::IsNormal
                | U::IsFinite
                | U::IsSubnormal
        ) | Action::Binary(B::CopySign)
    ) {
        return bits(a, action, input, output, p);
    }
    let half = matches!(lane, Lane::F16 | Lane::BF16);
    let arity = match action {
        Action::Unary(_) => 1,
        Action::Binary(_) => 2,
        Action::Ternary(Ternary::MulAdd) => 3,
        _ => unreachable!("verified scalar floating-point operation"),
    };
    let mut source = [p.x, p.y, p.z];
    if half {
        for (i, offset) in source[..arity].iter_mut().enumerate() {
            let dst = TEMP + 8 * i as i32;
            convert::emit(a, lane, Lane::F64, *offset, dst)?;
            *offset = dst;
        }
    }
    for (&offset, reg) in source[..arity].iter().zip([xmm0, xmm1, xmm2]) {
        if single {
            a.movss(reg, dword_ptr(rsp + offset))?;
        } else {
            a.movq(reg, qword_ptr(rsp + offset))?;
        }
    }
    macro_rules! fp {
        ($s:ident,$d:ident) => {
            if single {
                a.$s(xmm0, xmm1)
            } else {
                a.$d(xmm0, xmm1)
            }
        };
    }
    match action {
        Action::Unary(U::Fract) => {
            let format = if single { Lane::F32 } else { Lane::F64 };
            round_bits(a, format, U::Trunc, source[0], TEMP + 24)?;
            if single {
                a.subss(xmm0, dword_ptr(rsp + TEMP + 24))?;
            } else {
                a.subsd(xmm0, qword_ptr(rsp + TEMP + 24))?;
            }
        }
        Action::Binary(op @ (B::Min | B::Max)) => {
            // A NaN operand yields the other one. MINSS/MAXSS return the
            // second operand for a NaN first operand and for equal operands,
            // so equal ones (the two zeros) combine their sign bits instead.
            let done = a.create_label();
            let ordered = a.create_label();
            if single {
                a.ucomiss(xmm1, xmm1)?;
            } else {
                a.ucomisd(xmm1, xmm1)?;
            }
            a.jp(done)?;
            if single {
                a.ucomiss(xmm0, xmm1)?;
            } else {
                a.ucomisd(xmm0, xmm1)?;
            }
            a.jp(ordered)?;
            a.jne(ordered)?;
            if op == B::Min {
                a.orps(xmm0, xmm1)?;
            } else {
                a.andps(xmm0, xmm1)?;
            }
            a.jmp(done)?;
            bind(a, ordered)?;
            if op == B::Min {
                fp!(minss, minsd)?;
            } else {
                fp!(maxss, maxsd)?;
            }
            bind(a, done)?;
        }
        Action::Binary(B::Add) => fp!(addss, addsd)?,
        Action::Binary(B::Sub) => fp!(subss, subsd)?,
        Action::Binary(B::Mul) => fp!(mulss, mulsd)?,
        Action::Binary(B::Div) => fp!(divss, divsd)?,
        Action::Unary(U::Sqrt) => {
            if single {
                a.sqrtss(xmm0, xmm0)?;
            } else {
                a.sqrtsd(xmm0, xmm0)?;
            }
        }
        Action::Unary(U::Recip) => {
            a.movaps(xmm1, xmm0)?;
            a.mov(eax, 1)?;
            if single {
                a.cvtsi2ss(xmm0, eax)?;
                a.divss(xmm0, xmm1)?;
            } else {
                a.cvtsi2sd(xmm0, eax)?;
                a.divsd(xmm0, xmm1)?;
            }
        }
        Action::Binary(
            op @ (B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge | B::Ordered | B::Unordered),
        ) => {
            if single {
                a.ucomiss(xmm0, xmm1)?;
            } else {
                a.ucomisd(xmm0, xmm1)?;
            }
            match op {
                B::Eq | B::Lt | B::Le => {
                    match op {
                        B::Eq => a.sete(al),
                        B::Lt => a.setb(al),
                        _ => a.setbe(al),
                    }?;
                    a.setnp(cl)?;
                    a.and(al, cl)?;
                }
                B::Ne => {
                    a.setne(al)?;
                    a.setp(cl)?;
                    a.or(al, cl)?;
                }
                B::Gt => a.seta(al)?,
                B::Ge => a.setae(al)?,
                B::Ordered => a.setnp(al)?,
                _ => a.setp(al)?,
            }
            a.movzx(eax, al)?;
            a.neg(rax)?;
            return put(a, p.out, output.lane().unwrap().bytes());
        }
        Action::Ternary(Ternary::MulAdd) if half => return half_fma(a, lane, p.out),
        _ => {
            a.mov(
                r10,
                math_function(action, lane).expect("verified scalar math operation"),
            )?;
            a.call(r10)?;
        }
    }
    if single {
        a.movss(dword_ptr(rsp + p.out), xmm0)
    } else if half {
        a.movq(qword_ptr(rsp + TEMP), xmm0)?;
        convert::emit(a, Lane::F64, lane, TEMP, p.out)
    } else {
        a.movq(qword_ptr(rsp + p.out), xmm0)
    }
}
