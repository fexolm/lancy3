use super::{
    A, C, Cond, Result,
    inst::{Inst, IntBinary, IntOp, Masked, Packed, Select, class},
};
use crate::{
    hir::{
        Binary, Function, Lane, Op, Simd, Terminator, Ternary, Type, Value,
        transforms::ControlFlowGraph,
    },
    lir::{
        self,
        isel::{self, Edges, Root},
    },
    module::FunctionId,
    numbers::Math,
};
use regalloc2::{PRegSet, VReg};
use splice_target::{Profile, Simd as Config};

pub(super) fn select(
    f: &Function,
    cfg: &ControlFlowGraph,
    config: Config,
    entry: bool,
    callee: impl Fn(&Function) -> (FunctionId, PRegSet),
) -> Result<lir::Function<Inst>> {
    let reg = |v: Value| VReg::new(v.index(), class(f.value_type(v)));
    let edges = Edges::new(f, cfg);
    let mut selected = isel::select(f, cfg, |ctx, root, out| {
        let ty = |v| f.value_type(v);
        let Root::Inst(i) = root else {
            let Root::Term(term) = root else {
                unreachable!()
            };
            out.push(match term {
                Terminator::Return(v) => {
                    ctx.require(v);
                    Inst::Return(ty(v), reg(v), entry)
                }
                Terminator::Jump(_) => Inst::Jump,
                Terminator::Branch { cond, .. } => {
                    ctx.require(cond);
                    Inst::Branch(reg(cond))
                }
            });
            return Ok(());
        };
        let op = *f.op(i);
        let result = || f.result(i).expect("value-producing operation");
        let dst = || reg(result());
        let immediate = match op {
            Op::Bin(op @ (Binary::Add | Binary::Sub), a, b) if ty(a) == Type::I32 => {
                let code = if op == Binary::Add {
                    C::Add_32_addsub_imm
                } else {
                    C::Sub_32_addsub_imm
                };
                let form = |v| ctx.constant(v).and_then(add_immediate);
                form(b)
                    .map(|imm| (a, imm))
                    .or_else(|| {
                        (op == Binary::Add)
                            .then(|| form(a).map(|imm| (b, imm)))
                            .flatten()
                    })
                    .map(|(src, (imm, shift))| (code, src, imm, shift))
            }
            _ => None,
        };
        let mut input = |v| {
            ctx.require(v);
            reg(v)
        };
        if let Some((code, src, imm, shift)) = immediate {
            out.push(Inst::Imm(code, imm, shift, dst(), input(src)));
            return Ok(());
        }
        let inst = match op {
            Op::Const(n) => Inst::Literal(Type::I32, n as u32 as u128, dst()),
            Op::Literal(lane, bits) => Inst::Literal(lane.scalar(), bits, dst()),
            Op::Simd(Simd::LaneCount(lane)) => {
                Inst::Literal(Type::I32, (config.bytes / lane.bytes()) as u128, dst())
            }
            Op::Simd(Simd::Iota(lane)) if sve(config.profile) && scalar_int(lane) => {
                Inst::SveIota(arrangements(lane.bytes()).1, dst())
            }
            Op::Simd(Simd::Iota(lane)) if config.bytes == 16 => {
                let (math, _) = Math::from_op(op, ty).unwrap();
                let bytes = math
                    .eval(&[[0; 64]; 3], config.bytes)
                    .expect("iota is total");
                let bits = u128::from_le_bytes(*bytes.first_chunk().unwrap());
                Inst::Literal(Type::Vector(lane), bits, dst())
            }
            Op::Simd(Simd::AnyTrue(v) | Simd::AllTrue(v)) if ty(v).lane().unwrap().bytes() <= 8 => {
                let (neon, scalable) = arrangements(ty(v).lane().unwrap().bytes());
                let arr = if sve(config.profile) { scalable } else { neon };
                Inst::MaskTest(
                    matches!(op, Op::Simd(Simd::AllTrue(_))),
                    arr,
                    dst(),
                    input(v),
                )
            }
            Op::Simd(Simd::ExtractLane(src, idx) | Simd::InsertLane(src, idx, _)) => {
                let lane = ty(src).lane().unwrap().scalar();
                let insert = match op {
                    Op::Simd(Simd::InsertLane(_, _, value)) => Some(input(value)),
                    _ => None,
                };
                Inst::LaneAccess {
                    lane,
                    src: input(src),
                    index: input(idx),
                    insert,
                    dst: dst(),
                }
            }
            Op::Simd(Simd::Splat(v)) if ty(v).lane().unwrap().bytes() <= 8 => {
                let (neon, scalable) = arrangements(ty(v).lane().unwrap().bytes());
                let (code, arr) = if sve(config.profile) {
                    (C::Dup_z_r, scalable)
                } else {
                    (C::Dup_asimdgen, neon)
                };
                Inst::Splat(code, arr, ty(v), dst(), input(v))
            }
            Op::Ternary(Ternary::MulAdd, a, b, c) if fp(ty(a)) => {
                let arr = if ty(a) == Type::F32 { A::S } else { A::D };
                Inst::Fma(C::Fmadd_scalar, arr, dst(), [input(a), input(b), input(c)])
            }
            Op::Simd(Simd::Ternary(Ternary::MulAdd, a, b, c))
                if matches!(ty(a), Type::Vector(Lane::F32 | Lane::F64)) =>
            {
                let (neon, scalable) = arrangements(ty(a).lane().unwrap().bytes());
                let (code, arr) = if sve(config.profile) {
                    (C::Fmla_z_p_zz, scalable)
                } else {
                    (C::Fmla_asimd, neon)
                };
                Inst::Fma(code, arr, dst(), [input(a), input(b), input(c)])
            }
            Op::Simd(Simd::Compress(value, mask))
                if sve(config.profile) && matches!(ty(value).lane().unwrap().bytes(), 4 | 8) =>
            {
                Inst::Compact(
                    arrangements(ty(value).lane().unwrap().bytes()).1,
                    dst(),
                    input(value),
                    input(mask),
                )
            }
            Op::Ternary(Ternary::Select, mask, a, b) => Inst::Select(
                Select::Scalar(ty(a)),
                dst(),
                input(mask),
                input(a),
                input(b),
            ),
            Op::Simd(Simd::Ternary(kind @ (Ternary::Select | Ternary::BitSelect), mask, a, b))
                if kind == Ternary::BitSelect || ty(a).lane().unwrap().bytes() <= 8 =>
            {
                let scalable = sve(config.profile);
                let arr = if kind == Ternary::Select {
                    let (neon, sve) = arrangements(ty(a).lane().unwrap().bytes());
                    Some(if scalable { sve } else { neon })
                } else {
                    None
                };
                let form = if scalable {
                    Select::Sve(arr)
                } else {
                    Select::Neon(arr)
                };
                Inst::Select(form, dst(), input(mask), input(a), input(b))
            }
            Op::Simd(Simd::Reduce(Binary::Add, value))
                if ty(value).lane().is_some_and(scalar_int) =>
            {
                let size = ty(value).lane().unwrap().bytes();
                let (neon, scalable) = arrangements(size);
                Inst::Sum(
                    if sve(config.profile) { scalable } else { neon },
                    size,
                    dst(),
                    input(value),
                )
            }
            Op::Cast(to, v)
                if ty(v).lane().is_some_and(scalar_int) && to.lane().is_some_and(scalar_int) =>
            {
                Inst::IntCast(ty(v), to, dst(), input(v))
            }
            Op::Bitcast(to, v) if class(to) != class(ty(v)) => {
                Inst::BitMove(ty(v), to, dst(), input(v))
            }
            Op::Copy(v) | Op::Bitcast(_, v) | Op::Simd(Simd::Bitcast(_, v)) => {
                Inst::Copy(dst(), input(v))
            }
            Op::BoundsCheck(index, length) => Inst::Check(input(index), input(length)),
            Op::PtrOffset(base, index, scale) => {
                Inst::Offset(scale, dst(), input(base), input(index))
            }
            Op::Load(ty, ptr) => Inst::Load(ty, dst(), input(ptr)),
            Op::Simd(Simd::Load(lane, ptr)) => Inst::Load(Type::Vector(lane), dst(), input(ptr)),
            Op::Store(ptr, v) | Op::Simd(Simd::Store(ptr, v)) => {
                Inst::Store(ty(v), input(ptr), input(v))
            }
            Op::Simd(
                Simd::MaskedLoad {
                    address,
                    mask,
                    passthrough: value,
                }
                | Simd::MaskedStore {
                    address,
                    mask,
                    value,
                },
            ) => Inst::Masked {
                kind: masked(
                    ty(value),
                    config.profile,
                    address.indices.is_some(),
                    f.result(i).is_some(),
                ),
                ty: ty(value),
                base: input(address.base),
                indices: address.indices.map(&mut input),
                mask: input(mask),
                value: input(value),
                dst: f.result(i).map(reg),
            },
            Op::Call(site) => {
                let (func, args) = f.call(site);
                let mut offset = 0;
                for &arg in args {
                    let ty = ty(arg);
                    out.push(Inst::StoreArg(ty, offset, input(arg)));
                    offset += ty.bytes(config.bytes).next_multiple_of(8);
                }
                let (target, writes) = callee(func);
                Inst::Call(target, writes, ty(result()), dst())
            }
            op => {
                if let Op::Cast(to, v) = op
                    && let Some(code) = float_cast(ty(v), to)
                {
                    out.push(Inst::FloatCast(code, ty(v), to, dst(), input(v)));
                    return Ok(());
                }
                if let Op::Simd(Simd::Cast(to, v)) = op
                    && let Some((code, arr)) =
                        packed_cast(ty(v).lane().unwrap(), to, config.profile)
                {
                    out.push(Inst::PackedCast(code, arr, dst(), input(v)));
                    return Ok(());
                }
                if let Op::Bin(op, a, b) = op
                    && let Some(native) = scalar_binary(op, ty(a))
                {
                    out.push(Inst::Int(native, dst(), input(a), input(b)));
                    return Ok(());
                }
                if let Op::Bin(op, a, b) = op
                    && let Some((code, arr)) = float_binary(op, ty(a))
                {
                    out.push(Inst::Float(code, arr, dst(), input(a), input(b)));
                    return Ok(());
                }
                if let Op::Bin(op, a, b) = op
                    && let Some((cond, arr)) = float_compare(op, ty(a))
                {
                    out.push(Inst::FloatCompare(cond, arr, dst(), input(a), input(b)));
                    return Ok(());
                }
                if let Op::Bin(op, a, b) = op
                    && matches!(ty(a), Type::F32 | Type::F64)
                    && let Some(code) = neon_bits(op)
                {
                    out.push(Inst::Packed(
                        Packed::Neon(code, A::B16),
                        dst(),
                        input(a),
                        input(b),
                    ));
                    return Ok(());
                }
                if let Op::Simd(Simd::Bin(op, a, b)) = op
                    && let Some(native) = packed_binary(op, ty(a), config.profile)
                {
                    out.push(Inst::Packed(native, dst(), input(a), input(b)));
                    return Ok(());
                }
                if let Op::Simd(Simd::Bin(op, a, b)) = op
                    && let Some((native, swap)) = packed_compare(op, ty(a), config.profile)
                {
                    let (a, b) = if swap { (b, a) } else { (a, b) };
                    out.push(Inst::Packed(native, dst(), input(a), input(b)));
                    return Ok(());
                }
                let (math, args) = Math::from_op(op, ty).expect("verified numeric operation");
                for (n, arg) in args.into_iter().enumerate() {
                    if let Some(arg) = arg {
                        out.push(Inst::LibArg(ty(arg), n, input(arg)));
                    }
                }
                let sizes = args.map(|arg| arg.map_or(0, |v| ty(v).bytes(config.bytes)));
                out.push(Inst::Libcall(math, sizes));
                Inst::LibResult(ty(result()), dst())
            }
        };
        out.push(inst);
        Ok(())
    })?;

    let mut body = edges.function()?;
    for &b in cfg.reverse_postorder() {
        if b == f.entry() {
            let mut offset = 0;
            for &v in f.params(b) {
                let ty = f.value_type(v);
                body.push(Inst::Arg(ty, offset, reg(v)));
                offset += ty.bytes(config.bytes).next_multiple_of(8);
            }
        }
        for inst in selected[b].drain(..) {
            body.push(inst);
        }
        edges.finish(&mut body, b, reg, || Inst::Jump)?;
    }
    Ok(body)
}

fn add_immediate(n: i32) -> Option<(u16, u8)> {
    if (0..4096).contains(&n) {
        Some((n as u16, 0))
    } else if (0..1 << 24).contains(&n) && n & 4095 == 0 {
        Some(((n >> 12) as u16, 12))
    } else {
        None
    }
}

fn scalar_int(lane: Lane) -> bool {
    !lane.float() && lane.bytes() <= 8
}

fn sve(profile: Profile) -> bool {
    matches!(profile, Profile::Sve | Profile::Sve2)
}

fn arrangements(bytes: usize) -> (A, A) {
    let index = bytes.trailing_zeros() as usize;
    (
        [A::B16, A::H8, A::S4, A::D2][index],
        [A::B, A::H, A::S, A::D][index],
    )
}

fn masked(ty: Type, profile: Profile, indexed: bool, load: bool) -> Masked {
    let size = ty.lane().unwrap().bytes();
    if !sve(profile) || size > 8 {
        return Masked::Lanes;
    }
    let code = match (load, indexed, size) {
        (true, true, 4) => C::Ld1w_z_p_bz_sxtw,
        (true, true, 8) => C::Ld1d_z_p_bz_lsl,
        (true, false, 1) => C::Ld1b_z_p_bi_u8,
        (true, false, 2) => C::Ld1h_z_p_bi_u16,
        (true, false, 4) => C::Ld1w_z_p_bi_u32,
        (true, false, 8) => C::Ld1d_z_p_bi_u64,
        (false, false, 1) => C::St1b_z_p_bi_,
        (false, false, 2) => C::St1h_z_p_bi_,
        (false, false, 4) => C::St1w_z_p_bi_,
        (false, false, 8) => C::St1d_z_p_bi_,
        // Overlapping scatter lanes must commit in HIR lane order.
        _ => return Masked::Lanes,
    };
    Masked::Sve {
        code,
        arrangement: arrangements(size).1,
    }
}

fn float_binary(op: Binary, ty: Type) -> Option<(C, A)> {
    let arr = match ty {
        Type::F32 => A::S,
        Type::F64 => A::D,
        _ => return None,
    };
    let code = match op {
        Binary::Add => C::Fadd_scalar,
        Binary::Sub => C::Fsub_scalar,
        Binary::Mul => C::Fmul_scalar,
        Binary::Div => C::Fdiv_scalar,
        _ => return None,
    };
    Some((code, arr))
}

fn fp(ty: Type) -> bool {
    matches!(ty, Type::F32 | Type::F64)
}

fn float_cast(from: Type, to: Type) -> Option<C> {
    match (from, to) {
        (Type::F32, Type::F64) => Some(C::Fcvt_d_s),
        (Type::F64, Type::F32) => Some(C::Fcvt_s_d),
        (from, to) if fp(to) && from.lane().is_some_and(scalar_int) => {
            Some(if from.lane().unwrap().signed() {
                C::Scvtf_scalar
            } else {
                C::Ucvtf_scalar
            })
        }
        (from, to) if fp(from) && to.lane().is_some_and(scalar_int) => {
            Some(if to.lane().unwrap().signed() {
                C::Fcvtzs_scalar
            } else {
                C::Fcvtzu_scalar
            })
        }
        _ => None,
    }
}

fn packed_cast(from: Lane, to: Lane, profile: Profile) -> Option<(C, A)> {
    let size = from.bytes();
    if size != to.bytes() || !matches!(size, 4 | 8) || from.float() == to.float() {
        return None;
    }
    let (neon, scalable) = arrangements(size);
    let code = match (sve(profile), from.float(), from.signed() || to.signed()) {
        (true, true, true) => C::Fcvtzs_z_p_z,
        (true, true, false) => C::Fcvtzu_z_p_z,
        (true, false, true) => C::Scvtf_z_p_z,
        (true, false, false) => C::Ucvtf_z_p_z,
        (false, true, true) => C::Fcvtzs_asimd,
        (false, true, false) => C::Fcvtzu_asimd,
        (false, false, true) => C::Scvtf_asimd,
        (false, false, false) => C::Ucvtf_asimd,
    };
    Some((code, if sve(profile) { scalable } else { neon }))
}

fn float_compare(op: Binary, ty: Type) -> Option<(Cond, A)> {
    let arr = match ty {
        Type::F32 => A::S,
        Type::F64 => A::D,
        _ => return None,
    };
    let cond = match op {
        Binary::Eq => Cond::Eq,
        Binary::Ne => Cond::Ne,
        Binary::Lt => Cond::Mi,
        Binary::Le => Cond::Ls,
        Binary::Gt => Cond::Gt,
        Binary::Ge => Cond::Ge,
        Binary::Ordered => Cond::Vc,
        Binary::Unordered => Cond::Vs,
        _ => return None,
    };
    Some((cond, arr))
}

fn packed_compare(op: Binary, ty: Type, profile: Profile) -> Option<(Packed, bool)> {
    use Binary::*;
    let Type::Vector(lane) = ty else { return None };
    if lane.bytes() > 8 || matches!(lane, Lane::F16 | Lane::BF16) {
        return None;
    }
    let (neon, sve_arr) = arrangements(lane.bytes());
    let swap = matches!(op, Lt | Le);
    let floating = lane.float();
    let signed = lane.signed();
    let scalable = sve(profile);
    let code = match (scalable, floating, signed, op) {
        (true, true, _, Eq) => C::Fcmeq_p_p_zz,
        (true, true, _, Ne) => C::Fcmne_p_p_zz,
        (true, true, _, Gt | Lt) => C::Fcmgt_p_p_zz,
        (true, true, _, Ge | Le) => C::Fcmge_p_p_zz,
        (true, false, _, Eq) => C::Cmpeq_p_p_zz,
        (true, false, _, Ne) => C::Cmpne_p_p_zz,
        (true, false, true, Gt | Lt) => C::Cmpgt_p_p_zz,
        (true, false, true, Ge | Le) => C::Cmpge_p_p_zz,
        (true, false, false, Gt | Lt) => C::Cmphi_p_p_zz,
        (true, false, false, Ge | Le) => C::Cmphs_p_p_zz,
        (false, true, _, Eq | Ne) => C::Fcmeq_asimdsame,
        (false, true, _, Gt | Lt) => C::Fcmgt_asimdsame,
        (false, true, _, Ge | Le) => C::Fcmge_asimdsame,
        (false, false, _, Eq | Ne) => C::Cmeq_asimdsame,
        (false, false, true, Gt | Lt) => C::Cmgt_asimdsame,
        (false, false, true, Ge | Le) => C::Cmge_asimdsame,
        (false, false, false, Gt | Lt) => C::Cmhi_asimdsame,
        (false, false, false, Ge | Le) => C::Cmhs_asimdsame,
        _ => return None,
    };
    let packed = if scalable {
        Packed::SveCompare {
            code,
            arrangement: sve_arr,
        }
    } else {
        Packed::NeonCompare {
            code,
            arrangement: neon,
            invert: op == Ne,
        }
    };
    Some((packed, swap))
}

fn scalar_binary(op: Binary, ty: Type) -> Option<IntBinary> {
    use Binary::*;
    use IntOp::*;
    let lane = ty.lane()?;
    let size = lane.bytes();
    if size > 8 || (lane.float() && (size > 2 || !matches!(op, And | Or | Xor))) {
        return None;
    }
    let choose = |word, double| if size == 8 { double } else { word };
    let signed = lane.signed();
    let native = match op {
        Add => Three(choose(C::Add_32_addsub_shift, C::Add_64_addsub_shift)),
        Sub => Three(choose(C::Sub_32_addsub_shift, C::Sub_64_addsub_shift)),
        And => Three(choose(C::And_32_log_shift, C::And_64_log_shift)),
        Or => Three(choose(C::Orr_32_log_shift, C::Orr_64_log_shift)),
        Xor => Three(choose(C::Eor_32_log_shift, C::Eor_64_log_shift)),
        Mul => Multiply(choose(C::Madd_32a_dp_3src, C::Madd_64a_dp_3src)),
        Div | Rem => Divide {
            signed,
            remainder: op == Rem,
        },
        Shl => Three(choose(C::Lslv_32_dp_2src, C::Lslv_64_dp_2src)),
        Shr if signed => Three(choose(C::Asrv_32_dp_2src, C::Asrv_64_dp_2src)),
        Shr => Three(choose(C::Lsrv_32_dp_2src, C::Lsrv_64_dp_2src)),
        Eq => Compare(Cond::Eq),
        Ne => Compare(Cond::Ne),
        Lt => Compare(if signed { Cond::Lt } else { Cond::Cc }),
        Le => Compare(if signed { Cond::Le } else { Cond::Ls }),
        Gt => Compare(if signed { Cond::Gt } else { Cond::Hi }),
        Ge => Compare(if signed { Cond::Ge } else { Cond::Cs }),
        Min => MinMax(if signed { Cond::Le } else { Cond::Ls }),
        Max => MinMax(if signed { Cond::Ge } else { Cond::Cs }),
        _ => return None,
    };
    Some(IntBinary {
        op: native,
        size,
        signed_inputs: signed && matches!(op, Lt | Le | Gt | Ge | Min | Max | Div | Rem | Shr),
        wrap_shift: size < 4 && matches!(op, Shl | Shr),
    })
}

fn packed_binary(op: Binary, ty: Type, profile: Profile) -> Option<Packed> {
    use Binary::*;
    let Type::Vector(lane) = ty else { return None };
    let size = lane.bytes();
    let bitwise = matches!(op, And | Or | Xor);
    if !bitwise {
        if matches!(lane, Lane::F32 | Lane::F64) {
            let (neon, scalable) = arrangements(size);
            if sve(profile) {
                let code = match op {
                    Add => C::Fadd_z_zz_,
                    Sub => C::Fsub_z_zz_,
                    Mul => C::Fmul_z_zz_,
                    Div => return Some(Packed::SveDestructive(C::Fdiv_z_p_zz_, scalable, false)),
                    _ => return None,
                };
                return Some(Packed::Sve(code, scalable));
            }
            let code = match op {
                Add => C::Fadd_asimdsame,
                Sub => C::Fsub_asimdsame,
                Mul => C::Fmul_asimdsame,
                Div => C::Fdiv_asimdsame,
                _ => return None,
            };
            return Some(Packed::Neon(code, neon));
        }
        if lane.float() || size > 8 {
            return None;
        }
    }
    if sve(profile) {
        let a = if bitwise { A::D } else { arrangements(size).1 };
        let code = match op {
            Add => C::Add_z_zz_,
            Sub => C::Sub_z_zz_,
            Mul if profile == Profile::Sve2 => C::Mul_z_zz_,
            Mul => return Some(Packed::SveDestructive(C::Mul_z_p_zz_, a, true)),
            And => C::And_z_zz_,
            Or => C::Orr_z_zz_,
            Xor => C::Eor_z_zz_,
            _ => return None,
        };
        Some(Packed::Sve(code, a))
    } else {
        let a = if bitwise {
            A::B16
        } else {
            arrangements(size).0
        };
        let code = match op {
            Add => C::Add_asimdsame_only,
            Sub => C::Sub_asimdsame_only,
            Mul if size <= 4 => C::Mul_asimdsame_only,
            And => C::And_asimdsame_only,
            Or => C::Orr_asimdsame_only,
            Xor => C::Eor_asimdsame_only,
            _ => return None,
        };
        Some(Packed::Neon(code, a))
    }
}

fn neon_bits(op: Binary) -> Option<C> {
    match op {
        Binary::And => Some(C::And_asimdsame_only),
        Binary::Or => Some(C::Orr_asimdsame_only),
        Binary::Xor => Some(C::Eor_asimdsame_only),
        _ => None,
    }
}
