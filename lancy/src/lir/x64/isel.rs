use super::{
    Features,
    inst::simd::Native,
    inst::{Address, Alu, Cond, Inst, Quotient, Rhs, class},
};
use crate::CompileError;
use crate::hir::{Binary, Function, Op, Scale, Terminator, Value, transforms::ControlFlowGraph};
use crate::hir::{Simd, Type};
use crate::lir::isel::{self, Context, Edges, Root};
use crate::numbers::Math;
use crate::{lir, module::FunctionId};
use regalloc2::{PRegSet, RegClass, VReg};

fn reg(f: &Function, v: Value) -> VReg {
    VReg::new(v.index(), class(f.value_type(v)))
}

pub(super) fn select(
    f: &Function,
    cfg: &ControlFlowGraph,
    root: bool,
    callee_info: impl Fn(&Function) -> (FunctionId, PRegSet),
) -> Result<lir::Function<Inst>, CompileError> {
    let reg = |v| reg(f, v);
    let cpu = Features::host();
    let edges = Edges::new(f, cfg);
    let mut body = edges.function()?;
    let mut blocks = isel::select(f, cfg, |ctx, root, out| {
        let mut rules = Rules {
            ctx,
            f,
            body: &mut body,
            out,
        };
        match root {
            Root::Term(term) => rules.branch(term),
            Root::Inst(i) => {
                let dst = || reg(f.result(i).expect("value-producing instruction"));
                let op = *f.op(i);
                if let Op::Bin(kind, a, b) = op
                    && f.value_type(a) == Type::I32
                    && let Some(inst) = rules.i32_binary(kind, dst(), a, b)?
                {
                    rules.out.push(inst);
                    return Ok(());
                }
                let op = match op {
                    Op::Const(n) => Inst::Imm(dst(), n),
                    Op::Literal(l, n) => Inst::Literal(l, n, dst()),
                    Op::Simd(Simd::Iota(l)) => Inst::Iota(l, dst()),
                    Op::Bitcast(to, v) if class(to) == class(f.value_type(v)) => {
                        Inst::Copy(dst(), rules.input(v))
                    }
                    Op::Bitcast(to, v) => Inst::BitMove(f.value_type(v), to, dst(), rules.input(v)),
                    Op::Simd(Simd::Bitcast(_, v)) => Inst::Copy(dst(), rules.input(v)),
                    Op::Copy(v) => Inst::Copy(dst(), rules.input(v)),
                    Op::Bin(
                        kind @ (Binary::Add
                        | Binary::Sub
                        | Binary::Mul
                        | Binary::And
                        | Binary::Or
                        | Binary::Xor),
                        a,
                        b,
                    ) if class(f.value_type(a)) == RegClass::Int
                        && !f.value_type(a).lane().unwrap().float() =>
                    {
                        Inst::IntBinary(
                            kind,
                            f.value_type(a),
                            dst(),
                            rules.input(a),
                            rules.input(b),
                        )
                    }
                    Op::Simd(Simd::LaneCount(lane)) => {
                        Inst::Imm(dst(), (cpu.bytes / lane.bytes()) as i32)
                    }
                    Op::BoundsCheck(i, n) => {
                        if rules.ctx.constant(n) == Some(1)
                            && let Some(Op::Simd(Simd::AnyTrue(mask))) = rules.ctx.single_use(i)
                        {
                            Inst::CheckMask(rules.input(mask))
                        } else {
                            Inst::CheckBounds(rules.input(i), rules.input(n))
                        }
                    }
                    Op::PtrOffset(base, i, scale) => {
                        Inst::Lea(dst(), rules.offset(base, i, scale)?)
                    }
                    Op::Simd(Simd::MaskedLoad {
                        address,
                        mask,
                        passthrough: value,
                    })
                    | Op::Simd(Simd::MaskedStore {
                        address,
                        mask,
                        value,
                    }) => Inst::Masked {
                        ty: f.value_type(value),
                        base: rules.input(address.base),
                        indices: address.indices.map(|v| rules.input(v)),
                        temp: if cpu.bytes == 32
                            && cpu.avx10 == 0
                            && cpu.avx2
                            && address.indices.is_some()
                            && f.result(i).is_some()
                            && matches!(f.value_type(value).lane().unwrap().bytes(), 4 | 8)
                        {
                            Some(rules.body.temp(RegClass::Vector)?)
                        } else {
                            None
                        },
                        mask: rules.input(mask),
                        value: rules.input(value),
                        dst: f.result(i).map(reg),
                    },
                    Op::Simd(Simd::Load(l, p)) => {
                        Inst::Load(Type::Vector(l), dst(), rules.address(p)?)
                    }
                    Op::Load(ty, p) => Inst::Load(ty, dst(), rules.address(p)?),
                    Op::Store(p, v) | Op::Simd(Simd::Store(p, v)) => {
                        Inst::Store(f.value_type(v), rules.address(p)?, rules.input(v))
                    }
                    Op::Call(site) => {
                        let (callee, args) = f.call(site);
                        let mut index = 0;
                        for &arg in args {
                            let ty = f.value_type(arg);
                            let value = rules.input(arg);
                            rules.out.push(Inst::StoreArg(index, ty, value));
                            index += ty.bytes(cpu.bytes).div_ceil(8);
                        }
                        let (target, written) = callee_info(callee);
                        Inst::call(target, dst(), written)
                    }
                    op => {
                        let (math, args) =
                            Math::from_op(op, |v| f.value_type(v)).expect("numeric instruction");
                        let args = args.map(|v| v.map(|v| rules.input(v)));
                        match Native::select(math, cpu) {
                            Some(native) => {
                                let mut temps = [None; 6];
                                for t in &mut temps[..native.temporaries()] {
                                    *t = Some(rules.body.temp(RegClass::Vector)?);
                                }
                                Inst::Native(native, dst(), args, temps)
                            }
                            None => Inst::Lanes(math, dst(), args),
                        }
                    }
                };
                rules.out.push(op);
            }
        }
        Ok(())
    })?;
    for &b in cfg.reverse_postorder() {
        let entry = b == f.entry();
        if entry {
            let argv = if root && f.arity() != 0 {
                let addr = body.temp(RegClass::Int)?;
                body.push(Inst::Argv(addr));
                Some(addr)
            } else {
                None
            };
            let mut index = 0;
            for &arg in f.params(b) {
                let ty = f.value_type(arg);
                body.push(Inst::LoadArg {
                    dst: reg(arg),
                    argv,
                    ty,
                    index,
                });
                index += ty.bytes(cpu.bytes).div_ceil(8);
            }
        }
        for op in blocks[b].drain(..) {
            body.push(op);
        }
        edges.finish(&mut body, b, reg, || Inst::Jump)?;
    }
    Ok(body)
}

struct Rules<'a> {
    ctx: &'a mut Context,
    f: &'a Function,
    body: &'a mut lir::Function<Inst>,
    out: &'a mut Vec<Inst>,
}

impl Rules<'_> {
    fn input(&mut self, v: Value) -> VReg {
        self.ctx.require(v);
        reg(self.f, v)
    }

    fn rhs(&mut self, v: Value) -> Rhs {
        match self.ctx.constant(v) {
            Some(n) => Rhs::Imm(n),
            None => Rhs::Reg(self.input(v)),
        }
    }

    fn branch(&mut self, term: Terminator) {
        let inst = match term {
            Terminator::Return(v) => Inst::Ret(self.input(v)),
            Terminator::Jump(_) => Inst::Jump,
            Terminator::Branch { cond, .. } => {
                if let Some(Op::Simd(Simd::AnyTrue(v))) = self.ctx.single_use(cond) {
                    Inst::BranchAny(self.input(v))
                } else if let Some(Op::Bin(kind, a, b)) = self.ctx.single_use(cond)
                    && self.f.value_type(a) == crate::hir::Type::I32
                    && let Some(cc) = Cond::from_binary(kind)
                {
                    Inst::BrCmp(cc, self.input(a), self.rhs(b))
                } else {
                    Inst::Branch(self.input(cond))
                }
            }
        };
        self.out.push(inst);
    }

    fn address(&mut self, v: Value) -> Result<Address, CompileError> {
        match self.ctx.single_use(v) {
            Some(Op::PtrOffset(base, index, scale)) => self.offset(base, index, scale),
            _ => Ok(Address {
                base: self.input(v),
                index: None,
                disp: 0,
            }),
        }
    }

    fn offset(&mut self, base: Value, index: Value, scale: Scale) -> Result<Address, CompileError> {
        let base = self.input(base);
        if let Some(disp) = self
            .ctx
            .constant(index)
            .and_then(|i| i.checked_mul(1 << scale as u8))
        {
            return Ok(Address {
                base,
                index: None,
                disp,
            });
        }
        let wide = self.body.temp(RegClass::Int)?;
        let index = self.input(index);
        self.out.push(Inst::Movsxd(wide, index));
        Ok(Address {
            base,
            index: Some((wide, scale)),
            disp: 0,
        })
    }

    fn i32_binary(
        &mut self,
        kind: Binary,
        dst: VReg,
        mut a: Value,
        mut b: Value,
    ) -> Result<Option<Inst>, CompileError> {
        if let Some(cc) = Cond::from_binary(kind) {
            return Ok(Some(Inst::Compare(cc, dst, self.input(a), self.rhs(b))));
        }
        if matches!(
            kind,
            Binary::Add | Binary::Mul | Binary::And | Binary::Or | Binary::Xor
        ) && self.ctx.constant(a).is_some()
            && self.ctx.constant(b).is_none()
        {
            std::mem::swap(&mut a, &mut b);
        }
        Ok(Some(match kind {
            Binary::Div | Binary::Rem => {
                if let Some(divisor) = self.ctx.constant(b) {
                    let remainder = kind == Binary::Rem;
                    if divisor == 1 {
                        return Ok(Some(if remainder {
                            Inst::Imm(dst, 0)
                        } else {
                            Inst::Copy(dst, self.input(a))
                        }));
                    }
                    if !matches!(divisor, 0 | -1) {
                        return Ok(Some(Inst::DivImm {
                            remainder,
                            dst,
                            value: self.input(a),
                            divisor,
                            quotient: Quotient::for_divisor(divisor),
                        }));
                    }
                }
                let a = self.input(a);
                let hi = self.body.temp(RegClass::Int)?;
                self.out.push(Inst::Cdq { lo: a, hi });
                Inst::IDiv {
                    remainder: kind == Binary::Rem,
                    dst,
                    lo: a,
                    hi,
                    divisor: self.input(b),
                }
            }
            Binary::Shl | Binary::Shr => Inst::Shift {
                left: kind == Binary::Shl,
                dst,
                value: self.input(a),
                count: self.rhs(b),
            },
            Binary::Add => Inst::Alu(Alu::Add, dst, self.input(a), self.rhs(b)),
            Binary::Sub => Inst::Alu(Alu::Sub, dst, self.input(a), self.rhs(b)),
            Binary::Mul => Inst::Alu(Alu::Mul, dst, self.input(a), self.rhs(b)),
            Binary::And => Inst::Alu(Alu::And, dst, self.input(a), self.rhs(b)),
            Binary::Or => Inst::Alu(Alu::Or, dst, self.input(a), self.rhs(b)),
            Binary::Xor => Inst::Alu(Alu::Xor, dst, self.input(a), self.rhs(b)),
            Binary::Min => Inst::Alu(Alu::Min, dst, self.input(a), Rhs::Reg(self.input(b))),
            Binary::Max => Inst::Alu(Alu::Max, dst, self.input(a), Rhs::Reg(self.input(b))),
            _ => return Ok(None),
        }))
    }
}
