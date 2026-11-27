//! How AArch64 instructions read in listings: a family name, then the
//! attributes that are not operands. Encodings print by their name in the
//! Arm architecture description, types and operations as HIR spells them.
use super::half::{Action as HalfAction, Form as HalfForm};
use super::inst::{Expand, Inst, IntBinary, IntOp, Masked, Packed, Select, Shuffle};
use super::reduce::Form as ReduceForm;
use super::special::{Action as SpecialAction, Form as SpecialForm};
use super::unary::{FloatOp, FloatRegs, Form as UnaryForm};
use super::width::{Conversion, Narrow, Widen};
use super::{A, C, Cond, Config};
use crate::hir::text::{TypeName, binary_name, lane_name, trap_name, unary_name};
use crate::hir::{Lane, Type};
use crate::lir::printer::Named;
use regalloc2::{PReg, RegClass};

/// The words of one mnemonic, separated by spaces.
#[derive(Default)]
struct Words(Vec<String>);

impl Words {
    fn new(family: &str) -> Self {
        Self(vec![family.into()])
    }

    fn word(mut self, word: impl ToString) -> Self {
        self.0.push(word.to_string());
        self
    }

    fn code(self, code: C) -> Self {
        self.word(code.name().to_ascii_lowercase())
    }

    fn arrangement(self, arrangement: A) -> Self {
        self.word(arrangement.name())
    }

    fn ty(self, ty: Type) -> Self {
        self.word(TypeName(ty))
    }

    fn lane(self, lane: Lane) -> Self {
        self.word(lane_name(lane))
    }

    fn cond(self, cond: Cond) -> Self {
        self.word(cond.name())
    }

    /// `name` when `set`.
    fn flag(self, set: bool, name: &str) -> Self {
        if set { self.word(name) } else { self }
    }

    fn sve(self, scalable: bool) -> Self {
        self.flag(scalable, "sve")
    }

    fn bytes(self, size: usize) -> Self {
        self.word(format!("x{size}"))
    }

    fn done(self) -> String {
        self.0.join(" ")
    }
}

fn int(binary: IntBinary) -> Words {
    let words = match binary.op {
        IntOp::Three(code) => Words::new("int").code(code),
        IntOp::Multiply(code) => Words::new("int.mul").code(code),
        IntOp::Divide { signed, remainder } => {
            Words::new(if remainder { "int.rem" } else { "int.div" }).flag(signed, "signed")
        }
        IntOp::Compare(cond) => Words::new("int.cmp").cond(cond),
        IntOp::MinMax(cond) => Words::new("int.minmax").cond(cond),
    };
    words
        .bytes(binary.size)
        .flag(binary.signed_inputs, "signed_inputs")
        .flag(binary.wrap_shift, "wrap_shift")
}

fn unary(form: UnaryForm) -> Words {
    let words = Words::new("unary");
    match form {
        UnaryForm::Integer(op, size) => words.word(unary_name(op)).bytes(size),
        UnaryForm::Neon(code, arr) => words.code(code).arrangement(arr),
        UnaryForm::Sve(code, arr) => words.word("sve").code(code).arrangement(arr),
        UnaryForm::NeonPopcnt(size) => words.word("popcnt").bytes(size),
        UnaryForm::NeonCtz(arr) => words.word("ctz").arrangement(arr),
        UnaryForm::NeonCountZeros64(op) => words.word(unary_name(op)).bytes(8),
        UnaryForm::SveCtz(arr) => words.word("sve").word("ctz").arrangement(arr),
        UnaryForm::HalfSign(op, packed, scalable) => words
            .word("half")
            .word(unary_name(op))
            .flag(packed, "packed")
            .sve(scalable),
        UnaryForm::Float(op, regs) => {
            let words = match op {
                FloatOp::Native(code) => words.code(code),
                FloatOp::Fract(code) => words.word("fract").code(code),
                FloatOp::Recip => words.word("recip"),
            };
            match regs {
                FloatRegs::Scalar(arr) => words.arrangement(arr),
                FloatRegs::Neon(arr) => words.word("neon").arrangement(arr),
                FloatRegs::Sve(arr) => words.word("sve").arrangement(arr),
            }
        }
    }
}

fn half(form: HalfForm) -> Words {
    let words = Words::new("half").lane(form.lane);
    let words = match form.action {
        HalfAction::Binary(code) => words.code(code),
        HalfAction::Compare(cond) => words.word("cmp").cond(cond),
        HalfAction::Unary(op) => words.word(unary_name(op)),
        HalfAction::Cast(lane) => words.word("cast").lane(lane),
        HalfAction::MulAdd => words.word("fma"),
    };
    words.flag(form.packed, "packed").sve(form.scalable)
}

fn special(form: SpecialForm) -> Words {
    let words = Words::new("special").lane(form.lane);
    let words = match form.action {
        SpecialAction::MinMax(max) => words.word(if max { "max" } else { "min" }),
        SpecialAction::CopySign => words.word("copysign"),
        SpecialAction::Classify(op) => words.word(unary_name(op)),
        SpecialAction::Signum => words.word("signum"),
    };
    words.flag(form.packed, "packed").sve(form.scalable)
}

fn conversion(words: Words, form: Conversion) -> Words {
    words.lane(form.from).lane(form.to).sve(form.scalable)
}

fn widen(form: Widen) -> Words {
    let words = Words::new("widen");
    match form {
        Widen::Neon(code, from, to) => words.code(code).arrangement(from).arrangement(to),
        Widen::Sve {
            code,
            from,
            to,
            high,
        } => words
            .word("sve")
            .code(code)
            .arrangement(from)
            .arrangement(to)
            .flag(high, "high"),
        Widen::Wide {
            signed,
            scalable,
            high,
        } => words
            .word("wide")
            .flag(signed, "signed")
            .sve(scalable)
            .flag(high, "high"),
        Widen::Convert(form, high) => conversion(words.word("convert"), form).flag(high, "high"),
    }
}

fn narrow(form: Narrow) -> Words {
    let words = Words::new("narrow");
    match form {
        Narrow::Neon { codes, from, to } => words
            .code(codes[0])
            .code(codes[1])
            .arrangement(from)
            .arrangement(to[0])
            .arrangement(to[1]),
        Narrow::Sve { to, float } => words.word("sve").arrangement(to).flag(float, "float"),
        Narrow::NeonWide => words.word("wide"),
        Narrow::Convert(form) => conversion(words.word("convert"), form),
    }
}

fn reduce(form: ReduceForm) -> Words {
    let words = Words::new("reduce");
    match form {
        ReduceForm::Integer(code, arr, size) => words.code(code).arrangement(arr).bytes(size),
        ReduceForm::NeonBits(code, size) => words.word("bits").code(code).bytes(size),
        ReduceForm::NeonMinMax64(cond) => words.word("minmax").cond(cond).bytes(8),
        ReduceForm::FloatAdd(arr, scalable) => words.word("fadd").arrangement(arr).sve(scalable),
        ReduceForm::Ordered(op, lane, scalable) => words
            .word("ordered")
            .word(binary_name(op))
            .lane(lane)
            .sve(scalable),
    }
}

fn packed(form: Packed) -> Words {
    let words = Words::new("packed");
    match form {
        Packed::Neon(code, arr) => words.code(code).arrangement(arr),
        Packed::Sve(code, arr) => words.word("sve").code(code).arrangement(arr),
        Packed::SveDestructive(code, arr, swap) => words
            .word("sve")
            .code(code)
            .arrangement(arr)
            .flag(swap, "swap"),
        Packed::NeonMinMax64(code, signed) => words.code(code).flag(signed, "signed"),
        Packed::NeonShift(code, arr, right) => words
            .word("shift")
            .code(code)
            .arrangement(arr)
            .flag(right, "right"),
        Packed::SveShift(code, arr) => words.word("sve").word("shift").code(code).arrangement(arr),
        Packed::FloatOrder(arr, ordered) => words
            .word(if ordered { "ordered" } else { "unordered" })
            .arrangement(arr),
        Packed::NeonCompare {
            code,
            arrangement,
            invert,
        } => words
            .word("cmp")
            .code(code)
            .arrangement(arrangement)
            .flag(invert, "invert"),
        Packed::SveCompare { code, arrangement } => words
            .word("sve")
            .word("cmp")
            .code(code)
            .arrangement(arrangement),
    }
}

fn select(form: Select) -> Words {
    let words = Words::new("select");
    let arranged = |words: Words, arr: Option<A>| match arr {
        Some(arr) => words.arrangement(arr),
        None => words.word("bits"),
    };
    match form {
        Select::Scalar(ty) => words.ty(ty),
        Select::ScalarBits => words.word("bits"),
        Select::Neon(arr) => arranged(words, arr),
        Select::Sve(arr) => arranged(words.word("sve"), arr),
        Select::Wide(scalable) => words.word("wide").sve(scalable),
    }
}

impl Named for Inst {
    fn mnemonic(&self, _: &Config, _: &()) -> String {
        match *self {
            Self::Arg(ty, index, _) => Words::new("arg").ty(ty).word(index),
            Self::Literal(ty, bits, _) => Words::new("literal").ty(ty).word(format!("{bits:#x}")),
            Self::Copy(..) => Words::new("copy"),
            Self::BitMove(from, to, ..) => Words::new("bitcast").ty(from).ty(to),
            Self::Imm(code, imm, shift, ..) => Words::new("imm")
                .code(code)
                .word(imm)
                .word(format!("lsl={shift}")),
            Self::Int(binary, ..) => int(binary),
            Self::WideMath(form, ..) => Words::new("wide")
                .word(binary_name(form.op))
                .flag(form.signed, "signed")
                .flag(form.packed, "packed")
                .sve(form.scalable),
            Self::WideUnary(op, scalable, ..) => {
                Words::new("wide").word(unary_name(op)).sve(scalable)
            }
            Self::WideAdd {
                subtract, scalable, ..
            } => Words::new("wide")
                .word(if subtract { "sub" } else { "add" })
                .sve(scalable),
            Self::Unary(form, ..) => unary(form),
            Self::Half(form, ..) => half(form),
            Self::HalfCast(from, to, ..) => Words::new("half.cast").lane(from).lane(to),
            Self::IntCast(from, to, ..) => Words::new("cast.int").ty(from).ty(to),
            Self::WideCast(from, to, ..) => Words::new("cast.wide").ty(from).ty(to),
            Self::FloatCast(code, from, to, ..) => {
                Words::new("cast.float").code(code).ty(from).ty(to)
            }
            Self::PackedCast(code, arr, ..) => {
                Words::new("cast.packed").code(code).arrangement(arr)
            }
            Self::Select(form, ..) => select(form),
            Self::Fma(code, arr, ..) => Words::new("fma").code(code).arrangement(arr),
            Self::Sve2Ternary(code, ..) => Words::new("sve2.ternary").code(code),
            Self::Shuffle(form, ..) => match form {
                Shuffle::Neon(arr, tables) => Words::new("shuffle")
                    .arrangement(arr)
                    .word(format!("tables={tables}")),
                Shuffle::Sve(arr) => Words::new("shuffle").word("sve").arrangement(arr),
                Shuffle::Wide(scalable) => Words::new("shuffle").word("wide").sve(scalable),
            },
            Self::Compact { size, scalable, .. } => Words::new("compact").bytes(size).sve(scalable),
            Self::VectorDivide(form, ..) => Words::new(if form.remainder {
                "vector.rem"
            } else {
                "vector.div"
            })
            .bytes(form.size)
            .flag(form.signed, "signed")
            .sve(form.scalable),
            Self::Rotate(form, ..) => Words::new(if form.right { "rotr" } else { "rotl" })
                .bytes(form.size)
                .flag(form.packed, "packed")
                .sve(form.scalable),
            Self::ScalarSaturate(code, size, ..) => Words::new("saturate").code(code).bytes(size),
            Self::NeonMul64(..) => Words::new("mul64"),
            Self::FloatSpecial(form, ..) => special(form),
            Self::Expand(form, ..) => match form {
                Expand::Neon(arr, tables) => Words::new("expand")
                    .arrangement(arr)
                    .word(format!("tables={tables}")),
                Expand::Sve(arr) => Words::new("expand").word("sve").arrangement(arr),
                Expand::Sve2(arr) => Words::new("expand").word("sve2").arrangement(arr),
                Expand::Wide(profile) => Words::new("expand").word("wide").word(profile.name()),
            },
            Self::Widen(form, ..) => widen(form),
            Self::Narrow(form, ..) => narrow(form),
            Self::Reduce(form, ..) => reduce(form),
            Self::Float(code, arr, ..) => Words::new("float").code(code).arrangement(arr),
            Self::FloatCompare(cond, arr, ..) => {
                Words::new("float.cmp").cond(cond).arrangement(arr)
            }
            Self::Splat(code, arr, ty, ..) => {
                Words::new("splat").code(code).arrangement(arr).ty(ty)
            }
            Self::SveIota(lane, _) => Words::new("iota").lane(lane),
            Self::SveSplat128(..) => Words::new("splat128"),
            Self::MaskTest(all, arr, ..) => {
                Words::new(if all { "mask.all" } else { "mask.any" }).arrangement(arr)
            }
            Self::WideMaskTest(all, scalable, ..) => {
                Words::new(if all { "mask.all" } else { "mask.any" })
                    .word("wide")
                    .sve(scalable)
            }
            Self::LaneAccess { lane, insert, .. } => Words::new(if insert.is_some() {
                "insert"
            } else {
                "extract"
            })
            .ty(lane),
            Self::Packed(form, ..) => packed(form),
            Self::PackedShiftImm(code, arr, count, ..) => Words::new("packed.shift")
                .code(code)
                .arrangement(arr)
                .word(format!("imm={count}")),
            Self::Check(..) => Words::new("bounds"),
            Self::TrapIf(_, trap) => Words::new("trap_if").word(trap_name(trap)),
            Self::Trap(trap) => Words::new("trap").word(trap_name(trap)),
            Self::Offset(scale, ..) => Words::new("offset").word(1 << scale as u8),
            Self::Load(ty, ..) => Words::new("load").ty(ty),
            Self::Store(ty, ..) => Words::new("store").ty(ty),
            Self::DebugStore(ty, offset, _) => Words::new("debug_store").ty(ty).word(offset),
            Self::DebugPoint(point) => Words::new("debug_point").word(point),
            Self::Masked {
                kind,
                ty,
                indices,
                dst,
                ..
            } => {
                let words = Words::new(if dst.is_some() {
                    "maskedload"
                } else {
                    "maskedstore"
                })
                .ty(ty)
                .flag(indices.is_some(), "indexed");
                match kind {
                    Masked::Neon => words,
                    Masked::Sve { code, arrangement } => {
                        words.word("sve").code(code).arrangement(arrangement)
                    }
                    Masked::SveNarrow(code) => words.word("sve.narrow").code(code),
                    Masked::SveWide(code) => words.word("sve.wide").code(code),
                }
            }
            Self::StoreArg(ty, index, _) => Words::new("store_arg").ty(ty).word(index),
            Self::Call(target, ..) => Words::new("call").word(format!("f{}", target.index())),
            Self::LibArg(ty, index, _) => Words::new("libarg").ty(ty).word(index),
            Self::Libcall(action, input, output, _) => {
                Words::new("libcall").word(action).ty(input).ty(output)
            }
            Self::RustCall(function, ..) => Words::new("rust_call").word(function.name),
            Self::CallScratch(offset, bytes, _) => {
                Words::new("call_scratch").word(offset).word(bytes)
            }
            Self::LibResult(ty, _) => Words::new("libresult").ty(ty),
            Self::Jump => Words::new("jump"),
            Self::Branch(_) => Words::new("branch"),
            Self::Return(ty, _, root) => Words::new("ret").ty(ty).flag(root, "root"),
        }
        .done()
    }

    fn register(_: &Config, reg: PReg) -> String {
        let n = reg.hw_enc();
        match reg.class() {
            RegClass::Int => format!("x{n}"),
            RegClass::Float => format!("p{n}"),
            RegClass::Vector => format!("v{n}"),
        }
    }
}
