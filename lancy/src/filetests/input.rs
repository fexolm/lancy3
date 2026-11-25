//! Small, test-only spelling of Lancy SSA. Build the same Function as the frontend.
use crate::hir::{
    Binary, Block, Edge, Function, Lane, Op, Scale, Simd, SliceParam, Terminator, Ternary, Type,
    Unary, Value,
};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) type Lines<'a> = Vec<(usize, &'a str)>;
type Names<'a, T> = HashMap<&'a str, T>;
type Result<T> = std::result::Result<T, String>;
type Parsed<T> = std::result::Result<T, ParseError>;

/// A malformed filetest, or IR that the builder rejects while it is parsed.
#[derive(Debug)]
pub(crate) enum ParseError {
    Syntax(String),
    /// The builder's error, after the location prefixes of `at`.
    Invalid(String, crate::hir::IrError),
}

impl ParseError {
    fn at(self, location: impl std::fmt::Display) -> Self {
        match self {
            Self::Syntax(message) => Self::Syntax(format!("{location}: {message}")),
            Self::Invalid(place, error) => Self::Invalid(format!("{location}: {place}"), error),
        }
    }
}

impl From<String> for ParseError {
    fn from(message: String) -> Self {
        Self::Syntax(message)
    }
}

impl From<&str> for ParseError {
    fn from(message: &str) -> Self {
        Self::Syntax(message.to_owned())
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Syntax(message) => f.write_str(message),
            Self::Invalid(place, error) => write!(f, "{place}invalid IR: {error}"),
        }
    }
}

fn ty(s: &str) -> Result<Type> {
    if s == "ptr" {
        return Ok(Type::Ptr);
    }
    let vector = s.strip_prefix("vec.");
    let lane = match vector.unwrap_or(s) {
        "i8" => Lane::I8,
        "u8" => Lane::U8,
        "i16" => Lane::I16,
        "u16" => Lane::U16,
        "i32" => Lane::I32,
        "u32" => Lane::U32,
        "i64" => Lane::I64,
        "u64" => Lane::U64,
        "i128" => Lane::I128,
        "u128" => Lane::U128,
        "isize" => Lane::Isize,
        "usize" => Lane::Usize,
        "f16" => Lane::F16,
        "bf16" => Lane::BF16,
        "f32" => Lane::F32,
        "f64" => Lane::F64,
        _ => return Err(format!("unknown type {s}")),
    };
    Ok(if vector.is_some() {
        Type::Vector(lane)
    } else {
        lane.scalar()
    })
}

fn binary(name: &str) -> Result<Binary> {
    use Binary::*;
    Ok(match name {
        "add" => Add,
        "sub" => Sub,
        "mul" => Mul,
        "div" => Div,
        "rem" => Rem,
        "and" => And,
        "or" => Or,
        "xor" => Xor,
        "eq" => Eq,
        "ne" => Ne,
        "lt" => Lt,
        "le" => Le,
        "gt" => Gt,
        "ge" => Ge,
        "shl" => Shl,
        "shr" => Shr,
        "min" => Min,
        "max" => Max,
        "ordered" => Ordered,
        "unordered" => Unordered,
        "copysign" => CopySign,
        "pow" => Pow,
        "atan2" => Atan2,
        "hypot" => Hypot,
        "rotateleft" => RotateLeft,
        "rotateright" => RotateRight,
        "saturatingadd" => SaturatingAdd,
        "saturatingsub" => SaturatingSub,
        _ => return Err(format!("unknown instruction {name}")),
    })
}

fn element(s: &str) -> Result<Lane> {
    let t = ty(s)?;
    t.lane()
        .filter(|l| l.scalar() == t)
        .ok_or_else(|| format!("expected scalar element, got {s}"))
}

fn unary(name: &str) -> Result<Unary> {
    Ok(match name {
        "isnan" => Unary::IsNan,
        "isinfinite" => Unary::IsInfinite,
        "isfinite" => Unary::IsFinite,
        "isnormal" => Unary::IsNormal,
        "issubnormal" => Unary::IsSubnormal,
        "signbit" => Unary::SignBit,
        "signum" => Unary::Signum,
        "nextup" => Unary::NextUp,
        "nextdown" => Unary::NextDown,
        "neg" => Unary::Neg,
        "abs" => Unary::Abs,
        "not" => Unary::Not,
        "popcnt" => Unary::Popcnt,
        "leadingzeros" => Unary::LeadingZeros,
        "trailingzeros" => Unary::TrailingZeros,
        "sqrt" => Unary::Sqrt,
        "floor" => Unary::Floor,
        "ceil" => Unary::Ceil,
        "trunc" => Unary::Trunc,
        "round" => Unary::Round,
        "nearest" => Unary::Nearest,
        "fract" => Unary::Fract,
        "recip" => Unary::Recip,
        "exp" => Unary::Exp,
        "exp2" => Unary::Exp2,
        "ln" => Unary::Ln,
        "log2" => Unary::Log2,
        "log10" => Unary::Log10,
        "sin" => Unary::Sin,
        "cos" => Unary::Cos,
        "tan" => Unary::Tan,
        "asin" => Unary::Asin,
        "acos" => Unary::Acos,
        "atan" => Unary::Atan,
        "sinh" => Unary::Sinh,
        "cosh" => Unary::Cosh,
        "tanh" => Unary::Tanh,
        "asinh" => Unary::Asinh,
        "acosh" => Unary::Acosh,
        "atanh" => Unary::Atanh,
        "expm1" => Unary::ExpM1,
        "ln1p" => Unary::Ln1p,
        "cbrt" => Unary::Cbrt,
        _ => return Err(format!("unknown instruction {name}")),
    })
}

fn simd(name: &str, args: &[&str], v: impl Fn(&str) -> Result<Value>) -> Result<Simd> {
    Ok(match (name, args) {
        ("lanes", [t]) => Simd::LaneCount(element(t)?),
        ("iota", [t]) => Simd::Iota(element(t)?),
        ("shuffle", [x, y, i]) => Simd::Shuffle(v(x)?, v(y)?, v(i)?),
        ("any", [x]) => Simd::AnyTrue(v(x)?),
        ("all", [x]) => Simd::AllTrue(v(x)?),
        ("reduce", [op, x]) => Simd::Reduce(binary(op)?, v(x)?),
        (part @ ("widenlo" | "widenhi"), [t, x]) => Simd::Widen {
            to: element(t)?,
            high: part == "widenhi",
            value: v(x)?,
        },
        ("narrow", [t, x, y]) => Simd::Narrow(element(t)?, v(x)?, v(y)?),
        ("splat", [x]) => Simd::Splat(v(x)?),
        ("extract", [x, i]) => Simd::ExtractLane(v(x)?, v(i)?),
        ("insert", [x, i, y]) => Simd::InsertLane(v(x)?, v(i)?, v(y)?),
        (op @ ("maskedload" | "maskedstore"), [p, idx, mask, x]) => {
            let address = crate::hir::VectorAddress {
                base: v(p)?,
                indices: if *idx == "-" { None } else { Some(v(idx)?) },
            };
            if op == "maskedload" {
                Simd::MaskedLoad {
                    address,
                    mask: v(mask)?,
                    passthrough: v(x)?,
                }
            } else {
                Simd::MaskedStore {
                    address,
                    mask: v(mask)?,
                    value: v(x)?,
                }
            }
        }

        ("compress", [x, m]) => Simd::Compress(v(x)?, v(m)?),
        ("expand", [x, m]) => Simd::Expand(v(x)?, v(m)?),
        ("load", [t, p]) => Simd::Load(element(t)?, v(p)?),
        ("store", [p, x]) => Simd::Store(v(p)?, v(x)?),
        ("cast", [t, x]) => Simd::Cast(element(t)?, v(x)?),
        ("bitcast", [t, x]) => Simd::Bitcast(element(t)?, v(x)?),
        ("fma", [x, y, z]) => Simd::Ternary(Ternary::MulAdd, v(x)?, v(y)?, v(z)?),
        ("bitselect", [x, y, z]) => Simd::Ternary(Ternary::BitSelect, v(x)?, v(y)?, v(z)?),
        ("select", [x, y, z]) => Simd::Ternary(Ternary::Select, v(x)?, v(y)?, v(z)?),
        (name, [x]) => Simd::Unary(unary(name)?, v(x)?),
        (name, [x, y]) => Simd::Bin(binary(name)?, v(x)?, v(y)?),
        _ => return Err(format!("invalid SIMD instruction {name}")),
    })
}

fn params<'a>(words: &[&'a str]) -> Result<Vec<(&'a str, Type)>> {
    words
        .iter()
        .map(|word| {
            let (name, t) = word
                .split_once(':')
                .ok_or_else(|| format!("expected name:type, got {word}"))?;
            Ok((name, ty(t)?))
        })
        .collect()
}

fn define<'a, T>(names: &mut Names<'a, T>, name: &'a str, item: T) -> Result<()> {
    if name.is_empty() || names.insert(name, item).is_some() {
        return Err(format!("empty or duplicate name '{name}'"));
    }
    Ok(())
}

fn lookup<T: Copy>(names: &Names<T>, name: &str) -> Result<T> {
    names
        .get(name)
        .copied()
        .ok_or_else(|| format!("unknown name '{name}'"))
}

fn edge(
    f: &mut Function,
    blocks: &Names<Block>,
    values: &Names<Value>,
    words: &[&str],
) -> Result<Edge> {
    let (target, args) = words.split_first().ok_or("expected branch target")?;
    let args = args
        .iter()
        .map(|arg| lookup(values, arg))
        .collect::<Result<Vec<_>>>()?;
    Ok(f.add_edge(lookup(blocks, target)?, &args))
}

fn instruction<'a>(
    f: &mut Function,
    block: Block,
    words: &[&'a str],
    values: &mut Names<'a, Value>,
    blocks: &Names<Block>,
    functions: &Names<Arc<Function>>,
) -> Parsed<bool> {
    let v = |name: &str| lookup(values, name);
    let term = match words {
        ["ret", name] => Some(Terminator::Return(v(name)?)),
        ["jump", args @ ..] => Some(Terminator::Jump(edge(f, blocks, values, args)?)),
        ["br", cond, arms @ ..] => {
            let pos = arms
                .iter()
                .position(|s| *s == "|")
                .ok_or("expected | between edges")?;
            Some(Terminator::Branch {
                cond: v(cond)?,
                yes: edge(f, blocks, values, &arms[..pos])?,
                no: edge(f, blocks, values, &arms[pos + 1..])?,
            })
        }
        _ => None,
    };
    if let Some(term) = term {
        f.set_terminator(block, term);
        return Ok(true);
    }
    let (dst, words) = match words {
        [dst, "=", op @ ..] => (Some(*dst), op),
        _ => (None, words),
    };
    let op = match words {
        [name, args @ ..] if name.starts_with("simd.") => Op::Simd(simd(&name[5..], args, v)?),
        ["const", n] => Op::Const(super::integer(n)?),
        ["literal", t, n] => {
            let lane = element(t)?;
            let bits = u128::from_str_radix(n.trim_start_matches("0x"), 16)
                .map_err(|_| "expected literal bits in hex")?;
            Op::literal(lane, bits)
        }
        ["cast", t, x] => Op::Cast(ty(t)?, v(x)?),
        ["bitcast", t, x] => Op::Bitcast(ty(t)?, v(x)?),
        ["fma", x, y, z] => Op::Ternary(Ternary::MulAdd, v(x)?, v(y)?, v(z)?),
        ["select", x, y, z] => Op::Ternary(Ternary::Select, v(x)?, v(y)?, v(z)?),
        ["bitselect", x, y, z] => Op::Ternary(Ternary::BitSelect, v(x)?, v(y)?, v(z)?),
        ["copy", x] => Op::Copy(v(x)?),
        ["bounds", i, n] => Op::BoundsCheck(v(i)?, v(n)?),
        ["check_overflow", cond] => Op::CheckOverflow(v(cond)?),
        ["rust_call", "identity_or_error", value, fail] => Op::RustCall(
            &crate::hir::RustImport {
                name: "identity_or_error",
                resolve: || identity_or_error,
            },
            v(value)?,
            v(fail)?,
        ),
        ["rust_call", "double", value, unused] => Op::RustCall(
            &crate::hir::RustImport {
                name: "double",
                resolve: || double,
            },
            v(value)?,
            v(unused)?,
        ),
        ["rust_packet_call", "sum4", args @ ..] => {
            fn sum(words: &[i32]) -> i32 {
                words.iter().copied().fold(0, i32::wrapping_add)
            }
            let args = args.iter().map(|arg| v(arg)).collect::<Result<Vec<_>>>()?;
            Op::RustPacketCall(f.add_packet_call(
                &crate::hir::RustPacketImport {
                    name: "sum4",
                    arity: 4,
                    resolve: || sum,
                },
                0,
                &args,
            ))
        }
        ["offset", p, i, stride] => Op::PtrOffset(
            v(p)?,
            v(i)?,
            match *stride {
                "1" => Scale::One,
                "2" => Scale::Two,
                "4" => Scale::Four,
                "8" => Scale::Eight,
                _ => return Err(format!("invalid byte stride {stride}").into()),
            },
        ),
        ["call_scratch", offset, bytes] => Op::CallScratch {
            offset: super::integer(offset)? as u32,
            bytes: super::integer(bytes)? as u32,
        },
        ["debug_store", offset, x] => Op::DebugStore(super::integer(offset)? as u32, v(x)?),
        ["load", t, p] => Op::Load(ty(t)?, v(p)?),
        ["store", p, x] => Op::Store(v(p)?, v(x)?),
        ["call", name, args @ ..] => {
            let callee = functions
                .get(name)
                .ok_or_else(|| format!("unknown callee {name}"))?;
            let args = args.iter().map(|arg| v(arg)).collect::<Result<Vec<_>>>()?;
            Op::Call(f.add_call(Arc::clone(callee), &args))
        }
        [name, x] => Op::Unary(unary(name)?, v(x)?),
        [name, a, b] => Op::Bin(binary(name)?, v(a)?, v(b)?),
        _ => return Err(format!("invalid instruction: {}", words.join(" ")).into()),
    };
    let inst = f
        .append_inst(block, op)
        .map_err(|error| ParseError::Invalid(String::new(), error))?;
    match (dst, f.result(inst)) {
        (Some(name), Some(value)) => define(values, name, value)?,
        (None, None) => {}
        _ => return Err("result binding does not match the instruction".into()),
    }
    Ok(false)
}

/// Return `value`, or report the trap whose code is `fail`.
extern "C" fn identity_or_error(value: i32, fail: i32) -> i64 {
    match crate::Trap::from_code(fail as u32) {
        Some(trap) => trap.status(),
        None => i64::from(value as u32),
    }
}

extern "C" fn double(value: i32, _: i32) -> i64 {
    i64::from(value.wrapping_mul(2) as u32)
}

fn body<'a>(
    signature: &[&'a str],
    lines: &[(usize, &'a str)],
    functions: &Names<Arc<Function>>,
) -> Parsed<Function> {
    let [args @ .., "->", ret] = signature else {
        return Err("expected arguments -> type".into());
    };
    let args = params(args)?;
    let mut f = Function::new(&args.iter().map(|a| a.1).collect::<Vec<_>>(), ty(ret)?);
    let mut block = f.entry();
    let mut blocks = Names::from([("entry", block)]);
    let mut values = Names::new();
    for ((name, _), &v) in args.iter().zip(f.params(block)) {
        define(&mut values, name, v)?;
    }
    // Allocate block parameters before parsing forward edges or backedges.
    for &(line, text) in lines {
        let words: Vec<_> = text.split_whitespace().collect();
        if let ["block", name, args @ ..] = words.as_slice() {
            let b = f.create_block();
            let at_line = |e: String| format!("line {line}: {e}");
            define(&mut blocks, name, b).map_err(&at_line)?;
            for (name, t) in params(args).map_err(&at_line)? {
                define(&mut values, name, f.append_typed_param(b, t)).map_err(&at_line)?;
            }
        }
    }
    let mut ended = false;
    for &(line, text) in lines {
        let words: Vec<_> = text.split_whitespace().collect();
        if let ["block", name, ..] = words.as_slice() {
            block = blocks[name];
            ended = false;
        } else if let ["slices", first, layout @ ..] = words.as_slice() {
            let first = first
                .parse::<usize>()
                .map_err(|_| format!("line {line}: invalid slice parameter index"))?;
            let slices = layout
                .iter()
                .map(|&entry| match entry {
                    "read2" => Ok((2, false)),
                    "read4" => Ok((4, false)),
                    "read8" => Ok((8, false)),
                    "write2" => Ok((2, true)),
                    "write4" => Ok((4, true)),
                    "write8" => Ok((8, true)),
                    _ => Err(format!("line {line}: invalid slice layout {entry}")),
                })
                .collect::<Result<Box<_>>>()?;
            let params = f.params(f.entry());
            let slices = slices
                .iter()
                .enumerate()
                .map(|(i, &(element, writable))| {
                    let pair = first + 2 * i;
                    Ok(SliceParam {
                        ptr: *params.get(pair).ok_or("slice past the parameters")?,
                        len: *params.get(pair + 1).ok_or("slice past the parameters")?,
                        element,
                        writable,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            f.set_slice_params(slices)
                .map_err(|error| ParseError::Invalid(format!("line {line}: "), error))?;
        } else {
            if ended {
                return Err(format!("line {line}: instruction after terminator").into());
            }
            ended = instruction(&mut f, block, &words, &mut values, &blocks, functions)
                .map_err(|e| e.at(format_args!("line {line}")))?;
        }
    }
    Ok(f)
}

pub(crate) fn parse(mut lines: &[(usize, &str)]) -> Parsed<Arc<Function>> {
    let mut functions = Names::new();
    while let Some((&(line, header), rest)) = lines.split_first() {
        let words: Vec<_> = header.split_whitespace().collect();
        let ["fn", name, signature @ ..] = words.as_slice() else {
            return Err(format!("line {line}: expected function header").into());
        };
        let len = rest
            .iter()
            .position(|(_, s)| s.split_whitespace().next() == Some("fn"))
            .unwrap_or(rest.len());
        let f = body(signature, &rest[..len], &functions)
            .map_err(|e| e.at(format_args!("fn {name} (line {line})")))?;
        define(&mut functions, name, Arc::new(f))?;
        lines = &rest[len..];
    }
    functions.remove("main").ok_or("missing fn main".into())
}
