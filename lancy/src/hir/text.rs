//! The textual form of Lancy HIR: one syntax that [`Function`]'s `Display`
//! prints and [`parse`] reads, used by fixtures, listings and debugging.
//!
//! ```text
//! fn callee(x:i32) -> i32
//!   ret x
//! fn main(p:ptr n:i32 x:i32) -> i32
//!   slice p n read4
//!   one = const 1
//!   jump head(one)
//! block head(i:i32)
//!   v:vec.f32 = load vec.f32 p
//!   r = call callee(i)
//!   more = lt r n
//!   br more head(r) | done()
//! block done()
//!   ret i
//! ```
//!
//! Functions are listed callees first; the last one is the entry. Its first
//! block is implicit and takes the function's parameters. A result may name
//! its type (`v:vec.f32`); the printer always does. Values and blocks may
//! have any name without spaces or parentheses; the printer names them `vN`
//! and `bN`.
use super::{
    Binary, Block, Call, Edge, Function, IrError, Lane, Op, RustFunction, Scale, Simd, SliceParam,
    Terminator, Ternary, Type, Unary, Value, VectorAddress,
};
use crate::Trap;
use std::collections::HashMap;
use std::fmt::{self, Display, Write};
use std::sync::Arc;

const LANES: [(Lane, &str); 16] = [
    (Lane::I8, "i8"),
    (Lane::U8, "u8"),
    (Lane::I16, "i16"),
    (Lane::U16, "u16"),
    (Lane::I32, "i32"),
    (Lane::U32, "u32"),
    (Lane::I64, "i64"),
    (Lane::U64, "u64"),
    (Lane::I128, "i128"),
    (Lane::U128, "u128"),
    (Lane::Isize, "isize"),
    (Lane::Usize, "usize"),
    (Lane::F16, "f16"),
    (Lane::BF16, "bf16"),
    (Lane::F32, "f32"),
    (Lane::F64, "f64"),
];

const BINARIES: [(Binary, &str); 28] = [
    (Binary::Add, "add"),
    (Binary::Sub, "sub"),
    (Binary::Mul, "mul"),
    (Binary::Div, "div"),
    (Binary::Rem, "rem"),
    (Binary::And, "and"),
    (Binary::Or, "or"),
    (Binary::Xor, "xor"),
    (Binary::Eq, "eq"),
    (Binary::Ne, "ne"),
    (Binary::Lt, "lt"),
    (Binary::Le, "le"),
    (Binary::Gt, "gt"),
    (Binary::Ge, "ge"),
    (Binary::Shl, "shl"),
    (Binary::Shr, "shr"),
    (Binary::Min, "min"),
    (Binary::Max, "max"),
    (Binary::Ordered, "ordered"),
    (Binary::Unordered, "unordered"),
    (Binary::CopySign, "copysign"),
    (Binary::Pow, "pow"),
    (Binary::Atan2, "atan2"),
    (Binary::Hypot, "hypot"),
    (Binary::RotateLeft, "rotateleft"),
    (Binary::RotateRight, "rotateright"),
    (Binary::SaturatingAdd, "saturatingadd"),
    (Binary::SaturatingSub, "saturatingsub"),
];

const UNARIES: [(Unary, &str); 43] = [
    (Unary::Neg, "neg"),
    (Unary::Abs, "abs"),
    (Unary::Not, "not"),
    (Unary::Popcnt, "popcnt"),
    (Unary::LeadingZeros, "leadingzeros"),
    (Unary::TrailingZeros, "trailingzeros"),
    (Unary::Sqrt, "sqrt"),
    (Unary::Floor, "floor"),
    (Unary::Ceil, "ceil"),
    (Unary::Trunc, "trunc"),
    (Unary::Round, "round"),
    (Unary::Nearest, "nearest"),
    (Unary::Fract, "fract"),
    (Unary::Recip, "recip"),
    (Unary::Exp, "exp"),
    (Unary::Exp2, "exp2"),
    (Unary::Ln, "ln"),
    (Unary::Log2, "log2"),
    (Unary::Log10, "log10"),
    (Unary::Sin, "sin"),
    (Unary::Cos, "cos"),
    (Unary::Tan, "tan"),
    (Unary::Asin, "asin"),
    (Unary::Acos, "acos"),
    (Unary::Atan, "atan"),
    (Unary::IsNan, "isnan"),
    (Unary::IsInfinite, "isinfinite"),
    (Unary::IsFinite, "isfinite"),
    (Unary::IsNormal, "isnormal"),
    (Unary::IsSubnormal, "issubnormal"),
    (Unary::SignBit, "signbit"),
    (Unary::Signum, "signum"),
    (Unary::NextUp, "nextup"),
    (Unary::NextDown, "nextdown"),
    (Unary::Sinh, "sinh"),
    (Unary::Cosh, "cosh"),
    (Unary::Tanh, "tanh"),
    (Unary::Asinh, "asinh"),
    (Unary::Acosh, "acosh"),
    (Unary::Atanh, "atanh"),
    (Unary::ExpM1, "expm1"),
    (Unary::Ln1p, "ln1p"),
    (Unary::Cbrt, "cbrt"),
];

const TERNARIES: [(Ternary, &str); 3] = [
    (Ternary::MulAdd, "fma"),
    (Ternary::Select, "select"),
    (Ternary::BitSelect, "bitselect"),
];

const SCALES: [(Scale, &str); 4] = [
    (Scale::One, "1"),
    (Scale::Two, "2"),
    (Scale::Four, "4"),
    (Scale::Eight, "8"),
];

fn name_of<T: PartialEq + Copy>(table: &[(T, &'static str)], item: T) -> &'static str {
    table.iter().find(|(x, _)| *x == item).unwrap().1
}

fn find<T: Copy>(table: &[(T, &str)], name: &str, what: &str) -> std::result::Result<T, String> {
    table
        .iter()
        .find(|(_, n)| *n == name)
        .map(|&(x, _)| x)
        .ok_or_else(|| format!("unknown {what} {name}"))
}

/// The spelling of operations, lanes and traps, shared with LIR listings.
pub(crate) fn binary_name(op: Binary) -> &'static str {
    name_of(&BINARIES, op)
}

pub(crate) fn unary_name(op: Unary) -> &'static str {
    name_of(&UNARIES, op)
}

pub(crate) fn ternary_name(op: Ternary) -> &'static str {
    name_of(&TERNARIES, op)
}

pub(crate) fn trap_name(trap: Trap) -> String {
    format!("{trap:?}")
}

pub(crate) fn lane_name(lane: Lane) -> &'static str {
    name_of(&LANES, lane)
}

/// The spelling of a type: `ptr`, a lane (`i32`) or a vector (`vec.i32`).
pub(crate) struct TypeName(pub(crate) Type);

impl Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Type::Ptr => f.write_str("ptr"),
            Type::Vector(lane) => write!(f, "vec.{}", lane_name(lane)),
            ty => f.write_str(lane_name(ty.lane().unwrap())),
        }
    }
}

fn slice_layout(element: u8, writable: bool) -> String {
    format!("{}{element}", if writable { "write" } else { "read" })
}

struct V(Value);

impl Display for V {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0.index())
    }
}

fn values(list: &[Value]) -> String {
    list.iter()
        .map(|&v| V(v).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Print `f` as function `name`; `callee` names the function a call site calls.
pub(crate) fn write_function(
    out: &mut impl Write,
    name: &str,
    f: &Function,
    callee: &dyn Fn(Call) -> String,
) -> fmt::Result {
    let entry = f.entry();
    let params = |b: Block| {
        f.params(b)
            .iter()
            .map(|&v| format!("{}:{}", V(v), TypeName(f.value_type(v))))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let block = |b: Block| {
        if b == entry {
            "entry".to_owned()
        } else {
            format!("b{}", b.index())
        }
    };
    let edge = |e: Edge| {
        let (target, args) = f.edge(e);
        format!("{}({})", block(target), values(args))
    };
    writeln!(
        out,
        "fn {name}({}) -> {}",
        params(entry),
        TypeName(f.return_type())
    )?;
    for slice in f.slice_params() {
        writeln!(
            out,
            "  slice {} {} {}",
            V(slice.ptr),
            V(slice.len),
            slice_layout(slice.element, slice.writable)
        )?;
    }
    let order = std::iter::once(entry).chain(f.blocks().filter(|&b| b != entry));
    for b in order {
        if b != entry {
            writeln!(out, "block {}({})", block(b), params(b))?;
        }
        for inst in f.instructions(b) {
            out.write_str("  ")?;
            if let Some(result) = f.result(inst) {
                write!(out, "{}:{} = ", V(result), TypeName(f.value_type(result)))?;
            }
            write_op(out, *f.op(inst), f, callee)?;
            out.write_char('\n')?;
        }
        match *f.terminator(b) {
            Terminator::Return(v) => writeln!(out, "  ret {}", V(v))?,
            Terminator::Jump(e) => writeln!(out, "  jump {}", edge(e))?,
            Terminator::Branch { cond, yes, no } => {
                writeln!(out, "  br {} {} | {}", V(cond), edge(yes), edge(no))?
            }
        }
    }
    Ok(())
}

fn write_op(
    out: &mut impl Write,
    op: Op,
    f: &Function,
    callee: &dyn Fn(Call) -> String,
) -> fmt::Result {
    let index =
        |address: VectorAddress| address.indices.map_or("-".to_owned(), |i| V(i).to_string());
    match op {
        Op::Const(n) => write!(out, "const {n}"),
        Op::Literal(lane, bits) => write!(out, "literal {} {bits:#x}", lane_name(lane)),
        Op::Unary(op, x) => write!(out, "{} {}", name_of(&UNARIES, op), V(x)),
        Op::Bin(op, x, y) => write!(out, "{} {} {}", name_of(&BINARIES, op), V(x), V(y)),
        Op::Ternary(op, x, y, z) => {
            write!(
                out,
                "{} {} {} {}",
                name_of(&TERNARIES, op),
                V(x),
                V(y),
                V(z)
            )
        }
        Op::Cast(ty, x) => write!(out, "cast {} {}", TypeName(ty), V(x)),
        Op::Bitcast(ty, x) => write!(out, "bitcast {} {}", TypeName(ty), V(x)),
        Op::Copy(x) => write!(out, "copy {}", V(x)),
        Op::BoundsCheck(i, n) => write!(out, "bounds {} {}", V(i), V(n)),
        Op::TrapIf(cond, trap) => write!(out, "trap_if {} {}", V(cond), trap_name(trap)),
        Op::Trap(trap) => write!(out, "trap {}", trap_name(trap)),
        Op::PtrOffset(p, i, scale) => {
            write!(out, "offset {} {} {}", V(p), V(i), name_of(&SCALES, scale))
        }
        Op::CallScratch { offset, bytes } => write!(out, "call_scratch {offset} {bytes}"),
        Op::Load(ty, p) => write!(out, "load {} {}", TypeName(ty), V(p)),
        Op::Store(p, x) => write!(out, "store {} {}", V(p), V(x)),
        Op::DebugStore(offset, x) => write!(out, "debug_store {offset} {}", V(x)),
        Op::DebugPoint(point) => write!(out, "debug_point {point}"),
        Op::Call(site) => write!(out, "call {}({})", callee(site), values(f.call(site).1)),
        Op::RustCall(function, args) => {
            write!(out, "rust_call {} {}", function.name, values(&args))
        }
        Op::Simd(simd) => match simd {
            Simd::LaneCount(lane) => write!(out, "lanes {}", lane_name(lane)),
            Simd::Iota(lane) => write!(out, "iota {}", lane_name(lane)),
            Simd::Shuffle(x, y, i) => write!(out, "shuffle {} {} {}", V(x), V(y), V(i)),
            Simd::Reduce(op, x) => write!(out, "reduce {} {}", name_of(&BINARIES, op), V(x)),
            Simd::AnyTrue(x) => write!(out, "any {}", V(x)),
            Simd::AllTrue(x) => write!(out, "all {}", V(x)),
            Simd::Compress(x, m) => write!(out, "compress {} {}", V(x), V(m)),
            Simd::Expand(x, m) => write!(out, "expand {} {}", V(x), V(m)),
            Simd::Widen { to, high, value } => write!(
                out,
                "{} {} {}",
                if high { "widenhi" } else { "widenlo" },
                lane_name(to),
                V(value)
            ),
            Simd::Narrow(lane, x, y) => write!(out, "narrow {} {} {}", lane_name(lane), V(x), V(y)),
            Simd::Splat(x) => write!(out, "splat {}", V(x)),
            Simd::ExtractLane(x, i) => write!(out, "extract {} {}", V(x), V(i)),
            Simd::InsertLane(x, i, y) => write!(out, "insert {} {} {}", V(x), V(i), V(y)),
            Simd::MaskedLoad {
                address,
                mask,
                passthrough,
            } => write!(
                out,
                "maskedload {} {} {} {}",
                V(address.base),
                index(address),
                V(mask),
                V(passthrough)
            ),
            Simd::MaskedStore {
                address,
                mask,
                value,
            } => write!(
                out,
                "maskedstore {} {} {} {}",
                V(address.base),
                index(address),
                V(mask),
                V(value)
            ),
            Simd::FirstLanes(lane, n) => write!(out, "firstlanes {} {}", lane_name(lane), V(n)),
            Simd::MaskCount(m) => write!(out, "maskcount {}", V(m)),
            Simd::CompressStore {
                base,
                mask,
                value,
                limit,
            } => write!(
                out,
                "compressstore {} {} {} {}",
                V(base),
                V(mask),
                V(value),
                V(limit)
            ),
        },
    }
}

/// The function and its callees, callees first, each named once: the
/// function itself `main`, its callees `f0`, `f1`, ...
impl Display for Function {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn visit<'a>(f: &'a Arc<Function>, order: &mut Vec<&'a Arc<Function>>) {
            if order.iter().any(|g| Arc::ptr_eq(g, f)) {
                return;
            }
            for callee in f.callees() {
                visit(callee, order);
            }
            order.push(f);
        }
        let mut order = Vec::new();
        for callee in self.callees() {
            visit(callee, &mut order);
        }
        let name = |f: &Function, call| {
            let callee = f.call(call).0;
            let index = order.iter().position(|g| Arc::ptr_eq(g, callee)).unwrap();
            format!("f{index}")
        };
        for (i, f) in order.iter().enumerate() {
            write_function(out, &format!("f{i}"), f, &|call| name(f, call))?;
        }
        write_function(out, "main", self, &|call| name(self, call))
    }
}

impl Function {
    /// The functions this one calls, in the order of their call sites.
    fn callees(&self) -> impl Iterator<Item = &Arc<Function>> + '_ {
        self.blocks()
            .flat_map(|b| self.instructions(b))
            .filter_map(|inst| match *self.op(inst) {
                Op::Call(call) => Some(self.call(call).0),
                _ => None,
            })
    }
}

/// A malformed text, or IR that the builder rejects while it is read.
#[derive(Debug)]
pub enum ParseError {
    Syntax(String),
    /// The builder's error, after the location of the operation.
    Invalid(String, IrError),
}

impl ParseError {
    fn at(self, location: impl Display) -> Self {
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

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(message) => f.write_str(message),
            Self::Invalid(place, error) => write!(f, "{place}invalid IR: {error}"),
        }
    }
}

impl std::error::Error for ParseError {}

type Result<T> = std::result::Result<T, ParseError>;
type Names<'a, T> = HashMap<&'a str, T>;

fn ty(s: &str) -> std::result::Result<Type, String> {
    if s == "ptr" {
        return Ok(Type::Ptr);
    }
    match s.strip_prefix("vec.") {
        Some(lane) => find(&LANES, lane, "lane").map(Type::Vector),
        None => find(&LANES, s, "type").map(Lane::scalar),
    }
}

fn lane(s: &str) -> std::result::Result<Lane, String> {
    find(&LANES, s, "lane")
}

fn integer<T: std::str::FromStr>(s: &str) -> std::result::Result<T, String> {
    s.parse().map_err(|_| format!("expected a number, got {s}"))
}

/// An `i32` in decimal, or its bits in hex (`0x80000000`).
fn word(s: &str) -> std::result::Result<i32, String> {
    let s = s.replace('_', "");
    match s.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16).map(|bits| bits as i32),
        None => s.parse(),
    }
    .map_err(|_| format!("expected i32, got {s}"))
}

fn define<'a, T>(names: &mut Names<'a, T>, name: &'a str, item: T) -> Result<()> {
    if name.is_empty() || names.insert(name, item).is_some() {
        return Err(format!("empty or duplicate name '{name}'").into());
    }
    Ok(())
}

fn lookup<T: Copy>(names: &Names<T>, name: &str) -> Result<T> {
    names
        .get(name)
        .copied()
        .ok_or_else(|| format!("unknown name '{name}'").into())
}

/// Split `name ( items ) rest` into the name, the items and the rest.
fn group<'w, 'a>(words: &'w [&'a str]) -> Result<(&'a str, &'w [&'a str], &'w [&'a str])> {
    let [name, "(", rest @ ..] = words else {
        return Err(format!("expected name(...), got {}", words.join(" ")).into());
    };
    let close = rest.iter().position(|w| *w == ")").ok_or("expected )")?;
    Ok((name, &rest[..close], &rest[close + 1..]))
}

fn typed<'a>(words: &[&'a str]) -> Result<Vec<(&'a str, Type)>> {
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

/// The words of a line, with parentheses as words of their own.
fn split(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for word in line.split_whitespace() {
        let mut rest = word;
        while let Some(at) = rest.find(['(', ')']) {
            if at > 0 {
                out.push(&rest[..at]);
            }
            out.push(&rest[at..at + 1]);
            rest = &rest[at + 1..];
        }
        if !rest.is_empty() {
            out.push(rest);
        }
    }
    out
}

/// Read the functions of `text`; the last one is the result and the others
/// its callees. `rust` names the Rust functions that `rust_call` may call.
pub fn parse(text: &str, rust: &[&'static RustFunction]) -> Result<Function> {
    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .map(|(i, line)| (i + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let mut functions = Names::new();
    let mut last = None;
    let mut rest = &lines[..];
    while let Some(&(line, header)) = rest.first() {
        let len = rest[1..]
            .iter()
            .position(|(_, s)| s.starts_with("fn "))
            .map_or(rest.len(), |n| n + 1);
        let (name, f) = Parser {
            functions: &functions,
            rust,
            values: Names::new(),
            blocks: Names::new(),
        }
        .function(header, &rest[1..len])
        .map_err(|e| e.at(format_args!("line {line}")))?;
        let f = Arc::new(f);
        define(&mut functions, name, Arc::clone(&f))?;
        last = Some(f);
        rest = &rest[len..];
    }
    let last = last.ok_or("expected a function")?;
    drop(functions);
    Ok(Arc::try_unwrap(last).unwrap_or_else(|f| (*f).clone()))
}

struct Parser<'a, 'n> {
    functions: &'n Names<'a, Arc<Function>>,
    rust: &'n [&'static RustFunction],
    values: Names<'a, Value>,
    blocks: Names<'a, Block>,
}

impl<'a> Parser<'a, '_> {
    fn function(
        mut self,
        header: &'a str,
        lines: &[(usize, &'a str)],
    ) -> Result<(&'a str, Function)> {
        let words = split(header);
        let ["fn", rest @ ..] = &words[..] else {
            return Err("expected fn name(params) -> type".into());
        };
        let (name, params, rest) = group(rest)?;
        let ["->", ret] = rest else {
            return Err("expected -> type".into());
        };
        let params = typed(params)?;
        let mut f = Function::new(&params.iter().map(|p| p.1).collect::<Vec<_>>(), ty(ret)?);
        let entry = f.entry();
        self.blocks.insert("entry", entry);
        for (&(name, _), &v) in params.iter().zip(f.params(entry)) {
            define(&mut self.values, name, v)?;
        }
        // Blocks and their parameters exist before any edge names them.
        for &(line, text) in lines {
            let words = split(text);
            if let ["block", rest @ ..] = &words[..] {
                let at = |e: ParseError| e.at(format_args!("line {line}"));
                let (name, params, _) = group(rest).map_err(at)?;
                let b = f.create_block();
                define(&mut self.blocks, name, b).map_err(at)?;
                for (name, t) in typed(params).map_err(at)? {
                    let v = f.append_typed_param(b, t);
                    define(&mut self.values, name, v).map_err(at)?;
                }
            }
        }
        let mut block = entry;
        let mut ended = false;
        let mut slices = Vec::new();
        for &(line, text) in lines {
            let words = split(text);
            let at = |e: ParseError| e.at(format_args!("line {line}"));
            match &words[..] {
                ["block", name, ..] => {
                    block = self.blocks[name];
                    ended = false;
                }
                ["slice", ptr, len, layout] => {
                    let (writable, element) = if let Some(n) = layout.strip_prefix("write") {
                        (true, n)
                    } else if let Some(n) = layout.strip_prefix("read") {
                        (false, n)
                    } else {
                        return Err(at(format!("invalid slice layout {layout}").into()));
                    };
                    slices.push(SliceParam {
                        ptr: self.value(ptr).map_err(at)?,
                        len: self.value(len).map_err(at)?,
                        element: integer(element).map_err(|e| at(e.into()))?,
                        writable,
                    });
                }
                words if ended => {
                    return Err(at(
                        format!("{} after the terminator", words.join(" ")).into()
                    ));
                }
                words => ended = self.line(&mut f, block, words).map_err(at)?,
            }
        }
        if !slices.is_empty() {
            f.set_slice_params(slices)
                .map_err(|e| ParseError::Invalid(String::new(), e))?;
        }
        Ok((name, f))
    }

    fn value(&self, name: &str) -> Result<Value> {
        lookup(&self.values, name)
    }

    fn edge(&self, f: &mut Function, words: &[&'a str]) -> Result<Edge> {
        let (target, args, rest) = group(words)?;
        if !rest.is_empty() {
            return Err(format!("unexpected {}", rest.join(" ")).into());
        }
        let args = args
            .iter()
            .map(|a| self.value(a))
            .collect::<Result<Vec<_>>>()?;
        Ok(f.add_edge(lookup(&self.blocks, target)?, &args))
    }

    /// Append an instruction or set the terminator, which ends the block.
    fn line(&mut self, f: &mut Function, block: Block, words: &[&'a str]) -> Result<bool> {
        let term = match words {
            ["ret", x] => Some(Terminator::Return(self.value(x)?)),
            ["jump", edge @ ..] => Some(Terminator::Jump(self.edge(f, edge)?)),
            ["br", cond, arms @ ..] => {
                let bar = arms
                    .iter()
                    .position(|w| *w == "|")
                    .ok_or("expected | between edges")?;
                Some(Terminator::Branch {
                    cond: self.value(cond)?,
                    yes: self.edge(f, &arms[..bar])?,
                    no: self.edge(f, &arms[bar + 1..])?,
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
        let op = self.op(f, words)?;
        let inst = f
            .append_inst(block, op)
            .map_err(|error| ParseError::Invalid(String::new(), error))?;
        match (dst, f.result(inst)) {
            (Some(dst), Some(value)) => {
                let name = match dst.split_once(':') {
                    Some((name, t)) => {
                        let expected = ty(t)?;
                        if f.value_type(value) != expected {
                            return Err(format!(
                                "{name} is {}, not {t}",
                                TypeName(f.value_type(value))
                            )
                            .into());
                        }
                        name
                    }
                    None => dst,
                };
                define(&mut self.values, name, value)?;
            }
            (None, None) => {}
            _ => return Err("result binding does not match the instruction".into()),
        }
        Ok(false)
    }

    fn op(&self, f: &mut Function, words: &[&'a str]) -> Result<Op> {
        let v = |name: &str| self.value(name);
        let trap = |name: &str| {
            Trap::ALL
                .into_iter()
                .find(|&trap| trap_name(trap) == name)
                .ok_or_else(|| format!("unknown trap {name}"))
        };
        let address = |base: &str, indices: &str| -> Result<VectorAddress> {
            Ok(VectorAddress {
                base: v(base)?,
                indices: if indices == "-" {
                    None
                } else {
                    Some(v(indices)?)
                },
            })
        };
        Ok(match words {
            ["const", n] => Op::Const(word(n)?),
            ["literal", t, bits] => {
                let bits = u128::from_str_radix(bits.trim_start_matches("0x"), 16)
                    .map_err(|_| format!("expected literal bits in hex, got {bits}"))?;
                Op::literal(lane(t)?, bits)
            }
            ["cast", t, x] => Op::Cast(ty(t)?, v(x)?),
            ["bitcast", t, x] => Op::Bitcast(ty(t)?, v(x)?),
            ["copy", x] => Op::Copy(v(x)?),
            ["bounds", i, n] => Op::BoundsCheck(v(i)?, v(n)?),
            ["trap_if", cond, name] => Op::TrapIf(v(cond)?, trap(name)?),
            ["trap", name] => Op::Trap(trap(name)?),
            ["offset", p, i, scale] => {
                Op::PtrOffset(v(p)?, v(i)?, find(&SCALES, scale, "byte stride")?)
            }
            ["call_scratch", offset, bytes] => Op::CallScratch {
                offset: integer(offset)?,
                bytes: integer(bytes)?,
            },
            ["load", t, p] => Op::Load(ty(t)?, v(p)?),
            ["store", p, x] => Op::Store(v(p)?, v(x)?),
            ["debug_store", offset, x] => Op::DebugStore(integer(offset)?, v(x)?),
            ["debug_point", point] => Op::DebugPoint(integer(point)?),
            ["call", rest @ ..] => {
                let (name, args, rest) = group(rest)?;
                if !rest.is_empty() {
                    return Err(format!("unexpected {}", rest.join(" ")).into());
                }
                let callee = self
                    .functions
                    .get(name)
                    .ok_or_else(|| format!("unknown function {name}"))?;
                let args = args.iter().map(|a| v(a)).collect::<Result<Vec<_>>>()?;
                Op::Call(f.add_call(Arc::clone(callee), &args))
            }
            ["rust_call", name, receiver, operation, packet, word] => Op::RustCall(
                self.rust
                    .iter()
                    .find(|function| function.name == *name)
                    .ok_or_else(|| format!("unknown Rust function {name}"))?,
                [v(receiver)?, v(operation)?, v(packet)?, v(word)?],
            ),
            ["lanes", t] => Op::Simd(Simd::LaneCount(lane(t)?)),
            ["iota", t] => Op::Simd(Simd::Iota(lane(t)?)),
            ["shuffle", x, y, i] => Op::Simd(Simd::Shuffle(v(x)?, v(y)?, v(i)?)),
            ["reduce", op, x] => Op::Simd(Simd::Reduce(find(&BINARIES, op, "reduction")?, v(x)?)),
            ["any", x] => Op::Simd(Simd::AnyTrue(v(x)?)),
            ["all", x] => Op::Simd(Simd::AllTrue(v(x)?)),
            ["compress", x, m] => Op::Simd(Simd::Compress(v(x)?, v(m)?)),
            ["expand", x, m] => Op::Simd(Simd::Expand(v(x)?, v(m)?)),
            [part @ ("widenlo" | "widenhi"), t, x] => Op::Simd(Simd::Widen {
                to: lane(t)?,
                high: *part == "widenhi",
                value: v(x)?,
            }),
            ["narrow", t, x, y] => Op::Simd(Simd::Narrow(lane(t)?, v(x)?, v(y)?)),
            ["splat", x] => Op::Simd(Simd::Splat(v(x)?)),
            ["extract", x, i] => Op::Simd(Simd::ExtractLane(v(x)?, v(i)?)),
            ["insert", x, i, y] => Op::Simd(Simd::InsertLane(v(x)?, v(i)?, v(y)?)),
            ["maskedload", p, i, m, x] => Op::Simd(Simd::MaskedLoad {
                address: address(p, i)?,
                mask: v(m)?,
                passthrough: v(x)?,
            }),
            ["maskedstore", p, i, m, x] => Op::Simd(Simd::MaskedStore {
                address: address(p, i)?,
                mask: v(m)?,
                value: v(x)?,
            }),
            ["firstlanes", t, n] => Op::Simd(Simd::FirstLanes(lane(t)?, v(n)?)),
            ["maskcount", m] => Op::Simd(Simd::MaskCount(v(m)?)),
            ["compressstore", p, m, x, limit] => Op::Simd(Simd::CompressStore {
                base: v(p)?,
                mask: v(m)?,
                value: v(x)?,
                limit: v(limit)?,
            }),
            [name, x, y, z] => {
                Op::Ternary(find(&TERNARIES, name, "operation")?, v(x)?, v(y)?, v(z)?)
            }
            [name, x, y] => Op::Bin(find(&BINARIES, name, "operation")?, v(x)?, v(y)?),
            [name, x] => Op::Unary(find(&UNARIES, name, "operation")?, v(x)?),
            _ => return Err(format!("invalid instruction: {}", words.join(" ")).into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printed_text_reads_back_as_the_same_function() {
        // Block parameters are numbered before the results.
        let text = "fn f0(v0:vec.f32) -> vec.f32
  v1:vec.f32 = fma v0 v0 v0
  ret v1
fn main(v0:ptr v1:i32 v2:i32) -> i32
  slice v0 v1 write4
  v4:f64 = literal f64 0x3ff0000000000000
  v5:i32 = cast i32 v4
  debug_point 7
  debug_store 0 v5
  v6:vec.f32 = load vec.f32 v0
  v7:vec.f32 = call f0(v6)
  store v0 v7
  jump b1(v5)
block b1(v3:i32)
  v8:i32 = trap_if v3 Overflow
  ret v3
";
        // `trap_if` has no result: the binding is an error, not a new value.
        let error = parse(text, &[]).unwrap_err().to_string();
        assert!(error.contains("line 15: result binding"), "{error}");
        let text = text.replace("v8:i32 = trap_if", "trap_if");
        let f = parse(&text, &[]).unwrap();
        assert_eq!(f.to_string(), text);
    }

    #[test]
    fn a_named_result_type_must_be_the_operation_type() {
        let error = parse("fn main(x:i32) -> i32\n  y:i64 = add x x\n  ret y", &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("line 2: y is i32, not i64"), "{error}");
    }
}
