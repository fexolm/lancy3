//! Inspect a specialized kernel using `my_kernel::build::<splice_text::Text>(...)`.
//! The listing follows Splice's scalar semantics (including wrapping arithmetic
//! and comparison masks). Labels and mutable registers make control flow explicit.
//!
//! ```
//! use splice::kernel;
//! use splice_text::Text;
//!
//! #[kernel]
//! fn adjust(#[comptime] double: bool, value: i32) -> i32 {
//!     if double { value.wrapping_mul(2) } else { value.wrapping_add(5) }
//! }
//!
//! let text: String = adjust::build::<Text>(true)?;
//! assert!(text.starts_with("fn adjust(value: i32)"));
//! assert!(text.contains("value * "));
//! assert!(!text.contains(" + "));
//! # Ok::<(), String>(())
//! ```
#![forbid(unsafe_code)]

use splice::hir::{Binary, Buffer, Instr, Reg};
use splice::{Codegen, Function};
use std::fmt::{self, Write};
use std::sync::Arc;

/// A backend whose output is a deterministic listing of the specialized call graph.
pub struct Text;

impl Codegen for Text {
    type Output = String;

    fn build_with_options(
        arity: usize,
        kinds: &[splice::BufferKind],
        body: splice::frontend::Body<'_>,
        options: splice::BuildOptions,
    ) -> splice::Result<Self::Output> {
        Self::compile(&splice::frontend::materialize_with_options(
            arity, kinds, body, options,
        )?)
    }

    fn compile(root: &Function) -> splice::Result<String> {
        let _span = tracing::debug_span!(target: "splice::compile", "render_text_ir").entered();
        let mut output = String::new();
        render(root, &mut output).map_err(|error| error.to_string())?;
        Ok(output)
    }
}

fn symbol(function: &Function, id: usize) -> String {
    match (function.source().map(|s| s.name), id) {
        (Some(name), 0) => name.to_owned(),
        (Some(name), id) => format!("{name}.{id}"),
        (None, 0) => "entry".to_owned(),
        (None, id) => format!("function.{id}"),
    }
}

/// '%' keeps anonymous temporaries distinct from all Rust parameter names.
struct Operand(Option<&'static str>, char, u32);

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(name) => f.write_str(name),
            None => write!(f, "%{}{}", self.1, self.2),
        }
    }
}

fn render(root: &Function, out: &mut String) -> fmt::Result {
    let (bodies, ids) = root.reachable_functions();
    for (id, f) in bodies.into_iter().enumerate() {
        let reg = |Reg(i)| {
            Operand(
                f.source().and_then(|s| s.scalars.get(i as usize)).copied(),
                'r',
                i,
            )
        };
        let buf = |Buffer(i)| {
            Operand(
                f.source().and_then(|s| s.buffers.get(i as usize)).copied(),
                'b',
                i,
            )
        };
        if id != 0 {
            writeln!(out)?;
        }
        write!(out, "fn {}(", symbol(f, id))?;
        for i in 0..f.arity() {
            if i != 0 {
                out.push_str(", ");
            }
            write!(out, "{}: i32", reg(Reg(i as u32)))?;
        }
        for (i, kind) in f.buffer_kinds().iter().enumerate() {
            if i != 0 || f.arity() != 0 {
                out.push_str(", ");
            }
            let access = if kind.writable() { "&mut " } else { "&" };
            write!(
                out,
                "{}: {access}[{}]",
                buf(Buffer(i as u32)),
                kind.element().name()
            )?;
        }
        writeln!(out, ") -> i32 {{")?;
        for instruction in f.instructions() {
            match instruction {
                Instr::Debug(_) => {}
                Instr::Simd { dst, inst, args } => {
                    use splice::hir::SimdOp::*;
                    write!(out, "    ")?;
                    if let Some(d) = dst {
                        write!(out, "{} = ", reg(*d))?;
                    }
                    let (op, buffer) = match inst.op {
                        Load(b) => ("load", Some(b)),
                        Store(b) => ("store", Some(b)),
                        LoadSelect(b) => ("load_select", Some(b)),
                        StoreSelect(b) => ("store_select", Some(b)),
                        Gather(b) => ("gather", Some(b)),
                        Scatter(b) => ("scatter", Some(b)),
                        GatherSelect(b) => ("gather_select", Some(b)),
                        ScatterSelect(b) => ("scatter_select", Some(b)),
                        Splat(_) | SplatScalar => ("splat", None),
                        Lanes => ("lanes", None),
                        FirstN => ("first_n", None),
                        Iota => ("iota", None),
                        Shuffle(order) => (
                            match order {
                                splice::simd::Shuffle::InterleaveLow => "interleave_low",
                                splice::simd::Shuffle::InterleaveHigh => "interleave_high",
                                splice::simd::Shuffle::DeinterleaveEven => "deinterleave_even",
                                splice::simd::Shuffle::DeinterleaveOdd => "deinterleave_odd",
                            },
                            None,
                        ),
                        Compress => ("compress", None),
                        Expand => ("expand", None),
                        Swizzle => ("swizzle", None),
                        AlignmentPrefix(b) => ("alignment_prefix", Some(b)),
                        PrefixSum => ("prefix_sum", None),
                        Shr(_) => ("shr", None),
                        RotateLeft(_) => ("rotate_left", None),
                        Count => ("count", None),
                        CompressStore(b) => ("compress_store", Some(b)),
                        ExtractWrapping => ("extract_wrapping", None),
                        ReplaceWrapping => ("replace_wrapping", None),
                        Last => ("last", None),
                        Add => ("add", None),
                        Sub => ("sub", None),
                        Mul => ("mul", None),
                        Div => ("div", None),
                        Min => ("min", None),
                        Max => ("max", None),
                        And => ("and", None),
                        Or => ("or", None),
                        Xor => ("xor", None),
                        Not => ("not", None),
                        Eq => ("eq", None),
                        Ne => ("ne", None),
                        Lt => ("lt", None),
                        Le => ("le", None),
                        Gt => ("gt", None),
                        Ge => ("ge", None),
                        MaskAnd => ("mask_and", None),
                        MaskOr => ("mask_or", None),
                        MaskXor => ("mask_xor", None),
                        MaskNot => ("mask_not", None),
                        Any => ("any", None),
                        All => ("all", None),
                        Select => ("select", None),
                    };
                    write!(out, "simd.{op}.{}", inst.element.name())?;
                    if let Some(b) = buffer {
                        write!(out, " {}", buf(b))?;
                    }
                    if let Shr(shift) | RotateLeft(shift) = inst.op {
                        write!(out, " {shift}")?;
                    }
                    if let Splat(bits) = inst.op {
                        write!(out, " {bits:#x}")?;
                    }
                    for &r in args {
                        write!(out, " {}", reg(r))?;
                    }
                    writeln!(out)?;
                }
                Instr::Const(dst, n) => writeln!(out, "    {} = {n}", reg(*dst))?,
                Instr::Copy(dst, src) => writeln!(out, "    {} = {}", reg(*dst), reg(*src))?,
                Instr::Bin(dst, op, a, b) => {
                    writeln!(
                        out,
                        "    {} = {}",
                        reg(*dst),
                        BinaryExpr(*op, reg(*a), reg(*b))
                    )?;
                }
                Instr::BufferLen { dst, buffer } => {
                    writeln!(out, "    {} = {}.len()", reg(*dst), buf(*buffer))?
                }
                Instr::Load { dst, buffer, index } => {
                    writeln!(out, "    {} = {}[{}]", reg(*dst), buf(*buffer), reg(*index))?
                }
                Instr::Store {
                    buffer,
                    index,
                    value,
                } => writeln!(
                    out,
                    "    {}[{}] = {}",
                    buf(*buffer),
                    reg(*index),
                    reg(*value)
                )?,
                Instr::Label(label) => writeln!(out, "L{}:", label.0)?,
                Instr::Jump(label) => writeln!(out, "    goto L{}", label.0)?,
                Instr::Branch { cond, yes, no } => {
                    writeln!(out, "    if {} goto L{} else L{}", reg(*cond), yes.0, no.0)?
                }
                Instr::Return(value) => writeln!(out, "    return {}", reg(*value))?,
                Instr::Call {
                    dst,
                    function,
                    args,
                    buffers,
                } => {
                    let callee = ids[&Arc::as_ptr(function)];
                    write!(out, "    {} = {}(", reg(*dst), symbol(function, callee))?;
                    for (i, arg) in args
                        .iter()
                        .copied()
                        .map(reg)
                        .chain(buffers.iter().copied().map(buf))
                        .enumerate()
                    {
                        if i != 0 {
                            out.push_str(", ");
                        }
                        write!(out, "{arg}")?;
                    }
                    writeln!(out, ")")?;
                }
            }
        }
        writeln!(out, "}}")?;
    }
    Ok(())
}

struct BinaryExpr(Binary, Operand, Operand);

impl fmt::Display for BinaryExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(op, a, b) = self;
        let infix = match op {
            Binary::Add => "+",
            Binary::Sub => "-",
            Binary::Mul => "*",
            Binary::Div => "/",
            Binary::Rem => "%",
            Binary::And => "&",
            Binary::Or => "|",
            Binary::Xor => "^",
            Binary::Eq => "==",
            Binary::Ne => "!=",
            Binary::Lt => "<",
            Binary::Le => "<=",
            Binary::Gt => ">",
            Binary::Ge => ">=",
            Binary::Shl => "<<",
            Binary::Shr => ">>",
            Binary::Min => return write!(f, "min({a}, {b})"),
            Binary::Max => return write!(f, "max({a}, {b})"),
        };
        write!(f, "{a} {infix} {b}")
    }
}
