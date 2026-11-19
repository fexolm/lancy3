//! Small, test-only spelling of Lancy SSA. Build the same Function as the frontend.
use crate::hir::{Binary, Block, Edge, Function, Op, Scale, Terminator, Type, Value};
use std::collections::HashMap;
use std::sync::Arc;

pub(super) type Lines<'a> = Vec<(usize, &'a str)>;
type Names<'a, T> = HashMap<&'a str, T>;
type Result<T> = std::result::Result<T, String>;

fn ty(s: &str) -> Result<Type> {
    match s {
        "i32" => Ok(Type::I32),
        "ptr" => Ok(Type::Ptr),
        _ => Err(format!("unknown type {s}")),
    }
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
) -> Result<bool> {
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
        ["const", n] => Op::Const(super::integer(n)?),
        ["copy", x] => Op::Copy(v(x)?),
        ["bounds", i, n] => Op::BoundsCheck(v(i)?, v(n)?),
        ["offset", p, i, stride] => Op::PtrOffset(
            v(p)?,
            v(i)?,
            match *stride {
                "1" => Scale::One,
                "2" => Scale::Two,
                "4" => Scale::Four,
                "8" => Scale::Eight,
                _ => return Err(format!("invalid byte stride {stride}")),
            },
        ),
        ["load", t, p] => Op::Load(ty(t)?, v(p)?),
        ["store", p, x] => Op::Store(v(p)?, v(x)?),
        ["call", name, args @ ..] => {
            let callee = functions
                .get(name)
                .ok_or_else(|| format!("unknown callee {name}"))?;
            let args = args.iter().map(|arg| v(arg)).collect::<Result<Vec<_>>>()?;
            Op::Call(f.add_call(Arc::clone(callee), &args))
        }
        [name, a, b] => {
            use Binary::*;
            let op = match *name {
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
                _ => return Err(format!("unknown instruction {name}")),
            };
            Op::Bin(op, v(a)?, v(b)?)
        }
        _ => return Err(format!("invalid instruction: {}", words.join(" "))),
    };
    match (dst, f.append(block, op)) {
        (Some(name), Some(value)) => define(values, name, value)?,
        (None, None) => {}
        _ => return Err("result binding does not match the instruction".into()),
    }
    Ok(false)
}

fn body<'a>(
    signature: &[&'a str],
    lines: &[(usize, &'a str)],
    functions: &Names<Arc<Function>>,
) -> Result<Function> {
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
        } else {
            if ended {
                return Err(format!("line {line}: instruction after terminator"));
            }
            ended = instruction(&mut f, block, &words, &mut values, &blocks, functions)
                .map_err(|e| format!("line {line}: {e}"))?;
        }
    }
    Ok(f)
}

pub(super) fn parse(mut lines: &[(usize, &str)]) -> Result<Arc<Function>> {
    let mut functions = Names::new();
    while let Some((&(line, header), rest)) = lines.split_first() {
        let words: Vec<_> = header.split_whitespace().collect();
        let ["fn", name, signature @ ..] = words.as_slice() else {
            return Err(format!("line {line}: expected function header"));
        };
        let len = rest
            .iter()
            .position(|(_, s)| s.split_whitespace().next() == Some("fn"))
            .unwrap_or(rest.len());
        let f = body(signature, &rest[..len], &functions)
            .map_err(|e| format!("fn {name} (line {line}): {e}"))?;
        define(&mut functions, name, Arc::new(f))?;
        lines = &rest[len..];
    }
    functions.remove("main").ok_or("missing fn main".into())
}
