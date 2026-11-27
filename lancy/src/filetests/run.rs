//! `@run args -> result [| before | after]...`: typed inputs, one expected
//! result or trap, and one byte buffer pair per pointer parameter.
use super::Result;
use crate::hir::ssa::{InterpretError, buffer_address};
use crate::hir::{Function, Lane, Type};
use crate::support::numeric::Bits;
use crate::{CallError, Kernel, Trap};

pub(crate) struct Run {
    line: usize,
    args: Vec<String>,
    expected: std::result::Result<String, CallError>,
    buffers: Vec<(Vec<u8>, Vec<u8>)>,
}

fn bytes(text: &str) -> Result<Vec<u8>> {
    text.split_whitespace()
        .map(|s| {
            if s.len() != 2 {
                return Err(format!("expected a two-digit hex byte, got {s}"));
            }
            u8::from_str_radix(s, 16).map_err(|_| format!("invalid byte {s}"))
        })
        .collect()
}

/// Split on whitespace outside `[...]`, so `vec.i32[1 2]` is one literal.
fn literals(text: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    for c in text.chars() {
        match c {
            '[' => depth += 1,
            ']' if depth == 0 => return Err(format!("unbalanced ] in {text}")),
            ']' => depth -= 1,
            c if c.is_whitespace() && depth == 0 => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
                continue;
            }
            _ => {}
        }
        current.push(c);
    }
    if depth != 0 {
        return Err(format!("unbalanced [ in {text}"));
    }
    if !current.is_empty() {
        out.push(current);
    }
    Ok(out)
}

const LANES: [(&str, Lane); 16] = [
    ("isize", Lane::Isize),
    ("usize", Lane::Usize),
    ("i128", Lane::I128),
    ("u128", Lane::U128),
    ("bf16", Lane::BF16),
    ("f16", Lane::F16),
    ("f32", Lane::F32),
    ("f64", Lane::F64),
    ("i16", Lane::I16),
    ("u16", Lane::U16),
    ("i32", Lane::I32),
    ("u32", Lane::U32),
    ("i64", Lane::I64),
    ("u64", Lane::U64),
    ("i8", Lane::I8),
    ("u8", Lane::U8),
];

fn lane_name(lane: Lane) -> &'static str {
    LANES.iter().find(|(_, l)| *l == lane).unwrap().0
}

/// One lane value: hexadecimal is the bit pattern, decimal the number.
fn lane_bits(lane: Lane, text: &str) -> Result<u128> {
    let size = lane.bytes();
    let mask = u128::MAX >> (128 - 8 * size);
    let digits = text.replace('_', "");
    let fits = |bits: u128| (bits & !mask == 0).then_some(bits);
    let bits = if let Some(hex) = digits.strip_prefix("0x") {
        u128::from_str_radix(hex, 16).ok().and_then(fits)
    } else {
        match lane {
            Lane::F32 => digits.parse::<f32>().ok().map(|x| x.to_bits().into()),
            Lane::F64 => digits.parse::<f64>().ok().map(|x| x.to_bits().into()),
            Lane::F16 | Lane::BF16 => {
                return Err(format!("write {} as hexadecimal bits", lane_name(lane)));
            }
            lane if lane.signed() => digits.parse::<i128>().ok().and_then(|x| {
                let shift = 128 - 8 * size as u32;
                (((x << shift) >> shift) == x).then_some(x as u128 & mask)
            }),
            _ => digits.parse::<u128>().ok().and_then(fits),
        }
    };
    bits.ok_or_else(|| format!("{text} is not a {} value", lane_name(lane)))
}

/// A literal of type `ty`: `7`, `-3i64`, `1.5f32`, `0x3c00_f16`, or
/// `vec.i32[1 2]`, whose lanes repeat to fill the vector. An unsuffixed
/// scalar takes the expected type.
fn literal(text: &str, ty: Type, width: usize) -> Result<Bits> {
    let mut bits = [0; 64];
    let mismatch = |written: &str| format!("{text}: expected {ty:?}, written as {written}");
    if let Some(vector) = text.strip_prefix("vec.") {
        let (name, lanes) = vector
            .strip_suffix(']')
            .and_then(|v| v.split_once('['))
            .ok_or_else(|| format!("expected vec.<lane>[...], got {text}"))?;
        let &(_, lane) = LANES
            .iter()
            .find(|(known, _)| *known == name)
            .ok_or_else(|| format!("unknown lane type {name}"))?;
        if ty != Type::Vector(lane) {
            return Err(mismatch(&format!("vec.{name}")));
        }
        let lanes: Vec<_> = lanes
            .split_whitespace()
            .map(|x| lane_bits(lane, x))
            .collect::<Result<_>>()?;
        let count = width / lane.bytes();
        if lanes.is_empty() || !count.is_multiple_of(lanes.len()) {
            return Err(format!(
                "{text}: {} lanes do not repeat to fill {count}",
                lanes.len()
            ));
        }
        let size = lane.bytes();
        for (i, chunk) in bits[..width].chunks_exact_mut(size).enumerate() {
            chunk.copy_from_slice(&lanes[i % lanes.len()].to_le_bytes()[..size]);
        }
        return Ok(bits);
    }
    let hex = text.trim_start_matches('-').starts_with("0x");
    let (value, written) = LANES
        .iter()
        .find_map(|&(name, lane)| {
            let value = text.strip_suffix(name)?;
            // A hexadecimal digit run could end in `f16`; require `_` there.
            let value = if hex { value.strip_suffix('_')? } else { value };
            Some((value, Some(lane)))
        })
        .unwrap_or((text, None));
    let lane = match (ty, written) {
        (Type::Vector(_) | Type::Ptr, _) => return Err(mismatch("a scalar")),
        (ty, Some(lane)) if lane.scalar() != ty => return Err(mismatch(lane_name(lane))),
        (ty, _) => ty.lane().unwrap(),
    };
    let negative = value.strip_prefix('-').filter(|v| v.starts_with("0x"));
    let lane_value = match negative {
        // `-0x1` is the negated bit pattern, as for i32 fixtures.
        Some(hex) => lane_bits(lane, hex)?.wrapping_neg() & (u128::MAX >> (128 - 8 * lane.bytes())),
        None => lane_bits(lane, value)?,
    };
    bits[..lane.bytes()].copy_from_slice(&lane_value.to_le_bytes()[..lane.bytes()]);
    Ok(bits)
}

fn show(ty: Type, bits: &Bits, width: usize) -> String {
    match ty {
        Type::Vector(lane) => {
            let size = lane.bytes();
            let lanes: Vec<_> = bits[..width]
                .chunks_exact(size)
                .map(|chunk| format!("{:#x}", crate::support::numeric::read(chunk)))
                .collect();
            format!("vec.{}[{}]", lane_name(lane), lanes.join(" "))
        }
        ty => format!(
            "{:#x}_{}",
            crate::support::numeric::read(&bits[..ty.bytes(width)]),
            lane_name(ty.lane().unwrap())
        ),
    }
}

/// The run's inputs and expectation, typed by `f`'s signature.
struct Typed {
    args: Vec<Bits>,
    expected: std::result::Result<Bits, CallError>,
    ret: Type,
    width: usize,
}

impl Typed {
    fn show(&self, result: &std::result::Result<Bits, impl std::fmt::Display>) -> String {
        match result {
            Ok(bits) => show(self.ret, bits, self.width),
            Err(error) => format!("!{{{error}}}"),
        }
    }

    fn same(&self, actual: &Bits, expected: &Bits) -> bool {
        let n = self.ret.bytes(self.width);
        actual[..n] == expected[..n]
    }
}

impl Run {
    pub(crate) fn parse(line: usize, text: &str) -> Result<Self> {
        let mut parts = text.split('|');
        let call = parts.next().unwrap();
        let rest: Vec<_> = parts.collect();
        if !rest.len().is_multiple_of(2) {
            return Err("expected args -> result [| bytes before | bytes after]...".into());
        }
        let buffers = rest
            .chunks_exact(2)
            .map(|pair| Ok((bytes(pair[0])?, bytes(pair[1])?)))
            .collect::<Result<_>>()?;
        let (args, result) = call
            .split_once("->")
            .ok_or("expected @run args -> result")?;
        let expected = match result.trim() {
            "!bounds" => Err(CallError::Trap(Trap::Bounds)),
            "!zero" => Err(CallError::Trap(Trap::DivisionByZero)),
            "!overflow" => Err(CallError::Trap(Trap::DivisionOverflow)),
            "!arithmetic_overflow" => Err(CallError::Trap(Trap::Overflow)),
            "!rust_panic" => Err(CallError::Trap(Trap::RustPanic)),
            "!abi" => Err(CallError::Trap(Trap::Abi)),
            trap if trap.starts_with('!') => return Err(format!("unknown trap {trap}")),
            value => match literals(value)?.as_slice() {
                [value] => Ok(value.clone()),
                _ => return Err(format!("expected one result, got {value}")),
            },
        };
        Ok(Self {
            line,
            args: literals(args)?,
            expected,
            buffers,
        })
    }

    fn typed(&self, f: &Function) -> Result<Typed> {
        let width = crate::test_target::width();
        let at = |e: String| format!("line {}: {e}", self.line);
        let params = f.param_types();
        let pointers = params.iter().filter(|&&ty| ty == Type::Ptr).count();
        if pointers != self.buffers.len() || params.len() - pointers != self.args.len() {
            return Err(at(format!(
                "signature takes {} values and {pointers} buffers, the run gives {} and {}",
                params.len() - pointers,
                self.args.len(),
                self.buffers.len()
            )));
        }
        let (mut values, mut buffers) = (self.args.iter(), 0);
        let args = params
            .iter()
            .map(|&ty| {
                if ty == Type::Ptr {
                    buffers += 1;
                    let mut bits = [0; 64];
                    bits[..8].copy_from_slice(&buffer_address(buffers - 1).to_le_bytes());
                    Ok(bits)
                } else {
                    literal(values.next().unwrap(), ty, width)
                }
            })
            .collect::<Result<_>>()
            .map_err(at)?;
        let expected = match &self.expected {
            Ok(text) => Ok(literal(text, f.return_type(), width).map_err(at)?),
            Err(trap) => Err(trap.clone()),
        };
        Ok(Typed {
            args,
            expected,
            ret: f.return_type(),
            width,
        })
    }

    fn check_buffers(&self, stage: &str, buffers: &[Vec<u8>]) -> Result<()> {
        for (i, (actual, (_, expected))) in buffers.iter().zip(&self.buffers).enumerate() {
            if actual != expected {
                return Err(format!(
                    "line {}, {stage} buffer {i}: expected {expected:02x?}, got {actual:02x?}",
                    self.line
                ));
            }
        }
        Ok(())
    }

    fn initial(&self) -> Vec<Vec<u8>> {
        self.buffers
            .iter()
            .map(|(before, _)| before.clone())
            .collect()
    }

    pub(crate) fn interpret(&self, stage: &str, f: &Function) -> Result<()> {
        let typed = self.typed(f)?;
        let mut buffers = self.initial();
        let actual = f.interpret_buffers(&typed.args, &mut buffers);
        self.check_buffers(stage, &buffers)?;
        let matches = match (&actual, &typed.expected) {
            (Ok(actual), Ok(expected)) => typed.same(actual, expected),
            (Err(InterpretError::Trap(actual)), Err(expected)) => actual == expected,
            _ => false,
        };
        if !matches {
            return Err(format!(
                "line {}, {stage} {:?}: expected {}, got {}",
                self.line,
                self.args,
                typed.show(&typed.expected),
                typed.show(&actual)
            ));
        }
        Ok(())
    }

    /// # Safety
    /// The caller must have successfully interpreted this run against `f`,
    /// the source of `kernel`, including its checked buffer accesses.
    pub(crate) unsafe fn native(&self, f: &Function, kernel: &Kernel) -> Result<()> {
        let typed = self.typed(f)?;
        let mut buffers = self.initial();
        let mut words = Vec::new();
        for (bits, &ty) in typed.args.iter().zip(f.param_types()) {
            if ty == Type::Ptr {
                let buffer = (u64::from_le_bytes(bits[..8].try_into().unwrap()) >> 32) as usize;
                words.push(buffers[buffer].as_mut_ptr() as u64);
            } else {
                let n = ty.bytes(typed.width).div_ceil(8);
                words.extend(
                    bits[..n * 8]
                        .chunks_exact(8)
                        .map(|w| u64::from_le_bytes(w.try_into().unwrap())),
                );
            }
        }
        let mut result = vec![0; typed.ret.bytes(typed.width).div_ceil(8)];
        // SAFETY: Case::execute first interprets this same verified function,
        // inputs and initial bytes with checked accesses. Every executed access
        // is within these live buffers, including every active lane of a masked
        // access; CallScratch addresses lie in the kernel's own frame.
        let actual = unsafe { kernel.call_packed_unchecked(&words, &mut result) }.map(|()| {
            let mut bits = [0; 64];
            for (chunk, word) in bits.chunks_exact_mut(8).zip(&result) {
                chunk.copy_from_slice(&word.to_le_bytes());
            }
            bits
        });
        self.check_buffers("run", &buffers)?;
        let matches = match (&actual, &typed.expected) {
            (Ok(actual), Ok(expected)) => typed.same(actual, expected),
            (Err(actual), Err(expected)) => actual == expected,
            _ => false,
        };
        if !matches {
            return Err(format!(
                "line {}, run {:?}: expected {}, got {}",
                self.line,
                self.args,
                typed.show(&typed.expected),
                typed.show(&actual)
            ));
        }
        Ok(())
    }
}
