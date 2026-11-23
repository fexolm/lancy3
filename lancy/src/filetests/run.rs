use super::{Result, integer};
use crate::{CallError, ScalarKernel, hir::Function};

pub(crate) struct Run {
    line: usize,
    args: Vec<i32>,
    expected: std::result::Result<i32, CallError>,
    buffer: Option<(Vec<u8>, Vec<u8>)>,
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

impl Run {
    pub(crate) fn parse(line: usize, text: &str) -> Result<Self> {
        let parts: Vec<_> = text.split('|').collect();
        let (call, buffer) = match parts.as_slice() {
            [call] => (*call, None),
            [call, before, after] => (*call, Some((bytes(before)?, bytes(after)?))),
            _ => return Err("expected args -> result [| bytes before | bytes after]".into()),
        };
        let (args, result) = call
            .split_once("->")
            .ok_or("expected @run args -> result")?;
        let args = args
            .split_whitespace()
            .map(integer)
            .collect::<Result<Vec<_>>>()?;
        let expected = match result.trim() {
            "!bounds" => Err(CallError::OutOfBounds),
            "!zero" => Err(CallError::DivisionByZero),
            "!overflow" => Err(CallError::DivisionOverflow),
            "!arithmetic_overflow" => Err(CallError::IntegerOverflow),
            "!rust_panic" => Err(CallError::RustPanic),
            n => Ok(integer(n)?),
        };
        Ok(Self {
            line,
            args,
            expected,
            buffer,
        })
    }

    fn check_bytes(&self, stage: &str, actual: &[u8], expected: &[u8]) -> Result<()> {
        if actual != expected {
            return Err(format!(
                "line {}, {stage} buffer: expected {expected:02x?}, got {actual:02x?}",
                self.line
            ));
        }
        Ok(())
    }

    pub(crate) fn interpret(&self, stage: &str, f: &Function) -> Result<()> {
        let actual = if let Some((initial, expected)) = &self.buffer {
            let mut bytes = initial.clone();
            let result = f.interpret_buffer(&self.args, &mut bytes);
            self.check_bytes(stage, &bytes, expected)?;
            result
        } else {
            f.interpret(&self.args)
        };
        let expected = self.expected.map_err(|e| {
            match e {
                CallError::DivisionByZero => "integer division by zero",
                CallError::DivisionOverflow => "signed division overflow",
                CallError::IntegerOverflow => "kernel integer arithmetic overflow",
                CallError::RustPanic => "kernel Rust import panicked",
                CallError::OutOfBounds => "kernel slice index out of bounds",
                _ => unreachable!(),
            }
            .to_string()
        });
        if actual != expected {
            return Err(format!(
                "line {}, {stage} {:?}: expected {expected:?}, got {actual:?}",
                self.line, self.args
            ));
        }
        Ok(())
    }

    /// # Safety
    /// The caller must have successfully interpreted this run against the
    /// source of this exact kernel, including its checked buffer accesses.
    pub(crate) unsafe fn native(&self, kernel: &ScalarKernel) -> Result<()> {
        let actual = if let Some((initial, expected)) = &self.buffer {
            let mut bytes = initial.clone();
            let args: Vec<_> = std::iter::once(bytes.as_mut_ptr() as u64)
                .chain(self.args.iter().map(|&x| u64::from(x as u32)))
                .collect();
            // SAFETY: Case::execute first interprets this same verified function,
            // input and initial bytes with checked offsets. Every executed access
            // is within this live allocation, including every active lane of a masked access.
            let result = unsafe { kernel.call_raw(&args) }.map(|v| v as i32);
            self.check_bytes("run", &bytes, expected)?;
            result
        } else {
            let args: Vec<_> = self.args.iter().map(|&x| u64::from(x as u32)).collect();
            // SAFETY: the interpreter accepted this function only with an i32
            // signature, so no argument is a pointer. Its only native addresses
            // come from CallScratch inside the kernel's own frame.
            unsafe { kernel.call_raw(&args) }.map(|v| v as i32)
        };
        if actual != self.expected {
            return Err(format!(
                "line {}, run {:?}: expected {:?}, got {actual:?}",
                self.line, self.args, self.expected
            ));
        }
        Ok(())
    }
}
