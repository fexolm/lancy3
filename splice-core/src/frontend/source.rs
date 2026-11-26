use std::fmt::Write;

#[derive(Default)]
pub struct Source {
    pub functions: Vec<String>,
    pub transfer: Option<Transfer>,
    entry_name: String,
    /// The current unrolled loop has a `continue`, printed as a break out of
    /// its iteration block, or a `break`, printed as a break out of the loop.
    labels: UnrolledLabels,
}

/// Which labels the printed iterations of an unrolled loop need.
#[derive(Clone, Copy, Default)]
pub struct UnrolledLabels {
    iteration: bool,
    loop_: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    Return,
    Break,
    Continue,
}

pub fn block(body: &str) -> String {
    let mut out = String::from("{\n");
    for line in body.lines() {
        writeln!(out, "    {line}").unwrap();
    }
    out.push('}');
    out
}

impl Source {
    /// Print `continue` in an unrolled loop as a break out of its iteration.
    pub fn continue_unrolled(&mut self) -> String {
        self.labels.iteration = true;
        "break '__splice_iteration".to_owned()
    }

    /// Print `break` in an unrolled loop as a break out of the loop.
    pub fn break_unrolled(&mut self) -> String {
        self.labels.loop_ = true;
        "break '__splice_unrolled".to_owned()
    }

    /// Start printing an unrolled loop; returns the enclosing loop's labels.
    pub fn enter_unrolled(&mut self) -> UnrolledLabels {
        std::mem::take(&mut self.labels)
    }

    /// The label prefix of the iteration just printed.
    pub fn iteration_label(&mut self) -> &'static str {
        if std::mem::take(&mut self.labels.iteration) {
            "'__splice_iteration: "
        } else {
            ""
        }
    }

    /// Finish the unrolled loop `body`, labeled if it breaks.
    pub fn exit_unrolled(&mut self, enclosing: UnrolledLabels, body: &str) -> String {
        let labeled = std::mem::replace(&mut self.labels, enclosing).loop_;
        if labeled {
            format!("'__splice_unrolled: {}", block(body))
        } else {
            block(body)
        }
    }

    pub fn function(&mut self, name: &str) -> (usize, String) {
        let id = self.functions.len();
        self.functions.push(String::new());
        let name = if id == 0 {
            self.entry_name = name.trim_start_matches("r#").to_owned();
            name.to_owned()
        } else {
            let mut name = format!("{}_{id}", name.trim_start_matches("r#"));
            if name == self.entry_name {
                name.push('_');
            }
            name
        };
        (id, name)
    }
}

pub trait Argument {
    fn source(self) -> String;
}

impl Argument for String {
    fn source(self) -> String {
        self
    }
}

impl<T: std::fmt::Debug> Argument for super::scalar::Host<T> {
    fn source(self) -> String {
        literal(self.0)
    }
}

/// The arms of a `match` on a known value that its runtime guards leave
/// open, in order, as an `if` chain: each `(guard, body)` with a runtime
/// guard, then the arm that is selected when they all fail (no guard).
pub fn guarded_arms(arms: &[(Option<String>, String)]) -> String {
    let mut text = String::new();
    for (guard, body) in arms {
        match guard {
            Some(guard) => text.push_str(&format!("if {guard} {} else ", block(body))),
            None => {
                text.push_str(&if text.is_empty() {
                    body.clone()
                } else {
                    block(body)
                });
                break;
            }
        }
    }
    text
}

pub fn literal<T: std::fmt::Debug>(value: T) -> String {
    let repr = format!("{value:?}");
    match repr.as_str() {
        "NaN" => format!(
            "{}::NAN",
            std::any::type_name::<T>().trim_start_matches('&')
        ),
        "inf" => format!(
            "{}::INFINITY",
            std::any::type_name::<T>().trim_start_matches('&')
        ),
        "-inf" => format!(
            "{}::NEG_INFINITY",
            std::any::type_name::<T>().trim_start_matches('&')
        ),
        _ => repr,
    }
}

#[doc(hidden)]
pub struct Literal<'a, T>(pub &'a T);

/// Report opaque captures when a caller actually requests source output.
#[doc(hidden)]
pub trait TryLiteral {
    fn try_literal(self) -> crate::Result<String>;
}

impl<T: std::fmt::Debug> TryLiteral for &Literal<'_, T> {
    fn try_literal(self) -> crate::Result<String> {
        Ok(literal(self.0))
    }
}

impl<T> TryLiteral for &&Literal<'_, T> {
    fn try_literal(self) -> crate::Result<String> {
        Err(crate::BuildError::staging(
            "the specialized Rust value has no printable representation",
        ))
    }
}

/// Rust source that panics like the kernel trap for `fault`.
pub fn fault_panic(fault: super::scalar::HostFault) -> String {
    format!("panic!({:?})", fault.trap().to_string())
}

pub fn function(
    name: &str,
    params: &[String],
    result: &str,
    aliases: &[(String, String)],
    body: &str,
) -> String {
    let mut out = format!("fn {name}({}) -> {result} {{\n", params.join(", "));
    for alias in aliases {
        if body
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .any(|part| part == alias.0)
        {
            writeln!(out, "    {}", alias.1).unwrap();
        }
    }
    for line in body.lines() {
        writeln!(out, "    {line}").unwrap();
    }
    out.push_str("}\n");
    out
}
