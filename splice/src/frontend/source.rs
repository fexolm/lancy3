use std::fmt::Write;

#[derive(Default)]
pub struct Source {
    pub functions: Vec<String>,
    pub transfer: Option<Transfer>,
    entry_name: String,
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
        Err("the specialized Rust value has no printable representation".into())
    }
}

pub fn division_fault(divisor: i32) -> String {
    if divisor == 0 {
        r#"panic!("kernel integer division by zero")"#.to_owned()
    } else {
        r#"panic!("kernel signed division overflow")"#.to_owned()
    }
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
