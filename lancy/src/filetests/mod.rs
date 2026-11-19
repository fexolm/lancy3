//! SSA fixture syntax and execution shared by target-specific test runners.
pub(crate) mod input;
mod run;

pub(crate) use run::Run;
pub(crate) type Result<T> = std::result::Result<T, String>;

pub(crate) fn integer(s: &str) -> Result<i32> {
    let s = s.replace('_', "");
    if let Some(hex) = s.strip_prefix("0x") {
        u32::from_str_radix(hex, 16).map(|i| i as i32)
    } else {
        s.parse()
    }
    .map_err(|_| format!("expected i32, got {s}"))
}
