use std::sync::OnceLock;

pub(crate) fn enabled() -> proc_macro2::TokenStream {
    match explicit_mode() {
        Some(mode) => quote::quote!(#mode),
        None => quote::quote!(::core::cfg!(debug_assertions)),
    }
}

/// Stable Rust exposes `debug_assertions`, the default for overflow checking,
/// as a cfg. The independent override is supplied to the compiler driver.
pub(crate) fn explicit_mode() -> Option<bool> {
    static MODE: OnceLock<Option<bool>> = OnceLock::new();
    *MODE.get_or_init(|| {
        let args = std::env::args_os()
            .skip(1)
            .map(|arg| arg.to_string_lossy().into_owned())
            .flat_map(|arg| {
                if let Some(file) = arg.strip_prefix('@')
                    && let Ok(contents) = std::fs::read_to_string(file)
                {
                    return contents.lines().map(str::to_owned).collect();
                }
                vec![arg]
            });
        codegen_mode(args)
    })
}

fn codegen_mode(args: impl IntoIterator<Item = String>) -> Option<bool> {
    let mut args = args.into_iter();
    let mut mode = None;
    while let Some(arg) = args.next() {
        if arg == "--" {
            break;
        }
        let next;
        let option = if arg == "-C" || arg == "--codegen" {
            next = args.next();
            next.as_deref()
        } else {
            arg.strip_prefix("-C")
                .or_else(|| arg.strip_prefix("--codegen="))
        };
        let Some(option) = option else { continue };
        let (name, value) = option.split_once('=').unwrap_or((option, "yes"));
        if matches!(name, "overflow-checks" | "overflow_checks") {
            mode = match value {
                "y" | "yes" | "on" | "true" => Some(true),
                "n" | "no" | "off" | "false" => Some(false),
                _ => mode,
            };
        }
    }
    mode
}

#[cfg(test)]
mod tests {
    use super::codegen_mode;

    #[test]
    fn reads_only_the_last_explicit_overflow_codegen_option() {
        let mode = |args: &[&str]| codegen_mode(args.iter().map(ToString::to_string));
        assert_eq!(mode(&["-Copt-level=3", "-Cdebug-assertions=yes"]), None);
        assert_eq!(mode(&["--codegen", "overflow-checks=off"]), Some(false));
        assert_eq!(
            mode(&[
                "-C",
                "overflow-checks=n",
                "--codegen=overflow-checks",
                "--",
                "-Coverflow-checks=no",
            ]),
            Some(true)
        );
    }
}
