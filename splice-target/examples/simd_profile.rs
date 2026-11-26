//! Print the target `SPLICE_SIMD` selects on this machine: the profile and
//! vector width on the first line, the extensions every backend is given on
//! the second.
fn main() -> std::process::ExitCode {
    match splice_target::Profile::from_env()
        .and_then(|profile| splice_target::Target::host().map(|target| (profile, target)))
    {
        Ok((profile, target)) => {
            println!("{profile:?}, {} bytes", target.vector_bytes());
            let features = match (target.x86(), target.arm64()) {
                (Some(features), _) => format!("{features:?}"),
                (_, Some(features)) => format!("{features:?}"),
                (None, None) => String::new(),
            };
            // `Debug` lists every extension as `name: bool`; keep the enabled ones.
            let enabled: Vec<_> = features
                .split([',', '{', '}'])
                .filter_map(|field| field.trim().strip_suffix(": true"))
                .collect();
            println!("{}", enabled.join(" "));
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
