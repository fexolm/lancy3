fn main() -> std::process::ExitCode {
    match splice_target::Profile::from_env().and_then(|profile| {
        splice_target::Target::host().map(|target| (profile, target.vector_bytes()))
    }) {
        Ok((profile, bytes)) => {
            println!("{profile:?}, {bytes} bytes");
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
