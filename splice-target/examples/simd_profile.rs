fn main() -> std::process::ExitCode {
    match splice_target::simd() {
        Ok(simd) => {
            println!("{:?}, {} bytes", simd.profile, simd.bytes);
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
