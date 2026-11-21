use std::{
    fs::{File, OpenOptions},
    io,
    os::unix::fs::OpenOptionsExt,
    sync::Mutex,
};

static OUTPUT: Mutex<Option<(u32, io::Result<File>)>> = Mutex::new(None);

pub(super) fn with_file(write: impl FnOnce(&mut File) -> io::Result<()>) {
    let pid = std::process::id();
    let mut output = OUTPUT.lock().unwrap_or_else(|poison| poison.into_inner());
    if output.as_ref().is_none_or(|(owner, _)| *owner != pid) {
        // Open once, and only a fresh file. The shared temporary directory
        // can already contain a symlink or a stale map from a previous PID.
        let file = OpenOptions::new()
            .append(true)
            .create_new(true)
            .mode(0o600)
            .open(format!("/tmp/perf-{pid}.map"));
        *output = Some((pid, file));
    }
    let result = match &mut output.as_mut().expect("initialized above").1 {
        Ok(file) => write(file),
        Err(error) => Err(io::Error::from(error.kind())),
    };
    if let Err(error) = result {
        tracing::warn!(target: "splice::compile", %error, "could not write the perf map");
    }
}
