use std::{
    fs::{File, OpenOptions},
    io,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
    sync::Mutex,
};

static OUTPUT: Mutex<Option<Output>> = Mutex::new(None);

struct Output {
    pid: u32,
    file: io::Result<File>,
    warned: bool,
}

pub(super) fn with_file(write: impl FnOnce(&mut File) -> io::Result<()>) {
    let pid = std::process::id();
    let mut output = OUTPUT.lock().unwrap_or_else(|poison| poison.into_inner());
    let output = match &mut *output {
        Some(output) if output.pid == pid => output,
        // Open once per process, including after a fork.
        slot => slot.insert(Output {
            pid,
            file: open(Path::new(&format!("/tmp/perf-{pid}.map"))),
            warned: false,
        }),
    };
    let result = match &mut output.file {
        Ok(file) => write(file),
        Err(error) => Err(io::Error::from(error.kind())),
    };
    if let Err(error) = result
        && !std::mem::replace(&mut output.warned, true)
    {
        tracing::warn!(target: "splice::compile", %error, "could not write the perf map");
    }
}

/// Open the map for appending. The shared temporary directory can hold a
/// symlink or another user's file under this name, which are refused, and a
/// stale map from an earlier process with the same PID, which is truncated.
fn open(path: &Path) -> io::Result<File> {
    let file = OpenOptions::new()
        .append(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let metadata = file.metadata()?;
    // SAFETY: geteuid has no preconditions and cannot fail.
    let user = unsafe { libc::geteuid() };
    if !metadata.is_file() || metadata.uid() != user {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("{} is not a regular file of this user", path.display()),
        ));
    }
    file.set_len(0)?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::open;
    use std::io::Write;

    #[test]
    fn stale_maps_are_truncated_and_symlinks_refused() {
        let dir = std::env::temp_dir().join(format!("splice-perf-map-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stale = dir.join("perf-stale.map");
        std::fs::write(&stale, "1000 10 old_kernel\n").unwrap();
        let mut file = open(&stale).unwrap();
        writeln!(file, "2000 20 new_kernel").unwrap();
        assert_eq!(
            std::fs::read_to_string(&stale).unwrap(),
            "2000 20 new_kernel\n"
        );

        let target = dir.join("victim");
        std::fs::write(&target, "keep").unwrap();
        let link = dir.join("perf-link.map");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(open(&link).is_err());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "keep");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
