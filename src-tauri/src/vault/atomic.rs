use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Write `content` to `path` atomically: write to `{path}.tmp`, fsync, rename.
///
/// On Unix `rename` is atomic. On Windows it is atomic only when target is on
/// the same volume — true for our `app_data_dir` use case (PRD §9.2).
pub fn atomic_write(path: &Path, content: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let tmp = tmp_path_for(path);

    // Inline the write+fsync so we can do best-effort tmp cleanup on any
    // pre-rename failure without losing the original error.
    let write_result = (|| -> io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)?;
        file.write_all(content)?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }

    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(rename_err) => {
            // Cross-device rename (EXDEV) is the realistic failure here. Fall
            // back to copy+remove and warn so we know it happened.
            log::warn!(
                "[Rolo vault] atomic rename failed ({}), falling back to copy+remove",
                rename_err
            );
            let copy_result = fs::copy(&tmp, path).map(|_| ());
            let _ = fs::remove_file(&tmp);
            copy_result
        }
    }
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let tmp = path.to_path_buf();
    let mut os = tmp.into_os_string();
    os.push(".tmp");
    PathBuf::from(os)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn write_creates_file_with_content() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("hello.txt");
        atomic_write(&path, b"v1").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"v1");
    }

    #[test]
    fn second_write_replaces_target() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("hello.txt");
        atomic_write(&path, b"v1").unwrap();
        atomic_write(&path, b"v2").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"v2");
    }

    #[test]
    fn no_tmp_lingers_after_success() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("blob.bin");
        let big: Vec<u8> = (0..256 * 1024).map(|i| (i % 251) as u8).collect();
        atomic_write(&path, &big).unwrap();
        let tmp = dir.path().join("blob.bin.tmp");
        assert!(!tmp.exists(), "tmp file lingered after successful write");
        assert_eq!(fs::read(&path).unwrap().len(), big.len());
    }

    #[test]
    fn parent_dir_is_created() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("a").join("b").join("c").join("note.md");
        atomic_write(&nested, b"# hi").unwrap();
        assert_eq!(fs::read(&nested).unwrap(), b"# hi");
    }

    #[test]
    fn tmp_path_appends_suffix_preserving_extension() {
        let p = Path::new("/tmp/foo/meta.json");
        assert_eq!(tmp_path_for(p), PathBuf::from("/tmp/foo/meta.json.tmp"));
    }

    #[test]
    fn tmp_path_handles_no_extension() {
        let p = Path::new("/tmp/foo/log");
        assert_eq!(tmp_path_for(p), PathBuf::from("/tmp/foo/log.tmp"));
    }
}
