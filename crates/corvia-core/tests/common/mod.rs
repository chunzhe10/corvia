use corvia_core::config::Config;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

pub struct TestHarness {
    pub dir: TempDir,
    pub config: Config,
}

impl TestHarness {
    pub fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let config = Config::default();
        Self { dir, config }
    }

    pub fn base_dir(&self) -> &Path {
        self.dir.path()
    }

    pub fn copy_fixtures(&self) {
        let entries_dir = self.dir.path().join(".corvia/entries");
        std::fs::create_dir_all(&entries_dir).unwrap();
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        if fixtures_dir.exists() {
            for entry in std::fs::read_dir(&fixtures_dir).unwrap() {
                let entry = entry.unwrap();
                if entry.path().extension().is_some_and(|e| e == "md") {
                    let dest = entries_dir.join(entry.file_name());
                    std::fs::copy(entry.path(), dest).unwrap();
                }
            }
        }
    }
}
