fn main() {
    // Rebuild when the env var changes (toggling dashboard embedding)
    println!("cargo:rerun-if-env-changed=CORVIA_DASHBOARD_DIR");

    // Set CORVIA_DASHBOARD_DIR for rust-embed if not already set.
    // This ensures `cargo build` works without the env var — the embed
    // will simply be empty (placeholder directory with no files).
    if std::env::var("CORVIA_DASHBOARD_DIR").is_err() {
        let placeholder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("dashboard-placeholder");
        println!(
            "cargo:rustc-env=CORVIA_DASHBOARD_DIR={}",
            placeholder.display()
        );
    }
}
