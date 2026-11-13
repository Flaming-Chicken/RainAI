//! Embeds the attribution dictionary only when it exists locally.
//!
//! `data/rain/attributions.bin` is generated from the source ledger and is not
//! in git, so a fresh checkout doesn't have it. Without this check
//! `include_bytes!` would fail the build. When the file is absent the app
//! shows "Attribution dictionary unavailable offline" instead.

use std::path::Path;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rainai_embedded_attributions)");
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let dict = Path::new(&manifest_dir).join("../../data/rain/attributions.bin");
    println!("cargo::rerun-if-changed={}", dict.display());

    if dict.is_file() {
        println!("cargo::rustc-cfg=rainai_embedded_attributions");
        println!(
            "cargo::rustc-env=RAINAI_EMBEDDED_ATTRIBUTIONS={}",
            dict.display()
        );
    }
}
