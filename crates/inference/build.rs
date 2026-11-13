//! Embeds the tier-0 ternary weights only when they exist locally.
//!
//! The weights are not in git (see PLAN step 4: they ship as a release asset),
//! so a fresh checkout has no `data/slice_0_ternary.bin`. Without this check
//! `include_bytes!` would fail the build. When the file is absent the crate
//! builds with no embedded model, and weights are loaded at runtime instead.

use std::path::Path;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rainai_embedded_weights)");
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let weights = Path::new(&manifest_dir).join("data/slice_0_ternary.bin");
    println!("cargo::rerun-if-changed={}", weights.display());

    if weights.is_file() {
        println!("cargo::rustc-cfg=rainai_embedded_weights");
        println!(
            "cargo::rustc-env=RAINAI_EMBEDDED_WEIGHTS={}",
            weights.display()
        );
    } else {
        println!(
            "cargo::warning=no {} found; building without embedded model weights",
            weights.display()
        );
    }
}
