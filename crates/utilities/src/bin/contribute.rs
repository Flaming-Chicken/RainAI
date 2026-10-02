//! CLI tool for standardized rain audio data contribution, validation, and dataset ingestion.
//!
//! # Usage
//!
//! ```bash
//! # Generate an annotated contribution manifest template:
//! cargo run -p utilities --bin rainai_contribute -- template --out my_rain_contribution.json
//!
//! # Validate a manifest or local audio directory:
//! cargo run -p utilities --bin rainai_contribute -- validate my_rain_contribution.json
//! cargo run -p utilities --bin rainai_contribute -- validate /path/to/my/recordings
//!
//! # Import a local directory of audio recordings:
//! cargo run -p utilities --bin rainai_contribute -- import-dir /path/to/recordings --surface tin_roof --license "CC0" --author "Alice"
//!
//! # Inspect current dataset balance and underrepresented surfaces:
//! cargo run -p utilities --bin rainai_contribute -- quota
//! ```

use anyhow::Result;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use utilities::contribute::{
    generate_manifest_template, import_local_directory, validate_audio_file, validate_manifest,
    AudioQualityThresholds, CanonicalSurface, LocalImportOptions, MicrophoneSetup,
    PrecipitationRate, RainContributionManifest,
};
use utilities::ingest::{DownloadItem, SurfaceBalanceQuota};

fn print_usage() {
    println!(
        r#"RainAI Community Data Contribution CLI v1.0
Standardized ingestion and acoustic validation for community precipitation audio.

USAGE:
    rainai_contribute <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    template                     Generate an annotated example contribution manifest
        --out <FILE>             Output path (default: rain_contribution_template.json)

    validate <PATH>              Acoustically validate a manifest (.json) or folder of WAV files

    import-dir <DIR>             Import a local directory of audio recordings into the dataset
        --surface <SURFACE>      Canonical surface (asphalt, pavement, tin_roof, canvas_tent,
                                 foliage, wood_deck, glass, puddle_shallow, water_deep)
        --rate <RATE>            Precipitation rate (drizzle, light_rain, moderate_rain, heavy_rain, violent_storm)
        --mic <MIC>              Microphone geometry (mono, stereo_spaced, stereo_ortf, binaural_in_ear, etc.)
        --license <LICENSE>      Open license string (e.g. "CC0 1.0 Universal", "CC-BY 4.0")
        --author <AUTHOR>        Contributor / field recordist name
        --target <DIR>           Target destination directory (default: Data/rain)

    import-manifest <FILE>       Import all files declared in a contribution manifest
        --target <DIR>           Target destination directory (default: Data/rain)

    quota                        Display current dataset surface diversity and quotas

EXAMPLES:
    rainai_contribute template --out contribution.json
    rainai_contribute validate ./my_recordings/
    rainai_contribute import-dir ./my_recordings/ --surface tin_roof --author "Liam" --license "CC0"
    rainai_contribute quota
"#
    );
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "template" => {
            let mut out_path = PathBuf::from("rain_contribution_template.json");
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--out" && i + 1 < args.len() {
                    out_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
                i += 1;
            }
            let template = generate_manifest_template();
            template.save_to_file(&out_path)?;
            println!("[+] Generated contribution manifest template at: {:?}", out_path);
            println!("    Fill in your audio file paths, surfaces, and licenses, then run:");
            println!("    rainai_contribute validate {:?}", out_path);
        }
        "validate" => {
            if args.len() < 3 {
                eprintln!("[!] Missing path argument for validate.");
                print_usage();
                return Ok(());
            }
            let path = Path::new(&args[2]);
            if !path.exists() {
                eprintln!("[!] Path does not exist: {:?}", path);
                return Ok(());
            }

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                println!("[*] Validating contribution manifest: {:?}", path);
                let manifest = RainContributionManifest::load_from_file(path)?;
                let base_dir = path.parent();
                let res = validate_manifest(&manifest, base_dir, None)?;

                println!("\n=== Manifest Validation Report ===");
                println!("Dataset Name:        {}", manifest.dataset_name);
                println!("Contributor:         {}", manifest.contributor_name);
                println!("Total Declarations:  {}", manifest.sources.len());
                println!("Valid & Approved:    {}", res.valid_entries.len());
                println!("Rejected / Failed:   {}", res.rejected_entries.len());
                println!("Total Duration:      {:.1}s", res.total_duration_secs);
                println!("Normalized Diversity:{:.1}%", res.normalized_diversity * 100.0);

                if !res.rejected_entries.is_empty() {
                    println!("\n[!] Rejection Diagnostics:");
                    for rej in &res.rejected_entries {
                        println!("  - [{}] {}: {}", rej.source_id, rej.file_path, rej.rejection_reason);
                    }
                }

                if res.is_passing {
                    println!("\n[+] SUCCESS: All audio assets passed licensing and acoustic screening!");
                } else {
                    println!("\n[!] WARNING: Some items failed validation. Please resolve issues above.");
                }
            } else if path.is_dir() {
                println!("[*] Scanning and acoustically screening directory: {:?}", path);
                let thresholds = AudioQualityThresholds::default();
                let mut passed = 0;
                let mut failed = 0;

                for entry in walkdir(path)? {
                    if entry.extension().and_then(|s| s.to_str()).unwrap_or("").eq_ignore_ascii_case("wav") {
                        match validate_audio_file(&entry, &thresholds) {
                            Ok((m, sha, dur)) => {
                                println!("  [PASS] {:?} ({:.1}s, RMS={:.4}, SHA256={:.8}..)", entry.file_name().unwrap_or_default(), dur, m.rms_energy, sha);
                                passed += 1;
                            }
                            Err(e) => {
                                println!("  [FAIL] {:?}: {}", entry.file_name().unwrap_or_default(), e);
                                failed += 1;
                            }
                        }
                    }
                }
                println!("\nDirectory Validation Summary: {} passed, {} failed.", passed, failed);
            } else if path.is_file() {
                let thresholds = AudioQualityThresholds::default();
                match validate_audio_file(path, &thresholds) {
                    Ok((m, sha, dur)) => {
                        println!("[PASS] {:?} is a valid rain recording!", path);
                        println!("  Duration:          {:.2}s", dur);
                        println!("  Sample Rate:       {} Hz", m.sample_rate);
                        println!("  Channels:          {}", m.channels);
                        println!("  RMS Energy:        {:.5}", m.rms_energy);
                        println!("  Peak Amplitude:    {:.4}", m.peak_amplitude);
                        println!("  Clipping Ratio:    {:.4}%", m.clipping_ratio * 100.0);
                        println!("  Spectral Flatness: {:.4}", m.spectral_flatness);
                        println!("  High Freq Ratio:   {:.4}", m.high_freq_ratio);
                        println!("  SHA256:            {}", sha);
                    }
                    Err(e) => {
                        eprintln!("[FAIL] Audio validation error: {}", e);
                    }
                }
            }
        }
        "import-dir" => {
            if args.len() < 3 {
                eprintln!("[!] Missing directory argument for import-dir.");
                print_usage();
                return Ok(());
            }
            let input_dir = PathBuf::from(&args[2]);
            let mut options = LocalImportOptions::default();

            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--surface" if i + 1 < args.len() => {
                        options.default_surface = Some(CanonicalSurface::from_tag(&args[i + 1]));
                        i += 1;
                    }
                    "--rate" if i + 1 < args.len() => {
                        options.default_rate = PrecipitationRate::from_tag(&args[i + 1]);
                        i += 1;
                    }
                    "--mic" if i + 1 < args.len() => {
                        options.default_mic = MicrophoneSetup::from_tag(&args[i + 1]);
                        i += 1;
                    }
                    "--license" if i + 1 < args.len() => {
                        options.default_license = args[i + 1].clone();
                        i += 1;
                    }
                    "--author" if i + 1 < args.len() => {
                        options.author = args[i + 1].clone();
                        i += 1;
                    }
                    "--target" if i + 1 < args.len() => {
                        options.target_dir = PathBuf::from(&args[i + 1]);
                        i += 1;
                    }
                    _ => {}
                }
                i += 1;
            }

            let report = import_local_directory(&input_dir, &options)?;
            println!("\n=== Ingestion Report ===");
            println!("Discovered candidates: {}", report.total_discovered);
            println!("Successfully imported: {}", report.successfully_imported);
            println!("Rejected files:        {}", report.rejected_count);
            println!("Total audio duration:  {:.1}s", report.total_duration_secs);
            println!("Target directory:      {}", report.target_directory);
            println!("Surface distribution:  {:?}", report.surface_distribution);
        }
        "import-manifest" => {
            if args.len() < 3 {
                eprintln!("[!] Missing manifest file argument.");
                print_usage();
                return Ok(());
            }
            let manifest_path = Path::new(&args[2]);
            let manifest = RainContributionManifest::load_from_file(manifest_path)?;
            let base_dir = manifest_path.parent();

            let mut options = LocalImportOptions::default();
            if args.len() >= 5 && args[3] == "--target" {
                options.target_dir = PathBuf::from(&args[4]);
            }

            println!("[*] Validating manifest sources before import...");
            let res = validate_manifest(&manifest, base_dir, Some(&options.thresholds))?;
            println!("Approved {} / {} sources.", res.valid_entries.len(), manifest.sources.len());

            let mut imported = 0;
            for entry in &res.valid_entries {
                let src_path = if let Some(base) = base_dir {
                    base.join(&entry.source.file_path)
                } else {
                    PathBuf::from(&entry.source.file_path)
                };

                if src_path.exists() {
                    let dest_name = format!(
                        "contrib_{}_{}_{}",
                        entry.source.surface.as_str(),
                        entry.source.author.replace(' ', "_"),
                        src_path.file_name().unwrap_or_default().to_string_lossy()
                    );
                    let dest = options.target_dir.join(&dest_name);
                    fs::create_dir_all(&options.target_dir)?;
                    fs::copy(&src_path, &dest)?;
                    imported += 1;
                }
            }
            println!("[+] Imported {} audio files into {:?}", imported, options.target_dir);
        }
        "quota" => {
            println!("[*] Auditing active dataset surface balance & diversity...");
            let sources_path = shared::paths::WorkspacePaths::resolve_sources()
                .unwrap_or_else(|| PathBuf::from("sources.json"));

            if !sources_path.exists() {
                println!("[!] sources.json catalog not found at {:?}", sources_path);
                return Ok(());
            }

            let content = fs::read_to_string(&sources_path)?;
            let sources: Vec<DownloadItem> = serde_json::from_str(&content)?;

            let mut quota = SurfaceBalanceQuota::new(10);
            for s in &sources {
                quota.record(CanonicalSurface::from_tag(&s.category));
            }

            println!("\n=== Surface Balance Quota Register ===");
            println!("Total Audio Sources:   {}", quota.total_samples());
            println!("Shannon Diversity:     {:.3} nats", quota.shannon_entropy());
            println!("Normalized Diversity:  {:.1}%", quota.normalized_diversity() * 100.0);
            println!("\nSurface Distribution:");
            for surface in CanonicalSurface::ALL {
                let count = quota.counts.get(&surface).copied().unwrap_or(0);
                let bar_len = (count / 2).min(30);
                let bar = "=".repeat(bar_len);
                println!("  {:<15} [{:>3}] |{}", surface.as_str(), count, bar);
            }

            let underrepresented = quota.underrepresented_surfaces();
            if !underrepresented.is_empty() {
                println!("\n[!] Underrepresented Surfaces (Contributions needed):");
                for (s, count) in underrepresented {
                    println!("  - {:<15} (current count: {}, target: {})", s.as_str(), count, quota.target_per_surface);
                }
            } else {
                println!("\n[+] All 9 canonical surfaces satisfy target quotas!");
            }
        }
        _ => {
            print_usage();
        }
    }

    Ok(())
}

fn walkdir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                files.extend(walkdir(&path)?);
            } else {
                files.push(path);
            }
        }
    }
    Ok(files)
}
