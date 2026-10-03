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
    assert_not_main_branch, generate_manifest_template, import_local_directory, validate_audio_file,
    validate_manifest, AudioQualityThresholds, LocalImportOptions, MicrophoneSetup,
    PrecipitationRate, RainContributionManifest,
};
use utilities::ingest::{
    chrono_lite_timestamp, DownloadItem, LicenseVerifier, ProvenanceManifest, ProvenanceRecord,
    TagBalanceQuota,
};

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
    triage [DIR]                 Inspect quarantined contributions and failure diagnostics (default: Data/staging/quarantine)
    pull-approved [DIR]          Promote approved staging records to dev branch Data/raw/ and sources.json
    reconcile [TARGET_DIR]       Promote approved quarantine items and reconcile duplicate metadata in manifest_provenance.json
    export-attributions          Compile manifest_provenance.json into a zero-copy binary dictionary
        --manifest <FILE>        Source provenance manifest (default: Data/rain/manifest_provenance.json)
        --out <FILE>             Output binary path (default: Data/rain/attributions.bin)

EXAMPLES:
    rainai_contribute template --out contribution.json
    rainai_contribute validate ./my_recordings/
    rainai_contribute import-dir ./my_recordings/ --surface tin_roof --author "Liam" --license "CC0"
    rainai_contribute quota
    rainai_contribute triage
    rainai_contribute reconcile
    rainai_contribute pull-approved
    rainai_contribute export-attributions --out Data/rain/attributions.bin
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
                        options.default_tags = vec![args[i + 1].clone()];
                        i += 1;
                    }
                    "--rate" if i + 1 < args.len() => {
                        options.default_rate = Some(PrecipitationRate::from_tag(&args[i + 1]));
                        i += 1;
                    }
                    "--mic" if i + 1 < args.len() => {
                        options.default_mic = Some(MicrophoneSetup::from_tag(&args[i + 1]));
                        i += 1;
                    }
                    "--license" if i + 1 < args.len() => {
                        options.default_license = Some(args[i + 1].clone());
                        i += 1;
                    }
                    "--author" if i + 1 < args.len() => {
                        options.author = Some(args[i + 1].clone());
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
                    let first_tag = entry.source.tags.first().cloned().unwrap_or_else(|| "audio".to_string());
                    let author_str = entry.source.author.as_deref().unwrap_or("Anonymous").replace(' ', "_");
                    let dest_name = format!(
                        "contrib_{}_{}_{}",
                        first_tag,
                        author_str,
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

            let mut quota = TagBalanceQuota::new();
            for s in &sources {
                quota.record(&[s.category.clone()]);
            }

            println!("\n=== Tags Balance Quota Register ===");
            println!("Total Audio Sources:   {}", quota.total_samples());
            println!("\nTags Distribution:");
            let mut counts: Vec<_> = quota.counts.iter().collect();
            counts.sort_by_key(|a| std::cmp::Reverse(*a.1));
            for (tag, count) in counts {
                let bar_len = (count / 2).min(30);
                let bar = "=".repeat(bar_len);
                println!("  {:<15} [{:>3}] |{}", tag, count, bar);
            }
        }
        "triage" => {
            let quarantine_dir = if args.len() >= 3 {
                PathBuf::from(&args[2])
            } else {
                PathBuf::from("Data/staging/quarantine")
            };

            println!("\n=== RainAI Staging Quarantine Triage ===");
            println!("Inspecting quarantine directory: {:?}", quarantine_dir);

            if !quarantine_dir.exists() {
                println!("[+] Quarantine queue is empty! Zero rejected submissions.");
                return Ok(());
            }

            let mut count = 0;
            for entry in fs::read_dir(&quarantine_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    count += 1;
                    let content = fs::read_to_string(&path)?;
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                        println!("\n[!] Quarantined Item #{}: {}", count, path.file_stem().unwrap_or_default().to_string_lossy());
                        println!("    Reason:  {}", val.get("quarantine_reason").and_then(|v| v.as_str()).unwrap_or("Unknown reason"));
                        println!("    License: {}", val.get("license").and_then(|v| v.as_str()).unwrap_or("Missing"));
                        println!("    Author:  {}", val.get("author").and_then(|v| v.as_str()).unwrap_or("Anonymous"));
                        println!("    Tags:    {}", val.get("tags").map(|v| v.to_string()).unwrap_or_default());
                        if let Some(url) = val.get("url").and_then(|v| v.as_str()) {
                            println!("    Source:  {}", url);
                        }
                    }
                }
            }

            if count == 0 {
                println!("[+] No quarantined records found in {:?}", quarantine_dir);
            } else {
                println!("\nTotal Quarantined Items: {}", count);
            }
        }
        "pull-approved" => {
            assert_not_main_branch()?;

            let approved_dir = if args.len() >= 3 {
                PathBuf::from(&args[2])
            } else {
                PathBuf::from("Data/staging/approved")
            };

            let target_raw_dir = PathBuf::from("Data/raw");
            println!("\n=== Promoting Approved Staging Audio to Git LFS ===");
            println!("Reading approved staging from: {:?}", approved_dir);
            println!("Target Git LFS destination:    {:?}", target_raw_dir);

            if !approved_dir.exists() {
                println!("[!] Approved staging directory does not exist: {:?}", approved_dir);
                return Ok(());
            }

            fs::create_dir_all(&target_raw_dir)?;
            let mut promoted = 0;

            for entry in fs::read_dir(&approved_dir)? {
                let entry = entry?;
                let path = entry.path();
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();

                // Copy approved audio blobs directly
                if ext == "flac" || ext == "opus" || ext == "ogg" || ext == "mp3" || ext == "wav" {
                    let dest = target_raw_dir.join(path.file_name().unwrap());
                    fs::copy(&path, &dest)?;
                    promoted += 1;
                    println!("  [+] Promoted {:?} -> {:?}", path.file_name().unwrap(), dest);
                }
            }

            println!("\nSuccessfully promoted {} approved files into Git LFS (Data/raw/).", promoted);
        }
        "reconcile" => {
            assert_not_main_branch()?;

            let target_dir = if args.len() >= 3 {
                PathBuf::from(&args[2])
            } else {
                shared::paths::WorkspacePaths::resolve_attributions()
                    .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                    .unwrap_or_else(|| PathBuf::from("Data/rain"))
            };

            println!("\n=== Reconciling Dataset Provenance & Staging ===");
            println!("Target dataset directory: {:?}", target_dir);

            // 1. Reconcile Quarantine -> Approved if any quarantined asset now qualifies
            let quarantine_dir = PathBuf::from("Data/staging/quarantine");
            let approved_dir = PathBuf::from("Data/staging/approved");
            let mut promoted_from_quarantine = 0;

            if quarantine_dir.exists() {
                fs::create_dir_all(&approved_dir)?;
                for entry in fs::read_dir(&quarantine_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Ok(content) = fs::read_to_string(&path) {
                            if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&content) {
                                let lic_str = val.get("license").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();
                                let (ok, tier, _reason) = LicenseVerifier::verify(&lic_str);
                                let dsp_passed = val.get("dsp_passed").and_then(|v| v.as_bool()).unwrap_or(true);

                                if ok && tier.is_approved() && dsp_passed {
                                    // Move JSON sidecar to approved
                                    let filename = path.file_name().unwrap();
                                    let dest_json = approved_dir.join(filename);
                                    if let Some(obj) = val.as_object_mut() {
                                        obj.insert("target_prefix".to_string(), serde_json::Value::String("staging/approved".to_string()));
                                        obj.insert("quarantine_reason".to_string(), serde_json::Value::Null);
                                        obj.insert("promoted_at".to_string(), serde_json::Value::String(chrono_lite_timestamp()));
                                    }
                                    fs::write(&dest_json, serde_json::to_string_pretty(&val)?)?;
                                    fs::remove_file(&path)?;

                                    // Move corresponding audio blob if present
                                    let stem = path.file_stem().unwrap().to_string_lossy();
                                    for ext in ["flac", "wav", "m4a", "opus", "mp3"] {
                                        let old_blob = quarantine_dir.join(format!("{}.{}", stem, ext));
                                        if old_blob.exists() {
                                            let new_blob = approved_dir.join(format!("{}.{}", stem, ext));
                                            fs::copy(&old_blob, &new_blob)?;
                                            fs::remove_file(&old_blob)?;
                                        }
                                    }

                                    promoted_from_quarantine += 1;
                                    println!("  [+] Promoted {:?} from quarantine to approved (License: {})", filename, lic_str);
                                }
                            }
                        }
                    }
                }
            }

            // 2. Reconcile manifest_provenance.json deduplicating identical SHA-256 hashes
            let prov_manifest_path = target_dir.join("manifest_provenance.json");
            let mut merged_duplicates = 0;

            if prov_manifest_path.exists() {
                let content = fs::read_to_string(&prov_manifest_path)?;
                let manifest: ProvenanceManifest = serde_json::from_str(&content)?;
                let original_count = manifest.records.len();

                // Group by SHA-256
                let mut hash_map: std::collections::HashMap<String, Vec<ProvenanceRecord>> = std::collections::HashMap::new();
                for r in manifest.records {
                    hash_map.entry(r.sha256.clone()).or_default().push(r);
                }

                let mut reconciled_records: Vec<ProvenanceRecord> = Vec::new();
                let mut tags_distribution: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

                for (_sha, records) in hash_map {
                    if records.len() > 1 {
                        merged_duplicates += records.len() - 1;
                        let mut base = records[0].clone();
                        for extra in records.into_iter().skip(1) {
                            base.reconcile_with(extra);
                        }
                        for t in &base.tags {
                            *tags_distribution.entry(t.clone()).or_insert(0) += 1;
                        }
                        reconciled_records.push(base);
                    } else if let Some(single) = records.into_iter().next() {
                        for t in &single.tags {
                            *tags_distribution.entry(t.clone()).or_insert(0) += 1;
                        }
                        reconciled_records.push(single);
                    }
                }

                reconciled_records.sort_by(|a, b| a.filename.cmp(&b.filename));

                let reconciled_manifest = ProvenanceManifest {
                    generated_at_utc: chrono_lite_timestamp(),
                    total_sources: reconciled_records.len(),
                    tags_distribution,
                    records: reconciled_records,
                };

                let json = serde_json::to_string_pretty(&reconciled_manifest)?;
                fs::write(&prov_manifest_path, json)?;
                println!(
                    "[+] Harmonized manifest_provenance.json: merged {} duplicate records ({} -> {} unique sources).",
                    merged_duplicates, original_count, reconciled_manifest.total_sources
                );
            }

            println!("\nReconciliation Complete:");
            println!("  Quarantined items promoted: {}", promoted_from_quarantine);
            println!("  Duplicate provenance records merged: {}", merged_duplicates);
        }
        "export-attributions" => {
            let mut manifest_path = PathBuf::from("Data/rain/manifest_provenance.json");
            let mut out_path = PathBuf::from("Data/rain/attributions.bin");
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--manifest" && i + 1 < args.len() {
                    manifest_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                } else if args[i] == "--out" && i + 1 < args.len() {
                    out_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
                i += 1;
            }

            if !manifest_path.exists() {
                eprintln!("[!] Provenance manifest not found at: {:?}", manifest_path);
                return Ok(());
            }

            println!("[*] Loading provenance manifest from: {:?}", manifest_path);
            let content = fs::read_to_string(&manifest_path)?;
            let manifest: ProvenanceManifest = serde_json::from_str(&content)?;

            let mut inputs = Vec::with_capacity(manifest.records.len());
            for rec in manifest.records {
                let contributor = if !rec.contributors.is_empty() {
                    rec.contributors.join(", ")
                } else {
                    "RainAI Community".to_string()
                };

                let license = rec.license.unwrap_or_else(|| "Unknown".to_string());
                let surface = rec.tags.first().cloned().unwrap_or_else(|| "ambient".to_string());
                let tier = rec.license_tier.preference_rank();

                inputs.push(shared::attribution::AttributionRecordInput {
                    sha256_hex: rec.sha256,
                    contributor,
                    license,
                    license_tier: tier,
                    surface,
                });
            }

            let num_records = inputs.len();
            let binary_data = shared::attribution::compile_binary_attribution_dictionary(inputs);

            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }

            utilities::ingest::atomic_write(&out_path, &binary_data)?;
            println!(
                "[+] Successfully exported {} attribution records to binary dictionary: {:?} ({} bytes)",
                num_records, out_path, binary_data.len()
            );
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
