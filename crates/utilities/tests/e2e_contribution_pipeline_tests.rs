//! End-to-End (E2E) Integration Tests for RainAI Data Contribution, Staging,
//! Acoustic DSP Screening, Content-Addressed Deduplication, Quarantine Promotion,
//! and Non-Blocking Dataset Attribution.

use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs;
use std::path::Path;
use utilities::contribute::{
    import_local_directory, validate_audio_file, validate_manifest, AudioQualityThresholds,
    LocalImportOptions, MicrophoneSetup, PrecipitationRate, RainContributionManifest,
    RainSourceEntry,
};
use utilities::ingest::{
    compute_file_sha256, evaluate_attribution_policy,
    AttributionPolicyOutcome, LicenseTier, LicenseVerifier, ProvenanceManifest,
};

/// Generates a test WAV file with realistic broadband cavitation noise simulating rainfall.
fn create_synthetic_rain_wav(
    path: &Path,
    duration_secs: f32,
    sample_rate: u32,
    amplitude: f32,
) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).expect("Failed to create test WAV writer");

    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut rng_state: u64 = 987654321;

    for i in 0..num_samples {
        // High-frequency broadband noise generator
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let white = (rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0;

        // Modulate with droplet bursts
        let burst = ((i as f32 * 0.04).sin()).abs() * 0.6 + 0.4;
        let sample = white * amplitude * burst;

        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(pcm).unwrap(); // Left
        writer.write_sample(pcm).unwrap(); // Right
    }
    writer.finalize().expect("Failed to finalize test WAV");
}

#[test]
fn test_e2e_full_contribution_lifecycle() {
    let test_root = std::env::temp_dir().join(format!("rainai_e2e_lifecycle_{}", std::process::id()));
    let staging_quarantine = test_root.join("staging").join("quarantine");
    let staging_approved = test_root.join("staging").join("approved");
    let input_dir = test_root.join("incoming_recordings");
    let dataset_dir = test_root.join("dataset_rain");

    fs::create_dir_all(&staging_quarantine).unwrap();
    fs::create_dir_all(&staging_approved).unwrap();
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(&dataset_dir).unwrap();

    // =========================================================================
    // Stage 1: Synthesis of Environmental Rain Audio
    // =========================================================================
    let audio_file_path = input_dir.join("rain_canvas_tent_storm.wav");
    create_synthetic_rain_wav(&audio_file_path, 2.5, 48000, 0.25);

    let sha256 = compute_file_sha256(&audio_file_path).expect("Failed computing SHA-256");
    assert!(!sha256.is_empty());

    // Verify acoustic quality passes DSP screening
    let thresholds = AudioQualityThresholds::default();
    let (metrics, validated_sha, duration) =
        validate_audio_file(&audio_file_path, &thresholds).expect("DSP screening failed");

    assert_eq!(validated_sha, sha256);
    assert!((2.4..=2.6).contains(&duration));
    assert!(metrics.is_valid_rain_texture);
    assert!(metrics.rms_energy >= thresholds.min_rms_energy);

    // =========================================================================
    // Stage 2: Contribution Manifest with Unknown License (Quarantine Trigger)
    // =========================================================================
    let mut manifest = RainContributionManifest::new("E2E Test Set", "Alice Contributor");
    manifest.add_source(RainSourceEntry {
        id: "tent_heavy_01".to_string(),
        file_path: "rain_canvas_tent_storm.wav".to_string(),
        url: None,
        tags: vec!["canvas_tent".to_string(), "camping".to_string()],
        precipitation_rate: Some(PrecipitationRate::HeavyRain),
        environment: Some("Alpine campground forest ridge".to_string()),
        microphone_setup: Some(MicrophoneSetup::StereoOrtf),
        sample_rate: Some(48000),
        license: Some("Unknown".to_string()), // Triggers quarantine
        author: Some("Alice".to_string()),
        notes: Some("Field recording under high winds".to_string()),
        sha256: Some(sha256.clone()),
        descriptions: vec!["Heavy rain drumming rhythmically on taut canvas tent roof".to_string()],
        alternate_licenses: Vec::new(),
        contributors: vec!["Alice".to_string()],
    });

    let val_res = validate_manifest(&manifest, Some(&input_dir), Some(&thresholds))
        .expect("Manifest validation failed");

    // License was Unknown, so it must be rejected from direct ingestion
    assert_eq!(val_res.valid_entries.len(), 0);
    assert_eq!(val_res.rejected_entries.len(), 1);
    assert!(val_res.rejected_entries[0].rejection_reason.contains("Quarantined"));

    // Staging sidecar written to quarantine
    let quarantine_sidecar = staging_quarantine.join(format!("{}.json", sha256));
    let quarantine_audio = staging_quarantine.join(format!("{}.wav", sha256));
    fs::copy(&audio_file_path, &quarantine_audio).unwrap();

    let quarantine_payload = serde_json::json!({
        "sha256": sha256,
        "filename": "rain_canvas_tent_storm.wav",
        "license": "Unknown",
        "license_approved": false,
        "dsp_passed": true,
        "tags": ["canvas_tent", "camping"],
        "descriptions": ["Heavy rain drumming rhythmically on taut canvas tent roof"],
        "author": "Alice",
        "contributors": ["Alice"],
        "quarantine_reason": "UNAPPROVED_OR_MISSING_LICENSE"
    });
    fs::write(&quarantine_sidecar, serde_json::to_string_pretty(&quarantine_payload).unwrap()).unwrap();
    assert!(quarantine_sidecar.exists());

    // =========================================================================
    // Stage 3: Re-Contribution & Metadata Reconciliation (License Discovery)
    // =========================================================================
    // Later, the same audio is re-submitted with an approved license, extra description, and additional contributor
    let recontributed_entry = RainSourceEntry {
        id: "tent_heavy_01_annotated".to_string(),
        file_path: "rain_canvas_tent_storm.wav".to_string(),
        url: Some("https://archive.org/details/rain_canvas_tent".to_string()),
        tags: vec!["canvas_tent".to_string(), "mountain_storm".to_string()],
        precipitation_rate: Some(PrecipitationRate::ViolentStorm),
        environment: None,
        microphone_setup: None,
        sample_rate: None,
        license: Some("RainAI-FC-Proprietary-License".to_string()), // Approved grant
        author: Some("Bob".to_string()),
        notes: None,
        sha256: Some(sha256.clone()),
        descriptions: vec!["Acoustic high-frequency fabric saturation splatter texture".to_string()],
        alternate_licenses: Vec::new(),
        contributors: vec!["Bob".to_string()],
    };

    let mut reconciled_entry = manifest.sources[0].clone();
    reconciled_entry.reconcile_with(&recontributed_entry);

    // Assert reconciliation results
    assert_eq!(reconciled_entry.tags, vec!["canvas_tent", "camping", "mountain_storm"]);
    assert_eq!(reconciled_entry.descriptions.len(), 2);
    assert!(reconciled_entry.descriptions.contains(&"Heavy rain drumming rhythmically on taut canvas tent roof".to_string()));
    assert!(reconciled_entry.descriptions.contains(&"Acoustic high-frequency fabric saturation splatter texture".to_string()));
    assert_eq!(reconciled_entry.contributors, vec!["Alice", "Bob"]);
    assert_eq!(reconciled_entry.license.as_deref(), Some("RainAI-FC-Proprietary-License"));
    assert_eq!(reconciled_entry.url.as_deref(), Some("https://archive.org/details/rain_canvas_tent"));

    // =========================================================================
    // Stage 4: Promotion from Quarantine to Approved Staging
    // =========================================================================
    let (is_approved, tier, _) = LicenseVerifier::verify(reconciled_entry.license.as_deref().unwrap());
    assert!(is_approved);
    assert_eq!(tier, LicenseTier::ProjectProprietary);

    // Promote JSON sidecar and audio blob
    let approved_sidecar = staging_approved.join(format!("{}.json", sha256));
    let approved_audio = staging_approved.join(format!("{}.wav", sha256));

    let approved_payload = serde_json::json!({
        "sha256": sha256,
        "filename": "rain_canvas_tent_storm.wav",
        "license": "RainAI-FC-Proprietary-License",
        "alternate_licenses": ["Unknown"],
        "license_approved": true,
        "dsp_passed": true,
        "tags": reconciled_entry.tags,
        "descriptions": reconciled_entry.descriptions,
        "contributors": reconciled_entry.contributors,
        "promoted": true
    });
    fs::write(&approved_sidecar, serde_json::to_string_pretty(&approved_payload).unwrap()).unwrap();
    fs::copy(&quarantine_audio, &approved_audio).unwrap();

    // Clean quarantine
    fs::remove_file(&quarantine_sidecar).unwrap();
    fs::remove_file(&quarantine_audio).unwrap();

    assert!(approved_sidecar.exists());
    assert!(approved_audio.exists());
    assert!(!quarantine_sidecar.exists());

    // =========================================================================
    // Stage 5: Ingestion into Dataset Directory with Content-Addressed Deduplication
    // =========================================================================
    let options1 = LocalImportOptions {
        default_tags: vec!["canvas_tent".to_string()],
        default_rate: Some(PrecipitationRate::HeavyRain),
        default_mic: Some(MicrophoneSetup::StereoOrtf),
        default_license: Some("RainAI-FC-Proprietary-License".to_string()),
        author: Some("Alice".to_string()),
        target_dir: dataset_dir.clone(),
        update_sources_json: false,
        update_attributions: false,
        thresholds: AudioQualityThresholds::default(),
    };

    let report1 = import_local_directory(&input_dir, &options1).expect("Import 1 failed");
    assert_eq!(report1.successfully_imported, 1);

    let audio_files_after_1: Vec<_> = fs::read_dir(&dataset_dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    assert_eq!(audio_files_after_1.len(), 1);

    // Re-run import with an identical-hash file under a completely different name and new tags
    let duplicate_dir = test_root.join("incoming_duplicates");
    fs::create_dir_all(&duplicate_dir).unwrap();
    let dup_file = duplicate_dir.join("completely_different_name_same_data.wav");
    fs::copy(&audio_file_path, &dup_file).unwrap();

    let options2 = LocalImportOptions {
        default_tags: vec!["waterproof_fabric".to_string(), "monsoon".to_string()],
        default_rate: Some(PrecipitationRate::HeavyRain),
        default_mic: Some(MicrophoneSetup::StereoOrtf),
        default_license: Some("RainAI-FC-Proprietary-License".to_string()),
        author: Some("Bob".to_string()),
        target_dir: dataset_dir.clone(),
        update_sources_json: false,
        update_attributions: false,
        thresholds: AudioQualityThresholds::default(),
    };

    let report2 = import_local_directory(&duplicate_dir, &options2).expect("Import 2 failed");
    assert_eq!(report2.successfully_imported, 1);

    // ZERO BINARY DUPLICATION ASSERTION:
    // Still exactly 1 physical WAV file on disk!
    let audio_files_after_2: Vec<_> = fs::read_dir(&dataset_dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    assert_eq!(audio_files_after_2.len(), 1);

    // Verify manifest_provenance.json was harmonized
    let prov_path = dataset_dir.join("manifest_provenance.json");
    let prov_manifest: ProvenanceManifest =
        serde_json::from_str(&fs::read_to_string(&prov_path).unwrap()).unwrap();

    assert_eq!(prov_manifest.records.len(), 1);
    let record = &prov_manifest.records[0];
    assert_eq!(record.sha256, sha256);
    assert!(record.tags.contains(&"canvas_tent".to_string()));
    assert!(record.tags.contains(&"waterproof_fabric".to_string()));
    assert!(record.tags.contains(&"monsoon".to_string()));
    assert!(record.contributors.contains(&"Alice".to_string()));
    assert!(record.contributors.contains(&"Bob".to_string()));
    assert_eq!(record.license.as_deref(), Some("RainAI-FC-Proprietary-License"));

    // =========================================================================
    // Stage 6: Non-Blocking Attribution Outcome Evaluation
    // =========================================================================
    // With runtime contributor:
    let out_attributed = evaluate_attribution_policy(
        LicenseTier::ProjectProprietary,
        Some("Alice"),
        "RainAI-FC-Proprietary-License",
    );
    assert_eq!(
        out_attributed,
        AttributionPolicyOutcome::Attributed {
            contributor: "Alice".to_string(),
            license: "RainAI-FC-Proprietary-License".to_string(),
            tier: LicenseTier::ProjectProprietary,
        }
    );

    // When runtime contributor is untraced, model output is NEVER blocked!
    let out_untraced = evaluate_attribution_policy(
        LicenseTier::ProjectProprietary,
        None,
        "RainAI-FC-Proprietary-License",
    );
    match out_untraced {
        AttributionPolicyOutcome::GlobalTrainingAttributed { ref note, tier } => {
            assert_eq!(tier, LicenseTier::ProjectProprietary);
            assert!(note.contains("Permanent dataset-level attribution active"));
            assert!(note.contains("output served unconditionally"));
        }
        _ => panic!("Expected GlobalTrainingAttributed for untraced contributor"),
    }

    // =========================================================================
    // Stage 7: Clean Teardown
    // =========================================================================
    let _ = fs::remove_dir_all(&test_root);
}
