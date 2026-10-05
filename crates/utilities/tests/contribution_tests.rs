//! Integration tests for standardized community rain data contribution, validation,
//! acoustic quality screening, and local dataset directory ingestion.

use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs;
use std::path::{Path, PathBuf};
use utilities::contribute::{
    AudioQualityThresholds, LocalImportOptions, MicrophoneSetup, PrecipitationRate,
    RainContributionManifest, RainSourceEntry, generate_manifest_template, import_local_directory,
    validate_audio_file, validate_manifest,
};

/// Generates a test WAV file with realistic broadband cavitation noise simulating rainfall.
fn create_synthetic_rain_wav(
    path: &Path,
    duration_secs: f32,
    sample_rate: u32,
    amplitude: f32,
    clipping: bool,
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
    let mut rng_state: u64 = 123456789;

    for i in 0..num_samples {
        // High-frequency broadband noise generator (droplet cavitation)
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let white = (rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0;

        // Modulate with droplet impact bursts
        let burst = ((i as f32 * 0.05).sin()).abs() * 0.5 + 0.5;
        let mut sample = white * amplitude * burst;

        if clipping && i % 10 == 0 {
            sample = 1.5; // Trigger severe clipping
        }

        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(pcm).unwrap(); // Left
        writer.write_sample(pcm).unwrap(); // Right
    }
    writer.finalize().expect("Failed to finalize test WAV");
}

/// Generates a silent test WAV file.
fn create_silent_wav(path: &Path, duration_secs: f32, sample_rate: u32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).expect("Failed to create silent WAV");
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    for _ in 0..num_samples {
        writer.write_sample(0i16).unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn test_manifest_serialization_and_template() {
    let template = generate_manifest_template();
    assert_eq!(template.manifest_version, "1.0");
    assert!(!template.sources.is_empty());
    assert_eq!(
        template.default_license.as_deref(),
        Some("CC0 1.0 Universal")
    );

    // Verify template contains tags
    let tags: Vec<String> = template
        .sources
        .iter()
        .flat_map(|s| s.tags.clone())
        .collect();
    assert!(tags.iter().any(|t| t == "tin_roof"));
    assert!(tags.iter().any(|t| t == "foliage"));
    assert!(tags.iter().any(|t| t == "glass"));

    // Roundtrip JSON serialization
    let json = template
        .to_json_pretty()
        .expect("Failed to serialize template");
    let roundtrip = RainContributionManifest::from_json(&json).expect("Failed to deserialize");

    assert_eq!(roundtrip.dataset_name, template.dataset_name);
    assert_eq!(roundtrip.sources.len(), template.sources.len());
    assert_eq!(roundtrip.sources[0].id, template.sources[0].id);
    assert_eq!(roundtrip.sources[0].tags, template.sources[0].tags);
}

#[test]
fn test_license_and_quality_screening_in_manifest_validation() {
    let mut manifest = RainContributionManifest::new("Test Contribution", "Researcher");

    // Approved license entry (remote URL)
    manifest.add_source(RainSourceEntry {
        id: "valid_cc0_src".to_string(),
        file_path: "remote_rain.wav".to_string(),
        url: Some("https://example.org/rain.wav".to_string()),
        tags: vec!["pavement".to_string()],
        precipitation_rate: Some(PrecipitationRate::ModerateRain),
        environment: Some("Sidewalk".to_string()),
        microphone_setup: Some(MicrophoneSetup::StereoSpaced),
        sample_rate: Some(48000),
        license: Some("CC0 1.0 Universal".to_string()),
        author: Some("Alice".to_string()),
        notes: None,
        sha256: None,
        descriptions: vec!["Wet pavement drizzle".to_string()],
        alternate_licenses: Vec::new(),
        contributors: vec!["Alice".to_string()],
    });

    // Rejected license entry (NonCommercial clause)
    manifest.add_source(RainSourceEntry {
        id: "rejected_nc_src".to_string(),
        file_path: "nc_rain.wav".to_string(),
        url: Some("https://example.org/nc.wav".to_string()),
        tags: vec!["asphalt".to_string()],
        precipitation_rate: Some(PrecipitationRate::HeavyRain),
        environment: Some("Street".to_string()),
        microphone_setup: Some(MicrophoneSetup::Mono),
        sample_rate: Some(48000),
        license: Some("CC-BY-NC 4.0".to_string()),
        author: Some("Bob".to_string()),
        notes: None,
        sha256: None,
        descriptions: vec!["Street downpour".to_string()],
        alternate_licenses: Vec::new(),
        contributors: vec!["Bob".to_string()],
    });

    let res = validate_manifest(&manifest, None, None).expect("Validation execution failed");

    assert_eq!(res.valid_entries.len(), 1);
    assert_eq!(res.rejected_entries.len(), 1);
    assert_eq!(res.valid_entries[0].source.id, "valid_cc0_src");
    assert_eq!(res.rejected_entries[0].source_id, "rejected_nc_src");
    assert!(
        res.rejected_entries[0]
            .rejection_reason
            .contains("NonCommercial")
    );
    assert!(!res.is_passing);
}

#[test]
fn test_audio_validation_on_synthetic_pcm_wav() {
    let temp_dir = std::env::temp_dir().join(format!("rainai_test_pcm_{}", std::process::id()));
    let _ = fs::create_dir_all(&temp_dir);
    let wav_path = temp_dir.join("valid_rain_sample.wav");

    // Create 3.0s realistic rain audio at 48 kHz
    create_synthetic_rain_wav(&wav_path, 3.0, 48000, 0.25, false);

    let thresholds = AudioQualityThresholds::default();
    let (metrics, sha, duration) = validate_audio_file(&wav_path, &thresholds)
        .expect("Valid rain audio rejected unexpectedly");

    assert!((2.9..=3.1).contains(&duration));
    assert_eq!(metrics.sample_rate, 48000);
    assert_eq!(metrics.channels, 2);
    assert!(metrics.rms_energy > thresholds.min_rms_energy);
    assert!(metrics.clipping_ratio <= thresholds.max_clipping_ratio);
    assert!(metrics.spectral_flatness >= thresholds.min_spectral_flatness);
    assert!(!sha.is_empty());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_audio_validation_rejects_silent_and_clipped() {
    let temp_dir = std::env::temp_dir().join(format!("rainai_test_rej_{}", std::process::id()));
    let _ = fs::create_dir_all(&temp_dir);
    let thresholds = AudioQualityThresholds::default();

    // 1. Silent file
    let silent_path = temp_dir.join("silent.wav");
    create_silent_wav(&silent_path, 3.0, 48000);
    let silent_res = validate_audio_file(&silent_path, &thresholds);
    assert!(silent_res.is_err());
    let err_msg = silent_res.unwrap_err().to_string();
    assert!(err_msg.contains("RMS energy too low"));

    // 2. Clipped file
    let clipped_path = temp_dir.join("clipped.wav");
    create_synthetic_rain_wav(&clipped_path, 3.0, 48000, 0.90, true);
    let clipped_res = validate_audio_file(&clipped_path, &thresholds);
    assert!(clipped_res.is_err());
    let err_msg2 = clipped_res.unwrap_err().to_string();
    assert!(err_msg2.contains("severely clipped"));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_import_local_directory_end_to_end() {
    let temp_dir = std::env::temp_dir().join(format!("rainai_test_import_{}", std::process::id()));
    let input_dir = temp_dir.join("raw_input");
    let target_dir = temp_dir.join("dataset_target");

    let _ = fs::create_dir_all(&input_dir);
    let _ = fs::create_dir_all(&target_dir);

    // Create 3 valid audio recordings with different names
    create_synthetic_rain_wav(
        &input_dir.join("rain_tin_roof_porch.wav"),
        2.5,
        48000,
        0.20,
        false,
    );
    create_synthetic_rain_wav(
        &input_dir.join("heavy_deluge_asphalt_road.wav"),
        2.5,
        48000,
        0.22,
        false,
    );
    create_synthetic_rain_wav(
        &input_dir.join("window_glass_patter.wav"),
        2.5,
        48000,
        0.18,
        false,
    );

    // Create 1 invalid silent file that should be rejected
    create_silent_wav(&input_dir.join("bad_silent_recording.wav"), 2.5, 48000);

    let options = LocalImportOptions {
        default_tags: Vec::new(),
        default_rate: Some(PrecipitationRate::ModerateRain),
        default_mic: Some(MicrophoneSetup::StereoSpaced),
        default_license: Some("CC0 1.0 Universal".to_string()),
        author: Some("Alice Cooper".to_string()),
        target_dir: target_dir.clone(),
        update_sources_json: false,
        update_attributions: false,
        thresholds: AudioQualityThresholds::default(),
    };

    let report = import_local_directory(&input_dir, &options).expect("Directory import failed");

    assert_eq!(report.total_discovered, 4);
    assert_eq!(report.successfully_imported, 3);
    assert_eq!(report.rejected_count, 1);
    assert_eq!(report.rejected_reasons[0].source_id, "bad_silent_recording");

    // Verify files were physically copied to target directory
    let target_wav_files: Vec<PathBuf> = fs::read_dir(&target_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    assert_eq!(target_wav_files.len(), 3);
    assert!(target_dir.join("manifest_provenance.json").exists());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_precipitation_rate_and_mic_setup_tagging() {
    assert_eq!(
        PrecipitationRate::from_tag("misty_drizzle"),
        PrecipitationRate::Drizzle
    );
    assert_eq!(
        PrecipitationRate::from_tag("light_shower"),
        PrecipitationRate::LightRain
    );
    assert_eq!(
        PrecipitationRate::from_tag("heavy_downpour"),
        PrecipitationRate::HeavyRain
    );
    assert_eq!(
        PrecipitationRate::from_tag("thunder_storm"),
        PrecipitationRate::ViolentStorm
    );
    assert_eq!(
        PrecipitationRate::from_tag("regular_rain"),
        PrecipitationRate::ModerateRain
    );

    assert_eq!(
        MicrophoneSetup::from_tag("in_ear_binaural_mics"),
        MicrophoneSetup::BinauralInEar
    );
    assert_eq!(
        MicrophoneSetup::from_tag("ambisonic_b_format"),
        MicrophoneSetup::AmbisonicFoa
    );
    assert_eq!(
        MicrophoneSetup::from_tag("higher_order_hoa_16ch"),
        MicrophoneSetup::AmbisonicHoa
    );
    assert_eq!(
        MicrophoneSetup::from_tag("underwater_hydrophone"),
        MicrophoneSetup::Hydrophone
    );
    assert_eq!(
        MicrophoneSetup::from_tag("piezo_contact_disc"),
        MicrophoneSetup::ContactMic
    );
    assert_eq!(
        MicrophoneSetup::from_tag("spaced_pair_omnis"),
        MicrophoneSetup::StereoSpaced
    );
}

#[test]
fn test_source_entry_reconciliation_multi_license_and_descriptions() {
    let mut entry1 = RainSourceEntry {
        id: "sample_001".to_string(),
        file_path: "audio/rain_roof.wav".to_string(),
        url: None,
        tags: vec!["tin_roof".to_string(), "metal".to_string()],
        precipitation_rate: Some(PrecipitationRate::ModerateRain),
        environment: Some("Barn shed".to_string()),
        microphone_setup: Some(MicrophoneSetup::StereoOrtf),
        sample_rate: Some(48000),
        license: Some("CC-BY 4.0".to_string()),
        author: Some("Alice".to_string()),
        notes: Some("Original field recording from 2024".to_string()),
        sha256: Some("abcdef123456".to_string()),
        descriptions: vec!["Rhythmic rain drumming on corrugated tin roof".to_string()],
        alternate_licenses: Vec::new(),
        contributors: vec!["Alice".to_string()],
    };

    let entry2 = RainSourceEntry {
        id: "sample_001_recontributed".to_string(),
        file_path: "audio/rain_roof.wav".to_string(),
        url: Some("https://archive.org/details/rain_roof".to_string()),
        tags: vec![
            "tin_roof".to_string(),
            "rural".to_string(),
            "monsoon".to_string(),
        ],
        precipitation_rate: Some(PrecipitationRate::HeavyRain),
        environment: None,
        microphone_setup: None,
        sample_rate: None,
        license: Some("RainAI-FC-Proprietary-License".to_string()),
        author: Some("Bob".to_string()),
        notes: Some("Re-annotated perspective".to_string()),
        sha256: Some("abcdef123456".to_string()),
        descriptions: vec![
            "Resonant high-frequency metallic splatter texture".to_string(),
            "Rhythmic rain drumming on corrugated tin roof".to_string(), // Duplicate description should be deduplicated
        ],
        alternate_licenses: Vec::new(),
        contributors: vec!["Bob".to_string()],
    };

    entry1.reconcile_with(&entry2);

    // 1. Tags reconciled without duplicates
    assert_eq!(entry1.tags, vec!["tin_roof", "metal", "rural", "monsoon"]);

    // 2. Parallel descriptions preserved without duplicates
    assert_eq!(entry1.descriptions.len(), 3); // "Rhythmic...", "Original...", "Resonant..."
    assert!(
        entry1
            .descriptions
            .contains(&"Rhythmic rain drumming on corrugated tin roof".to_string())
    );
    assert!(
        entry1
            .descriptions
            .contains(&"Resonant high-frequency metallic splatter texture".to_string())
    );

    // 3. Multi-contributors merged
    assert_eq!(entry1.contributors, vec!["Alice", "Bob"]);

    // 4. Predominant license selection: ProjectProprietary (Rank 6) > CC-BY 4.0 (Rank 4)
    assert_eq!(
        entry1.license.as_deref(),
        Some("RainAI-FC-Proprietary-License")
    );
    assert!(entry1.alternate_licenses.contains(&"CC-BY 4.0".to_string()));

    // 5. Backfilled metadata
    assert_eq!(
        entry1.url.as_deref(),
        Some("https://archive.org/details/rain_roof")
    );
}

#[test]
fn test_validate_manifest_content_addressed_deduplication() {
    let temp_dir =
        std::env::temp_dir().join(format!("rainai_test_manifest_dedup_{}", std::process::id()));
    let _ = fs::create_dir_all(&temp_dir);
    let wav_path = temp_dir.join("shared_rain_audio.wav");

    create_synthetic_rain_wav(&wav_path, 2.0, 48000, 0.20, false);

    let mut manifest = RainContributionManifest::new("Deduplication Test", "Curator");

    // First entry pointing to the WAV file
    manifest.add_source(RainSourceEntry {
        id: "first_submission".to_string(),
        file_path: "shared_rain_audio.wav".to_string(),
        url: None,
        tags: vec!["pavement".to_string()],
        precipitation_rate: Some(PrecipitationRate::ModerateRain),
        environment: Some("Street".to_string()),
        microphone_setup: Some(MicrophoneSetup::StereoSpaced),
        sample_rate: Some(48000),
        license: Some("CC-BY 4.0".to_string()),
        author: Some("Alice".to_string()),
        notes: None,
        sha256: None,
        descriptions: vec!["Urban sidewalk rain".to_string()],
        alternate_licenses: Vec::new(),
        contributors: vec!["Alice".to_string()],
    });

    // Second duplicate entry pointing to the exact same audio file with different metadata & license
    manifest.add_source(RainSourceEntry {
        id: "duplicate_submission".to_string(),
        file_path: "shared_rain_audio.wav".to_string(),
        url: None,
        tags: vec!["asphalt".to_string(), "pavement".to_string()],
        precipitation_rate: Some(PrecipitationRate::ModerateRain),
        environment: Some("Downtown".to_string()),
        microphone_setup: Some(MicrophoneSetup::StereoSpaced),
        sample_rate: Some(48000),
        license: Some("RainAI-FC-Proprietary-License".to_string()),
        author: Some("Bob".to_string()),
        notes: None,
        sha256: None,
        descriptions: vec!["Dense droplet impact on asphalt surface".to_string()],
        alternate_licenses: Vec::new(),
        contributors: vec!["Bob".to_string()],
    });

    let res =
        validate_manifest(&manifest, Some(&temp_dir), None).expect("Manifest validation failed");

    // Only 1 unique valid entry exists after content-addressed deduplication
    assert_eq!(res.valid_entries.len(), 1);
    assert_eq!(res.rejected_entries.len(), 0);

    // Duration is counted only once (2.0s, not 4.0s)
    assert!((1.9..=2.1).contains(&res.total_duration_secs));

    // Metadata is reconciled
    let reconciled = &res.valid_entries[0].source;
    assert_eq!(reconciled.tags, vec!["pavement", "asphalt"]);
    assert_eq!(
        reconciled.license.as_deref(),
        Some("RainAI-FC-Proprietary-License")
    );
    assert!(
        reconciled
            .alternate_licenses
            .contains(&"CC-BY 4.0".to_string())
    );
    assert_eq!(reconciled.contributors, vec!["Alice", "Bob"]);
    assert_eq!(reconciled.descriptions.len(), 2);

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_import_local_directory_content_addressed_deduplication() {
    let temp_dir =
        std::env::temp_dir().join(format!("rainai_test_import_dedup_{}", std::process::id()));
    let input_dir = temp_dir.join("inputs");
    let target_dir = temp_dir.join("dataset");

    let _ = fs::create_dir_all(&input_dir);
    let _ = fs::create_dir_all(&target_dir);

    let audio1_path = input_dir.join("source_rain_sample.wav");
    create_synthetic_rain_wav(&audio1_path, 2.0, 48000, 0.22, false);

    // 1. Initial Import
    let options1 = LocalImportOptions {
        default_tags: vec!["tin_roof".to_string()],
        default_rate: Some(PrecipitationRate::ModerateRain),
        default_mic: Some(MicrophoneSetup::StereoSpaced),
        default_license: Some("CC-BY 4.0".to_string()),
        author: Some("Recordist_A".to_string()),
        target_dir: target_dir.clone(),
        update_sources_json: false,
        update_attributions: false,
        thresholds: AudioQualityThresholds::default(),
    };

    let report1 = import_local_directory(&input_dir, &options1).expect("First import failed");
    assert_eq!(report1.successfully_imported, 1);

    // Verify 1 audio file and 1 manifest in target dir
    let audio_files_after_run1: Vec<_> = fs::read_dir(&target_dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    assert_eq!(audio_files_after_run1.len(), 1);

    // 2. Secondary Import with IDENTICAL audio data (same content/hash) under a different filename
    let duplicate_input_dir = temp_dir.join("duplicate_inputs");
    let _ = fs::create_dir_all(&duplicate_input_dir);
    let audio2_path = duplicate_input_dir.join("different_filename_identical_audio.wav");
    fs::copy(&audio1_path, &audio2_path).unwrap();

    let options2 = LocalImportOptions {
        default_tags: vec!["metal_surface".to_string(), "monsoon".to_string()],
        default_rate: Some(PrecipitationRate::HeavyRain),
        default_mic: Some(MicrophoneSetup::StereoSpaced),
        default_license: Some("RainAI-FC-Proprietary-License".to_string()),
        author: Some("Recordist_B".to_string()),
        target_dir: target_dir.clone(),
        update_sources_json: false,
        update_attributions: false,
        thresholds: AudioQualityThresholds::default(),
    };

    let report2 =
        import_local_directory(&duplicate_input_dir, &options2).expect("Second import failed");
    assert_eq!(report2.successfully_imported, 1);

    // CRITICAL: Binary storage was NOT duplicated! Still only 1 WAV file on disk!
    let audio_files_after_run2: Vec<_> = fs::read_dir(&target_dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    assert_eq!(audio_files_after_run2.len(), 1);

    // Inspect manifest_provenance.json in target dir
    let prov_path = target_dir.join("manifest_provenance.json");
    let content = fs::read_to_string(&prov_path).unwrap();
    let manifest: utilities::ingest::ProvenanceManifest = serde_json::from_str(&content).unwrap();

    // Still only 1 record, but reconciled
    assert_eq!(manifest.records.len(), 1);
    let record = &manifest.records[0];
    assert!(record.tags.contains(&"tin_roof".to_string()));
    assert!(record.tags.contains(&"metal_surface".to_string()));
    assert!(record.tags.contains(&"monsoon".to_string()));
    assert_eq!(
        record.license.as_deref(),
        Some("RainAI-FC-Proprietary-License")
    );
    assert!(record.alternate_licenses.contains(&"CC-BY 4.0".to_string()));
    assert!(record.contributors.contains(&"Recordist_A".to_string()));
    assert!(record.contributors.contains(&"Recordist_B".to_string()));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_sidecar_metadata_extraction() {
    let temp_dir = std::env::temp_dir().join(format!(
        "sidecar_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&temp_dir).unwrap();

    let audio_path = temp_dir.join("forest_rain.wav");
    create_synthetic_rain_wav(&audio_path, 1.0, 48000, 0.4, false);

    // 1. JSON sidecar
    let json_path = temp_dir.join("forest_rain.json");
    let json_content = serde_json::json!({
        "license": "CC0 1.0 Universal",
        "author": "FieldRecordist",
        "surface": "foliage",
        "tags": ["pine", "canopy"],
        "description": "Rain drops trickling through hemlock branches"
    });
    fs::write(&json_path, json_content.to_string()).unwrap();

    let meta = utilities::contribute::extract_sidecar_metadata(&audio_path)
        .expect("Extraction failed")
        .expect("No metadata returned");

    assert_eq!(meta.license.as_deref(), Some("CC0 1.0 Universal"));
    assert_eq!(meta.author.as_deref(), Some("FieldRecordist"));
    assert_eq!(meta.surface_tag.as_deref(), Some("foliage"));
    assert!(meta.tags.contains(&"foliage".to_string()));
    assert!(meta.tags.contains(&"pine".to_string()));
    assert_eq!(meta.descriptions.len(), 1);

    let _ = fs::remove_file(json_path);

    // 2. TXT sidecar
    let txt_path = temp_dir.join("forest_rain.txt");
    let txt_content = "license: CC-BY-4.0\nauthor: Alice\nsurface: tin_roof\ntags: metal, shed\nnotes: Heavy roof downpour";
    fs::write(&txt_path, txt_content).unwrap();

    let meta_txt = utilities::contribute::extract_sidecar_metadata(&audio_path)
        .expect("Extraction failed")
        .expect("No metadata returned");

    assert_eq!(meta_txt.license.as_deref(), Some("CC-BY-4.0"));
    assert_eq!(meta_txt.author.as_deref(), Some("Alice"));
    assert_eq!(meta_txt.surface_tag.as_deref(), Some("tin_roof"));
    assert!(meta_txt.tags.contains(&"tin_roof".to_string()));
    assert!(meta_txt.tags.contains(&"metal".to_string()));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_process_input_drop_folder_quarantine_routing() {
    let temp_dir = std::env::temp_dir().join(format!(
        "drop_quar_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let input_dir = temp_dir.join("input");
    let quar_dir = temp_dir.join("quarantine");
    let target_dir = temp_dir.join("target");
    let ledger_path = temp_dir.join("ledger.json");
    let prov_path = temp_dir.join("provenance.json");

    fs::create_dir_all(&input_dir).unwrap();

    // 1. Audio with NO license -> Quarantine
    let no_lic_path = input_dir.join("unlicensed_audio.wav");
    create_synthetic_rain_wav(&no_lic_path, 1.0, 48000, 0.4, false);

    // 2. Audio with Restricted / Non-commercial license -> Quarantine
    let nc_audio_path = input_dir.join("nc_restricted.wav");
    create_synthetic_rain_wav(&nc_audio_path, 1.0, 48000, 0.4, false);
    let nc_sidecar = input_dir.join("nc_restricted.json");
    fs::write(
        &nc_sidecar,
        serde_json::json!({
            "license": "CC-BY-NC-4.0", // NonCommercial not approved
            "author": "Someone"
        })
        .to_string(),
    )
    .unwrap();

    // 3. Corrupt/Silent audio with valid license -> Quarantine
    let silent_path = input_dir.join("silent_rain.wav");
    create_silent_wav(&silent_path, 1.0, 48000);
    let silent_sidecar = input_dir.join("silent_rain.json");
    fs::write(
        &silent_sidecar,
        serde_json::json!({
            "license": "CC0 1.0 Universal",
            "author": "Someone"
        })
        .to_string(),
    )
    .unwrap();

    // Run input drop processing
    let result = utilities::contribute::process_input_drop_folder(
        &input_dir,
        &quar_dir,
        &target_dir,
        &ledger_path,
        &prov_path,
    )
    .expect("Drop folder processing failed");

    assert_eq!(result.total_scanned, 3);
    assert_eq!(result.quarantined_count, 3);
    assert_eq!(result.standardized_count, 0);

    // Input dir should now be empty of those files
    assert!(!no_lic_path.exists());
    assert!(!nc_audio_path.exists());
    assert!(!silent_path.exists());

    // Quarantine dir should hold quarantined files and json records
    assert!(quar_dir.join("unlicensed_audio.wav").exists());
    assert!(quar_dir.join("unlicensed_audio_quarantine.json").exists());
    assert!(quar_dir.join("nc_restricted.wav").exists());
    assert!(quar_dir.join("nc_restricted_quarantine.json").exists());
    assert!(quar_dir.join("silent_rain.wav").exists());
    assert!(quar_dir.join("silent_rain_quarantine.json").exists());

    // Verify quarantine JSON contents
    let quar_meta: utilities::contribute::QuarantineRecord = serde_json::from_str(
        &fs::read_to_string(quar_dir.join("unlicensed_audio_quarantine.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(quar_meta.status, "QUARANTINE");
    assert!(
        quar_meta
            .quarantine_reason
            .contains("UNAPPROVED_OR_MISSING_LICENSE")
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_process_input_drop_folder_standardization_and_promotion() {
    let temp_dir = std::env::temp_dir().join(format!(
        "drop_std_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let input_dir = temp_dir.join("input");
    let quar_dir = temp_dir.join("quarantine");
    let target_dir = temp_dir.join("target");
    let ledger_path = temp_dir.join("ledger.json");
    let prov_path = temp_dir.join("provenance.json");

    fs::create_dir_all(&input_dir).unwrap();

    // Create a 44.1kHz mono rain file with CC-BY license in JSON sidecar
    let test_audio = input_dir.join("tin_shed_downpour.wav");
    create_synthetic_rain_wav(&test_audio, 2.0, 48000, 0.45, false);

    let sidecar = input_dir.join("tin_shed_downpour.json");
    fs::write(
        &sidecar,
        serde_json::json!({
            "license": "CC-BY-4.0",
            "author": "AcousticRecordist",
            "surface": "tin_roof",
            "tags": ["tin_roof", "corrugated_metal"],
            "descriptions": ["Loud rain droplets hitting tin shed"]
        })
        .to_string(),
    )
    .unwrap();

    let result = utilities::contribute::process_input_drop_folder(
        &input_dir,
        &quar_dir,
        &target_dir,
        &ledger_path,
        &prov_path,
    )
    .expect("Processing drop folder failed");

    assert_eq!(result.total_scanned, 1);
    assert_eq!(result.quarantined_count, 0);
    assert_eq!(result.standardized_count, 1);

    // Original file and sidecar removed from input
    assert!(!test_audio.exists());
    assert!(!sidecar.exists());

    // Standardized file exists in target directory
    assert_eq!(result.standardized_files.len(), 1);
    let std_file = &result.standardized_files[0];
    assert!(std_file.exists());

    // Verify target WAV specification: 48kHz stereo Float
    let reader = hound::WavReader::open(std_file).expect("Failed opening standardized WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 48000);
    assert_eq!(spec.channels, 2);

    // Verify Persistent Data Utility Ledger was created and populated
    let ledger = utilities::data_worker::DataUtilityLedger::load_or_create(&ledger_path);
    assert_eq!(ledger.records.len(), 1);
    let record = ledger.records.values().next().unwrap();
    assert_eq!(record.surface_tag, "tin_roof");
    assert!(record.acoustic_quality_score > 0.0);
    assert!(record.current_utility_score > 0.0);
    assert!(record.is_cached_locally);

    // Verify Provenance Manifest was created and populated
    let prov: utilities::ingest::ProvenanceManifest =
        serde_json::from_str(&fs::read_to_string(&prov_path).unwrap()).unwrap();
    assert_eq!(prov.records.len(), 1);
    assert_eq!(prov.records[0].license.as_deref(), Some("CC-BY-4.0"));
    assert!(prov.records[0].tags.contains(&"tin_roof".to_string()));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_smooth_rolling_quota_eviction() {
    let temp_dir = std::env::temp_dir().join(format!(
        "smooth_quota_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&temp_dir).unwrap();

    let manifest_path = temp_dir.join("manifest.json");
    let ledger_path = utilities::data_worker::ledger_path_for_manifest(&manifest_path);

    let mut manifest_entries: std::collections::HashMap<
        String,
        utilities::features::AudioMetadata,
    > = std::collections::HashMap::new();
    let mut ledger = utilities::data_worker::DataUtilityLedger::default();

    // Create 10 synthetic files
    for i in 0..10 {
        let chunk_id = format!("chunk_{:03}", i);
        let wav_path = temp_dir.join(format!("{}.wav", chunk_id));
        create_synthetic_rain_wav(&wav_path, 1.0, 48000, 0.4, false);

        let q_score = (i as f32 + 1.0) / 10.0; // Increasing quality scores
        let u_score = q_score;

        manifest_entries.insert(
            chunk_id.clone(),
            utilities::features::AudioMetadata {
                path: wav_path.display().to_string(),
                filename: format!("{}.wav", chunk_id),
                sample_rate: 48000,
                channels: 2,
                duration_secs: 1.0,
                rms_energy: 0.1,
                rain_rate: 20.0,
                droplet_density: 0.5,
                drops_per_second: 200.0,
                high_freq_ratio: 0.2,
                spectral_centroid: 3000.0,
                spectral_rolloff: 6000.0,
                spectral_flatness: 0.3,
                surface_tag: "tin_roof".to_string(),
            },
        );

        ledger.upsert_record(utilities::data_worker::DataChunkUtilityRecord {
            chunk_id: chunk_id.clone(),
            sha256: format!("sha_{}", i),
            source_id: format!("{}.wav", chunk_id),
            surface_tag: "tin_roof".to_string(),
            material_properties: [0.5; 7],
            acoustic_quality_score: q_score,
            loss_ema: 1.0,
            gradient_norm_ema: 1.0,
            times_trained: 0,
            last_trained_timestamp: 0,
            information_novelty: 1.0,
            current_utility_score: u_score,
            retraining_priority: u_score,
            is_cached_locally: true,
        });
    }

    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest_entries).unwrap(),
    )
    .unwrap();
    ledger.save_to_file(&ledger_path).unwrap();

    let total_size = utilities::data_worker::DatabaseHealthWorker::calculate_dir_size(&temp_dir);
    // Set max_bytes such that soft ceiling (90%) is exceeded, and evict with batch limit 3
    let artificial_max_bytes = (total_size as f64 * 0.8) as u64;

    let quotas = vec![utilities::autopilot::SurfaceQuota {
        surface: "tin_roof".to_string(),
        count: 10,
        deficit_count: 0,
        proportion: 1.0,
        target_proportion: 0.11,
    }];

    let evicted = utilities::data_worker::DatabaseHealthWorker::enforce_rolling_quota_smooth(
        &temp_dir,
        &manifest_path,
        &quotas,
        artificial_max_bytes,
        3, // Small chunk batch limit
    );

    assert_eq!(
        evicted, 3,
        "Smooth eviction respected chunk batch limit of 3"
    );

    // Re-load ledger to verify score history was preserved while marked evicted
    let reloaded_ledger = utilities::data_worker::DataUtilityLedger::load_or_create(&ledger_path);
    assert_eq!(
        reloaded_ledger.records.len(),
        10,
        "All 10 score records preserved in ledger"
    );
    let evicted_records: Vec<_> = reloaded_ledger
        .records
        .values()
        .filter(|r| !r.is_cached_locally)
        .collect();
    assert_eq!(
        evicted_records.len(),
        3,
        "Exactly 3 records marked evicted in ledger"
    );

    // The 3 lowest quality chunks (chunk_000, chunk_001, chunk_002) should have been evicted
    assert!(evicted_records.iter().any(|r| r.chunk_id == "chunk_000"));
    assert!(evicted_records.iter().any(|r| r.chunk_id == "chunk_001"));
    assert!(evicted_records.iter().any(|r| r.chunk_id == "chunk_002"));

    let _ = fs::remove_dir_all(&temp_dir);
}
