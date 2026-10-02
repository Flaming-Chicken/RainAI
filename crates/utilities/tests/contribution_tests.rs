//! Integration tests for standardized community rain data contribution, validation,
//! acoustic quality screening, and local dataset directory ingestion.

use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs;
use std::path::{Path, PathBuf};
use utilities::contribute::{
    generate_manifest_template, import_local_directory, validate_audio_file, validate_manifest,
    AudioQualityThresholds, CanonicalSurface, LocalImportOptions, MicrophoneSetup,
    PrecipitationRate, RainContributionManifest, RainSourceEntry,
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
    assert_eq!(template.default_license, "CC0 1.0 Universal");

    // Verify template contains multiple canonical surfaces
    let surfaces: Vec<CanonicalSurface> = template.sources.iter().map(|s| s.surface).collect();
    assert!(surfaces.contains(&CanonicalSurface::TinRoof));
    assert!(surfaces.contains(&CanonicalSurface::Foliage));
    assert!(surfaces.contains(&CanonicalSurface::Glass));

    // Roundtrip JSON serialization
    let json = template.to_json_pretty().expect("Failed to serialize template");
    let roundtrip = RainContributionManifest::from_json(&json).expect("Failed to deserialize");

    assert_eq!(roundtrip.dataset_name, template.dataset_name);
    assert_eq!(roundtrip.sources.len(), template.sources.len());
    assert_eq!(roundtrip.sources[0].id, template.sources[0].id);
    assert_eq!(roundtrip.sources[0].surface, template.sources[0].surface);
}

#[test]
fn test_license_and_quality_screening_in_manifest_validation() {
    let mut manifest = RainContributionManifest::new("Test Contribution", "Researcher", "CC0");

    // Approved license entry (remote URL)
    manifest.add_source(RainSourceEntry {
        id: "valid_cc0_src".to_string(),
        file_path: "remote_rain.wav".to_string(),
        url: Some("https://example.org/rain.wav".to_string()),
        surface: CanonicalSurface::Pavement,
        precipitation_rate: PrecipitationRate::ModerateRain,
        environment: Some("Sidewalk".to_string()),
        microphone_setup: MicrophoneSetup::StereoSpaced,
        sample_rate: Some(48000),
        license: "CC0 1.0 Universal".to_string(),
        author: "Alice".to_string(),
        notes: None,
        sha256: None,
    });

    // Rejected license entry (NonCommercial clause)
    manifest.add_source(RainSourceEntry {
        id: "rejected_nc_src".to_string(),
        file_path: "nc_rain.wav".to_string(),
        url: Some("https://example.org/nc.wav".to_string()),
        surface: CanonicalSurface::Asphalt,
        precipitation_rate: PrecipitationRate::HeavyRain,
        environment: Some("Street".to_string()),
        microphone_setup: MicrophoneSetup::Mono,
        sample_rate: Some(48000),
        license: "CC-BY-NC 4.0".to_string(),
        author: "Bob".to_string(),
        notes: None,
        sha256: None,
    });

    let res = validate_manifest(&manifest, None, None).expect("Validation execution failed");

    assert_eq!(res.valid_entries.len(), 1);
    assert_eq!(res.rejected_entries.len(), 1);
    assert_eq!(res.valid_entries[0].source.id, "valid_cc0_src");
    assert_eq!(res.rejected_entries[0].source_id, "rejected_nc_src");
    assert!(res.rejected_entries[0].rejection_reason.contains("NonCommercial"));
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
    let (metrics, sha, duration) =
        validate_audio_file(&wav_path, &thresholds).expect("Valid rain audio rejected unexpectedly");

    assert!((2.9..=3.1).contains(&duration));
    assert_eq!(metrics.sample_rate, 48000);
    assert_eq!(metrics.channels, 2);
    assert!(metrics.rms_energy > thresholds.min_rms_energy);
    assert!(metrics.clipping_ratio <= thresholds.max_clipping_ratio);
    assert!(metrics.high_freq_ratio >= thresholds.min_high_freq_ratio);
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

    // 3. Too short file (0.5s < 2.0s min)
    let short_path = temp_dir.join("too_short.wav");
    create_synthetic_rain_wav(&short_path, 0.5, 48000, 0.25, false);
    let short_res = validate_audio_file(&short_path, &thresholds);
    assert!(short_res.is_err());
    let err_msg3 = short_res.unwrap_err().to_string();
    assert!(err_msg3.contains("Audio too short"));

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
    create_synthetic_rain_wav(&input_dir.join("rain_tin_roof_porch.wav"), 2.5, 48000, 0.20, false);
    create_synthetic_rain_wav(&input_dir.join("heavy_deluge_asphalt_road.wav"), 2.5, 48000, 0.22, false);
    create_synthetic_rain_wav(&input_dir.join("window_glass_patter.wav"), 2.5, 48000, 0.18, false);

    // Create 1 invalid silent file that should be rejected
    create_silent_wav(&input_dir.join("bad_silent_recording.wav"), 2.5, 48000);

    let options = LocalImportOptions {
        default_surface: None, // Auto-infer from filenames!
        default_rate: PrecipitationRate::ModerateRain,
        default_mic: MicrophoneSetup::StereoSpaced,
        default_license: "CC0 1.0 Universal".to_string(),
        author: "Alice Cooper".to_string(),
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

    // Verify auto-inference correctly identified TinRoof, Asphalt, and Glass surfaces
    assert!(report.surface_distribution.contains_key("tin_roof"));
    assert!(report.surface_distribution.contains_key("asphalt"));
    assert!(report.surface_distribution.contains_key("glass"));

    // Verify files were physically copied to target directory
    let target_files: Vec<PathBuf> = fs::read_dir(&target_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(target_files.len(), 3);

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_precipitation_rate_and_mic_setup_tagging() {
    assert_eq!(PrecipitationRate::from_tag("misty_drizzle"), PrecipitationRate::Drizzle);
    assert_eq!(PrecipitationRate::from_tag("light_shower"), PrecipitationRate::LightRain);
    assert_eq!(PrecipitationRate::from_tag("heavy_downpour"), PrecipitationRate::HeavyRain);
    assert_eq!(PrecipitationRate::from_tag("thunder_storm"), PrecipitationRate::ViolentStorm);
    assert_eq!(PrecipitationRate::from_tag("regular_rain"), PrecipitationRate::ModerateRain);

    assert_eq!(MicrophoneSetup::from_tag("in_ear_binaural_mics"), MicrophoneSetup::BinauralInEar);
    assert_eq!(MicrophoneSetup::from_tag("ambisonic_b_format"), MicrophoneSetup::AmbisonicFoa);
    assert_eq!(MicrophoneSetup::from_tag("higher_order_hoa_16ch"), MicrophoneSetup::AmbisonicHoa);
    assert_eq!(MicrophoneSetup::from_tag("underwater_hydrophone"), MicrophoneSetup::Hydrophone);
    assert_eq!(MicrophoneSetup::from_tag("piezo_contact_disc"), MicrophoneSetup::ContactMic);
    assert_eq!(MicrophoneSetup::from_tag("spaced_pair_omnis"), MicrophoneSetup::StereoSpaced);
}
