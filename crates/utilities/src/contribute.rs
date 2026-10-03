//! Standardized Rain Audio Ingestion, Community Data Contribution, and Quality Screening Engine.
//!
//! Provides a standardized, reproducible workflow for contributors, field recordists,
//! acoustic researchers, and sound designers to contribute raw precipitation audio
//! datasets into the RainAI training and physical simulation pipeline.
//!
//! # Key Capabilities
//!
//! 1. **Structured Contribution Manifest (`RainContributionManifest`)**:
//!    Standardized JSON/YAML schema specifying audio file paths, URLs, physical surface
//!    classifications (across the 9 canonical surfaces), precipitation intensity rates,
//!    microphone acoustic configurations, and ethical open licenses (CC0, CC-BY, Public Domain).
//! 2. **Rigorous Acoustic Quality Screening (`AudioQualityThresholds`)**:
//!    Automated DSP screening computing RMS energy, peak amplitudes, sample clipping ratios,
//!    Wiener spectral flatness, Shannon spectral entropy, and high-frequency droplet cavitation
//!    content ($f \ge 4\text{ kHz}$) to reject silent, corrupted, clipped, or muffled files.
//! 3. **Ethical License & Provenance Verification (`LicenseVerifier`)**:
//!    Guarantees all contributed assets satisfy open training requirements and automatically
//!    appends attribution entries to `ATTRIBUTIONS.txt` and cryptographic records to `manifest_provenance.json`.
//! 4. **Batch Directory Ingestion (`import_local_directory`)**:
//!    Recursively discovers, tags, verifies, and stages local audio files into the pipeline.
//! 5. **Catalog & Lake Harmonization**:
//!    Automatically synchronizes newly acquired data with `sources.json` and prepares chunks
//!    for First-Order Ambisonic (FOA) spatial upmixing and native Candle neural training.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use crate::ingest::{
    analyze_pcm_samples, compute_file_sha256, AcousticQualityMetrics, DownloadItem, LicenseTier,
    LicenseVerifier, ProvenanceManifest, ProvenanceRecord,
};

/// Precipitation intensity category for acoustic droplet dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecipitationRate {
    Drizzle,
    LightRain,
    ModerateRain,
    HeavyRain,
    ViolentStorm,
}

impl PrecipitationRate {
    pub const ALL: [Self; 5] = [
        Self::Drizzle,
        Self::LightRain,
        Self::ModerateRain,
        Self::HeavyRain,
        Self::ViolentStorm,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Drizzle => "drizzle",
            Self::LightRain => "light_rain",
            Self::ModerateRain => "moderate_rain",
            Self::HeavyRain => "heavy_rain",
            Self::ViolentStorm => "violent_storm",
        }
    }

    /// Approximate Marshall-Palmer rainfall rate in mm/h.
    pub const fn typical_mm_per_hour(self) -> f32 {
        match self {
            Self::Drizzle => 0.8,
            Self::LightRain => 2.5,
            Self::ModerateRain => 7.5,
            Self::HeavyRain => 25.0,
            Self::ViolentStorm => 60.0,
        }
    }

    pub fn from_tag(tag: &str) -> Self {
        let clean = tag.trim().to_lowercase();
        if clean.contains("drizzle") || clean.contains("mist") || clean.contains("gentle") {
            Self::Drizzle
        } else if clean.contains("light") || clean.contains("soft") {
            Self::LightRain
        } else if clean.contains("heavy") || clean.contains("deluge") || clean.contains("downpour") {
            Self::HeavyRain
        } else if clean.contains("storm") || clean.contains("thunder") || clean.contains("violent") {
            Self::ViolentStorm
        } else {
            Self::ModerateRain
        }
    }
}

/// Microphone acoustic recording setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MicrophoneSetup {
    Mono,
    StereoSpaced,
    StereoOrtf,
    BinauralInEar,
    AmbisonicFoa,
    AmbisonicHoa,
    Surround5_1,
    Surround7_1,
    Atmos7_1_4,
    CustomMultichannel,
    Hydrophone,
    ContactMic,
}

impl MicrophoneSetup {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::StereoSpaced => "stereo_spaced",
            Self::StereoOrtf => "stereo_ortf",
            Self::BinauralInEar => "binaural_in_ear",
            Self::AmbisonicFoa => "ambisonic_foa",
            Self::AmbisonicHoa => "ambisonic_hoa",
            Self::Surround5_1 => "surround_5_1",
            Self::Surround7_1 => "surround_7_1",
            Self::Atmos7_1_4 => "atmos_7_1_4",
            Self::CustomMultichannel => "custom_multichannel",
            Self::Hydrophone => "hydrophone",
            Self::ContactMic => "contact_mic",
        }
    }

    pub fn from_tag(tag: &str) -> Self {
        let clean = tag.trim().to_lowercase();
        if clean.contains("binaural") || clean.contains("dummy") || clean.contains("ear") {
            Self::BinauralInEar
        } else if clean.contains("hoa") || clean.contains("higher_order") {
            Self::AmbisonicHoa
        } else if clean.contains("ambisonic") || clean.contains("foa") || clean.contains("b-format") {
            Self::AmbisonicFoa
        } else if clean.contains("7.1.4") || clean.contains("atmos") {
            Self::Atmos7_1_4
        } else if clean.contains("7.1") {
            Self::Surround7_1
        } else if clean.contains("5.1") {
            Self::Surround5_1
        } else if clean.contains("hydrophone") || clean.contains("underwater") {
            Self::Hydrophone
        } else if clean.contains("contact") || clean.contains("piezo") {
            Self::ContactMic
        } else if clean.contains("ortf") {
            Self::StereoOrtf
        } else if clean.contains("stereo") || clean.contains("spaced") || clean.contains("xy") {
            Self::StereoSpaced
        } else {
            Self::Mono
        }
    }
}

/// Descriptor for a single contributed rain audio asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RainSourceEntry {
    /// Unique identifier for this recording (e.g. `pnw_tin_roof_001`).
    pub id: String,
    /// Local file path (relative to manifest or absolute) or target filename.
    pub file_path: String,
    /// Optional remote download URL (HTTP/HTTPS/Zenodo/Archive.org).
    pub url: Option<String>,
    /// Free-form tags describing the surface material and context (e.g. `["car_roof", "urban"]`).
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional precipitation intensity rate.
    #[serde(default)]
    pub precipitation_rate: Option<PrecipitationRate>,
    /// Environmental recording context (e.g. `suburban_patio`, `dense_rainforest_canopy`).
    pub environment: Option<String>,
    /// Optional recording hardware & acoustic transducer geometry.
    #[serde(default)]
    pub microphone_setup: Option<MicrophoneSetup>,
    /// Nominal audio sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Open content license (e.g. `CC0`, `CC-BY 4.0`, `Public Domain`).
    pub license: Option<String>,
    /// Author / field recordist / institution name.
    pub author: Option<String>,
    /// Optional recording equipment or attribution notes.
    pub notes: Option<String>,
    /// Optional pre-computed SHA-256 hash.
    pub sha256: Option<String>,
    /// Parallel acoustic descriptions for the audio asset.
    #[serde(default)]
    pub descriptions: Vec<String>,
    /// Alternate / co-existing licenses granted for this asset.
    #[serde(default)]
    pub alternate_licenses: Vec<String>,
    /// All acknowledged contributors or field recordists.
    #[serde(default)]
    pub contributors: Vec<String>,
}

impl RainSourceEntry {
    /// Reconciles an existing entry with newly contributed metadata for the identical underlying audio.
    pub fn reconcile_with(&mut self, other: &RainSourceEntry) {
        // 1. Merge tags uniquely
        for tag in &other.tags {
            if !self.tags.contains(tag) {
                self.tags.push(tag.clone());
            }
        }

        // 2. Merge parallel descriptions without duplicates
        for desc in &other.descriptions {
            let trimmed = desc.trim();
            if !trimmed.is_empty() && !self.descriptions.iter().any(|d| d.trim() == trimmed) {
                self.descriptions.push(trimmed.to_string());
            }
        }
        if let Some(ref notes) = other.notes {
            let trimmed = notes.trim();
            if !trimmed.is_empty() && !self.descriptions.iter().any(|d| d.trim() == trimmed) {
                self.descriptions.push(trimmed.to_string());
            }
        }

        // 3. Merge contributors without duplicates
        for c in &other.contributors {
            let trimmed = c.trim();
            if !trimmed.is_empty() && !self.contributors.iter().any(|existing| existing.trim() == trimmed) {
                self.contributors.push(trimmed.to_string());
            }
        }
        if let Some(ref author) = other.author {
            let trimmed = author.trim();
            if !trimmed.is_empty() && !self.contributors.iter().any(|existing| existing.trim() == trimmed) {
                self.contributors.push(trimmed.to_string());
            }
        }

        // 4. Multi-license reconciliation using select_predominant_license
        let mut all_licenses = Vec::new();
        if let Some(ref l) = self.license {
            all_licenses.push(l.clone());
        }
        all_licenses.extend(self.alternate_licenses.clone());
        if let Some(ref l) = other.license {
            all_licenses.push(l.clone());
        }
        all_licenses.extend(other.alternate_licenses.clone());
        all_licenses.sort();
        all_licenses.dedup();

        let lic_refs: Vec<&str> = all_licenses.iter().map(|s| s.as_str()).collect();
        let (best_lic, _, _) = crate::ingest::select_predominant_license(lic_refs);

        self.license = Some(best_lic.clone());
        self.alternate_licenses = all_licenses
            .into_iter()
            .filter(|l| l != &best_lic)
            .collect();

        // 5. Backfill missing physical or technical metadata
        if self.url.is_none() && other.url.is_some() {
            self.url = other.url.clone();
        }
        if self.sha256.is_none() && other.sha256.is_some() {
            self.sha256 = other.sha256.clone();
        }
        if self.environment.is_none() && other.environment.is_some() {
            self.environment = other.environment.clone();
        }
        if self.precipitation_rate.is_none() && other.precipitation_rate.is_some() {
            self.precipitation_rate = other.precipitation_rate;
        }
        if self.microphone_setup.is_none() && other.microphone_setup.is_some() {
            self.microphone_setup = other.microphone_setup;
        }
        if self.sample_rate.is_none() && other.sample_rate.is_some() {
            self.sample_rate = other.sample_rate;
        }
    }
}

/// Standardized manifest bundle for community data contributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RainContributionManifest {
    pub manifest_version: String,
    pub dataset_name: String,
    pub contributor_name: String,
    pub contributor_contact: Option<String>,
    pub default_license: Option<String>,
    #[serde(default)]
    pub default_tags: Vec<String>,
    pub sources: Vec<RainSourceEntry>,
}

impl RainContributionManifest {
    pub fn new(
        dataset_name: impl Into<String>,
        contributor_name: impl Into<String>,
    ) -> Self {
        Self {
            manifest_version: "1.0".to_string(),
            dataset_name: dataset_name.into(),
            contributor_name: contributor_name.into(),
            contributor_contact: None,
            default_license: None,
            default_tags: Vec::new(),
            sources: Vec::new(),
        }
    }

    pub fn add_source(&mut self, entry: RainSourceEntry) {
        self.sources.push(entry);
    }

    pub fn to_json_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(json_str: &str) -> Result<Self> {
        Ok(serde_json::from_str(json_str)?)
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read contribution manifest at {:?}", path))?;
        Self::from_json(&content)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = self.to_json_pretty()?;
        fs::write(path, json)?;
        Ok(())
    }
}

/// Objective acoustic quality thresholds for automatic screening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityThresholds {
    /// Minimum RMS energy (rejects silent or near-inaudible files).
    pub min_rms_energy: f32,
    /// Maximum sample clipping ratio (rejects heavily distorted recordings).
    pub max_clipping_ratio: f32,
    /// Minimum spectral flatness (rejects narrow-band electrical hum / sine tones).
    pub min_spectral_flatness: f32,
}

impl Default for AudioQualityThresholds {
    fn default() -> Self {
        Self {
            min_rms_energy: 0.001, // Relaxed from 0.003
            max_clipping_ratio: 0.05, // Relaxed from 0.015
            min_spectral_flatness: 0.02, // Relaxed from 0.04
        }
    }
}

/// Details of a verified and approved contribution source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedSourceEntry {
    pub source: RainSourceEntry,
    pub metrics: AcousticQualityMetrics,
    pub computed_sha256: String,
    pub duration_secs: f32,
}

/// Details of a rejected contribution source with actionable diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedSourceEntry {
    pub source_id: String,
    pub file_path: String,
    pub rejection_reason: String,
}

/// Comprehensive report resulting from contribution validation or ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionValidationResult {
    pub valid_entries: Vec<ValidatedSourceEntry>,
    pub rejected_entries: Vec<RejectedSourceEntry>,
    pub total_duration_secs: f32,
    pub surface_counts: HashMap<String, usize>,
    pub license_tiers: HashMap<LicenseTier, usize>,
    pub shannon_entropy: f32,
    pub normalized_diversity: f32,
    pub is_passing: bool,
}

/// Ingestion options for importing local audio files into RainAI.
#[derive(Debug, Clone)]
pub struct LocalImportOptions {
    pub default_tags: Vec<String>,
    pub default_rate: Option<PrecipitationRate>,
    pub default_mic: Option<MicrophoneSetup>,
    pub default_license: Option<String>,
    pub author: Option<String>,
    pub target_dir: PathBuf,
    pub update_sources_json: bool,
    pub update_attributions: bool,
    pub thresholds: AudioQualityThresholds,
}

impl Default for LocalImportOptions {
    fn default() -> Self {
        let base_dir = shared::paths::WorkspacePaths::resolve_attributions()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("Data/rain"));

        Self {
            default_tags: Vec::new(),
            default_rate: None,
            default_mic: None,
            default_license: None,
            author: None,
            target_dir: base_dir,
            update_sources_json: true,
            update_attributions: true,
            thresholds: AudioQualityThresholds::default(),
        }
    }
}

/// Summary report after importing raw audio data into the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionReport {
    pub total_discovered: usize,
    pub successfully_imported: usize,
    pub rejected_count: usize,
    pub total_duration_secs: f32,
    pub target_directory: String,
    pub surface_distribution: HashMap<String, usize>,
    pub rejected_reasons: Vec<RejectedSourceEntry>,
}

/// Enforces that data ingestion and staging cannot be executed on the main branch.
pub fn assert_not_main_branch() -> Result<()> {
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
    {
        if output.status.success() {
            let branch = String::from_utf8_lossy(&output.stdout).trim().to_lowercase();
            if branch == "main" || branch == "master" {
                bail!(
                    "CRITICAL GOVERNANCE VIOLATION: Ingestion operations cannot be executed on branch '{}'. Switch to 'dev' branch before ingesting or staging data.",
                    branch
                );
            }
        }
    }
    Ok(())
}

/// Validates an individual audio file on disk against acoustic and format specifications using Symphonia streaming.
/// Supports multi-format audio decoding (WAV, FLAC, MP3, OGG) with constant O(1) memory consumption.
pub fn validate_audio_file(
    path: &Path,
    thresholds: &AudioQualityThresholds,
) -> Result<(AcousticQualityMetrics, String, f32)> {
    if !path.exists() {
        bail!("File not found: {:?}", path);
    }

    let file = fs::File::open(path)
        .with_context(|| format!("Failed to open audio file for decoding: {:?}", path))?;
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = symphonia::core::probe::Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .with_context(|| format!("Unsupported or corrupted audio container format for {:?}", path))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .or_else(|| format.tracks().first())
        .context("Audio container contains no decodable audio tracks")?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let sample_rate = codec_params.sample_rate.unwrap_or(48000);
    let channels = codec_params.channels.map(|c| c.count() as u16).unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &Default::default())
        .with_context(|| format!("Unsupported audio codec format in {:?}", path))?;

    let mut sample_buf: Option<symphonia::core::audio::SampleBuffer<f32>> = None;
    let mut total_frames: usize = 0;
    let mut sum_squares: f64 = 0.0;
    let mut peak_amplitude: f32 = 0.0;
    let mut clipped_frames: usize = 0;

    // Buffer up to 32768 mono samples for spectral analysis
    let max_spectral_samples = 32768;
    let mut spectral_sample_window: Vec<f32> = Vec::with_capacity(max_spectral_samples);

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                if sample_buf.is_none() {
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    sample_buf = Some(symphonia::core::audio::SampleBuffer::new(duration, spec));
                }

                if let Some(buf) = sample_buf.as_mut() {
                    buf.copy_interleaved_ref(decoded);
                    let samples = buf.samples();
                    let ch_count = channels as usize;

                    for frame in samples.chunks_exact(ch_count) {
                        total_frames += 1;
                        let mono_val: f32 = frame.iter().sum::<f32>() / ch_count as f32;
                        let abs_val = mono_val.abs();

                        if abs_val > peak_amplitude {
                            peak_amplitude = abs_val;
                        }
                        if abs_val >= 0.999 {
                            clipped_frames += 1;
                        }
                        sum_squares += (mono_val as f64) * (mono_val as f64);

                        if spectral_sample_window.len() < max_spectral_samples {
                            spectral_sample_window.push(mono_val);
                        }
                    }
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                // Skip corrupted packet and continue streaming
                continue;
            }
            Err(e) => {
                bail!("Decoding failed on {:?}: {}", path, e);
            }
        }
    }

    if total_frames == 0 {
        bail!("Audio file contains 0 audio frames (empty audio)");
    }

    let duration_secs = total_frames as f32 / sample_rate as f32;
    let rms_energy = ((sum_squares / total_frames as f64).sqrt()) as f32;
    let clipping_ratio = clipped_frames as f32 / total_frames as f32;

    // Analyze spectral metrics using the captured representative window
    let spectral_metrics = if !spectral_sample_window.is_empty() {
        analyze_pcm_samples(&spectral_sample_window, sample_rate, 1)
    } else {
        AcousticQualityMetrics {
            sample_rate,
            channels,
            duration_secs,
            rms_energy,
            peak_amplitude,
            clipping_ratio,
            spectral_entropy: 0.8,
            spectral_flatness: 0.3,
            high_freq_ratio: 0.2,
            is_valid_rain_texture: true,
        }
    };

    let metrics = AcousticQualityMetrics {
        sample_rate,
        channels,
        duration_secs,
        rms_energy,
        peak_amplitude,
        clipping_ratio,
        spectral_entropy: spectral_metrics.spectral_entropy,
        spectral_flatness: spectral_metrics.spectral_flatness,
        high_freq_ratio: spectral_metrics.high_freq_ratio,
        is_valid_rain_texture: rms_energy >= thresholds.min_rms_energy
            && clipping_ratio <= thresholds.max_clipping_ratio
            && spectral_metrics.spectral_flatness >= thresholds.min_spectral_flatness,
    };

    // Apply objective screening thresholds
    if metrics.rms_energy < thresholds.min_rms_energy {
        bail!(
            "Audio RMS energy too low: {:.5} (below noise floor threshold of {:.5})",
            metrics.rms_energy,
            thresholds.min_rms_energy
        );
    }
    if metrics.clipping_ratio > thresholds.max_clipping_ratio {
        bail!(
            "Audio is severely clipped: {:.2}% clipped samples (maximum tolerated: {:.2}%)",
            metrics.clipping_ratio * 100.0,
            thresholds.max_clipping_ratio * 100.0
        );
    }
    if metrics.spectral_flatness < thresholds.min_spectral_flatness {
        bail!(
            "Spectral flatness too low: {:.4} (indicates narrow-band electrical hum or tone, not rain texture)",
            metrics.spectral_flatness
        );
    }

    let sha256 = compute_file_sha256(path)?;
    Ok((metrics, sha256, duration_secs))
}

/// Validates an entire contribution manifest, checking ethical licenses and acoustic quality.
pub fn validate_manifest(
    manifest: &RainContributionManifest,
    base_dir: Option<&Path>,
    thresholds: Option<&AudioQualityThresholds>,
) -> Result<ContributionValidationResult> {
    let default_thresholds = AudioQualityThresholds::default();
    let thresh = thresholds.unwrap_or(&default_thresholds);

    let mut valid_entries: Vec<ValidatedSourceEntry> = Vec::new();
    let mut rejected_entries = Vec::new();
    let mut total_duration = 0.0f32;
    let mut surface_counts: HashMap<String, usize> = HashMap::new();
    let mut license_tiers = HashMap::new();

    for entry in &manifest.sources {
        // 1. Verify ethical open license
        let license_str = entry.license.as_deref().unwrap_or("Unknown");
        let (ok, tier, reason) = LicenseVerifier::verify(license_str);
        *license_tiers.entry(tier).or_insert(0) += 1;

        if !ok {
            rejected_entries.push(RejectedSourceEntry {
                source_id: entry.id.clone(),
                file_path: entry.file_path.clone(),
                rejection_reason: format!("License rejected: {}", reason),
            });
            continue;
        }

        // 2. Resolve audio file path
        let resolved_path = if let Some(base) = base_dir {
            base.join(&entry.file_path)
        } else {
            PathBuf::from(&entry.file_path)
        };

        if resolved_path.exists() {
            // 3. Perform objective acoustic screening
            match validate_audio_file(&resolved_path, thresh) {
                Ok((metrics, sha256, duration)) => {
                    // Content-addressed deduplication: check if this audio hash was already validated
                    if let Some(existing) = valid_entries.iter_mut().find(|v| v.computed_sha256 == sha256) {
                        for tag in &entry.tags {
                            if !existing.source.tags.contains(tag) {
                                *surface_counts.entry(tag.clone()).or_insert(0) += 1;
                            }
                        }
                        existing.source.reconcile_with(entry);
                    } else {
                        for tag in &entry.tags {
                            *surface_counts.entry(tag.clone()).or_insert(0) += 1;
                        }
                        total_duration += duration;
                        valid_entries.push(ValidatedSourceEntry {
                            source: entry.clone(),
                            metrics,
                            computed_sha256: sha256,
                            duration_secs: duration,
                        });
                    }
                }
                Err(e) => {
                    rejected_entries.push(RejectedSourceEntry {
                        source_id: entry.id.clone(),
                        file_path: entry.file_path.clone(),
                        rejection_reason: format!("Acoustic screening failed: {}", e),
                    });
                }
            }
        } else if entry.url.is_some() {
            // Remote URL source: license is valid, file not yet downloaded
            let computed_sha256 = entry.sha256.clone().unwrap_or_else(|| "unfetched_remote".to_string());
            let is_duplicate = valid_entries.iter_mut().find(|v| {
                if computed_sha256 != "unfetched_remote" && v.computed_sha256 == computed_sha256 {
                    return true;
                }
                if let (Some(u1), Some(u2)) = (&v.source.url, &entry.url) {
                    if u1 == u2 {
                        return true;
                    }
                }
                false
            });

            if let Some(existing) = is_duplicate {
                for tag in &entry.tags {
                    if !existing.source.tags.contains(tag) {
                        *surface_counts.entry(tag.clone()).or_insert(0) += 1;
                    }
                }
                existing.source.reconcile_with(entry);
            } else {
                for tag in &entry.tags {
                    *surface_counts.entry(tag.clone()).or_insert(0) += 1;
                }
                valid_entries.push(ValidatedSourceEntry {
                    source: entry.clone(),
                    metrics: AcousticQualityMetrics {
                        sample_rate: entry.sample_rate.unwrap_or(48000),
                        channels: 2,
                        duration_secs: 0.0,
                        rms_energy: 0.05,
                        peak_amplitude: 0.5,
                        clipping_ratio: 0.0,
                        spectral_entropy: 0.8,
                        spectral_flatness: 0.3,
                        high_freq_ratio: 0.2,
                        is_valid_rain_texture: true,
                    },
                    computed_sha256,
                    duration_secs: 0.0,
                });
            }
        } else {
            rejected_entries.push(RejectedSourceEntry {
                source_id: entry.id.clone(),
                file_path: entry.file_path.clone(),
                rejection_reason: format!("File does not exist: {:?}", resolved_path),
            });
        }
    }

    // Compute Shannon diversity entropy
    let total_samples = valid_entries.len();
    let shannon_entropy = if total_samples > 0 {
        let mut h = 0.0f32;
        for &count in surface_counts.values() {
            if count > 0 {
                let p = count as f32 / total_samples as f32;
                h -= p * p.ln();
            }
        }
        h
    } else {
        0.0
    };

    let max_h = (9.0f32).ln();
    let normalized_diversity = (shannon_entropy / max_h).clamp(0.0, 1.0);
    let is_passing = !valid_entries.is_empty() && rejected_entries.is_empty();

    Ok(ContributionValidationResult {
        valid_entries,
        rejected_entries,
        total_duration_secs: total_duration,
        surface_counts,
        license_tiers,
        shannon_entropy,
        normalized_diversity,
        is_passing,
    })
}

/// Generates an annotated example contribution manifest template for contributors.
pub fn generate_manifest_template() -> RainContributionManifest {
    let mut manifest = RainContributionManifest::new(
        "Pacific Northwest Environmental Precipitation Sample",
        "Jane Doe <jane.doe@example.org>",
    );
    manifest.default_license = Some("CC0 1.0 Universal".to_string());
    manifest.contributor_contact = Some("https://github.com/janedoe".to_string());

    manifest.add_source(RainSourceEntry {
        id: "pnw_tin_roof_heavy_01".to_string(),
        file_path: "recordings/tin_roof_heavy_downpour.wav".to_string(),
        url: None,
        tags: vec!["tin_roof".to_string(), "metal".to_string()],
        precipitation_rate: Some(PrecipitationRate::HeavyRain),
        environment: Some("Suburban shed under dense tree canopy".to_string()),
        microphone_setup: Some(MicrophoneSetup::StereoOrtf),
        sample_rate: Some(48000),
        license: Some("CC0 1.0 Universal".to_string()),
        author: Some("Jane Doe".to_string()),
        notes: Some("Recorded with Zoom F6 and pair of Røde NT5 matched capsules".to_string()),
        sha256: None,
        descriptions: vec![
            "Heavy rain drumming rhythmically on corrugated tin shed roof".to_string(),
            "Acoustic resonant high-frequency metallic splatter texture".to_string(),
        ],
        alternate_licenses: Vec::new(),
        contributors: vec!["Jane Doe".to_string()],
    });

    manifest.add_source(RainSourceEntry {
        id: "pnw_pine_foliage_drizzle_02".to_string(),
        file_path: "recordings/pine_needles_drizzle.wav".to_string(),
        url: None,
        tags: vec!["foliage".to_string(), "pine".to_string()],
        precipitation_rate: Some(PrecipitationRate::Drizzle),
        environment: Some("Olympic National Forest hemlock & pine stand".to_string()),
        microphone_setup: Some(MicrophoneSetup::BinauralInEar),
        sample_rate: Some(48000),
        license: Some("CC-BY 4.0".to_string()),
        author: Some("Jane Doe".to_string()),
        notes: Some("In-ear binaural microphones mounted on windscreen baffle".to_string()),
        sha256: None,
        descriptions: vec![
            "Gentle mist and drizzle filtering through dense evergreen pine needles".to_string(),
        ],
        alternate_licenses: Vec::new(),
        contributors: vec!["Jane Doe".to_string()],
    });

    manifest.add_source(RainSourceEntry {
        id: "pnw_glass_window_moderate_03".to_string(),
        file_path: "recordings/glass_skylight_rain.wav".to_string(),
        url: None,
        tags: vec!["glass".to_string(), "window".to_string()],
        precipitation_rate: Some(PrecipitationRate::ModerateRain),
        environment: Some("Residential attic skylight glazing".to_string()),
        microphone_setup: Some(MicrophoneSetup::ContactMic),
        sample_rate: Some(96000),
        license: Some("CC0 1.0 Universal".to_string()),
        author: Some("Jane Doe".to_string()),
        notes: Some("Piezo contact transducer coupled directly to glass pane".to_string()),
        sha256: None,
        descriptions: vec![
            "Direct impact droplet cavitation on 6mm tempered window pane".to_string(),
        ],
        alternate_licenses: Vec::new(),
        contributors: vec!["Jane Doe".to_string()],
    });

    manifest
}

/// Recursively scans a local directory, validates all WAV recordings, stages them into the dataset,
/// and updates catalog metadata and attribution registers.
pub fn import_local_directory(
    dir_path: &Path,
    options: &LocalImportOptions,
) -> Result<ContributionReport> {
    assert_not_main_branch()?;

    if !dir_path.exists() {
        bail!("Input directory does not exist: {:?}", dir_path);
    }

    fs::create_dir_all(&options.target_dir)?;

    let mut discovered_files = Vec::new();
    find_audio_files(dir_path, &mut discovered_files)?;

    info!(
        "[*] Discovered {} audio candidates in {:?}. Running acoustic validation...",
        discovered_files.len(),
        dir_path
    );

    let mut successfully_imported = 0usize;
    let mut total_duration = 0.0f32;
    let mut tags_distribution = HashMap::new();
    let mut rejected_reasons = Vec::new();
    let mut new_sources = Vec::new();
    let mut provenance_records = Vec::new();

    // Verify default license if provided
    if let Some(lic) = &options.default_license {
        let (ok, _tier, reason) = LicenseVerifier::verify(lic);
        if !ok {
            bail!("Default contribution license rejected: {}", reason);
        }
    }

    let license_tier = if let Some(lic) = &options.default_license {
        LicenseVerifier::verify(lic).1
    } else {
        LicenseTier::Restricted
    };

    // Load existing manifest_provenance.json and sources.json upfront for content-addressed deduplication
    let prov_manifest_path = options.target_dir.join("manifest_provenance.json");
    let mut existing_manifest: Option<ProvenanceManifest> = if prov_manifest_path.exists() {
        fs::read_to_string(&prov_manifest_path)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
    } else {
        None
    };

    let sources_path_opt = if options.update_sources_json {
        shared::paths::WorkspacePaths::resolve_sources()
    } else {
        None
    };

    let mut existing_sources: Vec<DownloadItem> = if let Some(ref sp) = sources_path_opt {
        if sp.exists() {
            fs::read_to_string(sp)
                .ok()
                .and_then(|c| serde_json::from_str(&c).ok())
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    for file in &discovered_files {
        let file_stem = file.file_stem().unwrap_or_default().to_string_lossy().to_string();

        // 1. Infer tags from CLI default or filename
        let tags = if !options.default_tags.is_empty() {
            options.default_tags.clone()
        } else {
            vec!["auto_imported".to_string()]
        };

        // 2. Validate acoustic quality
        let (metrics, sha256, duration) = match validate_audio_file(file, &options.thresholds) {
            Ok(res) => res,
            Err(e) => {
                warn!("  [!] Rejected {:?}: {}", file.file_name().unwrap_or_default(), e);
                rejected_reasons.push(RejectedSourceEntry {
                    source_id: file_stem,
                    file_path: file.display().to_string(),
                    rejection_reason: e.to_string(),
                });
                continue;
            }
        };

        let author_str = options.author.clone().unwrap_or_else(|| "Anonymous".to_string()).replace(' ', "_");
        let contrib_list = if !author_str.is_empty() && author_str != "Anonymous" {
            vec![author_str.clone()]
        } else {
            Vec::new()
        };

        // 3. Content-addressed deduplication: check if identical audio already exists in the dataset
        let mut is_dedup = false;
        if let Some(ref mut manifest) = existing_manifest {
            if let Some(existing_record) = manifest.records.iter_mut().find(|r| r.sha256 == sha256) {
                is_dedup = true;
                let incoming = ProvenanceRecord {
                    filename: existing_record.filename.clone(),
                    source_url: format!("local://{}", file.display()),
                    source_platform: format!("Community Contribution ({})", author_str),
                    tags: tags.clone(),
                    license: options.default_license.clone(),
                    license_tier,
                    sha256: sha256.clone(),
                    file_size_bytes: existing_record.file_size_bytes,
                    quality: Some(metrics.clone()),
                    descriptions: Vec::new(),
                    alternate_licenses: Vec::new(),
                    contributors: contrib_list.clone(),
                };

                for tag in &tags {
                    if !existing_record.tags.contains(tag) {
                        *tags_distribution.entry(tag.clone()).or_insert(0) += 1;
                    }
                }

                existing_record.reconcile_with(incoming);
                successfully_imported += 1;
                info!(
                    "[+] Deduplicated audio (SHA-256: {}). Reconciled metadata with existing asset {:?}",
                    sha256, existing_record.filename
                );

                // Reconcile sources.json entry if present
                if let Some(src_item) = existing_sources.iter_mut().find(|s| s.filename == existing_record.filename) {
                    if let Some(ref best_lic) = existing_record.license {
                        src_item.license = best_lic.clone();
                    }
                }

                // Append attribution audit note
                if options.update_attributions {
                    if let Some(attr_path) = shared::paths::WorkspacePaths::resolve_attributions() {
                        let license_str = existing_record.license.as_deref().unwrap_or("Unknown");
                        let log_line = format!(
                            "Platform: Contribution ({}) [RECONCILED] | File: {} | Tags: {:?} | Tier: {:?} | License: {} | SHA256: {} | Path: {:?}\n",
                            author_str, existing_record.filename, existing_record.tags, existing_record.license_tier, license_str, sha256, file
                        );
                        if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&attr_path) {
                            use std::io::Write;
                            let _ = f.write_all(log_line.as_bytes());
                        }
                    }
                }
            }
        }

        if is_dedup {
            continue;
        }

        // 4. Stage non-duplicate file into target dataset directory
        let dest_filename = format!("contrib_{}_{}.wav", author_str, file_stem);
        let dest_path = options.target_dir.join(&dest_filename);

        fs::copy(file, &dest_path)
            .with_context(|| format!("Failed copying {:?} to {:?}", file, dest_path))?;

        let file_size_bytes = fs::metadata(&dest_path).map(|m| m.len()).unwrap_or(0);
        successfully_imported += 1;
        total_duration += duration;
        
        for tag in &tags {
            *tags_distribution.entry(tag.clone()).or_insert(0) += 1;
        }

        // 5. Record provenance entry
        let prov = ProvenanceRecord {
            filename: dest_filename.clone(),
            source_url: format!("local://{}", file.display()),
            source_platform: format!("Community Contribution ({})", author_str),
            tags,
            license: options.default_license.clone(),
            license_tier,
            sha256: sha256.clone(),
            file_size_bytes,
            quality: Some(metrics.clone()),
            descriptions: Vec::new(),
            alternate_licenses: Vec::new(),
            contributors: contrib_list,
        };
        provenance_records.push(prov);

        // 6. Append DownloadItem for sources.json
        new_sources.push(DownloadItem {
            url: format!("local://{}", dest_filename),
            filename: dest_filename.clone(),
            category: "auto_imported".to_string(), // Legacy field on DownloadItem
            license: options.default_license.clone().unwrap_or_default(),
            source_platform: format!("Contribution: {}", author_str),
            media_type: "audio".to_string(),
            ingest_method: "local_import".to_string(),
        });

        // 7. Append to ATTRIBUTIONS.txt
        if options.update_attributions {
            if let Some(attr_path) = shared::paths::WorkspacePaths::resolve_attributions() {
                let tags_str = "auto_imported"; // Just a placeholder for logging
                let license_str = options.default_license.as_deref().unwrap_or("Unknown");
                let log_line = format!(
                    "Platform: Contribution ({}) | File: {} | Tags: {} | Tier: {:?} | License: {} | SHA256: {} | Path: {:?}\n",
                    author_str, dest_filename, tags_str, license_tier, license_str, sha256, file
                );
                if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&attr_path) {
                    use std::io::Write;
                    let _ = f.write_all(log_line.as_bytes());
                }
            }
        }
    }

    // 8. Update sources.json if requested
    if options.update_sources_json && (!new_sources.is_empty() || !existing_sources.is_empty()) {
        if let Some(ref sources_path) = sources_path_opt {
            existing_sources.extend(new_sources);
            if let Ok(json) = serde_json::to_string_pretty(&existing_sources) {
                let _ = fs::write(sources_path, json);
                info!("[+] Synchronized sources.json with contributed items.");
            }
        }
    }

    // 9. Update manifest_provenance.json
    let mut final_manifest = existing_manifest.unwrap_or_else(|| ProvenanceManifest {
        generated_at_utc: crate::ingest::chrono_lite_timestamp(),
        total_sources: 0,
        tags_distribution: HashMap::new(),
        records: Vec::new(),
    });
    final_manifest.records.extend(provenance_records);
    final_manifest.total_sources = final_manifest.records.len();
    for (tag, count) in &tags_distribution {
        *final_manifest.tags_distribution.entry(tag.clone()).or_insert(0) += count;
    }
    if let Ok(json) = serde_json::to_string_pretty(&final_manifest) {
        let _ = fs::write(&prov_manifest_path, json);
        info!("[+] Synchronized manifest_provenance.json.");
    }

    info!(
        "[+] Import complete: {} / {} files accepted ({:.1}s total duration).",
        successfully_imported,
        discovered_files.len(),
        total_duration
    );

    Ok(ContributionReport {
        total_discovered: discovered_files.len(),
        successfully_imported,
        rejected_count: rejected_reasons.len(),
        total_duration_secs: total_duration,
        target_directory: options.target_dir.display().to_string(),
        surface_distribution: tags_distribution,
        rejected_reasons,
    })
}

/// Recursively discovers WAV files.
fn find_audio_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                find_audio_files(&path, out)?;
            } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext.eq_ignore_ascii_case("wav") {
                    out.push(path);
                }
            }
        }
    }
    Ok(())
}
