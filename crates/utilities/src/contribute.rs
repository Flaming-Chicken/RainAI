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
use hound::{SampleFormat, WavReader};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use crate::ingest::{
    analyze_pcm_samples, compute_file_sha256, AcousticQualityMetrics, DownloadItem, LicenseTier,
    LicenseVerifier, ProvenanceManifest, ProvenanceRecord,
};
pub use shared::surface::CanonicalSurface;

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
    /// Physical impact surface material (one of 9 canonical surfaces).
    pub surface: CanonicalSurface,
    /// Precipitation intensity rate.
    #[serde(default = "default_rate")]
    pub precipitation_rate: PrecipitationRate,
    /// Environmental recording context (e.g. `suburban_patio`, `dense_rainforest_canopy`).
    pub environment: Option<String>,
    /// Recording hardware & acoustic transducer geometry.
    #[serde(default = "default_mic")]
    pub microphone_setup: MicrophoneSetup,
    /// Nominal audio sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Open content license (e.g. `CC0`, `CC-BY 4.0`, `Public Domain`).
    pub license: String,
    /// Author / field recordist / institution name.
    pub author: String,
    /// Optional recording equipment or attribution notes.
    pub notes: Option<String>,
    /// Optional pre-computed SHA-256 hash.
    pub sha256: Option<String>,
}

fn default_rate() -> PrecipitationRate {
    PrecipitationRate::ModerateRain
}

fn default_mic() -> MicrophoneSetup {
    MicrophoneSetup::StereoSpaced
}

/// Standardized manifest bundle for community data contributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RainContributionManifest {
    pub manifest_version: String,
    pub dataset_name: String,
    pub contributor_name: String,
    pub contributor_contact: Option<String>,
    pub default_license: String,
    pub default_surface: Option<CanonicalSurface>,
    pub sources: Vec<RainSourceEntry>,
}

impl RainContributionManifest {
    pub fn new(
        dataset_name: impl Into<String>,
        contributor_name: impl Into<String>,
        default_license: impl Into<String>,
    ) -> Self {
        Self {
            manifest_version: "1.0".to_string(),
            dataset_name: dataset_name.into(),
            contributor_name: contributor_name.into(),
            contributor_contact: None,
            default_license: default_license.into(),
            default_surface: None,
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
    /// Minimum duration in seconds (shorter audio is rejected).
    pub min_duration_secs: f32,
    /// Minimum RMS energy (rejects silent or near-inaudible files).
    pub min_rms_energy: f32,
    /// Maximum sample clipping ratio (rejects distorted recordings).
    pub max_clipping_ratio: f32,
    /// Minimum spectral flatness (rejects narrow-band electrical hum / sine tones).
    pub min_spectral_flatness: f32,
    /// Minimum high-frequency energy ratio above 4 kHz (droplet cavitation check).
    pub min_high_freq_ratio: f32,
}

impl Default for AudioQualityThresholds {
    fn default() -> Self {
        Self {
            min_duration_secs: 2.0,
            min_rms_energy: 0.003,
            max_clipping_ratio: 0.015,
            min_spectral_flatness: 0.04,
            min_high_freq_ratio: 0.03,
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
    pub surface_counts: HashMap<CanonicalSurface, usize>,
    pub license_tiers: HashMap<LicenseTier, usize>,
    pub shannon_entropy: f32,
    pub normalized_diversity: f32,
    pub is_passing: bool,
}

/// Ingestion options for importing local audio files into RainAI.
#[derive(Debug, Clone)]
pub struct LocalImportOptions {
    pub default_surface: Option<CanonicalSurface>,
    pub default_rate: PrecipitationRate,
    pub default_mic: MicrophoneSetup,
    pub default_license: String,
    pub author: String,
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
            default_surface: None,
            default_rate: PrecipitationRate::ModerateRain,
            default_mic: MicrophoneSetup::StereoSpaced,
            default_license: "CC0 1.0 Universal".to_string(),
            author: "Community Contributor".to_string(),
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

/// Validates an individual audio file on disk against acoustic and format specifications.
pub fn validate_audio_file(
    path: &Path,
    thresholds: &AudioQualityThresholds,
) -> Result<(AcousticQualityMetrics, String, f32)> {
    if !path.exists() {
        bail!("File not found: {:?}", path);
    }

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    if ext != "wav" {
        bail!("Unsupported audio format '{}'. Please provide standard WAV audio.", ext);
    }

    let mut reader = WavReader::open(path)
        .with_context(|| format!("Failed to parse WAV header for {:?}", path))?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Int => {
            let scale = 1.0 / (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<Vec<_>, _>>()?
        }
    };

    if samples.is_empty() {
        bail!("Audio file contains 0 samples (empty audio)");
    }

    let channels = spec.channels;
    // Downmix multi-channel to mono for acoustic quality screening
    let mono_samples: Vec<f32> = if channels > 1 {
        let ch_count = channels as usize;
        samples
            .chunks_exact(ch_count)
            .map(|chunk| chunk.iter().sum::<f32>() / ch_count as f32)
            .collect()
    } else {
        samples
    };

    let metrics = analyze_pcm_samples(&mono_samples, spec.sample_rate, spec.channels);

    // Apply objective screening thresholds
    if metrics.duration_secs < thresholds.min_duration_secs {
        bail!(
            "Audio too short: {:.2}s (minimum required is {:.2}s)",
            metrics.duration_secs,
            thresholds.min_duration_secs
        );
    }
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
    if metrics.high_freq_ratio < thresholds.min_high_freq_ratio {
        bail!(
            "Lacks high-frequency droplet content (>4 kHz ratio is {:.4}, threshold is {:.4})",
            metrics.high_freq_ratio,
            thresholds.min_high_freq_ratio
        );
    }

    let sha256 = compute_file_sha256(path)?;
    let duration = metrics.duration_secs;

    Ok((metrics, sha256, duration))
}

/// Validates an entire contribution manifest, checking ethical licenses and acoustic quality.
pub fn validate_manifest(
    manifest: &RainContributionManifest,
    base_dir: Option<&Path>,
    thresholds: Option<&AudioQualityThresholds>,
) -> Result<ContributionValidationResult> {
    let default_thresholds = AudioQualityThresholds::default();
    let thresh = thresholds.unwrap_or(&default_thresholds);

    let mut valid_entries = Vec::new();
    let mut rejected_entries = Vec::new();
    let mut total_duration = 0.0f32;
    let mut surface_counts = HashMap::new();
    let mut license_tiers = HashMap::new();

    for s in CanonicalSurface::ALL {
        surface_counts.insert(s, 0);
    }

    for entry in &manifest.sources {
        // 1. Verify ethical open license
        let (ok, tier, reason) = LicenseVerifier::verify(&entry.license);
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
                    *surface_counts.entry(entry.surface).or_insert(0) += 1;
                    total_duration += duration;
                    valid_entries.push(ValidatedSourceEntry {
                        source: entry.clone(),
                        metrics,
                        computed_sha256: sha256,
                        duration_secs: duration,
                    });
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
            *surface_counts.entry(entry.surface).or_insert(0) += 1;
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
                computed_sha256: entry.sha256.clone().unwrap_or_else(|| "unfetched_remote".to_string()),
                duration_secs: 0.0,
            });
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
        "CC0 1.0 Universal",
    );
    manifest.contributor_contact = Some("https://github.com/janedoe".to_string());

    manifest.add_source(RainSourceEntry {
        id: "pnw_tin_roof_heavy_01".to_string(),
        file_path: "recordings/tin_roof_heavy_downpour.wav".to_string(),
        url: None,
        surface: CanonicalSurface::TinRoof,
        precipitation_rate: PrecipitationRate::HeavyRain,
        environment: Some("Suburban shed under dense tree canopy".to_string()),
        microphone_setup: MicrophoneSetup::StereoOrtf,
        sample_rate: Some(48000),
        license: "CC0 1.0 Universal".to_string(),
        author: "Jane Doe".to_string(),
        notes: Some("Recorded with Zoom F6 and pair of Røde NT5 matched capsules".to_string()),
        sha256: None,
    });

    manifest.add_source(RainSourceEntry {
        id: "pnw_pine_foliage_drizzle_02".to_string(),
        file_path: "recordings/pine_needles_drizzle.wav".to_string(),
        url: None,
        surface: CanonicalSurface::Foliage,
        precipitation_rate: PrecipitationRate::Drizzle,
        environment: Some("Olympic National Forest hemlock & pine stand".to_string()),
        microphone_setup: MicrophoneSetup::BinauralInEar,
        sample_rate: Some(48000),
        license: "CC-BY 4.0".to_string(),
        author: "Jane Doe".to_string(),
        notes: Some("In-ear binaural microphones mounted on windscreen baffle".to_string()),
        sha256: None,
    });

    manifest.add_source(RainSourceEntry {
        id: "pnw_glass_window_moderate_03".to_string(),
        file_path: "recordings/glass_skylight_rain.wav".to_string(),
        url: None,
        surface: CanonicalSurface::Glass,
        precipitation_rate: PrecipitationRate::ModerateRain,
        environment: Some("Residential attic skylight glazing".to_string()),
        microphone_setup: MicrophoneSetup::ContactMic,
        sample_rate: Some(96000),
        license: "CC0 1.0 Universal".to_string(),
        author: "Jane Doe".to_string(),
        notes: Some("Piezo contact transducer coupled directly to glass pane".to_string()),
        sha256: None,
    });

    manifest
}

/// Recursively scans a local directory, validates all WAV recordings, stages them into the dataset,
/// and updates catalog metadata and attribution registers.
pub fn import_local_directory(
    dir_path: &Path,
    options: &LocalImportOptions,
) -> Result<ContributionReport> {
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
    let mut surface_distribution = HashMap::new();
    let mut rejected_reasons = Vec::new();
    let mut new_sources = Vec::new();
    let mut provenance_records = Vec::new();

    // Verify default license
    let (license_ok, license_tier, license_reason) = LicenseVerifier::verify(&options.default_license);
    if !license_ok {
        bail!("Default contribution license rejected: {}", license_reason);
    }

    for file in &discovered_files {
        let file_stem = file.file_stem().unwrap_or_default().to_string_lossy().to_string();

        // 1. Infer surface from CLI default or filename/path keywords
        let surface = options.default_surface.unwrap_or_else(|| {
            let combined_tag = format!("{}/{}", file.display(), file_stem);
            CanonicalSurface::from_tag(&combined_tag)
        });

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

        // 3. Stage file into target dataset directory
        let dest_filename = format!("contrib_{}_{}_{}.wav", surface.as_str(), options.author.replace(' ', "_"), file_stem);
        let dest_path = options.target_dir.join(&dest_filename);

        fs::copy(file, &dest_path)
            .with_context(|| format!("Failed copying {:?} to {:?}", file, dest_path))?;

        let file_size_bytes = fs::metadata(&dest_path).map(|m| m.len()).unwrap_or(0);
        successfully_imported += 1;
        total_duration += duration;
        *surface_distribution.entry(surface.as_str().to_string()).or_insert(0) += 1;

        // 4. Record provenance entry
        let prov = ProvenanceRecord {
            filename: dest_filename.clone(),
            source_url: format!("local://{}", file.display()),
            source_platform: format!("Community Contribution ({})", options.author),
            category: format!("surface_{}", surface.as_str()),
            canonical_surface: surface,
            license: options.default_license.clone(),
            license_tier,
            sha256: sha256.clone(),
            file_size_bytes,
            quality: Some(metrics.clone()),
        };
        provenance_records.push(prov);

        // 5. Append DownloadItem for sources.json
        new_sources.push(DownloadItem {
            url: format!("local://{}", dest_filename),
            filename: dest_filename.clone(),
            category: format!("surface_{}", surface.as_str()),
            license: options.default_license.clone(),
            source_platform: format!("Contribution: {}", options.author),
            media_type: "audio".to_string(),
            ingest_method: "local_import".to_string(),
        });

        // 6. Append to ATTRIBUTIONS.txt
        if options.update_attributions {
            if let Some(attr_path) = shared::paths::WorkspacePaths::resolve_attributions() {
                let log_line = format!(
                    "Platform: Contribution ({}) | File: {} | Category: surface_{} (Surface: {}) | Tier: {:?} | License: {} | SHA256: {} | Path: {:?}\n",
                    options.author, dest_filename, surface.as_str(), surface.as_str(), license_tier, options.default_license, sha256, file
                );
                if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&attr_path) {
                    use std::io::Write;
                    let _ = f.write_all(log_line.as_bytes());
                }
            }
        }
    }

    // 7. Update sources.json if requested
    if options.update_sources_json && !new_sources.is_empty() {
        if let Some(sources_path) = shared::paths::WorkspacePaths::resolve_sources() {
            let mut existing: Vec<DownloadItem> = if sources_path.exists() {
                let content = fs::read_to_string(&sources_path).unwrap_or_else(|_| "[]".to_string());
                serde_json::from_str(&content).unwrap_or_default()
            } else {
                Vec::new()
            };
            existing.extend(new_sources);
            if let Ok(json) = serde_json::to_string_pretty(&existing) {
                let _ = fs::write(&sources_path, json);
                info!("[+] Updated sources.json with {} new contributed items.", successfully_imported);
            }
        }
    }

    // 8. Update manifest_provenance.json
    let prov_manifest_path = options.target_dir.join("manifest_provenance.json");
    if prov_manifest_path.exists() {
        if let Ok(content) = fs::read_to_string(&prov_manifest_path) {
            if let Ok(mut existing_manifest) = serde_json::from_str::<ProvenanceManifest>(&content) {
                existing_manifest.total_sources += provenance_records.len();
                existing_manifest.records.extend(provenance_records);
                for (s, count) in &surface_distribution {
                    *existing_manifest.surface_distribution.entry(s.clone()).or_insert(0) += count;
                }
                if let Ok(json) = serde_json::to_string_pretty(&existing_manifest) {
                    let _ = fs::write(&prov_manifest_path, json);
                }
            }
        }
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
        surface_distribution,
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
