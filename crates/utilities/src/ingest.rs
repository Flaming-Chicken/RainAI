//! Dataset Ingestion, Ethical License Verification, Acoustic Quality Screening, and Async Download Engine.
//!
//! Provides:
//! - Multi-source async downloads with exponential backoff retries and HF auth.
//! - Ethical licensing verification (`LicenseVerifier`) enforcing CC0, CC-BY, and Public Domain.
//! - Acoustic quality metrics calculation (`AcousticQualityMetrics`) measuring RMS, clipping ratio,
//!   Wiener spectral flatness, and Shannon spectral entropy.
//! - 9-class physical surface classification & quota tracking (`SurfaceBalanceQuota`).
//! - SHA-256 cryptographic provenance records (`ProvenanceRecord`, `ProvenanceManifest`).

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use hound::{SampleFormat, WavReader};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File as StdFile;
use std::io::Read;
use std::path::Path;
use std::time::Duration;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadItem {
    pub url: String,
    pub filename: String,
    pub category: String,
    pub license: String,
    pub source_platform: String,
    #[serde(default = "default_media_type")]
    pub media_type: String,
    #[serde(default = "default_ingest_method")]
    pub ingest_method: String,
}

fn default_media_type() -> String {
    "audio".to_string()
}
fn default_ingest_method() -> String {
    "direct_http".to_string()
}

/// Canonical grant text for user contributions under the default proprietary license.
pub const RAINAI_FC_PROPRIETARY_LICENSE_TEXT: &str = "\
By submitting this data and metadata, I grant Spodeian, Flaming Chicken, and their respective affiliates, \
successors, and assigns a worldwide, non-exclusive, royalty-free, perpetual, irrevocable, and sublicensable \
right to use, reproduce, modify, adapt, publish, translate, create derivative works from, distribute, and \
publicly display this data for any purpose, including commercial and non-commercial applications. This \
explicitly includes, without limitation, the right to use the data to train, test, and validate machine \
learning models for the RainAI project and any other current or future projects. I represent and warrant that \
I own or have the necessary rights to grant this license.";

/// Plain-language transparency copy displayed in the UI to build contributor trust regarding RainAI's XAI architecture.
pub const RAINAI_FC_TRANSPARENCY_NOTE: &str = "\
While this license allows us to use your data freely to build RainAI, our system is designed for transparency. \
We track the metadata of all contributions, meaning you will always be credited when your specific data \
directly influences our explainable AI's outputs.";

/// Mandatory contributor rights warranty statement.
pub const CONTRIBUTOR_WARRANTY_STATEMENT: &str = "\
I represent and warrant that I own or have the necessary rights to grant this license.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LicenseTier {
    PublicDomain,
    AttributionOnly,
    ShareAlike,
    ProjectProprietary,
    Unknown,
    Restricted,
}

impl LicenseTier {
    /// Determines whether this license tier requires contributor attribution.
    #[inline]
    pub fn requires_attribution(&self) -> bool {
        matches!(self, LicenseTier::AttributionOnly | LicenseTier::ShareAlike)
    }

    /// Whether this tier is approved for the training corpus (PLAN.md D2).
    /// ShareAlike is recognised but excluded.
    #[inline]
    pub fn is_approved(&self) -> bool {
        matches!(
            self,
            LicenseTier::PublicDomain
                | LicenseTier::AttributionOnly
                | LicenseTier::ProjectProprietary
        )
    }

    /// Whether this submission must be immediately placed in quarantine pending license discovery or scraping.
    #[inline]
    pub fn is_quarantined_pending_discovery(&self) -> bool {
        matches!(self, LicenseTier::Unknown)
    }

    /// Returns a numeric preference rank for license reconciliation when multiple licenses are granted for the same asset.
    /// Higher values indicate licenses that give the project greater flexibility
    /// for commercial applications, model weight distribution, and ecosystem development.
    ///
    /// 1. ProjectProprietary (RainAI-FC-Proprietary-License) -> 6 (Highest flexibility)
    /// 2. PublicDomain (CC0 / Unlicense / PDDL) -> 5
    /// 3. AttributionOnly (CC-BY 4.0/3.0/2.0, MIT, Apache 2.0) -> 4
    /// 4. ShareAlike (CC-BY-SA 4.0/3.0) -> 3
    /// 5. Unknown (Pending automated license scraping) -> 2
    /// 6. Restricted (Non-Commercial / No-Derivatives) -> 1
    #[inline]
    pub fn preference_rank(&self) -> u8 {
        match self {
            Self::ProjectProprietary => 6,
            Self::PublicDomain => 5,
            Self::AttributionOnly => 4,
            Self::ShareAlike => 3,
            Self::Unknown => 2,
            Self::Restricted => 1,
        }
    }

    /// Returns human-readable label for the policy tier.
    pub fn policy_description(&self) -> &'static str {
        match self {
            Self::PublicDomain => "Public Domain (CC0 1.0 / public domain / PD Mark)",
            Self::AttributionOnly => "Attribution (CC BY 3.0 / 4.0)",
            Self::ShareAlike => "Share-Alike (CC BY-SA; excluded from the corpus)",
            Self::ProjectProprietary => {
                "RainAI / Spodeian / Flaming Chicken Proprietary (Commercial grant with XAI transparency)"
            }
            Self::Unknown => "Unknown / Pending License Scraping (Immediately Quarantined)",
            Self::Restricted => "Restricted / Ineligible (NC, ND, platform or unlisted license)",
        }
    }
}

/// Given a collection of candidate license strings granted for the same asset across contributions,
/// returns the predominant license that is most favorable for the project,
/// alongside its tier and verification diagnosis.
pub fn select_predominant_license<'a, I>(licenses: I) -> (String, LicenseTier, &'static str)
where
    I: IntoIterator<Item = &'a str>,
{
    let mut best_lic = "Unknown".to_string();
    let mut best_tier = LicenseTier::Unknown;
    let mut best_reason = "No licenses provided";
    let mut best_rank = 0u8;

    for lic in licenses {
        let (ok, tier, reason) = LicenseVerifier::verify(lic);
        let rank = tier.preference_rank();
        if rank > best_rank || (rank == best_rank && ok && !best_tier.is_approved()) {
            best_rank = rank;
            best_lic = lic.to_string();
            best_tier = tier;
            best_reason = reason;
        }
    }

    (best_lic, best_tier, best_reason)
}

/// Status of attribution resolution for an XAI (Explainable AI) generated output.
///
/// In RainAI, all training data is permanently registered upon ingestion into `ATTRIBUTIONS.txt`
/// and model provenance cards. Therefore, individual generated soundscape outputs are **never blocked**.
/// Runtime XAI attribution is an **optional explainability feature** rather than a blocking mandate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionPolicyOutcome {
    /// Runtime XAI attribution resolved and dynamically linked to model output.
    Attributed {
        contributor: String,
        license: String,
        tier: LicenseTier,
    },
    /// Runtime XAI untraced, but global training attribution is permanently recorded in ATTRIBUTIONS.txt. Output is safely served.
    GlobalTrainingAttributed { note: String, tier: LicenseTier },
    /// Public domain or unconstrained asset; no individual attribution needed. Output is safely served.
    Unconstrained { note: String, tier: LicenseTier },
}

/// Evaluates attribution resolution for an XAI model output.
///
/// Returns an `AttributionPolicyOutcome`. Because training data is permanently attributed
/// in `ATTRIBUTIONS.txt` and `DATA_BIBLIOGRAPHY.md` at training time, model outputs are
/// never blocked.
pub fn evaluate_attribution_policy(
    tier: LicenseTier,
    contributor: Option<&str>,
    license_id: &str,
) -> AttributionPolicyOutcome {
    if let Some(author) = contributor.filter(|a| !a.trim().is_empty()) {
        AttributionPolicyOutcome::Attributed {
            contributor: author.to_string(),
            license: license_id.to_string(),
            tier,
        }
    } else if tier == LicenseTier::PublicDomain {
        AttributionPolicyOutcome::Unconstrained {
            note:
                "Public domain asset; unconstrained generation with permanent bibliography record."
                    .to_string(),
            tier,
        }
    } else {
        AttributionPolicyOutcome::GlobalTrainingAttributed {
            note: format!(
                "Permanent dataset-level attribution active for '{}' in ATTRIBUTIONS.txt; output served unconditionally.",
                tier.policy_description()
            ),
            tier,
        }
    }
}

/// Decides whether a license string is allowed into the training corpus (PLAN.md D2).
///
/// Allowed, and nothing else:
/// - CC0 1.0 and public domain (including the PD Mark) -> [`LicenseTier::PublicDomain`]
/// - CC BY 3.0 / 4.0 -> [`LicenseTier::AttributionOnly`]
/// - the RainAI contributor grant -> [`LicenseTier::ProjectProprietary`]
///
/// ShareAlike, NonCommercial and NoDerivatives licenses, platform licenses
/// (Mixkit, Pixabay, ...), software licenses (MIT, Apache, ...) and anything
/// unrecognised are rejected. Matching is on whole tokens, so "Unlicense" is
/// not read as containing "nc", nor "standard" as containing "nd".
pub struct LicenseVerifier;

impl LicenseVerifier {
    fn tokens(clean: &str) -> Vec<&str> {
        clean
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '.'))
            .filter(|t| !t.is_empty())
            .collect()
    }

    pub fn verify(license: &str) -> (bool, LicenseTier, &'static str) {
        let clean = license.trim().to_lowercase();
        let tokens = Self::tokens(&clean);
        let has = |t: &str| tokens.contains(&t);

        // 1. Unknown / unspecified: quarantined until a license is read from the source.
        if clean.is_empty()
            || matches!(
                clean.as_str(),
                "unknown" | "unspecified" | "none" | "to be scraped" | "n/a"
            )
            || clean.contains("pending")
        {
            return (
                false,
                LicenseTier::Unknown,
                "Quarantined: no license recorded; read it from the source's metadata first",
            );
        }

        // 2. The RainAI contributor grant (see RAINAI_FC_PROPRIETARY_LICENSE_TEXT).
        if clean.contains("rainai-fc")
            || clean.contains("rainai-proprietary")
            || clean.contains("spodeian-permission")
        {
            return (
                true,
                LicenseTier::ProjectProprietary,
                "Approved: RainAI contributor grant (contributor warrants they hold the rights)",
            );
        }

        // 3. NonCommercial / NoDerivatives, on whole tokens or creativecommons.org URLs.
        if has("nc")
            || clean.contains("noncommercial")
            || clean.contains("non-commercial")
            || clean.contains("/by-nc")
        {
            return (
                false,
                LicenseTier::Restricted,
                "Rejected: NonCommercial clause",
            );
        }
        if has("nd")
            || clean.contains("noderivatives")
            || clean.contains("no-derivatives")
            || clean.contains("/by-nd")
        {
            return (
                false,
                LicenseTier::Restricted,
                "Rejected: NoDerivatives clause",
            );
        }

        // 4. ShareAlike: recognised, but excluded by D2.
        if has("sa")
            || clean.contains("sharealike")
            || clean.contains("share-alike")
            || clean.contains("/by-sa")
        {
            return (
                false,
                LicenseTier::ShareAlike,
                "Rejected: ShareAlike is excluded from the corpus (D2)",
            );
        }

        // 5. CC0 and public domain.
        if has("cc0")
            || clean.contains("publicdomain/zero")
            || clean.contains("public domain")
            || has("pdm")
            || clean.contains("publicdomain/mark")
            || clean.contains("licenses/publicdomain")
        {
            return (
                true,
                LicenseTier::PublicDomain,
                "Approved: CC0 1.0 / public domain",
            );
        }

        // 6. CC BY 3.0 / 4.0 ("CC-BY 4.0", "Attribution 4.0", or the license URL).
        let is_cc_by = (has("cc") && has("by"))
            || has("attribution")
            || clean.contains("creativecommons.org/licenses/by/");
        if is_cc_by {
            if has("4.0") || has("3.0") || clean.contains("/by/4.0") || clean.contains("/by/3.0") {
                return (
                    true,
                    LicenseTier::AttributionOnly,
                    "Approved: CC BY 3.0 / 4.0 (creator, title, source and changes must be recorded)",
                );
            }
            return (
                false,
                LicenseTier::Unknown,
                "Quarantined: CC BY without a 3.0/4.0 version; confirm the version at the source",
            );
        }

        (
            false,
            LicenseTier::Restricted,
            "Rejected: not on the allow-list (CC0, public domain, CC BY 3.0/4.0)",
        )
    }
}

/// Tracks distribution of tags in the dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TagBalanceQuota {
    pub counts: HashMap<String, usize>,
}

impl TagBalanceQuota {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    pub fn record(&mut self, tags: &[String]) {
        for tag in tags {
            *self.counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    pub fn total_samples(&self) -> usize {
        self.counts.values().sum()
    }
}

/// Acoustic quality metrics for validation and filtering of downloaded precipitation audio.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AcousticQualityMetrics {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_secs: f32,
    pub rms_energy: f32,
    pub peak_amplitude: f32,
    pub clipping_ratio: f32,
    pub spectral_entropy: f32,
    pub spectral_flatness: f32,
    pub high_freq_ratio: f32,
    pub is_valid_rain_texture: bool,
}

/// Computes acoustic quality metrics from mono/downmixed PCM floating point audio samples.
pub fn analyze_pcm_samples(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> AcousticQualityMetrics {
    let n = samples.len();
    if n == 0 {
        return AcousticQualityMetrics {
            sample_rate,
            channels,
            duration_secs: 0.0,
            rms_energy: 0.0,
            peak_amplitude: 0.0,
            clipping_ratio: 0.0,
            spectral_entropy: 0.0,
            spectral_flatness: 0.0,
            high_freq_ratio: 0.0,
            is_valid_rain_texture: false,
        };
    }

    let duration_secs = n as f32 / sample_rate as f32;
    let mut sum_sq = 0.0f32;
    let mut peak_amplitude = 0.0f32;
    let mut clipped_count = 0usize;

    for &s in samples {
        let abs_s = s.abs();
        sum_sq += s * s;
        if abs_s > peak_amplitude {
            peak_amplitude = abs_s;
        }
        if abs_s >= 0.999 {
            clipped_count += 1;
        }
    }

    let rms_energy = (sum_sq / n as f32).sqrt();
    let clipping_ratio = clipped_count as f32 / n as f32;

    // FFT spectral analysis
    const FFT_SIZE: usize = 1024;
    const HOP_SIZE: usize = 512;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let num_hops = n.saturating_sub(FFT_SIZE) / HOP_SIZE;
    let mut total_entropy = 0.0f32;
    let mut total_flatness = 0.0f32;
    let mut total_hf_ratio = 0.0f32;
    let mut valid_hops = 0usize;

    let mut buffer = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        FFT_SIZE
    ];
    let half = FFT_SIZE / 2;
    let bin_hz = sample_rate as f32 / FFT_SIZE as f32;
    let hf_bin_start = (4000.0 / bin_hz).clamp(1.0, (half - 1) as f32) as usize;

    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos()))
        .collect();

    for h in 0..num_hops {
        let offset = h * HOP_SIZE;
        for (i, c) in buffer.iter_mut().enumerate().take(FFT_SIZE) {
            *c = Complex {
                re: samples[offset + i] * window[i],
                im: 0.0,
            };
        }

        fft.process(&mut buffer);

        let mut sum_power = 0.0f32;
        let mut log_power_sum = 0.0f32;
        let mut hf_power = 0.0f32;
        let mut powers = Vec::with_capacity(half);

        for (k, c) in buffer.iter().enumerate().take(half) {
            let p = c.re * c.re + c.im * c.im + 1e-12;
            powers.push(p);
            sum_power += p;
            log_power_sum += p.ln();
            if k >= hf_bin_start {
                hf_power += p;
            }
        }

        if sum_power > 1e-8 {
            // Shannon spectral entropy
            let mut hop_entropy = 0.0f32;
            let inv_sum = 1.0 / sum_power;
            for &p in &powers {
                let prob = p * inv_sum;
                if prob > 1e-9 {
                    hop_entropy -= prob * prob.log2();
                }
            }
            let norm_entropy = (hop_entropy / (half as f32).log2()).clamp(0.0, 1.0);

            // Spectral flatness (Wiener entropy)
            let geom_mean = (log_power_sum / half as f32).exp();
            let arith_mean = sum_power / half as f32;
            let flatness = (geom_mean / (arith_mean + 1e-12)).clamp(0.0, 1.0);

            // High frequency ratio
            let hf_ratio = (hf_power / sum_power).clamp(0.0, 1.0);

            total_entropy += norm_entropy;
            total_flatness += flatness;
            total_hf_ratio += hf_ratio;
            valid_hops += 1;
        }
    }

    let spectral_entropy = if valid_hops > 0 {
        total_entropy / valid_hops as f32
    } else {
        0.5
    };
    let spectral_flatness = if valid_hops > 0 {
        total_flatness / valid_hops as f32
    } else {
        0.2
    };
    let high_freq_ratio = if valid_hops > 0 {
        total_hf_ratio / valid_hops as f32
    } else {
        0.2
    };

    // Quality gate: require audible signal, limited clipping, broadband entropy
    let is_valid_rain_texture = rms_energy >= 1e-4
        && peak_amplitude >= 1e-3
        && clipping_ratio <= 0.05
        && spectral_entropy >= 0.35;

    AcousticQualityMetrics {
        sample_rate,
        channels,
        duration_secs,
        rms_energy,
        peak_amplitude,
        clipping_ratio,
        spectral_entropy,
        spectral_flatness,
        high_freq_ratio,
        is_valid_rain_texture,
    }
}

/// Reads a WAV audio file from disk and computes its `AcousticQualityMetrics`.
pub fn analyze_wav_file(path: &Path) -> Result<AcousticQualityMetrics> {
    let mut reader = WavReader::open(path).context("Failed to open WAV for acoustic inspection")?;
    let spec = reader.spec();

    let raw_samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        SampleFormat::Int => {
            let scale = 1.0 / (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok().map(|v| v as f32 * scale))
                .collect()
        }
    };

    let channels = spec.channels as usize;
    let n_frames = raw_samples.len() / channels.max(1);
    let mut mono = Vec::with_capacity(n_frames);
    let inv_ch = 1.0 / channels as f32;

    for frame in raw_samples.chunks_exact(channels) {
        mono.push(frame.iter().sum::<f32>() * inv_ch);
    }

    Ok(analyze_pcm_samples(&mono, spec.sample_rate, spec.channels))
}

/// Computes the cryptographic SHA-256 hexadecimal hash of a file on disk.
pub fn compute_file_sha256(path: &Path) -> Result<String> {
    let mut file = StdFile::open(path).with_context(|| format!("Failed to open {:?}", path))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];

    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Atomically writes content to a file via a temporary file, fsync, and atomic rename.
///
/// Guarantees Atomicity and Durability without leaving orphaned backup or history files.
pub fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create parent directory at {:?}", parent))?;

    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    let tmp_path = parent.join(format!("{}.tmp.{}", file_name, std::process::id()));

    let mut file = std::fs::File::create(&tmp_path)
        .with_context(|| format!("Failed to create temporary file at {:?}", tmp_path))?;

    use std::io::Write;
    file.write_all(content)
        .with_context(|| format!("Failed to write data to {:?}", tmp_path))?;
    file.sync_all()
        .with_context(|| format!("Failed to fsync {:?}", tmp_path))?;
    drop(file);

    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        bail!(
            "Failed to atomically rename {:?} to {:?}: {}",
            tmp_path,
            path,
            e
        );
    }

    Ok(())
}

/// Provenance metadata record for an ingested audio asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub filename: String,
    pub source_url: String,
    pub source_platform: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub license: Option<String>,
    pub license_tier: LicenseTier,
    pub sha256: String,
    pub file_size_bytes: u64,
    pub quality: Option<AcousticQualityMetrics>,
    #[serde(default)]
    pub descriptions: Vec<String>,
    #[serde(default)]
    pub alternate_licenses: Vec<String>,
    #[serde(default)]
    pub contributors: Vec<String>,
}

impl ProvenanceRecord {
    /// Reconciles an existing provenance record with newly contributed metadata
    /// for the identical underlying audio SHA-256 hash.
    pub fn reconcile_with(&mut self, other: ProvenanceRecord) {
        // 1. Merge tags uniquely
        for tag in other.tags {
            if !self.tags.contains(&tag) {
                self.tags.push(tag);
            }
        }

        // 2. Merge parallel descriptions without duplicates
        for desc in other.descriptions {
            let trimmed = desc.trim();
            if !trimmed.is_empty() && !self.descriptions.iter().any(|d| d.trim() == trimmed) {
                self.descriptions.push(trimmed.to_string());
            }
        }

        // 3. Merge contributors without duplicates
        for contrib in other.contributors {
            let trimmed = contrib.trim();
            if !trimmed.is_empty() && !self.contributors.iter().any(|c| c.trim() == trimmed) {
                self.contributors.push(trimmed.to_string());
            }
        }

        // 4. Collect all candidate licenses and pick predominant
        let mut all_licenses = Vec::new();
        if let Some(ref l) = self.license {
            all_licenses.push(l.clone());
        }
        all_licenses.extend(self.alternate_licenses.clone());
        if let Some(ref l) = other.license {
            all_licenses.push(l.clone());
        }
        all_licenses.extend(other.alternate_licenses);

        all_licenses.sort();
        all_licenses.dedup();

        let lic_refs: Vec<&str> = all_licenses.iter().map(|s| s.as_str()).collect();
        let (best_lic, best_tier, _) = select_predominant_license(lic_refs);

        self.license = Some(best_lic.clone());
        self.license_tier = best_tier;
        self.alternate_licenses = all_licenses
            .into_iter()
            .filter(|l| l != &best_lic)
            .collect();

        // 5. Prefer populated acoustic quality metrics if not present
        if self.quality.is_none() && other.quality.is_some() {
            self.quality = other.quality;
        }
    }
}

/// Complete dataset provenance audit manifest.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProvenanceManifest {
    pub generated_at_utc: String,
    pub total_sources: usize,
    pub tags_distribution: HashMap<String, usize>,
    pub records: Vec<ProvenanceRecord>,
}

pub async fn download_file_with_retry(
    client: &reqwest::Client,
    url: &str,
    destination: &Path,
) -> Result<()> {
    let mut retries = 3;
    let mut delay = Duration::from_secs(2);

    loop {
        let mut request = client.get(url);

        if url.contains("huggingface.co") {
            if let Ok(token) = std::env::var("HF_TOKEN") {
                request = request.bearer_auth(token);
            }
        }

        match request.send().await {
            Ok(resp) if resp.status().is_success() => {
                let mut file = File::create(destination).await?;
                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    let data = chunk?;
                    file.write_all(&data).await?;
                }
                return Ok(());
            }
            Ok(resp) => {
                let status = resp.status();
                if status.is_client_error() {
                    bail!("Fatal HTTP status: {}", status);
                }
                if retries == 0 {
                    bail!("Failed with HTTP status {} after retries", status);
                }
            }
            Err(e) => {
                if retries == 0 {
                    bail!("Network error: {}", e);
                }
            }
        }

        retries -= 1;
        tokio::time::sleep(delay).await;
        delay *= 2;
    }
}

pub fn chrono_lite_timestamp() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{}.{:03}s_since_epoch",
        duration.as_secs(),
        duration.subsec_millis()
    )
}

use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
};

/// In-process dataset ingestion pipeline callable on a background thread.
pub fn run_ingestion_pipeline(
    stop_signal: Arc<AtomicBool>,
    log_tx: Option<Sender<String>>,
) -> Result<usize> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(run_ingestion_pipeline_async(stop_signal, log_tx))
}

/// Candidate locations of the committed source registry.
pub const SOURCE_REGISTRY_CANDIDATES: &[&str] = &[
    "source_registry.json",
    "../source_registry.json",
    "../../source_registry.json",
];

/// Downloads every allowed item from the sources in `source_registry.json`
/// (PLAN.md D2). Each item's license is read from the platform's metadata and
/// checked with [`LicenseVerifier`]; anything not allowed is skipped and
/// logged. Every downloaded file gets a ledger record, and the attribution
/// file, provenance manifest and local catalog are regenerated from the
/// ledger. Returns the number of newly downloaded files.
pub async fn run_ingestion_pipeline_async(
    stop_signal: Arc<AtomicBool>,
    log_tx: Option<Sender<String>>,
) -> Result<usize> {
    use crate::sources::{
        DataPaths, Ledger, LedgerRecord, STANDARD_CHANGES, SourceRegistry, fetch_candidates,
        ledger_filename, regenerate_outputs,
    };

    let emit_log = |msg: String| {
        tracing::info!("{}", msg);
        if let Some(ref tx) = log_tx {
            let _ = tx.send(msg);
        }
    };

    let Some(registry_path) = SOURCE_REGISTRY_CANDIDATES
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
    else {
        emit_log("[!] source_registry.json not found.".to_string());
        return Ok(0);
    };
    let registry = SourceRegistry::load(&registry_path)?;
    let root = registry_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let paths = DataPaths::under(&root);
    tokio::fs::create_dir_all(&paths.rain_dir).await?;

    let mut ledger = Ledger::load(&paths.ledger())?;
    let mut known_items: std::collections::HashSet<String> =
        ledger.iter().map(|r| r.item_url.clone()).collect();
    let mut known_hashes: std::collections::HashSet<String> =
        ledger.iter().map(|r| r.sha256.clone()).collect();

    emit_log(format!(
        "[*] {} sources in {}; {} files already in the ledger",
        registry.sources.len(),
        registry_path.display(),
        ledger.len()
    ));

    let client = reqwest::Client::builder()
        .user_agent("RainAI-Dataset-Collector/5.0 (+https://github.com/Spodeian/RainAI)")
        .timeout(Duration::from_secs(60))
        .pool_idle_timeout(Duration::from_secs(15))
        .build()?;

    let mut downloaded_count = 0usize;
    'sources: for spec in &registry.sources {
        if stop_signal.load(Ordering::Relaxed) {
            emit_log("[!] Audio ingest aborted by user token.".to_string());
            break;
        }
        let candidates = match fetch_candidates(&client, spec).await {
            Ok(c) => c,
            Err(e) => {
                emit_log(format!(
                    "  [!] {} ({}): {:#}",
                    spec.id(),
                    spec.adapter_name(),
                    e
                ));
                continue;
            }
        };
        emit_log(format!(
            "  [*] {} ({}): {} candidates",
            spec.id(),
            spec.adapter_name(),
            candidates.len()
        ));

        for cand in candidates {
            if stop_signal.load(Ordering::Relaxed) {
                emit_log("[!] Audio ingest aborted by user token.".to_string());
                break 'sources;
            }
            if known_items.contains(&cand.item_url) {
                continue;
            }
            let (allowed, tier, reason) = LicenseVerifier::verify(&cand.license);
            if !allowed || tier == LicenseTier::ProjectProprietary {
                emit_log(format!("    [-] {}: {}", cand.item_url, reason));
                continue;
            }
            if tier == LicenseTier::AttributionOnly && cand.creator.trim().is_empty() {
                emit_log(format!(
                    "    [-] {}: CC BY item has no creator to credit",
                    cand.item_url
                ));
                continue;
            }

            let file = ledger_filename(spec.id(), &cand);
            let dest = paths.rain_dir.join(&file);
            if let Some(dir) = dest.parent() {
                tokio::fs::create_dir_all(dir).await?;
            }
            if let Err(e) = download_file_with_retry(&client, &cand.download_url, &dest).await {
                emit_log(format!("    [!] {}: {}", cand.download_url, e));
                let _ = tokio::fs::remove_file(&dest).await;
                continue;
            }
            let sha256 = compute_file_sha256(&dest)?;
            if !known_hashes.insert(sha256.clone()) {
                emit_log(format!(
                    "    [=] {}: duplicate of a ledgered file",
                    cand.item_url
                ));
                let _ = tokio::fs::remove_file(&dest).await;
                continue;
            }

            let rec = LedgerRecord {
                file,
                sha256,
                source_id: spec.id().to_string(),
                platform: spec.adapter_name().to_string(),
                category: spec.category().to_string(),
                item_url: cand.item_url.clone(),
                download_url: cand.download_url,
                creator: cand.creator,
                title: cand.title,
                license: cand.license,
                license_url: cand.license_url,
                license_tier: tier,
                retrieved_utc: chrono_lite_timestamp(),
                changes: STANDARD_CHANGES.to_string(),
            };
            Ledger::append(&paths.ledger(), &rec)?;
            emit_log(format!("    [+] {} ({:?})", rec.file, tier));
            known_items.insert(cand.item_url);
            ledger.push(rec);
            downloaded_count += 1;
        }
    }

    let total = regenerate_outputs(&paths)?;
    emit_log(format!(
        "[+] Ingest finished: {} new files, {} in the ledger. Regenerated {}, {} and {}.",
        downloaded_count,
        total,
        paths.attributions().display(),
        paths.provenance().display(),
        paths.catalog.display()
    ));
    Ok(downloaded_count)
}
