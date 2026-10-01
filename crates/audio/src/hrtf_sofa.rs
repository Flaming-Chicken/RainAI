//! Personalized Head-Related Transfer Function (HRTF) and SOFA spatialization.
//!
//! Provides coordinate mapping (azimuth, elevation, distance), spherical interpolation,
//! and binaural time-domain convolution with measured Head-Related Impulse Responses (HRIRs).

use serde::{Deserialize, Serialize};

/// 3D spherical sound source position.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SphericalPosition {
    /// Azimuth angle in degrees: [-180.0, 180.0] (0 = straight ahead, 90 = right, -90 = left)
    pub azimuth_deg: f32,
    /// Elevation angle in degrees: [-90.0, 90.0] (0 = horizontal plane, 90 = directly above)
    pub elevation_deg: f32,
    /// Distance from head center in meters
    pub distance_m: f32,
}

impl SphericalPosition {
    pub fn new(azimuth_deg: f32, elevation_deg: f32, distance_m: f32) -> Self {
        Self {
            azimuth_deg: azimuth_deg.clamp(-180.0, 180.0),
            elevation_deg: elevation_deg.clamp(-90.0, 90.0),
            distance_m: distance_m.max(0.1),
        }
    }
}

/// Pair of Head-Related Impulse Responses (HRIR) for left and right ears.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrirPair {
    pub position: SphericalPosition,
    pub left_ir: Vec<f32>,
    pub right_ir: Vec<f32>,
}

/// SOFA (Spatially Oriented Format for Acoustics) Binaural Spatializer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SofaSpatializer {
    pub impulse_responses: Vec<HrirPair>,
    pub sample_rate: u32,
}

impl SofaSpatializer {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            impulse_responses: Vec::new(),
            sample_rate,
        }
    }

    /// Adds a measured HRIR pair for a spherical coordinate.
    pub fn add_hrir(&mut self, position: SphericalPosition, left_ir: Vec<f32>, right_ir: Vec<f32>) {
        self.impulse_responses.push(HrirPair {
            position,
            left_ir,
            right_ir,
        });
    }

    /// Finds the nearest measured HRIR pair by Euclidean angular distance.
    pub fn find_nearest_hrir(&self, pos: SphericalPosition) -> Option<&HrirPair> {
        if self.impulse_responses.is_empty() {
            return None;
        }

        self.impulse_responses.iter().min_by(|a, b| {
            let dist_a = (a.position.azimuth_deg - pos.azimuth_deg).powi(2)
                + (a.position.elevation_deg - pos.elevation_deg).powi(2);
            let dist_b = (b.position.azimuth_deg - pos.azimuth_deg).powi(2)
                + (b.position.elevation_deg - pos.elevation_deg).powi(2);
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Performs in-place zero-allocation binaural convolution of a mono audio buffer into caller-provided left and right output slices.
    pub fn spatialize_mono_into(
        &self,
        input: &[f32],
        pos: SphericalPosition,
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let n = input.len();
        if n == 0 {
            return;
        }

        if let Some(hrir) = self.find_nearest_hrir(pos) {
            let ir_len = hrir.left_ir.len().min(hrir.right_ir.len());
            let out_len = (n + ir_len.saturating_sub(1)).min(left_out.len()).min(right_out.len());
            left_out[..out_len].fill(0.0);
            right_out[..out_len].fill(0.0);

            // Vector-unrolled time-domain convolution with auto-vectorization
            let l_ir = &hrir.left_ir[..ir_len];
            let r_ir = &hrir.right_ir[..ir_len];

            for i in 0..n {
                let s = input[i];
                let max_j = ir_len.min(out_len.saturating_sub(i));
                let l_slice = &mut left_out[i..i + max_j];
                let r_slice = &mut right_out[i..i + max_j];

                let active_chunks = max_j / 4;
                for c in 0..active_chunks {
                    let b = c * 4;
                    l_slice[b] += s * l_ir[b];
                    l_slice[b + 1] += s * l_ir[b + 1];
                    l_slice[b + 2] += s * l_ir[b + 2];
                    l_slice[b + 3] += s * l_ir[b + 3];

                    r_slice[b] += s * r_ir[b];
                    r_slice[b + 1] += s * r_ir[b + 1];
                    r_slice[b + 2] += s * r_ir[b + 2];
                    r_slice[b + 3] += s * r_ir[b + 3];
                }
                for j in (active_chunks * 4)..max_j {
                    l_slice[j] += s * l_ir[j];
                    r_slice[j] += s * r_ir[j];
                }
            }

            // Attenuate by 1 / distance (inverse square law for sound pressure)
            let atten = 1.0 / pos.distance_m.max(0.1);
            for sample in &mut left_out[..out_len] {
                *sample *= atten;
            }
            for sample in &mut right_out[..out_len] {
                *sample *= atten;
            }
        } else {
            // Fallback: simple stereo panning based on azimuth
            let pan = (pos.azimuth_deg / 180.0).clamp(-1.0, 1.0);
            let left_gain = ((1.0 - pan) * 0.5).sqrt();
            let right_gain = ((1.0 + pan) * 0.5).sqrt();

            for (i, &s) in input.iter().enumerate().take(left_out.len().min(right_out.len())) {
                left_out[i] = s * left_gain;
                right_out[i] = s * right_gain;
            }
        }
    }

    /// Performs binaural convolution of a mono audio buffer into a stereo (left, right) buffer.
    pub fn spatialize_mono(&self, input: &[f32], pos: SphericalPosition) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        if let Some(hrir) = self.find_nearest_hrir(pos) {
            let ir_len = hrir.left_ir.len().min(hrir.right_ir.len());
            let out_len = n + ir_len.saturating_sub(1);
            let mut left_out = vec![0.0f32; out_len];
            let mut right_out = vec![0.0f32; out_len];
            self.spatialize_mono_into(input, pos, &mut left_out, &mut right_out);
            (left_out, right_out)
        } else {
            let mut left_out = vec![0.0f32; n];
            let mut right_out = vec![0.0f32; n];
            self.spatialize_mono_into(input, pos, &mut left_out, &mut right_out);
            (left_out, right_out)
        }
    }

    /// Loads a custom impulse response from raw file bytes (`.wav` or `.sofa` JSON).
    /// Stores the raw bytes into `spodeian-cache` for zero-copy reuse and updates the active HRIR.
    pub fn load_custom_ir_from_bytes(
        &mut self,
        file_name: &str,
        data: &[u8],
        cache_dir: Option<&std::path::Path>,
    ) -> Result<CustomIrMetadata, String> {
        if data.is_empty() {
            return Err("Custom IR file payload is empty".to_string());
        }

        let is_wav = file_name.to_lowercase().ends_with(".wav");
        let mut left_ir = Vec::new();
        let mut right_ir = Vec::new();
        let mut sample_rate = self.sample_rate;
        let mut channels = 1;

        if is_wav {
            let cursor = std::io::Cursor::new(data);
            let mut reader = hound::WavReader::new(cursor)
                .map_err(|e| format!("Invalid WAV format: {}", e))?;
            let spec = reader.spec();
            channels = spec.channels as usize;
            sample_rate = spec.sample_rate;

            let raw_samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => reader
                    .samples::<f32>()
                    .filter_map(Result::ok)
                    .collect(),
                hound::SampleFormat::Int => {
                    let max_val = (1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
                    reader
                        .samples::<i32>()
                        .filter_map(Result::ok)
                        .map(|s| s as f32 / max_val)
                        .collect()
                }
            };

            if raw_samples.is_empty() {
                return Err("WAV file contains no decodable audio samples".to_string());
            }

            if channels == 1 {
                left_ir = raw_samples.clone();
                right_ir = raw_samples;
            } else {
                for (i, &s) in raw_samples.iter().enumerate() {
                    if i % channels == 0 {
                        left_ir.push(s);
                    } else if i % channels == 1 {
                        right_ir.push(s);
                    }
                }
                if right_ir.is_empty() {
                    right_ir = left_ir.clone();
                }
            }
        } else {
            // Assume JSON/SOFA formatted impulse response coefficients
            if let Ok(val) = serde_json::from_slice::<serde_json::Value>(data) {
                let left_val = val.get("left_ir").or_else(|| val.get("left"));
                let right_val = val.get("right_ir").or_else(|| val.get("right"));

                if let Some(arr) = left_val.and_then(|v| v.as_array()) {
                    left_ir = arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
                    if let Some(r_arr) = right_val.and_then(|v| v.as_array()) {
                        right_ir = r_arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
                    } else {
                        right_ir = left_ir.clone();
                    }
                    if let Some(sr) = val.get("sample_rate").or_else(|| val.get("SampleRate")).and_then(|v| v.as_u64()) {
                        sample_rate = sr as u32;
                    }
                    channels = 2;
                } else if let Some(arr) = val.as_array() {
                    left_ir = arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
                    right_ir = left_ir.clone();
                } else {
                    return Err("Unsupported SOFA/JSON IR schema".to_string());
                }
            } else {
                return Err("Could not parse file as WAV or SOFA/JSON".to_string());
            }
        }

        let sample_count = left_ir.len().max(right_ir.len());
        if sample_count == 0 {
            return Err("Decoded impulse response contains 0 samples".to_string());
        }

        // Store into spodeian-cache CAS for zero-copy reuse
        let sha256_hash = spodeian_cache::ContentAddressedStorage::compute_sha256(data);
        let tier = spodeian_cache::PreferentialRouter::determine_tier(data.len(), "audio/wav", true);

        if let Some(dir) = cache_dir {
            if let Ok(cas) = spodeian_cache::ContentAddressedStorage::new(dir) {
                let _ = cas.put(data);
            }
        }

        // Replace default center HRIR with user custom impulse response
        self.impulse_responses.retain(|h| h.position != SphericalPosition::new(0.0, 0.0, 1.0));
        self.add_hrir(SphericalPosition::new(0.0, 0.0, 1.0), left_ir, right_ir);

        Ok(CustomIrMetadata {
            name: file_name.to_string(),
            sha256_hash,
            format: if is_wav { "wav".to_string() } else { "sofa".to_string() },
            sample_rate,
            channels,
            sample_count,
            byte_size: data.len(),
            tier,
        })
    }
}

/// Metadata describing custom impulse response (IR) files for HRTF spatialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomIrMetadata {
    pub name: String,
    pub sha256_hash: String,
    pub format: String,
    pub sample_rate: u32,
    pub channels: usize,
    pub sample_count: usize,
    pub byte_size: usize,
    pub tier: spodeian_cache::StorageTier,
}

/// Binaural Ambisonic decoding algorithms supported by the spatializer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum BinauralDecoderMode {
    /// 32-tap time-domain FIR filterbank based on Google Resonance Audio.
    ResonanceAudio32Tap = 0,
    /// Custom measured SOFA or WAV HRIR direct convolution.
    SofaCustomIr = 1,
    /// Fast stereo speaker downmix without HRTF coloration.
    StereoDownmix = 2,
}

impl BinauralDecoderMode {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::ResonanceAudio32Tap,
            1 => Self::SofaCustomIr,
            2 => Self::StereoDownmix,
            _ => Self::ResonanceAudio32Tap,
        }
    }

    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Lock-free atomic holder for runtime binaural decoder mode switching.
#[derive(Debug)]
pub struct AtomicBinauralDecoderMode {
    inner: std::sync::atomic::AtomicU8,
}

impl AtomicBinauralDecoderMode {
    pub fn new(mode: BinauralDecoderMode) -> Self {
        Self {
            inner: std::sync::atomic::AtomicU8::new(mode.to_u8()),
        }
    }

    pub fn load(&self, order: std::sync::atomic::Ordering) -> BinauralDecoderMode {
        BinauralDecoderMode::from_u8(self.inner.load(order))
    }

    pub fn store(&self, mode: BinauralDecoderMode, order: std::sync::atomic::Ordering) {
        self.inner.store(mode.to_u8(), order);
    }

    pub fn swap(&self, mode: BinauralDecoderMode, order: std::sync::atomic::Ordering) -> BinauralDecoderMode {
        BinauralDecoderMode::from_u8(self.inner.swap(mode.to_u8(), order))
    }
}

