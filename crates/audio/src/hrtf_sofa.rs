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
}
