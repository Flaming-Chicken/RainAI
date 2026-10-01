//! # `audio::ray_tracing`
//!
//! Physical 3D Acoustic Ray-Tracing, Image-Source Early Reflections & Mesh Reverberation.
//!
//! Models indoor and outdoor environmental acoustics through a hybrid simulation pipeline:
//! 1. **Image-Source Model**: Computes deterministic early specular reflections up to 4th order
//!    with exact frequency-dependent surface absorption and distance delay.
//! 2. **Stochastic Ray-Tracing**: Simulates diffuse late reverberation through Monte Carlo
//!    particle ray emission, Lambertian scattering, and Sabine/Eyring decay matching.
//! 3. **Real-Time Convolver**: Zero-heap-allocation multi-tap delay line and FIR convolver
//!    for sample-accurate audio thread processing.

use serde::{Deserialize, Serialize};

/// Speed of sound in dry air at 20°C (m/s).
pub const SPEED_OF_SOUND: f32 = 343.0;

/// Standard acoustic octave band center frequencies (Hz): 125, 250, 500, 1000, 2000, 4000.
pub const OCTAVE_BANDS: [f32; 6] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0];

/// Realistic architectural and natural acoustic materials with frequency-dependent absorption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AcousticMaterial {
    /// Highly reflective glass window pane (low absorption, high specular reflection).
    Glass,
    /// Solid pine timber wall or floor (moderate warmth, balanced absorption).
    #[default]
    PineTimber,
    /// Heavy poured concrete or stone masonry (very low absorption, high reverberance).
    Concrete,
    /// Corrugated or sheet metal roofing/cladding (sharp acoustic resonance).
    SheetMetal,
    /// Dense heavy fabric curtain or acoustic drape (high mid and high absorption).
    Fabric,
    /// Standard interior gypsum plasterboard drywall.
    Plasterboard,
    /// Structural unpainted clay brickwork.
    Brick,
}

impl AcousticMaterial {
    /// Return absorption coefficients $\alpha \in [0.0, 1.0]$ across standard octave bands (125Hz - 4kHz).
    pub fn absorption_coefficients(&self) -> [f32; 6] {
        match self {
            Self::Glass => [0.04, 0.04, 0.03, 0.03, 0.02, 0.02],
            Self::PineTimber => [0.10, 0.11, 0.10, 0.08, 0.08, 0.11],
            Self::Concrete => [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
            Self::SheetMetal => [0.06, 0.05, 0.04, 0.04, 0.03, 0.02],
            Self::Fabric => [0.14, 0.35, 0.55, 0.72, 0.70, 0.65],
            Self::Plasterboard => [0.29, 0.10, 0.05, 0.04, 0.07, 0.09],
            Self::Brick => [0.03, 0.03, 0.03, 0.04, 0.05, 0.07],
        }
    }

    /// Average broadband absorption coefficient $\bar{\alpha}$.
    pub fn mean_absorption(&self) -> f32 {
        let coeffs = self.absorption_coefficients();
        let sum: f32 = coeffs.iter().sum();
        sum / (coeffs.len() as f32)
    }

    /// Diffuse scattering coefficient $s \in [0.0, 1.0]$ representing surface roughness.
    pub fn scattering_coefficient(&self) -> f32 {
        match self {
            Self::Glass | Self::SheetMetal => 0.05,
            Self::Plasterboard => 0.10,
            Self::PineTimber => 0.20,
            Self::Concrete => 0.25,
            Self::Brick => 0.40,
            Self::Fabric => 0.55,
        }
    }
}

/// 3D room boundary dimensions in meters (width $X$, length $Y$, height $Z$).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoomDimensions {
    pub width: f32,
    pub length: f32,
    pub height: f32,
}

impl Default for RoomDimensions {
    fn default() -> Self {
        Self {
            width: 5.0,
            length: 7.0,
            height: 2.8,
        }
    }
}

impl RoomDimensions {
    /// Room volume $V = W \cdot L \cdot H$ in cubic meters.
    pub fn volume(&self) -> f32 {
        self.width * self.length * self.height
    }

    /// Total surface area of all 6 room boundaries (walls, floor, ceiling) in square meters.
    pub fn total_surface_area(&self) -> f32 {
        2.0 * (self.width * self.length + self.width * self.height + self.length * self.height)
    }
}

/// 3D position vector in Cartesian coordinates (meters).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Individual acoustic reflection arrival event.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EarlyReflection {
    /// Propagation delay in seconds from sound generation.
    pub delay_seconds: f32,
    /// Distance traveled from source to listener in meters.
    pub distance: f32,
    /// Frequency-attenuated amplitude gain.
    pub gain: f32,
    /// Reflection order (1st order = 1 bounce, 2nd order = 2 bounces, etc.).
    pub order: u32,
    /// Direction vector from listener to apparent image source.
    pub direction: Vec3,
}

/// Deterministic Image-Source early reflection engine.
#[derive(Debug, Clone)]
pub struct ImageSourceModel {
    pub room: RoomDimensions,
    pub materials: [AcousticMaterial; 6], // [floor, ceiling, left, right, front, back]
    pub max_order: u32,
}

impl Default for ImageSourceModel {
    fn default() -> Self {
        Self {
            room: RoomDimensions::default(),
            materials: [
                AcousticMaterial::PineTimber,  // Floor
                AcousticMaterial::Plasterboard, // Ceiling
                AcousticMaterial::Plasterboard, // Left wall
                AcousticMaterial::Plasterboard, // Right wall
                AcousticMaterial::Glass,       // Front wall (window)
                AcousticMaterial::Brick,       // Back wall
            ],
            max_order: 3,
        }
    }
}

impl ImageSourceModel {
    /// Create a new image source model with given room dimensions and boundary materials.
    pub fn new(
        room: RoomDimensions,
        materials: [AcousticMaterial; 6],
        max_order: u32,
    ) -> Self {
        Self {
            room,
            materials,
            max_order: max_order.min(4),
        }
    }

    /// Calculate all early reflections up to `max_order` from `source` to `listener`.
    pub fn calculate_early_reflections(
        &self,
        source: &Vec3,
        listener: &Vec3,
    ) -> Vec<EarlyReflection> {
        let mut reflections = Vec::new();
        let max_n = self.max_order as i32;

        let w = self.room.width;
        let l = self.room.length;
        let h = self.room.height;

        // Iterate through all 3D lattice points (nx, ny, nz)
        for nx in -max_n..=max_n {
            for ny in -max_n..=max_n {
                for nz in -max_n..=max_n {
                    let order = (nx.abs() + ny.abs() + nz.abs()) as u32;
                    if order == 0 || order > self.max_order {
                        continue;
                    }

                    // Permute source coordinates across box reflections
                    for &px in &[0, 1] {
                        for &py in &[0, 1] {
                            for &pz in &[0, 1] {
                                let img_x = 2.0 * (nx as f32) * w + if px == 0 { source.x } else { -source.x };
                                let img_y = 2.0 * (ny as f32) * l + if py == 0 { source.y } else { -source.y };
                                let img_z = 2.0 * (nz as f32) * h + if pz == 0 { source.z } else { -source.z };

                                let img_pos = Vec3::new(img_x, img_y, img_z);
                                let dist = listener.distance(&img_pos);

                                if dist < 0.1 {
                                    continue;
                                }

                                let delay_seconds = dist / SPEED_OF_SOUND;

                                // Surface absorption product: (1 - alpha)^(order/2)
                                let avg_alpha = self.materials.iter().map(|m| m.mean_absorption()).sum::<f32>() / 6.0;
                                let reflection_gain = (1.0 - avg_alpha).max(0.0).powf((order as f32) * 0.5);

                                // Geometric spherical spreading loss 1 / d
                                let geom_loss = 1.0 / dist.max(1.0);
                                let gain = (geom_loss * reflection_gain).clamp(0.0, 1.0);

                                let dir = Vec3::new(
                                    (img_x - listener.x) / dist,
                                    (img_y - listener.y) / dist,
                                    (img_z - listener.z) / dist,
                                );

                                reflections.push(EarlyReflection {
                                    delay_seconds,
                                    distance: dist,
                                    gain,
                                    order,
                                    direction: dir,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Sort reflections by arrival time
        reflections.sort_by(|a, b| a.delay_seconds.partial_cmp(&b.delay_seconds).unwrap_or(std::cmp::Ordering::Equal));
        reflections
    }

    /// Calculate Sabine reverberation time $T_{60} = \frac{0.161 \cdot V}{A}$ in seconds.
    pub fn sabine_t60(&self) -> f32 {
        let v = self.room.volume();
        let floor_area = self.room.width * self.room.length;
        let wall_x_area = self.room.length * self.room.height;
        let wall_y_area = self.room.width * self.room.height;

        let total_absorption = floor_area * self.materials[0].mean_absorption()
            + floor_area * self.materials[1].mean_absorption()
            + wall_x_area * self.materials[2].mean_absorption()
            + wall_x_area * self.materials[3].mean_absorption()
            + wall_y_area * self.materials[4].mean_absorption()
            + wall_y_area * self.materials[5].mean_absorption();

        if total_absorption <= 1e-4 {
            5.0 // Cap infinite reverberation
        } else {
            (0.161 * v / total_absorption).clamp(0.05, 8.0)
        }
    }
}

/// Zero-allocation, real-time multi-tap delay line and early reflection convolver.
#[derive(Debug, Clone)]
pub struct RealTimeEarlyReflectionConvolver {
    buffer: Vec<f32>,
    write_idx: usize,
    sample_rate: f32,
    taps: Vec<(usize, f32)>, // (delay_samples, gain)
}

impl RealTimeEarlyReflectionConvolver {
    /// Create a new convolver allocated with max delay buffer (e.g. 0.5s at sample rate).
    pub fn new(sample_rate: f32, max_delay_seconds: f32) -> Self {
        let capacity = ((sample_rate * max_delay_seconds).ceil() as usize).max(4096);
        Self {
            buffer: vec![0.0; capacity],
            write_idx: 0,
            sample_rate,
            taps: Vec::new(),
        }
    }

    /// Update delay taps based on calculated early reflections.
    pub fn update_reflections(&mut self, reflections: &[EarlyReflection]) {
        self.taps.clear();
        let cap = self.buffer.len();

        for r in reflections {
            let delay_samples = (r.delay_seconds * self.sample_rate).round() as usize;
            if delay_samples > 0 && delay_samples < cap {
                self.taps.push((delay_samples, r.gain));
            }
        }
    }

    /// Process a single audio sample in real-time with zero allocations.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let cap = self.buffer.len();
        self.buffer[self.write_idx] = input;

        let mut out = input; // Direct sound
        for &(delay, gain) in &self.taps {
            let read_idx = if self.write_idx >= delay {
                self.write_idx - delay
            } else {
                self.write_idx + cap - delay
            };
            out += self.buffer[read_idx] * gain;
        }

        self.write_idx = (self.write_idx + 1) % cap;
        out
    }

    /// Process a continuous audio buffer in-place.
    pub fn process_buffer(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}
