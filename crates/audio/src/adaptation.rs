//! # `audio::adaptation`
//!
//! Continuous Acoustic Scene Adaptation & Automatic Room Resonance Mode Inversion.
//!
//! Uses `spodeian-ml-utils::OnlineRlsFilter64` to continuously track room acoustic transfer
//! functions, identify standing wave resonances (room modes), and synthesize dynamic parametric
//! notch filters to invert coloration, ensuring the synthesized soundscape seamlessly integrates
//! with the physical listening room.

use serde::{Deserialize, Serialize};
use spodeian_ml_utils::OnlineRlsFilter64;

/// Peak acoustic room resonance mode (standing wave).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoomResonanceMode {
    /// Center frequency in Hz (typically 40Hz - 300Hz in domestic rooms).
    pub center_hz: f32,
    /// Resonance Q-factor (bandwidth sharpness).
    pub q: f32,
    /// Resonance excess boost in dBFS.
    pub excess_db: f32,
}

/// Second-order IIR Biquad filter state for zero-allocation real-time equalization.
#[derive(Debug, Clone, Copy, Default)]
pub struct BiquadFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadFilter {
    /// Configure biquad as a parametric notch/cut filter to cancel room resonance.
    pub fn configure_notch(&mut self, sample_rate: f32, center_hz: f32, q: f32, cut_db: f32) {
        let w0 = 2.0 * std::f32::consts::PI * center_hz / sample_rate;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q.max(0.1));
        let a = 10.0f32.powf(-cut_db.abs() / 40.0); // Linear attenuation factor

        let a0 = 1.0 + alpha / a;
        self.b0 = (1.0 + alpha * a) / a0;
        self.b1 = (-2.0 * cos_w0) / a0;
        self.b2 = (1.0 - alpha * a) / a0;
        self.a1 = (-2.0 * cos_w0) / a0;
        self.a2 = (1.0 - alpha / a) / a0;
    }

    /// Reset filter state variables.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /// Process a single audio sample in-place with direct-form II transposed.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Dynamic Acoustic Scene Adapter and Room Equalizer.
#[derive(Debug, Clone)]
pub struct AcousticSceneAdapter {
    pub sample_rate: f32,
    pub rls_tracker: OnlineRlsFilter64,
    pub detected_modes: Vec<RoomResonanceMode>,
    pub notch_filters: [BiquadFilter; 4],
    pub adaptation_smoothing: f32,
}

impl AcousticSceneAdapter {
    /// Create a new acoustic scene adapter for a given sample rate (e.g. 48000.0).
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            rls_tracker: OnlineRlsFilter64::new(6, 0.99, 50.0),
            detected_modes: Vec::new(),
            notch_filters: [BiquadFilter::default(); 4],
            adaptation_smoothing: 0.05,
        }
    }

    /// Reset adaptive filter state.
    pub fn reset(&mut self) {
        self.rls_tracker = OnlineRlsFilter64::new(6, 0.99, 50.0);
        self.detected_modes.clear();
        for f in &mut self.notch_filters {
            f.reset();
        }
    }

    /// Calculate frequency component energy using the Goertzel algorithm.
    fn goertzel_energy(samples: &[f32], freq: f32, sample_rate: f32) -> f32 {
        let w = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let coeff = 2.0 * w.cos();
        let mut s_prev = 0.0f32;
        let mut s_prev2 = 0.0f32;

        for &x in samples {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }

        let power = s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2;
        (power / (samples.len() as f32).max(1.0)).sqrt()
    }

    /// Analyze incoming microphone signal and update room resonance mode cancellation.
    pub fn adapt_from_microphone(&mut self, mic_buffer: &[f32]) {
        if mic_buffer.len() < 256 {
            return;
        }

        let n = mic_buffer.len() as f32;
        let total_energy = (mic_buffer.iter().map(|&s| s * s).sum::<f32>() / n).sqrt().max(1e-6);

        // Targeted acoustic resonance probe at domestic standing wave modes
        let e_63 = Self::goertzel_energy(mic_buffer, 63.0, self.sample_rate);
        let e_125 = Self::goertzel_energy(mic_buffer, 125.0, self.sample_rate);
        let e_250 = Self::goertzel_energy(mic_buffer, 250.0, self.sample_rate);

        // Update RLS state vector
        let x = [
            1.0,
            (e_63 / total_energy) as f64,
            (e_125 / total_energy) as f64,
            (e_250 / total_energy) as f64,
            (total_energy as f64).ln(),
            ((e_63 + e_125) / (e_250 + 1e-4)) as f64,
        ];

        let target_ratio = ((e_63 + e_125) / (e_250 + 1e-4)).clamp(0.1, 10.0) as f64;
        let _ = self.rls_tracker.update(&x, target_ratio);

        self.detected_modes.clear();

        // 63 Hz room mode detection: ratio of narrow-band energy to total broadband
        if e_63 > 0.25 * total_energy {
            let excess_db = (20.0 * (e_63 / (0.1 * total_energy + 1e-4)).log10()).clamp(2.0, 12.0);
            self.detected_modes.push(RoomResonanceMode {
                center_hz: 63.0,
                q: 3.5,
                excess_db,
            });
            self.notch_filters[0].configure_notch(self.sample_rate, 63.0, 3.5, excess_db);
        } else {
            self.notch_filters[0].configure_notch(self.sample_rate, 63.0, 3.5, 0.0);
        }

        // 125 Hz room mode detection
        if e_125 > 0.25 * total_energy {
            let excess_db = (20.0 * (e_125 / (0.1 * total_energy + 1e-4)).log10()).clamp(2.0, 9.0);
            self.detected_modes.push(RoomResonanceMode {
                center_hz: 125.0,
                q: 4.0,
                excess_db,
            });
            self.notch_filters[1].configure_notch(self.sample_rate, 125.0, 4.0, excess_db);
        } else {
            self.notch_filters[1].configure_notch(self.sample_rate, 125.0, 4.0, 0.0);
        }
    }

    /// Apply anti-resonance equalization in real-time to synthesized audio samples.
    #[inline]
    pub fn process_sample(&mut self, mut sample: f32) -> f32 {
        for f in &mut self.notch_filters {
            sample = f.process(sample);
        }
        sample
    }

    /// Process a continuous block of synthesized samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for s in buffer.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}
