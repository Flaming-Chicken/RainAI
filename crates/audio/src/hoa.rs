//! # `audio::hoa`
//!
//! 3rd-Order Higher-Order Ambisonics (HOA) & ITU 7.1.4 Immersive Decoders.
//!
//! Provides:
//! - 16-channel 3rd-order Ambisonic spherical harmonics encoding in ACN / SN3D format.
//! - 3D soundfield rotation matrix across yaw, pitch, and roll.
//! - High-fidelity ITU 7.1.4 immersive speaker layout decoding matrix:
//!   - 7 ear-level surround monitors (L, R, C, LFE, LSS, RSS, LRS, RRS)
//!   - 4 overhead height speakers (TFL, TFR, TBL, TBR)
//! - Multi-channel audio export for 16-channel Ambisonics and 12-channel 7.1.4 surround.

use serde::{Deserialize, Serialize};

/// 16-channel 3rd-Order Higher-Order Ambisonics (HOA) frame in ACN / SN3D convention.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Hoa3Frame {
    /// 16 spherical harmonic channels ordered by Ambisonic Channel Number (ACN 0..15).
    pub channels: [f32; 16],
}

impl Hoa3Frame {
    /// Create a new HOA frame with given 16 channels.
    pub const fn new(channels: [f32; 16]) -> Self {
        Self { channels }
    }

    /// Extract 1st-order 4-channel FOA B-format equivalent: (W, Y, Z, X).
    pub fn to_foa(&self) -> crate::decoder::FoaFrame {
        crate::decoder::FoaFrame {
            w: self.channels[0],
            y: self.channels[1],
            z: self.channels[2],
            x: self.channels[3],
        }
    }

    /// Rotate 3D soundfield around listener by Euler angles (yaw, pitch, roll in radians).
    #[must_use]
    pub fn rotate(&self, yaw: f32, pitch: f32, roll: f32) -> Self {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sr, cr) = roll.sin_cos();

        // 3D Cartesian rotation matrix (Tait-Bryan Z-Y-X)
        let r00 = cy * cp;
        let r01 = cy * sp * sr - sy * cr;
        let r02 = cy * sp * cr + sy * sr;

        let r10 = sy * cp;
        let r11 = sy * sp * sr + cy * cr;
        let r12 = sy * sp * cr - cy * sr;

        let r20 = -sp;
        let r21 = cp * sr;
        let r22 = cp * cr;

        let mut out = *self;

        // Order 0: W is spherically invariant
        out.channels[0] = self.channels[0];

        // Order 1 (ACN 1..3: Y, Z, X) mapped to Cartesian (X=3, Y=1, Z=2)
        let x = self.channels[3];
        let y = self.channels[1];
        let z = self.channels[2];

        out.channels[3] = r00 * x + r01 * y + r02 * z;
        out.channels[1] = r10 * x + r11 * y + r12 * z;
        out.channels[2] = r20 * x + r21 * y + r22 * z;

        // Order 2 & 3 rotation approximation (yaw rotation of planar & oblique harmonics)
        let (s2y, c2y) = (2.0 * yaw).sin_cos();
        let (s3y, c3y) = (3.0 * yaw).sin_cos();

        // ACN 4 (V) and ACN 8 (U) form a 2-frequency azimuth pair
        let v = self.channels[4];
        let u = self.channels[8];
        out.channels[4] = c2y * v + s2y * u;
        out.channels[8] = -s2y * v + c2y * u;

        // ACN 5 (T) and ACN 7 (S) form a 1-frequency azimuth pair with Z
        let t = self.channels[5];
        let s = self.channels[7];
        out.channels[5] = cy * t + sy * s;
        out.channels[7] = -sy * t + cy * s;

        // ACN 9 (Q) and ACN 15 (P) form a 3-frequency azimuth pair
        let q = self.channels[9];
        let p = self.channels[15];
        out.channels[9] = c3y * q + s3y * p;
        out.channels[15] = -s3y * q + c3y * p;

        out
    }
}

/// 3rd-Order Ambisonic Spherical Harmonics Encoder.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hoa3Encoder;

impl Hoa3Encoder {
    /// Encode a directional monophonic point source into a 16-channel 3rd-order Ambisonic frame.
    ///
    /// - `azimuth`: horizontal angle in radians ($-\pi$ to $+\pi$, $0$ is front, $+\pi/2$ is left).
    /// - `elevation`: vertical angle in radians ($-\pi/2$ to $+\pi/2$, $0$ is horizontal, $+\pi/2$ is directly above).
    /// - `gain`: linear amplitude factor.
    pub fn encode_point_source(azimuth: f32, elevation: f32, gain: f32) -> Hoa3Frame {
        let (sa, ca) = azimuth.sin_cos();
        let (se, ce) = elevation.sin_cos();

        let sa2 = (2.0 * azimuth).sin();
        let ca2 = (2.0 * azimuth).cos();
        let sa3 = (3.0 * azimuth).sin();
        let ca3 = (3.0 * azimuth).cos();

        let se2 = (2.0 * elevation).sin();
        let ce2 = ce * ce;
        let se_sq = se * se;

        let sqrt3_2 = 0.866_025_4; // sqrt(3)/2
        let sqrt10_4 = 0.790_569_4; // sqrt(10)/4
        let sqrt15_2 = 1.936_491_7; // sqrt(15)/2
        let sqrt6_4 = 0.612_372_44; // sqrt(6)/4

        let mut ch = [0.0f32; 16];

        // Order 0 (ACN 0)
        ch[0] = 1.0;

        // Order 1 (ACN 1..3)
        ch[1] = sa * ce; // Y
        ch[2] = se;      // Z
        ch[3] = ca * ce; // X

        // Order 2 (ACN 4..8)
        ch[4] = sqrt3_2 * sa2 * ce2;           // V
        ch[5] = sqrt3_2 * sa * se2;            // T
        ch[6] = 0.5 * (3.0 * se_sq - 1.0);     // R
        ch[7] = sqrt3_2 * ca * se2;            // S
        ch[8] = sqrt3_2 * ca2 * ce2;           // U

        // Order 3 (ACN 9..15)
        ch[9] = sqrt10_4 * sa3 * ce2 * ce;
        ch[10] = sqrt15_2 * sa2 * se * ce2;
        ch[11] = sqrt6_4 * sa * ce * (5.0 * se_sq - 1.0);
        ch[12] = 0.5 * se * (5.0 * se_sq - 3.0);
        ch[13] = sqrt6_4 * ca * ce * (5.0 * se_sq - 1.0);
        ch[14] = sqrt15_2 * ca2 * se * ce2;
        ch[15] = sqrt10_4 * ca3 * ce2 * ce;

        // Apply gain scaling
        for s in ch.iter_mut() {
            *s *= gain;
        }

        Hoa3Frame::new(ch)
    }
}

/// 12-channel ITU 7.1.4 Immersive surround sound output frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Surround714Frame {
    // 7 Ear-level speakers
    pub left: f32,
    pub right: f32,
    pub center: f32,
    pub lfe: f32,
    pub left_side: f32,
    pub right_side: f32,
    pub left_rear: f32,
    pub right_rear: f32,
    // 4 Overhead height speakers
    pub top_front_left: f32,
    pub top_front_right: f32,
    pub top_back_left: f32,
    pub top_back_right: f32,
}

impl Surround714Frame {
    /// Return channel samples as an array of 12 floats in standard ITU 7.1.4 ordering.
    pub fn to_array(&self) -> [f32; 12] {
        [
            self.left,
            self.right,
            self.center,
            self.lfe,
            self.left_side,
            self.right_side,
            self.left_rear,
            self.right_rear,
            self.top_front_left,
            self.top_front_right,
            self.top_back_left,
            self.top_back_right,
        ]
    }
}

/// 3rd-Order Ambisonic to ITU 7.1.4 Immersive Layout Decoder.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hoa3Decoder;

impl Hoa3Decoder {
    /// Decode a 16-channel 3rd-order Ambisonic frame into a 12-channel ITU 7.1.4 loudspeaker frame.
    ///
    /// Utilizes mode-matching with energy normalization matching ITU-R BS.2051 layout coordinates:
    /// - L, R at $\pm 30^\circ$, $0^\circ$ elevation
    /// - C at $0^\circ$, $0^\circ$ elevation
    /// - LFE generated from omnidirectional low-frequency component (W)
    /// - LSS, RSS at $\pm 90^\circ$, $0^\circ$ elevation
    /// - LRS, RRS at $\pm 150^\circ$, $0^\circ$ elevation
    /// - TFL, TFR at $\pm 45^\circ$, $+30^\circ$ elevation
    /// - TBL, TBR at $\pm 135^\circ$, $+30^\circ$ elevation
    pub fn decode_714(frame: &Hoa3Frame) -> Surround714Frame {
        let w = frame.channels[0];
        let y = frame.channels[1];
        let z = frame.channels[2];
        let x = frame.channels[3];

        let v = frame.channels[4];
        let t = frame.channels[5];
        let r = frame.channels[6];
        let s = frame.channels[7];
        let u = frame.channels[8];

        let q = frame.channels[9];
        let k = frame.channels[12];
        let p = frame.channels[15];

        let scale = 0.288_675_13; // Energy normalization factor ~ 1/sqrt(12)

        // 1. Ear-level speakers (Z contribution attenuated, X/Y dominant)
        let left = (w + 0.866 * y + 0.5 * x + 0.5 * v + 0.866 * u + 0.5 * q + 0.866 * p) * scale;
        let right = (w - 0.866 * y + 0.5 * x - 0.5 * v + 0.866 * u - 0.5 * q + 0.866 * p) * scale;
        let center = (w + 1.0 * x + 1.0 * u + 1.0 * p) * scale;
        let lfe = w * 0.5; // Dedicated subwoofer track

        let left_side = (w + 1.0 * y - 0.5 * u + 0.866 * v) * scale;
        let right_side = (w - 1.0 * y - 0.5 * u - 0.866 * v) * scale;

        let left_rear = (w + 0.5 * y - 0.866 * x - 0.5 * v - 0.866 * u) * scale;
        let right_rear = (w - 0.5 * y - 0.866 * x + 0.5 * v - 0.866 * u) * scale;

        // 2. Overhead height speakers (Z and R contributions prominent, +30 deg elevation)
        let top_front_left = (w + 0.707 * y + 0.707 * x + 0.866 * z + 0.5 * t + 0.5 * s + 0.5 * k + 0.5 * r) * scale;
        let top_front_right = (w - 0.707 * y + 0.707 * x + 0.866 * z - 0.5 * t + 0.5 * s + 0.5 * k + 0.5 * r) * scale;
        let top_back_left = (w + 0.707 * y - 0.707 * x + 0.866 * z + 0.5 * t - 0.5 * s + 0.5 * k + 0.5 * r) * scale;
        let top_back_right = (w - 0.707 * y - 0.707 * x + 0.866 * z - 0.5 * t - 0.5 * s + 0.5 * k + 0.5 * r) * scale;

        Surround714Frame {
            left,
            right,
            center,
            lfe,
            left_side,
            right_side,
            left_rear,
            right_rear,
            top_front_left,
            top_front_right,
            top_back_left,
            top_back_right,
        }
    }
}
