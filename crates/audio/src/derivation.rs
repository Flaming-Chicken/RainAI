//! # `audio::derivation`
//!
//! Automatic Inference & Parameter Derivation Engine.
//!
//! High-level user intent (Performance Profile, Listening Setup, Sound Layers)
//! automatically dictates low-level DSP parameters, audio buffer sizing, neural solver
//! tolerances, and droplet voice densities, eliminating manual guessing and legacy governors.

use crate::decoder::DecodeMode;
use inference::compute_router::FlowSolverAlgorithm;
use serde::{Deserialize, Serialize};
use shared::rain::{GovernorOptimizationProfile, QualityTier};

/// High-level listening environment configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ListeningSetup {
    /// Stereo desktop monitor speakers (default setup)
    #[default]
    DesktopSpeakers,
    /// Binaural virtual acoustics with HRTF filtering
    BinauralHeadphones,
    /// Wireless Bluetooth (A2DP) with +100ms jitter reserve
    WirelessBluetooth,
    /// Surround sound 7.1 home theater array
    Surround71,
    /// Raw Ambisonic B-format passthrough
    RawFoaPassthrough,
}

impl ListeningSetup {
    pub fn label(self) -> &'static str {
        match self {
            Self::DesktopSpeakers => "Desktop Speakers (Default)",
            Self::BinauralHeadphones => "Headphones (Binaural HRTF)",
            Self::WirelessBluetooth => "Wireless (Bluetooth A2DP)",
            Self::Surround71 => "Home Theater (7.1 Surround)",
            Self::RawFoaPassthrough => "Ambisonic Passthrough (VR / Spatial)",
        }
    }

    /// Converts to Ambisonic DecodeMode
    pub fn to_decode_mode(self) -> DecodeMode {
        match self {
            Self::DesktopSpeakers => DecodeMode::StereoSpeakers,
            Self::BinauralHeadphones => DecodeMode::BinauralHeadphones,
            Self::WirelessBluetooth => DecodeMode::BinauralHeadphones,
            Self::Surround71 => DecodeMode::Surround71,
            Self::RawFoaPassthrough => DecodeMode::RawFoaPassthrough,
        }
    }

    /// From DecodeMode and bluetooth flag
    pub fn from_decode_mode(mode: DecodeMode, is_wireless: bool) -> Self {
        if is_wireless {
            Self::WirelessBluetooth
        } else {
            match mode {
                DecodeMode::StereoSpeakers => Self::DesktopSpeakers,
                DecodeMode::BinauralHeadphones => Self::BinauralHeadphones,
                DecodeMode::Surround71 => Self::Surround71,
                DecodeMode::RawFoaPassthrough => Self::RawFoaPassthrough,
            }
        }
    }
}

pub use shared::rain::SoundLayers;

/// Fully derived low-level audio, DSP, and neural solver parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedParameters {
    /// Target ringbuffer duration in milliseconds
    pub target_buffer_ms: f32,
    /// Neural waveshaper quantization tier
    pub waveshaper_tier: QualityTier,
    /// Neural flow ODE solver algorithm & tolerance
    pub flow_solver: FlowSolverAlgorithm,
    /// Local error truncation tolerance for adaptive solvers
    pub solver_tolerance: f32,
    /// Maximum simultaneous physical drop impulses
    pub droplet_voice_cap: usize,
    /// Mute procedural noise-burst bands 0..=9 when physical raindrop synthesizer is active
    pub suppress_procedural_droplets: bool,
}

/// Evaluates high-level user configuration to automatically derive optimal parameters.
pub fn derive_parameters(
    profile: GovernorOptimizationProfile,
    listening_setup: ListeningSetup,
    layers: SoundLayers,
) -> DerivedParameters {
    // 1. Buffer Size & Latency calculation
    let base_buffer_ms = match profile {
        GovernorOptimizationProfile::EcoBatterySaver => 60.0,
        GovernorOptimizationProfile::LowLatencyInteractive => 15.0,
        GovernorOptimizationProfile::BalancedAdaptive => 45.0,
        GovernorOptimizationProfile::StudioMaster => 120.0,
        GovernorOptimizationProfile::BluetoothA2DPSink => 150.0,
    };

    let target_buffer_ms = if listening_setup == ListeningSetup::WirelessBluetooth {
        base_buffer_ms + 100.0 // +100ms jitter safety reserve
    } else {
        base_buffer_ms
    };

    // 2. Waveshaper Precision / Quantization
    let waveshaper_tier = match profile {
        GovernorOptimizationProfile::EcoBatterySaver => QualityTier::Ternary158,
        GovernorOptimizationProfile::LowLatencyInteractive
        | GovernorOptimizationProfile::BalancedAdaptive
        | GovernorOptimizationProfile::BluetoothA2DPSink => QualityTier::AdaptiveMinimum,
        GovernorOptimizationProfile::StudioMaster => QualityTier::StudioFp32,
    };

    // 3. Flow Solver Configuration
    let (flow_solver, solver_tolerance) = match profile {
        GovernorOptimizationProfile::EcoBatterySaver => (
            FlowSolverAlgorithm::AdaptiveRk23 {
                tol: 1e-2,
                initial_h: 0.2,
            },
            1e-2,
        ),
        GovernorOptimizationProfile::LowLatencyInteractive => (
            FlowSolverAlgorithm::AdaptiveHeun2 {
                tol: 5e-3,
                initial_h: 0.05,
            },
            5e-3,
        ),
        GovernorOptimizationProfile::BalancedAdaptive
        | GovernorOptimizationProfile::BluetoothA2DPSink => (
            FlowSolverAlgorithm::AdaptiveRk45 {
                tol: 1e-3,
                initial_h: 0.1,
            },
            1e-3,
        ),
        GovernorOptimizationProfile::StudioMaster => (
            FlowSolverAlgorithm::AdaptiveTsit5 {
                tol: 1e-4,
                initial_h: 0.05,
            },
            1e-4,
        ),
    };

    // 4. Droplet Voice Density
    let droplet_voice_cap = match profile {
        GovernorOptimizationProfile::EcoBatterySaver => 128,
        GovernorOptimizationProfile::LowLatencyInteractive => 256,
        GovernorOptimizationProfile::BalancedAdaptive
        | GovernorOptimizationProfile::BluetoothA2DPSink => 512,
        GovernorOptimizationProfile::StudioMaster => 2048,
    };

    // 5. Procedural Droplet Band Suppression (DSP Partitioning)
    // When physical raindrops are active, suppress procedural noise-burst bands 0..=9
    let suppress_procedural_droplets = layers.raindrops;

    DerivedParameters {
        target_buffer_ms,
        waveshaper_tier,
        flow_solver,
        solver_tolerance,
        droplet_voice_cap,
        suppress_procedural_droplets,
    }
}
