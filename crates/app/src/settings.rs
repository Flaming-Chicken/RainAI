//! # `app::settings`
//!
//! Centralized Settings Store & Schema Migration Foundations.
//!
//! Decouples persistent configuration from live audio state and guarantees
//! non-destructive backwards-compatible schema upgrades.

use audio::derivation::{DerivedParameters, ListeningSetup, SoundLayers, derive_parameters};
use inference::compute_router::FlowSolverAlgorithm;
use serde::{Deserialize, Serialize};
use shared::rain::{GovernorOptimizationProfile, QualityTier};

/// Current session schema format version.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Central persistent user settings state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SettingsState {
    /// Schema format version for migrations
    pub schema_version: u32,

    // High-Level User Intent
    pub profile: GovernorOptimizationProfile,
    pub listening_setup: ListeningSetup,
    pub layers: SoundLayers,
    pub evolve: bool,
    pub ai_enabled: bool,

    // Audio & Acoustic DSP Settings
    pub master_volume: f32,
    pub noise_masking_enabled: bool,
    pub noise_masking_threshold_db: f32,
    pub hrtf_profile: String,
    pub custom_ir_hash: Option<String>,

    // Manual Advanced Overrides (None = Auto-Derived from Profile + Intent)
    pub manual_buffer_ms: Option<f32>,
    pub manual_solver: Option<FlowSolverAlgorithm>,
    pub manual_solver_tolerance: Option<f32>,
    pub manual_waveshaper_tier: Option<QualityTier>,
    pub manual_droplet_voice_cap: Option<usize>,

    // UI & Display Preferences
    pub show_spectrogram: bool,
    pub spectrogram_height: f32,
    pub show_provenance_dock: bool,
    pub show_advanced_settings: bool,
    pub active_preset_name: String,
}

impl Default for SettingsState {
    fn default() -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            profile: GovernorOptimizationProfile::BalancedAdaptive,
            listening_setup: ListeningSetup::DesktopSpeakers, // Desktop speakers default format
            layers: SoundLayers::default(), // Raindrops: true, Rain wash: true, AI texture: false
            evolve: false,                  // Off by default
            ai_enabled: false,              // Off by default (models training)
            master_volume: 0.60,
            noise_masking_enabled: false,
            noise_masking_threshold_db: -40.0,
            hrtf_profile: "Kemar-Compact-Standard".to_string(),
            custom_ir_hash: None,
            manual_buffer_ms: None,
            manual_solver: None,
            manual_solver_tolerance: None,
            manual_waveshaper_tier: None,
            manual_droplet_voice_cap: None,
            show_spectrogram: true,
            spectrogram_height: 120.0,
            show_provenance_dock: false,
            show_advanced_settings: false,
            active_preset_name: "Gentle Summer Rain".to_string(),
        }
    }
}

impl SettingsState {
    /// Evaluates high-level choices into active parameters, applying any explicit manual overrides.
    pub fn resolve_parameters(&self) -> DerivedParameters {
        let mut derived = derive_parameters(self.profile, self.listening_setup, self.layers);

        if let Some(buf_ms) = self.manual_buffer_ms {
            derived.target_buffer_ms = buf_ms;
        }
        if let Some(solver) = self.manual_solver {
            derived.flow_solver = solver;
        }
        if let Some(tol) = self.manual_solver_tolerance {
            derived.solver_tolerance = tol;
        }
        if let Some(tier) = self.manual_waveshaper_tier {
            derived.waveshaper_tier = tier;
        }
        if let Some(cap) = self.manual_droplet_voice_cap {
            derived.droplet_voice_cap = cap;
        }

        derived
    }

    /// Resets all manual overrides back to `Auto (Derived)`.
    pub fn reset_advanced_overrides_to_auto(&mut self) {
        self.manual_buffer_ms = None;
        self.manual_solver = None;
        self.manual_solver_tolerance = None;
        self.manual_waveshaper_tier = None;
        self.manual_droplet_voice_cap = None;
    }
}
