//! Sleep timer with smooth fade-out gain envelope and A/B + Favorites preset bank helpers.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Sleep timer state with configurable fade-out tail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SleepTimer {
    pub active: bool,
    pub total_secs: f32,
    pub remaining_secs: f32,
    pub fade_out_secs: f32,
}

impl Default for SleepTimer {
    fn default() -> Self {
        Self {
            active: false,
            total_secs: 1800.0,
            remaining_secs: 1800.0,
            fade_out_secs: 30.0,
        }
    }
}

impl SleepTimer {
    /// Starts a timer for `duration_secs` with a `fade_out_secs` fade tail.
    pub fn start(&mut self, duration_secs: f32, fade_out_secs: f32) {
        self.total_secs = duration_secs.max(1.0);
        self.remaining_secs = self.total_secs;
        self.fade_out_secs = fade_out_secs.clamp(0.0, self.total_secs);
        self.active = true;
    }

    /// Advances the timer by `dt_secs` and returns `(current_gain, finished)`.
    pub fn advance(&mut self, dt_secs: f32) -> (f32, bool) {
        if !self.active {
            return (1.0, false);
        }
        self.remaining_secs = (self.remaining_secs - dt_secs.max(0.0)).max(0.0);
        if self.remaining_secs <= 0.0 {
            self.active = false;
            return (0.0, true);
        }
        if self.fade_out_secs > 0.0 && self.remaining_secs < self.fade_out_secs {
            let t = (self.remaining_secs / self.fade_out_secs).clamp(0.0, 1.0);
            // Smooth cosine fade-out
            let gain = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
            return (gain, false);
        }
        (1.0, false)
    }
}

/// A/B comparison slot + favorites tracking for soundscape presets.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetBankState {
    pub slot_a: Option<String>,
    pub slot_b: Option<String>,
    pub active_is_b: bool,
    pub favorites: BTreeSet<String>,
}

impl PresetBankState {
    pub fn toggle_ab(&mut self) -> Option<&str> {
        self.active_is_b = !self.active_is_b;
        if self.active_is_b {
            self.slot_b.as_deref()
        } else {
            self.slot_a.as_deref()
        }
    }

    pub fn toggle_favorite(&mut self, preset_id: &str) -> bool {
        if self.favorites.contains(preset_id) {
            self.favorites.remove(preset_id);
            false
        } else {
            self.favorites.insert(preset_id.to_string());
            true
        }
    }
}
