//! 64-Dimensional Hierarchical Conditioning Vector Schema, Layout Definitions,
//! and Bidirectional Text/Audio Inversion Engine.
//!
//! Provides canonical slice ranges and zero-heap encoding for the continuous conditioning
//! manifold shared between neural training (Mamba2 MoE, Spatial VAE) and runtime inference.
//!
//! Total dimensionality:
//! - 0..10  (10 dims): Macro Atmosphere & Controls
//! - 10..17 (7 dims):  Continuous Physical Material Properties (Hardness, Resonance, Damping, Impedance, Roughness, Water Depth, Cavity)
//! - 17..21 (4 dims):  Aeroacoustic Wind Dynamics
//! - 21..39 (18 dims): Spatialized Side Acoustic Sounds
//! - 39     (1 dim):   Physical Parameter Drift
//! - 40..46 (6 dims):  Physical Acoustic Summary Metrics
//! - 46..64 (18 dims): Expanded Open Semantic Projection
//! Total = 64 dimensions.

use std::ops::Range;

/// Total dimensionality of the continuous conditioning vector:
/// 10 (Macro Atmosphere) + 7 (Material Continuum) + 4 (Wind) + 18 (Side Sounds)
/// + 1 (Drift) + 6 (Acoustic Metrics) + 18 (Semantic Projection) = 64.
pub const CONDITION_DIM: usize = 64;

/// Precise slice layout of the 64-dimensional conditioning vector.
pub struct ConditioningLayout;

impl ConditioningLayout {
    /// 10-dimensional macro atmosphere & controls:
    /// [0: intensity, 1: runoff, 2: temperature, 3: humidity, 4: pitch_angle,
    ///  5: distance, 6: enclosure, 7: air_absorption, 8: wind_speed, 9: wind_azimuth]
    pub const MACRO_ATMOSPHERE: Range<usize> = 0..10;

    /// Alias for backwards compatibility.
    pub const BASE_CONTROLS: Range<usize> = 0..10;

    /// 7-dimensional continuous physical material properties (replaces rigid discrete surface bins):
    /// [10: hardness, 11: resonance_freq, 12: damping, 13: acoustic_impedance,
    ///  14: roughness, 15: water_depth, 16: cavity_hollowness]
    pub const MATERIAL_PROPERTIES: Range<usize> = 10..17;

    /// Alias for material properties.
    pub const SURFACES: Range<usize> = 10..17;

    /// 4-dimensional aeroacoustic wind dynamics:
    /// [17: speed, 18: gustiness, 19: turbulence, 20: howl]
    pub const WIND_DYNAMICS: Range<usize> = 17..21;

    /// 18-dimensional spatialized side acoustic elements:
    /// Insects (3), Birds (3), Fireplace (4), Thunder (4), Traffic (4).
    pub const SIDE_SOUNDS: Range<usize> = 21..39;

    /// 1-dimensional physical parameter drift tolerance (index 39).
    pub const DRIFT_INDEX: usize = 39;

    /// 6-dimensional physical acoustic summary metrics:
    /// [40: spectral_centroid, 41: spectral_spread, 42: spectral_skewness,
    ///  43: transient_density, 44: rms_energy, 45: diffuseness]
    pub const ACOUSTIC_METRICS: Range<usize> = 40..46;

    /// 18-dimensional compact semantic harmonic projection (open continuous space).
    pub const SEMANTIC_PROJECTION: Range<usize> = 46..64;

    /// Alias for semantic projection.
    pub const CLAP: Range<usize> = 46..64;

    /// Total dimension.
    pub const TOTAL_DIM: usize = CONDITION_DIM;
}

/// Physical and geometric range constraint for a conditioning vector dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionRange {
    /// Unipolar non-negative physical magnitude or energetic ratio [0.0, 1.0].
    UnipolarZeroToOne,
    /// Bipolar zero-centered spatial direction, elevation, thermal deviation, or semantic projection [-1.0, 1.0].
    BipolarMinusOneToOne,
}

impl DimensionRange {
    #[inline]
    pub const fn min_bound(self) -> f32 {
        match self {
            Self::UnipolarZeroToOne => 0.0,
            Self::BipolarMinusOneToOne => -1.0,
        }
    }

    #[inline]
    pub const fn max_bound(self) -> f32 {
        1.0
    }

    #[inline]
    pub fn clamp(self, val: f32) -> f32 {
        if val.is_nan() {
            0.0
        } else if val == f32::INFINITY {
            self.max_bound()
        } else if val == f32::NEG_INFINITY {
            self.min_bound()
        } else {
            val.clamp(self.min_bound(), self.max_bound())
        }
    }
}

/// Physical prior range specification for all 64 conditioning vector dimensions.
///
/// Follows Approach C (Physical Priors with Learned Latent Re-Projection):
/// - Unipolar [0.0, 1.0] for physical magnitudes, masses, and energies.
/// - Bipolar [-1.0, 1.0] for spatial panning, angles, thermal deviations, and semantic latents.
pub const DIMENSION_RANGES: [DimensionRange; CONDITION_DIM] = {
    use DimensionRange::*;
    [
        // 0..10: Macro Atmosphere & Controls
        UnipolarZeroToOne,    // 0: rain rate / intensity [0, 1]
        UnipolarZeroToOne,    // 1: runoff volume [0, 1]
        BipolarMinusOneToOne, // 2: temperature deviation (-1 freezing to +1 tropical) [-1, 1]
        UnipolarZeroToOne,    // 3: humidity ratio [0, 1]
        BipolarMinusOneToOne, // 4: pitch angle / inclination (-1 downward to +1 upward) [-1, 1]
        UnipolarZeroToOne,    // 5: listener distance [0, 1]
        UnipolarZeroToOne,    // 6: acoustic enclosure / boundary isolation [0, 1]
        UnipolarZeroToOne,    // 7: air absorption HF damping [0, 1]
        UnipolarZeroToOne,    // 8: wind speed [0, 1]
        BipolarMinusOneToOne, // 9: wind azimuth angle (-pi to +pi mapped to [-1, 1])
        // 10..17: Material Properties (Strict non-negative physical properties [0, 1])
        UnipolarZeroToOne, // 10: hardness
        UnipolarZeroToOne, // 11: resonance_freq
        UnipolarZeroToOne, // 12: damping
        UnipolarZeroToOne, // 13: acoustic_impedance
        UnipolarZeroToOne, // 14: roughness
        UnipolarZeroToOne, // 15: water_depth
        UnipolarZeroToOne, // 16: cavity_hollowness
        // 17..21: Aeroacoustic Wind Dynamics (Energy magnitudes [0, 1])
        UnipolarZeroToOne, // 17: speed
        UnipolarZeroToOne, // 18: gustiness
        UnipolarZeroToOne, // 19: turbulence
        UnipolarZeroToOne, // 20: howl
        // 21..39: Spatialized Side Sounds (Gains [0, 1], Panning/Elevations [-1, 1])
        UnipolarZeroToOne,    // 21: insect density
        UnipolarZeroToOne,    // 22: insect proximity
        BipolarMinusOneToOne, // 23: insect azimuth
        UnipolarZeroToOne,    // 24: bird activity
        UnipolarZeroToOne,    // 25: bird proximity
        BipolarMinusOneToOne, // 26: bird elevation
        UnipolarZeroToOne,    // 27: fireplace intensity
        UnipolarZeroToOne,    // 28: fireplace crackle
        BipolarMinusOneToOne, // 29: fireplace azimuth
        BipolarMinusOneToOne, // 30: fireplace elevation
        UnipolarZeroToOne,    // 31: thunder proximity
        UnipolarZeroToOne,    // 32: thunder rumble length
        BipolarMinusOneToOne, // 33: thunder azimuth
        BipolarMinusOneToOne, // 34: thunder elevation
        UnipolarZeroToOne,    // 35: traffic distance
        UnipolarZeroToOne,    // 36: traffic wetness
        BipolarMinusOneToOne, // 37: traffic azimuth start
        BipolarMinusOneToOne, // 38: traffic azimuth end
        // 39: Drift Index (Zero-centered cyclic deviation [-1, 1])
        BipolarMinusOneToOne, // 39: parameter drift state
        // 40..46: Physical Acoustic Summary Metrics
        UnipolarZeroToOne,    // 40: spectral centroid [0, 1]
        UnipolarZeroToOne,    // 41: spectral spread / flatness [0, 1]
        BipolarMinusOneToOne, // 42: spectral skewness / tilt (-1 dark rumble to +1 bright sizzle) [-1, 1]
        UnipolarZeroToOne,    // 43: transient / droplet density [0, 1]
        UnipolarZeroToOne,    // 44: RMS energy [0, 1]
        UnipolarZeroToOne,    // 45: diffuseness [0, 1]
        // 46..64: Semantic Harmonic Projection (18-dim unit hypersphere S^17 [-1, 1])
        BipolarMinusOneToOne, // 46
        BipolarMinusOneToOne, // 47
        BipolarMinusOneToOne, // 48
        BipolarMinusOneToOne, // 49
        BipolarMinusOneToOne, // 50
        BipolarMinusOneToOne, // 51
        BipolarMinusOneToOne, // 52
        BipolarMinusOneToOne, // 53
        BipolarMinusOneToOne, // 54
        BipolarMinusOneToOne, // 55
        BipolarMinusOneToOne, // 56
        BipolarMinusOneToOne, // 57
        BipolarMinusOneToOne, // 58
        BipolarMinusOneToOne, // 59
        BipolarMinusOneToOne, // 60
        BipolarMinusOneToOne, // 61
        BipolarMinusOneToOne, // 62
        BipolarMinusOneToOne, // 63
    ]
};

/// Clamps and sanitizes each dimension of a 64-dimensional conditioning vector
/// according to its physical prior specification.
#[inline]
pub fn sanitize_conditioning_vector(u: &mut [f32; CONDITION_DIM]) {
    for i in 0..CONDITION_DIM {
        u[i] = DIMENSION_RANGES[i].clamp(u[i]);
    }
}

/// Assembles a 64-dimensional conditioning vector without heap allocation
/// and enforces physical prior constraints across all dimensions.
#[inline]
pub fn encode_conditioning_vector(
    macro_atmosphere: &[f32; 10],
    material_properties: &[f32; 7],
    wind_dynamics: &[f32; 4],
    side_sounds: &[f32; 18],
    drift: f32,
    acoustic_metrics: &[f32; 6],
    semantic_projection: &[f32; 18],
) -> [f32; CONDITION_DIM] {
    let mut u = [0.0f32; CONDITION_DIM];
    u[ConditioningLayout::MACRO_ATMOSPHERE].copy_from_slice(macro_atmosphere);
    u[ConditioningLayout::MATERIAL_PROPERTIES].copy_from_slice(material_properties);
    u[ConditioningLayout::WIND_DYNAMICS].copy_from_slice(wind_dynamics);
    u[ConditioningLayout::SIDE_SOUNDS].copy_from_slice(side_sounds);
    u[ConditioningLayout::DRIFT_INDEX] = drift;
    u[ConditioningLayout::ACOUSTIC_METRICS].copy_from_slice(acoustic_metrics);
    u[ConditioningLayout::SEMANTIC_PROJECTION].copy_from_slice(semantic_projection);
    sanitize_conditioning_vector(&mut u);
    u
}

/// Generates a normalized 18-dimensional semantic embedding projection
/// from a descriptive text prompt (e.g. "Heavy thunderstorm on tin roof with gusts").
pub fn compute_text_semantic_projection(prompt: &str) -> [f32; 18] {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return [1.0 / (18.0f32).sqrt(); 18];
    }

    let mut emb = [0.0f32; 18];
    let words: Vec<&str> = trimmed.split_whitespace().collect();

    for (w_idx, word) in words.iter().enumerate() {
        let clean = word.to_lowercase();
        let mut hash: u64 = 0xcbf29ce484222325;
        for b in clean.bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }

        // Project word hash onto harmonic basis vectors
        for i in 0..18 {
            let phase = ((hash.wrapping_add((i * 31) as u64)) % 65536) as f32 / 65536.0;
            let weight = 1.0 / (1.0 + (w_idx as f32 * 0.15));
            emb[i] += (2.0 * std::f32::consts::PI * phase).sin() * weight;
        }
    }

    // L2-normalize to unit hypersphere
    let norm_sq: f32 = emb.iter().map(|v| v * v).sum();
    let inv_norm = if norm_sq > 1e-12 {
        1.0 / norm_sq.sqrt()
    } else {
        1.0 / (18.0f32).sqrt()
    };

    for v in emb.iter_mut() {
        *v *= inv_norm;
    }

    emb
}

/// Alias for text semantic embedding projection.
#[inline]
pub fn compute_text_embedding(prompt: &str) -> [f32; 18] {
    compute_text_semantic_projection(prompt)
}

/// Computes both the 6-dimensional physical acoustic metrics and 18-dimensional
/// compact semantic projection from raw audio samples.
pub fn compute_audio_acoustic_and_semantic(
    samples: &[f32],
    sample_rate: u32,
) -> ([f32; 6], [f32; 18]) {
    if samples.is_empty() {
        return ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1.0 / (18.0f32).sqrt(); 18]);
    }

    let sr = (sample_rate as f32).max(8000.0);
    let mut energy_sum = 0.0f32;
    let mut zcr_count = 0usize;
    let mut peak = 0.0f32;

    for i in 0..samples.len() {
        let s = samples[i].abs();
        if s > peak {
            peak = s;
        }
        energy_sum += s * s;
        if i > 0 && (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
            zcr_count += 1;
        }
    }

    let n = samples.len() as f32;
    let rms = (energy_sum / n).sqrt();
    let zcr = zcr_count as f32 / n;
    let centroid = (zcr * sr * 0.5).clamp(50.0, 16000.0);
    let norm_centroid = (centroid / 8000.0).clamp(0.0, 1.0);
    let crest_factor = if rms > 1e-5 { peak / rms } else { 1.0 };
    let transient_density = ((crest_factor - 1.0) / 6.0).clamp(0.0, 1.0);

    let metrics = [
        norm_centroid,
        0.5, // spread
        0.5, // skewness
        transient_density,
        (rms * 10.0).clamp(0.0, 1.0),
        0.5, // diffuseness
    ];

    let mut semantic = [0.0f32; 18];
    for i in 0..18 {
        let f = (i as f32 / 18.0) * (sr * 0.5);
        let dist = (f - centroid).abs() / (centroid * 0.5 + 100.0);
        let weight = (-dist).exp();
        let phase = i as f32 * 0.45 + rms * 8.0;
        semantic[i] = weight * phase.cos() + (1.0 - weight) * phase.sin() * 0.5;
    }

    // L2-normalize semantic
    let norm_sq: f32 = semantic.iter().map(|v| v * v).sum();
    let inv_norm = if norm_sq > 1e-12 {
        1.0 / norm_sq.sqrt()
    } else {
        1.0 / (18.0f32).sqrt()
    };
    for v in &mut semantic {
        *v *= inv_norm;
    }

    (metrics, semantic)
}

/// Generates a 24-dimensional compact acoustic latent embedding (6 metrics + 18 semantic)
/// from raw audio samples.
pub fn compute_audio_summary_embedding(samples: &[f32], sample_rate: u32) -> [f32; 24] {
    let (metrics, sem) = compute_audio_acoustic_and_semantic(samples, sample_rate);
    let mut out = [0.0f32; 24];
    out[0..6].copy_from_slice(&metrics);
    out[6..24].copy_from_slice(&sem);
    out
}

/// Inverts a natural language prompt description directly into physical rain parameters,
/// surfaces, wind, and side sounds in real time.
///
/// Modifies the `RainState` immediately, enabling instant audible feedback on both
/// physical DSP and neural synthesis without requiring manual confirmation.
pub fn invert_text_to_rain_state(prompt: &str, state: &mut crate::rain::RainState) -> bool {
    let lower = prompt.trim().to_lowercase();
    if lower.is_empty() {
        return false;
    }

    let mut matched = false;

    // 1. Precipitation Intensity & Runoff
    if lower.contains("drizzle")
        || lower.contains("mist")
        || lower.contains("sprinkle")
        || lower.contains("light rain")
        || lower.contains("gentle rain")
    {
        state.weather.intensity = 0.22;
        state.weather.runoff = 0.18;
        matched = true;
    } else if lower.contains("cloudburst")
        || lower.contains("torrential")
        || lower.contains("deluge")
        || lower.contains("downpour")
        || lower.contains("heavy")
        || lower.contains("storm")
    {
        state.weather.intensity = 0.88;
        state.weather.runoff = 0.82;
        matched = true;
    } else if lower.contains("steady")
        || lower.contains("moderate")
        || lower.contains("summer")
        || lower.contains("calm")
    {
        state.weather.intensity = 0.50;
        state.weather.runoff = 0.45;
        matched = true;
    }

    // 2. Physical Resonant Surfaces & Continuum Materials
    if lower.contains("umbrella") || lower.contains("parasol") {
        state.surfaces.canvas_tent = 0.65;
        state.surfaces.glass_window = 0.25;
        state.surfaces.puddle_shallow = 0.10;
        state.surfaces.tin = 0.0;
        state.surfaces.pavement = 0.0;
        state.surfaces.leaves_broad = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.wood_deck = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("car") || lower.contains("windshield") || lower.contains("sunroof") {
        state.surfaces.glass_window = 0.75;
        state.surfaces.tin = 0.20;
        state.surfaces.pavement = 0.05;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.leaves_broad = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.puddle_shallow = 0.0;
        state.surfaces.wood_deck = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("gravel") || lower.contains("pebbles") || lower.contains("stones") {
        state.surfaces.pavement = 0.70;
        state.surfaces.wood_deck = 0.20;
        state.surfaces.puddle_shallow = 0.10;
        state.surfaces.tin = 0.0;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.leaves_broad = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.glass_window = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("tin")
        || lower.contains("metal")
        || lower.contains("corrugated")
        || lower.contains("roof")
    {
        state.surfaces.tin = 0.85;
        state.surfaces.pavement = 0.05;
        state.surfaces.leaves_broad = 0.05;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.puddle_shallow = 0.05;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.glass_window = 0.0;
        state.surfaces.wood_deck = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("canvas")
        || lower.contains("tent")
        || lower.contains("tarp")
        || lower.contains("camping")
    {
        state.surfaces.canvas_tent = 0.80;
        state.surfaces.leaves_broad = 0.10;
        state.surfaces.puddle_shallow = 0.10;
        state.surfaces.tin = 0.0;
        state.surfaces.pavement = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.glass_window = 0.0;
        state.surfaces.wood_deck = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("pine") || lower.contains("needle") || lower.contains("conifer") {
        state.surfaces.pine_needles = 0.70;
        state.surfaces.leaves_broad = 0.20;
        state.surfaces.puddle_shallow = 0.10;
        state.surfaces.tin = 0.0;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.pavement = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.glass_window = 0.0;
        state.surfaces.wood_deck = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("foliage")
        || lower.contains("leaf")
        || lower.contains("leaves")
        || lower.contains("canopy")
        || lower.contains("forest")
        || lower.contains("jungle")
        || lower.contains("woods")
        || lower.contains("trees")
    {
        state.surfaces.leaves_broad = 0.65;
        state.surfaces.pine_needles = 0.25;
        state.surfaces.puddle_shallow = 0.10;
        state.surfaces.tin = 0.0;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.pavement = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.glass_window = 0.0;
        state.surfaces.wood_deck = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("window")
        || lower.contains("glass")
        || lower.contains("pane")
        || lower.contains("skylight")
    {
        state.surfaces.glass_window = 0.80;
        state.surfaces.tin = 0.10;
        state.surfaces.pavement = 0.10;
        state.surfaces.leaves_broad = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.puddle_shallow = 0.0;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.wood_deck = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("deck")
        || lower.contains("wood")
        || lower.contains("patio")
        || lower.contains("porch")
    {
        state.surfaces.wood_deck = 0.75;
        state.surfaces.pavement = 0.15;
        state.surfaces.puddle_shallow = 0.10;
        state.surfaces.tin = 0.0;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.leaves_broad = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.glass_window = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("pavement")
        || lower.contains("asphalt")
        || lower.contains("concrete")
        || lower.contains("street")
        || lower.contains("road")
        || lower.contains("urban")
        || lower.contains("city")
    {
        state.surfaces.pavement = 0.65;
        state.surfaces.puddle_shallow = 0.35;
        state.surfaces.tin = 0.0;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.leaves_broad = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.glass_window = 0.0;
        state.surfaces.wood_deck = 0.0;
        state.sound_layers.raindrops = true;
        matched = true;
    } else if lower.contains("puddle")
        || lower.contains("water")
        || lower.contains("lake")
        || lower.contains("pond")
        || lower.contains("river")
    {
        state.surfaces.water_deep = 0.60;
        state.surfaces.puddle_shallow = 0.40;
        state.surfaces.tin = 0.0;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.leaves_broad = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.pavement = 0.0;
        state.surfaces.glass_window = 0.0;
        state.surfaces.wood_deck = 0.0;
        matched = true;
    }

    // 3. Aeroacoustic Wind Dynamics
    if lower.contains("howl")
        || lower.contains("howling")
        || lower.contains("gale")
        || lower.contains("blizzard")
    {
        state.wind.speed = 0.85;
        state.wind.howl = 0.88;
        state.wind.gustiness = 0.78;
        state.wind.turbulence = 0.75;
        state.sound_layers.rain_wash = true;
        matched = true;
    } else if lower.contains("gust")
        || lower.contains("gusty")
        || lower.contains("whipping")
        || lower.contains("windy")
    {
        state.wind.speed = 0.65;
        state.wind.gustiness = 0.72;
        state.wind.turbulence = 0.60;
        state.wind.howl = 0.45;
        state.sound_layers.rain_wash = true;
        matched = true;
    } else if lower.contains("breeze") || lower.contains("gentle breeze") {
        state.wind.speed = 0.28;
        state.wind.gustiness = 0.22;
        state.wind.turbulence = 0.20;
        state.wind.howl = 0.05;
        matched = true;
    } else if lower.contains("still") || lower.contains("no wind") {
        state.wind.speed = 0.05;
        state.wind.gustiness = 0.05;
        state.wind.turbulence = 0.05;
        state.wind.howl = 0.0;
        matched = true;
    }

    // 4. Spatial Side Sounds
    if lower.contains("thunder") || lower.contains("lightning") {
        state.side_sounds.thunder_proximity = 0.72;
        state.side_sounds.thunder_rumble_length = 0.78;
        matched = true;
    }
    if lower.contains("bird")
        || lower.contains("birds")
        || lower.contains("chirp")
        || lower.contains("songbird")
    {
        state.side_sounds.bird_activity = 0.68;
        state.side_sounds.bird_proximity = 0.60;
        matched = true;
    }
    if lower.contains("insect")
        || lower.contains("insects")
        || lower.contains("cicada")
        || lower.contains("cicadas")
        || lower.contains("cricket")
        || lower.contains("crickets")
    {
        state.side_sounds.insect_density = 0.70;
        state.side_sounds.insect_proximity = 0.55;
        matched = true;
    }
    if lower.contains("fire")
        || lower.contains("fireplace")
        || lower.contains("campfire")
        || lower.contains("hearth")
        || lower.contains("cozy")
    {
        state.side_sounds.fireplace_intensity = 0.75;
        state.side_sounds.fireplace_crackle_rate = 0.65;
        matched = true;
    }
    if lower.contains("traffic")
        || lower.contains("car")
        || lower.contains("cars")
        || lower.contains("highway")
        || lower.contains("road")
    {
        state.side_sounds.traffic_distance = 0.35;
        state.side_sounds.traffic_wetness = 0.80;
        matched = true;
    }

    // 5. Ensure valid sound layers
    state.sound_layers.ensure_valid();

    matched
}

/// Inverts physical features from an empirical audio recording into RainState parameters.
pub fn invert_audio_to_rain_state(
    samples: &[f32],
    sample_rate: u32,
    state: &mut crate::rain::RainState,
) -> bool {
    if samples.is_empty() {
        return false;
    }

    let sr = (sample_rate as f32).max(8000.0);
    let mut energy_sum = 0.0f32;
    let mut zcr_count = 0usize;
    let mut peak = 0.0f32;

    for i in 0..samples.len() {
        let s = samples[i].abs();
        if s > peak {
            peak = s;
        }
        energy_sum += s * s;
        if i > 0 && (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
            zcr_count += 1;
        }
    }

    let n = samples.len() as f32;
    let rms = (energy_sum / n).sqrt();
    let zcr = zcr_count as f32 / n;
    let centroid = zcr * sr * 0.5;
    let crest_factor = if rms > 1e-5 { peak / rms } else { 1.0 };

    // Update weather intensity
    state.weather.intensity = (rms * 12.0).clamp(0.15, 1.0);
    state.weather.runoff = (state.weather.intensity * 0.85).clamp(0.1, 1.0);

    // Invert surfaces based on spectral centroid
    if centroid > 3800.0 {
        // High frequency: metal, tin, glass
        state.surfaces.tin = 0.70;
        state.surfaces.glass_window = 0.20;
        state.surfaces.pavement = 0.10;
        state.surfaces.leaves_broad = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.puddle_shallow = 0.0;
        state.surfaces.canvas_tent = 0.0;
        state.surfaces.wood_deck = 0.0;
    } else if centroid > 2200.0 {
        // Mid frequency: pavement, foliage, wood
        state.surfaces.pavement = 0.45;
        state.surfaces.leaves_broad = 0.35;
        state.surfaces.wood_deck = 0.20;
        state.surfaces.tin = 0.0;
        state.surfaces.glass_window = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.water_deep = 0.0;
        state.surfaces.puddle_shallow = 0.0;
        state.surfaces.canvas_tent = 0.0;
    } else {
        // Low/damped frequency: canvas tent, puddles, deep water
        state.surfaces.canvas_tent = 0.50;
        state.surfaces.puddle_shallow = 0.30;
        state.surfaces.water_deep = 0.20;
        state.surfaces.tin = 0.0;
        state.surfaces.glass_window = 0.0;
        state.surfaces.pavement = 0.0;
        state.surfaces.leaves_broad = 0.0;
        state.surfaces.pine_needles = 0.0;
        state.surfaces.wood_deck = 0.0;
    }

    // Invert sound layers and wind from dynamics
    if crest_factor > 3.2 {
        state.sound_layers.raindrops = true;
    }
    if rms > 0.04 || crest_factor < 2.8 {
        state.sound_layers.rain_wash = true;
    }
    state.sound_layers.ensure_valid();

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_text_semantic_projection_unit_norm() {
        let empty = compute_text_semantic_projection("");
        let norm_empty: f32 = empty.iter().map(|v| v * v).sum();
        assert!((norm_empty - 1.0).abs() < 1e-4);

        let prompt = "Heavy summer downpour on corrugated tin roof with howling wind";
        let emb = compute_text_semantic_projection(prompt);
        let norm: f32 = emb.iter().map(|v| v * v).sum();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_text_semantic_projection_distinct_prompts() {
        let p1 =
            compute_text_semantic_projection("Gentle meadow drizzle with birds and soft foliage");
        let p2 =
            compute_text_semantic_projection("Roaring category 5 tropical storm on sheet metal");

        let dot: f32 = p1.iter().zip(p2.iter()).map(|(a, b)| a * b).sum();
        assert!(dot < 0.90);
    }

    #[test]
    fn test_compute_audio_summary_embedding() {
        let empty = compute_audio_summary_embedding(&[], 48000);
        assert_eq!(empty.len(), 24);

        // Generate synthetic sine pulse
        let mut samples = Vec::with_capacity(1000);
        for i in 0..1000 {
            let t = i as f32 / 48000.0;
            samples.push((2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        let emb = compute_audio_summary_embedding(&samples, 48000);
        assert_eq!(emb.len(), 24);
    }

    #[test]
    fn test_encode_conditioning_vector_layout_64() {
        let macro_atm = [0.5f32; 10];
        let mat_props = [0.8f32; 7];
        let wind = [0.25f32; 4];
        let side = [0.05f32; 18];
        let drift = 0.75f32;
        let metrics = [0.6f32; 6];
        let semantic = [0.25f32; 18];

        let vec = encode_conditioning_vector(
            &macro_atm, &mat_props, &wind, &side, drift, &metrics, &semantic,
        );
        assert_eq!(vec.len(), CONDITION_DIM);
        assert_eq!(CONDITION_DIM, 64);
        assert_eq!(vec[ConditioningLayout::DRIFT_INDEX], 0.75);
        assert_eq!(vec[ConditioningLayout::MACRO_ATMOSPHERE][0], 0.5);
        assert_eq!(vec[ConditioningLayout::MATERIAL_PROPERTIES][0], 0.8);
        assert_eq!(vec[ConditioningLayout::WIND_DYNAMICS][0], 0.25);
        assert_eq!(vec[ConditioningLayout::SIDE_SOUNDS][0], 0.05);
        assert_eq!(vec[ConditioningLayout::ACOUSTIC_METRICS][0], 0.6);
        assert_eq!(vec[ConditioningLayout::SEMANTIC_PROJECTION][0], 0.25);
    }

    #[test]
    fn test_invert_text_to_rain_state() {
        let mut state = crate::rain::RainState::default();
        let prompt = "Heavy summer downpour on golf umbrella with howling wind and thunder";
        let res = invert_text_to_rain_state(prompt, &mut state);
        assert!(res);
        assert!(state.weather.intensity > 0.8);
        assert!(state.surfaces.canvas_tent > 0.5); // taut skin umbrella
        assert!(state.wind.howl > 0.8);
        assert!(state.side_sounds.thunder_proximity > 0.7);
        assert!(state.sound_layers.raindrops);
    }

    #[test]
    fn test_invert_audio_to_rain_state() {
        let mut state = crate::rain::RainState::default();
        let mut samples = Vec::with_capacity(4800);
        for i in 0..4800 {
            let t = i as f32 / 48000.0;
            // 5 kHz high-pitched pulse (simulating metal/tin)
            samples.push((2.0 * std::f32::consts::PI * 5000.0 * t).sin() * 0.5);
        }
        let res = invert_audio_to_rain_state(&samples, 48000, &mut state);
        assert!(res);
        assert!(state.surfaces.tin > 0.6);
    }

    #[test]
    fn test_dimension_ranges_and_sanitization() {
        assert_eq!(DIMENSION_RANGES.len(), 64);

        // Rain intensity is unipolar
        assert_eq!(DIMENSION_RANGES[0], DimensionRange::UnipolarZeroToOne);
        assert_eq!(DIMENSION_RANGES[0].clamp(-0.5), 0.0);
        assert_eq!(DIMENSION_RANGES[0].clamp(1.5), 1.0);

        // Temperature deviation is bipolar
        assert_eq!(DIMENSION_RANGES[2], DimensionRange::BipolarMinusOneToOne);
        assert_eq!(DIMENSION_RANGES[2].clamp(-0.8), -0.8);
        assert_eq!(DIMENSION_RANGES[2].clamp(-2.0), -1.0);

        // Semantic dimension 46 is bipolar
        assert_eq!(DIMENSION_RANGES[46], DimensionRange::BipolarMinusOneToOne);
        assert_eq!(DIMENSION_RANGES[46].clamp(-0.35), -0.35);

        // Material properties are strictly unipolar
        assert_eq!(DIMENSION_RANGES[10], DimensionRange::UnipolarZeroToOne);
        assert_eq!(DIMENSION_RANGES[10].clamp(-1.0), 0.0);

        // Sanitizer handles NaNs gracefully (converts to zero)
        let mut raw_nan = [f32::NAN; 64];
        sanitize_conditioning_vector(&mut raw_nan);
        for v in raw_nan {
            assert_eq!(v, 0.0);
        }

        // Sanitizer handles positive infinities (clamps to upper extreme 1.0)
        let mut raw_pos_inf = [f32::INFINITY; 64];
        sanitize_conditioning_vector(&mut raw_pos_inf);
        for v in raw_pos_inf {
            assert_eq!(v, 1.0);
        }

        // Sanitizer handles negative infinities (clamps to lower extreme: 0.0 unipolar, -1.0 bipolar)
        let mut raw_neg_inf = [f32::NEG_INFINITY; 64];
        sanitize_conditioning_vector(&mut raw_neg_inf);
        assert_eq!(raw_neg_inf[0], 0.0); // intensity (unipolar)
        assert_eq!(raw_neg_inf[2], -1.0); // temperature (bipolar)
        assert_eq!(raw_neg_inf[10], 0.0); // hardness (unipolar)
        assert_eq!(raw_neg_inf[46], -1.0); // semantic projection (bipolar)
    }
}
