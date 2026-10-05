//! RainAI domain models, Macro-to-Micro conditioning mapping, and audio synthesis state.

use serde::{Deserialize, Serialize};

pub use crate::conditioning::CONDITION_DIM;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QualityTier {
    Ternary158,
    #[default]
    AdaptiveMinimum,
    HighInt16,
    StudioFp32,
}

impl QualityTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ternary158 => "Ternary 1.58-Bit (Ultra-Fast / Add-Only)",
            Self::AdaptiveMinimum => "Adaptive QAT (Fast Startup)",
            Self::HighInt16 => "High Quality (INT16 On-Demand)",
            Self::StudioFp32 => "Studio Master (FP32 On-Demand)",
        }
    }

    pub fn download_size_label(self) -> &'static str {
        match self {
            Self::Ternary158 => "~1.1 MB (Included / Instant)",
            Self::AdaptiveMinimum => "~2.4 MB (Included)",
            Self::HighInt16 => "~9.8 MB (Download)",
            Self::StudioFp32 => "~38.4 MB (Download)",
        }
    }

    pub fn is_download_required(self) -> bool {
        match self {
            Self::Ternary158 | Self::AdaptiveMinimum => false,
            Self::HighInt16 | Self::StudioFp32 => true,
        }
    }
}

/// Continuous mixture over 9 physical materials
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SurfaceMixture {
    pub tin: f32,
    pub leaves_broad: f32,
    pub pine_needles: f32,
    pub pavement: f32,
    pub water_deep: f32,
    pub puddle_shallow: f32,
    pub canvas_tent: f32,
    pub glass_window: f32,
    pub wood_deck: f32,
}

impl Default for SurfaceMixture {
    fn default() -> Self {
        Self {
            tin: 0.1,
            leaves_broad: 0.2,
            pine_needles: 0.1,
            pavement: 0.15,
            water_deep: 0.05,
            puddle_shallow: 0.15,
            canvas_tent: 0.05,
            glass_window: 0.1,
            wood_deck: 0.1,
        }
    }
}

impl SurfaceMixture {
    /// Returns the proportion for a given canonical surface.
    pub fn get(&self, surface: crate::surface::CanonicalSurface) -> f32 {
        use crate::surface::CanonicalSurface::*;
        match surface {
            TinRoof => self.tin,
            Foliage => (self.leaves_broad + self.pine_needles) * 0.5,
            Pavement | Asphalt => self.pavement,
            WaterDeep => self.water_deep,
            PuddleShallow => self.puddle_shallow,
            CanvasTent => self.canvas_tent,
            Glass => self.glass_window,
            WoodDeck => self.wood_deck,
        }
    }

    /// Sets the proportion for a given canonical surface.
    pub fn set(&mut self, surface: crate::surface::CanonicalSurface, val: f32) {
        use crate::surface::CanonicalSurface::*;
        match surface {
            TinRoof => self.tin = val,
            Foliage => {
                self.leaves_broad = val * 0.6;
                self.pine_needles = val * 0.4;
            }
            Pavement | Asphalt => self.pavement = val,
            WaterDeep => self.water_deep = val,
            PuddleShallow => self.puddle_shallow = val,
            CanvasTent => self.canvas_tent = val,
            Glass => self.glass_window = val,
            WoodDeck => self.wood_deck = val,
        }
    }

    /// Normalizes the surface values using a partition of unity (sum to 1.0)
    pub fn normalized(&self) -> [f32; 9] {
        let raw = [
            self.tin.max(0.0),
            self.leaves_broad.max(0.0),
            self.pine_needles.max(0.0),
            self.pavement.max(0.0),
            self.water_deep.max(0.0),
            self.puddle_shallow.max(0.0),
            self.canvas_tent.max(0.0),
            self.glass_window.max(0.0),
            self.wood_deck.max(0.0),
        ];
        let sum: f32 = raw.iter().sum();
        if sum <= 1e-6 {
            [1.0 / 9.0; 9]
        } else {
            let mut out = [0.0; 9];
            for i in 0..9 {
                out[i] = raw[i] / sum;
            }
            out
        }
    }

    /// Converts the current surface mixture into continuous physical material properties
    pub fn to_material_properties(&self) -> crate::surface::PhysicalMaterialProperties {
        let norm = self.normalized();
        let surfaces = crate::surface::CanonicalSurface::ALL;
        let mut props = crate::surface::PhysicalMaterialProperties {
            hardness: 0.0,
            resonance_freq: 0.0,
            damping: 0.0,
            acoustic_impedance: 0.0,
            roughness: 0.0,
            water_depth: 0.0,
            cavity_hollowness: 0.0,
        };
        for (i, &surf) in surfaces.iter().enumerate() {
            let weight = norm[i];
            let p = surf.material_properties();
            props.hardness += p.hardness * weight;
            props.resonance_freq += p.resonance_freq * weight;
            props.damping += p.damping * weight;
            props.acoustic_impedance += p.acoustic_impedance * weight;
            props.roughness += p.roughness * weight;
            props.water_depth += p.water_depth * weight;
            props.cavity_hollowness += p.cavity_hollowness * weight;
        }
        props
    }

    pub fn set_preset_forest(&mut self) {
        self.tin = 0.0;
        self.leaves_broad = 0.45;
        self.pine_needles = 0.35;
        self.pavement = 0.0;
        self.water_deep = 0.05;
        self.puddle_shallow = 0.1;
        self.canvas_tent = 0.0;
        self.glass_window = 0.0;
        self.wood_deck = 0.05;
    }

    pub fn set_preset_urban(&mut self) {
        self.tin = 0.25;
        self.leaves_broad = 0.05;
        self.pine_needles = 0.0;
        self.pavement = 0.40;
        self.water_deep = 0.0;
        self.puddle_shallow = 0.15;
        self.canvas_tent = 0.0;
        self.glass_window = 0.15;
        self.wood_deck = 0.0;
    }

    pub fn set_preset_tent(&mut self) {
        self.tin = 0.0;
        self.leaves_broad = 0.15;
        self.pine_needles = 0.15;
        self.pavement = 0.0;
        self.water_deep = 0.0;
        self.puddle_shallow = 0.1;
        self.canvas_tent = 0.60;
        self.glass_window = 0.0;
        self.wood_deck = 0.0;
    }

    pub fn set_preset_window(&mut self) {
        self.tin = 0.1;
        self.leaves_broad = 0.05;
        self.pine_needles = 0.0;
        self.pavement = 0.1;
        self.water_deep = 0.0;
        self.puddle_shallow = 0.05;
        self.canvas_tent = 0.0;
        self.glass_window = 0.70;
        self.wood_deck = 0.0;
    }
}

/// Wind physics parameters
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct WindParameters {
    pub speed: f32,
    pub gustiness: f32,
    pub turbulence: f32,
    pub howl: f32,
}

impl Default for WindParameters {
    fn default() -> Self {
        Self {
            speed: 0.35,
            gustiness: 0.2,
            turbulence: 0.15,
            howl: 0.1,
        }
    }
}

/// Spatialized side sounds positioned in the First-Order Ambisonic field
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SideSounds {
    // Insects
    pub insect_density: f32,
    pub insect_proximity: f32,
    pub insect_azimuth: f32, // -1.0 to 1.0 mapped to -pi to pi

    // Birds
    pub bird_activity: f32,
    pub bird_proximity: f32,
    pub bird_elevation: f32, // 0.0 to 1.0 mapped to 0 to pi/2

    // Fireplace
    pub fireplace_intensity: f32,
    pub fireplace_crackle_rate: f32,
    pub fireplace_azimuth: f32,
    pub fireplace_elevation: f32,

    // Thunder
    pub thunder_proximity: f32,
    pub thunder_rumble_length: f32,
    pub thunder_azimuth: f32,
    pub thunder_elevation: f32,

    // Traffic
    pub traffic_distance: f32,
    pub traffic_wetness: f32,
    pub traffic_azimuth_start: f32,
    pub traffic_azimuth_end: f32,
}

impl Default for SideSounds {
    fn default() -> Self {
        Self {
            insect_density: 0.1,
            insect_proximity: 0.8,
            insect_azimuth: 0.25,

            bird_activity: 0.05,
            bird_proximity: 0.9,
            bird_elevation: 0.4,

            fireplace_intensity: 0.0,
            fireplace_crackle_rate: 0.4,
            fireplace_azimuth: -0.3,
            fireplace_elevation: 0.0,

            thunder_proximity: 0.0,
            thunder_rumble_length: 0.6,
            thunder_azimuth: 0.7,
            thunder_elevation: 0.6,

            traffic_distance: 0.0,
            traffic_wetness: 0.8,
            traffic_azimuth_start: -0.8,
            traffic_azimuth_end: 0.8,
        }
    }
}

/// Base weather and acoustic space parameters
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BaseWeather {
    pub intensity: f32,
    pub runoff: f32,
    pub temperature: f32,
    pub humidity: f32,
    pub pitch_angle: f32,
    pub distance: f32,
    pub enclosure: f32, // 0.0: Outside in the open, 1.0: Deep indoors behind glass
}

impl Default for BaseWeather {
    fn default() -> Self {
        Self {
            intensity: 0.5,
            runoff: 0.4,
            temperature: 0.6,
            humidity: 0.85,
            pitch_angle: 0.1,
            distance: 0.3,
            enclosure: 0.2,
        }
    }
}

/// Real-time engine telemetry reported back to egui
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EngineTelemetry {
    pub buffer_health_ms: f32,
    pub cpu_headroom: f32,
    pub gpu_headroom: f32,
    pub delta_t_ms: f32,
    #[serde(default)]
    pub active_experts: usize,
    pub panic_factor: f32,
    pub jitter_factor: f32,
    pub quality_critic_score: f32,
    pub synthesis_blend: f32,
    pub effective_quant_floor: f32,
    pub effective_quant_ceiling: f32,
    #[serde(default)]
    pub governor_status: String,
    pub active_path_label: String,
    pub active_quantization_format: String,
    pub is_prebuffered: bool,
    pub active_optimization_profile: String,
    pub active_stress_profile: String,
    pub diffusion_bypassed: bool,
    pub ambisonic_order_reduced: bool,
    pub thinking_steps: usize,
    pub consistency_jump_active: bool,
    pub dynamic_buffer_bytes: usize,
    pub buffer_health_ratio: f32,
    pub user_thinking_steps_override: bool,
    pub is_offline_export_mode: bool,
    pub history_seconds_available: f32,
    pub env_max_buffer_bytes: usize,
    pub buffer_capacity_ms: f32,
    pub buffer_resize_cooldown: f32,
    pub quant_macro_cooldown: f32,
    pub buffer_resizes_count: usize,
    pub quant_swaps_count: usize,
}

impl Default for EngineTelemetry {
    fn default() -> Self {
        Self {
            buffer_health_ms: 45.0,
            cpu_headroom: 0.88,
            gpu_headroom: 0.92,
            delta_t_ms: 10.0,
            active_experts: 8,
            panic_factor: 0.0,
            jitter_factor: 0.02,
            quality_critic_score: 0.96,
            synthesis_blend: 0.0,
            effective_quant_floor: 1.58,
            effective_quant_ceiling: 8.0,
            governor_status: "Optimal Headroom".into(),
            active_path_label: "CPU Neural SIMD".into(),
            active_quantization_format: "INT8 (8-Bit, 256 Levels)".into(),
            is_prebuffered: false,
            active_optimization_profile: "Balanced Adaptive (Default)".into(),
            active_stress_profile: "0: Nominal Desktop (Pristine)".into(),
            diffusion_bypassed: false,
            ambisonic_order_reduced: false,
            thinking_steps: 3,
            consistency_jump_active: false,
            dynamic_buffer_bytes: 34560,
            buffer_health_ratio: 1.0,
            user_thinking_steps_override: false,
            is_offline_export_mode: false,
            history_seconds_available: 0.0,
            env_max_buffer_bytes: 65536,
            buffer_capacity_ms: 45.0,
            buffer_resize_cooldown: 2.0,
            quant_macro_cooldown: 15.0,
            buffer_resizes_count: 0,
            quant_swaps_count: 0,
        }
    }
}

/// Audio DSP smoothing and parameter interpolation mode
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DspSmoothingMode {
    #[default]
    MediatedLive, // Live play: Smooths physical parameters & prevents clicks
    DirectBypass,      // Direct raw parameter application without interpolation
    OfflineMaxQuality, // Non-realtime export: Maximum sample fidelity, unconstrained headroom
}

pub type MetaControllerInterceptionMode = DspSmoothingMode;

impl DspSmoothingMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::MediatedLive => "Physical Acoustic Smoothing (Live Physics)",
            Self::DirectBypass => "Direct Parameter Control (Manual Raw Bypass)",
            Self::OfflineMaxQuality => "Offline Studio Master Quality (Max Headroom)",
        }
    }
}

/// Operational performance and latency profiles
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PerformanceProfile {
    EcoBatterySaver,
    LowLatencyInteractive,
    #[default]
    BalancedAdaptive,
    StudioMaster,
    BluetoothA2DPSink,
}

pub type GovernorOptimizationProfile = PerformanceProfile;

impl PerformanceProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::EcoBatterySaver => "Eco Battery Saver (<0.5W, High Efficiency)",
            Self::LowLatencyInteractive => "Low-Latency Interactive (15ms Buffer, Snappy)",
            Self::BalancedAdaptive => "Balanced Adaptive (45ms Buffer, Optimal)",
            Self::StudioMaster => "Studio Master (120ms Buffer, High Fidelity)",
            Self::BluetoothA2DPSink => "Bluetooth A2DP Sink (150ms Safety Reserve)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::EcoBatterySaver => "Eco Battery",
            Self::LowLatencyInteractive => "Low Latency",
            Self::BalancedAdaptive => "Balanced",
            Self::StudioMaster => "Studio Master",
            Self::BluetoothA2DPSink => "Bluetooth Sink",
        }
    }

    pub fn target_buffer_ms(self) -> f32 {
        match self {
            Self::EcoBatterySaver => 30.0,
            Self::LowLatencyInteractive => 15.0,
            Self::BalancedAdaptive => 45.0,
            Self::StudioMaster => 120.0,
            Self::BluetoothA2DPSink => 150.0,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::EcoBatterySaver => {
                "Power-saving profile prioritizing thermal efficiency and minimal battery drain under 0.5W."
            }
            Self::LowLatencyInteractive => {
                "Ultra-fast response with a tight 15ms buffer for responsive real-time slider interaction."
            }
            Self::BalancedAdaptive => {
                "Default operational profile balancing robust buffer headroom with low audio latency (45ms target)."
            }
            Self::StudioMaster => {
                "Maximum fidelity priority locking 120ms buffer reserve for pristine playback."
            }
            Self::BluetoothA2DPSink => {
                "Extended 150ms safety reserve with jitter damping to prevent underruns on wireless Bluetooth audio sinks."
            }
        }
    }
}

/// Simulated hardware stress profiles matching RainAI deployment specification
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HardwareStressProfile {
    #[default]
    NominalDesktop,
    ThermalThrottlingCascade,
    GcWebAudioMicroStalls,
    UnifiedMemoryBusContention,
    DynamicGameDawInterference,
    BluetoothA2dpAudioSink,
    EcoSleepSoundscapeMode,
    HeterogeneousEcoreAsymmetry,
}

impl HardwareStressProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::NominalDesktop => "0: Nominal Desktop (Pristine)",
            Self::ThermalThrottlingCascade => "1: Thermal Throttling Cascade (Clock Collapse)",
            Self::GcWebAudioMicroStalls => "2: GC / WebAudio Micro-Stalls (Periodic Pauses)",
            Self::UnifiedMemoryBusContention => "3: Unified Memory Bus Contention (Transfer Choke)",
            Self::DynamicGameDawInterference => "4: Dynamic Game/DAW Interference (Host Bursts)",
            Self::BluetoothA2dpAudioSink => "5: Bluetooth A2DP Audio Sink (Jitter & Latency)",
            Self::EcoSleepSoundscapeMode => "6: Eco Sleep Soundscape Mode (Low-Power Throttle)",
            Self::HeterogeneousEcoreAsymmetry => {
                "7: Heterogeneous E-Core Asymmetry (Core Bouncing)"
            }
        }
    }

    pub fn profile_id(self) -> u8 {
        match self {
            Self::NominalDesktop => 0,
            Self::ThermalThrottlingCascade => 1,
            Self::GcWebAudioMicroStalls => 2,
            Self::UnifiedMemoryBusContention => 3,
            Self::DynamicGameDawInterference => 4,
            Self::BluetoothA2dpAudioSink => 5,
            Self::EcoSleepSoundscapeMode => 6,
            Self::HeterogeneousEcoreAsymmetry => 7,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::NominalDesktop => {
                "Unconstrained execution with maximum CPU/GPU headroom and minimal jitter."
            }
            Self::ThermalThrottlingCascade => {
                "Simulates thermal clock collapse with progressive compute latency and rising panic factor."
            }
            Self::GcWebAudioMicroStalls => {
                "Injects periodic 5-25ms garbage collector pauses into the buffer pipeline."
            }
            Self::UnifiedMemoryBusContention => {
                "Simulates memory bandwidth saturation, inducing jitter in weight/latent tensor transfers."
            }
            Self::DynamicGameDawInterference => {
                "Simulates competing heavy background workloads with sudden high-priority thread spikes."
            }
            Self::BluetoothA2dpAudioSink => {
                "Simulates wireless audio output with high transmission latency and variable packet dispatch."
            }
            Self::EcoSleepSoundscapeMode => {
                "Aggressively restricts compute to ultra-low frequency and throttles background tasks."
            }
            Self::HeterogeneousEcoreAsymmetry => {
                "Simulates thread migration bouncing between high-frequency P-cores and low-power E-cores."
            }
        }
    }
}

/// Colors of noise for subtractive synthesis shaping
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoiseColor {
    White,
    #[default]
    Pink,
    Brown,
    Blue,
    Violet,
}

impl NoiseColor {
    pub fn label(self) -> &'static str {
        match self {
            Self::White => "White (0 dB/oct - Flat & Crisp Spray)",
            Self::Pink => "Pink (-3 dB/oct - Natural Rain)",
            Self::Brown => "Brown (-6 dB/oct - Deep, Warm & Heavy Rain)",
            Self::Blue => "Blue (+3 dB/oct - High Mist Spray)",
            Self::Violet => "Violet (+6 dB/oct - Sharp Needles on Tin)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::White => "White Noise",
            Self::Pink => "Pink Noise",
            Self::Brown => "Brown Noise",
            Self::Blue => "Blue Noise",
            Self::Violet => "Violet Noise",
        }
    }

    pub fn spectral_decay_power(self) -> f32 {
        match self {
            Self::White => 0.0,
            Self::Pink => 0.5,
            Self::Brown => 1.0,
            Self::Blue => -0.5,
            Self::Violet => -1.0,
        }
    }
}

/// Core audio synthesis engine generator mode
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SynthesisMode {
    #[default]
    NeuralAi,
    PhysicalSynth,
    ProceduralFilterbank,
    HybridAdaptive,
}

impl SynthesisMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::NeuralAi => "Neural AI (Mamba2-MoE + VAE) (Preview)",
            Self::PhysicalSynth => "Physical Fluid Dynamics (Synth-Rain)",
            Self::ProceduralFilterbank => "Subtractive Procedural (16-Band)",
            Self::HybridAdaptive => "Hybrid Adaptive (Governor Dynamic Blend)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::NeuralAi => "Neural AI (Preview)",
            Self::PhysicalSynth => "Synth-Rain",
            Self::ProceduralFilterbank => "Procedural",
            Self::HybridAdaptive => "Hybrid",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::NeuralAi => {
                "Recurrent Mamba-2 SSM + Spatial VAE DDSP generative soundscape synthesis."
            }
            Self::PhysicalSynth => {
                "Physical acoustic simulation of Gunn-Kinzer droplet velocities, Ulbrich DSD, and impact cavitation."
            }
            Self::ProceduralFilterbank => {
                "16-band resonant subtractive filterbank with colored noise shaping (zero latency)."
            }
            Self::HybridAdaptive => {
                "Autonomous governor dynamically blending Neural and Procedural/Physical engines based on compute headroom."
            }
        }
    }
}

/// User-selectable sound layers (replaces opaque synthesis mode dropdown)
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoundLayers {
    /// Discrete particle impacts against physical surface resonators (tin, foliage, glass, etc.)
    pub raindrops: bool,
    /// Continuous ambient pink/brown noise wash, wind gusts, and distant rainfall
    pub rain_wash: bool,
    /// Generative neural DDSP texture (incoming / training)
    pub ai_texture: bool,
}

impl Default for SoundLayers {
    fn default() -> Self {
        Self {
            raindrops: true,
            rain_wash: true,
            ai_texture: false, // OFF by default (models training)
        }
    }
}

impl SoundLayers {
    /// Enforces the guardrail that at least one rain layer must remain active.
    pub fn ensure_valid(&mut self) {
        if !self.raindrops && !self.rain_wash && !self.ai_texture {
            self.rain_wash = true;
        }
    }
}

/// Complete RainAI engine state
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RainState {
    pub is_playing: bool,
    pub master_volume: f32,
    pub synthesis_mode: SynthesisMode,
    #[serde(default)]
    pub sound_layers: SoundLayers,
    pub quality_tier: QualityTier,
    pub noise_color: NoiseColor,
    pub evolve_enabled: bool,
    pub evolve_speed: f32,
    pub auto_quantize: bool,
    pub drift_time: f32,
    pub weather: BaseWeather,
    pub surfaces: SurfaceMixture,
    pub wind: WindParameters,
    pub side_sounds: SideSounds,
    pub preferred_format: Option<String>,
    pub optimization_profile: GovernorOptimizationProfile,
    pub stress_profile: HardwareStressProfile,
    #[serde(default = "default_thinking_steps")]
    pub thinking_steps: usize,
    #[serde(default)]
    pub use_consistency_jump: bool,
    #[serde(default)]
    pub user_thinking_steps: Option<usize>,
    #[serde(default)]
    pub conditioning_prompt: String,
    #[serde(default)]
    pub conditioning_audio_file: Option<String>,
    #[serde(skip)]
    pub custom_latent_embedding: Option<[f32; 24]>,
    #[serde(skip)]
    pub custom_clap_embedding: Option<[f32; 18]>,
    #[serde(default)]
    pub meta_mediation_mode: DspSmoothingMode,
    #[serde(default = "default_true")]
    pub history_recording_enabled: bool,
    #[serde(skip)]
    pub telemetry: EngineTelemetry,
}

fn default_thinking_steps() -> usize {
    3
}

fn default_true() -> bool {
    true
}

impl Default for RainState {
    fn default() -> Self {
        Self {
            is_playing: false,
            master_volume: 0.8,
            synthesis_mode: SynthesisMode::PhysicalSynth,
            sound_layers: SoundLayers::default(),
            quality_tier: QualityTier::AdaptiveMinimum,
            noise_color: NoiseColor::Pink,
            evolve_enabled: false, // Off by default
            evolve_speed: 0.2,
            auto_quantize: true,
            drift_time: 0.0,
            weather: BaseWeather::default(),
            surfaces: SurfaceMixture::default(),
            wind: WindParameters::default(),
            side_sounds: SideSounds::default(),
            preferred_format: None,
            optimization_profile: GovernorOptimizationProfile::default(),
            stress_profile: HardwareStressProfile::default(),
            thinking_steps: 3,
            use_consistency_jump: false,
            user_thinking_steps: None,
            conditioning_prompt: String::new(),
            conditioning_audio_file: None,
            custom_latent_embedding: None,
            custom_clap_embedding: None,
            meta_mediation_mode: MetaControllerInterceptionMode::default(),
            history_recording_enabled: true,
            telemetry: EngineTelemetry::default(),
        }
    }
}

impl RainState {
    /// Steps the procedural Brownian weather drift to keep the soundscape alive
    pub fn step_procedural_drift(&mut self, dt: f32) {
        if !self.evolve_enabled {
            return;
        }

        self.drift_time += dt * self.evolve_speed;
        let t = self.drift_time;

        // Subtle harmonic wind gusting
        let wind_drift = (t * 0.4).sin() * 0.15 + (t * 1.1).cos() * 0.05;
        self.wind.gustiness = (self.wind.gustiness + wind_drift * dt).clamp(0.0, 1.0);

        // Slow atmospheric humidity & runoff drift
        let runoff_drift = (t * 0.15).sin() * 0.08;
        self.weather.runoff = (self.weather.runoff + runoff_drift * dt).clamp(0.1, 1.0);

        // Gentle insect/bird diurnal fluctuation
        let bio_drift = (t * 0.25).cos() * 0.05;
        if self.side_sounds.bird_activity > 0.02 {
            self.side_sounds.bird_activity =
                (self.side_sounds.bird_activity + bio_drift * dt).clamp(0.0, 0.8);
        }
    }

    /// Converts current UI parameters into the 64-dim conditioning vector as a fixed array without heap allocation
    pub fn to_conditioning_array(&self) -> [f32; CONDITION_DIM] {
        let macro_atmosphere = [
            self.weather.intensity,
            self.weather.runoff,
            (self.weather.temperature - 0.5) * 2.0, // thermal deviation [-1.0 freezing to +1.0 tropical]
            self.weather.humidity,
            self.weather.pitch_angle, // inclination [-1.0 downward to +1.0 upward]
            self.weather.distance,
            self.weather.enclosure,
            (1.0 - self.weather.humidity) * 0.5, // air_absorption
            self.wind.speed,
            0.0, // wind_azimuth [-1.0 to +1.0]
        ];
        let material_props = self.surfaces.to_material_properties().to_array();
        let wind_dynamics = [
            self.wind.speed,
            self.wind.gustiness,
            self.wind.turbulence,
            self.wind.howl,
        ];
        let side_sounds = [
            self.side_sounds.insect_density,
            self.side_sounds.insect_proximity,
            self.side_sounds.insect_azimuth,
            self.side_sounds.bird_activity,
            self.side_sounds.bird_proximity,
            self.side_sounds.bird_elevation,
            self.side_sounds.fireplace_intensity,
            self.side_sounds.fireplace_crackle_rate,
            self.side_sounds.fireplace_azimuth,
            self.side_sounds.fireplace_elevation,
            self.side_sounds.thunder_proximity,
            self.side_sounds.thunder_rumble_length,
            self.side_sounds.thunder_azimuth,
            self.side_sounds.thunder_elevation,
            self.side_sounds.traffic_distance,
            self.side_sounds.traffic_wetness,
            self.side_sounds.traffic_azimuth_start,
            self.side_sounds.traffic_azimuth_end,
        ];
        let drift = 0.2;

        let (acoustic_metrics, semantic_projection) = if let Some(custom) =
            self.custom_latent_embedding
        {
            let mut m = [0.0f32; 6];
            let mut s = [0.0f32; 18];
            m.copy_from_slice(&custom[0..6]);
            s.copy_from_slice(&custom[6..24]);
            (m, s)
        } else {
            let spectral_centroid =
                (1500.0 + material_props[0] * 3500.0 + self.weather.intensity * 1000.0) / 8000.0;
            let spectral_spread = (0.3 + self.wind.turbulence * 0.4).clamp(0.0, 1.0);
            let spectral_skewness = (material_props[0] - material_props[2]).clamp(-1.0, 1.0); // [-1.0 dark thud to +1.0 bright ring]
            let transient_density =
                (self.weather.intensity * (material_props[0] + 0.3)).clamp(0.0, 1.0);
            let rms_energy = (self.weather.intensity * 0.8 + 0.1).clamp(0.0, 1.0);
            let diffuseness =
                (self.weather.distance * 0.7 + self.weather.enclosure * 0.3).clamp(0.0, 1.0);
            let m = [
                spectral_centroid,
                spectral_spread,
                spectral_skewness,
                transient_density,
                rms_energy,
                diffuseness,
            ];
            let s = if let Some(custom_clap) = self.custom_clap_embedding {
                custom_clap
            } else if !self.conditioning_prompt.trim().is_empty() {
                crate::conditioning::compute_text_semantic_projection(&self.conditioning_prompt)
            } else {
                [1.0 / (18.0f32).sqrt(); 18]
            };
            (m, s)
        };

        crate::conditioning::encode_conditioning_vector(
            &macro_atmosphere,
            &material_props,
            &wind_dynamics,
            &side_sounds,
            drift,
            &acoustic_metrics,
            &semantic_projection,
        )
    }

    /// Converts the current UI parameters into the exact 64-dimensional conditioning vector u
    pub fn to_conditioning_vector(&self) -> Vec<f32> {
        self.to_conditioning_array().to_vec()
    }
}
