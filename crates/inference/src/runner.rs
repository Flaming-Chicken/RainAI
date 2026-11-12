//! Neural Inference Runner for block-based autoregressive audio synthesis with async fallback.

use crate::asset_manager::{AssetManager, ExecutionPath};
use crate::engram::EngramBank;
use crate::kernels;
use crate::model::{QuantizedLayer, QuantizedModelManifest};
use crate::weight_cache_manager::WeightCacheManager;
use crate::weight_loader::{LoadedLayer, WeightBuffer, WeightCache};

use shared::rain::{CONDITION_DIM, QualityTier};

/// Neural model status
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineStatus {
    Unloaded,
    DownloadingWeights,
    Ready,
    OffloadedToWebGpu,
}

/// Inference runner managing the active quality tier, fallbacks, and neural weights
#[derive(Clone, Debug)]
pub struct InferenceRunner {
    pub target_tier: QualityTier,
    pub active_tier: QualityTier,
    pub target_path: ExecutionPath,
    pub active_path: ExecutionPath,
    pub is_fallback_active: bool,
    pub status: EngineStatus,
    pub assets: AssetManager,
    pub manifest: Option<QuantizedModelManifest>,
    pub min_bit_width: f32,
    pub max_bit_width: f32,
    pub diffusion_bypass: bool,
    pub weight_cache: WeightCache,
    pub _layers: Vec<QuantizedLayer>,
    pub latent_state: [f32; 64],
    pub u_t_buffer: [f32; 64],
    pub crossfade_counter: usize,
    pub thinking_steps: usize,
    pub use_consistency_jump: bool,
    pub ssm_momentum_alpha: f32,
    pub cached_cond_hash: u64,
    pub cond_cache_valid: bool,
    pub engram_bank: Option<EngramBank>,
    pub latent_kv_cache: Vec<[f32; 32]>,
    pub stochastic_brown_driver: [f32; 64],
    pub rng_state: u64,
}

impl Default for InferenceRunner {
    fn default() -> Self {
        Self::new(QualityTier::AdaptiveMinimum, WeightCache::default())
    }
}

impl InferenceRunner {
    #[inline]
    fn dispatch_projection(
        layer: &LoadedLayer,
        input: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
    ) {
        match &layer.buffer {
            WeightBuffer::Ternary2Bit { packed, gamma } => {
                kernels::ternary_matmul_simd_f32(packed, input, output, *gamma);
                if let Some(b) = bias {
                    for (out, &b_val) in output.iter_mut().zip(b.iter()) {
                        *out += b_val;
                    }
                }
            }
            WeightBuffer::Int8 { weights, scale } => {
                kernels::int8_matmul_simd_f32(weights, input, output, *scale);
                if let Some(b) = bias {
                    for (out, &b_val) in output.iter_mut().zip(b.iter()) {
                        *out += b_val;
                    }
                }
            }
            WeightBuffer::Posit8 { raw, scale } => {
                kernels::posit8_matmul_simd_f32(raw, input, output, *scale);
                if let Some(b) = bias {
                    for (out, &b_val) in output.iter_mut().zip(b.iter()) {
                        *out += b_val;
                    }
                }
            }
            WeightBuffer::Bf16 { raw, scale } => {
                kernels::bf16_matmul_simd_f32(raw, input, output, *scale);
                if let Some(b) = bias {
                    for (out, &b_val) in output.iter_mut().zip(b.iter()) {
                        *out += b_val;
                    }
                }
            }
            _ => {
                kernels::dense_projection(input, &layer.weights, bias, output);
            }
        }
    }

    pub fn new(tier: QualityTier, weight_cache: WeightCache) -> Self {
        let assets = AssetManager::new();
        let (active_tier, is_fallback) = assets.resolve_effective_tier(tier);
        let manifest = QuantizedModelManifest::load_default_ternary().ok();

        let (min_bits, max_bits) = if let Some(ref m) = manifest {
            let min_b = m
                .activation_qat
                .per_channel_bit_widths
                .iter()
                .copied()
                .fold(32.0f32, f32::min);
            let max_b = m
                .activation_qat
                .per_channel_bit_widths
                .iter()
                .copied()
                .fold(1.58f32, f32::max);
            (min_b, max_b)
        } else {
            (1.58, 8.0)
        };

        Self {
            target_tier: tier,
            active_tier,
            target_path: ExecutionPath::CpuNeural,
            active_path: ExecutionPath::CpuNeural,
            is_fallback_active: is_fallback,
            status: EngineStatus::Ready,
            assets,
            manifest,
            min_bit_width: min_bits,
            max_bit_width: max_bits,
            diffusion_bypass: false,
            weight_cache,
            _layers: Vec::new(),
            latent_state: [0.0; 64],
            u_t_buffer: [0.0; 64],
            crossfade_counter: 0,
            thinking_steps: 3,
            use_consistency_jump: false,
            ssm_momentum_alpha: 0.10,
            cached_cond_hash: 0,
            cond_cache_valid: false,
            engram_bank: Some(EngramBank::new()),
            latent_kv_cache: Vec::new(),
            stochastic_brown_driver: [0.0; 64],
            rng_state: 88172645463325252,
        }
    }

    /// Sets the SSM state momentum damping coefficient (clamped 0.0..=0.5).
    /// Stabilizes recurrent trajectory dynamics during turbulent gusts.
    pub fn set_ssm_momentum_alpha(&mut self, alpha: f32) {
        self.ssm_momentum_alpha = alpha.clamp(0.0, 0.5);
    }

    /// Bidirectional lookahead trajectory smoothing on conditioning vectors.
    /// Uses a non-causal Gaussian kernel across the window to eliminate parameter jitter.
    pub fn smooth_conditioning_trajectory(trajectory: &mut [[f32; CONDITION_DIM]]) {
        let n = trajectory.len();
        if n < 3 {
            return;
        }
        let original = trajectory.to_vec();
        for i in 1..n - 1 {
            for d in 0..CONDITION_DIM {
                trajectory[i][d] =
                    0.25 * original[i - 1][d] + 0.50 * original[i][d] + 0.25 * original[i + 1][d];
            }
        }
    }

    /// Set latent deliberation thinking steps (clamped 1..=5)
    pub fn set_thinking_steps(&mut self, steps: usize) {
        self.thinking_steps = steps.clamp(1, 5);
    }

    /// Set pre-generated deliberation steps (clamped 1..=5).
    /// Dynamically alterable by the Invasive Meta-Controller or user override.
    pub fn set_pre_generated_steps(&mut self, steps: usize) {
        self.set_thinking_steps(steps);
    }

    /// Get current amount of pre-generated deliberation steps
    pub fn pre_generated_steps(&self) -> usize {
        self.thinking_steps
    }

    /// Enable or disable 1-step consistency distillation jump head
    pub fn set_use_consistency_jump(&mut self, enabled: bool) {
        self.use_consistency_jump = enabled;
    }

    /// Returns true if the consistency jump head weights are loaded in cache
    pub fn has_consistency_jump_head(&self) -> bool {
        self.weight_cache.contains("consistency_head.proj.weight")
    }

    /// Enable or disable latent diffusion bypass under emergency panic conditions
    pub fn set_diffusion_bypass(&mut self, bypass: bool) {
        self.diffusion_bypass = bypass;
    }

    #[inline]
    fn ensure_conditioning_projection(&mut self, conditioning: &[f32; CONDITION_DIM]) {
        // Fast hash-check for conditioning gateway cache validity over the 64-dimensional vector
        let mut cond_hash = 14695981039346656037u64;
        for (i, &v) in conditioning.iter().enumerate() {
            cond_hash = (cond_hash ^ (v.to_bits() as u64).wrapping_mul((i + 1) as u64))
                .wrapping_mul(1099511628211);
        }

        if !self.cond_cache_valid || self.cached_cond_hash != cond_hash {
            if let Some(cond_layer) = self.weight_cache.get("encoder.cond_proj.weight") {
                let bias = self.weight_cache.get("encoder.cond_proj.bias");
                let bias_ref = bias.as_ref().map(|b| b.weights.as_slice());
                Self::dispatch_projection(
                    &cond_layer,
                    conditioning,
                    bias_ref,
                    &mut self.u_t_buffer,
                );
            }
            kernels::simd_silu_in_place(&mut self.u_t_buffer);
            self.cached_cond_hash = cond_hash;
            self.cond_cache_valid = true;
        }
    }

    /// Fast 1-step distilled inference directly projecting conditioning through consistency jump head
    pub fn fast_consistency_step(
        &mut self,
        conditioning: &[f32; CONDITION_DIM],
    ) -> (f32, f32, f32, f32) {
        // 1. Conditioning Projection (cached when input parameters are invariant)
        self.ensure_conditioning_projection(conditioning);

        // 2. Consistency Jump Head Projection: u_t (64) -> FOA (4)
        let mut foa_out = [0.0; 4];
        if let Some(jump_layer) = self.weight_cache.get("consistency_head.proj.weight") {
            let bias = self.weight_cache.get("consistency_head.proj.bias");
            let bias_ref = bias.as_ref().map(|b| b.weights.as_slice());
            Self::dispatch_projection(&jump_layer, &self.u_t_buffer, bias_ref, &mut foa_out);
        } else {
            return self.step(conditioning);
        }

        let bit_scale = (self.max_bit_width / 8.0).clamp(0.5, 1.5);
        (
            foa_out[0] * bit_scale,
            foa_out[1] * bit_scale,
            foa_out[2] * bit_scale,
            foa_out[3] * bit_scale,
        )
    }

    /// Set requested target tier; triggers async download if needed and selects best available fallback
    pub fn set_target_tier(&mut self, tier: QualityTier) {
        self.target_tier = tier;
        if tier.is_download_required() && !self.assets.tier_state(tier).is_ready() {
            self.assets.trigger_download(tier);
            self.status = EngineStatus::DownloadingWeights;
        }

        let (eff_tier, is_fallback) = self.assets.resolve_effective_tier(tier);
        if eff_tier != self.active_tier {
            self.active_tier = eff_tier;
            self.crossfade_counter = 128; // 128-sample micro-crossfade to eliminate clicks
        }
        self.is_fallback_active = is_fallback;

        if !is_fallback {
            self.status = EngineStatus::Ready;
        }
    }

    /// Set requested execution hardware pathway; falls back to CPU if WebGPU pipeline is unready
    pub fn set_target_path(&mut self, path: ExecutionPath) {
        self.target_path = path;
        let prefer_webgpu = path == ExecutionPath::WebGpuNeural;
        let (eff_path, _is_fallback) = self.assets.resolve_effective_path(prefer_webgpu);
        self.active_path = eff_path;

        if eff_path == ExecutionPath::WebGpuNeural {
            self.status = EngineStatus::OffloadedToWebGpu;
        } else if self.status == EngineStatus::OffloadedToWebGpu {
            self.status = EngineStatus::Ready;
        }
    }

    /// Update dynamic continuous bit-width quantization bounds from autonomous governor
    pub fn update_quantization_bounds(&mut self, min_bits: f32, max_bits: f32) {
        self.min_bit_width = min_bits;
        self.max_bit_width = max_bits;
    }

    /// Returns active execution path
    pub fn active_path(&self) -> ExecutionPath {
        self.active_path
    }

    /// Polls background download progress and hot-swaps to target tier when ready
    pub fn poll_downloads(&mut self) {
        if self.assets.tier_state(self.target_tier).is_ready()
            && self.active_tier != self.target_tier
        {
            self.active_tier = self.target_tier;
            self.is_fallback_active = false;
            self.status = EngineStatus::Ready;
            self.crossfade_counter = 128;
        }
    }

    /// Advances the internal zero-allocation brown noise process (-6 dB/oct leaky integration)
    #[inline]
    pub fn step_brown_driver(&mut self) {
        for val in self.stochastic_brown_driver.iter_mut() {
            self.rng_state ^= self.rng_state >> 12;
            self.rng_state ^= self.rng_state << 25;
            self.rng_state ^= self.rng_state >> 27;
            let white = (self.rng_state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as f32
                / 4_294_967_296.0
                - 0.5;
            *val = *val * 0.96 + white * 0.04;
        }
    }

    /// Run one step of inference given the 64-dim conditioning vector
    /// Returns 4-channel FOA values (W, X, Y, Z)
    pub fn step(&mut self, conditioning: &[f32; CONDITION_DIM]) -> (f32, f32, f32, f32) {
        if self.use_consistency_jump && self.has_consistency_jump_head() {
            return self.fast_consistency_step(conditioning);
        }

        let cond_energy: f32 = conditioning.iter().take(32).sum::<f32>() / 32.0;

        // Emergency Latent Diffusion Bypass Mode
        if self.diffusion_bypass {
            let w = cond_energy * 1.414;
            let x = (conditioning[1 % CONDITION_DIM] - 0.5) * 2.0 * cond_energy;
            let y = (conditioning[2 % CONDITION_DIM] - 0.5) * 2.0 * cond_energy;
            let z = (conditioning[7 % CONDITION_DIM] - 0.5) * 2.0 * cond_energy;
            return (w, x, y, z);
        }

        // 1. Advance the stochastic brown noise driver and inject into conditioning projection
        self.step_brown_driver();
        self.ensure_conditioning_projection(conditioning);
        for (u, &b) in self
            .u_t_buffer
            .iter_mut()
            .zip(self.stochastic_brown_driver.iter())
        {
            *u += b * 0.08;
        }

        // 2. Mamba2 Recurrence across Thinking Deliberation Steps (1..=5)
        let iterations = self.thinking_steps.clamp(1, 5);

        for _ in 0..iterations {
            let prev_latent = self.latent_state;

            if let (Some(a_diag), Some(b_diag)) = (
                self.weight_cache.get("mamba.A_diag.weight"),
                self.weight_cache.get("mamba.B_diag.weight"),
            ) {
                kernels::step_recurrence_f32(
                    &mut self.latent_state,
                    &a_diag.weights,
                    &b_diag.weights,
                    &self.u_t_buffer,
                );
            } else {
                // Stochastic recurrent fallback: ensures living dynamic state evolution even when generator weights are uninitialized
                for (s, &u) in self.latent_state.iter_mut().zip(self.u_t_buffer.iter()) {
                    *s = *s * 0.94 + u * 0.06;
                }
            }

            // O(1) Engram physical prior gated lookup
            if let Some(ref engram) = self.engram_bank {
                engram.fuse_in_place(&mut self.latent_state, 0.15);
            }

            // Apply SSM state momentum damping: s_t = alpha * s_{t-1} + (1 - alpha) * s_{recurrence}
            if self.ssm_momentum_alpha > 0.0 {
                let alpha = self.ssm_momentum_alpha;
                for (curr, &prev) in self.latent_state.iter_mut().zip(prev_latent.iter()) {
                    *curr = prev * alpha + *curr * (1.0 - alpha);
                }
            }
        }

        // 4. Ambisonic FOA Projection: y_t = W_foa @ s_t
        let mut foa_out = [0.0; 4];
        if let Some(foa_layer) = self.weight_cache.get("decoder.foa_proj.weight") {
            Self::dispatch_projection(&foa_layer, &self.latent_state, None, &mut foa_out);
        } else {
            // Default physical spatial projection mapping latent components to FOA (W, X, Y, Z)
            foa_out[0] = (self.latent_state[0] * 0.4).tanh() * 1.2 + 1.3;
            foa_out[1] = self.latent_state[1].tanh() * 0.6;
            foa_out[2] = self.latent_state[2].tanh() * 0.6;
            foa_out[3] = self.latent_state[3].tanh() * 0.4;
        }

        if self.crossfade_counter > 0 {
            self.crossfade_counter -= 1;
        }

        // Bit-width scaling for energy compensation across tiers
        let bit_scale = (self.max_bit_width / 8.0).clamp(0.5, 1.5);

        (
            foa_out[0] * bit_scale,
            foa_out[1] * bit_scale,
            foa_out[2] * bit_scale,
            foa_out[3] * bit_scale,
        )
    }

    /// Asynchronously requests a quality tier upgrade and loads weights via IndexedDB/Network
    pub async fn upgrade_tier_async(&mut self, tier: QualityTier) -> Result<(), String> {
        self.set_target_tier(tier);

        if tier.is_download_required() && !self.assets.tier_state(tier).is_ready() {
            let new_cache = WeightCacheManager::load_tier(tier).await?;
            self.weight_cache = new_cache;
            self.cond_cache_valid = false;

            *self.assets.tier_state_mut(tier) = crate::asset_manager::AssetState::Ready;
            self.active_tier = tier;
            self.is_fallback_active = false;
            self.status = EngineStatus::Ready;
        }
        Ok(())
    }

    /// Evaluates a block-rate neural step (50-100 Hz) to produce continuous parametric control signals
    /// that dynamically modulate compiled procedural and physical DSP engines.
    pub fn step_parametric(
        &mut self,
        conditioning: &[f32; CONDITION_DIM],
    ) -> NeuralParametricControl {
        // Run recurrence and get the directional FOA vector
        let (w, x, y, z) = self.step(conditioning);

        let mut band_gains = [1.0f32; 16];
        let mut band_freq_drifts = [0.0f32; 16];

        // Map latent state dimensions 0..16 to filterbank gains with softplus/sigmoid shaping
        for i in 0..16 {
            let latent_val = self.latent_state[i];
            band_gains[i] = (latent_val.tanh() * 0.5 + 1.0).max(0.05);
        }

        // Map latent state dimensions 16..32 to resonant frequency micro-drifts
        for i in 0..16 {
            let drift_val = self.latent_state[16 + i];
            band_freq_drifts[i] = drift_val.tanh() * 0.08; // ±8% frequency variation
        }

        // Latent dimensions 32..36 map to droplet physical parameters and aeroacoustics
        let droplet_rate_mod = (self.latent_state[32].tanh() * 0.75 + 1.0).clamp(0.2, 2.5);
        let droplet_energy_mod = (self.latent_state[33].tanh() * 0.5 + 1.0).clamp(0.4, 2.0);
        let wind_gust_mod = (self.latent_state[34].tanh() * 0.6 + 1.0).clamp(0.2, 2.2);
        let wind_howl_mod = (self.latent_state[35].tanh() * 0.7 + 1.0).clamp(0.1, 2.5);

        NeuralParametricControl {
            band_gains,
            band_freq_drifts,
            droplet_rate_mod,
            droplet_energy_mod,
            wind_gust_mod,
            wind_howl_mod,
            spatial_vector: (w, x, y, z),
        }
    }
}

/// Continuous neural-parametric control signals produced by the Mamba2-MoE model at block rate (50-100 Hz).
/// These signals drive compiled procedural and physical DSP synthesizers without requiring 48kHz neural forward passes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuralParametricControl {
    /// 16-band subtractive filterbank gains (linear multipliers, nominal 1.0)
    pub band_gains: [f32; 16],
    /// 16-band resonant frequency drift offsets (semitone / fractional offset)
    pub band_freq_drifts: [f32; 16],
    /// Raindrop Poisson impact rate modulation (nominal 1.0)
    pub droplet_rate_mod: f32,
    /// Raindrop kinetic energy / diameter modulation (nominal 1.0)
    pub droplet_energy_mod: f32,
    /// Wind gustiness modulation (nominal 1.0)
    pub wind_gust_mod: f32,
    /// Wind howl resonance modulation (nominal 1.0)
    pub wind_howl_mod: f32,
    /// 3D directional Ambisonic vector (W, X, Y, Z)
    pub spatial_vector: (f32, f32, f32, f32),
}

impl Default for NeuralParametricControl {
    fn default() -> Self {
        Self {
            band_gains: [1.0; 16],
            band_freq_drifts: [0.0; 16],
            droplet_rate_mod: 1.0,
            droplet_energy_mod: 1.0,
            wind_gust_mod: 1.0,
            wind_howl_mod: 1.0,
            spatial_vector: (1.0, 0.0, 0.0, 0.0),
        }
    }
}
