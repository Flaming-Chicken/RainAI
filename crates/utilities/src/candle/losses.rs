//! Loss Functions and Regularizers for Neural Training.
//!
//! Includes 2nd-order physics trajectory losses, MoE load balancing and router z-loss,
//! expert diversity, Beta-VAE KL divergence, HWIL latency penalties, flow matching,
//! and multi-resolution reconstruction loss.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use super::*;

pub fn compute_physics_trajectory_loss(
    z_pred: &Tensor,
    z_target: &Tensor,
    z_prev: &Tensor,
    lambda_vel: f64,
) -> Result<Tensor> {
    let diff = (z_pred - z_target)?;
    let mse = diff.sqr()?.mean_all()?;

    let v_pred = (z_pred - z_prev)?;
    let v_target = (z_target - z_prev)?;
    let v_diff = (v_pred - v_target)?;
    let v_loss = v_diff.sqr()?.mean_all()?;

    let total = (&mse + (&v_loss * lambda_vel)?)?;
    Ok(total)
}

/// 2nd-Order Physics-Informed Trajectory Smoothness & Aerodynamic Drag Loss.
/// Evaluates:
/// 1. Position MSE: ||z_pred - z_target||^2
/// 2. 1st-order velocity continuity: Huber(v_pred - v_target)
/// 3. 2nd-order acceleration (jerk) smoothness: Huber(a_pred - a_target) where a_t = z_t - 2*z_{t-1} + z_{t-2}
/// 4. Terminal aerodynamic drag dissipation: ReLU(||v_pred||_2 - v_terminal)^2
pub fn compute_physics_trajectory_loss_v2(
    z_pred: &Tensor,
    z_target: &Tensor,
    z_prev: &Tensor,
    z_prev2: Option<&Tensor>,
    lambda_vel: f64,
    lambda_acc: f64,
    lambda_drag: f64,
    v_terminal: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let diff = (z_pred - z_target)?;
    let pos_loss = diff.sqr()?.mean_all()?;

    let v_pred = (z_pred - z_prev)?;
    let v_target = (z_target - z_prev)?;
    let v_diff = (v_pred.clone() - &v_target)?;
    let vel_loss = crate::stft_loss::huber_loss(&v_diff, 0.5)?;

    let acc_loss = if let Some(prev2) = z_prev2 {
        let two_z_prev = (z_prev * 2.0)?;
        let a_pred = ((z_pred - &two_z_prev)? + prev2)?;
        let a_target = ((z_target - &two_z_prev)? + prev2)?;
        let a_diff = (a_pred - a_target)?;
        crate::stft_loss::huber_loss(&a_diff, 0.5)?
    } else {
        Tensor::zeros((), DType::F32, z_pred.device())?
    };

    let v_pred_sq = v_pred.sqr()?.sum_keepdim(1)?.relu()?;
    let speed = (v_pred_sq + 1e-6)?.sqrt()?;
    let excess_speed = (speed - v_terminal)?.relu()?;
    let drag_loss = excess_speed.sqr()?.mean_all()?;

    let total_acc = (&pos_loss + (&vel_loss * lambda_vel)?)?;
    let total_drag = (&total_acc + (&acc_loss * lambda_acc)?)?;
    let total = (&total_drag + (&drag_loss * lambda_drag)?)?;

    Ok((total, pos_loss, acc_loss, drag_loss))
}

/// Smooth hyperbolic tangent soft-capping function: SoftCap(L, M) = M * tanh(L / M).
/// For normal losses (L << M), gradients are identical to L (1.0).
/// For extreme loss spikes (L >> M), the loss saturates smoothly at M with zero gradient,
/// making the training loop immune to anomalous batch gradient shocks.
pub fn soft_cap_loss(loss: &Tensor, max_val: f64) -> Result<Tensor> {
    let scaled = (loss / max_val)?;
    let capped = (scaled.tanh()? * max_val)?;
    Ok(capped)
}

/// Dimension Outlier Spike Suppression Loss:
/// Penalizes activation dimensions exceeding `tau_max` (default: 3.5 standard deviations)
/// and disproportionate Peak-to-Average Power Ratios (PAPR) across channels.
/// Directly suppresses activation outliers in attention KV caches and latent states.
pub fn compute_dimension_outlier_spike_loss(x: &Tensor, tau_max: f64) -> Result<Tensor> {
    let abs_x = x.abs()?;
    let tau_t = Tensor::full(tau_max as f32, x.shape(), x.device())?;
    let excess = (abs_x.broadcast_sub(&tau_t))?.relu()?;
    let threshold_penalty = excess.sqr()?.mean_all()?;

    // Peak-to-average power ratio across hidden channels
    let mean_mag = (abs_x.mean_keepdim(1)? + 1e-6)?;
    let max_mag = abs_x.max_keepdim(1)?;
    let papr = (max_mag.broadcast_div(&mean_mag)? - 1.0)?.relu()?;
    let papr_penalty = papr.sqr()?.mean_all()?;

    let total = (&threshold_penalty + (&papr_penalty * 0.1)?)?;
    Ok(total)
}

/// Isometric / Orthogonality Regularization Loss for Affine Alignment Matrices:
/// Penalizes deviation from exact isometry: L_ortho = ||W^T * W - I||_F^2.
/// Guarantees that learned affine alignment transforms preserve Euclidean norms,
/// bound singular values sigma_i(W) approx 1.0, and cannot collapse rank or explode activations.
pub fn compute_orthogonality_loss(weight: &Tensor) -> Result<Tensor> {
    let dims = weight.dims();
    if dims.len() != 2 || dims[0] != dims[1] {
        return Ok(Tensor::zeros((), DType::F32, weight.device())?);
    }
    let n = dims[0];
    let w_t_w = weight.t()?.matmul(weight)?;
    let eye = Tensor::eye(n, DType::F32, weight.device())?;
    let diff = (w_t_w - eye)?;
    let loss = diff.sqr()?.mean_all()?;
    Ok(loss)
}

/// Contractive Attractor Trajectory Regularization (Lyapunov Stability):
/// Penalizes runaway velocity expansion: ReLU(||z_{t+1} - z_t|| / (||z_t - z_{t-1}|| + eps) - max_ratio)^2.
/// Guarantees that recurrent state trajectories cannot exponentially diverge without external stimulus.
pub fn compute_contractive_loss(
    z_pred: &Tensor,
    z_prev: &Tensor,
    z_prev2: Option<&Tensor>,
    max_ratio: f64,
) -> Result<Tensor> {
    if let Some(prev2) = z_prev2 {
        let v_curr = (z_pred - z_prev)?.sqr()?.sum_keepdim(1)?.relu()?.sqrt()?;
        let v_prev = ((z_prev - prev2)?.sqr()?.sum_keepdim(1)?.relu()? + 1e-6)?.sqrt()?;
        let ratio = (v_curr.broadcast_div(&v_prev)? - max_ratio)?.relu()?;
        let loss = ratio.sqr()?.mean_all()?;
        Ok(loss)
    } else {
        Ok(Tensor::zeros((), DType::F32, z_pred.device())?)
    }
}

/// Auxiliary load balancing loss penalizing expert imbalance:
/// $\mathcal{L}_{aux} = N \sum_{e=1}^N f_e \cdot P_e$.
pub fn compute_moe_load_balancing_loss(router_probs: &Tensor) -> Result<Tensor> {
    // Mean probability per expert across batch: [8]
    let mean_probs = router_probs.mean(0)?;
    let sq = mean_probs.sqr()?;
    let sum_sq = sq.sum_all()?;
    let aux = (&sum_sq * (NUM_EXPERTS as f64))?;
    Ok(aux)
}

/// Router Z-Loss for MoE numerical stability and floating-point overflow prevention.
/// Penalizes extreme logit magnitudes: L_z = 1/B sum_b (log sum_e exp(z_{b, e}))^2.
pub fn compute_router_z_loss(router_logits: &Tensor) -> Result<Tensor> {
    let max_logit = router_logits.max_keepdim(1)?;
    let exp_diff = router_logits
        .broadcast_sub(&max_logit)?
        .clamp(-20.0f32, 20.0f32)?
        .exp()?;
    let sum_exp = exp_diff.sum_keepdim(1)?.clamp(1e-8f32, 1e8f32)?;
    let log_sum_exp = (&max_logit + &sum_exp.log()?)?;
    let z_loss = log_sum_exp.sqr()?.mean_all()?;
    Ok(z_loss)
}

/// Computes pairwise cosine similarity between router probability distributions across thinking steps.
/// Penalizes collinear expert activation, encouraging orthogonal specialist panels across deliberation depth:
/// L_div = 1 / (M choose 2) * sum_{j < k} (p_j . p_k) / (||p_j|| * ||p_k|| + eps)
pub fn compute_expert_diversity_loss(prob_history: &[Tensor]) -> Result<Tensor> {
    let m = prob_history.len();
    if m < 2 {
        return Ok(Tensor::zeros((), DType::F32, prob_history[0].device())?);
    }

    let mut pair_sim_sum = Tensor::zeros((), DType::F32, prob_history[0].device())?;
    let mut num_pairs = 0usize;
    let eps = 1e-6f64;

    for j in 0..m {
        for k in (j + 1)..m {
            let p_j = &prob_history[j];
            let p_k = &prob_history[k];
            let dot = (p_j * p_k)?.sum_keepdim(1)?;
            let norm_j = (p_j.sqr()?.sum_keepdim(1)? + (eps * eps))?.sqrt()?;
            let norm_k = (p_k.sqr()?.sum_keepdim(1)? + (eps * eps))?.sqrt()?;
            let denom = (&norm_j * &norm_k)?.clamp(1e-7f32, 1e7f32)?;
            let cos_sim = dot.broadcast_div(&denom)?.mean_all()?;
            pair_sim_sum = (&pair_sim_sum + &cos_sim)?;
            num_pairs += 1;
        }
    }

    if num_pairs > 0 {
        Ok((pair_sim_sum / (num_pairs as f64))?)
    } else {
        Ok(Tensor::zeros((), DType::F32, prob_history[0].device())?)
    }
}

/// Beta-VAE loss: Reconstruction MSE + $\beta \cdot \text{KL}(q(z|x) \| p(z))$.
/// Beta-VAE loss with Free-Bits thresholding to prevent posterior collapse:
/// For each latent dimension d, enforces KL_d >= free_bits nats.
/// Below free_bits, the gradient is 0, guaranteeing the latent code cannot be crushed into white noise.
/// Also computes the active latent units count (dimensions where Var_B(mu) > 0.01).
pub fn compute_beta_vae_loss_with_free_bits(
    pred_bands: &Tensor,
    target_bands: &Tensor,
    mu: &Tensor,
    logvar: &Tensor,
    beta: f64,
    free_bits: f64,
) -> Result<(Tensor, Tensor, Tensor, usize)> {
    let recon_diff = (pred_bands - target_bands)?;
    let recon_loss = recon_diff.sqr()?.mean_all()?;

    // Numerical clamp on logvar to [-12.0, 12.0] to prevent exponential overflow to infinity and NaN
    let logvar_clamped = logvar.clamp(-12.0f32, 12.0f32)?;
    let mu_sq = mu.sqr()?;
    let var = logvar_clamped.exp()?;
    let ones = Tensor::ones(logvar.shape(), DType::F32, logvar.device())?;
    // KL element-wise: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    let inner = (((&mu_sq + &var)? - &ones)? - &logvar_clamped)?;
    let kl_elements = (&inner * 0.5)?;

    // Mean KL per latent dimension across batch: [D]
    let kl_per_dim = kl_elements.mean(0)?;

    // Free-bits floor
    let kl_loss = if free_bits > 1e-6 {
        let fb = Tensor::full(free_bits as f32, kl_per_dim.shape(), kl_per_dim.device())?;
        let excess = (kl_per_dim.broadcast_sub(&fb))?.relu()?;
        (&excess + &fb)?.mean_all()?
    } else {
        kl_per_dim.mean_all()?
    };

    // Calculate active latent units (Var_B(mu) > 0.01)
    let active_units = if mu.dim(0)? > 1 {
        let b = mu.dim(0)? as f64;
        let mean_mu = mu.mean_keepdim(0)?;
        let diff = mu.broadcast_sub(&mean_mu)?;
        let var_mu = (diff.sqr()?.sum_keepdim(0)? / (b - 1.0))?;
        let var_vec = var_mu.flatten_all()?.to_vec1::<f32>()?;
        var_vec.iter().filter(|&&v| v > 0.01).count()
    } else {
        LATENT_DIM
    };

    let total = (&recon_loss + (&kl_loss * beta)?)?;
    Ok((total, recon_loss, kl_loss, active_units))
}

/// Backward-compatible Beta-VAE loss wrapper (zero free-bits threshold).
pub fn compute_beta_vae_loss(
    pred_bands: &Tensor,
    target_bands: &Tensor,
    mu: &Tensor,
    logvar: &Tensor,
    beta: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (total, recon_loss, kl, _active) =
        compute_beta_vae_loss_with_free_bits(pred_bands, target_bands, mu, logvar, beta, 0.0)?;
    Ok((total, recon_loss, kl))
}

/// Hardware-in-the-Loop (HWIL) governor budget penalty.
pub fn compute_hwil_penalty(
    active_experts: usize,
    budget_experts: usize,
    buffer_health_ms: f32,
    target_buffer_ms: f32,
) -> f32 {
    let expert_penalty = if active_experts > budget_experts {
        (active_experts - budget_experts) as f32 * 0.15
    } else {
        0.0
    };

    let buffer_deficit = (target_buffer_ms - buffer_health_ms).max(0.0) / target_buffer_ms.max(1.0);
    expert_penalty + buffer_deficit * 0.25
}

/// Continuous, differentiable Hardware-in-the-Loop (HWIL) governor budget and buffer penalty.
pub fn compute_continuous_hwil_penalty(
    buffer_health_ms: f32,
    target_buffer_ms: f32,
    active_experts: f32,
    budget_experts: f32,
) -> f32 {
    let buffer_deficit =
        ((target_buffer_ms - buffer_health_ms).max(0.0) / target_buffer_ms.max(1.0)).powi(2);
    let expert_excess = ((active_experts - budget_experts).max(0.0) * 0.15).powi(2);
    expert_excess + buffer_deficit * 0.35
}

/// Conditional Optimal Transport (OT) Flow Matching Loss.
/// $\mathcal{L}_{flow} = \| v_{pred} - (z_{target} - (1 - \sigma_{min}) z_{noise}) \|^2$.
/// Mirrors `compute_flow_matching_loss` from `src/models/mamba2_moe.py`.
pub fn compute_flow_matching_loss(
    pred_velocity: &Tensor,
    z_target: &Tensor,
    z_noise: &Tensor,
    sigma_min: f64,
) -> Result<Tensor> {
    let scale = 1.0 - sigma_min;
    let target_velocity = (z_target - (z_noise * scale)?)?;
    let diff = (pred_velocity - &target_velocity)?;
    let loss = diff.sqr()?.mean_all()?;
    Ok(loss)
}

/// Straight-Path Conditional Optimal Transport Flow Matching Loss.
/// Regularizes probability flow trajectories toward straight paths:
/// L_flow = ||v_pred - target_v||^2 + lambda_straight * ||v_pred - mean_v||^2.
pub fn compute_straight_flow_loss(
    pred_velocity: &Tensor,
    z_target: &Tensor,
    z_noise: &Tensor,
    sigma_min: f64,
    lambda_straight: f64,
) -> Result<(Tensor, Tensor)> {
    let scale = 1.0 - sigma_min;
    let target_velocity = (z_target - (z_noise * scale)?)?;
    let diff = (pred_velocity - &target_velocity)?;
    let base_flow = diff.sqr()?.mean_all()?;

    let mean_target = target_velocity.mean_keepdim(0)?;
    let curvature = pred_velocity
        .broadcast_sub(&mean_target)?
        .sqr()?
        .mean_all()?;
    let total = (&base_flow + (&curvature * lambda_straight)?)?;
    Ok((total, base_flow))
}

/// Hierarchical Multi-Resolution Reconstruction Loss evaluating error
/// across multiple acoustic sampling tiers (16kHz, 32kHz, 48kHz).
/// Mirrors `HierarchicalMultiResLoss` from `src/models/diff_autoencoder.py`.
pub fn compute_hierarchical_multi_res_loss(
    pred_audio: &Tensor,
    target_audio: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let diff = (pred_audio - target_audio)?;
    let l_fine = diff.abs()?.mean_all()?;
    let l_energy = diff.sqr()?.mean_all()?;
    let l_total = ((&l_fine + &l_energy)? * 0.5)?;
    Ok((l_total, l_fine, l_energy))
}

/// VICReg-style Latent Variance Hinge Loss preventing dimensional collapse:
/// L_var = 1/D sum_d relu(target_std - sqrt(Var_B(z_d) + eps))^2.
/// Enforces that all D latent channels maintain at least `target_std` (default 1.0)
/// spread across batch samples, preventing low-rank subspace collapse.
pub fn compute_latent_variance_loss(z: &Tensor, target_std: f64) -> Result<Tensor> {
    let batch_size = z.dim(0)?;
    if batch_size < 2 {
        return Ok(Tensor::zeros((), DType::F32, z.device())?);
    }
    let mean = z.mean_keepdim(0)?;
    let diff = z.broadcast_sub(&mean)?;
    let var = (diff.sqr()?.sum_keepdim(0)? / ((batch_size - 1) as f64))?;
    let std = (var + 1e-4)?.sqrt()?;
    let target_t = Tensor::full(target_std as f32, std.shape(), std.device())?;
    let hinge = (target_t - std)?.relu()?;
    let loss = hinge.sqr()?.mean_all()?;
    Ok(loss)
}

/// Router Shannon Entropy Loss & Perplexity Metric preventing MoE winner-take-all collapse:
/// Evaluates negative entropy across batch-averaged expert probabilities:
/// L_entropy = sum_e P_e * log(P_e + eps).
/// Minimizing this maximizes router entropy H(P).
/// Returns: (entropy_loss_tensor, perplexity_scalar, dead_expert_count).
pub fn compute_router_entropy_loss(router_probs: &Tensor) -> Result<(Tensor, f32, usize)> {
    let mean_probs = router_probs.mean(0)?; // [NUM_EXPERTS]
    let eps = 1e-8f32;
    let probs_vec = mean_probs.to_vec1::<f32>()?;

    let mut entropy = 0.0f32;
    let mut dead_count = 0usize;
    let dead_threshold = 0.10f32 / (NUM_EXPERTS as f32); // 0.0125 (under 1.25%)
    for &p in &probs_vec {
        if p > eps {
            entropy -= p * (p + eps).ln();
        }
        if p < dead_threshold {
            dead_count += 1;
        }
    }
    let perplexity = entropy.exp().clamp(1.0, NUM_EXPERTS as f32);

    let log_p = (mean_probs.clone() + (eps as f64))?.log()?;
    let neg_entropy = (mean_probs * log_p)?.sum_all()?;
    Ok((neg_entropy, perplexity, dead_count))
}

/// Trajectory Diversity Preservation Loss preventing mode collapse:
/// Enforces that predicted trajectory latents across different batch samples maintain
/// non-zero batch-wise standard deviation (default min_std = 0.25).
pub fn compute_trajectory_diversity_loss(z_pred: &Tensor, min_std: f64) -> Result<Tensor> {
    let batch_size = z_pred.dim(0)?;
    if batch_size < 2 {
        return Ok(Tensor::zeros((), DType::F32, z_pred.device())?);
    }
    let mean = z_pred.mean_keepdim(0)?;
    let diff = z_pred.broadcast_sub(&mean)?;
    let var = (diff.sqr()?.sum_keepdim(0)? / ((batch_size - 1) as f64))?;
    let std = (var + 1e-4)?.sqrt()?;
    let min_std_t = Tensor::full(min_std as f32, std.shape(), std.device())?;
    let penalty = (min_std_t - std)?.relu()?;
    let loss = penalty.sqr()?.mean_all()?;
    Ok(loss)
}

/// Live Model Health & Collapse Diagnostics snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCollapseDiagnostics {
    pub active_latents: usize, // Active latent channels (out of 64) with Var > 0.01
    pub latent_total_dim: usize, // 64
    pub router_perplexity: f32, // Router effective expert diversity [1.0 .. 8.0]
    pub dead_experts: usize,   // Experts receiving < 1.25% routing probability
    pub trajectory_variance: f32, // Mean standard deviation of predicted trajectories
    pub status_code: String,   // "OPTIMAL", "POSTERIOR_RISK", "ROUTER_STARVATION", "MODE_COLLAPSE"
    pub is_mitigating: bool,
}

impl Default for ModelCollapseDiagnostics {
    fn default() -> Self {
        Self {
            active_latents: LATENT_DIM,
            latent_total_dim: LATENT_DIM,
            router_perplexity: NUM_EXPERTS as f32,
            dead_experts: 0,
            trajectory_variance: 1.0,
            status_code: "OPTIMAL".to_string(),
            is_mitigating: false,
        }
    }
}

/// Bounded, Huber-smoothed dense soup distillation deficit loss.
/// Evaluates: SoftCap(Huber_delta(z_soup - z_pred), max_cap).
/// - For small differences (|e| <= delta): Quadratic 0.5 * e^2 for smooth sub-gradient convergence.
/// - For moderate differences (|e| > delta): Linear delta * |e| - 0.5 * delta^2 to eliminate quadratic divergence.
/// - For extreme divergence: SoftCap asymptotically bounds the total loss at `max_cap`, preventing runaway feedback loops.
pub fn compute_soup_deficit_loss(
    z_soup: &Tensor,
    z_pred: &Tensor,
    delta: f64,
    max_cap: f64,
) -> Result<Tensor> {
    let diff = (z_soup - z_pred)?;
    let huber = crate::stft_loss::huber_loss(&diff, delta)?;
    let capped = soft_cap_loss(&huber, max_cap)?;
    Ok(capped)
}

/// Regularization penalty bounding the Frobenius norm drift of residual expert weights relative to the shared base:
/// L_drift = sum_k relu(||Delta W_k||_F / (||W_base||_F + eps) - max_ratio)^2.
/// Guarantees that residual experts remain true low-rank perturbations without divergent weight explosions.
pub fn compute_expert_drift_loss(
    drift_pairs: &[(&Tensor, &Tensor)],
    max_ratio: f64,
) -> Result<Tensor> {
    if drift_pairs.is_empty() {
        return Ok(Tensor::zeros((), DType::F32, &Device::Cpu)?);
    }

    let dev = drift_pairs[0].0.device();
    let mut total_penalty = Tensor::zeros((), DType::F32, dev)?;
    let eps = 1e-6f64;

    for &(base_w, expert_w) in drift_pairs {
        let base_norm = (base_w.sqr()?.sum_all()? + (eps * eps))?.sqrt()?;
        let delta_w = (expert_w - base_w)?;
        let delta_norm = (delta_w.sqr()?.sum_all()? + (eps * eps))?.sqrt()?;
        let ratio = delta_norm.broadcast_div(&base_norm)?;
        let max_ratio_t = Tensor::full(max_ratio as f32, ratio.shape(), ratio.device())?;
        let excess = (ratio - max_ratio_t)?.relu()?;
        let penalty = excess.sqr()?.mean_all()?;
        total_penalty = (&total_penalty + &penalty)?;
    }

    let count = drift_pairs.len() as f64;
    Ok((total_penalty / count)?)
}

/// Quantization and Efficiency Optimization Loss for the Neural Waveshaper.
/// Regularizes the continuous learned bit-width beta towards minimal viable precision
/// and rewards parameter sparsity pruning to discover the Pareto-optimal default deployment tier.
pub fn compute_waveshaper_efficiency_loss(
    quantizer: &crate::candle::models::CandleLearnedQuantizer,
    target_bits: f64,
    lambda_bits: f64,
    lambda_prune: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let beta = &quantizer.beta;
    let target = Tensor::full(target_bits as f32, beta.shape(), beta.device())?;
    let excess_bits = (beta - &target)?.relu()?;
    let bits_penalty = excess_bits.sqr()?.mean_all()?;

    // Sparsity pruning incentive (rewarding larger dead zones in delta_prune up to 0.25)
    let prune_target = Tensor::full(
        0.25f32,
        quantizer.delta_prune.shape(),
        quantizer.delta_prune.device(),
    )?;
    let prune_margin = (&prune_target - &quantizer.delta_prune)?;
    let prune_penalty = prune_margin.relu()?.sqr()?.mean_all()?;

    let total = ((&bits_penalty * lambda_bits)? + (&prune_penalty * lambda_prune)?)?;
    Ok((total, bits_penalty, prune_penalty))
}

// ============================================================================
// Phase 39: Robust Physical Losses, Efficiency & Grokking Framework
// ============================================================================

/// Smooth C^inf Charbonnier / Pseudo-Huber loss: H_delta(e) = sqrt(e^2 + delta^2) - delta.
/// Everywhere differentiable with smooth derivatives and gradient bounded by 1.0.
pub fn compute_charbonnier_loss(diff: &Tensor, delta: f64) -> Result<Tensor> {
    let delta_sq = delta * delta;
    let delta_sq_t = Tensor::full(delta_sq as f32, diff.shape(), diff.device())?;
    let inner = (diff.sqr()? + delta_sq_t)?;
    let sqrt_term = inner.sqrt()?;
    let delta_t = Tensor::full(delta as f32, diff.shape(), diff.device())?;
    let loss = (sqrt_term - delta_t)?;
    loss.mean_all().map_err(Into::into)
}

/// Continuous-Time Vector Field Charbonnier Loss.
/// Evaluates the difference between predicted vector field f_theta(z, t, u) and target derivative dz*/dt.
pub fn compute_vector_field_charbonnier_loss(
    v_pred: &Tensor,
    v_target: &Tensor,
    delta: f64,
) -> Result<Tensor> {
    let diff = (v_pred - v_target)?;
    compute_charbonnier_loss(&diff, delta)
}

/// Smooth Softplus Aerodynamic Drag Dissipation Loss.
/// Replaces discontinuous ReLU kinks with smooth C^2 dissipation:
/// L_drag = (1/beta * ln(1 + exp(beta * (||v||_2 - v_terminal))))^2.
pub fn compute_smooth_drag_loss(v_pred: &Tensor, v_terminal: f64, beta: f64) -> Result<Tensor> {
    let v_sq = v_pred.sqr()?.sum_keepdim(1)?.relu()?;
    let speed = (v_sq + 1e-6)?.sqrt()?;
    let excess_speed = (speed - v_terminal)?;
    let scaled = (&excess_speed * beta)?;
    let softplus_term = models::candle_softplus(&scaled)?;
    let unscaled = (softplus_term / beta)?;
    let loss = unscaled.sqr()?.mean_all()?;
    Ok(loss)
}

/// Contractive Lyapunov Stability Regularization for learned ODE vector fields.
/// Enforces dissipative, non-divergent dynamics by penalizing positive trace on the symmetric Jacobian:
/// L_lyapunov = relu(Tr(J_sym) + gamma)^2 where J_sym = 0.5 * (J + J^T).
pub fn compute_lyapunov_stability_loss(jacobian: &Tensor, gamma: f64) -> Result<Tensor> {
    let dims = jacobian.dims();
    if dims.len() < 2 {
        return Ok(Tensor::zeros((), DType::F32, jacobian.device())?);
    }
    let j_sym = ((jacobian + jacobian.t()?)? * 0.5)?;
    let n = dims[dims.len() - 1];
    let eye = Tensor::eye(n, DType::F32, jacobian.device())?;
    let diag = (&j_sym * &eye)?.sum_all()?;
    let gamma_t = Tensor::full(gamma as f32, diag.shape(), diag.device())?;
    let penalty = (&diag + gamma_t)?.relu()?;
    let loss = penalty.sqr()?;
    Ok(loss)
}

/// Path Total Acceleration Regularization for Straight-Flow Matching:
/// L_straight = Charbonnier(dv/dt + (v . grad_z) v, delta).
/// Forcing total path acceleration to zero straightens probability trajectories,
/// enabling adaptive ODE solvers (RK45 / Dormand-Prince) to converge in <= 4 evaluations.
pub fn compute_straight_flow_total_accel_loss(
    dv_dt: &Tensor,
    v_grad_v: &Tensor,
    delta: f64,
) -> Result<Tensor> {
    let total_accel = (dv_dt + v_grad_v)?;
    compute_charbonnier_loss(&total_accel, delta)
}

/// Samples Log-Normal time steps and computes Min-SNR importance weights:
/// tau ~ N(0, 1), t = sigmoid(tau), w(t) = min(1.0, SNR(t) / snr_clip).
pub fn sample_lognormal_min_snr_time(
    batch_size: usize,
    device: &Device,
    snr_clip: f64,
) -> Result<(Tensor, Tensor)> {
    let normal_samples = Tensor::randn(0.0f32, 1.0f32, (batch_size, 1), device)?;
    let neg = (&normal_samples * (-1.0f64))?;
    let exp_term = (neg.exp()? + 1.0f64)?;
    let ones = Tensor::ones((batch_size, 1), DType::F32, device)?;
    let t = ones.broadcast_div(&exp_term)?;

    let one_minus_t = (&ones - &t)?;
    let t_sq = (t.sqr()? + 1e-6)?;
    let snr = one_minus_t.sqr()?.broadcast_div(&t_sq)?;
    let snr_clip_t = Tensor::full(snr_clip as f32, snr.shape(), device)?;
    let normalized_snr = snr.broadcast_div(&snr_clip_t)?;
    let weights = normalized_snr.clamp(0.01f32, 1.0f32)?;

    Ok((t, weights))
}

/// Multi-Scale Temporal Envelope Huber Loss (replacing deterministic instantaneous phase).
/// Evaluates temporal envelope matching across low-pass smoothed absolute waveforms
/// using differences at multiple strides [16, 32, 64] samples.
pub fn compute_multiscale_envelope_loss(
    audio_pred: &Tensor,
    audio_target: &Tensor,
) -> Result<Tensor> {
    let abs_pred = audio_pred.abs()?;
    let abs_target = audio_target.abs()?;
    let mut total_loss = Tensor::zeros((), DType::F32, audio_pred.device())?;

    let strides = [16usize, 32, 64];
    for &stride in &strides {
        let len = audio_pred.dim(audio_pred.dims().len() - 1)?;
        if len > stride * 2 {
            let num_pools = len / stride;
            let p_pred = abs_pred.narrow(audio_pred.dims().len() - 1, 0, num_pools * stride)?;
            let p_target =
                abs_target.narrow(audio_target.dims().len() - 1, 0, num_pools * stride)?;
            let diff = (p_pred - p_target)?;
            let huber = crate::stft_loss::huber_loss(&diff, 0.1)?;
            total_loss = (&total_loss + &huber)?;
        }
    }

    let count = strides.len() as f64;
    Ok((total_loss / count)?)
}

/// Ultrasonic Nyquist Anti-Aliasing Penalty (L_alias) for the Neural Waveshaper.
/// Penalizes high-frequency energy in the upper 10% of the spectrum (near Nyquist)
/// approximated by second-order finite differences (Laplacian high-pass filter):
/// HPF(x)[n] = x[n] - 2*x[n-1] + x[n-2].
pub fn compute_waveshaper_anti_aliasing_loss(audio_shaped: &Tensor) -> Result<Tensor> {
    let dims = audio_shaped.dims();
    let len = dims[dims.len() - 1];
    if len < 3 {
        return Ok(Tensor::zeros((), DType::F32, audio_shaped.device())?);
    }
    let x_0 = audio_shaped.narrow(dims.len() - 1, 2, len - 2)?;
    let x_1 = audio_shaped.narrow(dims.len() - 1, 1, len - 2)?;
    let x_2 = audio_shaped.narrow(dims.len() - 1, 0, len - 2)?;

    let two_x1 = (&x_1 * 2.0f64)?;
    let hp = ((&x_0 - &two_x1)? + &x_2)?;
    let loss = hp.sqr()?.mean_all()?;
    Ok(loss)
}

/// Bark-Weighted Multi-Scale STFT Loss.
/// Weights frequency bins with psychoacoustic perceptual curve emphasizing 1 kHz - 6 kHz rain textures.
pub fn compute_bark_weighted_stft_loss(pred_mag: &Tensor, target_mag: &Tensor) -> Result<Tensor> {
    let diff = (pred_mag - target_mag)?;
    let abs_diff = diff.abs()?;

    let num_bins = pred_mag.dim(pred_mag.dims().len() - 1)?;
    let mut weights = Vec::with_capacity(num_bins);
    for i in 0..num_bins {
        let f_norm = (i as f32) / (num_bins as f32);
        let dist = f_norm - 0.20f32;
        let w = 1.0f32 + 0.8f32 * (-15.0f32 * dist * dist).exp();
        weights.push(w);
    }
    let w_tensor = Tensor::from_vec(weights, (1, num_bins), pred_mag.device())?;
    let weighted_diff = abs_diff.broadcast_mul(&w_tensor)?;
    Ok(weighted_diff.mean_all()?)
}

/// Beta-VAE loss with Leaky Softplus Free-Bits:
/// Enforces soft margin: softplus(KL_d - free_bits) + alpha_leak * KL_d.
/// Eliminates zero-gradient locks and permanently dead latent units.
pub fn compute_beta_vae_loss_leaky_free_bits(
    pred_bands: &Tensor,
    target_bands: &Tensor,
    mu: &Tensor,
    logvar: &Tensor,
    beta: f64,
    free_bits: f64,
    alpha_leak: f64,
) -> Result<(Tensor, Tensor, Tensor, usize)> {
    let recon_diff = (pred_bands - target_bands)?;
    let recon_loss = recon_diff.sqr()?.mean_all()?;

    let logvar_clamped = logvar.clamp(-12.0f32, 12.0f32)?;
    let mu_sq = mu.sqr()?;
    let var = logvar_clamped.exp()?;
    let ones = Tensor::ones(logvar.shape(), DType::F32, logvar.device())?;
    let inner = (((&mu_sq + &var)? - &ones)? - &logvar_clamped)?;
    let kl_elements = (&inner * 0.5)?;
    let kl_per_dim = kl_elements.mean(0)?;

    let fb = Tensor::full(free_bits as f32, kl_per_dim.shape(), kl_per_dim.device())?;
    let margin = (kl_per_dim.broadcast_sub(&fb))?;
    let soft_margin = models::candle_softplus(&margin)?;
    let leak = (&kl_per_dim * alpha_leak)?;
    let kl_loss = (&soft_margin + &leak)?.mean_all()?;

    let active_units = if mu.dim(0)? > 1 {
        let b = mu.dim(0)? as f64;
        let mean_mu = mu.mean_keepdim(0)?;
        let diff = mu.broadcast_sub(&mean_mu)?;
        let var_mu = (diff.sqr()?.sum_keepdim(0)? / (b - 1.0))?;
        let var_vec = var_mu.flatten_all()?.to_vec1::<f32>()?;
        var_vec.iter().filter(|&&v| v > 0.01).count()
    } else {
        LATENT_DIM
    };

    let total = (&recon_loss + (&kl_loss * beta)?)?;
    Ok((total, recon_loss, kl_loss, active_units))
}

/// Ledoit-Wolf Covariance Shrinkage for Total Correlation Disentanglement Loss.
/// Regularizes sample covariance C_shrunk = (1 - lambda) * C + lambda * (Tr(C)/D) * I,
/// preventing rank-deficient singularities and gradient explosions under small batch sizes.
pub fn compute_ledoit_wolf_covariance_loss(z: &Tensor, shrinkage_lambda: f64) -> Result<Tensor> {
    let b = z.dim(0)?;
    if b < 2 {
        return Ok(Tensor::zeros((), DType::F32, z.device())?);
    }
    let d = z.dim(1)?;
    let mean = z.mean_keepdim(0)?;
    let centered = z.broadcast_sub(&mean)?;

    let cov = (centered.t()?.matmul(&centered)? / ((b - 1) as f64))?;
    let eye = Tensor::eye(d, DType::F32, z.device())?;
    let tr = (&cov * &eye)?.sum_all()?.to_scalar::<f32>()? as f64;
    let target_diag = tr / (d as f64);
    let target_matrix = (eye * target_diag)?;

    let shrunk_cov = (((&cov * (1.0 - shrinkage_lambda))?) + (&target_matrix * shrinkage_lambda)?)?;
    let diag = (&shrunk_cov * Tensor::eye(d, DType::F32, z.device())?)?;
    let off_diag = (shrunk_cov - diag)?;
    let tc_loss = off_diag.sqr()?.mean_all()?;
    Ok(tc_loss)
}

/// Projected Conflicting Gradients (PCGrad) Projection Operator.
/// If cosine similarity < 0 (conflicting gradients), projects g1 onto the normal plane of g2:
/// g1_proj = g1 - (g1 . g2 / ||g2||^2) * g2.
pub fn pcgrad_project(g1: &Tensor, g2: &Tensor) -> Result<(Tensor, Tensor)> {
    let dot = (g1 * g2)?.sum_all()?.to_scalar::<f32>()? as f64;
    if dot >= 0.0 {
        return Ok((g1.clone(), g2.clone()));
    }

    let norm_sq_1 = (g1.sqr()?.sum_all()?.to_scalar::<f32>()? as f64) + 1e-8;
    let norm_sq_2 = (g2.sqr()?.sum_all()?.to_scalar::<f32>()? as f64) + 1e-8;

    let proj_factor_1 = dot / norm_sq_2;
    let g1_sub = (g2 * proj_factor_1)?;
    let g1_proj = (g1 - &g1_sub)?;

    let proj_factor_2 = dot / norm_sq_1;
    let g2_sub = (g1 * proj_factor_2)?;
    let g2_proj = (g2 - &g2_sub)?;

    Ok((g1_proj, g2_proj))
}

/// Bounded Homoscedastic Multi-Task Uncertainty Loss.
/// Scales tasks via bounded variances: s_i = min_val + (max_val - min_val) * sigmoid(theta_i),
/// preventing the optimizer from blowing up variances to infinite values.
pub fn compute_bounded_uncertainty_loss(
    losses: &[Tensor],
    theta_params: &Tensor,
    min_log_var: f64,
    max_log_var: f64,
) -> Result<Tensor> {
    if losses.is_empty() {
        return Ok(Tensor::zeros((), DType::F32, &Device::Cpu)?);
    }
    let dev = losses[0].device();
    let mut total = Tensor::zeros((), DType::F32, dev)?;
    let span = max_log_var - min_log_var;

    let thetas = theta_params.to_vec1::<f32>()?;
    for (i, loss) in losses.iter().enumerate() {
        let th = thetas[i % thetas.len()] as f64;
        let sig = 1.0 / (1.0 + (-th).exp());
        let log_var = min_log_var + span * sig;
        let precision = (-log_var).exp();

        let weighted_loss = (loss * (0.5 * precision))?;
        let reg_term = Tensor::full((0.5 * log_var) as f32, weighted_loss.shape(), dev)?;
        let task_loss = (&weighted_loss + reg_term)?;
        total = (&total + &task_loss)?;
    }
    Ok(total)
}

/// Masked Acoustic & Physical Modeling (SpecAugment temporal & frequency occlusion).
/// Masks random blocks of frames (temporal) and channels (frequency), forcing
/// recurrent models to learn global continuity and physical imputation.
pub fn apply_spec_augment_mask(features: &Tensor, time_mask_ratio: f64) -> Result<Tensor> {
    let dims = features.dims();
    if dims.is_empty() {
        return Ok(features.clone());
    }
    let dev = features.device();
    let total_elems = features.elem_count();
    let mut mask_vec = vec![1.0f32; total_elems];

    let time_cutoff = (total_elems as f64 * (1.0 - time_mask_ratio)) as usize;
    for i in time_cutoff..total_elems {
        if (i % 5) == 0 {
            mask_vec[i] = 0.0f32;
        }
    }
    let mask_t = Tensor::from_vec(mask_vec, features.shape(), dev)?;
    features.broadcast_mul(&mask_t).map_err(Into::into)
}

/// L1 Activation Sparsity Loss for intermediate synthesis parameters.
/// Drives non-critical control envelopes precisely to zero for branch-prediction bypass in WASM.
pub fn compute_activation_l1_sparsity_loss(
    activations: &[&Tensor],
    lambda_l1: f64,
) -> Result<Tensor> {
    if activations.is_empty() {
        return Ok(Tensor::zeros((), DType::F32, &Device::Cpu)?);
    }
    let dev = activations[0].device();
    let mut total_l1 = Tensor::zeros((), DType::F32, dev)?;
    for &act in activations {
        let l1 = act.abs()?.mean_all()?;
        total_l1 = (&total_l1 + &l1)?;
    }
    let scaled = (total_l1 * lambda_l1)?;
    Ok(scaled)
}

/// Group Distributionally Robust Optimization (Group DRO) Loss.
/// Tracks and dynamically up-weights worst-performing acoustic domains:
/// w_k <- w_k * exp(eta * L_k), normalized so sum(w) = 1.0.
pub fn compute_group_dro_loss(
    domain_losses: &[Tensor],
    domain_weights: &mut [f64],
    eta: f64,
) -> Result<Tensor> {
    if domain_losses.is_empty() || domain_weights.is_empty() {
        return Ok(Tensor::zeros((), DType::F32, &Device::Cpu)?);
    }
    let dev = domain_losses[0].device();
    let n = domain_losses.len().min(domain_weights.len());

    let mut weight_sum = 0.0f64;
    for i in 0..n {
        let loss_val = domain_losses[i].to_scalar::<f32>()? as f64;
        domain_weights[i] *= (eta * loss_val).clamp(-10.0, 10.0).exp();
        weight_sum += domain_weights[i];
    }
    if weight_sum > 1e-8 {
        for w in domain_weights.iter_mut().take(n) {
            *w /= weight_sum;
        }
    }

    let mut total = Tensor::zeros((), DType::F32, dev)?;
    for i in 0..n {
        let weighted = (&domain_losses[i] * domain_weights[i])?;
        total = (&total + &weighted)?;
    }
    Ok(total)
}
