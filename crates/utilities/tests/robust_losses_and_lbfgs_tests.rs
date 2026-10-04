//! Integration Tests for Phase 39: Robust Physical Losses, Efficiency & Dynamic L-BFGS.

use candle_core::{DType, Device, Tensor, Var};
use utilities::candle::{
    DynamicLbfgs, DynamicLbfgsConfig, apply_spec_augment_mask,
    compute_activation_l1_sparsity_loss, compute_bark_weighted_stft_loss,
    compute_beta_vae_loss_leaky_free_bits, compute_bounded_uncertainty_loss,
    compute_charbonnier_loss, compute_group_dro_loss, compute_ledoit_wolf_covariance_loss,
    compute_lyapunov_stability_loss, compute_multiscale_envelope_loss, compute_smooth_drag_loss,
    compute_straight_flow_total_accel_loss, compute_vector_field_charbonnier_loss,
    compute_waveshaper_anti_aliasing_loss, pcgrad_project, sample_lognormal_min_snr_time,
};

#[test]
fn test_charbonnier_loss_smoothness_and_bounds() {
    let dev = Device::Cpu;
    let small_diff = Tensor::new(&[0.01f32, -0.02f32, 0.0f32], &dev).unwrap();
    let large_diff = Tensor::new(&[100.0f32, -50.0f32, 20.0f32], &dev).unwrap();

    let loss_small = compute_charbonnier_loss(&small_diff, 0.1).unwrap();
    let loss_large = compute_charbonnier_loss(&large_diff, 0.1).unwrap();

    let val_small = loss_small.to_scalar::<f32>().unwrap();
    let val_large = loss_large.to_scalar::<f32>().unwrap();

    assert!(val_small >= 0.0);
    assert!(val_large > val_small);
    // Large differences should scale approximately linearly (bounded subgradient), not quadratically
    assert!(val_large < 100.0);
}

#[test]
fn test_vector_field_charbonnier_and_smooth_drag() {
    let dev = Device::Cpu;
    let v_pred = Tensor::randn(0.0f32, 1.0f32, (4, 16), &dev).unwrap();
    let v_target = Tensor::randn(0.0f32, 1.0f32, (4, 16), &dev).unwrap();

    let vf_loss = compute_vector_field_charbonnier_loss(&v_pred, &v_target, 0.5).unwrap();
    assert!(vf_loss.to_scalar::<f32>().unwrap() >= 0.0);

    // Smooth softplus aerodynamic drag dissipation
    let drag_loss = compute_smooth_drag_loss(&v_pred, 1.5, 2.0).unwrap();
    let drag_val = drag_loss.to_scalar::<f32>().unwrap();
    assert!(drag_val >= 0.0);
    assert!(drag_val.is_finite());
}

#[test]
fn test_lyapunov_stability_contractive_penalty() {
    let dev = Device::Cpu;
    // Positive definite / expanding Jacobian -> should incur penalty
    let expanding_j = Tensor::eye(8, DType::F32, &dev).unwrap();
    let loss_expanding = compute_lyapunov_stability_loss(&expanding_j, 0.0).unwrap();
    let val_exp = loss_expanding.to_scalar::<f32>().unwrap();
    assert!(val_exp > 0.0);

    // Negative definite / dissipative Jacobian -> trace is negative, no penalty
    let dissipative_j = (expanding_j * (-2.0f64)).unwrap();
    let loss_dissipative = compute_lyapunov_stability_loss(&dissipative_j, 0.0).unwrap();
    let val_diss = loss_dissipative.to_scalar::<f32>().unwrap();
    assert_eq!(val_diss, 0.0);
}

#[test]
fn test_straight_flow_total_accel_and_min_snr() {
    let dev = Device::Cpu;
    let dv_dt = Tensor::zeros((2, 8), DType::F32, &dev).unwrap();
    let v_grad_v = Tensor::zeros((2, 8), DType::F32, &dev).unwrap();

    let straight_loss = compute_straight_flow_total_accel_loss(&dv_dt, &v_grad_v, 0.1).unwrap();
    assert_eq!(straight_loss.to_scalar::<f32>().unwrap(), 0.0);

    // Log-Normal time step sampling and Min-SNR weights
    let (t_steps, weights) = sample_lognormal_min_snr_time(16, &dev, 5.0).unwrap();
    assert_eq!(t_steps.dims(), &[16, 1]);
    assert_eq!(weights.dims(), &[16, 1]);

    let t_vec = t_steps.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &t in &t_vec {
        assert!(t > 0.0 && t < 1.0);
    }
}

#[test]
fn test_multiscale_envelope_and_waveshaper_anti_aliasing() {
    let dev = Device::Cpu;
    let audio_pred = Tensor::randn(0.0f32, 0.5f32, (2, 512), &dev).unwrap();
    let audio_target = Tensor::randn(0.0f32, 0.5f32, (2, 512), &dev).unwrap();

    let env_loss = compute_multiscale_envelope_loss(&audio_pred, &audio_target).unwrap();
    assert!(env_loss.to_scalar::<f32>().unwrap() >= 0.0);

    let alias_loss = compute_waveshaper_anti_aliasing_loss(&audio_pred).unwrap();
    assert!(alias_loss.to_scalar::<f32>().unwrap() >= 0.0);
}

#[test]
fn test_bark_weighted_stft_and_leaky_free_bits() {
    let dev = Device::Cpu;
    let pred_mag = Tensor::randn(0.0f32, 1.0f32, (2, 128), &dev).unwrap();
    let target_mag = Tensor::randn(0.0f32, 1.0f32, (2, 128), &dev).unwrap();

    let bark_loss = compute_bark_weighted_stft_loss(&pred_mag, &target_mag).unwrap();
    assert!(bark_loss.to_scalar::<f32>().unwrap() > 0.0);

    // Leaky Free-Bits VAE Loss
    let pred_bands = Tensor::randn(0.0f32, 0.5f32, (4, 16), &dev).unwrap();
    let target_bands = Tensor::randn(0.0f32, 0.5f32, (4, 16), &dev).unwrap();
    let mu = Tensor::randn(0.0f32, 0.1f32, (4, 64), &dev).unwrap();
    let logvar = Tensor::full(-2.0f32, (4, 64), &dev).unwrap();

    let (total, recon, kl, active) = compute_beta_vae_loss_leaky_free_bits(
        &pred_bands,
        &target_bands,
        &mu,
        &logvar,
        1.0,
        0.25,
        0.05,
    )
    .unwrap();

    assert!(total.to_scalar::<f32>().unwrap() > 0.0);
    assert!(recon.to_scalar::<f32>().unwrap() >= 0.0);
    assert!(kl.to_scalar::<f32>().unwrap() > 0.0);
    assert!(active <= 64);
}

#[test]
fn test_ledoit_wolf_covariance_shrinkage() {
    let dev = Device::Cpu;
    let z = Tensor::randn(0.0f32, 1.0f32, (8, 16), &dev).unwrap();
    let tc_loss = compute_ledoit_wolf_covariance_loss(&z, 0.15).unwrap();
    assert!(tc_loss.to_scalar::<f32>().unwrap() >= 0.0);
}

#[test]
fn test_pcgrad_projection_orthogonalization() {
    let dev = Device::Cpu;
    // Opposing gradients: g1 = [1, 0], g2 = [-1, 0]
    let g1 = Tensor::new(&[1.0f32, 0.0f32], &dev).unwrap();
    let g2 = Tensor::new(&[-1.0f32, 0.0f32], &dev).unwrap();

    let (g1_proj, g2_proj) = pcgrad_project(&g1, &g2).unwrap();

    // After projection, conflicting component should be eliminated
    let dot = (&g1_proj * &g2_proj)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(dot >= -1e-6);
}

#[test]
fn test_bounded_uncertainty_and_spec_augment_and_dro() {
    let dev = Device::Cpu;
    let l1 = Tensor::new(1.5f32, &dev).unwrap();
    let l2 = Tensor::new(0.5f32, &dev).unwrap();
    let thetas = Tensor::new(&[0.0f32, 0.0f32], &dev).unwrap();

    let bounded_loss = compute_bounded_uncertainty_loss(&[l1.clone(), l2.clone()], &thetas, -2.0, 2.0).unwrap();
    assert!(bounded_loss.to_scalar::<f32>().unwrap() > 0.0);

    // SpecAugment masking
    let features = Tensor::ones((4, 32), DType::F32, &dev).unwrap();
    let masked = apply_spec_augment_mask(&features, 0.25).unwrap();
    let sum_masked = masked.sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(sum_masked < 128.0);

    // L1 activation sparsity
    let act = Tensor::randn(0.0f32, 1.0f32, (4, 16), &dev).unwrap();
    let l1_loss = compute_activation_l1_sparsity_loss(&[&act], 0.01).unwrap();
    assert!(l1_loss.to_scalar::<f32>().unwrap() > 0.0);

    // Group DRO weight updating
    let mut domain_weights = [0.5f64, 0.5f64];
    let dro_loss = compute_group_dro_loss(&[l1, l2], &mut domain_weights, 0.1).unwrap();
    assert!(dro_loss.to_scalar::<f32>().unwrap() > 0.0);
    // Higher loss domain (l1=1.5) should receive higher weight than l2=0.5
    assert!(domain_weights[0] > domain_weights[1]);
}

#[test]
fn test_dynamic_lbfgs_optimizer_convergence() {
    let dev = Device::Cpu;
    // Quadratic minimization problem: f(w) = 0.5 * ||w - target||^2
    let target = Tensor::new(&[2.0f32, -3.0f32, 1.5f32], &dev).unwrap();
    let w_var = Var::from_tensor(&Tensor::zeros(3, DType::F32, &dev).unwrap()).unwrap();

    let config = DynamicLbfgsConfig {
        lr: 0.5,
        history_size: 5,
        weight_decay: 0.0,
        damping_factor: 0.2,
    };
    let mut opt = DynamicLbfgs::new(vec![w_var.clone()], config);

    let initial_loss = (w_var.as_tensor() - &target)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    for _ in 0..15 {
        let diff = (w_var.as_tensor() - &target).unwrap();
        let loss = diff.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
    }

    let final_loss = (w_var.as_tensor() - &target)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    assert!(final_loss < initial_loss * 0.05);
}
