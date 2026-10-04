use inference::asset_manager::{AssetManager, AssetState};
use inference::model::{PrecisionFormat, QuantizedModelManifest};
use inference::runner::{EngineStatus, InferenceRunner};
use inference::weight_loader::{LoadedLayer, WeightBuffer, WeightCache, WeightLoader};
use shared::rain::{CONDITION_DIM, QualityTier};

#[test]
fn test_ternary_2bit_pack_unpack_roundtrip() {
    let original = vec![0i8, 1, -1, 1, 0, 0, -1, -1, 1];
    let packed = WeightLoader::pack_ternary_2bit(&original);
    assert_eq!(packed.len(), 3);

    let unpacked = WeightLoader::unpack_ternary_2bit(&packed, original.len());
    assert_eq!(original, unpacked);
}

#[test]
fn test_weight_cache_operations() {
    let mut cache = WeightCache::new();
    let packed = vec![0x05; (64usize * 554).div_ceil(4)];
    let layer = LoadedLayer {
        name: "encoder.cond_proj.weight".to_string(),
        shape: vec![64, 554],
        scale: 0.125,
        format: PrecisionFormat::Ternary158,
        weights: vec![],
        packed_weights: packed.clone(),
        buffer: WeightBuffer::Ternary2Bit {
            packed,
            gamma: 0.125,
        },
    };

    cache.insert(layer);
    assert!(cache.contains("encoder.cond_proj.weight"));

    let retrieved = cache
        .get("encoder.cond_proj.weight")
        .expect("Layer must exist");
    assert_eq!(retrieved.num_elements(), 64 * 554);
    assert!(retrieved.weights.is_empty());
    assert!(!retrieved.packed_weights.is_empty());
}

#[test]
fn test_inference_step_dimension() {
    let weight_cache = WeightCache::default();
    let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum, weight_cache);
    let cond = [0.5f32; CONDITION_DIM];
    let (w, x, y, z) = runner.step(&cond);
    assert!(w.is_finite());
    assert!(x.is_finite());
    assert!(y.is_finite());
    assert!(z.is_finite());
}

#[test]
fn test_tier_transition_with_fallback() {
    let weight_cache = WeightCache::default();
    let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum, weight_cache);
    assert_eq!(runner.status, EngineStatus::Ready);
    assert_eq!(runner.active_tier, QualityTier::AdaptiveMinimum);

    runner.set_target_tier(QualityTier::StudioFp32);
    assert_eq!(runner.status, EngineStatus::DownloadingWeights);
    assert_eq!(runner.active_tier, QualityTier::AdaptiveMinimum);
    assert!(runner.is_fallback_active);

    runner.assets.fp32_state = AssetState::Ready;
    runner.poll_downloads();
    assert_eq!(runner.active_tier, QualityTier::StudioFp32);
    assert!(!runner.is_fallback_active);
    assert_eq!(runner.status, EngineStatus::Ready);
}

#[test]
fn test_load_default_ternary_manifest() {
    let manifest = QuantizedModelManifest::load_default_ternary()
        .expect("Embedded ternary manifest must parse cleanly");
    assert_eq!(manifest.tier, "ternary_1_58bit");
    assert_eq!(manifest.activation_qat.per_channel_bit_widths.len(), 64);
    assert!(manifest.layers.contains_key("encoder.cond_proj.weight"));
}

#[test]
fn test_fallback_chain_when_unready() {
    let mgr = AssetManager::new();
    let (eff, is_fallback) = mgr.resolve_effective_tier(QualityTier::StudioFp32);
    assert_eq!(eff, QualityTier::AdaptiveMinimum);
    assert!(is_fallback);
}

#[test]
fn test_geometric_midpoint_boundaries_and_pareto_allocation() {
    use inference::model::LayerRole;

    // 1. Invariant: Geometric level midpoints
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(0.2),
        PrecisionFormat::Pruned
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(1.0),
        PrecisionFormat::Ternary158
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(1.80),
        PrecisionFormat::Ternary158
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(1.81),
        PrecisionFormat::Int2
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(2.58),
        PrecisionFormat::Int2
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(2.60),
        PrecisionFormat::Posit8
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(3.58),
        PrecisionFormat::Posit8
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(3.60),
        PrecisionFormat::Int4
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(4.58),
        PrecisionFormat::Int4
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(4.60),
        PrecisionFormat::Int5
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(5.58),
        PrecisionFormat::Int5
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(5.60),
        PrecisionFormat::Int8
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(8.0),
        PrecisionFormat::Bf16
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(12.0),
        PrecisionFormat::Fp16
    );
    assert_eq!(
        PrecisionFormat::from_continuous_bit_width(18.0),
        PrecisionFormat::Fp32
    );

    // 2. Invariant: Natural per-layer Pareto allocation
    // SSM Recurrence: Retains Bf16 / Fp32 for recurrent stability
    assert_eq!(
        LayerRole::MambaStateSpaceRecurrence.down_quantization_fallback(8.0),
        PrecisionFormat::Bf16
    );
    assert_eq!(
        LayerRole::MambaStateSpaceRecurrence.down_quantization_fallback(5.0),
        PrecisionFormat::Int5
    );

    // Latent Bottleneck: Prioritizes Posit tapered precision around 0 dBFS
    assert_eq!(
        LayerRole::LatentBottleneck.down_quantization_fallback(6.0),
        PrecisionFormat::Posit16
    );
    assert_eq!(
        LayerRole::LatentBottleneck.down_quantization_fallback(3.0),
        PrecisionFormat::Posit8
    );

    // Ambisonic / Filters: Preserves unitary 3D rotation phase
    assert_eq!(
        LayerRole::AmbisonicRotation.down_quantization_fallback(12.0),
        PrecisionFormat::Fp32
    );
    assert_eq!(
        LayerRole::AmbisonicRotation.down_quantization_fallback(6.0),
        PrecisionFormat::Fp16
    );

    // Projections: Scales smoothly down to coarse integer / ternary
    assert_eq!(
        LayerRole::DenseProjection.down_quantization_fallback(1.5),
        PrecisionFormat::Ternary158
    );
}

#[test]
fn test_simd_kernels_posit_int8_bf16() {
    use inference::kernels;

    let activations = vec![1.0f32, -0.5, 2.0, 0.25];
    let mut out_posit = [0.0f32; 1];
    let mut out_int8 = [0.0f32; 1];
    let mut out_bf16 = [0.0f32; 1];

    // Posit8 kernel test
    let posit_raw = vec![0x40u8, 0x40, 0x40, 0x40]; // Some non-zero posit bytes
    kernels::posit8_matmul_simd_f32(&posit_raw, &activations, &mut out_posit, 1.0);
    assert!(out_posit[0].is_finite());

    // INT8 kernel test
    let int8_weights = vec![127i8, -64, 32, -16];
    kernels::int8_matmul_simd_f32(&int8_weights, &activations, &mut out_int8, 1.0);
    assert!(out_int8[0].is_finite());

    // BF16 kernel test
    let bf16_weights = vec![0x3F80u16, 0x3F80, 0x3F80, 0x3F80]; // 1.0 in BF16 is 0x3F80
    kernels::bf16_matmul_simd_f32(&bf16_weights, &activations, &mut out_bf16, 1.0);
    assert!(out_bf16[0].is_finite());
    // Sum = 1.0*1.0 + 1.0*(-0.5) + 1.0*2.0 + 1.0*0.25 = 2.75
    assert!((out_bf16[0] - 2.75).abs() < 1e-3);
}

#[test]
fn test_multi_iteration_expert_diversity() {
    use inference::kernels;

    let mut latent = [0.1f32; 64];
    // Create router weights where expert 0 has a higher affinity initially
    let mut router_weights = vec![0.0f32; 8 * 64];
    for d in 0..64 {
        router_weights[d] = 0.20; // Expert 0 favored
        router_weights[64 + d] = 0.15; // Expert 1 runner-up
        router_weights[2 * 64 + d] = 0.10; // Expert 2
    }

    let mut weights_iter0 = [0.0f32; 16];
    kernels::route_and_decay_with_history(
        &mut latent,
        &router_weights,
        8,
        0.75,
        0.85,
        None,
        0.0,
        Some(&mut weights_iter0),
    );

    assert!(
        weights_iter0[0] > weights_iter0[1],
        "Expert 0 should be top in iteration 0"
    );

    // In iteration 1, pass iteration 0 weights into deliberation history
    let mut delib_history = [0.0f32; 16];
    delib_history[..8].copy_from_slice(&weights_iter0[..8]);

    let mut weights_iter1 = [0.0f32; 16];
    kernels::route_and_decay_with_history(
        &mut latent,
        &router_weights,
        8,
        0.75,
        0.85,
        Some(&delib_history),
        0.60, // Strong deliberation diversity tabu
        Some(&mut weights_iter1),
    );

    // Dominant expert 0 must be dampened due to deliberation history tabu
    assert!(
        weights_iter1[0] < weights_iter0[0],
        "Expert 0 must be suppressed in subsequent deliberation iteration: iter0={}, iter1={}",
        weights_iter0[0],
        weights_iter1[0]
    );

    // Complementary expert pool (experts 1..8) must gain collective weight, diversifying the deliberation panel
    let comp_weight0: f32 = weights_iter0[1..8].iter().sum();
    let comp_weight1: f32 = weights_iter1[1..8].iter().sum();
    assert!(
        comp_weight1 > comp_weight0,
        "Complementary expert panel must gain mass across iterations: iter0={comp_weight0}, iter1={comp_weight1}"
    );
}

#[test]
fn test_pre_generated_steps_alterable() {
    let weight_cache = WeightCache::default();
    let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum, weight_cache);

    // Default thinking steps is 3
    assert_eq!(runner.pre_generated_steps(), 3);

    // Meta-controller sets steps to 4
    runner.set_pre_generated_steps(4);
    assert_eq!(runner.pre_generated_steps(), 4);

    // High stress drops to 1
    runner.set_pre_generated_steps(1);
    assert_eq!(runner.pre_generated_steps(), 1);

    // Clamping boundaries (1..=5)
    runner.set_pre_generated_steps(0);
    assert_eq!(runner.pre_generated_steps(), 1);

    runner.set_pre_generated_steps(99);
    assert_eq!(runner.pre_generated_steps(), 5);
}

#[test]
fn test_rk4_continuous_flow_trajectory_solver() {
    use inference::kernels::Rk4FlowSolver;

    // Test ODE: dx/dt = -2 * x with x(0) = 1.0.
    // Analytical solution: x(t) = exp(-2 * t). At t = 1.0, x(1.0) = exp(-2) ≈ 0.135335.
    let x0 = vec![1.0f32];
    let num_steps = 20;

    let final_x = Rk4FlowSolver::solve_trajectory(&x0, num_steps, |x, _t| vec![-2.0 * x[0]]);

    let expected = (-2.0f32).exp();
    assert!(
        (final_x[0] - expected).abs() < 1e-4,
        "RK4 numerical solution was {}, expected {}",
        final_x[0],
        expected
    );
}

#[test]
fn test_hardware_compute_router_candle_and_wgsl() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use inference::compute_router::{CandleFlowBackend, HardwareComputeRouter, WgslComputeBackend};
    use spodeian_ml_utils::VelocityFlowHead;
    use std::collections::HashMap;
    use std::sync::Arc;

    let device = Device::Cpu;
    let dim = 2;
    let num_freqs = 2;
    let hidden_dim = 4;

    // 1. Build Candle velocity flow head
    let mut tensors = HashMap::new();
    let time_dim = num_freqs * 2;
    tensors.insert(
        "flow_time_proj.weight".to_string(),
        Tensor::zeros((hidden_dim, time_dim), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_time_proj.bias".to_string(),
        Tensor::zeros(hidden_dim, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_hidden.weight".to_string(),
        Tensor::zeros((hidden_dim, dim + hidden_dim), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_hidden.bias".to_string(),
        Tensor::zeros(hidden_dim, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_out.weight".to_string(),
        Tensor::zeros((dim, hidden_dim), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_out.bias".to_string(),
        Tensor::zeros(dim, DType::F32, &device).unwrap(),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let head = VelocityFlowHead::new(vb, dim, num_freqs, hidden_dim).unwrap();

    let candle_backend = Arc::new(CandleFlowBackend::new("candle-cpu", head, device));
    let wgsl_backend = Arc::new(WgslComputeBackend::new(dim));

    // 2. Test router with Candle prioritized
    let router = HardwareComputeRouter::new(vec![candle_backend.clone(), wgsl_backend.clone()]);

    let active_backend = router.select_backend().unwrap();
    assert_eq!(active_backend.backend_name(), "candle-cpu");

    let x0 = vec![1.0f32, -0.5f32];
    let traj_candle = router.solve_trajectory(&x0, 5).unwrap();
    assert_eq!(traj_candle.len(), dim);

    // 3. Test fallback to WGSL when Candle is unavailable
    let unavailable_candle = Arc::new(CandleFlowBackend {
        name: "candle-cuda",
        head: VelocityFlowHead::new(
            VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu),
            dim,
            num_freqs,
            hidden_dim,
        )
        .unwrap_or_else(|_| {
            let mut t = HashMap::new();
            t.insert(
                "flow_time_proj.weight".into(),
                Tensor::zeros((hidden_dim, time_dim), DType::F32, &Device::Cpu).unwrap(),
            );
            t.insert(
                "flow_time_proj.bias".into(),
                Tensor::zeros(hidden_dim, DType::F32, &Device::Cpu).unwrap(),
            );
            t.insert(
                "flow_hidden.weight".into(),
                Tensor::zeros((hidden_dim, dim + hidden_dim), DType::F32, &Device::Cpu).unwrap(),
            );
            t.insert(
                "flow_hidden.bias".into(),
                Tensor::zeros(hidden_dim, DType::F32, &Device::Cpu).unwrap(),
            );
            t.insert(
                "flow_out.weight".into(),
                Tensor::zeros((dim, hidden_dim), DType::F32, &Device::Cpu).unwrap(),
            );
            t.insert(
                "flow_out.bias".into(),
                Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
            );
            VelocityFlowHead::new(
                VarBuilder::from_tensors(t, DType::F32, &Device::Cpu),
                dim,
                num_freqs,
                hidden_dim,
            )
            .unwrap()
        }),
        device: Device::Cpu,
        available: false, // Simulated CUDA hardware absent
    });

    let fallback_router =
        HardwareComputeRouter::new(vec![unavailable_candle, wgsl_backend.clone()]);

    let fallback_backend = fallback_router.select_backend().unwrap();
    assert_eq!(fallback_backend.backend_name(), "webgpu-wgsl-custom");

    let traj_wgsl = fallback_router.solve_trajectory(&x0, 5).unwrap();
    assert_eq!(traj_wgsl.len(), dim);
}

#[test]
fn test_step_parametric_control() {
    let weight_cache = WeightCache::default();
    let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum, weight_cache);
    let cond = [0.4f32; CONDITION_DIM];

    let ctrl = runner.step_parametric(&cond);

    assert_eq!(ctrl.band_gains.len(), 16);
    assert_eq!(ctrl.band_freq_drifts.len(), 16);

    for &gain in &ctrl.band_gains {
        assert!(
            gain.is_finite() && gain > 0.0,
            "Band gain must be finite and positive"
        );
    }

    for &drift in &ctrl.band_freq_drifts {
        assert!(
            drift.is_finite() && drift.abs() <= 0.15,
            "Frequency drift must be within expected bounds"
        );
    }

    assert!(ctrl.droplet_rate_mod >= 0.2 && ctrl.droplet_rate_mod <= 2.5);
    assert!(ctrl.droplet_energy_mod >= 0.4 && ctrl.droplet_energy_mod <= 2.0);
    assert!(ctrl.wind_gust_mod >= 0.2 && ctrl.wind_gust_mod <= 2.2);
    assert!(ctrl.wind_howl_mod >= 0.1 && ctrl.wind_howl_mod <= 2.5);

    let (w, x, y, z) = ctrl.spatial_vector;
    assert!(w.is_finite());
    assert!(x.is_finite());
    assert!(y.is_finite());
    assert!(z.is_finite());
}
