//! Integration tests verifying `sml-runtime`'s `SelfOptimizingMetaIntegrator`
//! continuous flow-matching solver inside `RainAI`'s `inference` crate.

use inference::kernels::adaptive_flow_solver::SelfOptimizingMetaIntegrator;

#[test]
fn test_inference_meta_integrator_flow_matching_trajectory() {
    // Integrate a continuous flow field dx/dt = -0.5 * (x - target) over t in [0, 2]
    let target = [0.25f32, -0.5f32, 0.75f32, -0.125f32];
    let x0 = [1.25f32, 0.5f32, -0.25f32, 0.875f32];

    let solver = SelfOptimizingMetaIntegrator::new(1e-4);
    let (xf, _diag, nfe) = solver.solve_interval(&x0, 0.0, 2.0, |x, _t| {
        x.iter()
            .zip(target.iter())
            .map(|(&xi, &ti)| -0.5 * (xi - ti))
            .collect()
    });

    assert!(nfe > 0);
    let decay = (-1.0f32).exp();
    for i in 0..4 {
        let expected = target[i] + (x0[i] - target[i]) * decay;
        assert!(
            (xf[i] - expected).abs() < 1e-2,
            "Flow matching latent mismatch at dim {i}: got={}, expected={expected}",
            xf[i]
        );
    }
}
