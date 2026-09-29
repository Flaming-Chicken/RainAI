//! Unit tests for Dormand-Prince (RK45), Bogacki-Shampine (RK23), and LearnedFlowController.
//!
//! Strictly isolated in tests/ with zero inline tests in production code and zero panics.

use inference::kernels::{AdaptiveStepResult, BogackiShampine23, DormandPrince45, LearnedFlowController};

#[test]
fn test_dormand_prince_single_step() {
    let x0 = vec![1.0f32, 2.0f32];
    let t = 0.0f32;
    let h = 0.1f32;
    let tol = 1e-4f32;

    // Linear decay ODE: dx/dt = -x
    let velocity = |x: &[f32], _t: f32| -> Vec<f32> {
        x.iter().map(|&v| -v).collect()
    };

    let result: AdaptiveStepResult = DormandPrince45::step(&x0, t, h, tol, velocity);

    assert_eq!(result.x_next.len(), 2);
    // Analytical solution: x(0.1) = x0 * e^(-0.1)
    let expected_0 = 1.0f32 * (-0.1f32).exp();
    let expected_1 = 2.0f32 * (-0.1f32).exp();

    assert!((result.x_next[0] - expected_0).abs() < 1e-5);
    assert!((result.x_next[1] - expected_1).abs() < 1e-5);
    assert!(result.accepted);
    assert!(result.error_norm < tol);
    assert!(result.recommended_h > 0.0);
}

#[test]
fn test_dormand_prince_adaptive_trajectory_convergence() {
    let x0 = vec![10.0f32];
    let initial_h = 0.05f32;
    let min_h = 0.001f32;
    let max_h = 0.20f32;
    let tol = 1e-4f32;

    // dx/dt = -0.5 * x => x(1.0) = 10.0 * e^(-0.5) = 6.0653066
    let velocity = |x: &[f32], _t: f32| -> Vec<f32> {
        vec![-0.5f32 * x[0]]
    };

    let solve_res = DormandPrince45::solve_adaptive_trajectory(
        &x0,
        initial_h,
        min_h,
        max_h,
        tol,
        500,
        velocity,
    );

    assert!(solve_res.is_ok());
    let (final_x, steps) = match solve_res {
        Ok(v) => v,
        Err(_) => (vec![0.0], 0),
    };

    let expected = 10.0f32 * (-0.5f32).exp();
    assert!((final_x[0] - expected).abs() < 1e-4);
    assert!(steps > 0);
}

#[test]
fn test_bogacki_shampine_low_power_solver() {
    let x0 = vec![5.0f32];
    let t = 0.0f32;
    let h = 0.05f32;
    let tol = 1e-3f32;

    // dx/dt = -x => x(0.05) = 5.0 * e^(-0.05)
    let velocity = |x: &[f32], _t: f32| -> Vec<f32> {
        vec![-x[0]]
    };

    let result = BogackiShampine23::step(&x0, t, h, tol, velocity);
    assert_eq!(result.x_next.len(), 1);

    let expected = 5.0f32 * (-0.05f32).exp();
    assert!((result.x_next[0] - expected).abs() < 1e-4);
    assert!(result.accepted);
}

#[test]
fn test_learned_flow_controller_curvature_damping() {
    let controller = LearnedFlowController::new(0.10, 0.01, 0.25);

    let v_straight = vec![0.5f32; 16];
    let h_straight = controller.predict_step_size(&v_straight, Some(&v_straight), 0.05);

    // Highly curved trajectory: sharp change in velocity vector
    let v_curved = vec![-0.8f32; 16];
    let h_curved = controller.predict_step_size(&v_curved, Some(&v_straight), 0.05);

    // Curvature must throttle step size down to preserve dynamic fidelity
    assert!(h_curved < h_straight);
    assert!(h_curved >= controller.min_h);
    assert!(h_straight <= controller.max_h);
}
