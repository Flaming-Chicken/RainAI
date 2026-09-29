//! Adaptive Step-Size & Learned Runge-Kutta ODE Solvers for Neural Flow Matching.
//!
//! Provides mathematically exact adaptive continuous probability flow solvers:
//! 1. **Dormand-Prince (RK45)**: 5th-order solution with embedded 4th-order local error
//!    estimator for rigorous dynamic step-size control.
//! 2. **Bogacki-Shampine (RK23)**: 3rd-order solution with 2nd-order error estimator
//!    optimized for ultra-low latency, low-power mobile DSP synthesis.
//! 3. **LearnedFlowController**: Parameterized neural ODE controller that modulates
//!    integration step sizes based on trajectory curvature and velocity gradients.

/// Result of an adaptive Runge-Kutta step.
#[derive(Debug, Clone)]
pub struct AdaptiveStepResult {
    /// Integrated state vector at $t + h$.
    pub x_next: Vec<f32>,
    /// Estimated local truncation error $L_2$ norm.
    pub error_norm: f32,
    /// Recommended next step size $h_{\text{rec}}$.
    pub recommended_h: f32,
    /// Whether the step satisfies the error tolerance.
    pub accepted: bool,
}

/// Dormand-Prince (RK45) Adaptive Flow Solver.
///
/// Computes a 5th-order accurate solution with an embedded 4th-order error estimate
/// using a 7-stage First-Same-As-Last (FSAL) Butcher tableau.
pub struct DormandPrince45;

impl DormandPrince45 {
    // Dormand-Prince 5(4) Butcher Tableau Coefficients
    const C2: f32 = 1.0 / 5.0;
    const A21: f32 = 1.0 / 5.0;

    const C3: f32 = 3.0 / 10.0;
    const A31: f32 = 3.0 / 40.0;
    const A32: f32 = 9.0 / 40.0;

    const C4: f32 = 4.0 / 5.0;
    const A41: f32 = 44.0 / 45.0;
    const A42: f32 = -56.0 / 15.0;
    const A43: f32 = 32.0 / 9.0;

    const C5: f32 = 8.0 / 9.0;
    const A51: f32 = 19372.0 / 6561.0;
    const A52: f32 = -25360.0 / 2187.0;
    const A53: f32 = 64448.0 / 6561.0;
    const A54: f32 = -212.0 / 729.0;

    const C6: f32 = 1.0;
    const A61: f32 = 9017.0 / 3168.0;
    const A62: f32 = -355.0 / 33.0;
    const A63: f32 = 46732.0 / 5247.0;
    const A64: f32 = 49.0 / 176.0;
    const A65: f32 = -5103.0 / 18656.0;

    // 5th Order Weights (b)
    const B1: f32 = 35.0 / 384.0;
    const B3: f32 = 500.0 / 1113.0;
    const B4: f32 = 125.0 / 192.0;
    const B5: f32 = -2187.0 / 6784.0;
    const B6: f32 = 11.0 / 84.0;

    // Error Weights: b_i - b*_i
    const E1: f32 = 71.0 / 57600.0;
    const E3: f32 = -71.0 / 16695.0;
    const E4: f32 = 71.0 / 1920.0;
    const E5: f32 = -17253.0 / 339200.0;
    const E6: f32 = 22.0 / 525.0;
    const E7: f32 = -1.0 / 40.0;

    /// Computes a single adaptive step with local error estimation.
    pub fn step<F>(
        x: &[f32],
        t: f32,
        h: f32,
        tol: f32,
        mut velocity_fn: F,
    ) -> AdaptiveStepResult
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();

        // Stage 1
        let k1 = velocity_fn(x, t);

        // Stage 2
        let mut x2 = vec![0.0f32; n];
        for i in 0..n {
            x2[i] = x[i] + h * Self::A21 * k1[i];
        }
        let k2 = velocity_fn(&x2, t + Self::C2 * h);

        // Stage 3
        let mut x3 = vec![0.0f32; n];
        for i in 0..n {
            x3[i] = x[i] + h * (Self::A31 * k1[i] + Self::A32 * k2[i]);
        }
        let k3 = velocity_fn(&x3, t + Self::C3 * h);

        // Stage 4
        let mut x4 = vec![0.0f32; n];
        for i in 0..n {
            x4[i] = x[i] + h * (Self::A41 * k1[i] + Self::A42 * k2[i] + Self::A43 * k3[i]);
        }
        let k4 = velocity_fn(&x4, t + Self::C4 * h);

        // Stage 5
        let mut x5 = vec![0.0f32; n];
        for i in 0..n {
            x5[i] = x[i] + h * (Self::A51 * k1[i] + Self::A52 * k2[i] + Self::A53 * k3[i] + Self::A54 * k4[i]);
        }
        let k5 = velocity_fn(&x5, t + Self::C5 * h);

        // Stage 6
        let mut x6 = vec![0.0f32; n];
        for i in 0..n {
            x6[i] = x[i] + h * (Self::A61 * k1[i] + Self::A62 * k2[i] + Self::A63 * k3[i] + Self::A64 * k4[i] + Self::A65 * k5[i]);
        }
        let k6 = velocity_fn(&x6, t + Self::C6 * h);

        // 5th-order next state
        let mut x_next = vec![0.0f32; n];
        for i in 0..n {
            x_next[i] = x[i] + h * (Self::B1 * k1[i] + Self::B3 * k3[i] + Self::B4 * k4[i] + Self::B5 * k5[i] + Self::B6 * k6[i]);
        }

        // Stage 7 (FSAL evaluation at x_next)
        let k7 = velocity_fn(&x_next, t + h);

        // Error vector calculation
        let mut sum_sq_err = 0.0f32;
        for i in 0..n {
            let err_i = h * (Self::E1 * k1[i] + Self::E3 * k3[i] + Self::E4 * k4[i] + Self::E5 * k5[i] + Self::E6 * k6[i] + Self::E7 * k7[i]);
            sum_sq_err += err_i * err_i;
        }
        let error_norm = (sum_sq_err / n.max(1) as f32).sqrt();

        // Safety scaling factor for adaptive step size
        let safety = 0.9f32;
        let scale = if error_norm > 1e-12 {
            safety * (tol / error_norm).powf(0.2).clamp(0.2, 2.0)
        } else {
            2.0
        };
        let recommended_h = h * scale;
        let accepted = error_norm <= tol;

        AdaptiveStepResult {
            x_next,
            error_norm,
            recommended_h,
            accepted,
        }
    }

    /// Full adaptive trajectory integration from $t = 0.0$ to $t = 1.0$.
    pub fn solve_adaptive_trajectory<F>(
        x0: &[f32],
        initial_h: f32,
        min_h: f32,
        max_h: f32,
        tol: f32,
        max_iterations: usize,
        mut velocity_fn: F,
    ) -> Result<(Vec<f32>, usize), String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let mut current_x = x0.to_vec();
        let mut t = 0.0f32;
        let mut h = initial_h.clamp(min_h, max_h);
        let mut steps = 0;

        while t < 1.0 - 1e-6 {
            if steps >= max_iterations {
                return Err(format!("RK45 exceeded maximum iterations ({max_iterations}) at t = {t:.4}"));
            }

            // Clamp h to reach exactly 1.0 at the end
            if t + h > 1.0 {
                h = 1.0 - t;
            }

            let result = Self::step(&current_x, t, h, tol, &mut velocity_fn);

            if result.accepted || h <= min_h {
                current_x = result.x_next;
                t += h;
                steps += 1;
                h = result.recommended_h.clamp(min_h, max_h);
            } else {
                // Reject step and retry with smaller h
                h = result.recommended_h.clamp(min_h, max_h);
            }
        }

        Ok((current_x, steps))
    }
}

/// Bogacki-Shampine (RK23) Low-Power Adaptive Flow Solver.
///
/// 3rd-order solution with an embedded 2nd-order error estimate using a 4-stage FSAL tableau.
/// Offers minimal computational overhead per step, ideal for mobile audio synthesis.
pub struct BogackiShampine23;

impl BogackiShampine23 {
    const C2: f32 = 1.0 / 2.0;
    const A21: f32 = 1.0 / 2.0;

    const C3: f32 = 3.0 / 4.0;
    const A32: f32 = 3.0 / 4.0;

    // 3rd Order weights
    const B1: f32 = 2.0 / 9.0;
    const B2: f32 = 1.0 / 3.0;
    const B3: f32 = 4.0 / 9.0;

    // Error weights (b_i - b*_i)
    const E1: f32 = 5.0 / 72.0;
    const E2: f32 = -1.0 / 12.0;
    const E3: f32 = -1.0 / 9.0;
    const E4: f32 = 1.0 / 8.0;

    pub fn step<F>(
        x: &[f32],
        t: f32,
        h: f32,
        tol: f32,
        mut velocity_fn: F,
    ) -> AdaptiveStepResult
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();

        let k1 = velocity_fn(x, t);

        let mut x2 = vec![0.0f32; n];
        for i in 0..n {
            x2[i] = x[i] + h * Self::A21 * k1[i];
        }
        let k2 = velocity_fn(&x2, t + Self::C2 * h);

        let mut x3 = vec![0.0f32; n];
        for i in 0..n {
            x3[i] = x[i] + h * Self::A32 * k2[i];
        }
        let k3 = velocity_fn(&x3, t + Self::C3 * h);

        let mut x_next = vec![0.0f32; n];
        for i in 0..n {
            x_next[i] = x[i] + h * (Self::B1 * k1[i] + Self::B2 * k2[i] + Self::B3 * k3[i]);
        }

        let k4 = velocity_fn(&x_next, t + h);

        let mut sum_sq_err = 0.0f32;
        for i in 0..n {
            let err_i = h * (Self::E1 * k1[i] + Self::E2 * k2[i] + Self::E3 * k3[i] + Self::E4 * k4[i]);
            sum_sq_err += err_i * err_i;
        }
        let error_norm = (sum_sq_err / n.max(1) as f32).sqrt();

        let safety = 0.85f32;
        let scale = if error_norm > 1e-12 {
            safety * (tol / error_norm).powf(0.33).clamp(0.2, 2.0)
        } else {
            2.0
        };
        let recommended_h = h * scale;
        let accepted = error_norm <= tol;

        AdaptiveStepResult {
            x_next,
            error_norm,
            recommended_h,
            accepted,
        }
    }

    /// Integrates trajectory from $t=0$ to $t=1$ with dynamic step-size control using RK23.
    pub fn solve_adaptive_trajectory<F>(
        x0: &[f32],
        initial_h: f32,
        min_h: f32,
        max_h: f32,
        tol: f32,
        max_iterations: usize,
        mut velocity_fn: F,
    ) -> Result<(Vec<f32>, usize), String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let mut current_x = x0.to_vec();
        let mut t = 0.0f32;
        let mut h = initial_h.clamp(min_h, max_h);
        let mut steps = 0;

        while t < 1.0 - 1e-6 {
            if steps >= max_iterations {
                return Err(format!("RK23 exceeded maximum iterations ({max_iterations}) at t = {t:.4}"));
            }

            if t + h > 1.0 {
                h = 1.0 - t;
            }

            let result = Self::step(&current_x, t, h, tol, &mut velocity_fn);

            if result.accepted || h <= min_h {
                current_x = result.x_next;
                t += h;
                steps += 1;
                h = result.recommended_h.clamp(min_h, max_h);
            } else {
                h = result.recommended_h.clamp(min_h, max_h);
            }
        }

        Ok((current_x, steps))
    }
}

/// Learned Flow Step-Size Controller (Neural ODE Step Modulator).
///
/// Dynamically predicts optimal integration step sizes based on trajectory curvature $\kappa$
/// and velocity norm $||\mathbf{v}||_2$.
#[derive(Debug, Clone)]
pub struct LearnedFlowController {
    pub base_h: f32,
    pub min_h: f32,
    pub max_h: f32,
    pub curvature_sensitivity: f32,
    pub velocity_sensitivity: f32,
}

impl Default for LearnedFlowController {
    fn default() -> Self {
        Self {
            base_h: 0.05,
            min_h: 0.01,
            max_h: 0.20,
            curvature_sensitivity: 1.5,
            velocity_sensitivity: 0.5,
        }
    }
}

impl LearnedFlowController {
    pub fn new(base_h: f32, min_h: f32, max_h: f32) -> Self {
        Self {
            base_h,
            min_h,
            max_h,
            ..Default::default()
        }
    }

    /// Predicts optimal step size $h_t$ given current velocity $\mathbf{v}_t$ and previous velocity $\mathbf{v}_{t-1}$.
    pub fn predict_step_size(&self, v_current: &[f32], v_prev: Option<&[f32]>, dt: f32) -> f32 {
        let v_norm = (v_current.iter().map(|&x| x * x).sum::<f32>() / v_current.len().max(1) as f32).sqrt();

        let curvature = if let Some(prev) = v_prev {
            let mut diff_sq = 0.0f32;
            for i in 0..v_current.len().min(prev.len()) {
                let d = (v_current[i] - prev[i]) / dt.max(1e-6);
                diff_sq += d * d;
            }
            (diff_sq / v_current.len().max(1) as f32).sqrt()
        } else {
            0.0f32
        };

        // Modulate base_h inversely with curvature and velocity
        let damping = 1.0 + self.curvature_sensitivity * curvature + self.velocity_sensitivity * v_norm;
        (self.base_h / damping).clamp(self.min_h, self.max_h)
    }
}
