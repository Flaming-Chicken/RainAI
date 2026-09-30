//! Trait-based Hardware Compute Router for Continuous RK4 Flow Trajectories.
//!
//! Prioritizes Hugging Face Candle (CPU/GPU) hardware acceleration with seamless fallback
//! to custom WebGPU WGSL compute shaders.

use crate::kernels::rk4_flow_solver::Rk4FlowSolver;
use candle_core::Device;
use spodeian_ml_utils::VelocityFlowHead;
use std::sync::Arc;

/// Unified Hardware Compute Backend contract.
pub trait ComputeBackend: std::fmt::Debug + Send + Sync {
    /// Identifier for the compute backend (e.g. "candle-cpu", "candle-cuda", "webgpu-wgsl").
    fn backend_name(&self) -> &'static str;

    /// Health check assessing hardware capability and context validity.
    fn is_available(&self) -> bool {
        true
    }

    /// Evaluates continuous probability flow velocity $v_\theta(x, t) \in \mathbb{R}^d$.
    fn evaluate_flow_velocity(&self, x: &[f32], t: f32) -> Result<Vec<f32>, String>;
}

/// Candle-based neural flow velocity backend (CPU or CUDA).
pub struct CandleFlowBackend {
    pub name: &'static str,
    pub head: VelocityFlowHead,
    pub device: Device,
    pub available: bool,
}

impl std::fmt::Debug for CandleFlowBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleFlowBackend")
            .field("name", &self.name)
            .field("dim", &self.head.dim)
            .field("device", &self.device)
            .field("available", &self.available)
            .finish()
    }
}

impl CandleFlowBackend {
    pub fn new(name: &'static str, head: VelocityFlowHead, device: Device) -> Self {
        Self {
            name,
            head,
            device,
            available: true,
        }
    }
}

impl ComputeBackend for CandleFlowBackend {
    fn backend_name(&self) -> &'static str {
        self.name
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn evaluate_flow_velocity(&self, x: &[f32], t: f32) -> Result<Vec<f32>, String> {
        self.head
            .predict_velocity(x, t, &self.device)
            .map_err(|e| format!("Candle velocity evaluation error: {e}"))
    }
}

/// Custom WGSL compute shader simulation fallback backend.
#[derive(Debug)]
pub struct WgslComputeBackend {
    pub dim: usize,
    pub is_context_active: bool,
}

impl WgslComputeBackend {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            is_context_active: true,
        }
    }
}

impl ComputeBackend for WgslComputeBackend {
    fn backend_name(&self) -> &'static str {
        "webgpu-wgsl-custom"
    }

    fn is_available(&self) -> bool {
        self.is_context_active
    }

    fn evaluate_flow_velocity(&self, x: &[f32], t: f32) -> Result<Vec<f32>, String> {
        assert_eq!(x.len(), self.dim);
        // Custom hardware WGSL kernel emulation: time-harmonic velocity field
        let omega = 2.0 * std::f32::consts::PI * t;
        let mut v = Vec::with_capacity(self.dim);
        for (i, &xi) in x.iter().enumerate() {
            let phase = i as f32 * 0.5;
            v.push(-xi * (omega + phase).sin() + 0.1 * (omega * 2.0).cos());
        }
        Ok(v)
    }
}

use serde::{Deserialize, Serialize};

/// Supported Continuous Flow ODE Solver Algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FlowSolverAlgorithm {
    /// Fixed-step Runge-Kutta 4th Order.
    FixedRk4 { steps: usize },
    /// Adaptive Dormand-Prince 5(4) with dynamic error tolerance.
    AdaptiveRk45 { tol: f32, initial_h: f32 },
    /// Low-power mobile adaptive Bogacki-Shampine 2(3).
    AdaptiveRk23 { tol: f32, initial_h: f32 },
    /// High-efficiency Tsitouras 5(4) solver for smooth flow matching trajectories.
    AdaptiveTsit5 { tol: f32, initial_h: f32 },
    /// Heun's 2nd-order adaptive predictor-corrector (EDM/Karras formulation).
    AdaptiveHeun2 { tol: f32, initial_h: f32 },
    /// DPM-Solver++ 2nd-order fast multistep exponential integrator.
    DpmSolverPP { steps: usize },
    /// Neural ODE adaptive step modulator with trajectory curvature damping.
    LearnedCurvature { tol: f32, initial_h: f32 },
}

impl FlowSolverAlgorithm {
    pub fn name(&self) -> &'static str {
        match self {
            Self::FixedRk4 { .. } => "Fixed RK4 (Classic 4th-Order)",
            Self::AdaptiveRk45 { .. } => "Adaptive RK45 (Dormand-Prince)",
            Self::AdaptiveRk23 { .. } => "Adaptive RK23 (Bogacki-Shampine)",
            Self::AdaptiveTsit5 { .. } => "Adaptive Tsit5 (Tsitouras 5(4) FSAL)",
            Self::AdaptiveHeun2 { .. } => "Adaptive Heun2 (EDM/Karras 2nd-Order)",
            Self::DpmSolverPP { .. } => "DPM-Solver++ (Fast Multistep)",
            Self::LearnedCurvature { .. } => "Learned Curvature (Neural ODE Damped)",
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            Self::FixedRk4 { .. } => "Fixed RK4",
            Self::AdaptiveRk45 { .. } => "RK45",
            Self::AdaptiveRk23 { .. } => "RK23",
            Self::AdaptiveTsit5 { .. } => "Tsit5",
            Self::AdaptiveHeun2 { .. } => "Heun2",
            Self::DpmSolverPP { .. } => "DPM++",
            Self::LearnedCurvature { .. } => "Learned Curvature",
        }
    }
}

impl Default for FlowSolverAlgorithm {
    fn default() -> Self {
        Self::FixedRk4 { steps: 10 }
    }
}

/// Dynamic Hardware Compute Router.
///
/// Dispatches RK4 probability flow integration across a prioritized tier of compute backends.
#[derive(Debug)]
pub struct HardwareComputeRouter {
    backends: Vec<Arc<dyn ComputeBackend>>,
}

impl HardwareComputeRouter {
    pub fn new(backends: Vec<Arc<dyn ComputeBackend>>) -> Self {
        Self { backends }
    }

    /// Select highest-priority operational compute backend.
    pub fn select_backend(&self) -> Option<&Arc<dyn ComputeBackend>> {
        self.backends.iter().find(|b| b.is_available())
    }

    /// Solve full continuous probability flow trajectory using RK4 solver
    /// via the selected hardware backend.
    pub fn solve_trajectory(&self, x0: &[f32], num_steps: usize) -> Result<Vec<f32>, String> {
        let backend = self
            .select_backend()
            .ok_or_else(|| "No operational compute backend available for RK4 flow".to_string())?;

        let mut error_opt: Option<String> = None;
        let final_x = Rk4FlowSolver::solve_trajectory(x0, num_steps, |x_curr, t_curr| {
            match backend.evaluate_flow_velocity(x_curr, t_curr) {
                Ok(v) => v,
                Err(e) => {
                    error_opt = Some(e);
                    vec![0.0f32; x_curr.len()]
                }
            }
        });

        if let Some(err) = error_opt {
            Err(err)
        } else {
            Ok(final_x)
        }
    }

    /// Solve continuous probability flow trajectory using a specified solver algorithm.
    pub fn solve_trajectory_with_solver(
        &self,
        x0: &[f32],
        solver: FlowSolverAlgorithm,
    ) -> Result<Vec<f32>, String> {
        let backend = self
            .select_backend()
            .ok_or_else(|| "No operational compute backend available for flow solver".to_string())?;

        match solver {
            FlowSolverAlgorithm::FixedRk4 { steps } => self.solve_trajectory(x0, steps),
            FlowSolverAlgorithm::AdaptiveRk45 { tol, initial_h } => {
                let mut error_opt = None;
                let (final_x, _) = crate::kernels::adaptive_flow_solver::DormandPrince45::solve_adaptive_trajectory(
                    x0,
                    initial_h,
                    1e-4,
                    0.25,
                    tol,
                    500,
                    |x_curr, t_curr| match backend.evaluate_flow_velocity(x_curr, t_curr) {
                        Ok(v) => v,
                        Err(e) => {
                            error_opt = Some(e);
                            vec![0.0f32; x_curr.len()]
                        }
                    },
                )?;
                if let Some(err) = error_opt {
                    Err(err)
                } else {
                    Ok(final_x)
                }
            }
            FlowSolverAlgorithm::AdaptiveRk23 { tol, initial_h } => {
                let mut error_opt = None;
                let (final_x, _) = crate::kernels::adaptive_flow_solver::BogackiShampine23::solve_adaptive_trajectory(
                    x0,
                    initial_h,
                    1e-4,
                    0.25,
                    tol,
                    500,
                    |x_curr, t_curr| match backend.evaluate_flow_velocity(x_curr, t_curr) {
                        Ok(v) => v,
                        Err(e) => {
                            error_opt = Some(e);
                            vec![0.0f32; x_curr.len()]
                        }
                    },
                )?;
                if let Some(err) = error_opt {
                    Err(err)
                } else {
                    Ok(final_x)
                }
            }
            FlowSolverAlgorithm::AdaptiveTsit5 { tol, initial_h } => {
                let mut error_opt = None;
                let (final_x, _) = crate::kernels::adaptive_flow_solver::Tsitouras54::solve_adaptive_trajectory(
                    x0,
                    initial_h,
                    1e-4,
                    0.25,
                    tol,
                    500,
                    |x_curr, t_curr| match backend.evaluate_flow_velocity(x_curr, t_curr) {
                        Ok(v) => v,
                        Err(e) => {
                            error_opt = Some(e);
                            vec![0.0f32; x_curr.len()]
                        }
                    },
                )?;
                if let Some(err) = error_opt {
                    Err(err)
                } else {
                    Ok(final_x)
                }
            }
            FlowSolverAlgorithm::AdaptiveHeun2 { tol, initial_h } => {
                let mut error_opt = None;
                let (final_x, _) = crate::kernels::adaptive_flow_solver::HeunAdaptive2::solve_adaptive_trajectory(
                    x0,
                    initial_h,
                    1e-4,
                    0.25,
                    tol,
                    500,
                    |x_curr, t_curr| match backend.evaluate_flow_velocity(x_curr, t_curr) {
                        Ok(v) => v,
                        Err(e) => {
                            error_opt = Some(e);
                            vec![0.0f32; x_curr.len()]
                        }
                    },
                )?;
                if let Some(err) = error_opt {
                    Err(err)
                } else {
                    Ok(final_x)
                }
            }
            FlowSolverAlgorithm::DpmSolverPP { steps } => {
                let mut error_opt = None;
                let final_x = crate::kernels::adaptive_flow_solver::DpmSolverPP::solve_fast_trajectory(
                    x0,
                    steps,
                    |x_curr, t_curr| match backend.evaluate_flow_velocity(x_curr, t_curr) {
                        Ok(v) => v,
                        Err(e) => {
                            error_opt = Some(e);
                            vec![0.0f32; x_curr.len()]
                        }
                    },
                )?;
                if let Some(err) = error_opt {
                    Err(err)
                } else {
                    Ok(final_x)
                }
            }
            FlowSolverAlgorithm::LearnedCurvature { tol, initial_h } => {
                let controller = crate::kernels::adaptive_flow_solver::LearnedFlowController::new(initial_h, 0.005, 0.25);
                let mut current_x = x0.to_vec();
                let mut t = 0.0f32;
                let mut h = initial_h;
                let mut v_prev: Option<Vec<f32>> = None;
                let mut steps = 0;
                let mut error_opt = None;

                while t < 1.0 - 1e-6 {
                    if steps >= 500 {
                        return Err(format!("Learned controller exceeded 500 iterations at t = {t:.4}"));
                    }
                    if t + h > 1.0 {
                        h = 1.0 - t;
                    }
                    let v_curr = backend.evaluate_flow_velocity(&current_x, t)?;
                    h = controller.predict_step_size(&v_curr, v_prev.as_deref(), h);
                    let result = crate::kernels::adaptive_flow_solver::BogackiShampine23::step(&current_x, t, h, tol, |x_c, t_c| {
                        match backend.evaluate_flow_velocity(x_c, t_c) {
                            Ok(v) => v,
                            Err(e) => {
                                error_opt = Some(e);
                                vec![0.0f32; x0.len()]
                            }
                        }
                    });
                    if let Some(err) = error_opt {
                        return Err(err);
                    }
                    current_x = result.x_next;
                    v_prev = Some(v_curr);
                    t += h;
                    steps += 1;
                }
                Ok(current_x)
            }
        }
    }
}
