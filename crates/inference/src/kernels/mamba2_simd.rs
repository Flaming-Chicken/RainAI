//! Vectorized Mamba2 State-Space Recurrence Kernel: s_t = A * s_{t-1} + B * u_t.
//!
//! Re-exported from central `spodeian-ml-rs::sml-runtime::ssm`.

pub use spodeian_ml_utils::ssm::step_recurrence_f32;
