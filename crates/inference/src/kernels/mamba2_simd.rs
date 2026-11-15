//! Vectorized Mamba2 State-Space Recurrence Kernel: s_t = A * s_{t-1} + B * u_t.
//!
//! Re-exported from central `spodeian-core::sml-runtime::ssm`.

pub use sml_runtime::ssm::step_recurrence_f32;
