# RainAI Development Roadmap

---

## Phase 1: Core Neural DSP & Multi-Tier Runtime [COMPLETE]
- [x] Mamba-2 MoE trajectory generator in PyTorch and Candle.
- [x] WebGPU hardware-accelerated WGSL compute shader backend.
- [x] Multi-tier quantization (1.58-bit Ternary, INT8, FP16, FP32).
- [x] Ambisonic First-Order Ambisonics (FOA) real-time spatializer.

---

## Phase 2: Interoperability & Parity Verification [COMPLETE]
- [x] PyTorch canonical source of truth with automated Safetensors export.
- [x] Automated Cross-Framework Parity Test Suite (`parity_candle_pytorch.rs`).
- [x] Runge-Kutta 4th Order (RK4) continuous trajectory flow integrator.
- [x] Formal Data Provenance & Bibliography Register (`DATA_BIBLIOGRAPHY.md`).
- [x] Universal `cargo-ndk` Android mobile deployment pipeline (`scripts/build-android.ps1`, `scripts/build-android.sh`).

---

## Phase 3: Continuous-Time Edge Architecture & Version 1.0.0 [CURRENT]
- [x] Continuous-Time Neural ODE adaptive trajectory solvers (**Dormand-Prince RK45**, Bogacki-Shampine, Heun, DPM-Solver++, **Learned Solvers (Implicit RK, PEC)**).
- [x] Correlated stochastic brown noise driver ($-6\text{ dB/oct}$) to guarantee non-convergent, living latent trajectories.
- [x] Real-time neural-parametric frequency micro-drifts (`apply_drifts`) dynamically modulating 16-band biquad filterbanks.
- [x] Standardized 48 kHz WebAudio synthesis target with sub-32 kHz safe rate clamping and telemetry warnings.
- [x] Zero-Copy Explainable AI (XAI) binary attribution dictionary (`attributions.bin`) and live Soundscape Provenance HUD.
- [x] Native compiled Markdown [`PRIVACY_POLICY.md`](PRIVACY_POLICY.md) modal dialog.
- [x] Robust CI/CD workflow token prioritization (`PRIVATE_READ_ACCESS`).

---

## Phase 4: Continual Adaptation & Neural Waveshaping
- [ ] Audio-rate L0 neural waveshaping for nonlinear organic texture.
- [ ] On-the-job Test-Time Adaptation (TTA) of spatial reflection heads.
- [ ] Dynamic Engram Bank retrieval-augmented acoustic memory.
- [ ] Sharpness-Aware Minimization (SAM) training for cross-space generalization.
