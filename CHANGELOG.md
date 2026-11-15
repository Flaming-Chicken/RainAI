# Changelog

All notable changes to `RainAI` are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Semantic Versioning.

## [2.0.0] - 2026-11-14

### Added
- Web-only Turnstile desktop flow: desktop **Contribute** launches `/#contribute` directly in the browser.
- Wired `sml-runtime` (`SelfOptimizingMeta`, `SelfOptimizingMetaConfig`, `create_self_optimizing_meta`), `spodeian-provenance`, and `spodeian-audio`.
- Added `SleepTimer` cosine fade-out state & `PresetBankState` (`A/B` toggle + favorites) in `crates/shared/src/sleep_timer.rs`.
- Added `crates/audio/tests/dsp_regression_tests.rs` and `crates/inference/tests/meta_flow_tests.rs` (`188` workspace tests).

### Changed
- Clarified SafeTensors + Candle (`sml-runtime`) native inference path and runtime `ffmpeg` detection for compressed exports.
- Removed dead local `TaskKind::DataContribution` queue and purged legacy studio text references.
