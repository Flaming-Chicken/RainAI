use audio::derivation::{ListeningSetup, SoundLayers, derive_parameters};
use inference::compute_router::FlowSolverAlgorithm;
use shared::rain::{GovernorOptimizationProfile, QualityTier};

#[test]
fn test_default_derivation_desktop_balanced() {
    let layers = SoundLayers::default();
    assert!(layers.raindrops);
    assert!(layers.rain_wash);
    assert!(!layers.ai_texture);

    let derived = derive_parameters(
        GovernorOptimizationProfile::BalancedAdaptive,
        ListeningSetup::DesktopSpeakers,
        layers,
    );

    assert_eq!(derived.target_buffer_ms, 45.0);
    assert_eq!(derived.waveshaper_tier, QualityTier::AdaptiveMinimum);
    assert_eq!(derived.droplet_voice_cap, 512);
    assert!(derived.suppress_procedural_droplets);
    assert!(matches!(
        derived.flow_solver,
        FlowSolverAlgorithm::AdaptiveRk45 { .. }
    ));
}

#[test]
fn test_battery_saver_derivation() {
    let layers = SoundLayers::default();
    let derived = derive_parameters(
        GovernorOptimizationProfile::EcoBatterySaver,
        ListeningSetup::DesktopSpeakers,
        layers,
    );

    assert_eq!(derived.target_buffer_ms, 60.0);
    assert_eq!(derived.waveshaper_tier, QualityTier::Ternary158);
    assert_eq!(derived.droplet_voice_cap, 128);
    assert!(matches!(
        derived.flow_solver,
        FlowSolverAlgorithm::AdaptiveRk23 { .. }
    ));
    assert_eq!(derived.solver_tolerance, 1e-2);
}

#[test]
fn test_wireless_bluetooth_jitter_reserve() {
    let layers = SoundLayers::default();
    let derived = derive_parameters(
        GovernorOptimizationProfile::BalancedAdaptive,
        ListeningSetup::WirelessBluetooth,
        layers,
    );

    // 45ms base + 100ms wireless reserve = 145ms
    assert_eq!(derived.target_buffer_ms, 145.0);
}

#[test]
fn test_procedural_partitioning_when_raindrops_disabled() {
    let layers = SoundLayers {
        raindrops: false,
        rain_wash: true,
        ai_texture: false,
    };
    let derived = derive_parameters(
        GovernorOptimizationProfile::StudioMaster,
        ListeningSetup::BinauralHeadphones,
        layers,
    );

    assert_eq!(derived.target_buffer_ms, 120.0);
    assert_eq!(derived.waveshaper_tier, QualityTier::StudioFp32);
    assert_eq!(derived.droplet_voice_cap, 2048);
    // When raindrops are OFF, procedural droplet bands must NOT be suppressed
    assert!(!derived.suppress_procedural_droplets);
    assert!(matches!(
        derived.flow_solver,
        FlowSolverAlgorithm::AdaptiveTsit5 { .. }
    ));
}

#[test]
fn test_sound_layers_guardrail() {
    let mut layers = SoundLayers {
        raindrops: false,
        rain_wash: false,
        ai_texture: false,
    };
    layers.ensure_valid();
    // Guardrail prevents all layers from being off
    assert!(layers.rain_wash);
}
