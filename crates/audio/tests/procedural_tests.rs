use audio::decoder::FoaFrame;
use audio::{
    LEARNED_DRIFT, NOMINAL_BAND_FREQS, NOMINAL_BAND_Q, ProceduralSynthesizer,
    SubtractiveFilterbank16,
};
use shared::RainState;

#[test]
fn test_procedural_silence_when_stopped() {
    let mut synth = ProceduralSynthesizer::new(48000.0);
    let state = RainState {
        is_playing: false,
        ..Default::default()
    };

    let frame = synth.process_frame(&state);
    assert_eq!(frame, FoaFrame::default());
}

#[test]
fn test_procedural_audio_generation() {
    let mut synth = ProceduralSynthesizer::new(48000.0);
    let state = RainState {
        is_playing: true,
        master_volume: 1.0,
        ..Default::default()
    };

    let mut nonzero_count = 0;
    for _ in 0..100 {
        let frame = synth.process_frame(&state);
        if frame.w.abs() > 1e-4 {
            nonzero_count += 1;
        }
    }
    assert!(
        nonzero_count > 90,
        "Procedural synthesizer should generate continuous audio"
    );
}

#[test]
fn test_subtractive_filterbank_16_bands() {
    let filterbank = SubtractiveFilterbank16::new(48000.0);
    assert_eq!(filterbank.filters.len(), 16);
    assert_eq!(NOMINAL_BAND_FREQS.len(), 16);
    assert_eq!(NOMINAL_BAND_Q.len(), 16);
    assert_eq!(LEARNED_DRIFT.len(), 16);

    const { assert!(LEARNED_DRIFT[0] > 0.0) };
    const { assert!(LEARNED_DRIFT[3] < 0.0) };
}

#[test]
fn test_procedural_parallel_buffer_generation() {
    let mut synth = ProceduralSynthesizer::new(48000.0);
    let state = RainState {
        is_playing: true,
        master_volume: 1.0,
        ..Default::default()
    };

    let mut buffer = vec![FoaFrame::default(); 512];
    synth.process_buffer_parallel(&state, &mut buffer, 128);

    let active_frames = buffer.iter().filter(|f| f.w.abs() > 1e-4).count();
    assert!(
        active_frames > 450,
        "Parallel procedural synthesis should generate non-zero audio across chunks: got {active_frames}/512"
    );
}

#[test]
fn test_neural_parametric_modulation_procedural() {
    use inference::NeuralParametricControl;

    let mut synth = ProceduralSynthesizer::new(48000.0);
    let state = RainState {
        is_playing: true,
        master_volume: 1.0,
        ..Default::default()
    };

    let mut ctrl = NeuralParametricControl::default();
    ctrl.band_gains[0] = 3.0; // Boost tin band
    ctrl.droplet_rate_mod = 2.0;
    ctrl.spatial_vector = (1.5, 0.5, -0.5, 0.8);

    let mut modulated_frames = 0;
    for _ in 0..128 {
        let frame = synth.process_frame_modulated(&state, Some(&ctrl));
        if frame.w.abs() > 1e-4 {
            modulated_frames += 1;
        }
    }
    assert!(
        modulated_frames > 115,
        "Modulated procedural synthesis should generate audio frames"
    );
}

#[test]
fn test_neural_parametric_modulation_physical() {
    use audio::PhysicalRainSynthesizer;
    use inference::NeuralParametricControl;

    let mut physical = PhysicalRainSynthesizer::new(48000.0);
    let state = RainState {
        is_playing: true,
        master_volume: 1.0,
        ..Default::default()
    };

    let ctrl = NeuralParametricControl {
        droplet_rate_mod: 1.5,
        droplet_energy_mod: 1.2,
        spatial_vector: (1.2, 0.4, 0.2, -0.3),
        ..Default::default()
    };

    let mut energy_sum = 0.0f32;
    for _ in 0..512 {
        let frame = physical.process_frame_modulated(&state, Some(&ctrl));
        energy_sum += frame.w.abs();
    }
    assert!(
        energy_sum > 0.01,
        "Modulated physical synthesizer should generate active droplet acoustics"
    );
}
