//! Integration tests for AudioGraph, AudioNodes, and AudioSessionMediator in RainAI.

use audio::decoder::DecodeMode;
use audio::graph::{
    AmbisonicDecoderNode, AudioGraph, AudioNode, GainNode, NoiseMaskerNode,
    PhysicalSynthesizerNode, ProceduralSynthesizerNode,
};
use audio::hrtf_sofa::{AtomicBinauralDecoderMode, BinauralDecoderMode};
use audio::physical::PhysicalRainSynthesizer;
use audio::procedural::ProceduralSynthesizer;
use audio::session::{AudioCommand, AudioCommandQueue, AudioSessionMediator, AudioTelemetryQueue};
use shared::rain::RainState;
use std::sync::atomic::Ordering;

#[test]
fn test_binaural_decoder_mode_atomic() {
    let atomic_mode = AtomicBinauralDecoderMode::new(BinauralDecoderMode::ResonanceAudio32Tap);
    assert_eq!(atomic_mode.load(Ordering::Relaxed), BinauralDecoderMode::ResonanceAudio32Tap);

    atomic_mode.store(BinauralDecoderMode::SofaCustomIr, Ordering::Relaxed);
    assert_eq!(atomic_mode.load(Ordering::Relaxed), BinauralDecoderMode::SofaCustomIr);

    let old = atomic_mode.swap(BinauralDecoderMode::StereoDownmix, Ordering::Relaxed);
    assert_eq!(old, BinauralDecoderMode::SofaCustomIr);
    assert_eq!(atomic_mode.load(Ordering::Relaxed), BinauralDecoderMode::StereoDownmix);
}

#[test]
fn test_audio_graph_procedural_to_gain() {
    let mut graph = AudioGraph::new(128);
    let sample_rate = 44100;
    let proc_synth = ProceduralSynthesizer::new(sample_rate as f32);
    let mut state = RainState::default();
    state.is_playing = true;

    let proc_node = Box::new(ProceduralSynthesizerNode::new(proc_synth, state));
    let gain_node = Box::new(GainNode::new(0.5));

    let n0 = graph.add_node(proc_node);
    let n1 = graph.add_node(gain_node);

    // Connect procedural mono output to gain channel 0 and channel 1
    graph.connect(n0, 0, n1, 0).expect("Connect ch0 failed");
    graph.connect(n0, 0, n1, 1).expect("Connect ch1 failed");

    let mut out_l = vec![0.0f32; 128];
    let mut out_r = vec![0.0f32; 128];
    let mut stereo_ptrs: [&mut [f32]; 2] = [&mut out_l, &mut out_r];

    graph.process(&mut stereo_ptrs, 128);

    // Ensure audio frames were rendered
    let sum_l: f32 = out_l.iter().map(|s| s.abs()).sum();
    let sum_r: f32 = out_r.iter().map(|s| s.abs()).sum();
    assert!(sum_l > 0.0, "Left channel should have non-zero energy");
    assert!(sum_r > 0.0, "Right channel should have non-zero energy");
}

#[test]
fn test_audio_graph_physical_to_ambisonic_decoder() {
    let mut graph = AudioGraph::new(64);
    let sample_rate = 44100;
    let phys_synth = PhysicalRainSynthesizer::new(sample_rate as f32);
    let mut state = RainState::default();
    state.is_playing = true;

    let phys_node = Box::new(PhysicalSynthesizerNode::new(phys_synth, state));
    let decoder_node = Box::new(AmbisonicDecoderNode::new(DecodeMode::BinauralHeadphones));

    let n0 = graph.add_node(phys_node);
    let n1 = graph.add_node(decoder_node);

    // Connect 4 B-format channels (W, Y, Z, X) to decoder
    for ch in 0..4 {
        graph.connect(n0, ch, n1, ch).expect("B-format connect failed");
    }

    let mut out_l = vec![0.0f32; 64];
    let mut out_r = vec![0.0f32; 64];
    let mut stereo_ptrs: [&mut [f32]; 2] = [&mut out_l, &mut out_r];

    graph.process(&mut stereo_ptrs, 64);

    let sum_energy: f32 = out_l.iter().chain(out_r.iter()).map(|s| s.abs()).sum();
    assert!(sum_energy > 0.0, "Ambisonic physical output should produce signal");
}

#[test]
fn test_noise_masker_node() {
    let mut masker = NoiseMaskerNode::new(44100.0);
    assert_eq!(masker.name(), "NoiseMaskerNode");
    assert_eq!(masker.num_inputs(), 1);
    assert_eq!(masker.num_outputs(), 1);

    let input = vec![0.5f32; 32];
    let mut output = vec![0.0f32; 32];
    let in_ptrs: [&[f32]; 1] = [&input];
    let mut out_ptrs: [&mut [f32]; 1] = [&mut output];

    masker.process(&in_ptrs, &mut out_ptrs, 32);
    let out_sum: f32 = output.iter().map(|s| s.abs()).sum();
    assert!(out_sum > 0.0);
}

#[test]
fn test_session_mediator_and_queues() {
    let mut graph = AudioGraph::new(64);
    let sample_rate = 44100;
    let proc_synth = ProceduralSynthesizer::new(sample_rate as f32);
    let mut state = RainState::default();
    state.is_playing = true;

    let proc_node = Box::new(ProceduralSynthesizerNode::new(proc_synth, state));
    let gain_node = Box::new(GainNode::new(1.0));

    let n0 = graph.add_node(proc_node);
    let n1 = graph.add_node(gain_node);

    graph.connect(n0, 0, n1, 0).unwrap();
    graph.connect(n0, 0, n1, 1).unwrap();

    let (cmd_tx, cmd_rx) = AudioCommandQueue::new();
    let (telemetry_tx, mut telemetry_rx) = AudioTelemetryQueue::new();

    let mut mediator = AudioSessionMediator::new(graph, cmd_rx, telemetry_tx);

    let mut out_l = vec![0.0f32; 64];
    let mut out_r = vec![0.0f32; 64];

    // Render quantum 1
    mediator.render_quantum(&mut [&mut out_l, &mut out_r], 64);
    let snap1 = telemetry_rx.poll_latest().expect("Expected telemetry snapshot");
    assert!(snap1.frames_rendered >= 64);
    assert!(!snap1.is_muted);

    // Send mute command
    cmd_tx.send(AudioCommand::SetMute(true)).unwrap();

    // Render quantum 2 with mute
    mediator.render_quantum(&mut [&mut out_l, &mut out_r], 64);
    assert_eq!(out_l[0], 0.0);
    assert_eq!(out_r[0], 0.0);

    // Send volume adjustment
    cmd_tx.send(AudioCommand::SetMute(false)).unwrap();
    cmd_tx.send(AudioCommand::SetMasterVolume(0.5)).unwrap();

    // Render quantum 3
    mediator.render_quantum(&mut [&mut out_l, &mut out_r], 64);
    let snap3 = telemetry_rx.poll_latest().expect("Expected telemetry snapshot");
    assert!(snap3.frames_rendered >= 192);
}
