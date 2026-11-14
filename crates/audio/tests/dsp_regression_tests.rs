//! Golden DSP & real-time audio safety regression tests for `audio`.

use audio::buffer_guard::BufferGuard;
use audio::decoder::DecodeMode;
use audio::export::{render_wav_stream, write_wav_header};
use audio::procedural::ProceduralSynthesizer;
use shared::rain::RainState;
use std::io::Cursor;

#[test]
fn test_procedural_synth_finite_bounded_output() {
    let mut synth = ProceduralSynthesizer::new(48_000.0);
    let state = RainState {
        is_playing: true,
        master_volume: 1.0,
        ..Default::default()
    };
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;

    for _ in 0..4096 {
        let frame = synth.process_frame(&state);
        for ch in [frame.w, frame.y, frame.z, frame.x] {
            assert!(ch.is_finite(), "Non-finite FOA sample produced: {ch}");
            assert!(ch.abs() <= 4.0, "Sample exceeded safety bound: {ch}");
            max_abs = max_abs.max(ch.abs());
            sum_sq += (ch as f64) * (ch as f64);
        }
    }

    assert!(max_abs > 0.0, "Procedural synthesizer output should be non-silent");
    assert!(sum_sq > 0.0);
}

#[test]
fn test_buffer_guard_steady_state_zero_underruns() {
    let mut guard = BufferGuard::new(48_000.0, 120.0);
    for _ in 0..50 {
        let action = guard.update(6000, 512, 512.0 / 48_000.0);
        assert!(!action.is_starving);
        assert!(action.synthesis_blend.is_finite());
        assert_eq!(guard.underrun_count(), 0);
    }
}

#[test]
fn test_wav_header_and_stream_export_layout() {
    let mut header = Vec::new();
    write_wav_header(&mut header, 2, 48_000, 256).expect("write_wav_header should succeed");
    assert_eq!(header.len(), 44);
    assert_eq!(&header[0..4], b"RIFF");
    assert_eq!(&header[8..12], b"WAVE");
    assert_eq!(&header[12..16], b"fmt ");
    assert_eq!(&header[36..40], b"data");

    let state = RainState::default();
    let mut buf = Cursor::new(Vec::new());
    let mut last_progress = 0.0f32;
    let bytes = render_wav_stream(
        &state,
        0.05,
        48_000,
        DecodeMode::StereoSpeakers,
        &mut buf,
        |p| last_progress = p,
    )
    .expect("render_wav_stream should succeed");

    let raw = buf.into_inner();
    assert!(bytes > 44);
    assert_eq!(raw.len(), bytes as usize);
    assert_eq!(&raw[0..4], b"RIFF");
    assert!(last_progress > 0.0);
}
