use audio::buffer_guard::BufferGuard;

#[test]
fn test_buffer_guard_initial_state() {
    let guard = BufferGuard::new(48000.0, 45.0);
    assert_eq!(guard.target_buffer_ms, 45.0);
    assert_eq!(guard.sample_rate, 48000.0);
    assert_eq!(guard.synthesis_blend(), 0.0);
    assert_eq!(guard.compute_deficit(), 0.0);
    assert_eq!(guard.underrun_count(), 0);
}

#[test]
fn test_buffer_guard_healthy_playback() {
    let mut guard = BufferGuard::new(48000.0, 45.0);
    // 45ms at 48kHz = 2160 frames
    let available_frames = 2160;
    let frames_needed = 256;
    let dt = 256.0 / 48000.0;

    let action = guard.update(available_frames, frames_needed, dt);
    assert!(!action.is_starving);
    assert_eq!(action.frames_to_generate, 0);
    assert!(action.synthesis_blend < 0.01);
    assert_eq!(action.compute_deficit, 0.0);
    assert!((action.buffer_health_ms - 45.0).abs() < 0.1);
}

#[test]
fn test_buffer_guard_underrun_emergency_crossfade() {
    let mut guard = BufferGuard::new(48000.0, 45.0);
    // Almost starved buffer (only 32 frames available, but 256 needed)
    let available_frames = 32;
    let frames_needed = 256;
    let dt = 0.05; // 50ms timestep

    let action = guard.update(available_frames, frames_needed, dt);
    assert!(action.is_starving);
    assert!(action.frames_to_generate > 0);
    assert!(action.synthesis_blend > 0.3); // ramping towards 1.0
    assert!(action.compute_deficit > 0.8);
    assert_eq!(guard.underrun_count(), 1);

    // After another step, blend should continue moving toward 1.0 (emergency procedural fallback)
    let action2 = guard.update(available_frames, frames_needed, dt);
    assert!(action2.synthesis_blend > action.synthesis_blend);
}

#[test]
fn test_buffer_guard_recovery() {
    let mut guard = BufferGuard::new(48000.0, 45.0);
    // Start starved
    guard.update(0, 256, 0.1);
    assert!(guard.synthesis_blend() > 0.5);

    // Buffer replenished to target headroom (2160 frames)
    for _ in 0..10 {
        guard.update(2160, 256, 0.05);
    }
    // Should smoothly recover back down to 0.0
    assert!(guard.synthesis_blend() < 0.05);
    assert_eq!(guard.compute_deficit(), 0.0);
}
