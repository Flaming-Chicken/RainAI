//! Integration tests for Edge Spatial Audio DSP, Image-Source Room Acoustic Modeling,
//! Higher-Order Ambisonics (HOA 3rd-Order) 7.1.4 Decoders, and Adaptive Equalization.

use audio::adaptation::AcousticSceneAdapter;
use audio::hoa::{Hoa3Decoder, Hoa3Encoder};
use audio::ray_tracing::{
    AcousticMaterial, ImageSourceModel, RealTimeEarlyReflectionConvolver, RoomDimensions, Vec3,
};

#[test]
fn test_room_acoustics_and_sabine_t60() {
    let room = RoomDimensions {
        width: 5.0,
        length: 7.0,
        height: 3.0,
    };
    assert!((room.volume() - 105.0).abs() < 1e-4);
    assert!((room.total_surface_area() - 142.0).abs() < 1e-4);

    let model = ImageSourceModel::new(
        room,
        [
            AcousticMaterial::Concrete,
            AcousticMaterial::Concrete,
            AcousticMaterial::Concrete,
            AcousticMaterial::Concrete,
            AcousticMaterial::Concrete,
            AcousticMaterial::Concrete,
        ],
        3,
    );

    let t60 = model.sabine_t60();
    assert!(
        t60 > 2.0,
        "Concrete room must have high reverberation time T60, got {t60}"
    );

    // Damped room with heavy fabric
    let damped_model = ImageSourceModel::new(
        room,
        [
            AcousticMaterial::Fabric,
            AcousticMaterial::Fabric,
            AcousticMaterial::Fabric,
            AcousticMaterial::Fabric,
            AcousticMaterial::Fabric,
            AcousticMaterial::Fabric,
        ],
        3,
    );
    let damped_t60 = damped_model.sabine_t60();
    assert!(
        damped_t60 < t60,
        "Fabric room must have significantly shorter T60 than concrete room"
    );
}

#[test]
fn test_image_source_early_reflections_calculation() {
    let room = RoomDimensions {
        width: 4.0,
        length: 5.0,
        height: 2.5,
    };
    let model = ImageSourceModel::new(
        room,
        [
            AcousticMaterial::PineTimber,
            AcousticMaterial::Plasterboard,
            AcousticMaterial::Plasterboard,
            AcousticMaterial::Plasterboard,
            AcousticMaterial::Glass,
            AcousticMaterial::Brick,
        ],
        2,
    );

    let source = Vec3::new(1.0, 1.0, 1.2);
    let listener = Vec3::new(2.5, 3.0, 1.2);

    let reflections = model.calculate_early_reflections(&source, &listener);
    assert!(
        !reflections.is_empty(),
        "Must generate early reflection arrival events"
    );

    // Verify reflections are sorted chronologically by delay
    for window in reflections.windows(2) {
        assert!(
            window[0].delay_seconds <= window[1].delay_seconds,
            "Reflections must be ordered by arrival time"
        );
    }

    // Verify orders are within max order 2
    for r in &reflections {
        assert!(r.order <= 2);
        assert!(r.gain > 0.0 && r.gain <= 1.0);
    }
}

#[test]
fn test_realtime_early_reflection_convolver() {
    let sample_rate = 48000.0;
    let mut convolver = RealTimeEarlyReflectionConvolver::new(sample_rate, 0.2);

    let model = ImageSourceModel::default();
    let source = Vec3::new(1.0, 1.0, 1.0);
    let listener = Vec3::new(2.0, 2.0, 1.0);
    let reflections = model.calculate_early_reflections(&source, &listener);

    convolver.update_reflections(&reflections);

    // Process unit impulse input [1.0, 0.0, 0.0, ...]
    let mut buffer = vec![0.0f32; 1024];
    buffer[0] = 1.0;

    convolver.process_buffer(&mut buffer);

    // Direct sound must be preserved
    assert_eq!(buffer[0], 1.0);

    // Energy must be distributed into delay taps
    let non_zero_count = buffer.iter().filter(|&&s| s.abs() > 1e-6).count();
    assert!(
        non_zero_count > 1,
        "Convolver must introduce early reflection echoes"
    );
}

#[test]
fn test_hoa3_encoder_spherical_harmonics() {
    // 1. Center Front Source: azimuth = 0, elevation = 0
    let front = Hoa3Encoder::encode_point_source(0.0, 0.0, 1.0);
    assert!((front.channels[0] - 1.0).abs() < 1e-4, "W channel must be 1.0");
    assert!((front.channels[3] - 1.0).abs() < 1e-4, "X channel must be 1.0 (Front)");
    assert!(front.channels[1].abs() < 1e-4, "Y channel must be 0.0 (No left-right)");
    assert!(front.channels[2].abs() < 1e-4, "Z channel must be 0.0 (No elevation)");

    // 2. Pure Left Source: azimuth = pi/2, elevation = 0
    let left = Hoa3Encoder::encode_point_source(std::f32::consts::FRAC_PI_2, 0.0, 1.0);
    assert!((left.channels[0] - 1.0).abs() < 1e-4);
    assert!((left.channels[1] - 1.0).abs() < 1e-4, "Y channel must be 1.0 (Left)");
    assert!(left.channels[3].abs() < 1e-4, "X channel must be 0.0");

    // 3. Directly Overhead Source: elevation = pi/2
    let overhead = Hoa3Encoder::encode_point_source(0.0, std::f32::consts::FRAC_PI_2, 1.0);
    assert!((overhead.channels[0] - 1.0).abs() < 1e-4);
    assert!((overhead.channels[2] - 1.0).abs() < 1e-4, "Z channel must be 1.0 (Zenith)");
    assert!(overhead.channels[1].abs() < 1e-4);
    assert!(overhead.channels[3].abs() < 1e-4);
}

#[test]
fn test_hoa3_soundfield_rotation() {
    let front = Hoa3Encoder::encode_point_source(0.0, 0.0, 1.0);

    // Rotate 90 degrees counter-clockwise (yaw = pi/2): front should become right
    let rotated = front.rotate(std::f32::consts::FRAC_PI_2, 0.0, 0.0);

    // W is invariant to rotation
    assert_eq!(rotated.channels[0], front.channels[0]);

    // X was 1.0, after 90 deg yaw rotation X becomes 0 and Y becomes -1.0 (or vice versa depending on handedness)
    assert!(rotated.channels[3].abs() < 1e-3);
    assert!((rotated.channels[1].abs() - 1.0).abs() < 1e-3);
}

#[test]
fn test_hoa3_itu_714_decoder() {
    // 1. Center sound source
    let front = Hoa3Encoder::encode_point_source(0.0, 0.0, 1.0);
    let out_714_front = Hoa3Decoder::decode_714(&front);

    assert!(
        out_714_front.center > out_714_front.left_rear,
        "Front source must produce stronger center speaker output than rear"
    );

    // 2. Overhead sound source
    let overhead = Hoa3Encoder::encode_point_source(0.0, std::f32::consts::FRAC_PI_2, 1.0);
    let out_714_height = Hoa3Decoder::decode_714(&overhead);

    let height_sum = out_714_height.top_front_left
        + out_714_height.top_front_right
        + out_714_height.top_back_left
        + out_714_height.top_back_right;

    assert!(
        height_sum > 0.5,
        "Overhead source must drive overhead speakers prominently"
    );
}

#[test]
fn test_acoustic_scene_adaptation_and_mode_inversion() {
    let sample_rate = 48000.0;
    let mut adapter = AcousticSceneAdapter::new(sample_rate);

    // Create synthetic microphone buffer with resonant 63 Hz room mode boom
    let mut mic_buffer = vec![0.0f32; 1024];
    for (i, s) in mic_buffer.iter_mut().enumerate() {
        let t = (i as f32) / sample_rate;
        // Strong 63Hz resonance + small background noise
        *s = 0.8 * (2.0 * std::f32::consts::PI * 63.0 * t).sin() + 0.05 * (t * 1000.0).sin();
    }

    adapter.adapt_from_microphone(&mic_buffer);

    // Adapter should detect the 63Hz resonance mode
    let has_63_mode = adapter.detected_modes.iter().any(|m| m.center_hz == 63.0);
    assert!(has_63_mode, "Adapter must identify 63Hz room resonance mode");

    // Process a resonant 63Hz input through the adapter's notch filter
    let mut audio_signal = vec![0.0f32; 512];
    for (i, s) in audio_signal.iter_mut().enumerate() {
        let t = (i as f32) / sample_rate;
        *s = (2.0 * std::f32::consts::PI * 63.0 * t).sin();
    }

    let input_energy: f32 = audio_signal.iter().map(|x| x * x).sum();
    adapter.process_buffer(&mut audio_signal);
    let output_energy: f32 = audio_signal.iter().map(|x| x * x).sum();

    assert!(
        output_energy < input_energy,
        "Notch filter must attenuate resonant 63Hz room standing wave"
    );
}
