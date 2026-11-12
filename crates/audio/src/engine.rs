//! Cross-platform real-time audio playback engine.
//!
//! Provides native low-latency output via `cpal` on desktop, and `web-sys` WebAudio
//! integration on WebAssembly, with lock-free state synchronization from the UI thread.

use crate::buffer_guard::BufferGuard;
use crate::decoder::{AmbisonicDecoder, DecodeMode};
use crate::history::AcousticHistoryBuffer;
use crate::physical::PhysicalRainSynthesizer;
use crate::procedural::ProceduralSynthesizer;
use shared::rain::{EngineTelemetry, RainState};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Hard floor for the synthesis/output sample rate. Below this the 8 kHz-and-up noise-shaped
/// bands (rain "air", mist, transducer floor) alias or vanish, and the brown-noise corner
/// frequencies drift far from their tuned values. Lowering it requires an anti-aliasing stage.
pub const MIN_OUTPUT_SAMPLE_RATE: f32 = 32_000.0;
/// Rate requested from the platform when the default device/context rate is unsuitable.
pub const TARGET_OUTPUT_SAMPLE_RATE: f32 = 48_000.0;
/// Upper sanity bound; anything above is treated as a misreport.
pub const MAX_OUTPUT_SAMPLE_RATE: f32 = 192_000.0;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("No audio output device found")]
    NoOutputDevice,
    #[error("Device configuration error: {0}")]
    ConfigError(String),
    #[error("Stream build error: {0}")]
    StreamError(String),
    #[error("WebAudio error: {0}")]
    WebAudioError(String),
}

/// Circular FIFO buffer for stereo audio frames
#[derive(Debug, Clone)]
pub struct AudioRingBuffer {
    buffer: Vec<f32>, // interleaved stereo [L, R, L, R, ...]
    capacity_frames: usize,
    read_pos: usize,
    write_pos: usize,
    available_frames: usize,
}

impl AudioRingBuffer {
    pub fn new(capacity_frames: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity_frames * 2],
            capacity_frames,
            read_pos: 0,
            write_pos: 0,
            available_frames: 0,
        }
    }

    #[inline]
    pub fn capacity_frames(&self) -> usize {
        self.capacity_frames
    }

    #[inline]
    pub fn size_in_bytes(&self) -> usize {
        self.capacity_frames * 2 * std::mem::size_of::<f32>()
    }

    #[inline]
    pub fn dynamic_bytes_used(&self) -> usize {
        self.available_frames * 2 * std::mem::size_of::<f32>()
    }

    #[inline]
    pub fn available_frames(&self) -> usize {
        self.available_frames
    }

    #[inline]
    pub fn free_frames(&self) -> usize {
        self.capacity_frames.saturating_sub(self.available_frames)
    }

    /// Dynamically resizes the ring buffer capacity relative to target performance
    /// while preserving existing audio samples without clicks.
    pub fn resize_relative_to_performance(&mut self, new_capacity: usize) {
        let new_capacity = new_capacity.clamp(128, 96000);
        if new_capacity == self.capacity_frames {
            return;
        }
        let mut new_buf = vec![0.0f32; new_capacity * 2];
        let frames_to_copy = self.available_frames.min(new_capacity);
        for i in 0..frames_to_copy {
            let src_idx = ((self.read_pos + i) % self.capacity_frames) * 2;
            let dst_idx = i * 2;
            new_buf[dst_idx] = self.buffer[src_idx];
            new_buf[dst_idx + 1] = self.buffer[src_idx + 1];
        }
        self.buffer = new_buf;
        self.capacity_frames = new_capacity;
        self.read_pos = 0;
        self.write_pos = frames_to_copy % new_capacity;
        self.available_frames = frames_to_copy;
    }

    /// Evaluates buffer health relative to target capacity in frames
    #[inline]
    pub fn relative_health_ratio(&self, target_capacity_frames: usize) -> f32 {
        if target_capacity_frames == 0 {
            1.0
        } else {
            (self.available_frames as f32 / target_capacity_frames as f32).clamp(0.0, 2.0)
        }
    }

    #[inline]
    pub fn push_frame(&mut self, left: f32, right: f32) -> bool {
        if self.available_frames >= self.capacity_frames {
            return false;
        }
        let idx = self.write_pos * 2;
        self.buffer[idx] = left;
        self.buffer[idx + 1] = right;
        self.write_pos = (self.write_pos + 1) % self.capacity_frames;
        self.available_frames += 1;
        true
    }

    #[inline]
    pub fn pop_frame(&mut self) -> Option<(f32, f32)> {
        if self.available_frames == 0 {
            return None;
        }
        let idx = self.read_pos * 2;
        let l = self.buffer[idx];
        let r = self.buffer[idx + 1];
        self.read_pos = (self.read_pos + 1) % self.capacity_frames;
        self.available_frames -= 1;
        Some((l, r))
    }

    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.available_frames = 0;
    }
}

/// Zero-allocation, smooth soft-knee tanh saturation limiter.
/// Limits audio signals strictly to [-1.0, 1.0], preventing harsh digital clipping.
#[inline]
pub fn soft_limit(sample: f32) -> f32 {
    if sample.abs() < 0.8 {
        sample
    } else {
        (sample * 0.95).tanh()
    }
}

/// Dynamic crest factor compressor and envelope follower.
/// Prevents transient harshness during heavy downpours while preserving
/// delicate micro-droplet impact clarity through adaptive soft-knee gain reduction.
#[derive(Debug, Clone)]
pub struct CrestFactorGovernor {
    peak_env: f32,
    rms_env: f32,
    gain: f32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl CrestFactorGovernor {
    pub fn new(sample_rate: f32) -> Self {
        let sr = sample_rate.max(1.0);
        let attack_coeff = (-1.0 / (0.005 * sr)).exp();
        let release_coeff = (-1.0 / (0.080 * sr)).exp();
        Self {
            peak_env: 0.0,
            rms_env: 0.0,
            gain: 1.0,
            attack_coeff,
            release_coeff,
        }
    }

    #[inline]
    pub fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        let max_abs = left.abs().max(right.abs());
        let sq = (left * left + right * right) * 0.5;

        if max_abs > self.peak_env {
            self.peak_env = max_abs;
        } else {
            self.peak_env =
                self.peak_env * self.release_coeff + max_abs * (1.0 - self.release_coeff);
        }

        self.rms_env = self.rms_env * self.release_coeff + sq * (1.0 - self.release_coeff);
        let rms = self.rms_env.sqrt().max(1e-5);
        let crest_factor = self.peak_env / rms;

        let target_gain = if crest_factor > 4.5 {
            (4.5 / crest_factor).powf(0.5)
        } else if self.peak_env > 0.85 {
            0.85 / self.peak_env
        } else {
            1.0
        };

        if target_gain < self.gain {
            self.gain = self.gain * self.attack_coeff + target_gain * (1.0 - self.attack_coeff);
        } else {
            self.gain = self.gain * self.release_coeff + target_gain * (1.0 - self.release_coeff);
        }

        (left * self.gain, right * self.gain)
    }
}

/// Shared thread-safe state container between UI thread and Audio thread
#[derive(Clone, Debug)]
pub struct SharedAudioState {
    pub rain: Arc<RwLock<RainState>>,
    pub decode_mode: Arc<RwLock<DecodeMode>>,
    pub orientation: Arc<RwLock<[f32; 3]>>,
    pub telemetry: Arc<RwLock<EngineTelemetry>>,
    pub ring_buffer: Arc<RwLock<AudioRingBuffer>>,
    pub history: Arc<RwLock<AcousticHistoryBuffer>>,
}

impl Default for SharedAudioState {
    fn default() -> Self {
        Self {
            rain: Arc::new(RwLock::new(RainState::default())),
            decode_mode: Arc::new(RwLock::new(DecodeMode::default())),
            orientation: Arc::new(RwLock::new([0.0, 0.0, 0.0])),
            telemetry: Arc::new(RwLock::new(EngineTelemetry::default())),
            ring_buffer: Arc::new(RwLock::new(AudioRingBuffer::new(12000))),
            history: Arc::new(RwLock::new(AcousticHistoryBuffer::new(30.0, 48000.0))),
        }
    }
}

impl SharedAudioState {
    pub fn new(state: RainState, mode: DecodeMode) -> Self {
        let telemetry = Arc::new(RwLock::new(state.telemetry.clone()));
        Self {
            rain: Arc::new(RwLock::new(state)),
            decode_mode: Arc::new(RwLock::new(mode)),
            orientation: Arc::new(RwLock::new([0.0, 0.0, 0.0])),
            telemetry,
            ring_buffer: Arc::new(RwLock::new(AudioRingBuffer::new(12000))),
            history: Arc::new(RwLock::new(AcousticHistoryBuffer::new(30.0, 48000.0))),
        }
    }

    pub fn update_rain(&self, state: &RainState) {
        if let Ok(mut lock) = self.rain.write() {
            let mut new_state = state.clone();
            if let Ok(tele_lock) = self.telemetry.read() {
                new_state.telemetry = tele_lock.clone();
            }
            *lock = new_state;
        }
    }

    pub fn update_telemetry(&self, telemetry: &EngineTelemetry) {
        if let Ok(mut lock) = self.telemetry.try_write() {
            *lock = telemetry.clone();
        }
    }

    pub fn get_telemetry(&self) -> EngineTelemetry {
        if let Ok(lock) = self.telemetry.read() {
            lock.clone()
        } else {
            EngineTelemetry::default()
        }
    }

    pub fn set_decode_mode(&self, mode: DecodeMode) {
        if let Ok(mut lock) = self.decode_mode.write() {
            *lock = mode;
        }
    }

    pub fn set_orientation(&self, yaw: f32, pitch: f32, roll: f32) {
        if let Ok(mut lock) = self.orientation.write() {
            *lock = [yaw, pitch, roll];
        }
    }
}

/// Native desktop audio runner using CPAL with dedicated background neural inference thread
#[cfg(not(target_arch = "wasm32"))]
pub struct DesktopAudioEngine {
    _stream: cpal::Stream,
    pub state: SharedAudioState,
    _shutdown: Arc<std::sync::atomic::AtomicBool>,
    _inference_thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for DesktopAudioEngine {
    fn drop(&mut self) {
        self._shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl DesktopAudioEngine {
    pub fn start(initial_state: RainState, mode: DecodeMode) -> Result<Self, AudioError> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(AudioError::NoOutputDevice)?;

        let supported_config = device
            .default_output_config()
            .map_err(|e| AudioError::ConfigError(e.to_string()))?;

        let sample_rate = supported_config.sample_rate().0 as f32;
        let channels = supported_config.channels() as usize;

        let quality_tier = initial_state.quality_tier;
        let cached_rain = initial_state.clone();
        let state = SharedAudioState::new(initial_state, mode);
        let audio_state = state.clone();

        let mut synth = ProceduralSynthesizer::new(sample_rate);
        let mut physical_synth = PhysicalRainSynthesizer::new(sample_rate);

        // Load embedded ternary weights by default to avoid blocking audio thread on initialization
        let weight_cache =
            inference::weight_loader::WeightLoader::load_embedded_ternary().unwrap_or_default();
        let mut runner = inference::runner::InferenceRunner::new(quality_tier, weight_cache);

        let mut decoder = AmbisonicDecoder::new(mode);
        let mut buffer_guard = BufferGuard::new(sample_rate, 45.0);
        let mut crest_governor = CrestFactorGovernor::new(sample_rate);
        let mut cached_rain = cached_rain;
        let mut rb = AudioRingBuffer::new(12000);

        let err_fn = |err| tracing::error!("An error occurred on the audio stream: {err}");

        let (param_tx, param_rx) =
            std::sync::mpsc::sync_channel::<inference::NeuralParametricControl>(128);
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_state = state.clone();
        let worker_shutdown = shutdown.clone();

        let inference_handle = std::thread::Builder::new()
            .name("rainai-neural-synthesis".into())
            .spawn(move || {
                let weight_cache = inference::weight_loader::WeightLoader::load_embedded_ternary()
                    .unwrap_or_default();
                let mut runner =
                    inference::runner::InferenceRunner::new(quality_tier, weight_cache);

                while !worker_shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                    let rain = if let Ok(guard) = worker_state.rain.read() {
                        guard.clone()
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        continue;
                    };

                    if !rain.is_playing {
                        std::thread::sleep(std::time::Duration::from_millis(15));
                        continue;
                    }

                    let cond = rain.to_conditioning_array();
                    let ctrl = runner.step_parametric(&cond);

                    let _ = param_tx.try_send(ctrl);
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            })
            .ok();

        let mut latest_neural_ctrl: Option<inference::NeuralParametricControl> = None;

        let stream = match supported_config.sample_format() {
            cpal::SampleFormat::F32 => device
                .build_output_stream(
                    &supported_config.into(),
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        let dt = (data.len() / channels.max(1)) as f32 / sample_rate.max(1.0);
                        let frames_needed = data.len() / channels.max(1);

                        while let Ok(ctrl) = param_rx.try_recv() {
                            latest_neural_ctrl = Some(ctrl);
                        }

                        // Wait-free synchronization: try_read avoids blocking the real-time audio thread
                        if let Ok(guard) = audio_state.rain.try_read() {
                            cached_rain = guard.clone();
                        }
                        let mut rain = cached_rain.clone();

                        if let Ok(dm) = audio_state.decode_mode.try_read() {
                            decoder.mode = *dm;
                        }
                        if let Ok(ori) = audio_state.orientation.try_read() {
                            decoder.set_orientation(ori[0], ori[1], ori[2]);
                        }

                        buffer_guard
                            .set_target_buffer_ms(rain.optimization_profile.target_buffer_ms());
                        let action = buffer_guard.update(rb.available_frames(), frames_needed, dt);

                        rain.telemetry.buffer_health_ms = action.buffer_health_ms;
                        rain.telemetry.synthesis_blend = action.synthesis_blend;
                        rain.telemetry.cpu_headroom = 1.0 - action.compute_deficit;
                        rain.telemetry.buffer_capacity_ms =
                            (rb.capacity_frames() as f32 / sample_rate.max(1.0)) * 1000.0;
                        rain.telemetry.is_prebuffered =
                            rb.available_frames() >= action.target_headroom_frames;

                        let frames_to_generate =
                            action.frames_to_generate.min(frames_needed.max(256) * 4);

                        if frames_to_generate > 0 {
                            runner.set_target_tier(rain.quality_tier);
                            runner.set_thinking_steps(rain.thinking_steps);
                            runner.set_use_consistency_jump(rain.use_consistency_jump);

                            // Condition synthesis as active to prime the buffer
                            let mut synth_state = rain.clone();
                            synth_state.is_playing = true;
                            let cond = synth_state.to_conditioning_array();

                            for _ in 0..frames_to_generate {
                                let layers = rain.sound_layers;
                                let suppress_droplets = layers.raindrops;

                                let phys_foa = if layers.raindrops {
                                    physical_synth.process_frame_modulated(
                                        &synth_state,
                                        latest_neural_ctrl.as_ref(),
                                    )
                                } else {
                                    crate::decoder::FoaFrame::default()
                                };

                                let proc_foa = if layers.rain_wash {
                                    synth.process_frame_partitioned(
                                        &synth_state,
                                        latest_neural_ctrl.as_ref(),
                                        suppress_droplets,
                                    )
                                } else {
                                    crate::decoder::FoaFrame::default()
                                };

                                let ai_foa = if layers.ai_texture {
                                    synth.process_frame_modulated(
                                        &synth_state,
                                        latest_neural_ctrl.as_ref(),
                                    )
                                } else {
                                    crate::decoder::FoaFrame::default()
                                };

                                let foa = crate::decoder::FoaFrame::new(
                                    phys_foa.w + proc_foa.w + ai_foa.w,
                                    phys_foa.x + proc_foa.x + ai_foa.x,
                                    phys_foa.y + proc_foa.y + ai_foa.y,
                                    phys_foa.z + proc_foa.z + ai_foa.z,
                                );

                                // Record into AcousticHistoryBuffer if recording is enabled
                                if rain.history_recording_enabled {
                                    if let Ok(mut hist) = audio_state.history.try_write() {
                                        hist.record_frame(&cond, foa);
                                    }
                                }

                                let stereo = decoder.decode_stereo(foa);

                                let (crest_l, crest_r) =
                                    crest_governor.process(stereo.left, stereo.right);
                                let limited_l = soft_limit(crest_l);
                                let limited_r = soft_limit(crest_r);
                                if !rb.push_frame(limited_l, limited_r) {
                                    break;
                                }
                            }
                        }

                        let current_available = rb.available_frames();
                        let buffer_ms = (current_available as f32 / sample_rate) * 1000.0;
                        rain.telemetry.buffer_health_ms = buffer_ms;
                        rain.telemetry.buffer_health_ratio =
                            rb.relative_health_ratio(action.target_headroom_frames);
                        if let Ok(hist) = audio_state.history.try_read() {
                            rain.telemetry.history_seconds_available = hist.available_seconds();
                        }

                        // 2. Playback consumption
                        if rain.is_playing {
                            rain.telemetry.is_prebuffered = false;
                            for frame_chunk in data.chunks_mut(channels) {
                                if let Some((l, r)) = rb.pop_frame() {
                                    let out_l = soft_limit(l * rain.master_volume);
                                    let out_r = soft_limit(r * rain.master_volume);
                                    if channels >= 2 {
                                        frame_chunk[0] = out_l;
                                        frame_chunk[1] = out_r;
                                        for extra in &mut frame_chunk[2..] {
                                            *extra = 0.0;
                                        }
                                    } else if channels == 1 {
                                        frame_chunk[0] = (out_l + out_r) * 0.5;
                                    }
                                } else {
                                    for s in frame_chunk.iter_mut() {
                                        *s = 0.0;
                                    }
                                }
                            }
                        } else {
                            // Paused: output silence to hardware
                            for s in data.iter_mut() {
                                *s = 0.0;
                            }

                            if current_available >= action.target_headroom_frames.saturating_sub(64)
                            {
                                rain.telemetry.is_prebuffered = true;
                                rain.telemetry.governor_status =
                                    "Pre-Buffered & Ready (Happy)".into();
                            } else {
                                rain.telemetry.is_prebuffered = false;
                                rain.telemetry.governor_status = format!(
                                    "Pre-Buffering... ({:.0}ms / {:.0}ms)",
                                    buffer_ms, buffer_guard.target_buffer_ms
                                );
                            }
                        }

                        audio_state.update_telemetry(&rain.telemetry);
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| AudioError::StreamError(e.to_string()))?,
            _ => {
                return Err(AudioError::StreamError(
                    "Unsupported sample format (expected F32)".into(),
                ));
            }
        };

        stream
            .play()
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        Ok(Self {
            _stream: stream,
            state,
            _shutdown: shutdown,
            _inference_thread: inference_handle,
        })
    }
}

/// WebAssembly WebAudio runner
#[cfg(target_arch = "wasm32")]
pub struct WebAudioEngine {
    pub state: SharedAudioState,
    _ctx: web_sys::AudioContext,
    _processor: web_sys::ScriptProcessorNode,
    _closure: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::AudioProcessingEvent)>,
}

#[cfg(target_arch = "wasm32")]
impl WebAudioEngine {
    pub fn start(initial_state: RainState, mode: DecodeMode) -> Result<Self, AudioError> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen::closure::Closure;

        // Reuse window.__rainAudioContext from index.js when it satisfies the output-rate floor;
        // otherwise create a fresh context that explicitly requests TARGET_OUTPUT_SAMPLE_RATE
        // (the browser resamples to the hardware rate, so the DSP never runs below the floor).
        let existing_ctx = web_sys::window()
            .and_then(|win| {
                js_sys::Reflect::get(&win, &wasm_bindgen::JsValue::from_str("__rainAudioContext"))
                    .ok()
            })
            .and_then(|v| v.dyn_into::<web_sys::AudioContext>().ok())
            .filter(|c| {
                let sr = c.sample_rate();
                sr.is_finite() && (MIN_OUTPUT_SAMPLE_RATE..=MAX_OUTPUT_SAMPLE_RATE).contains(&sr)
            });

        let ctx = match existing_ctx {
            Some(ctx) => ctx,
            None => {
                let opts = web_sys::AudioContextOptions::new();
                opts.set_sample_rate(TARGET_OUTPUT_SAMPLE_RATE);
                web_sys::AudioContext::new_with_context_options(&opts)
                    .or_else(|_| web_sys::AudioContext::new())
                    .map_err(|e| AudioError::WebAudioError(format!("{e:?}")))?
            }
        };

        if let Some(win) = web_sys::window() {
            let _ = js_sys::Reflect::set(
                &win,
                &wasm_bindgen::JsValue::from_str("__rainAudioContext"),
                &ctx,
            );
        }

        let raw_sr = ctx.sample_rate();
        let sample_rate = if raw_sr.is_finite()
            && (MIN_OUTPUT_SAMPLE_RATE..=MAX_OUTPUT_SAMPLE_RATE).contains(&raw_sr)
        {
            raw_sr
        } else if raw_sr.is_finite() && raw_sr >= 8000.0 {
            web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(&format!(
                "AudioContext sample rate {raw_sr} Hz is below the {MIN_OUTPUT_SAMPLE_RATE} Hz floor. Synthesis running with sub-Nyquist anti-aliasing / safe clamping."
            )));
            raw_sr
        } else {
            web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(&format!(
                "AudioContext sample rate {raw_sr} Hz is invalid. Defaulting to {TARGET_OUTPUT_SAMPLE_RATE} Hz."
            )));
            TARGET_OUTPUT_SAMPLE_RATE
        };
        let quality_tier = initial_state.quality_tier;
        let cached_rain = initial_state.clone();
        let state = SharedAudioState::new(initial_state, mode);

        let processor = ctx
            .create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(
                2048, 0, 2,
            )
            .map_err(|e| AudioError::WebAudioError(format!("{e:?}")))?;

        let audio_state = state.clone();
        let mut synth = ProceduralSynthesizer::new(sample_rate);
        let mut physical_synth = PhysicalRainSynthesizer::new(sample_rate);

        let weight_cache =
            inference::weight_loader::WeightLoader::load_embedded_ternary().unwrap_or_default();
        let mut runner = inference::runner::InferenceRunner::new(quality_tier, weight_cache);

        let mut decoder = AmbisonicDecoder::new(mode);
        let mut buffer_guard = BufferGuard::new(sample_rate, 45.0);
        let mut crest_governor = CrestFactorGovernor::new(sample_rate);
        let mut cached_rain = cached_rain;
        let mut rb = AudioRingBuffer::new(12000);
        let mut left_out = vec![0.0f32; 2048];
        let mut right_out = vec![0.0f32; 2048];

        let closure = Closure::wrap(Box::new(move |event: web_sys::AudioProcessingEvent| {
            let output_buffer = match event.output_buffer() {
                Ok(buf) => buf,
                Err(_) => return,
            };

            let frames_needed = output_buffer.length() as usize;
            if left_out.len() != frames_needed {
                left_out.resize(frames_needed, 0.0);
                right_out.resize(frames_needed, 0.0);
            }

            let dt = frames_needed as f32 / sample_rate.max(1.0);

            // Wait-free synchronization: try_read avoids blocking the real-time audio thread
            if let Ok(guard) = audio_state.rain.try_read() {
                cached_rain = guard.clone();
            }
            let mut rain = cached_rain.clone();

            if let Ok(dm) = audio_state.decode_mode.try_read() {
                decoder.mode = *dm;
            }
            if let Ok(ori) = audio_state.orientation.try_read() {
                decoder.set_orientation(ori[0], ori[1], ori[2]);
            }

            buffer_guard.set_target_buffer_ms(rain.optimization_profile.target_buffer_ms());
            let action = buffer_guard.update(rb.available_frames(), frames_needed, dt);

            rain.telemetry.buffer_health_ms = action.buffer_health_ms;
            rain.telemetry.synthesis_blend = action.synthesis_blend;
            rain.telemetry.cpu_headroom = 1.0 - action.compute_deficit;
            rain.telemetry.buffer_capacity_ms =
                (rb.capacity_frames() as f32 / sample_rate.max(1.0)) * 1000.0;
            rain.telemetry.is_prebuffered = rb.available_frames() >= action.target_headroom_frames;

            // Incremental headroom budgeting: clamp generation to frames_needed + 128 to prevent main thread spikes
            let max_batch = frames_needed + 128;
            let frames_to_generate = action.frames_to_generate.min(max_batch);

            if frames_to_generate > 0 {
                runner.set_target_tier(rain.quality_tier);
                runner.set_thinking_steps(rain.thinking_steps);
                runner.set_use_consistency_jump(rain.use_consistency_jump);

                let mut synth_state = rain.clone();
                synth_state.is_playing = true;
                let _blend = action.synthesis_blend;
                let cond = synth_state.to_conditioning_array();

                // Evaluate neural-parametric control at block rate (~50-100Hz)
                let neural_ctrl = runner.step_parametric(&cond);

                for _ in 0..frames_to_generate {
                    let layers = rain.sound_layers;
                    let suppress_droplets = layers.raindrops;

                    let phys_foa = if layers.raindrops {
                        physical_synth.process_frame_modulated(&synth_state, Some(&neural_ctrl))
                    } else {
                        crate::decoder::FoaFrame::default()
                    };

                    let proc_foa = if layers.rain_wash {
                        synth.process_frame_partitioned(
                            &synth_state,
                            Some(&neural_ctrl),
                            suppress_droplets,
                        )
                    } else {
                        crate::decoder::FoaFrame::default()
                    };

                    let ai_foa = if layers.ai_texture {
                        synth.process_frame_modulated(&synth_state, Some(&neural_ctrl))
                    } else {
                        crate::decoder::FoaFrame::default()
                    };

                    let foa = crate::decoder::FoaFrame::new(
                        phys_foa.w + proc_foa.w + ai_foa.w,
                        phys_foa.x + proc_foa.x + ai_foa.x,
                        phys_foa.y + proc_foa.y + ai_foa.y,
                        phys_foa.z + proc_foa.z + ai_foa.z,
                    );

                    // Record into AcousticHistoryBuffer if recording is enabled
                    if rain.history_recording_enabled {
                        if let Ok(mut hist) = audio_state.history.try_write() {
                            hist.record_frame(&cond, foa);
                        }
                    }

                    let stereo = decoder.decode_stereo(foa);

                    let (crest_l, crest_r) = crest_governor.process(stereo.left, stereo.right);
                    let limited_l = soft_limit(crest_l);
                    let limited_r = soft_limit(crest_r);
                    if !rb.push_frame(limited_l, limited_r) {
                        break;
                    }
                }
            }

            let current_available = rb.available_frames();
            let buffer_ms = (current_available as f32 / sample_rate) * 1000.0;
            rain.telemetry.buffer_health_ms = buffer_ms;
            rain.telemetry.buffer_health_ratio =
                rb.relative_health_ratio(action.target_headroom_frames);
            if let Ok(hist) = audio_state.history.try_read() {
                rain.telemetry.history_seconds_available = hist.available_seconds();
            }

            if rain.is_playing {
                rain.telemetry.is_prebuffered = false;
                for i in 0..frames_needed {
                    if let Some((l, r)) = rb.pop_frame() {
                        left_out[i] = soft_limit(l * rain.master_volume);
                        right_out[i] = soft_limit(r * rain.master_volume);
                    } else {
                        left_out[i] = 0.0;
                        right_out[i] = 0.0;
                    }
                }
            } else {
                for i in 0..frames_needed {
                    left_out[i] = 0.0;
                    right_out[i] = 0.0;
                }

                if current_available >= action.target_headroom_frames.saturating_sub(64) {
                    rain.telemetry.is_prebuffered = true;
                    rain.telemetry.governor_status = "Pre-Buffered & Ready (Happy)".into();
                } else {
                    rain.telemetry.is_prebuffered = false;
                    rain.telemetry.governor_status = format!(
                        "Pre-Buffering... ({:.0}ms / {:.0}ms)",
                        buffer_ms, buffer_guard.target_buffer_ms
                    );
                }
            }

            audio_state.update_telemetry(&rain.telemetry);

            let _ = output_buffer.copy_to_channel(&left_out, 0);
            let _ = output_buffer.copy_to_channel(&right_out, 1);
        }) as Box<dyn FnMut(web_sys::AudioProcessingEvent)>);

        processor.set_onaudioprocess(Some(closure.as_ref().unchecked_ref()));
        processor
            .connect_with_audio_node(&ctx.destination())
            .map_err(|e| AudioError::WebAudioError(format!("{e:?}")))?;

        Ok(Self {
            state,
            _ctx: ctx,
            _processor: processor,
            _closure: closure,
        })
    }

    pub fn resume(&self) -> Result<(), AudioError> {
        let _ = self._ctx.resume();
        Ok(())
    }
}
