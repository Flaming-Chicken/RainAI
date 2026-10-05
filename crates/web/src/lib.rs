//! WebAssembly (WASM) Entry Point and PWA Runner for RainAI.
//!
//! Handles client-side WebGPU shader pipeline initialization, WebAudio audio worklet contexts,
//! canvas resizing, console panic hooks, and JavaScript/TypeScript interop bindings.

#[cfg(target_arch = "wasm32")]
use app::TemplateApp;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, prelude::*};

#[cfg(target_arch = "wasm32")]
use audio::{
    engine::soft_limit, physical::PhysicalRainSynthesizer, procedural::ProceduralSynthesizer,
};
#[cfg(target_arch = "wasm32")]
use inference::{runner::InferenceRunner, weight_loader::WeightLoader};
#[cfg(target_arch = "wasm32")]
use shared::rain::{CONDITION_DIM, QualityTier, RainState};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn main() {
    // Universal telemetry & logging initialization
    spodeian_telemetry::init_default();

    // Spawn the async eframe WebRunner natively
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .and_then(|win| win.document())
            .expect("Failed to get document");
        let canvas = document
            .get_element_by_id("egui_canvas")
            .expect("Canvas element 'egui_canvas' not found")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Failed to cast element to HtmlCanvasElement");

        let web_options = eframe::WebOptions::default();
        let runner = eframe::WebRunner::new();
        let _ = runner
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
            )
            .await;
    });
}

/// WASM binding for the AudioWorklet to drive neural-parametric synthesis.
///
/// Dispatches the AI model at block rate (~50-100 Hz) to produce parametric control
/// vectors that modulate compiled, high-performance WebAudio DSP synthesizers
/// (Procedural DDSP Filterbank and Physical Rain Cavitation Synthesizer).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmInferenceNode {
    runner: InferenceRunner,
    synth: ProceduralSynthesizer,
    physical_synth: PhysicalRainSynthesizer,
    state: RainState,
    neural_enabled: bool,
    use_physical: bool,
    // Pinned pre-allocated scratch buffers for zero-allocation WebAudio transfers
    planar_buffer: Vec<f32>,
    stereo_buffer: Vec<f32>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmInferenceNode {
    /// Initializes the inference engine, loads the embedded ternary weights,
    /// and instantiates the compiled procedural and physical DSP synthesizers.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmInferenceNode, JsValue> {
        let cache = WeightLoader::load_embedded_ternary()
            .map_err(|e| JsValue::from_str(&format!("Failed to load embedded weights: {}", e)))?;

        let runner = InferenceRunner::new(QualityTier::Ternary158, cache);
        let sample_rate = 48000.0;
        let synth = ProceduralSynthesizer::new(sample_rate);
        let physical_synth = PhysicalRainSynthesizer::new(sample_rate);
        let state = RainState::default();

        // Pre-allocate up to 1024 frames of 4-channel FOA (4096 floats) and 2-channel Stereo (2048 floats)
        let planar_buffer = vec![0.0f32; 4096];
        let stereo_buffer = vec![0.0f32; 2048];

        Ok(Self {
            runner,
            synth,
            physical_synth,
            state,
            neural_enabled: true,
            use_physical: false,
            planar_buffer,
            stereo_buffer,
        })
    }

    /// Processes a single frame of conditioning data and returns 4-channel FOA (W, X, Y, Z).
    pub fn step_frame(&mut self, conditioning: &[f32]) -> Result<Vec<f32>, JsValue> {
        if conditioning.len() != CONDITION_DIM {
            return Err(JsValue::from_str(&format!(
                "Conditioning vector must be exactly {} elements",
                CONDITION_DIM
            )));
        }

        let mut cond_array = [0.0f32; CONDITION_DIM];
        cond_array.copy_from_slice(conditioning);

        let (w, x, y, z) = self.runner.step(&cond_array);
        Ok(vec![w, x, y, z])
    }

    /// Returns raw pointer to the internal pinned planar 4-channel FOA scratch buffer.
    pub fn planar_buffer_ptr(&self) -> *const f32 {
        self.planar_buffer.as_ptr()
    }

    /// Returns raw pointer to the internal pinned stereo 2-channel scratch buffer.
    pub fn stereo_buffer_ptr(&self) -> *const f32 {
        self.stereo_buffer.as_ptr()
    }

    /// Zero-allocation block-rate rendering into internal pinned First-Order Ambisonic (FOA) buffer.
    ///
    /// The neural model evaluates `step_parametric` ONCE per quantum/block, and modulates the
    /// compiled procedural or physical DSP pipeline across `frames`.
    ///
    /// Rendered data is available in WASM linear memory at `planar_buffer_ptr()`.
    /// Returns total samples rendered (`frames * 4`).
    pub fn render_block_planar(
        &mut self,
        conditioning: &[f32],
        frames: usize,
    ) -> Result<usize, JsValue> {
        if conditioning.len() != CONDITION_DIM {
            return Err(JsValue::from_str(&format!(
                "Conditioning vector must be exactly {} elements",
                CONDITION_DIM
            )));
        }

        let total_samples = frames * 4;
        if self.planar_buffer.len() < total_samples {
            self.planar_buffer.resize(total_samples, 0.0f32);
        }

        let modulation = if self.neural_enabled {
            let mut cond_array = [0.0f32; CONDITION_DIM];
            cond_array.copy_from_slice(conditioning);
            Some(self.runner.step_parametric(&cond_array))
        } else {
            None
        };

        let (w_slice, rest) = self.planar_buffer[..total_samples].split_at_mut(frames);
        let (x_slice, rest) = rest.split_at_mut(frames);
        let (y_slice, z_slice) = rest.split_at_mut(frames);

        for i in 0..frames {
            let foa = if self.use_physical {
                self.physical_synth
                    .process_frame_modulated(&self.state, modulation.as_ref())
            } else {
                self.synth
                    .process_frame_modulated(&self.state, modulation.as_ref())
            };

            w_slice[i] = soft_limit(foa.w);
            x_slice[i] = soft_limit(foa.x);
            y_slice[i] = soft_limit(foa.y);
            z_slice[i] = soft_limit(foa.z);
        }

        Ok(total_samples)
    }

    /// Zero-allocation block-rate rendering into internal pinned Stereo buffer.
    ///
    /// Decodes Ambisonic FOA to binaural/stereo:
    /// `L = W * 0.707 + Y * 0.5`, `R = W * 0.707 - Y * 0.5`.
    ///
    /// Rendered data is available in WASM linear memory at `stereo_buffer_ptr()`.
    /// Returns total samples rendered (`frames * 2`).
    pub fn render_block_stereo(
        &mut self,
        conditioning: &[f32],
        frames: usize,
    ) -> Result<usize, JsValue> {
        if conditioning.len() != CONDITION_DIM {
            return Err(JsValue::from_str(&format!(
                "Conditioning vector must be exactly {} elements",
                CONDITION_DIM
            )));
        }

        let total_samples = frames * 2;
        if self.stereo_buffer.len() < total_samples {
            self.stereo_buffer.resize(total_samples, 0.0f32);
        }

        let modulation = if self.neural_enabled {
            let mut cond_array = [0.0f32; CONDITION_DIM];
            cond_array.copy_from_slice(conditioning);
            Some(self.runner.step_parametric(&cond_array))
        } else {
            None
        };

        let (left_slice, right_slice) = self.stereo_buffer[..total_samples].split_at_mut(frames);

        for i in 0..frames {
            let foa = if self.use_physical {
                self.physical_synth
                    .process_frame_modulated(&self.state, modulation.as_ref())
            } else {
                self.synth
                    .process_frame_modulated(&self.state, modulation.as_ref())
            };

            let left = foa.w * 0.707 + foa.y * 0.5;
            let right = foa.w * 0.707 - foa.y * 0.5;

            left_slice[i] = soft_limit(left * self.state.master_volume);
            right_slice[i] = soft_limit(right * self.state.master_volume);
        }

        Ok(total_samples)
    }

    /// Backwards-compatible block rendering returning a copied Vec<f32>.
    pub fn step_block_planar(
        &mut self,
        conditioning: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let total = self.render_block_planar(conditioning, frames)?;
        Ok(self.planar_buffer[..total].to_vec())
    }

    /// Backwards-compatible block rendering returning a copied Vec<f32>.
    pub fn step_block_stereo(
        &mut self,
        conditioning: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let total = self.render_block_stereo(conditioning, frames)?;
        Ok(self.stereo_buffer[..total].to_vec())
    }

    /// Deprecated no-op: expert pruning has been removed in favor of dense recurrence.
    pub fn set_active_experts(&mut self, _experts: usize) {}

    /// Toggle neural AI modulation on or off (falls back to pure procedural DSP).
    pub fn set_neural_enabled(&mut self, enabled: bool) {
        self.neural_enabled = enabled;
    }

    /// Switch between procedural filterbank synthesis and physical rain synthesis.
    pub fn set_use_physical(&mut self, use_physical: bool) {
        self.use_physical = use_physical;
    }

    /// Adjust rain intensity [0.0, 1.0].
    pub fn set_rain_intensity(&mut self, intensity: f32) {
        self.state.weather.intensity = intensity.clamp(0.0, 1.0);
    }

    /// Adjust wind speed in m/s [0.0, 40.0].
    pub fn set_wind_speed(&mut self, speed: f32) {
        self.state.wind.speed = speed.clamp(0.0, 40.0);
    }

    /// Adjust master volume [0.0, 1.0].
    pub fn set_master_volume(&mut self, volume: f32) {
        self.state.master_volume = volume.clamp(0.0, 1.0);
    }
}
