//! Minimal real-time audio buffer guard and underrun protector.
//!
//! Replaces legacy multi-variable MetaGovernor with a lightweight, deterministic
//! buffer monitor that maintains smooth playback, crossfades to procedural audio
//! on underrun emergencies, and computes platform-independent compute-budget metrics
//! for learned inference ODE solvers.

#[derive(Debug, Clone)]
pub struct BufferGuard {
    pub target_buffer_ms: f32,
    pub sample_rate: f32,
    current_blend: f32, // 0.0 = full primary/neural/physical, 1.0 = procedural emergency fallback
    underrun_count: usize,
    compute_deficit: f32, // 0.0 = plenty of headroom, 1.0 = exhausted
}

impl BufferGuard {
    pub fn new(sample_rate: f32, target_buffer_ms: f32) -> Self {
        Self {
            target_buffer_ms,
            sample_rate,
            current_blend: 0.0,
            underrun_count: 0,
            compute_deficit: 0.0,
        }
    }

    /// Evaluates current ring buffer health and updates smooth crossfade blend.
    pub fn update(
        &mut self,
        available_frames: usize,
        frames_needed: usize,
        dt: f32,
    ) -> BufferGuardAction {
        let buffer_ms = (available_frames as f32 / self.sample_rate.max(1.0)) * 1000.0;
        let target_frames = ((self.target_buffer_ms / 1000.0) * self.sample_rate) as usize;

        let is_starving = available_frames < frames_needed;
        if is_starving {
            self.underrun_count += 1;
        }

        // Compute deficit: how far below target buffer we are
        let deficit_ratio = (1.0 - (buffer_ms / self.target_buffer_ms.max(1.0))).clamp(0.0, 1.0);
        self.compute_deficit = deficit_ratio;

        // Smooth crossfade to procedural: tau ~ 100ms
        let target_blend = if is_starving || buffer_ms < 10.0 {
            1.0 // emergency procedural fallback
        } else if buffer_ms < self.target_buffer_ms * 0.5 {
            0.5 * deficit_ratio
        } else {
            0.0 // healthy primary stream
        };

        let slew = (dt / 0.100).clamp(0.0, 1.0);
        self.current_blend += (target_blend - self.current_blend) * slew;

        let frames_to_generate = if available_frames < target_frames {
            target_frames - available_frames
        } else {
            0
        };

        BufferGuardAction {
            synthesis_blend: self.current_blend,
            compute_deficit: self.compute_deficit,
            frames_to_generate,
            target_headroom_frames: target_frames,
            buffer_health_ms: buffer_ms,
            is_starving,
        }
    }

    pub fn set_target_buffer_ms(&mut self, ms: f32) {
        self.target_buffer_ms = ms;
    }

    pub fn synthesis_blend(&self) -> f32 {
        self.current_blend
    }

    pub fn compute_deficit(&self) -> f32 {
        self.compute_deficit
    }

    pub fn underrun_count(&self) -> usize {
        self.underrun_count
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferGuardAction {
    pub synthesis_blend: f32,
    pub compute_deficit: f32,
    pub frames_to_generate: usize,
    pub target_headroom_frames: usize,
    pub buffer_health_ms: f32,
    pub is_starving: bool,
}
