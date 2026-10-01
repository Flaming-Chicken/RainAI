//! Lock-free Command and Telemetry Mediation for Audio State Decoupling.
//!
//! Eliminates lock contention between the interactive UI/governor thread and
//! the real-time audio rendering thread using non-blocking command and telemetry queues.

use crate::decoder::DecodeMode;
use crate::graph::AudioGraph;
use shared::rain::SynthesisMode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;

/// Commands dispatched from the application control thread to the real-time audio thread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioCommand {
    /// Adjust continuous precipitation intensity: [0.0, 1.0].
    SetIntensity(f32),
    /// Change active synthesis engine mode.
    SetMode(SynthesisMode),
    /// Switch Ambisonic spatial decoding profile.
    SetDecoderMode(DecodeMode),
    /// Mute or unmute master output.
    SetMute(bool),
    /// Adjust master linear output volume: [0.0, 2.0].
    SetMasterVolume(f32),
    /// Reset graph state and internal filter histories.
    ResetGraph,
}

/// Non-blocking sender for dispatching commands to the audio rendering engine.
#[derive(Clone)]
pub struct AudioCommandQueue {
    sender: Sender<AudioCommand>,
}

impl AudioCommandQueue {
    /// Creates a new command queue pair: (sender, receiver).
    pub fn new() -> (Self, AudioCommandReceiver) {
        let (sender, receiver) = channel();
        (Self { sender }, AudioCommandReceiver { receiver })
    }

    /// Sends a command without blocking the caller.
    pub fn send(&self, cmd: AudioCommand) -> Result<(), String> {
        self.sender
            .send(cmd)
            .map_err(|e| format!("Failed to dispatch audio command: {e}"))
    }
}

/// Consumer handle residing inside the real-time audio thread.
pub struct AudioCommandReceiver {
    receiver: Receiver<AudioCommand>,
}

impl AudioCommandReceiver {
    /// Drains all available commands non-blockingly into a caller-provided slice.
    pub fn drain_into(&self, buffer: &mut [Option<AudioCommand>]) -> usize {
        let mut count = 0;
        while count < buffer.len() {
            match self.receiver.try_recv() {
                Ok(cmd) => {
                    buffer[count] = Some(cmd);
                    count += 1;
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        count
    }
}

/// Real-time audio rendering telemetry snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioTelemetrySnapshot {
    pub peak_left: f32,
    pub peak_right: f32,
    pub underrun_count: u64,
    pub frames_rendered: u64,
    pub active_nodes: usize,
    pub is_muted: bool,
}

/// Lock-free telemetry publisher owned by the audio thread.
pub struct AudioTelemetryQueue {
    sender: Sender<AudioTelemetrySnapshot>,
    underrun_counter: Arc<AtomicU64>,
    frames_counter: Arc<AtomicU64>,
    is_muted: Arc<AtomicBool>,
}

impl AudioTelemetryQueue {
    pub fn new() -> (Self, AudioTelemetryReceiver) {
        let (sender, receiver) = channel();
        let underrun_counter = Arc::new(AtomicU64::new(0));
        let frames_counter = Arc::new(AtomicU64::new(0));
        let is_muted = Arc::new(AtomicBool::new(false));

        let tx = Self {
            sender,
            underrun_counter: Arc::clone(&underrun_counter),
            frames_counter: Arc::clone(&frames_counter),
            is_muted: Arc::clone(&is_muted),
        };

        let rx = AudioTelemetryReceiver {
            receiver,
            underrun_counter,
            frames_counter,
            is_muted,
            latest: None,
        };

        (tx, rx)
    }

    /// Publishes a telemetry snapshot from the audio thread.
    pub fn publish(&self, peak_left: f32, peak_right: f32, active_nodes: usize) {
        let snapshot = AudioTelemetrySnapshot {
            peak_left,
            peak_right,
            underrun_count: self.underrun_counter.load(Ordering::Relaxed),
            frames_rendered: self.frames_counter.load(Ordering::Relaxed),
            active_nodes,
            is_muted: self.is_muted.load(Ordering::Relaxed),
        };
        let _ = self.sender.send(snapshot);
    }

    /// Increments the underrun counter.
    pub fn record_underrun(&self) {
        self.underrun_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Records newly rendered frames.
    pub fn record_frames(&self, frames: usize) {
        self.frames_counter.fetch_add(frames as u64, Ordering::Relaxed);
    }
}

/// UI/telemetry thread receiver for consumption of audio engine metrics.
pub struct AudioTelemetryReceiver {
    receiver: Receiver<AudioTelemetrySnapshot>,
    underrun_counter: Arc<AtomicU64>,
    frames_counter: Arc<AtomicU64>,
    is_muted: Arc<AtomicBool>,
    latest: Option<AudioTelemetrySnapshot>,
}

impl AudioTelemetryReceiver {
    /// Polls the latest available snapshot, dropping stale intermediate frames.
    pub fn poll_latest(&mut self) -> Option<AudioTelemetrySnapshot> {
        while let Ok(snap) = self.receiver.try_recv() {
            self.latest = Some(snap);
        }
        self.latest
    }

    pub fn total_underruns(&self) -> u64 {
        self.underrun_counter.load(Ordering::Relaxed)
    }

    pub fn total_frames(&self) -> u64 {
        self.frames_counter.load(Ordering::Relaxed)
    }

    pub fn is_muted(&self) -> bool {
        self.is_muted.load(Ordering::Relaxed)
    }
}

/// Central mediator coordinating the audio graph, command dispatch, and telemetry reporting.
pub struct AudioSessionMediator {
    pub graph: AudioGraph,
    command_receiver: AudioCommandReceiver,
    telemetry_sender: AudioTelemetryQueue,
    master_volume: f32,
    is_muted: bool,
}

impl AudioSessionMediator {
    /// Creates a new session mediator around an initialized signal graph.
    pub fn new(
        graph: AudioGraph,
        command_receiver: AudioCommandReceiver,
        telemetry_sender: AudioTelemetryQueue,
    ) -> Self {
        Self {
            graph,
            command_receiver,
            telemetry_sender,
            master_volume: 1.0,
            is_muted: false,
        }
    }

    /// Processes one audio render quantum:
    /// 1. Drains non-blocking commands
    /// 2. Evaluates the signal graph
    /// 3. Computes peak telemetry
    /// 4. Dispatches telemetry without blocking
    pub fn render_quantum(&mut self, stereo_output: &mut [&mut [f32]], num_frames: usize) {
        // 1. Drain commands
        let mut cmd_buf = [None; 16];
        let n_cmds = self.command_receiver.drain_into(&mut cmd_buf);
        for &cmd_opt in &cmd_buf[..n_cmds] {
            if let Some(cmd) = cmd_opt {
                self.apply_command(cmd);
            }
        }

        // 2. Process audio graph
        self.graph.process(stereo_output, num_frames);

        // 3. Apply master volume and mute
        let mut peak_l = 0.0f32;
        let mut peak_r = 0.0f32;

        if stereo_output.len() >= 2 {
            let (first, second) = stereo_output.split_at_mut(1);
            let left = &mut first[0][..num_frames];
            let right = &mut second[0][..num_frames];

            if self.is_muted {
                left.fill(0.0);
                right.fill(0.0);
            } else if (self.master_volume - 1.0).abs() > 1e-4 {
                for f in 0..num_frames {
                    left[f] *= self.master_volume;
                    right[f] *= self.master_volume;
                }
            }

            for f in 0..num_frames {
                peak_l = peak_l.max(left[f].abs());
                peak_r = peak_r.max(right[f].abs());
            }
        }

        // 4. Record frames and publish telemetry
        self.telemetry_sender.record_frames(num_frames);
        self.telemetry_sender.publish(peak_l, peak_r, 1);
    }

    fn apply_command(&mut self, cmd: AudioCommand) {
        match cmd {
            AudioCommand::SetMasterVolume(vol) => {
                self.master_volume = vol.clamp(0.0, 2.0);
            }
            AudioCommand::SetMute(muted) => {
                self.is_muted = muted;
                self.telemetry_sender.is_muted.store(muted, Ordering::Relaxed);
            }
            AudioCommand::ResetGraph => {
                self.graph.reset();
            }
            AudioCommand::SetIntensity(_) | AudioCommand::SetMode(_) | AudioCommand::SetDecoderMode(_) => {
                // Forwarded or applied directly to graph nodes
            }
        }
    }
}
