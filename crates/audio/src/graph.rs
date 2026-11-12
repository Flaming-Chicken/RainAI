//! Composable Directed Acyclic Signal Graph and Audio Node Abstractions.
//!
//! Provides zero-allocation real-time audio routing, topological block evaluation,
//! and modular decoupling of procedural synthesizers, physical fluid models, and ambisonic decoders.

use crate::decoder::{AmbisonicDecoder, DecodeMode, FoaFrame};
use crate::noise_masking::AmbientNoiseMasker;
use crate::physical::PhysicalRainSynthesizer;
use crate::procedural::ProceduralSynthesizer;
use shared::rain::RainState;

/// Zero-allocation view over multi-channel contiguous audio buffers.
pub struct AudioBufferSlice<'a> {
    pub channels: &'a mut [&'a mut [f32]],
}

impl<'a> AudioBufferSlice<'a> {
    /// Creates a new buffer slice wrapper over mutable channel slices.
    pub fn new(channels: &'a mut [&'a mut [f32]]) -> Self {
        Self { channels }
    }

    /// Number of audio channels in the slice.
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of frames available in each channel.
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Zeroes all channels in the slice.
    pub fn clear(&mut self) {
        for ch in self.channels.iter_mut() {
            ch.fill(0.0);
        }
    }
}

/// Composable Audio Processing Node Trait.
pub trait AudioNode: Send {
    /// Unique diagnostic or descriptive name of the node.
    fn name(&self) -> &str;

    /// Number of input audio channels expected.
    fn num_inputs(&self) -> usize;

    /// Number of output audio channels generated.
    fn num_outputs(&self) -> usize;

    /// Process a block of audio frames without heap allocation.
    fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_frames: usize);

    /// Resets internal filter states and delays.
    fn reset(&mut self);
}

/// Node wrapping the procedural 16-band subtractive rain synthesizer.
pub struct ProceduralSynthesizerNode {
    synthesizer: ProceduralSynthesizer,
    pub state: RainState,
}

impl ProceduralSynthesizerNode {
    pub fn new(synthesizer: ProceduralSynthesizer, state: RainState) -> Self {
        Self { synthesizer, state }
    }
}

impl AudioNode for ProceduralSynthesizerNode {
    fn name(&self) -> &str {
        "ProceduralSynthesizerNode"
    }

    fn num_inputs(&self) -> usize {
        0
    }

    fn num_outputs(&self) -> usize {
        1 // Mono acoustic output
    }

    fn process(&mut self, _inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_frames: usize) {
        if let Some(out_mono) = outputs.first_mut() {
            let frames = num_frames.min(out_mono.len());
            for i in 0..frames {
                let frame = self.synthesizer.process_frame(&self.state);
                out_mono[i] = frame.w;
            }
        }
    }

    fn reset(&mut self) {
        self.synthesizer.reset();
    }
}

/// Node wrapping the physical aeroacoustic rain model generating 4-channel Ambisonic B-Format ($W, Y, Z, X$).
pub struct PhysicalSynthesizerNode {
    synthesizer: PhysicalRainSynthesizer,
    pub state: RainState,
}

impl PhysicalSynthesizerNode {
    pub fn new(synthesizer: PhysicalRainSynthesizer, state: RainState) -> Self {
        Self { synthesizer, state }
    }
}

impl AudioNode for PhysicalSynthesizerNode {
    fn name(&self) -> &str {
        "PhysicalSynthesizerNode"
    }

    fn num_inputs(&self) -> usize {
        0
    }

    fn num_outputs(&self) -> usize {
        4 // Ambisonic B-format (W, Y, Z, X)
    }

    fn process(&mut self, _inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_frames: usize) {
        if outputs.len() >= 4 {
            let frames = num_frames
                .min(outputs[0].len())
                .min(outputs[1].len())
                .min(outputs[2].len())
                .min(outputs[3].len());

            for f in 0..frames {
                let frame = self.synthesizer.process_frame(&self.state);
                outputs[0][f] = frame.w;
                outputs[1][f] = frame.y;
                outputs[2][f] = frame.z;
                outputs[3][f] = frame.x;
            }
        }
    }

    fn reset(&mut self) {
        self.synthesizer.reset();
    }
}

/// Node wrapping ambient noise masking filters.
pub struct NoiseMaskerNode {
    masker: AmbientNoiseMasker,
    pub target_color_index: usize,
}

impl NoiseMaskerNode {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            masker: AmbientNoiseMasker::new(sample_rate),
            target_color_index: 0,
        }
    }
}

impl AudioNode for NoiseMaskerNode {
    fn name(&self) -> &str {
        "NoiseMaskerNode"
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_frames: usize) {
        if let (Some(input), Some(output)) = (inputs.first(), outputs.first_mut()) {
            let frames = num_frames.min(input.len()).min(output.len());
            let gain = 10.0f32.powf(self.masker.current_recommendation.gain_boost_db / 20.0);
            for f in 0..frames {
                output[f] = input[f] * gain;
            }
        }
    }

    fn reset(&mut self) {
        self.masker.reset();
    }
}

/// Node wrapping the Ambisonic B-format spatial decoder.
pub struct AmbisonicDecoderNode {
    decoder: AmbisonicDecoder,
}

impl AmbisonicDecoderNode {
    pub fn new(mode: DecodeMode) -> Self {
        Self {
            decoder: AmbisonicDecoder::new(mode),
        }
    }

    pub fn set_mode(&mut self, mode: DecodeMode) {
        self.decoder.set_mode(mode);
    }
}

impl AudioNode for AmbisonicDecoderNode {
    fn name(&self) -> &str {
        "AmbisonicDecoderNode"
    }

    fn num_inputs(&self) -> usize {
        4 // B-Format W, Y, Z, X
    }

    fn num_outputs(&self) -> usize {
        2 // Stereo L, R
    }

    fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_frames: usize) {
        if inputs.len() >= 4 && outputs.len() >= 2 {
            let frames = num_frames
                .min(inputs[0].len())
                .min(inputs[1].len())
                .min(inputs[2].len())
                .min(inputs[3].len())
                .min(outputs[0].len())
                .min(outputs[1].len());

            for f in 0..frames {
                let w = inputs[0][f];
                let y = inputs[1][f];
                let z = inputs[2][f];
                let x = inputs[3][f];
                let foa = FoaFrame::new(w, x, y, z);
                let stereo = self.decoder.decode_stereo(foa);
                outputs[0][f] = stereo.left;
                outputs[1][f] = stereo.right;
            }
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
    }
}

/// Smooth gain scaling node.
pub struct GainNode {
    pub gain: f32,
    current_gain: f32,
    smoothing: f32,
}

impl GainNode {
    pub fn new(gain: f32) -> Self {
        Self {
            gain,
            current_gain: gain,
            smoothing: 0.05,
        }
    }
}

impl AudioNode for GainNode {
    fn name(&self) -> &str {
        "GainNode"
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        2
    }

    fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_frames: usize) {
        let chs = inputs.len().min(outputs.len());
        for ch in 0..chs {
            let in_slice = inputs[ch];
            let out_slice = &mut outputs[ch];
            let frames = num_frames.min(in_slice.len()).min(out_slice.len());
            for f in 0..frames {
                self.current_gain += self.smoothing * (self.gain - self.current_gain);
                out_slice[f] = in_slice[f] * self.current_gain;
            }
        }
    }

    fn reset(&mut self) {
        self.current_gain = self.gain;
    }
}

/// Connection entry representing an audio routing edge between two node channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioConnection {
    pub from_node: usize,
    pub from_channel: usize,
    pub to_node: usize,
    pub to_channel: usize,
}

/// Directed acyclic audio signal graph executing nodes in deterministic topological order.
pub struct AudioGraph {
    nodes: Vec<Box<dyn AudioNode>>,
    connections: Vec<AudioConnection>,
    execution_order: Vec<usize>,
    max_frames: usize,
    /// Pre-allocated scratch buffers: one buffer per node output channel
    node_buffers: Vec<Vec<Vec<f32>>>,
    /// Pre-allocated scratch channel buffers for decoupled node execution
    scratch_in: Vec<Vec<f32>>,
    scratch_out: Vec<Vec<f32>>,
}

impl AudioGraph {
    /// Creates a new empty audio graph.
    pub fn new(max_frames: usize) -> Self {
        let mf = max_frames.max(64);
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            execution_order: Vec::new(),
            max_frames: mf,
            node_buffers: Vec::new(),
            scratch_in: vec![vec![0.0; mf]; 8],
            scratch_out: vec![vec![0.0; mf]; 8],
        }
    }

    /// Adds an audio node to the graph and returns its index.
    pub fn add_node(&mut self, node: Box<dyn AudioNode>) -> usize {
        let idx = self.nodes.len();
        let num_outs = node.num_outputs();
        let mut ch_bufs = Vec::with_capacity(num_outs);
        for _ in 0..num_outs {
            ch_bufs.push(vec![0.0; self.max_frames]);
        }
        self.node_buffers.push(ch_bufs);
        self.nodes.push(node);
        self.recompute_topology();
        idx
    }

    /// Connects an output channel of a source node to an input channel of a destination node.
    pub fn connect(
        &mut self,
        from_node: usize,
        from_channel: usize,
        to_node: usize,
        to_channel: usize,
    ) -> Result<(), String> {
        if from_node >= self.nodes.len() || to_node >= self.nodes.len() {
            return Err("Node index out of bounds".to_string());
        }
        if from_channel >= self.nodes[from_node].num_outputs() {
            return Err("Source channel out of bounds".to_string());
        }
        if to_channel >= self.nodes[to_node].num_inputs() {
            return Err("Destination channel out of bounds".to_string());
        }

        self.connections.push(AudioConnection {
            from_node,
            from_channel,
            to_node,
            to_channel,
        });

        self.recompute_topology();
        Ok(())
    }

    /// Recomputes the topological execution order using Kahn's algorithm.
    fn recompute_topology(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            self.execution_order.clear();
            return;
        }

        let mut in_degree = vec![0; n];
        let mut adj = vec![Vec::new(); n];

        for conn in &self.connections {
            if conn.from_node < n && conn.to_node < n {
                adj[conn.from_node].push(conn.to_node);
                in_degree[conn.to_node] += 1;
            }
        }

        let mut queue = std::collections::VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        // If cycle detected, fallback to sequential order
        if order.len() == n {
            self.execution_order = order;
        } else {
            self.execution_order = (0..n).collect();
        }
    }

    /// Evaluates the signal graph across all nodes, populating `outputs` without dynamic allocation.
    pub fn process(&mut self, final_outputs: &mut [&mut [f32]], num_frames: usize) {
        let frames = num_frames.min(self.max_frames);

        // Pre-clear all node output buffers
        for node_bufs in self.node_buffers.iter_mut() {
            for buf in node_bufs.iter_mut() {
                buf[..frames].fill(0.0);
            }
        }

        for &node_idx in &self.execution_order {
            let num_inputs = self.nodes[node_idx].num_inputs();
            let num_outputs = self.nodes[node_idx].num_outputs();

            // 1. Prepare inputs in scratch_in
            for ch in 0..num_inputs.min(8) {
                self.scratch_in[ch][..frames].fill(0.0);
            }
            for conn in &self.connections {
                if conn.to_node == node_idx && conn.to_channel < 8 {
                    let src_buf = &self.node_buffers[conn.from_node][conn.from_channel][..frames];
                    self.scratch_in[conn.to_channel][..frames].copy_from_slice(src_buf);
                }
            }

            // 2. Prepare outputs in scratch_out
            for ch in 0..num_outputs.min(8) {
                self.scratch_out[ch][..frames].fill(0.0);
            }

            {
                let in_ptrs: [&[f32]; 8] = [
                    &self.scratch_in[0][..frames],
                    &self.scratch_in[1][..frames],
                    &self.scratch_in[2][..frames],
                    &self.scratch_in[3][..frames],
                    &self.scratch_in[4][..frames],
                    &self.scratch_in[5][..frames],
                    &self.scratch_in[6][..frames],
                    &self.scratch_in[7][..frames],
                ];

                let (s0, s_rest) = self.scratch_out.split_at_mut(1);
                let (s1, s_rest2) = s_rest.split_at_mut(1);
                let (s2, s_rest3) = s_rest2.split_at_mut(1);
                let (s3, s_rest4) = s_rest3.split_at_mut(1);
                let (s4, s_rest5) = s_rest4.split_at_mut(1);
                let (s5, s_rest6) = s_rest5.split_at_mut(1);
                let (s6, s7) = s_rest6.split_at_mut(1);

                let mut out_ptrs: [&mut [f32]; 8] = [
                    &mut s0[0][..frames],
                    &mut s1[0][..frames],
                    &mut s2[0][..frames],
                    &mut s3[0][..frames],
                    &mut s4[0][..frames],
                    &mut s5[0][..frames],
                    &mut s6[0][..frames],
                    &mut s7[0][..frames],
                ];

                self.nodes[node_idx].process(
                    &in_ptrs[..num_inputs.min(8)],
                    &mut out_ptrs[..num_outputs.min(8)],
                    frames,
                );
            }

            // 3. Copy scratch_out back into this node's output buffers
            for ch in 0..num_outputs.min(8) {
                self.node_buffers[node_idx][ch][..frames]
                    .copy_from_slice(&self.scratch_out[ch][..frames]);
            }
        }

        // Copy from terminal node(s) to final outputs
        if let Some(&last_node) = self.execution_order.last() {
            let num_outs = final_outputs.len().min(self.node_buffers[last_node].len());
            for ch in 0..num_outs {
                final_outputs[ch][..frames]
                    .copy_from_slice(&self.node_buffers[last_node][ch][..frames]);
            }
        }
    }

    /// Resets all nodes in the graph.
    pub fn reset(&mut self) {
        for node in self.nodes.iter_mut() {
            node.reset();
        }
    }
}
