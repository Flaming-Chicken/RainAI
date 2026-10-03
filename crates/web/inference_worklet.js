import init, { WasmInferenceNode } from './pkg/web.js';

class RainInferenceProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.initialized = false;
        this.inferenceNode = null;
        this.conditioningBuffer = new Float32Array(554);
        this.sharedConditioning = null;
        
        this.port.onmessage = async (event) => {
            const { type, payload } = event.data;
            
            if (type === 'INIT_WASM') {
                try {
                    await init(payload.wasmModule);
                    this.inferenceNode = new WasmInferenceNode();
                    this.initialized = true;
                    this.port.postMessage({ type: 'READY' });
                } catch (error) {
                    this.port.postMessage({ type: 'ERROR', message: error.toString() });
                }
            } else if (type === 'SET_SHARED_BUFFER') {
                // Lock-free high-rate updates mapped directly to the UI thread
                this.sharedConditioning = new Float32Array(payload.sharedBuffer);
            } else if (type === 'SET_EXPERTS') {
                if (this.inferenceNode) this.inferenceNode.set_active_experts(payload.experts);
            } else if (type === 'SET_NEURAL_ENABLED') {
                if (this.inferenceNode) this.inferenceNode.set_neural_enabled(payload.enabled);
            } else if (type === 'SET_USE_PHYSICAL') {
                if (this.inferenceNode) this.inferenceNode.set_use_physical(payload.use_physical);
            } else if (type === 'SET_WEATHER') {
                if (this.inferenceNode) {
                    if (payload.intensity !== undefined) this.inferenceNode.set_rain_intensity(payload.intensity);
                    if (payload.wind !== undefined) this.inferenceNode.set_wind_speed(payload.wind);
                    if (payload.volume !== undefined) this.inferenceNode.set_master_volume(payload.volume);
                }
            }
        };
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        
        if (!this.initialized || !output || output.length === 0) {
            return true; 
        }

        const bufferSize = output[0].length;

        if (this.sharedConditioning) {
            this.conditioningBuffer.set(this.sharedConditioning);
        }

        try {
            if (output.length >= 4) {
                // Ambisonic FOA 4-channel mode: W, X, Y, Z
                const planar = this.inferenceNode.step_block_planar(this.conditioningBuffer, bufferSize);
                output[0].set(planar.subarray(0, bufferSize));
                output[1].set(planar.subarray(bufferSize, bufferSize * 2));
                output[2].set(planar.subarray(bufferSize * 2, bufferSize * 3));
                output[3].set(planar.subarray(bufferSize * 3, bufferSize * 4));
            } else if (output.length >= 2) {
                // Binaural / Stereo 2-channel mode: Left, Right
                const stereo = this.inferenceNode.step_block_stereo(this.conditioningBuffer, bufferSize);
                output[0].set(stereo.subarray(0, bufferSize));
                output[1].set(stereo.subarray(bufferSize, bufferSize * 2));
            } else if (output.length === 1) {
                // Mono mode
                const stereo = this.inferenceNode.step_block_stereo(this.conditioningBuffer, bufferSize);
                const left = stereo.subarray(0, bufferSize);
                const right = stereo.subarray(bufferSize, bufferSize * 2);
                for (let i = 0; i < bufferSize; i++) {
                    output[0][i] = (left[i] + right[i]) * 0.5;
                }
            }
        } catch (e) {
            for (let ch = 0; ch < output.length; ch++) {
                output[ch].fill(0.0);
            }
        }

        return true;
    }
}

registerProcessor('rain-inference-processor', RainInferenceProcessor);
