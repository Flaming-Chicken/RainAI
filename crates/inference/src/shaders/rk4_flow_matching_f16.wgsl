// WebGPU Compute Shader: Continuous Trajectory Runge-Kutta 4th Order (RK4) Half-Precision (FP16) Kernel
// Exploit Android & Mobile GPU f16 WGSL extension for 2x-4x throughput & minimal thermal dissipation.

enable f16;

struct Rk4UniformsF16 {
    dim: u32,
    h: f16,        // step size dt
    t: f16,        // current flow time [0.0, 1.0]
    substep: u32,  // 0: compute k1, 1: compute k2, 2: compute k3, 3: compute k4 & integrate
};

@group(0) @binding(0) var<uniform> uniforms: Rk4UniformsF16;
@group(0) @binding(1) var<storage, read_write> x_current: array<f16>;
@group(0) @binding(2) var<storage, read> velocity_k: array<f16>; // current evaluation of v_theta
@group(0) @binding(3) var<storage, read_write> x_temp: array<f16>;    // intermediate state for next substep
@group(0) @binding(4) var<storage, read_write> k_accumulator: array<f16>; // accumulated weighted sum

@compute @workgroup_size(64)
fn rk4_step_f16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.dim) {
        return;
    }

    let h = uniforms.h;
    let vk = velocity_k[idx];

    switch (uniforms.substep) {
        // Substep 0: Received k1 = v(x_t, t)
        case 0u: {
            k_accumulator[idx] = vk;
            x_temp[idx] = x_current[idx] + 0.5h * h * vk;
        }

        // Substep 1: Received k2 = v(x_t + h/2 * k1, t + h/2)
        case 1u: {
            k_accumulator[idx] = k_accumulator[idx] + 2.0h * vk;
            x_temp[idx] = x_current[idx] + 0.5h * h * vk;
        }

        // Substep 2: Received k3 = v(x_t + h/2 * k2, t + h/2)
        case 2u: {
            k_accumulator[idx] = k_accumulator[idx] + 2.0h * vk;
            x_temp[idx] = x_current[idx] + h * vk;
        }

        // Substep 3: Received k4 = v(x_t + h * k3, t + h)
        // Final integration: x_{t+h} = x_t + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        case 3u: {
            let total_k = k_accumulator[idx] + vk;
            x_current[idx] = x_current[idx] + (h / 6.0h) * total_k;
            x_temp[idx] = x_current[idx];
        }

        default: {}
    }
}
