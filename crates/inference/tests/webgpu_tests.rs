use futures::executor::block_on;
use wgpu::{
    DeviceDescriptor, Features, Limits, PowerPreference, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource,
};

#[test]
fn test_wgsl_shader_compilation() {
    let instance = wgpu::Instance::default();
    let adapter_fut = instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::LowPower,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    });

    if let Some(adapter) = block_on(async { adapter_fut.await.ok() }) {
        let device_fut = adapter.request_device(&DeviceDescriptor {
            label: Some("Shader Compilation Test Device"),
            required_features: Features::empty(),
            required_limits: Limits::downlevel_webgl2_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        });

        if let Some((device, _queue)) = block_on(async { device_fut.await.ok() }) {
            let shaders = [
                (
                    "layer_forward.wgsl",
                    include_str!("../src/shaders/layer_forward.wgsl"),
                ),
                ("mamba2.wgsl", include_str!("../src/shaders/mamba2.wgsl")),
                (
                    "foa_projection.wgsl",
                    include_str!("../src/shaders/foa_projection.wgsl"),
                ),
                (
                    "consistency_jump.wgsl",
                    include_str!("../src/shaders/consistency_jump.wgsl"),
                ),
                (
                    "mamba2_deliberation.wgsl",
                    include_str!("../src/shaders/mamba2_deliberation.wgsl"),
                ),
                (
                    "binaural_convolver.wgsl",
                    include_str!("../src/shaders/binaural_convolver.wgsl"),
                ),
                ("dequant.wgsl", include_str!("../src/shaders/dequant.wgsl")),
                (
                    "mamba2_ssd.wgsl",
                    include_str!("../src/shaders/mamba2_ssd.wgsl"),
                ),
                (
                    "mla_attention.wgsl",
                    include_str!("../src/shaders/mla_attention.wgsl"),
                ),
                (
                    "rk4_flow_matching.wgsl",
                    include_str!("../src/shaders/rk4_flow_matching.wgsl"),
                ),
                (
                    "droplet_panning.wgsl",
                    include_str!("../../app/src/shaders/droplet_panning.wgsl"),
                ),
            ];

            for (name, source) in shaders {
                let _module = device.create_shader_module(ShaderModuleDescriptor {
                    label: Some(name),
                    source: ShaderSource::Wgsl(source.into()),
                });
            }

            // Test FP16 shader if logical device enabled SHADER_F16
            if device.features().contains(Features::SHADER_F16) {
                let _module_f16 = device.create_shader_module(ShaderModuleDescriptor {
                    label: Some("rk4_flow_matching_f16.wgsl"),
                    source: ShaderSource::Wgsl(
                        include_str!("../src/shaders/rk4_flow_matching_f16.wgsl").into(),
                    ),
                });
            }
        }
    }
}
