//! Native Desktop Studio Executable for RainAI: Real-Time Neural & Physical Spatial Soundscape Synthesis.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use app::TemplateApp;
use eframe::NativeOptions;
use eframe::egui;
use spodeian_telemetry::init_default;
use tracing::{info, warn};

struct CliArgs {
    ir_path: Option<String>,
    export_path: Option<String>,
    duration: f32,
    headless: bool,
}

fn parse_cli_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut ir_path = None;
    let mut export_path = None;
    let mut duration = 10.0f32;
    let mut headless = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ir-path" => {
                if i + 1 < args.len() {
                    ir_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--export" => {
                if i + 1 < args.len() {
                    export_path = Some(args[i + 1].clone());
                    headless = true;
                    i += 1;
                }
            }
            "--duration" => {
                if i + 1 < args.len() {
                    if let Ok(d) = args[i + 1].parse::<f32>() {
                        duration = d;
                    }
                    i += 1;
                }
            }
            "--headless" => {
                headless = true;
            }
            _ => {}
        }
        i += 1;
    }

    CliArgs {
        ir_path,
        export_path,
        duration,
        headless,
    }
}

fn main() -> eframe::Result<()> {
    // Universal telemetry & logging initialization
    init_default();

    let cli_args = parse_cli_args();

    if cli_args.headless || cli_args.export_path.is_some() {
        info!("Running RainAI in headless batch mode (duration: {:.1}s)...", cli_args.duration);
        let mut rain = shared::preset::WeatherPreset::gentle_summer_rain().state;
        rain.is_playing = true;

        if let Some(ref ir_path_str) = cli_args.ir_path {
            let p = std::path::Path::new(ir_path_str);
            if p.exists() {
                if let Ok(bytes) = std::fs::read(p) {
                    let mut spatializer = audio::SofaSpatializer::new(48000);
                    let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("custom_ir.wav");
                    let cache_dir = std::path::Path::new("data/cache/ir");
                    match spatializer.load_custom_ir_from_bytes(fname, &bytes, Some(cache_dir)) {
                        Ok(meta) => {
                            info!("Loaded custom IR '{}' ({} samples, {} channels, sha256: {})", meta.name, meta.sample_count, meta.channels, meta.sha256_hash);
                        }
                        Err(e) => {
                            warn!("Failed to parse custom IR: {e}");
                        }
                    }
                }
            } else {
                warn!("Custom IR file not found: {ir_path_str}");
            }
        }

        let out_path = cli_args.export_path.unwrap_or_else(|| "rainai_export.wav".to_string());
        info!("Rendering {:.1}s audio to '{}'...", cli_args.duration, out_path);
        let mut file = std::fs::File::create(&out_path).map_err(|e| {
            eframe::Error::AppCreation(Box::new(e))
        })?;

        audio::render_wav_stream(
            &rain,
            cli_args.duration,
            48000,
            audio::DecodeMode::BinauralHeadphones,
            &mut file,
            |progress| {
                if (progress * 10.0).fract() < 0.05 {
                    info!("Export progress: {:.0}%", progress * 100.0);
                }
            },
        ).map_err(|e| {
            eframe::Error::AppCreation(Box::new(std::io::Error::other(e.to_string())))
        })?;

        info!("Headless export complete: '{}'", out_path);
        return Ok(());
    }

    let ir_path = cli_args.ir_path;

    // Native window viewport configurations
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("🌧 RainAI · Neural Spatial Soundscape Studio")
            .with_inner_size([1100.0, 750.0])
            .with_min_inner_size([700.0, 500.0])
            .with_active(true)
            .with_resizable(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "🌧 RainAI · Neural Spatial Soundscape Studio",
        options,
        Box::new(move |cc| {
            let mut app = TemplateApp::new(cc);
            if let Some(ref path_str) = ir_path {
                let path = std::path::Path::new(path_str);
                if path.exists() {
                    if let Ok(bytes) = std::fs::read(path) {
                        let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("custom_ir.wav");
                        let cache_dir = std::path::Path::new("data/cache/ir");
                        let _ = app.rain_view.load_custom_ir_bytes(file_name, &bytes, Some(cache_dir));
                    }
                }
            }
            Ok(Box::new(app))
        }),
    )
}
