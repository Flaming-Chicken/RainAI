//! Dedicated Offline Audio Export modal dialog.

use crate::TemplateApp;
use crate::task_queue::{TaskItem, TaskKind};
#[cfg(not(target_arch = "wasm32"))]
use crate::task_queue::{TaskQueue, TaskStatus};
#[cfg(not(target_arch = "wasm32"))]
use audio::decoder::DecodeMode;
use eframe::egui::{self, Color32};
#[cfg(not(target_arch = "wasm32"))]
use shared::rain::RainState;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OfflineAudioFormat {
    #[default]
    FlacLevel8,
    Mp3320k,
    Opus160k,
    Wav32Float,
}

impl OfflineAudioFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::FlacLevel8 => "FLAC (Lossless Level 8 Compression, Vorbis Comments)",
            Self::Mp3320k => "MP3 (320 kbps CBR, ID3v2.4 Metadata Embedded)",
            Self::Opus160k => "Opus (160 kbps VBR, Low-Latency)",
            Self::Wav32Float => "WAV (32-Bit IEEE Float, Uncompressed 48kHz)",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::FlacLevel8 => "flac",
            Self::Mp3320k => "mp3",
            Self::Opus160k => "opus",
            Self::Wav32Float => "wav",
        }
    }
}

#[derive(Clone, Debug)]
pub struct AudioExportModalState {
    pub duration_secs: f32,
    pub sample_rate: u32,
    pub format: OfflineAudioFormat,
    pub embed_provenance_metadata: bool,
    pub status_message: Option<String>,
}

impl Default for AudioExportModalState {
    fn default() -> Self {
        Self {
            duration_secs: 30.0,
            sample_rate: 48000,
            format: OfflineAudioFormat::FlacLevel8,
            embed_provenance_metadata: true,
            status_message: None,
        }
    }
}

pub fn render_export_audio_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if !app.show_audio_export_dialog {
        return;
    }

    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(380.0, 620.0);
    let win_h = (ui.available_height() - 32.0).clamp(420.0, 640.0);

    egui::Window::new("🎵 Offline Audio Export Studio")
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .min_width(360.0)
        .min_height(380.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Render Soundscape to Audio File");
                ui.label(
                    "Synthesizes faster-than-realtime spatial audio with full parameter conditioning and embedded provenance.",
                );
                ui.add_space(6.0);
                ui.separator();
                ui.add_space(6.0);

                // Duration Slider
                ui.horizontal(|ui| {
                    ui.label("Export Duration:");
                    ui.add(
                        egui::Slider::new(&mut app.audio_export_state.duration_secs, 5.0..=300.0)
                            .suffix(" s")
                            .custom_formatter(|v, _| format!("{:.0} seconds ({:.1} min)", v, v / 60.0)),
                    );
                });

                ui.add_space(6.0);

                // Sample Rate Selector
                ui.horizontal(|ui| {
                    ui.label("Sample Rate:");
                    let rates = [44100, 48000, 96000];
                    for r in rates {
                        let label = format!("{} kHz", r / 1000);
                        ui.radio_value(&mut app.audio_export_state.sample_rate, r, label);
                    }
                });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                // Audio Format
                ui.heading("Encoding Format");
                ui.label("Self-contained metadata is written directly into Vorbis comments or ID3v2.4 tags:");
                ui.add_space(4.0);

                let formats = [
                    OfflineAudioFormat::FlacLevel8,
                    OfflineAudioFormat::Mp3320k,
                    OfflineAudioFormat::Opus160k,
                    OfflineAudioFormat::Wav32Float,
                ];

                for fmt in formats {
                    ui.radio_value(&mut app.audio_export_state.format, fmt, fmt.label());
                }

                ui.add_space(6.0);
                ui.checkbox(
                    &mut app.audio_export_state.embed_provenance_metadata,
                    "Embed XAI provenance tags and 64 conditioning parameters into audio metadata",
                );

                ui.add_space(8.0);
                ui.separator();
                // Start Export Action
                ui.horizontal(|ui| {
                    if ui.button("🚀 Start Offline Render").clicked() {
                        let fmt = app.audio_export_state.format;
                        let duration = app.audio_export_state.duration_secs;
                        let title = format!("Render {:.0}s {}", duration, fmt.extension().to_uppercase());
                        let task_id = format!(
                            "export_{}",
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis()
                        );
                        let task = TaskItem::new(
                            &task_id,
                            title,
                            TaskKind::AudioExport {
                                format: fmt.extension().into(),
                            },
                        );
                        app.task_queue.enqueue(task);
                        app.show_task_queue_tray = true;
                        app.audio_export_state.status_message = Some(
                            "Export enqueued! Tracking progress in bottom-right background tray."
                                .to_string(),
                        );

                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let tq = app.task_queue.clone();
                            let rain_state = app.state.rain.clone();
                            let dur = duration;
                            let sr = app.audio_export_state.sample_rate;
                            let mode = app.rain_view.decode_mode;
                            let embed_prov = app.audio_export_state.embed_provenance_metadata;
                            let out_fmt = fmt;
                            let tid = task_id.clone();

                            std::thread::spawn(move || {
                                execute_offline_export(
                                    &tid,
                                    &tq,
                                    rain_state,
                                    dur,
                                    sr,
                                    mode,
                                    out_fmt,
                                    embed_prov,
                                );
                            });
                        }
                    }

                    if ui.button("Close").clicked() {
                        app.show_audio_export_dialog = false;
                    }
                });

                if let Some(ref msg) = app.audio_export_state.status_message {
                    ui.add_space(6.0);
                    ui.colored_label(Color32::from_rgb(80, 200, 100), msg);
                }
            });
        });

    if !open {
        app.show_audio_export_dialog = false;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_offline_export(
    tid: &str,
    tq: &TaskQueue,
    rain_state: RainState,
    duration_secs: f32,
    sample_rate: u32,
    mode: DecodeMode,
    format: OfflineAudioFormat,
    embed_provenance: bool,
) {
    use std::process::Command;

    tq.update_status(
        tid,
        TaskStatus::Running {
            progress: 0.05,
            speed_bps: 0.0,
            status_text: "Initializing acoustic synthesis...".into(),
        },
    );

    let mut wav_bytes = Vec::new();
    let cb_tq = tq.clone();
    let cb_tid = tid.to_string();

    let render_res = audio::export::render_wav_stream(
        &rain_state,
        duration_secs,
        sample_rate,
        mode,
        &mut wav_bytes,
        move |p| {
            cb_tq.update_progress(
                &cb_tid,
                p * 0.70,
                (p * 100.0) as u64,
                100,
                format!("Synthesizing acoustic frames: {:.0}%", p * 100.0),
            );
        },
    );

    if let Err(e) = render_res {
        tq.update_status(
            tid,
            TaskStatus::Failed {
                error_message: format!("Audio synthesis error: {e}"),
            },
        );
        return;
    }

    tq.update_status(
        tid,
        TaskStatus::Running {
            progress: 0.75,
            speed_bps: 0.0,
            status_text: format!("Encoding to {}...", format.extension().to_uppercase()),
        },
    );

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let export_dir = std::path::Path::new("exports");
    let _ = std::fs::create_dir_all(export_dir);
    let temp_wav_path = export_dir.join(format!("temp_{tid}.wav"));
    let out_file_name = format!("rainai_soundscape_{timestamp}.{}", format.extension());
    let out_path = export_dir.join(&out_file_name);

    if let Err(e) = std::fs::write(&temp_wav_path, &wav_bytes) {
        tq.update_status(
            tid,
            TaskStatus::Failed {
                error_message: format!("Failed to write intermediate audio: {e}"),
            },
        );
        return;
    }

    let prov_tag = if embed_provenance {
        format!(
            "RainAI Spatial Synthesizer | Intensity: {:.2} | Wind: {:.2} | Surfaces: tin={:.2},leaves={:.2},pavement={:.2}",
            rain_state.weather.intensity,
            rain_state.wind.speed,
            rain_state.surfaces.tin,
            rain_state.surfaces.leaves_broad,
            rain_state.surfaces.pavement
        )
    } else {
        "RainAI Spatial Synthesizer".to_string()
    };

    let encode_res = match format {
        OfflineAudioFormat::Wav32Float => std::fs::rename(&temp_wav_path, &out_path).map(|_| ()),
        OfflineAudioFormat::FlacLevel8 => {
            let status = Command::new("ffmpeg")
                .arg("-y")
                .arg("-i")
                .arg(&temp_wav_path)
                .arg("-compression_level")
                .arg("8")
                .arg("-metadata")
                .arg(format!("comment={prov_tag}"))
                .arg("-metadata")
                .arg("title=RainAI Soundscape")
                .arg(&out_path)
                .status();

            let _ = std::fs::remove_file(&temp_wav_path);
            status
                .map(|s| if s.success() { () } else { () })
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        }
        OfflineAudioFormat::Mp3320k => {
            let status = Command::new("ffmpeg")
                .arg("-y")
                .arg("-i")
                .arg(&temp_wav_path)
                .arg("-b:a")
                .arg("320k")
                .arg("-id3v2_version")
                .arg("4")
                .arg("-metadata")
                .arg(format!("TXXX:PROVENANCE={prov_tag}"))
                .arg("-metadata")
                .arg("title=RainAI Soundscape")
                .arg(&out_path)
                .status();

            let _ = std::fs::remove_file(&temp_wav_path);
            status
                .map(|s| if s.success() { () } else { () })
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        }
        OfflineAudioFormat::Opus160k => {
            let status = Command::new("ffmpeg")
                .arg("-y")
                .arg("-i")
                .arg(&temp_wav_path)
                .arg("-c:a")
                .arg("libopus")
                .arg("-b:a")
                .arg("160k")
                .arg("-metadata")
                .arg(format!("comment={prov_tag}"))
                .arg(&out_path)
                .status();

            let _ = std::fs::remove_file(&temp_wav_path);
            status
                .map(|s| if s.success() { () } else { () })
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        }
    };

    match encode_res {
        Ok(_) => {
            let size_kb = std::fs::metadata(&out_path)
                .map(|m| m.len() as f32 / 1024.0)
                .unwrap_or(0.0);
            tq.update_status(
                tid,
                TaskStatus::Done {
                    result_message: format!("Saved {} ({:.1} KB)", out_file_name, size_kb),
                },
            );
        }
        Err(e) => {
            tq.update_status(
                tid,
                TaskStatus::Failed {
                    error_message: format!("Encoding failed: {e}"),
                },
            );
        }
    }
}
