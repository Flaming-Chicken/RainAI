//! Consolidated Storage Hub, Asset Preloading, and Backup Transfer modal.

use crate::TemplateApp;
use crate::storage_manager::*;
use crate::task_queue::{TaskItem, TaskKind};
use eframe::egui::{self, Color32};
use shared::{export_to_compressed_bson, import_from_compressed_bson};

pub fn render_storage_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if !app.show_storage_modal {
        return;
    }

    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(380.0, 680.0);
    let win_h = (ui.available_height() - 32.0).clamp(480.0, 740.0);

    egui::Window::new("💾 Storage & Backup Management Hub")
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .min_width(360.0)
        .min_height(440.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Section 1: Diagnostics
                ui.heading("Storage Health & Durability");
                ui.add_space(4.0);

                let status_color = match app.storage_diag.is_persisted {
                    Some(true) => Color32::from_rgb(80, 200, 100),
                    Some(false) => Color32::from_rgb(240, 160, 50),
                    None => Color32::GRAY,
                };
                let status_label = match app.storage_diag.is_persisted {
                    Some(true) => "Persistent (Immune to browser eviction)",
                    Some(false) => "Ephemeral (May be cleared if disk is low)",
                    None => "Unknown / Querying...",
                };

                ui.horizontal(|ui| {
                    ui.label("Durability Status:");
                    ui.colored_label(status_color, egui::RichText::new(status_label).strong());
                });

                ui.horizontal(|ui| {
                    ui.label("Active Storage Tier:");
                    ui.label(egui::RichText::new(app.storage_diag.backend.label()).strong());
                });

                ui.horizontal(|ui| {
                    ui.label("PWA Installation:");
                    if app.storage_diag.is_pwa_installed {
                        ui.colored_label(Color32::from_rgb(80, 200, 100), "Installed (Native PWA)");
                    } else if app.storage_diag.pwa_install_available {
                        ui.label("Available to Install");
                    } else {
                        ui.label("Not Available in Browser Tab");
                    }
                });

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if app.storage_diag.is_persisted != Some(true) {
                        if ui.button("Request Persistence").on_hover_text("Ask browser permission to protect state from auto-eviction").clicked() {
                            request_persistent_storage();
                        }
                    }

                    if app.storage_diag.pwa_install_available && !app.storage_diag.is_pwa_installed {
                        if ui.button("Install Web App").on_hover_text("Install to homescreen/desktop for permanent offline durability").clicked() {
                            trigger_pwa_install();
                        }
                    }
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(6.0);

                // Section 2: Asset Preloading
                ui.heading("Asset Preloading");
                ui.label("Preload all acoustic neural weights, HRTF convolution tables, and impulse responses for instant zero-latency offline performance.");
                ui.add_space(6.0);

                egui::Frame::group(ui.style())
                    .fill(ui.visuals().faint_bg_color)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("📦 Embedded Ternary Weights (1.58-bit):");
                            ui.colored_label(Color32::from_rgb(80, 200, 100), "Cached (Embedded)");
                        });
                        ui.horizontal(|ui| {
                            ui.label("🎧 KEMAR Compact HRTF Dataset:");
                            ui.colored_label(Color32::from_rgb(80, 200, 100), "Loaded (In-Memory)");
                        });
                        ui.horizontal(|ui| {
                            ui.label("🏛 Custom Impulse Response Cache:");
                            if let Some(meta) = &app.rain_view.custom_ir_meta {
                                ui.colored_label(Color32::from_rgb(100, 180, 240), &meta.name);
                            } else {
                                ui.label("Default Algorithmic Ambisonics");
                            }
                        });

                        ui.add_space(6.0);
                        if ui.button("⚡ Preload All Assets").on_hover_text("Pre-cache all neural model checkpoints and spatial impulse responses").clicked() {
                            let task = TaskItem::new("preload_assets", "Preload All Acoustic Assets", TaskKind::AssetPreload { asset_id: "all_manifest_v1".into() });
                            app.task_queue.enqueue(task);
                            app.show_task_queue_tray = true;
                        }
                    });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(6.0);

                // Section 3: Backup & Transfer (Export, Import, Reset)
                ui.heading("Backup & Transfer");
                ui.label("Export, import, and backup your soundscape presets across devices.");
                ui.add_space(6.0);

                // Export row
                ui.label(egui::RichText::new("Export Configuration:").strong());
                ui.horizontal(|ui| {
                    if ui.button("📋 Copy JSON to Clipboard").clicked() {
                        if let Ok(json_str) = serde_json::to_string_pretty(&app.state.rain) {
                            ui.ctx().copy_text(json_str);
                            app.export_copied_notification = Some(ui.input(|i| i.time));
                        }
                    }

                    if ui.button("📥 Download .json File").clicked() {
                        if let Ok(json_str) = serde_json::to_string_pretty(&app.state.rain) {
                            trigger_text_download("rain_preset.json", &json_str, "application/json;charset=utf-8");
                        }
                    }

                    if ui.button("💾 Download .bson Backup").clicked() {
                        if let Ok(bytes) = export_to_compressed_bson(&app.state.rain) {
                            trigger_binary_download("rain_preset.bson", &bytes, "application/octet-stream");
                        }
                    }
                });

                if let Some(t) = app.export_copied_notification {
                    if ui.input(|i| i.time) - t < 3.0 {
                        ui.colored_label(Color32::GREEN, "✔ Soundscape configuration copied to clipboard!");
                    }
                }

                ui.add_space(8.0);

                // Import row
                ui.label(egui::RichText::new("Import Configuration:").strong());
                ui.label("Paste JSON / Base64 BSON below, or use clipboard / file upload:");
                ui.add_space(4.0);

                egui::ScrollArea::both().max_height(100.0).show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut app.import_text_buffer)
                            .font(egui::TextStyle::Monospace)
                            .hint_text("Paste JSON or Base64 BSON content here...")
                            .desired_width(f32::INFINITY)
                            .desired_rows(4),
                    );
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("Apply Pasted Import").clicked() {
                        let input = app.import_text_buffer.trim();
                        if input.is_empty() {
                            app.import_result_message = Some(Err("Import text is empty.".to_string()));
                        } else if input.starts_with('{') {
                            if let Ok(rain) = serde_json::from_str::<shared::RainState>(input) {
                                app.state.rain = rain;
                                app.import_result_message = Some(Ok("Successfully imported soundscape preset from JSON!".to_string()));
                                app.persist_state();
                            } else {
                                app.import_result_message = Some(Err("Invalid JSON format.".to_string()));
                            }
                        } else if let Ok(bytes) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, input) {
                            if let Ok(rain) = import_from_compressed_bson::<shared::RainState>(&bytes) {
                                app.state.rain = rain;
                                app.import_result_message = Some(Ok("Successfully imported soundscape preset from compressed BSON!".to_string()));
                                app.persist_state();
                            } else {
                                app.import_result_message = Some(Err("Invalid BSON payload.".to_string()));
                            }
                        } else {
                            app.import_result_message = Some(Err("Unrecognized format. Please provide valid JSON or Base64 BSON.".to_string()));
                        }
                    }

                    if ui.button("Clear Buffer").clicked() {
                        app.import_text_buffer.clear();
                        app.import_result_message = None;
                    }
                });

                if let Some(ref result) = app.import_result_message {
                    ui.add_space(4.0);
                    match result {
                        Ok(msg) => {
                            ui.colored_label(Color32::from_rgb(80, 200, 100), format!("✔ {msg}"));
                        }
                        Err(msg) => {
                            ui.colored_label(Color32::from_rgb(240, 80, 80), format!("✖ {msg}"));
                        }
                    }
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(6.0);

                // Section 4: Factory Reset
                ui.heading("Reset Soundscape");
                ui.label("Restore all physical surface matrices, atmospheric sliders, and flow solvers to factory defaults.");
                ui.add_space(4.0);

                if ui.button("↺ Reset All Parameters to Factory Defaults").clicked() {
                    app.show_reset_dialog = true;
                }

                ui.add_space(8.0);
                if ui.button("Close Hub").clicked() {
                    app.show_storage_modal = false;
                }
            });
        });

    if !open {
        app.show_storage_modal = false;
    }
}
