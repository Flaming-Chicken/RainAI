//! Modular UI dialogs, warning banners, settings, and background queue docks.

pub mod contribute;
pub mod export_audio;
pub mod help;
pub mod presets;
pub mod privacy;
pub mod provenance;
pub mod queue_dock;
pub mod settings;
pub mod storage;

pub use contribute::*;
pub use export_audio::*;
pub use help::*;
pub use presets::*;
pub use privacy::*;
pub use provenance::*;
pub use queue_dock::*;
pub use settings::*;
pub use storage::*;

use crate::TemplateApp;
use eframe::egui::{self, Color32};
use shared::export_to_compressed_bson;

pub fn render_warning_banners(app: &mut TemplateApp, ctx: &egui::Context) {
    let is_ephemeral = app.storage_diag.is_persisted == Some(false);
    let is_quota = app.storage_diag.quota_exceeded;

    let show_combined = is_ephemeral && is_quota && !app.dismissed_combined_warning;
    let show_ephemeral = is_ephemeral && !app.dismissed_ephemeral_warning;
    let show_quota = is_quota && !app.dismissed_quota_warning;

    if show_combined || show_ephemeral || show_quota {
        let (msg, fill_color, stroke_color, text_color) = if show_combined {
            (
                "Storage Alert: Storage is Ephemeral AND Quota Limit Exceeded!",
                Color32::from_rgba_premultiplied(35, 20, 20, 245),
                Color32::from_rgb(240, 80, 80),
                Color32::from_rgb(255, 120, 120),
            )
        } else if show_ephemeral {
            (
                "Ephemeral Storage: Browser may clear local data under storage pressure.",
                Color32::from_rgba_premultiplied(35, 28, 15, 245),
                Color32::from_rgb(220, 160, 30),
                Color32::from_rgb(255, 200, 80),
            )
        } else {
            (
                "Storage Quota Exceeded: State migrated to IndexedDB fallback tier.",
                Color32::from_rgba_premultiplied(35, 28, 15, 245),
                Color32::from_rgb(220, 160, 30),
                Color32::from_rgb(255, 200, 80),
            )
        };

        egui::Area::new(egui::Id::new("storage_warning_banner_area"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -20.0))
            .show(ctx, |ui| {
                egui::Frame::NONE
                    .fill(fill_color)
                    .stroke(egui::Stroke::new(1.0_f32, stroke_color))
                    .corner_radius(8)
                    .inner_margin(egui::Margin::symmetric(16, 10))
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                            ui.label(egui::RichText::new(msg).strong().color(text_color));
                            ui.add_space(6.0);
                            ui.with_layout(
                                egui::Layout::left_to_right(egui::Align::Center)
                                    .with_main_align(egui::Align::Center),
                                |ui| {
                                    if ui.button("Save .bson Backup").clicked() {
                                        if let Ok(bytes) =
                                            export_to_compressed_bson(&app.state.rain)
                                        {
                                            crate::storage_manager::trigger_binary_download(
                                                "rain_preset_backup.bson",
                                                &bytes,
                                                "application/octet-stream",
                                            );
                                        }
                                    }
                                    if ui.button("Request Persistence").clicked() {
                                        crate::storage_manager::request_persistent_storage();
                                    }
                                    if ui.button("Dismiss").clicked() {
                                        if show_combined {
                                            app.dismissed_combined_warning = true;
                                        } else if show_ephemeral {
                                            app.dismissed_ephemeral_warning = true;
                                        } else {
                                            app.dismissed_quota_warning = true;
                                        }
                                    }
                                },
                            );
                        });
                    });
            });
    }
}

pub fn render_dialogs(app: &mut TemplateApp, ui: &mut egui::Ui) {
    help::render_help_dialog(app, ui);
    settings::render_settings_dialog(app, ui);
    privacy::render_privacy_dialog(app, ui);
    storage::render_storage_dialog(app, ui);
    contribute::render_contribute_dialog(app, ui);
    presets::render_presets_dialog(app, ui);
    provenance::render_provenance_dialog(app, ui);
    export_audio::render_export_audio_dialog(app, ui);

    // Factory reset confirmation dialog
    if app.show_reset_dialog {
        egui::Window::new("Reset Soundscape Parameters?")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.label("Are you sure you want to reset all soundscape parameters to default factory preset?");
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button("Yes, Reset Soundscape").clicked() {
                        app.state.rain = shared::RainState::default();
                        app.rain_view.flow_solver = inference::compute_router::FlowSolverAlgorithm::AdaptiveRk45 {
                            tol: 1e-3,
                            initial_h: 0.1,
                        };
                        app.rain_view.simulated_thermal_level = 0.22;
                        app.settings = crate::settings::SettingsState::default();
                        app.show_reset_dialog = false;
                        app.persist_state();
                    }
                    if ui.button("Cancel").clicked() {
                        app.show_reset_dialog = false;
                    }
                });
            });
    }
}
