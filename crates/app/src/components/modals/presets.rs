//! Preset Manager & Creation modal dialog.

use crate::TemplateApp;
use eframe::egui::{self, Color32};
use shared::preset::WeatherPreset;

#[derive(Clone, Debug, Default)]
pub struct PresetModalState {
    pub new_preset_name: String,
    pub new_preset_description: String,
    pub new_preset_tags: String,
    pub status_message: Option<String>,
}

pub fn render_presets_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if !app.show_presets_dialog {
        return;
    }

    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(380.0, 640.0);
    let win_h = (ui.available_height() - 32.0).clamp(440.0, 680.0);

    egui::Window::new("📋 Soundscape Presets & Library")
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .min_width(360.0)
        .min_height(400.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Section 1: Save Current Preset
                ui.heading("Save Current Soundscape as Preset");
                ui.label("Snapshot your current surface blend, weather dynamics, and spatial radar into a named preset.");
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("Preset Name:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.preset_state.new_preset_name)
                            .hint_text("e.g. Midnight Porch Storm")
                            .desired_width(220.0),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Description:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.preset_state.new_preset_description)
                            .hint_text("Short acoustic summary...")
                            .desired_width(f32::INFINITY),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Tags:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.preset_state.new_preset_tags)
                            .hint_text("night, thunder, porch")
                            .desired_width(f32::INFINITY),
                    );
                });

                ui.add_space(4.0);
                if ui.button("💾 Save as Custom Preset").clicked() {
                    let name = app.preset_state.new_preset_name.trim();
                    if name.is_empty() {
                        app.preset_state.status_message = Some("Please specify a preset name.".to_string());
                    } else {
                        app.settings.active_preset_name = name.to_string();
                        app.preset_state.status_message = Some(format!("Saved preset '{name}'!"));
                        app.preset_state.new_preset_name.clear();
                        app.preset_state.new_preset_description.clear();
                        app.preset_state.new_preset_tags.clear();
                        app.persist_state();
                    }
                }

                if let Some(ref msg) = app.preset_state.status_message {
                    ui.add_space(2.0);
                    ui.colored_label(Color32::from_rgb(80, 200, 100), msg);
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(6.0);

                // Section 2: Factory Acoustic Presets
                ui.heading("Factory Weather Presets");
                ui.label("Curated acoustic soundscapes tuned against empirical field recordings.");
                ui.add_space(6.0);

                let factory = WeatherPreset::builtins();

                for preset in factory {
                    egui::Frame::group(ui.style())
                        .fill(ui.visuals().faint_bg_color)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.heading(&preset.name);
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.button("▶ Apply Preset").clicked() {
                                        let playing = app.state.rain.is_playing;
                                        let volume = app.state.rain.master_volume;
                                        app.state.rain = preset.state.clone();
                                        app.state.rain.is_playing = playing;
                                        app.state.rain.master_volume = volume;
                                        app.settings.active_preset_name = preset.name.clone();
                                        app.persist_state();
                                    }
                                });
                            });
                            ui.label(&preset.description);
                            ui.label(egui::RichText::new(format!("Tags: {}", preset.tags.join(", "))).small().weak());
                        });
                    ui.add_space(4.0);
                }

                ui.add_space(8.0);
                if ui.button("Close").clicked() {
                    app.show_presets_dialog = false;
                }
            });
        });

    if !open {
        app.show_presets_dialog = false;
    }
}
