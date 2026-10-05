//! Full Soundscape & Dataset Provenance modal dialog.

use crate::TemplateApp;
use eframe::egui::{self, Color32};

pub fn render_provenance_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if !app.show_provenance_dialog {
        return;
    }

    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(380.0, 720.0);
    let win_h = (ui.available_height() - 32.0).clamp(440.0, 720.0);

    egui::Window::new("📜 Full Soundscape & Training Provenance")
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .min_width(360.0)
        .min_height(400.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Attribution & Transparency (XAI)");
                ui.label(
                    "RainAI maps generative and physical acoustic energy back to empirical training datasets and field recordings.",
                );
                ui.add_space(6.0);

                egui::Frame::group(ui.style())
                    .fill(ui.visuals().faint_bg_color)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("🧠 Neural Model Status:");
                            ui.colored_label(Color32::from_rgb(120, 180, 240), "Retraining In Progress (Phase 41)");
                        });
                        ui.label("• Stale pre-trained weights have been flushed to ensure zero attribution leakage.\n• Next-generation checkpoints will feature direct tokenized contributor credit embedding.");
                    });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading("Verified Acoustic Training Corpora");
                ui.label("Empirical recordings referenced by physical fluid dynamics and subtractive filterbanks:");
                ui.add_space(6.0);

                let corpora = [
                    ("Spodeian Studio Recordings", "Flaming Chicken / Liam", "Proprietary / CC-BY-NC-SA 4.0", "Calibrated multi-surface field microphones: Corrugated tin, glass windows, tent canvas, gravel pavement."),
                    ("Ulbrich DSD Drop Spectra", "Ulbrich (1983) / Marshall-Palmer", "Public Domain / Academic", "Empirical gamma distribution governing raindrop diameter spectra from 0.1mm to 7.0mm."),
                    ("Gunn-Kinzer Terminal Velocity", "Gunn & Kinzer (1949)", "Public Domain / Physical", "Non-linear drag and terminal velocity equations for falling water spheres in atmosphere."),
                    ("Minnaert Bubble Acoustics", "Minnaert (1933) / Pumphrey & Crum", "Public Domain / Physical", "Frequency f0 = 3.26/r for underwater bubble resonance and impact cavitation chirps."),
                    ("ISO 140-18 Laboratory Rainfall", "ISO Standardization", "Standard Reference", "Acoustic measurement of rainfall sound on building elements and corrugated roofs."),
                ];

                for (name, author, license, desc) in corpora {
                    egui::Frame::group(ui.style())
                        .fill(ui.visuals().faint_bg_color)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.heading(name);
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.colored_label(Color32::from_rgb(100, 200, 140), license);
                                });
                            });
                            ui.label(egui::RichText::new(format!("Contributor / Source: {author}")).small().strong());
                            ui.label(desc);
                        });
                    ui.add_space(4.0);
                }

                ui.add_space(10.0);
                if ui.button("Close").clicked() {
                    app.show_provenance_dialog = false;
                }
            });
        });

    if !open {
        app.show_provenance_dialog = false;
    }
}
