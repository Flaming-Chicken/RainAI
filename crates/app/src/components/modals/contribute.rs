//! Community Data Contribution modal dialog.

use crate::TemplateApp;
use crate::task_queue::{TaskItem, TaskKind};
use eframe::egui::{self, Color32};

pub fn render_contribute_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if !app.show_contribute_dialog {
        return;
    }

    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(380.0, 700.0);
    let win_h = (ui.available_height() - 32.0).clamp(460.0, 760.0);

    egui::Window::new("🌧 Contribute Rain Audio Data")
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .min_width(360.0)
        .min_height(420.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Community Rain Audio Dataset Ingestion");
                ui.label(
                    "Help train RainAI's physical flow-matching models by contributing your own rainfall audio recordings.",
                );
                ui.add_space(6.0);
                ui.separator();
                ui.add_space(6.0);

                // Section 1: Submission Mode & Source
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut app.contribute_state.submission_mode, 0, "📁 Audio / Video File");
                    ui.selectable_value(&mut app.contribute_state.submission_mode, 1, "🔗 Remote URL (YouTube / Freesound)");
                });
                ui.add_space(6.0);

                if app.contribute_state.submission_mode == 0 {
                    ui.label(egui::RichText::new("File or Local Recording Path:").strong());
                    ui.add(
                        egui::TextEdit::singleline(&mut app.contribute_state.file_or_url)
                            .hint_text("Drag & drop audio/video file path or enter filename...")
                            .desired_width(f32::INFINITY),
                    );
                    ui.label(
                        egui::RichText::new("• Lossless PCM/WAV files are compressed to FLAC Level 8 client-side.\n• Lossy files (Opus, MP3, OGG) are preserved with zero transcoding.\n• Videos (MP4, MKV) have audio extracted client-side with video discarded prior to staging.")
                            .small()
                            .color(ui.visuals().weak_text_color()),
                    );
                } else {
                    ui.label(egui::RichText::new("Remote Source URL:").strong());
                    ui.add(
                        egui::TextEdit::singleline(&mut app.contribute_state.file_or_url)
                            .hint_text("https://youtube.com/watch?v=... or Freesound / Google Drive link")
                            .desired_width(f32::INFINITY),
                    );
                    ui.label(
                        egui::RichText::new("• Remote URLs are registered as verified catalog pointers.\n• Audio is pulled lazily during ML training passes.")
                            .small()
                            .color(ui.visuals().weak_text_color()),
                    );
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(6.0);

                // Section 2: Authors & Attributions (Dynamic multiple authors + attribution file)
                ui.heading("Authors & Attributions");
                ui.label("List each contributor separately, and optionally attach an attribution file.");
                ui.add_space(4.0);

                let author_count = app.contribute_state.authors.len();
                let mut to_remove = None;
                for (idx, author) in app.contribute_state.authors.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("Author #{}:", idx + 1));
                        ui.add(
                            egui::TextEdit::singleline(author)
                                .hint_text("e.g. Liam (spodeian) or Recording Studio")
                                .desired_width(240.0),
                        );
                        if author_count > 1 {
                            if ui.button("✖").on_hover_text("Remove this author").clicked() {
                                to_remove = Some(idx);
                            }
                        }
                    });
                }
                if let Some(idx) = to_remove {
                    app.contribute_state.authors.remove(idx);
                }

                if ui.button("➕ Add Author").on_hover_text("Add another author/recordist text box").clicked() {
                    app.contribute_state.authors.push(String::new());
                }

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Attribution File (Optional):");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.contribute_state.attribution_file_path)
                            .hint_text("Path to ATTRIBUTIONS.txt or credits file...")
                            .desired_width(f32::INFINITY),
                    );
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(6.0);

                // Section 3: Tags & Description (Inputs + File uploads)
                ui.heading("Metadata & Description");
                ui.label("User metadata is merged additively with any tags scraped from the audio file or link.");
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("Tags:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.contribute_state.tags_input)
                            .hint_text("tin_roof, rain_texture, thunder, forest")
                            .desired_width(f32::INFINITY),
                    );
                });

                ui.add_space(4.0);
                ui.label("Description / Field Notes:");
                ui.add(
                    egui::TextEdit::multiline(&mut app.contribute_state.description_input)
                        .hint_text("Describe recording environment, microphone setup, weather context...")
                        .desired_width(f32::INFINITY)
                        .desired_rows(3),
                );

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Description / Metadata File (Optional):");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.contribute_state.description_file_path)
                            .hint_text("Path to description.txt or metadata.json...")
                            .desired_width(f32::INFINITY),
                    );
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(6.0);

                // Section 4: Licensing
                ui.heading("Licensing Agreement");
                ui.add_space(4.0);

                let license_options = [
                    (
                        "RainAI-FC-Proprietary-License",
                        "RainAI-FC-Proprietary-License (Recommended Default: Commercial & non-commercial grant for RainAI; XAI attribution)",
                    ),
                    (
                        "CC0 1.0 Universal",
                        "CC0 1.0 Universal / Public Domain (Unconstrained public domain dedication)",
                    ),
                    (
                        "CC-BY 4.0",
                        "CC-BY 4.0 (Permissive Attribution: permanent dataset attribution)",
                    ),
                    (
                        "CC-BY-SA 4.0",
                        "CC-BY-SA 4.0 (Commercial Share-Alike: CC-BY-SA 4.0/3.0)",
                    ),
                    (
                        "Unknown / Unspecified",
                        "Unknown / Unspecified (Immediate Quarantine: Audio is quarantined until verified)",
                    ),
                    (
                        "Custom",
                        "Custom / Other License (Provide text or license file below)",
                    ),
                ];

                for (id, desc) in license_options {
                    ui.radio_value(&mut app.contribute_state.selected_license, id.to_string(), desc);
                }

                // Custom license handling
                if app.contribute_state.selected_license == "Custom" {
                    ui.add_space(6.0);
                    egui::Frame::group(ui.style())
                        .fill(ui.visuals().faint_bg_color)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Custom License Specification:").strong());
                            ui.add_space(2.0);
                            ui.add(
                                egui::TextEdit::multiline(&mut app.contribute_state.custom_license_text)
                                    .hint_text("Type custom license terms or grant text here...")
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(3),
                            );
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label("Or Submit License File:");
                                ui.add(
                                    egui::TextEdit::singleline(&mut app.contribute_state.custom_license_file_path)
                                        .hint_text("Path to LICENSE.txt...")
                                        .desired_width(f32::INFINITY),
                                );
                            });
                        });
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(6.0);

                // Section 5: Rights Warranty & Auto-Proceed Toggle
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(
                        &mut app.contribute_state.confirmed_rights_warranty,
                        egui::RichText::new("I represent and warrant that I have the right to provide and license this audio, and agree to the")
                            .strong(),
                    );
                    if ui.link("Privacy Policy").clicked() {
                        app.show_privacy_dialog = true;
                    }
                    ui.label(".");
                });

                ui.add_space(4.0);
                ui.checkbox(
                    &mut app.contribute_state.auto_proceed_after_processing,
                    "Automatically proceed with upload after processing completes (uncheck to review & finalise first)",
                );

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                // Section 6: Action Buttons
                let warranty_confirmed = app.contribute_state.confirmed_rights_warranty;
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(warranty_confirmed, |ui| {
                        if ui.button("🚀 Queue for Staging").clicked() {
                            if app.contribute_state.file_or_url.trim().is_empty() {
                                app.contribute_state.status_message = Some(Err("Please specify an audio file path or remote URL.".to_string()));
                            } else {
                                let title = if app.contribute_state.submission_mode == 0 {
                                    format!("Upload: {}", app.contribute_state.file_or_url)
                                } else {
                                    format!("Remote Catalog: {}", app.contribute_state.file_or_url)
                                };
                                let task = TaskItem::new("contrib_task", title, TaskKind::DataContribution { title: app.contribute_state.file_or_url.clone() });
                                app.task_queue.enqueue(task);
                                app.show_task_queue_tray = true;
                                app.contribute_state.status_message = Some(Ok("Task enqueued! Track progress in the background queue at the bottom right.".to_string()));
                            }
                        }
                    });

                    if ui.button("🛡 View Privacy Policy").clicked() {
                        app.show_privacy_dialog = true;
                    }

                    if ui.button("Close").clicked() {
                        app.show_contribute_dialog = false;
                    }
                });

                if !warranty_confirmed {
                    ui.add_space(4.0);
                    ui.colored_label(Color32::from_rgb(240, 160, 40), "⚠ Required: Please confirm the contributor rights warranty above before staging.");
                }

                if let Some(ref res) = app.contribute_state.status_message {
                    ui.add_space(6.0);
                    match res {
                        Ok(msg) => {
                            ui.colored_label(Color32::from_rgb(80, 200, 100), format!("✔ {msg}"));
                        }
                        Err(msg) => {
                            ui.colored_label(Color32::from_rgb(240, 80, 80), format!("✖ {msg}"));
                        }
                    }
                }
            });
        });

    if !open {
        app.show_contribute_dialog = false;
    }
}
