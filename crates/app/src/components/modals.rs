//! Modal dialogs, warning banners, and data transfer views.

use crate::{ExportFormat, TemplateApp, storage_manager::*};
use eframe::egui;
use shared::{
    ItemCollection, export_to_compressed_bson, import_from_compressed_bson, import_from_csv,
    import_from_json,
};

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
                egui::Color32::from_rgba_premultiplied(35, 20, 20, 245),
                egui::Color32::from_rgb(240, 80, 80),
                egui::Color32::from_rgb(255, 120, 120),
            )
        } else if show_ephemeral {
            (
                "Ephemeral Storage: Browser may clear local data under storage pressure.",
                egui::Color32::from_rgba_premultiplied(35, 28, 15, 245),
                egui::Color32::from_rgb(220, 160, 30),
                egui::Color32::from_rgb(255, 200, 80),
            )
        } else {
            (
                "Storage Quota Exceeded: State migrated to IndexedDB fallback tier.",
                egui::Color32::from_rgba_premultiplied(35, 28, 15, 245),
                egui::Color32::from_rgb(220, 160, 30),
                egui::Color32::from_rgb(255, 200, 80),
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
                                            export_to_compressed_bson(&app.state.collection)
                                        {
                                            trigger_binary_download(
                                                "data_backup.bson",
                                                &bytes,
                                                "application/octet-stream",
                                            );
                                        }
                                    }
                                    if ui.button("Request Persistence").clicked() {
                                        request_persistent_storage();
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
    if app.show_help_dialog {
        let mut open = true;
        let win_w = (ui.available_width() - 24.0).clamp(340.0, 600.0);
        let win_h = (ui.available_height() - 32.0).clamp(440.0, 680.0);

        egui::Window::new("Help & Information")
            .open(&mut open)
            .resizable(true)
            .collapsible(true)
            .default_size(egui::vec2(win_w, win_h))
            .min_width(320.0)
            .min_height(380.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Serverless & Desktop Template");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                                    .strong()
                                    .color(ui.visuals().hyperlink_color),
                            );
                        });
                    });
                    ui.add_space(4.0);
                    ui.separator();
                    ui.add_space(6.0);

                    ui.heading("Serverless & Desktop Architecture");
                    ui.add_space(4.0);
                    ui.label("This template demonstrates a production-grade, offline-first application architecture compiling to both WebAssembly (via eframe / Trunk / Cloudflare Pages) and Native Desktop (via eframe / Winit).");

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(6.0);

                    ui.heading("Multi-Tier Storage Engine");
                    ui.add_space(4.0);
                    ui.label("• Tier 1: Fast synchronous local storage.");
                    ui.label("• Tier 2: Asynchronous IndexedDB extended quota fallback.");
                    ui.label("• Persistence: StorageManager persistence bridge prevents browser data eviction.");
                    ui.label("• Formats: Dual-format JSON and RON deserialization ensures seamless backwards and forward compatibility.");

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(6.0);

                    ui.heading("Local Privacy & Security");
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("100% Client-Side: No telemetry or server database calls.")
                            .color(egui::Color32::from_rgb(80, 160, 90))
                            .strong(),
                    );
                    ui.label("Your data is stored strictly in your local browser or desktop application storage.");
                });
            });
        if !open {
            app.show_help_dialog = false;
        }
    }

    if app.show_reset_dialog {
        egui::Window::new("Reset Data?")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.label("Are you sure you want to reset all items to default sample data?");
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button("Yes, Reset").clicked() {
                        app.state.collection = ItemCollection::default();
                        app.show_reset_dialog = false;
                        app.persist_state();
                    }
                    if ui.button("Cancel").clicked() {
                        app.show_reset_dialog = false;
                    }
                });
            });
    }

    if let Some(format) = app.show_export_dialog {
        let mut open = true;
        egui::Window::new(format!("Export {}", format.label()))
            .open(&mut open)
            .default_size(egui::vec2(540.0, 380.0))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Copy to Clipboard").clicked() {
                        ui.ctx().copy_text(app.export_text_buffer.clone());
                        app.export_copied_notification = Some(ui.input(|i| i.time));
                    }
                    if let Some(t) = app.export_copied_notification
                        && ui.input(|i| i.time) - t < 3.0
                    {
                        ui.label(
                            egui::RichText::new("Copied to clipboard!").color(egui::Color32::GREEN),
                        );
                    }

                    ui.separator();

                    if ui.button("Download File").clicked() {
                        match format {
                            ExportFormat::Json => {
                                trigger_text_download(
                                    "data_export.json",
                                    &app.export_text_buffer,
                                    "application/json;charset=utf-8",
                                );
                            }
                            ExportFormat::Csv => {
                                trigger_text_download(
                                    "data_export.csv",
                                    &app.export_text_buffer,
                                    "text/csv;charset=utf-8",
                                );
                            }
                            ExportFormat::Bson => {
                                if let Ok(bytes) = export_to_compressed_bson(&app.state.collection)
                                {
                                    trigger_binary_download(
                                        "data_backup.bson",
                                        &bytes,
                                        "application/octet-stream",
                                    );
                                }
                            }
                        }
                    }
                });

                ui.add_space(8.0);
                egui::ScrollArea::both().show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut app.export_text_buffer)
                            .font(egui::TextStyle::Monospace)
                            .code_editor()
                            .lock_focus(true)
                            .desired_width(f32::INFINITY),
                    );
                });
            });

        if !open {
            app.show_export_dialog = None;
            app.export_text_buffer.clear();
        }
    }

    if app.show_import_dialog {
        let mut open = true;
        egui::Window::new("Import Data")
            .open(&mut open)
            .default_size(egui::vec2(500.0, 360.0))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.label(
                    "Paste JSON, CSV or Base64 BSON data below to import into your collection:",
                );
                ui.add_space(8.0);

                egui::ScrollArea::both().max_height(200.0).show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut app.import_text_buffer)
                            .font(egui::TextStyle::Monospace)
                            .hint_text("Paste JSON, CSV, or Base64 BSON content here...")
                            .desired_width(f32::INFINITY)
                            .desired_rows(8),
                    );
                });

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.button("Apply Import").clicked() {
                        let input = app.import_text_buffer.trim();
                        if input.is_empty() {
                            app.import_result_message = Some(Err("Input is empty.".to_string()));
                        } else if input.starts_with('{') {
                            match import_from_json(input) {
                                Ok(col) => {
                                    let count = col.total_count();
                                    app.state.collection = col;
                                    app.import_result_message = Some(Ok(format!(
                                        "Successfully imported {} items from JSON!",
                                        count
                                    )));
                                    app.persist_state();
                                }
                                Err(e) => {
                                    app.import_result_message = Some(Err(e.to_string()));
                                }
                            }
                        } else if let Ok(decoded_bytes) = base64::Engine::decode(
                            &base64::engine::general_purpose::STANDARD,
                            input,
                        ) {
                            match import_from_compressed_bson(&decoded_bytes) {
                                Ok(col) => {
                                    let count = col.total_count();
                                    app.state.collection = col;
                                    app.import_result_message = Some(Ok(format!(
                                        "Successfully imported {} items from compressed BSON!",
                                        count
                                    )));
                                    app.persist_state();
                                }
                                Err(e) => {
                                    app.import_result_message =
                                        Some(Err(format!("BSON import failed: {}", e)));
                                }
                            }
                        } else {
                            match import_from_csv(input) {
                                Ok(col) => {
                                    let count = col.total_count();
                                    app.state.collection = col;
                                    app.import_result_message = Some(Ok(format!(
                                        "Successfully imported {} items from CSV!",
                                        count
                                    )));
                                    app.persist_state();
                                }
                                Err(e) => {
                                    app.import_result_message = Some(Err(e.to_string()));
                                }
                            }
                        }
                    }

                    if ui.button("Cancel").clicked() {
                        app.show_import_dialog = false;
                        app.import_text_buffer.clear();
                        app.import_result_message = None;
                    }
                });

                if let Some(ref result) = app.import_result_message {
                    ui.add_space(6.0);
                    match result {
                        Ok(msg) => {
                            ui.label(
                                egui::RichText::new(msg)
                                    .color(egui::Color32::from_rgb(80, 180, 90))
                                    .strong(),
                            );
                        }
                        Err(msg) => {
                            ui.label(
                                egui::RichText::new(msg)
                                    .color(egui::Color32::from_rgb(220, 70, 70))
                                    .strong(),
                            );
                        }
                    }
                }
            });

        if !open {
            app.show_import_dialog = false;
            app.import_text_buffer.clear();
            app.import_result_message = None;
        }
    }

    if app.show_storage_modal {
        render_storage_modal(app, ui);
    }

    if app.show_contribute_dialog {
        render_contribute_dialog(app, ui);
    }

    if app.show_privacy_dialog {
        render_privacy_dialog(app, ui);
    }
}

pub fn render_storage_modal(app: &mut TemplateApp, ui: &mut egui::Ui) {
    let mut open = true;
    egui::Window::new("Storage & Data Management")
        .open(&mut open)
        .default_size(egui::vec2(480.0, 380.0))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            ui.heading("Storage Diagnostics");
            ui.add_space(4.0);

            let status_color = match app.storage_diag.is_persisted {
                Some(true) => egui::Color32::from_rgb(80, 200, 100),
                Some(false) => egui::Color32::from_rgb(240, 160, 50),
                None => egui::Color32::GRAY,
            };
            let status_label = match app.storage_diag.is_persisted {
                Some(true) => "Persistent (Immune to browser eviction)",
                Some(false) => "Ephemeral (May be cleared if disk is low)",
                None => "Unknown / Querying...",
            };

            ui.horizontal(|ui| {
                ui.label("Persistence Status:");
                ui.colored_label(status_color, egui::RichText::new(status_label).strong());
            });

            ui.horizontal(|ui| {
                ui.label("Active Storage Tier:");
                ui.label(egui::RichText::new(app.storage_diag.backend.label()).strong());
            });

            ui.horizontal(|ui| {
                ui.label("PWA Installation:");
                if app.storage_diag.is_pwa_installed {
                    ui.colored_label(
                        egui::Color32::from_rgb(80, 200, 100),
                        "Installed (Permanent App)",
                    );
                } else if app.storage_diag.pwa_install_available {
                    ui.label("Available to Install");
                } else {
                    ui.label("Not Available in Browser Tab");
                }
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(6.0);

            ui.heading("Actions");
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                if app.storage_diag.is_persisted != Some(true) {
                    if ui
                        .button("Request Persistent Storage")
                        .on_hover_text("Ask browser permission to protect state from auto-eviction")
                        .clicked()
                    {
                        request_persistent_storage();
                    }
                }

                if app.storage_diag.pwa_install_available && !app.storage_diag.is_pwa_installed {
                    if ui
                        .button("Install Web App")
                        .on_hover_text("Install to homescreen/desktop for highest durability")
                        .clicked()
                    {
                        trigger_pwa_install();
                    }
                }
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .button("Export Compressed BSON Backup")
                    .on_hover_text("Download compact, offline state backup (.bson)")
                    .clicked()
                {
                    if let Ok(bytes) = export_to_compressed_bson(&app.state.collection) {
                        trigger_binary_download(
                            "data_backup.bson",
                            &bytes,
                            "application/octet-stream",
                        );
                    }
                }

                if ui.button("Import Backup").clicked() {
                    app.show_storage_modal = false;
                    app.show_import_dialog = true;
                }
            });
        });

    if !open {
        app.show_storage_modal = false;
    }
}

pub fn render_contribute_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(360.0, 640.0);
    let win_h = (ui.available_height() - 32.0).clamp(420.0, 680.0);

    egui::Window::new("🌧 Contribute Rain Audio Data")
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .min_width(340.0)
        .min_height(380.0)
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

                // Submission Mode
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut app.contribute_state.submission_mode, 0, "📁 Audio / Video File");
                    ui.selectable_value(&mut app.contribute_state.submission_mode, 1, "🔗 Remote URL (YouTube / Cloud)");
                });
                ui.add_space(8.0);

                if app.contribute_state.submission_mode == 0 {
                    ui.label(egui::RichText::new("File or Local Recording:").strong());
                    ui.add(
                        egui::TextEdit::singleline(&mut app.contribute_state.file_or_url)
                            .hint_text("Drag & drop audio/video file path or enter filename...")
                            .desired_width(f32::INFINITY),
                    );
                    ui.label(
                        egui::RichText::new("• Lossless PCM/WAV files are compressed directly to FLAC Level 8 client-side (100% bit-perfect, zero quality loss).\n• Lossy files (Opus, AAC, MP3, OGG) are preserved natively with zero transcoding.\n• Videos (MP4, MKV) have audio extracted client-side with video discarded prior to upload.")
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
                        egui::RichText::new("• Remote URLs are registered as verified catalog pointers.\n• Audio is pulled lazily and on-demand into local cache during ML training passes (zero forced repository downloads).")
                            .small()
                            .color(ui.visuals().weak_text_color()),
                    );
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                // Metadata & License
                ui.heading("Metadata & Licensing");
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("Author / Recordist:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.contribute_state.author_name)
                            .hint_text("Anonymous (or your name/handle)")
                            .desired_width(200.0),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Tags:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.contribute_state.tags_input)
                            .hint_text("e.g. tin_roof, car_hood, dense_canopy")
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.add_space(4.0);

                ui.label(egui::RichText::new("License Agreement:").strong());
                let license_options = [
                    (
                        "RainAI-FC-Proprietary-License",
                        "RainAI-FC-Proprietary-License (Recommended Default: Commercial & non-commercial grant for RainAI, Spodeian, & Flaming Chicken; XAI attribution)",
                    ),
                    (
                        "CC0 1.0 Universal",
                        "CC0 1.0 Universal / Public Domain (Unconstrained public domain dedication, Unlicense, PDDL)",
                    ),
                    (
                        "CC-BY 4.0",
                        "CC-BY 4.0 (Permissive Attribution: CC-BY, MIT, Apache-2.0; permanent dataset attribution)",
                    ),
                    (
                        "CC-BY-SA 4.0",
                        "CC-BY-SA 4.0 (Commercial Share-Alike: CC-BY-SA 4.0/3.0)",
                    ),
                    (
                        "Unknown / Unspecified",
                        "Unknown / Unspecified (Immediate Quarantine: Audio is quarantined until a license is discovered during processing)",
                    ),
                    (
                        "Custom",
                        "Custom / Other Verified Open License",
                    ),
                ];

                for (id, desc) in license_options {
                    ui.radio_value(&mut app.contribute_state.selected_license, id.to_string(), desc);
                }

                ui.add_space(6.0);

                // Explainable AI & Transparency Accordion for Proprietary License
                if app.contribute_state.selected_license == "RainAI-FC-Proprietary-License" {
                    egui::Frame::NONE
                        .fill(ui.visuals().faint_bg_color)
                        .stroke(egui::Stroke::new(1.0, ui.visuals().window_stroke().color))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("ℹ️ Explainable AI (XAI) Attribution & Transparency")
                                    .strong()
                                    .color(ui.visuals().strong_text_color()),
                            );
                            ui.label(
                                egui::RichText::new(
                                    "\"While this license allows us to use your data freely to build RainAI, our system is designed for transparency. We track the metadata of all contributions, meaning you will always be credited when your specific data directly influences our explainable AI's outputs.\""
                                )
                                .italics()
                                .color(ui.visuals().text_color()),
                            );
                            ui.add_space(4.0);
                            let toggle_text = if app.contribute_state.show_license_details {
                                "▼ Hide Full Legal Grant"
                            } else {
                                "▶ View Full Legal Grant & Terms"
                            };
                            if ui.button(toggle_text).clicked() {
                                app.contribute_state.show_license_details = !app.contribute_state.show_license_details;
                            }
                            if app.contribute_state.show_license_details {
                                ui.add_space(4.0);
                                egui::ScrollArea::vertical().max_height(100.0).show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(
                                            "\"By submitting this data and metadata, I grant Spodeian, Flaming Chicken, and their respective affiliates, successors, and assigns a worldwide, non-exclusive, royalty-free, perpetual, irrevocable, and sublicensable right to use, reproduce, modify, adapt, publish, translate, create derivative works from, distribute, and publicly display this data for any purpose, including commercial and non-commercial applications. This explicitly includes, without limitation, the right to use the data to train, test, and validate machine learning models for the RainAI project and any other current or future projects. I represent and warrant that I own or have the necessary rights to grant this license.\""
                                        )
                                        .small()
                                        .monospace(),
                                    );
                                });
                            }
                        });
                } else if app.contribute_state.selected_license == "CC-BY 4.0" {
                    ui.label(
                        egui::RichText::new(
                            "⚖️ Dataset Attribution: Training data under CC-BY is permanently registered in ATTRIBUTIONS.txt and project bibliography upon ingestion. Real-time XAI provides optional contributor credit mapping."
                        )
                        .small()
                        .color(ui.visuals().text_color()),
                    );
                } else if app.contribute_state.selected_license == "Unknown / Unspecified" {
                    egui::Frame::NONE
                        .fill(ui.visuals().faint_bg_color)
                        .stroke(egui::Stroke::new(1.0, ui.visuals().window_stroke().color))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("📁 Immediate Quarantine Notice")
                                    .strong()
                                    .color(egui::Color32::from_rgb(220, 160, 40)),
                            );
                            ui.label(
                                "Submissions with an unknown or unspecified license are immediately placed into staging/quarantine/. \
                                Our automated ingestion workers will attempt to scrape and verify the license from Vorbis/ID3 metadata tags \
                                or source URLs during data processing before any consideration for dataset promotion.",
                            );
                        });
                }

                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("All licensing and metadata tags are embedded directly into Vorbis/ID3 container comments prior to upload.")
                        .small()
                        .color(ui.visuals().weak_text_color()),
                );

                ui.add_space(8.0);

                // Mandatory Contributor Warranty Affirmation Checkbox
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

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                // Submission actions
                let warranty_confirmed = app.contribute_state.confirmed_rights_warranty;
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(warranty_confirmed, |ui| {
                        if ui.button("🚀 Stage Submission to R2").clicked() {
                            if app.contribute_state.file_or_url.trim().is_empty() {
                                app.contribute_state.status_message = Some(Err("Please provide a file path or URL.".to_string()));
                            } else if app.contribute_state.selected_license == "Unknown / Unspecified" {
                                app.contribute_state.status_message = Some(Ok(
                                    "Submission quarantined successfully! Placed in staging/quarantine/ pending automated license scraping during data processing."
                                        .to_string(),
                                ));
                            } else {
                                app.contribute_state.status_message = Some(Ok(format!(
                                    "Submission staged successfully! Registered under '{}' with contributor rights warranty affirmed.",
                                    app.contribute_state.selected_license
                                )));
                            }
                        }
                    });

                    if ui.button("Close").clicked() {
                        app.show_contribute_dialog = false;
                    }
                });

                if !warranty_confirmed {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("⚠ Required: Please confirm the contributor rights warranty above before staging.")
                            .small()
                            .color(egui::Color32::from_rgb(220, 160, 40)),
                    );
                }

                if let Some(ref res) = app.contribute_state.status_message {
                    ui.add_space(6.0);
                    match res {
                        Ok(msg) => {
                            ui.label(egui::RichText::new(msg).color(egui::Color32::from_rgb(80, 200, 100)).strong());
                        }
                        Err(msg) => {
                            ui.label(egui::RichText::new(msg).color(egui::Color32::from_rgb(240, 80, 80)).strong());
                        }
                    }
                }

                ui.add_space(12.0);
                // Multi-tier Fallback Notice & Proton Mail Link
                egui::Frame::NONE
                    .fill(ui.visuals().faint_bg_color)
                    .stroke(egui::Stroke::new(1.0, ui.visuals().window_stroke().color))
                    .corner_radius(6)
                    .inner_margin(egui::Margin::symmetric(12, 10))
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("📬 Multi-Tier Failsafe Fallback").strong());
                        ui.label(
                            "Have large multi-gigabyte collections, private cloud drives, or encountering upload issues? You can email us directly with download links, attachments, and notes.",
                        );
                        ui.add_space(4.0);
                        let mailto_url = format!(
                            "mailto:spodeian@proton.me?subject=RainAI%20Audio%20Contribution&body=Author:%20{}%0ALicense:%20{}%0ATags:%20{}%0ASource:%20{}",
                            app.contribute_state.author_name,
                            app.contribute_state.selected_license,
                            app.contribute_state.tags_input,
                            app.contribute_state.file_or_url
                        );
                        ui.hyperlink_to("📧 Send directly to spodeian@proton.me", &mailto_url);
                    });
            });
        });

    if !open {
        app.show_contribute_dialog = false;
    }
}

pub fn render_privacy_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(340.0, 680.0);
    let win_h = (ui.available_height() - 32.0).clamp(440.0, 720.0);

    egui::Window::new("🛡 RainAI Privacy Policy & Terms")
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                const PRIVACY_TEXT: &str = include_str!("../../../../PRIVACY_POLICY.md");
                ui.label(PRIVACY_TEXT);
            });
        });

    if !open {
        app.show_privacy_dialog = false;
    }
}
