//! Top navigation bar and global header controls.

use crate::{ScreenConstraints, TemplateApp, storage_manager::*};
use eframe::egui;

pub fn render_navbar(app: &mut TemplateApp, ui: &mut egui::Ui, constraints: &ScreenConstraints) {
    egui::Panel::top("top_panel").show(ui, |ui| {
        ui.add_space(4.0);
        let title_text = if constraints.is_mobile {
            "🌧 RainAI Studio"
        } else {
            "🌧 RainAI · Neural Spatial Soundscape Studio"
        };
        let header_row_height = if constraints.is_mobile { 44.0 } else { 32.0 };

        ui.horizontal(|ui| {
            ui.set_height(header_row_height);

            if constraints.is_mobile {
                ui.label(egui::RichText::new(title_text).size(18.0).strong());
            } else {
                ui.heading(title_text);
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // 1. Theme Switcher
                let theme_text = if constraints.is_mobile {
                    format!("{} {}", app.state.config.theme.icon(), app.state.config.theme.label())
                } else {
                    format!("{} Theme: {}", app.state.config.theme.icon(), app.state.config.theme.label())
                };

                if ui
                    .button(theme_text)
                    .on_hover_text("Cycle visual themes: Dark, Warm Light, High Contrast")
                    .clicked()
                {
                    app.state.config.theme = app.state.config.theme.next();
                    app.persist_state();
                }

                // 2. Contribute Data (web-only Turnstile flow: desktop opens browser at /#contribute)
                let contribute_text = if constraints.is_mobile { "Contribute" } else { "🌧 Contribute" };
                if ui
                    .button(contribute_text)
                    .on_hover_text("Contribute precipitation audio recordings via the web Turnstile portal (/#contribute)")
                    .clicked()
                {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let portal_url = std::env::var("RAINAI_CONTRIBUTE_URL")
                            .or_else(|_| std::env::var("RAINAI_PORTAL_URL"))
                            .unwrap_or_else(|_| "https://rainai.spodeian.trade/#contribute".to_string());
                        ui.ctx().open_url(egui::OpenUrl::new_tab(&portal_url));
                        app.contribute_state.status_message = Some(Ok(format!(
                            "Launched Web Contribution Portal ({portal_url}) in your browser — Turnstile tokens are handled exclusively on web."
                        )));
                    }
                    app.show_contribute_dialog = true;
                }

                // 3. Settings Modal
                let settings_text = if constraints.is_mobile { "⚙ Settings" } else { "⚙ Settings" };
                if ui
                    .button(settings_text)
                    .on_hover_text("Audio listening format, performance profiles, and fine-grained DSP solver settings")
                    .clicked()
                {
                    app.show_settings_dialog = true;
                }

                // 4. Storage & Backups Hub
                let storage_text = match app.storage_diag.is_persisted {
                    Some(true) => if constraints.is_mobile { "💾 Storage (P)" } else { "💾 Storage: Persistent" },
                    Some(false) => if constraints.is_mobile { "💾 Storage (E)" } else { "💾 Storage: Ephemeral" },
                    None => "💾 Storage",
                };
                if ui
                    .button(storage_text)
                    .on_hover_text("Asset preloading, backup export/import (JSON/BSON), and diagnostics")
                    .clicked()
                {
                    app.show_storage_modal = true;
                }

                // 5. Dedicated Offline Audio Export
                let export_audio_text = if constraints.is_mobile { "🎵 Export" } else { "🎵 Export Audio" };
                if ui
                    .button(export_audio_text)
                    .on_hover_text("Render soundscape to FLAC Level 8, MP3 (320k), or WAV with embedded metadata")
                    .clicked()
                {
                    app.show_audio_export_dialog = true;
                }

                // 6. Presets Manager
                let presets_text = if constraints.is_mobile { "📋 Presets" } else { "📋 Presets" };
                if ui
                    .button(presets_text)
                    .on_hover_text("Browse factory presets and create new custom presets")
                    .clicked()
                {
                    app.show_presets_dialog = true;
                }

                // 7. Full Provenance
                if !constraints.is_mobile {
                    if ui
                        .button("📜 Provenance")
                        .on_hover_text("View full training corpus provenance and XAI attribution details")
                        .clicked()
                    {
                        app.show_provenance_dialog = true;
                    }
                }

                // 8. Decoupled Spectrogram Toggle
                let spec_text = if app.rain_view.show_spectrogram {
                    "📊 Spectrogram"
                } else {
                    "📊 Show Spectrogram"
                };
                if ui
                    .button(spec_text)
                    .on_hover_text("Toggle bottom real-time neural spectrogram waterfall panel")
                    .clicked()
                {
                    app.rain_view.show_spectrogram = !app.rain_view.show_spectrogram;
                }

                // 9. Help & Info
                let help_text = if constraints.is_mobile { "Help" } else { "Help" };
                if ui
                    .button(help_text)
                    .on_hover_text("Help, physics architecture & shortcuts")
                    .clicked()
                {
                    app.show_help_dialog = true;
                }

                // 10. Privacy Policy
                if !constraints.is_mobile {
                    if ui
                        .button("Privacy")
                        .on_hover_text("Privacy policy and contribution terms")
                        .clicked()
                    {
                        app.show_privacy_dialog = true;
                    }
                }

                // PWA Install Prompt Button
                if app.storage_diag.pwa_install_available && !app.storage_diag.is_pwa_installed {
                    if ui
                        .button("Install App")
                        .on_hover_text("Install application for permanent offline storage")
                        .clicked()
                    {
                        trigger_pwa_install();
                    }
                }
            });
        });
        ui.add_space(4.0);
    });
}
