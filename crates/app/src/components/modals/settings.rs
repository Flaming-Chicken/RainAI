//! Settings & Engine Configuration modal dialog.

use crate::TemplateApp;
use audio::derivation::ListeningSetup;
use eframe::egui::{self, Color32};
use inference::compute_router::FlowSolverAlgorithm;
use shared::rain::{GovernorOptimizationProfile, QualityTier};

pub fn render_settings_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if !app.show_settings_dialog {
        return;
    }

    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(380.0, 640.0);
    let win_h = (ui.available_height() - 32.0).clamp(460.0, 720.0);

    egui::Window::new("⚙ RainAI Settings & Performance")
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .min_width(360.0)
        .min_height(420.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Listening Setup");
                ui.label("Configures spatial Ambisonic decoding and device latency buffers.");
                ui.add_space(4.0);

                let current_setup = app.settings.listening_setup;
                egui::ComboBox::from_id_salt("settings_listening_setup_selector")
                    .selected_text(current_setup.label())
                    .show_ui(ui, |ui| {
                        let setups = [
                            ListeningSetup::DesktopSpeakers,
                            ListeningSetup::BinauralHeadphones,
                            ListeningSetup::WirelessBluetooth,
                            ListeningSetup::Surround71,
                            ListeningSetup::RawFoaPassthrough,
                        ];
                        for s in setups {
                            if ui.selectable_label(app.settings.listening_setup == s, s.label()).clicked() {
                                app.settings.listening_setup = s;
                                app.rain_view.decode_mode = s.to_decode_mode();
                                if let Some(audio) = &app.audio_state {
                                    audio.set_decode_mode(s.to_decode_mode());
                                }
                            }
                        }
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading("Performance & Optimization Profile");
                ui.label("Controls real-time DSP quality, solver tolerance, and buffer headroom.");
                ui.add_space(4.0);

                let profiles = [
                    (GovernorOptimizationProfile::BalancedAdaptive, "Balanced Adaptive (Recommended)", "Optimal blend of battery life and fluid acoustic fidelity."),
                    (GovernorOptimizationProfile::EcoBatterySaver, "Eco Battery Saver", "Prioritizes minimal CPU/GPU usage; extends battery on laptops and mobile devices."),
                    (GovernorOptimizationProfile::LowLatencyInteractive, "Low Latency Interactive (15ms)", "Ultra-tight 15ms buffer for responsive parameter modulation."),
                    (GovernorOptimizationProfile::StudioMaster, "Studio Master (High Fidelity)", "Deep 120ms buffer and maximum precision waveshaper synthesis."),
                    (GovernorOptimizationProfile::BluetoothA2DPSink, "Bluetooth A2DP Sink (+100ms Reserve)", "Extended jitter safety reserve to prevent wireless dropouts."),
                ];

                for (p, label, desc) in profiles {
                    if ui.radio_value(&mut app.settings.profile, p, label).changed() {
                        app.state.rain.optimization_profile = p;
                    }
                    ui.label(egui::RichText::new(format!("   {desc}")).small().weak());
                    ui.add_space(2.0);
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading("Sound Layers & Generative AI");
                ui.label("Select active acoustic layers. At least one layer must remain active.");
                ui.add_space(4.0);

                ui.horizontal_wrapped(|ui| {
                    if ui.checkbox(&mut app.settings.layers.raindrops, "💧 Raindrops (Physical particles)").changed() {
                        app.settings.layers.ensure_valid();
                        app.state.rain.sound_layers = app.settings.layers;
                    }
                    ui.add_space(8.0);
                    if ui.checkbox(&mut app.settings.layers.rain_wash, "🌊 Rain wash (Ambient bed)").changed() {
                        app.settings.layers.ensure_valid();
                        app.state.rain.sound_layers = app.settings.layers;
                    }
                    ui.add_space(8.0);
                    if ui.checkbox(&mut app.settings.layers.ai_texture, "🧠 AI Texture (Generative)").changed() {
                        app.settings.layers.ensure_valid();
                        app.state.rain.sound_layers = app.settings.layers;
                    }
                });

                ui.add_space(4.0);
                if !app.settings.ai_enabled {
                    ui.label(egui::RichText::new("ℹ Generative AI models are currently retraining. Physical and procedural layers synthesize 100% offline.").small().color(Color32::from_rgb(120, 180, 240)));
                }

                ui.add_space(6.0);
                if ui.checkbox(&mut app.settings.evolve, "🌱 Evolve (Continuous non-repeating acoustic drift)").changed() {
                    app.state.rain.evolve_enabled = app.settings.evolve;
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                // Derived parameters evaluation
                let derived = app.settings.resolve_parameters();

                // Advanced Settings Toggle & Drawer
                let adv_label = if app.settings.show_advanced_settings {
                    "🔽 Hide Advanced Audio & DSP Overrides"
                } else {
                    "▶ Show Advanced Audio & DSP Overrides"
                };
                if ui.button(adv_label).clicked() {
                    app.settings.show_advanced_settings = !app.settings.show_advanced_settings;
                }

                if app.settings.show_advanced_settings {
                    ui.add_space(6.0);
                    egui::Frame::group(ui.style())
                        .fill(ui.visuals().faint_bg_color)
                        .show(ui, |ui| {
                            ui.heading("Advanced Parameter Overrides");
                            ui.label("Low-level parameters are derived automatically. Overrides take precedence.");
                            ui.add_space(6.0);

                            // 1. Buffer Size
                            ui.horizontal(|ui| {
                                ui.label("Target Buffer:");
                                let is_manual = app.settings.manual_buffer_ms.is_some();
                                let mut buf_val = app.settings.manual_buffer_ms.unwrap_or(derived.target_buffer_ms);
                                if ui.add(egui::Slider::new(&mut buf_val, 10.0..=250.0).suffix(" ms")).changed() {
                                    app.settings.manual_buffer_ms = Some(buf_val);
                                }
                                if is_manual {
                                    ui.colored_label(Color32::from_rgb(240, 180, 80), "(Manual Override)");
                                    if ui.small_button("Auto").clicked() {
                                        app.settings.manual_buffer_ms = None;
                                    }
                                } else {
                                    ui.colored_label(Color32::from_rgb(100, 200, 120), "(Auto: Derived)");
                                }
                            });

                            // 2. Flow Solver Algorithm
                            ui.horizontal(|ui| {
                                ui.label("Flow Solver:");
                                let is_manual = app.settings.manual_solver.is_some();
                                let mut current_solver = app.settings.manual_solver.unwrap_or(derived.flow_solver);
                                egui::ComboBox::from_id_salt("adv_flow_solver_override")
                                    .selected_text(current_solver.label())
                                    .show_ui(ui, |ui| {
                                        let solvers = [
                                            FlowSolverAlgorithm::AdaptiveRk45 { tol: 1e-3, initial_h: 0.1 },
                                            FlowSolverAlgorithm::AdaptiveRk23 { tol: 1e-3, initial_h: 0.1 },
                                            FlowSolverAlgorithm::AdaptiveTsit5 { tol: 1e-3, initial_h: 0.1 },
                                            FlowSolverAlgorithm::AdaptiveHeun2 { tol: 1e-3, initial_h: 0.1 },
                                            FlowSolverAlgorithm::FixedRk4 { steps: 4 },
                                            FlowSolverAlgorithm::DpmSolverPP { steps: 3 },
                                            FlowSolverAlgorithm::TrainedPec { steps: 1 },
                                        ];
                                        for s in solvers {
                                            if ui.selectable_label(current_solver.label() == s.label(), s.label()).clicked() {
                                                current_solver = s;
                                                app.settings.manual_solver = Some(s);
                                                app.rain_view.flow_solver = s;
                                            }
                                        }
                                    });

                                if is_manual {
                                    ui.colored_label(Color32::from_rgb(240, 180, 80), "(Manual)");
                                    if ui.small_button("Auto").clicked() {
                                        app.settings.manual_solver = None;
                                        app.rain_view.flow_solver = derived.flow_solver;
                                    }
                                } else {
                                    ui.colored_label(Color32::from_rgb(100, 200, 120), "(Auto: Derived)");
                                }
                            });

                            // 3. Solver Error Tolerance
                            ui.horizontal(|ui| {
                                ui.label("Solver Tolerance:");
                                let is_manual = app.settings.manual_solver_tolerance.is_some();
                                let mut tol_val = app.settings.manual_solver_tolerance.unwrap_or(derived.solver_tolerance);
                                if ui.add(egui::Slider::new(&mut tol_val, 1e-4..=1e-1).logarithmic(true).custom_formatter(|v, _| format!("{v:.1e}"))).changed() {
                                    app.settings.manual_solver_tolerance = Some(tol_val);
                                }
                                if is_manual {
                                    ui.colored_label(Color32::from_rgb(240, 180, 80), "(Manual)");
                                    if ui.small_button("Auto").clicked() {
                                        app.settings.manual_solver_tolerance = None;
                                    }
                                } else {
                                    ui.colored_label(Color32::from_rgb(100, 200, 120), "(Auto: Derived)");
                                }
                            });

                            // 4. Droplet Voice Cap
                            ui.horizontal(|ui| {
                                ui.label("Droplet Voice Cap:");
                                let is_manual = app.settings.manual_droplet_voice_cap.is_some();
                                let mut cap_val = app.settings.manual_droplet_voice_cap.unwrap_or(derived.droplet_voice_cap);
                                if ui.add(egui::Slider::new(&mut cap_val, 32..=512)).changed() {
                                    app.settings.manual_droplet_voice_cap = Some(cap_val);
                                }
                                if is_manual {
                                    ui.colored_label(Color32::from_rgb(240, 180, 80), "(Manual)");
                                    if ui.small_button("Auto").clicked() {
                                        app.settings.manual_droplet_voice_cap = None;
                                    }
                                } else {
                                    ui.colored_label(Color32::from_rgb(100, 200, 120), "(Auto: Derived)");
                                }
                            });

                            // 5. Waveshaper Tier
                            ui.horizontal(|ui| {
                                ui.label("Waveshaper Tier:");
                                let is_manual = app.settings.manual_waveshaper_tier.is_some();
                                let mut tier_val = app.settings.manual_waveshaper_tier.unwrap_or(derived.waveshaper_tier);
                                egui::ComboBox::from_id_salt("adv_waveshaper_tier_selector")
                                    .selected_text(tier_val.label())
                                    .show_ui(ui, |ui| {
                                        let tiers = [
                                            QualityTier::Ternary158,
                                            QualityTier::AdaptiveMinimum,
                                            QualityTier::HighInt16,
                                            QualityTier::StudioFp32,
                                        ];
                                        for t in tiers {
                                            if ui.selectable_label(tier_val == t, t.label()).clicked() {
                                                tier_val = t;
                                                app.settings.manual_waveshaper_tier = Some(t);
                                                app.state.rain.quality_tier = t;
                                            }
                                        }
                                    });

                                if is_manual {
                                    ui.colored_label(Color32::from_rgb(240, 180, 80), "(Manual)");
                                    if ui.small_button("Auto").clicked() {
                                        app.settings.manual_waveshaper_tier = None;
                                        app.state.rain.quality_tier = derived.waveshaper_tier;
                                    }
                                } else {
                                    ui.colored_label(Color32::from_rgb(100, 200, 120), "(Auto: Derived)");
                                }
                            });

                            ui.add_space(8.0);
                            if ui.button("↺ Reset All Overrides to Auto (Derived)").clicked() {
                                app.settings.reset_advanced_overrides_to_auto();
                                let fresh_derived = app.settings.resolve_parameters();
                                app.rain_view.flow_solver = fresh_derived.flow_solver;
                                app.state.rain.quality_tier = fresh_derived.waveshaper_tier;
                            }
                        });
                }

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.button("Save & Close").clicked() {
                        crate::storage_manager::save_settings_state(None, &app.settings);
                        app.show_settings_dialog = false;
                    }
                });
            });
        });

    if !open {
        crate::storage_manager::save_settings_state(None, &app.settings);
        app.show_settings_dialog = false;
    }
}
