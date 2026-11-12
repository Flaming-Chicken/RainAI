//! Help & Studio Architecture modal dialog.

use crate::TemplateApp;
use eframe::egui;

pub fn render_help_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if !app.show_help_dialog {
        return;
    }

    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(340.0, 620.0);
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
                    ui.heading("🌧️ RainAI Neural & Physical Audio Studio");
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

                ui.heading("1. Physical Acoustics & Fluid Simulation");
                ui.add_space(4.0);
                ui.label("• 9 Continuous Surface Materials: Corrugated tin roof modal ringing (ISO 140-18), deep water bubble cavitation (Pumphrey & Crum / Minnaert chirps), asphalt splash shockwaves, canvas tent membrane resonance, and glass window pinging.");
                ui.label("• Drop Aerodynamics: Ulbrich Gamma Drop Size Distribution (D in [0.2, 5.5] mm) and Gunn-Kinzer terminal fall velocity ($v_t \\in [0.5, 9.65]$ m/s) coupled with Erpul wind vectors.");
                ui.label("• Particle Decoupling: Foreground discrete raindrops are rendered via modal particle impulse synthesis; diffuse ambient rain wash and wind gusts are rendered via the subtractive filterbank.");

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading("2. Continuous-Time Neural ODE & Flow Matching");
                ui.add_space(4.0);
                ui.label("• Continuous Generative Flow: Recurrent Mamba-2 SSM with brown noise stochastic driving (-6 dB/oct), learned acoustic vector fields, and spatial VAE latent decoders.");
                ui.label("• Advanced Flow Solvers: Supports Dormand-Prince (RK45), Bogacki-Shampine (RK23), Tsitouras 5(4), Heun2, DPM-Solver++, Trained PEC (1-NFE), and Trained Implicit RK (Surrogate 1-NFE).");
                ui.label("• 1-Step Consistency Distillation Jump: Instantaneous sub-millisecond Euler inference for battery-constrained mobile and WebGPU devices.");
                ui.label("• Incoming Models: AI texture layers are currently undergoing offline retraining with updated zero-leakage attribution sets.");

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading("3. 3D Spatial Audio & Ambisonic Decoders");
                ui.add_space(4.0);
                ui.label("• 4-Channel First-Order Ambisonics (FOA): W (Omni pressure), Y (Side dipole), Z (Elevation dome), X (Front-back gradient).");
                ui.label("• Decoders: Desktop Speakers (Phase-Correct Stereo default), Headphones (Binaural HRTF via Google Resonance Audio FIR), 7.1 Surround, or custom HRIR (.wav, .sofa) impulse responses.");

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading("4. Studio Controls & Navigation");
                ui.add_space(4.0);
                ui.label("• Play / Pause: Start or pause the live Ambisonic soundscape.");
                ui.label("• Sound Layers: Toggle physical Raindrops, continuous Rain wash, and generative AI texture independently.");
                ui.label("• ⚙ Settings: Access performance profiles, listening setups, and fine-grained DSP/solver overrides.");
                ui.label("• 💾 Storage: Preload all neural weights/impulse responses, and export/import full presets with copy-to-clipboard.");
                ui.label("• 🎵 Audio Export: Faster-than-realtime offline export to FLAC Level 8, MP3 (320k), or WAV with embedded XAI provenance tags.");
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("100% Client-Side Privacy: Runs offline in local browser WASM/WebAudio or native desktop with zero external cloud telemetry.")
                        .color(egui::Color32::from_rgb(80, 180, 100))
                        .strong(),
                );
            });
        });

    if !open {
        app.show_help_dialog = false;
    }
}
