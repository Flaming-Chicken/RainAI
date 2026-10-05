//! Privacy Policy modal dialog.

use crate::TemplateApp;
use eframe::egui;

pub fn render_privacy_dialog(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if !app.show_privacy_dialog {
        return;
    }

    let mut open = true;
    let win_w = (ui.available_width() - 24.0).clamp(340.0, 680.0);
    let win_h = (ui.available_height() - 32.0).clamp(440.0, 720.0);

    egui::Window::new("🛡 RainAI Privacy Policy & Contribution Terms")
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                const PRIVACY_TEXT: &str = include_str!("../../../../../PRIVACY_POLICY.md");
                ui.label(PRIVACY_TEXT);
            });
        });

    if !open {
        app.show_privacy_dialog = false;
    }
}
