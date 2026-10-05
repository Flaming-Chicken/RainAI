//! Bottom-right expandable background task queue dock.

use crate::TemplateApp;
use crate::task_queue::TaskStatus;
use eframe::egui::{self, Color32};

pub fn render_queue_dock(app: &mut TemplateApp, ctx: &egui::Context) {
    let tasks = app.task_queue.list();

    // Only show if there are tasks or if user explicitly toggled it open
    if tasks.is_empty() && !app.show_task_queue_tray {
        return;
    }

    egui::Area::new(egui::Id::new("task_queue_dock_area"))
        .order(egui::Order::Foreground)
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -16.0))
        .show(ctx, |ui| {
            if !app.show_task_queue_tray {
                // Collapsed Pill Button
                let count = tasks.iter().filter(|t| t.status.is_active()).count();
                let text = if count > 0 {
                    format!("⏳ Tasks ({count} Active)")
                } else {
                    format!("📋 Tasks ({})", tasks.len())
                };

                egui::Frame::NONE
                    .fill(Color32::from_rgba_premultiplied(25, 30, 40, 240))
                    .stroke(egui::Stroke::new(1.0, Color32::from_rgb(80, 140, 220)))
                    .corner_radius(16)
                    .inner_margin(egui::Margin::symmetric(12, 6))
                    .show(ui, |ui| {
                        if ui
                            .button(egui::RichText::new(text).strong().color(Color32::WHITE))
                            .clicked()
                        {
                            app.show_task_queue_tray = true;
                        }
                    });
            } else {
                // Expanded Dock Window
                egui::Frame::NONE
                    .fill(Color32::from_rgba_premultiplied(18, 22, 30, 250))
                    .stroke(egui::Stroke::new(1.0, Color32::from_rgb(60, 100, 160)))
                    .corner_radius(10)
                    .inner_margin(egui::Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        ui.set_width(340.0);

                        ui.horizontal(|ui| {
                            ui.heading("⏳ Background Queue");
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .small_button("✖")
                                        .on_hover_text("Minimize queue dock")
                                        .clicked()
                                    {
                                        app.show_task_queue_tray = false;
                                    }
                                    if ui.small_button("Clear Finished").clicked() {
                                        app.task_queue.clear_finished();
                                    }
                                },
                            );
                        });
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);

                        if tasks.is_empty() {
                            ui.label("No active or enqueued background tasks.");
                        } else {
                            egui::ScrollArea::vertical()
                                .max_height(200.0)
                                .show(ui, |ui| {
                                    for task in tasks {
                                        ui.group(|ui| {
                                            ui.horizontal(|ui| {
                                                ui.label(egui::RichText::new(&task.title).strong());
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| match &task.status {
                                                        TaskStatus::Queued => {
                                                            ui.colored_label(
                                                                Color32::from_rgb(180, 180, 180),
                                                                "Queued",
                                                            );
                                                        }
                                                        TaskStatus::Running { .. } => {
                                                            ui.colored_label(
                                                                Color32::from_rgb(100, 200, 255),
                                                                "Running",
                                                            );
                                                        }
                                                        TaskStatus::Paused { .. } => {
                                                            ui.colored_label(
                                                                Color32::from_rgb(255, 200, 80),
                                                                "Paused",
                                                            );
                                                        }
                                                        TaskStatus::AwaitingConfirmation {
                                                            ..
                                                        } => {
                                                            ui.colored_label(
                                                                Color32::from_rgb(255, 140, 80),
                                                                "Review",
                                                            );
                                                        }
                                                        TaskStatus::Done { .. } => {
                                                            ui.colored_label(
                                                                Color32::from_rgb(80, 220, 120),
                                                                "Done",
                                                            );
                                                        }
                                                        TaskStatus::Failed { .. } => {
                                                            ui.colored_label(
                                                                Color32::from_rgb(240, 80, 80),
                                                                "Failed",
                                                            );
                                                        }
                                                        TaskStatus::Cancelled => {
                                                            ui.colored_label(
                                                                Color32::GRAY,
                                                                "Cancelled",
                                                            );
                                                        }
                                                    },
                                                );
                                            });

                                            let progress = task.status.progress();
                                            ui.add(
                                                egui::ProgressBar::new(progress).show_percentage(),
                                            );

                                            match &task.status {
                                                TaskStatus::Running {
                                                    status_text,
                                                    speed_bps,
                                                    ..
                                                } => {
                                                    let speed = *speed_bps;
                                                    ui.horizontal(|ui| {
                                                        ui.label(
                                                            egui::RichText::new(status_text)
                                                                .small()
                                                                .weak(),
                                                        );
                                                        if speed > 0.0 {
                                                            ui.with_layout(
                                                                egui::Layout::right_to_left(
                                                                    egui::Align::Center,
                                                                ),
                                                                |ui| {
                                                                    ui.label(
                                                                        egui::RichText::new(
                                                                            format!(
                                                                                "{:.1} KB/s",
                                                                                speed / 1024.0
                                                                            ),
                                                                        )
                                                                        .small()
                                                                        .weak(),
                                                                    );
                                                                },
                                                            );
                                                        }
                                                    });
                                                    ui.horizontal(|ui| {
                                                        if ui.small_button("⏸ Pause").clicked() {
                                                            app.task_queue.pause(&task.id);
                                                        }
                                                        if ui.small_button("✖ Cancel").clicked() {
                                                            app.task_queue.cancel(&task.id);
                                                        }
                                                    });
                                                }
                                                TaskStatus::Paused { status_text, .. } => {
                                                    ui.label(
                                                        egui::RichText::new(status_text)
                                                            .small()
                                                            .weak(),
                                                    );
                                                    ui.horizontal(|ui| {
                                                        if ui.small_button("▶ Resume").clicked() {
                                                            app.task_queue.resume(&task.id);
                                                        }
                                                        if ui.small_button("✖ Cancel").clicked() {
                                                            app.task_queue.cancel(&task.id);
                                                        }
                                                    });
                                                }
                                                TaskStatus::AwaitingConfirmation { message } => {
                                                    ui.label(egui::RichText::new(message).small());
                                                    ui.horizontal(|ui| {
                                                        if ui
                                                            .button("✔ Finalise & Upload")
                                                            .clicked()
                                                        {
                                                            app.task_queue
                                                                .confirm_and_proceed(&task.id);
                                                        }
                                                        if ui.button("✖ Cancel").clicked() {
                                                            app.task_queue.cancel(&task.id);
                                                        }
                                                    });
                                                }
                                                TaskStatus::Done { result_message } => {
                                                    ui.label(
                                                        egui::RichText::new(result_message)
                                                            .small()
                                                            .color(Color32::from_rgb(80, 200, 100)),
                                                    );
                                                }
                                                TaskStatus::Failed { error_message } => {
                                                    ui.label(
                                                        egui::RichText::new(error_message)
                                                            .small()
                                                            .color(Color32::from_rgb(240, 80, 80)),
                                                    );
                                                }
                                                _ => {}
                                            }
                                        });
                                        ui.add_space(2.0);
                                    }
                                });
                        }
                    });
            }
        });
}
