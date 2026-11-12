#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod exporter;

use app::EconCalcApp;
use eframe::egui;
use exporter::ARCH_SHORT_LABEL;

fn main() -> eframe::Result<()> {
    let window_title = format!(
        "econ-calc — Positron Computational Economics & Mathematical IDE ({})",
        ARCH_SHORT_LABEL
    );

    let viewport = egui::ViewportBuilder::default()
        .with_title(window_title)
        .with_inner_size([1520.0, 920.0])
        .with_min_inner_size([1080.0, 680.0]);

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "econ-calc",
        native_options,
        Box::new(|cc| Ok(Box::new(EconCalcApp::new(cc)))),
    )
}
