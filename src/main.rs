#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod exporter;

use app::EconCalcApp;
use eframe::egui;

fn main() -> eframe::Result<()> {
    let viewport = egui::ViewportBuilder::default()
        .with_title("EconCalc Studio")
        .with_inner_size([1280.0, 780.0])
        .with_min_inner_size([800.0, 520.0]);

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
