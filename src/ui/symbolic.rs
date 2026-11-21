use crate::collections::IndexMap;
use crate::ui::bento::{CURVE_COLOR_SWATCHES, ThemePalette, pill_button};
use eframe::egui::{self, Color32, FontId, Pos2, RichText, Sense, Stroke, Ui, Vec2};
use egui_plot::LineStyle;
use std::collections::VecDeque;

// =============================================================================
// SLIDER ANIMATION MODES & SPECIFICATION
// =============================================================================

/// Desmos-style slider animation playback mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationMode {
    /// `⇄` Bounce back and forth between `min` and `max`.
    Bounce,
    /// `⇉` Repeat forward from `min` to `max`, wrapping back to `min`.
    LoopForward,
    /// `↱₁` Play once from current value to `max` and stop.
    PlayOnce,
    /// `→` Increment continuously without stopping at `max`.
    Continuous,
}

impl AnimationMode {
    pub fn icon(self) -> &'static str {
        match self {
            AnimationMode::Bounce => "⇄",
            AnimationMode::LoopForward => "⇉",
            AnimationMode::PlayOnce => "↱₁",
            AnimationMode::Continuous => "→",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            AnimationMode::Bounce => "Bounce (min ↔ max)",
            AnimationMode::LoopForward => "Loop Forward (min → max)",
            AnimationMode::PlayOnce => "Play Once (→ max)",
            AnimationMode::Continuous => "Continuous (→ ∞)",
        }
    }

    pub fn all() -> [AnimationMode; 4] {
        [
            AnimationMode::Bounce,
            AnimationMode::LoopForward,
            AnimationMode::PlayOnce,
            AnimationMode::Continuous,
        ]
    }
}

/// Ordered animation speed presets (`0.25x .. 8x`).
pub const ANIM_SPEED_PRESETS: &[f64] = &[0.25_f64, 0.5_f64, 1.0_f64, 2.0_f64, 4.0_f64, 8.0_f64];

/// Desmos-style slider bounds + 60 FPS animation playback specification.
#[derive(Debug, Clone)]
pub struct SliderSpec {
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub edit_bounds_open: bool,
    pub playing: bool,
    pub anim_mode: AnimationMode,
    /// `+1.0` moving forward, `-1.0` moving backward (used in `Bounce` mode).
    pub anim_dir: f64,
    pub anim_speed: f64,
}

impl SliderSpec {
    pub fn for_parameter(name: &str, initial_val: f64) -> Self {
        let lower = name.to_lowercase();
        let (min, max, step) = if lower.contains("alpha")
            || lower.contains("beta")
            || lower.contains("delta")
            || lower == "s"
            || lower == "c_1"
            || lower == "n"
        {
            (0.0_f64, 1.5_f64, 0.01_f64)
        } else if lower == "i" || lower.contains("income") || lower.contains("y_bar") {
            let max_v = (initial_val.abs() * 2.5_f64).max(50.0_f64);
            (0.0_f64, max_v, 0.5_f64)
        } else if initial_val.abs() <= 5.0_f64 {
            (-10.0_f64, 10.0_f64, 0.05_f64)
        } else {
            let bound = (initial_val.abs() * 2.0_f64).max(20.0_f64);
            (-bound, bound, 0.1_f64)
        };

        Self {
            min,
            max,
            step,
            edit_bounds_open: false,
            playing: false,
            anim_mode: AnimationMode::Bounce,
            anim_dir: 1.0_f64,
            anim_speed: 1.0_f64,
        }
    }

    pub fn slower(&mut self) {
        for &preset in ANIM_SPEED_PRESETS.iter().rev() {
            if preset < self.anim_speed - 1e-6_f64 {
                self.anim_speed = preset;
                return;
            }
        }
        self.anim_speed = ANIM_SPEED_PRESETS[0];
    }

    pub fn faster(&mut self) {
        for &preset in ANIM_SPEED_PRESETS {
            if preset > self.anim_speed + 1e-6_f64 {
                self.anim_speed = preset;
                return;
            }
        }
        if let Some(&last) = ANIM_SPEED_PRESETS.last() {
            self.anim_speed = last;
        }
    }

    /// Advances an animated slider value by `dt_secs` seconds (full sweep takes ~4.5s at `1x`).
    /// Returns the updated parameter value.
    pub fn advance_value(&mut self, current: f64, dt_secs: f64) -> f64 {
        if !self.playing {
            return current;
        }
        let span = (self.max - self.min).abs().max(1e-3_f64);
        let rate = (span / 4.5_f64) * self.anim_speed.clamp(0.1_f64, 16.0_f64);
        let dt = dt_secs.clamp(0.0_f64, 0.1_f64);
        let delta = rate * dt;

        let mut next = match self.anim_mode {
            AnimationMode::Bounce => {
                let mut v = current + self.anim_dir * delta;
                if v >= self.max {
                    v = self.max;
                    self.anim_dir = -1.0_f64;
                } else if v <= self.min {
                    v = self.min;
                    self.anim_dir = 1.0_f64;
                }
                v
            }
            AnimationMode::LoopForward => {
                let mut v = current + delta;
                if v > self.max {
                    v = self.min + (v - self.max).rem_euclid(span);
                }
                v
            }
            AnimationMode::PlayOnce => {
                let v = current + delta;
                if v >= self.max {
                    self.playing = false;
                    self.max
                } else {
                    v
                }
            }
            AnimationMode::Continuous => {
                let v = current + delta;
                if v > self.max {
                    self.max = v + span * 0.25_f64;
                }
                v
            }
        };

        if self.step > 0.0_f64 && self.step < span * 0.25_f64 {
            // Keep smooth visual motion while snapping only when step is coarse enough
            if delta >= self.step * 0.5_f64 {
                next = (next / self.step).round() * self.step;
            }
        }
        next
    }
}

// =============================================================================
// CURVE APPEARANCE & FULL X/Y/T DOMAIN CONFIGURATION
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveLineStyle {
    Solid,
    Dashed,
    Dotted,
}

impl CurveLineStyle {
    pub fn label(self) -> &'static str {
        match self {
            CurveLineStyle::Solid => "Solid",
            CurveLineStyle::Dashed => "Dashed",
            CurveLineStyle::Dotted => "Dotted",
        }
    }

    pub fn glyph(self) -> &'static str {
        match self {
            CurveLineStyle::Solid => "—",
            CurveLineStyle::Dashed => "- -",
            CurveLineStyle::Dotted => "···",
        }
    }

    pub fn to_egui(self) -> LineStyle {
        match self {
            CurveLineStyle::Solid => LineStyle::Solid,
            CurveLineStyle::Dashed => LineStyle::Dashed { length: 8.0_f32 },
            CurveLineStyle::Dotted => LineStyle::Dotted { spacing: 5.0_f32 },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionPlotRole {
    StandardCurve,
    AreaUnderCurve,
    SampledPoints,
    Surface3DOnly,
}

impl FunctionPlotRole {
    pub fn label(self) -> &'static str {
        match self {
            FunctionPlotRole::StandardCurve => "Curve y = f(x)",
            FunctionPlotRole::AreaUnderCurve => "Area Shade ∫f(x)",
            FunctionPlotRole::SampledPoints => "Sampled Points",
            FunctionPlotRole::Surface3DOnly => "3D Surface Only",
        }
    }
}

/// Per-function visual appearance, line/point toggles, opacity/glow, and X/Y/t domain controls.
#[derive(Debug, Clone)]
pub struct FunctionStyle {
    pub visible: bool,
    pub show_lines: bool,
    pub show_points: bool,
    pub color: Color32,
    pub line_style: CurveLineStyle,
    pub stroke_width: f32,
    pub opacity: f32,
    pub role: FunctionPlotRole,
    /// Optional explicit X-domain restriction `(x_min, x_max)` + expression strings (e.g. `"-2*pi"`, `"I / price_x"`).
    pub custom_domain: Option<(f64, f64)>,
    pub custom_x_min_expr: String,
    pub custom_x_max_expr: String,
    /// Optional explicit Y-image restriction `(y_min, y_max)` + expression strings.
    pub custom_y_range: Option<(f64, f64)>,
    pub custom_y_min_expr: String,
    pub custom_y_max_expr: String,
    /// Optional explicit parameter domain `(t_min, t_max)` for parametric/polar curves.
    pub custom_param_domain: Option<(f64, f64)>,
    pub custom_t_min_expr: String,
    pub custom_t_max_expr: String,
    pub settings_open: bool,
    pub appearance_popover_open: bool,
}

impl FunctionStyle {
    pub fn default_for_index(idx: usize) -> Self {
        let palette = [
            Color32::from_rgb(184, 74, 57),  // Cinnabar Rubric / Kinnabari (#B84A39)
            Color32::from_rgb(72, 118, 168), // Lapis Lazuli (#4876A8)
            Color32::from_rgb(76, 142, 104), // Malachite / Prasinos (#4C8E68)
            Color32::from_rgb(196, 152, 68), // Ochre Gold / Chrysochra (#C49844)
            Color32::from_rgb(142, 92, 130), // Imperial Porphyry (#8E5C82)
            Color32::from_rgb(168, 104, 62), // Burnt Sienna (#A8683E)
        ];
        Self {
            visible: true,
            show_lines: true,
            show_points: false,
            color: palette[idx % palette.len()],
            line_style: if idx == 1 {
                CurveLineStyle::Dashed
            } else {
                CurveLineStyle::Solid
            },
            stroke_width: 2.3_f32,
            opacity: 1.0_f32,
            role: FunctionPlotRole::StandardCurve,
            custom_domain: None,
            custom_x_min_expr: String::new(),
            custom_x_max_expr: String::new(),
            custom_y_range: None,
            custom_y_min_expr: String::new(),
            custom_y_max_expr: String::new(),
            custom_param_domain: None,
            custom_t_min_expr: "0".to_string(),
            custom_t_max_expr: "2*pi".to_string(),
            settings_open: false,
            appearance_popover_open: false,
        }
    }
}

// =============================================================================
// UNIFIED SYMBOLIC CELL LIST (DESMOS ERGONOMICS + DARK BENTO IDE)
// =============================================================================

#[derive(Debug, Clone)]
pub enum SymbolicCellKind {
    /// Unified single-entry formula cell (`f(x) = x^2`, `f(5)`, `x^2 - 5x + 6 = 0`,
    /// `y < x^2`, `(cos(t), sin(t))`, `tangent(f, 2)`, `surplus(f, g)`, `a = 2.5`, etc.).
    Expression {
        raw: String,
        /// Key in `workspace.functions` / `workspace.function_styles` (e.g. `"f(x)"` or `"__cell_3"`).
        bound_key: Option<String>,
        /// Two-way interactive LaTeX drawer open toggle.
        latex_drawer_open: bool,
        /// Editable draft text inside the LaTeX drawer.
        latex_draft: String,
        /// Whether `latex_draft` was edited by user and has a parse error pending.
        latex_error: Option<String>,
        /// Toggle between decimal and continued-fraction exact display (`⅓ ↔ 0.333`).
        show_fraction: bool,
        /// Toggle plotting constant evaluation point `(x0, f(x0))` on the 2D graph.
        plot_eval_point: bool,
        /// Toggle highlighting solved 1D equation roots on the 2D graph.
        highlight_roots: bool,
    },
    /// Dedicated parameter slider cell backed by `workspace.parameters[param_name]`.
    Slider {
        param_name: String,
    },
    /// Collapsible folder grouping child cells (`folder_id == Some(folder.id)`).
    Folder {
        title: String,
        collapsed: bool,
        visible: bool,
    },
    /// Inline economic / mathematical explanatory note cell.
    Note {
        text: String,
    },
    /// Interactive function value table `(x, f(x), f'(x))` synced to Data Viewer & R `data.frame`.
    Table {
        fn_sig: String,
        x_values: Vec<f64>,
        show_derivative: bool,
        dataset_name: String,
    },
}

#[derive(Debug, Clone)]
pub struct SymbolicCell {
    pub id: u64,
    /// Optional parent `Folder` cell ID; renders vertical connector line & inherits folder visibility.
    pub folder_id: Option<u64>,
    pub kind: SymbolicCellKind,
}

impl SymbolicCell {
    pub fn new_expression(id: u64, raw: impl Into<String>, bound_key: Option<String>) -> Self {
        let raw_s = raw.into();
        let latex_draft = crate::math::to_latex_from_ast(&raw_s);
        Self {
            id,
            folder_id: None,
            kind: SymbolicCellKind::Expression {
                raw: raw_s,
                bound_key,
                latex_drawer_open: false,
                latex_draft,
                latex_error: None,
                show_fraction: false,
                plot_eval_point: false,
                highlight_roots: true,
            },
        }
    }

    pub fn new_slider(id: u64, param_name: impl Into<String>) -> Self {
        Self {
            id,
            folder_id: None,
            kind: SymbolicCellKind::Slider {
                param_name: param_name.into(),
            },
        }
    }

    pub fn new_folder(id: u64, title: impl Into<String>) -> Self {
        Self {
            id,
            folder_id: None,
            kind: SymbolicCellKind::Folder {
                title: title.into(),
                collapsed: false,
                visible: true,
            },
        }
    }

    pub fn new_note(id: u64, text: impl Into<String>) -> Self {
        Self {
            id,
            folder_id: None,
            kind: SymbolicCellKind::Note { text: text.into() },
        }
    }

    pub fn new_table(
        id: u64,
        fn_sig: impl Into<String>,
        x_values: Vec<f64>,
        dataset_name: impl Into<String>,
    ) -> Self {
        Self {
            id,
            folder_id: None,
            kind: SymbolicCellKind::Table {
                fn_sig: fn_sig.into(),
                x_values,
                show_derivative: true,
                dataset_name: dataset_name.into(),
            },
        }
    }
}

// =============================================================================
// UNDO / REDO HISTORY STACK
// =============================================================================

#[derive(Debug, Clone)]
pub struct WorkspaceSnapshot {
    pub cells: Vec<SymbolicCell>,
    pub next_cell_id: u64,
    pub parameters: IndexMap<String, f64>,
    pub slider_specs: IndexMap<String, SliderSpec>,
    pub functions: IndexMap<String, String>,
    pub function_styles: IndexMap<String, FunctionStyle>,
}

#[derive(Debug, Clone, Default)]
pub struct UndoHistory {
    undo_stack: VecDeque<WorkspaceSnapshot>,
    redo_stack: VecDeque<WorkspaceSnapshot>,
}

impl UndoHistory {
    const MAX_ENTRIES: usize = 60;

    pub fn push(&mut self, snap: WorkspaceSnapshot) {
        if self.undo_stack.len() >= Self::MAX_ENTRIES {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(snap);
        self.redo_stack.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo(&mut self, current: WorkspaceSnapshot) -> Option<WorkspaceSnapshot> {
        let prev = self.undo_stack.pop_back()?;
        if self.redo_stack.len() >= Self::MAX_ENTRIES {
            self.redo_stack.pop_front();
        }
        self.redo_stack.push_back(current);
        Some(prev)
    }

    pub fn redo(&mut self, current: WorkspaceSnapshot) -> Option<WorkspaceSnapshot> {
        let next = self.redo_stack.pop_back()?;
        if self.undo_stack.len() >= Self::MAX_ENTRIES {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(current);
        Some(next)
    }
}

// =============================================================================
// COMPARATIVE STATICS BASELINE FREEZE, POIs, AXIS CONFIG & SHIFT-DRAG ZOOM
// =============================================================================

#[derive(Debug, Clone)]
pub struct FrozenCurve {
    pub label: String,
    pub color: Color32,
    pub branches: Vec<Vec<[f64; 2]>>,
}

/// Snapshot of curves & equilibrium before a parameter shock ("Freeze Baseline" `📌`).
#[derive(Debug, Clone)]
pub struct FrozenBaseline {
    #[allow(dead_code)]
    pub label: String,
    pub curves: Vec<FrozenCurve>,
    pub equilibrium: Option<(f64, f64)>,
}

/// User-pinned Point of Interest displayed with a persistent coordinate label on the 2D plot.
#[derive(Debug, Clone)]
pub struct PinnedPoi {
    pub x: f64,
    pub y: f64,
    pub label: String,
    pub color: Color32,
}

/// Spatial zone detected for `Shift + Hover` / `Shift + Click & Drag` axis zooming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftDragZone {
    XAxis,
    YAxis,
    BothXY,
}

/// Active `Shift + Click & Drag` interaction state on the 2D plot.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShiftDragState {
    pub active_zone: Option<ShiftDragZone>,
    pub hovered_zone: Option<ShiftDragZone>,
}

/// Global X & Y axis labels and optional manual grid step configuration.
#[derive(Debug, Clone)]
pub struct AxisDisplayConfig {
    pub x_label: String,
    pub y_label: String,
    pub x_step: Option<f64>,
    pub y_step: Option<f64>,
}

impl Default for AxisDisplayConfig {
    fn default() -> Self {
        Self {
            x_label: "x".to_string(),
            y_label: "y".to_string(),
            x_step: None,
            y_step: None,
        }
    }
}

// =============================================================================
// REUSABLE DARK SLATE BENTO SYMBOLIC UI WIDGETS
// =============================================================================

/// Renders the circular colored curve emblem inside the 36px left gutter.
/// Left-click toggles visibility; right-click opens the appearance popover.
pub fn render_gutter_curve_badge(
    ui: &mut Ui,
    color: Color32,
    visible: bool,
    line_style: CurveLineStyle,
    show_points: bool,
    pal: ThemePalette,
) -> egui::Response {
    let size = Vec2::splat(22.0_f32);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let center = rect.center();
    let radius = 9.5_f32;
    let p = ui.painter();

    if visible {
        p.circle_filled(center, radius, color);
        p.circle_stroke(center, radius, Stroke::new(1.0_f32, pal.border));
        let ink = Color32::from_rgb(238, 232, 218);
        let y = center.y;
        match (line_style, show_points) {
            (_, true) => {
                p.circle_filled(Pos2::new(center.x - 4.0_f32, y), 1.8_f32, ink);
                p.circle_filled(center, 1.8_f32, ink);
                p.circle_filled(Pos2::new(center.x + 4.0_f32, y), 1.8_f32, ink);
            }
            (CurveLineStyle::Solid, false) => {
                p.line_segment(
                    [Pos2::new(center.x - 5.0_f32, y + 1.5_f32), Pos2::new(center.x + 5.0_f32, y - 1.5_f32)],
                    Stroke::new(1.7_f32, ink),
                );
            }
            (CurveLineStyle::Dashed, false) => {
                p.line_segment(
                    [Pos2::new(center.x - 5.2_f32, y), Pos2::new(center.x - 1.2_f32, y)],
                    Stroke::new(1.7_f32, ink),
                );
                p.line_segment(
                    [Pos2::new(center.x + 1.2_f32, y), Pos2::new(center.x + 5.2_f32, y)],
                    Stroke::new(1.7_f32, ink),
                );
            }
            (CurveLineStyle::Dotted, false) => {
                p.circle_filled(Pos2::new(center.x - 4.0_f32, y), 1.3_f32, ink);
                p.circle_filled(center, 1.3_f32, ink);
                p.circle_filled(Pos2::new(center.x + 4.0_f32, y), 1.3_f32, ink);
            }
        }
    } else {
        p.circle_filled(center, radius, pal.extreme_bg);
        p.circle_stroke(center, radius, Stroke::new(1.8_f32, color.gamma_multiply(0.65_f32)));
    }
    resp
}

/// Renders the Desmos-style floating curve appearance controls (Lines/Points switches,
/// Solid/Dashed/Dotted stroke selector, Width & Opacity/Glow sliders, Role selector,
/// and curated color grid with `✓` checkmark + custom RGB picker).
pub fn render_curve_appearance_controls(
    ui: &mut Ui,
    id_salt: &str,
    style: &mut FunctionStyle,
    pal: ThemePalette,
) {
    ui.set_min_width(228.0_f32);
    ui.horizontal(|ui| {
        ui.checkbox(&mut style.show_lines, "Lines");
        ui.add_space(6.0_f32);
        ui.checkbox(&mut style.show_points, "Points");
        ui.add_space(6.0_f32);
        ui.checkbox(&mut style.visible, "Visible");
    });

    ui.add_space(4.0_f32);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Stroke:").color(pal.text_secondary));
        for ls in [
            CurveLineStyle::Solid,
            CurveLineStyle::Dashed,
            CurveLineStyle::Dotted,
        ] {
            let active = style.line_style == ls;
            if ui
                .add(pill_button(ls.glyph(), active, pal))
                .on_hover_text(ls.label())
                .clicked()
            {
                style.line_style = ls;
            }
        }
    });

    ui.add_space(4.0_f32);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Width:").color(pal.text_secondary));
        ui.add(
            egui::Slider::new(&mut style.stroke_width, 1.0_f32..=5.0_f32)
                .suffix(" px")
                .max_decimals(1),
        );
    });
    ui.horizontal(|ui| {
        ui.label(RichText::new("Opacity (α):").color(pal.text_secondary));
        ui.add(
            egui::Slider::new(&mut style.opacity, 0.1_f32..=1.0_f32)
                .max_decimals(2),
        );
    });

    ui.add_space(4.0_f32);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Role:").color(pal.text_secondary));
        egui::ComboBox::from_id_source(format!("appearance_role_{}", id_salt))
            .selected_text(style.role.label())
            .show_ui(ui, |ui| {
                for r in [
                    FunctionPlotRole::StandardCurve,
                    FunctionPlotRole::AreaUnderCurve,
                    FunctionPlotRole::SampledPoints,
                    FunctionPlotRole::Surface3DOnly,
                ] {
                    ui.selectable_value(&mut style.role, r, r.label());
                }
            });
    });

    ui.add_space(5.0_f32);
    ui.label(
        RichText::new("Color")
            .font(FontId::proportional(11.5_f32))
            .strong()
            .color(pal.text_secondary),
    );
    ui.add_space(2.0_f32);

    // 6x2 vibrant swatch grid with crisp `✓` checkmark on active color
    for chunk in CURVE_COLOR_SWATCHES.chunks(6) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 5.0_f32;
            for &swatch in chunk {
                let selected = style.color == swatch;
                let (rect, resp) = ui.allocate_exact_size(Vec2::splat(22.0_f32), Sense::click());
                let c = rect.center();
                ui.painter().circle_filled(c, 9.5_f32, swatch);
                if selected {
                    ui.painter().circle_stroke(
                        c,
                        10.5_f32,
                        Stroke::new(1.5_f32, pal.text_primary),
                    );
                    ui.painter().text(
                        c,
                        egui::Align2::CENTER_CENTER,
                        "✓",
                        FontId::proportional(12.0_f32),
                        Color32::from_rgb(14, 18, 26),
                    );
                } else if resp.hovered() {
                    ui.painter().circle_stroke(
                        c,
                        10.2_f32,
                        Stroke::new(1.2_f32, pal.text_secondary),
                    );
                }
                if resp.clicked() {
                    style.color = swatch;
                }
            }
        });
        ui.add_space(3.0_f32);
    }

    ui.horizontal(|ui| {
        ui.label(RichText::new("Custom RGB:").color(pal.text_muted));
        ui.color_edit_button_srgba(&mut style.color);
    });
}

/// Renders the Desmos-style slider Animation Mode & Speed popover (`⇄`, `⇉`, `↱₁`, `→` + `« 1x »`).
pub fn render_slider_anim_controls(ui: &mut Ui, spec: &mut SliderSpec, pal: ThemePalette) {
    ui.set_min_width(185.0_f32);
    ui.label(
        RichText::new("Animation Mode")
            .font(FontId::proportional(11.5_f32))
            .strong()
            .color(pal.accent_blue),
    );
    ui.add_space(3.0_f32);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0_f32;
        for mode in AnimationMode::all() {
            let active = spec.anim_mode == mode;
            if ui
                .add(pill_button(mode.icon(), active, pal))
                .on_hover_text(mode.label())
                .clicked()
            {
                spec.anim_mode = mode;
            }
        }
    });
    ui.label(
        RichText::new(spec.anim_mode.label())
            .font(FontId::proportional(10.8_f32))
            .color(pal.text_muted),
    );

    ui.separator();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Speed:").color(pal.text_secondary));
        if ui
            .add(pill_button("«", false, pal))
            .on_hover_text("Slower")
            .clicked()
        {
            spec.slower();
        }
        let spd_str = if (spec.anim_speed - spec.anim_speed.round()).abs() < 1e-6_f64 {
            format!("{}x", spec.anim_speed.round() as i64)
        } else {
            format!("{:.2}x", spec.anim_speed)
                .trim_end_matches('0')
                .to_string()
        };
        ui.label(
            RichText::new(spd_str)
                .font(FontId::monospace(12.0_f32))
                .strong()
                .color(pal.accent_blue),
        );
        if ui
            .add(pill_button("»", false, pal))
            .on_hover_text("Faster")
            .clicked()
        {
            spec.faster();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slider_animation_modes_and_undo_history() {
        let mut spec = SliderSpec::for_parameter("a", 0.0);
        spec.min = 0.0;
        spec.max = 10.0;
        spec.step = 0.0;
        spec.playing = true;
        spec.anim_speed = 4.0;

        // Bounce reaches max and reverses direction
        spec.anim_mode = AnimationMode::Bounce;
        let mut v = 9.5;
        for _ in 0..10 {
            v = spec.advance_value(v, 0.05);
        }
        assert!(v <= 10.0 && spec.anim_dir < 0.0);

        // LoopForward wraps back near min
        spec.anim_mode = AnimationMode::LoopForward;
        let wrapped = spec.advance_value(9.9, 0.08);
        assert!((0.0..5.0).contains(&wrapped));

        // PlayOnce stops playing at max
        spec.anim_mode = AnimationMode::PlayOnce;
        spec.playing = true;
        let done_v = spec.advance_value(9.9, 0.08);
        assert!((done_v - 10.0).abs() < 1e-9);
        assert!(!spec.playing);

        // Continuous expands max without stopping
        spec.anim_mode = AnimationMode::Continuous;
        spec.playing = true;
        let cont_v = spec.advance_value(9.9, 0.08);
        assert!(cont_v > 10.0 && spec.max > 10.0);

        // Speed presets
        spec.anim_speed = 1.0;
        spec.faster();
        assert!((spec.anim_speed - 2.0).abs() < 1e-9);
        spec.slower();
        assert!((spec.anim_speed - 1.0).abs() < 1e-9);

        // Undo / Redo history round-trip
        let mut hist = UndoHistory::default();
        let s1 = WorkspaceSnapshot {
            cells: vec![SymbolicCell::new_note(1, "v1")],
            next_cell_id: 2,
            parameters: IndexMap::new(),
            slider_specs: IndexMap::new(),
            functions: IndexMap::new(),
            function_styles: IndexMap::new(),
        };
        let s2 = WorkspaceSnapshot {
            cells: vec![SymbolicCell::new_note(1, "v2")],
            next_cell_id: 2,
            parameters: IndexMap::new(),
            slider_specs: IndexMap::new(),
            functions: IndexMap::new(),
            function_styles: IndexMap::new(),
        };
        hist.push(s1);
        assert!(hist.can_undo());
        let undone = hist.undo(s2).unwrap();
        assert!(matches!(&undone.cells[0].kind, SymbolicCellKind::Note { text } if text == "v1"));
        assert!(hist.can_redo());
        let redone = hist.redo(undone).unwrap();
        assert!(matches!(&redone.cells[0].kind, SymbolicCellKind::Note { text } if text == "v2"));
    }
}
