pub use crate::collections::IndexMap;
use crate::exporter::{
    ActiveModelKind, ExportOutcome, ExportPayload, GridRow, ProjectFileEntry, ReportFormat,
    export_and_launch_positron, list_project_directory, load_text_file, save_text_file,
};
use crate::math::{
    self, CompiledExpr, CompiledWorkspace, EvalContext, EvalScratch, PlottableEntry,
};
pub use crate::r_engine::RObject;
use crate::r_engine::{DesmosBridge, EmbeddedREngine, REngineMathBridge};
use crate::ui::bento::{
    ThemePalette, apply_bento_theme, capsule_frame, draw_glowing_plot_line, draw_principal_axes,
    header_icon_button, pill_button, render_bento_card_frame, render_card_sub_toolbar,
    render_card_tab_header, render_card_title_header,
};
use crate::ui::math_preview::{apply_smart_math_shortcuts, render_math_preview_strip};
use eframe::egui::{
    self, Color32, FontId, Margin, Pos2, RichText, Rounding, Sense, Stroke, Ui, Vec2,
};
use egui_extras as _;
use egui_plot::{
    Legend, Line, LineStyle, MarkerShape, Plot, PlotBounds, PlotPoints, Points, Polygon, VLine,
};
use nalgebra::{Matrix2, Vector2};
use std::path::PathBuf;

// =============================================================================
// DESMOS SLIDER SPECIFICATION & FUNCTION VISUAL STYLING
// =============================================================================

/// Desmos-style slider bounds specification with fixed min/max/step limits (prevents runaway values).
#[derive(Debug, Clone)]
pub struct SliderSpec {
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub edit_bounds_open: bool,
}

impl SliderSpec {
    pub fn for_parameter(name: &str, initial_val: f64) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("alpha")
            || lower.contains("beta")
            || lower.contains("delta")
            || lower == "s"
            || lower == "c_1"
            || lower == "n"
        {
            Self {
                min: 0.0_f64,
                max: 1.5_f64,
                step: 0.01_f64,
                edit_bounds_open: false,
            }
        } else if lower == "i" || lower.contains("income") || lower.contains("y_bar") {
            let max_v = (initial_val.abs() * 2.5_f64).max(50.0_f64);
            Self {
                min: 0.0_f64,
                max: max_v,
                step: 0.5_f64,
                edit_bounds_open: false,
            }
        } else if initial_val.abs() <= 5.0_f64 {
            Self {
                min: -10.0_f64,
                max: 10.0_f64,
                step: 0.05_f64,
                edit_bounds_open: false,
            }
        } else {
            let bound = (initial_val.abs() * 2.0_f64).max(20.0_f64);
            Self {
                min: -bound,
                max: bound,
                step: 0.1_f64,
                edit_bounds_open: false,
            }
        }
    }
}

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
            FunctionPlotRole::StandardCurve => "Curve y=f(x)",
            FunctionPlotRole::AreaUnderCurve => "Area Shade ∫f(x)",
            FunctionPlotRole::SampledPoints => "Sampled Points",
            FunctionPlotRole::Surface3DOnly => "3D Surface Only",
        }
    }
}

/// Per-function visual and domain configuration for graphing freedom.
#[derive(Debug, Clone)]
pub struct FunctionStyle {
    pub visible: bool,
    pub color: Color32,
    pub line_style: CurveLineStyle,
    pub stroke_width: f32,
    pub role: FunctionPlotRole,
    pub custom_domain: Option<(f64, f64)>,
    pub settings_open: bool,
}

impl FunctionStyle {
    pub fn default_for_index(idx: usize) -> Self {
        let palette = [
            Color32::from_rgb(248, 113, 113), // Coral / Neon Red (#F87171)
            Color32::from_rgb(56, 189, 248),  // Sky Blue (#38BDF8)
            Color32::from_rgb(16, 185, 129),  // Emerald (#10B981)
            Color32::from_rgb(167, 139, 250), // Violet (#A78BFA)
            Color32::from_rgb(251, 191, 36),  // Amber (#FBBF24)
            Color32::from_rgb(236, 72, 153),  // Pink (#EC4899)
        ];
        Self {
            visible: true,
            color: palette[idx % palette.len()],
            line_style: if idx == 1 {
                CurveLineStyle::Dashed
            } else {
                CurveLineStyle::Solid
            },
            stroke_width: 2.3_f32,
            role: FunctionPlotRole::StandardCurve,
            custom_domain: None,
            settings_open: false,
        }
    }
}

// =============================================================================
// DATASETS (RObject is defined in src/r_engine/object.rs and re-exported above)
// =============================================================================

#[derive(Debug, Clone)]
pub struct DataTable {
    pub name: String,
    pub points: Vec<[f64; 2]>,
    pub raw_csv: String,
    pub show_on_plot: bool,
    pub show_ols_line: bool,
    pub color: Color32,
}

impl DataTable {
    pub fn from_csv_str(
        name: impl Into<String>,
        csv_text: impl Into<String>,
        color: Color32,
    ) -> Self {
        let raw_csv = csv_text.into();
        let points = Self::parse_xy_csv(&raw_csv);
        Self {
            name: name.into(),
            points,
            raw_csv,
            show_on_plot: false,
            show_ols_line: false,
            color,
        }
    }

    pub fn sync_from_raw_csv(&mut self) {
        self.points = Self::parse_xy_csv(&self.raw_csv);
    }

    pub fn sync_raw_csv_from_points(&mut self) {
        let mut out = String::from("x,y\n");
        for [x, y] in &self.points {
            out.push_str(&format!("{:.6},{:.6}\n", x, y));
        }
        self.raw_csv = out;
    }

    pub fn parse_xy_csv(raw: &str) -> Vec<[f64; 2]> {
        let mut pts = Vec::new();
        for raw_line in raw.lines() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed
                .split([',', ';', '\t'])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                if let (Ok(x), Ok(y)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                    if x.is_finite() && y.is_finite() {
                        pts.push([x, y]);
                    }
                }
            }
        }
        pts
    }
}

/// Core Desmos + RStudio dynamic symbolic state.
#[derive(Debug, Clone)]
pub struct Workspace {
    pub parameters: IndexMap<String, f64>,
    pub slider_specs: IndexMap<String, SliderSpec>,
    pub functions: IndexMap<String, String>,
    pub function_styles: IndexMap<String, FunctionStyle>,
    pub compiled_math: CompiledWorkspace,
    pub r_objects: IndexMap<String, RObject>,
    pub datasets: Vec<DataTable>,
    pub active_dataset_idx: usize,
    pub markdown_notes: String,
    pub csv_scratchpad: String,
}

impl Workspace {
    pub fn new_default() -> Self {
        let mut ws = Self {
            parameters: IndexMap::new(),
            slider_specs: IndexMap::new(),
            functions: IndexMap::new(),
            function_styles: IndexMap::new(),
            compiled_math: CompiledWorkspace::default(),
            r_objects: IndexMap::new(),
            datasets: Vec::new(),
            active_dataset_idx: 0,
            markdown_notes: String::new(),
            csv_scratchpad: String::new(),
        };

        ws.set_parameter("A", 2.0_f64);
        ws.set_parameter("alpha", 0.45_f64);
        ws.set_parameter("beta", 0.55_f64);
        ws.set_parameter("price_x", 4.0_f64);
        ws.set_parameter("price_y", 5.0_f64);
        ws.set_parameter("I", 100.0_f64);
        ws.set_parameter("Y_bar", 18.0_f64);

        ws.set_function("f(x)", "(Y_bar / (A * x^alpha))^(1 / beta)");
        ws.set_function("g(x)", "(I - price_x * x) / price_y");

        ws.markdown_notes = "\
# Microeconomic Tangency & Optimization Notes

- **Isoquant / Indifference Curve** $f(x)$ traces combinations $(x, y)$ yielding constant output $\\bar{Y}$.
- **Budget / Isocost Line** $g(x) = \\frac{I - P_x x}{P_y}$ has constant slope $-\\frac{P_x}{P_y}$.
- **Optimality Condition:** $\\text{MRS}_{x,y} = \\frac{\\alpha y}{\\beta x} = \\frac{P_x}{P_y}$."
            .to_string();

        ws.csv_scratchpad = "\
x,y
1.5,16.8
3.0,13.5
5.0,11.2
8.0,8.6
11.25,11.0
14.0,6.2
18.0,4.5"
            .to_string();

        ws.datasets.push(DataTable::from_csv_str(
            "empirical_sample",
            ws.csv_scratchpad.clone(),
            Color32::from_rgb(239, 68, 68),
        ));

        ws
    }

    pub fn set_parameter(&mut self, name: impl Into<String>, val: f64) {
        let key = name.into();
        if self.slider_specs.get(&key).is_none() {
            let spec = SliderSpec::for_parameter(&key, val);
            self.slider_specs.insert(key.clone(), spec);
        }
        self.parameters.insert(key.clone(), val);
        self.r_objects.insert(key, RObject::scalar(val));
    }

    pub fn set_function(&mut self, sig: impl Into<String>, expr: impl Into<String>) {
        let key = sig.into();
        let idx = self.functions.len();
        if self.function_styles.get(&key).is_none() {
            self.function_styles
                .insert(key.clone(), FunctionStyle::default_for_index(idx));
        }
        self.functions.insert(key, expr.into());
        self.sync_math_cache();
    }

    pub fn sync_math_cache(&mut self) {
        self.compiled_math = CompiledWorkspace::build(&self.functions);
    }

    pub fn clear(&mut self) {
        self.parameters.clear();
        self.slider_specs.clear();
        self.functions.clear();
        self.function_styles.clear();
        self.compiled_math = CompiledWorkspace::default();
        self.r_objects.clear();
        self.markdown_notes.clear();
        self.csv_scratchpad.clear();
        for ds in &mut self.datasets {
            ds.show_on_plot = false;
            ds.show_ols_line = false;
        }
    }
}

// =============================================================================
// UI NAVIGATION ENUMS (ThemePalette lives in src/ui/bento.rs)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewportMode {
    Plot2D,
    Surface3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AxisZoomMode {
    BothXY,
    XOnly,
    YOnly,
}

impl AxisZoomMode {
    fn label(self) -> &'static str {
        match self {
            AxisZoomMode::BothXY => "XY",
            AxisZoomMode::XOnly => "X",
            AxisZoomMode::YOnly => "Y",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeftBottomTab {
    Files,
    Models,
    ReportFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileFilterKind {
    All,
    Scripts,
    Data,
    Notes,
}

impl FileFilterKind {
    fn label(self) -> &'static str {
        match self {
            FileFilterKind::All => "All files",
            FileFilterKind::Scripts => "Scripts (.R)",
            FileFilterKind::Data => "Data (.csv)",
            FileFilterKind::Notes => "Notes (.qmd/.md)",
        }
    }

    fn matches(self, name: &str, is_dir: bool) -> bool {
        if is_dir || self == FileFilterKind::All {
            return true;
        }
        let lower = name.to_lowercase();
        match self {
            FileFilterKind::All => true,
            FileFilterKind::Scripts => lower.ends_with(".r") || lower.ends_with(".econ"),
            FileFilterKind::Data => lower.ends_with(".csv") || lower.ends_with(".tsv"),
            FileFilterKind::Notes => {
                lower.ends_with(".qmd") || lower.ends_with(".rmd") || lower.ends_with(".md")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BottomDockTab {
    QuickConsole,
    ScriptEditor,
    NumericAndGraphSettings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightDockTab {
    Datasets,
    AnalysisAndLatex,
    NotesAndReports,
    HelpCenter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SurfaceColorMap {
    OceanThermal,
    Viridis,
    CoolWarm,
}

impl SurfaceColorMap {
    fn label(self) -> &'static str {
        match self {
            SurfaceColorMap::OceanThermal => "Ocean Thermal",
            SurfaceColorMap::Viridis => "Viridis",
            SurfaceColorMap::CoolWarm => "Cool-Warm",
        }
    }

    fn map_normalized(self, t: f32) -> (u8, u8, u8) {
        let t = t.clamp(0.0_f32, 1.0_f32);
        match self {
            SurfaceColorMap::OceanThermal => {
                let r = (30.0_f32 + 205.0_f32 * t) as u8;
                let g = (105.0_f32 + 120.0_f32 * (1.0_f32 - (2.0_f32 * t - 1.0_f32).abs())) as u8;
                let b = (240.0_f32 - 180.0_f32 * t) as u8;
                (r, g, b)
            }
            SurfaceColorMap::Viridis => {
                let r = (68.0_f32 + 185.0_f32 * t * t) as u8;
                let g = (1.0_f32 + 215.0_f32 * t) as u8;
                let b = (84.0_f32 + 90.0_f32 * (1.0_f32 - (2.0_f32 * t - 1.0_f32).abs())
                    - 50.0_f32 * t)
                    .clamp(20.0_f32, 255.0_f32) as u8;
                (r, g, b)
            }
            SurfaceColorMap::CoolWarm => {
                let r = (59.0_f32 + 180.0_f32 * t) as u8;
                let g = (76.0_f32 + 110.0_f32 * (1.0_f32 - (2.0_f32 * t - 1.0_f32).abs())) as u8;
                let b = (192.0_f32 - 155.0_f32 * t) as u8;
                (r, g, b)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnippetTemplate {
    CobbDouglas,
    BudgetConstraint,
    IsLmMacro,
    SolowGrowth,
    MonopolistProfit,
    AsymptoticRational,
    SaddleSurface3D,
}

impl SnippetTemplate {
    fn all() -> [SnippetTemplate; 7] {
        [
            SnippetTemplate::CobbDouglas,
            SnippetTemplate::BudgetConstraint,
            SnippetTemplate::IsLmMacro,
            SnippetTemplate::SolowGrowth,
            SnippetTemplate::MonopolistProfit,
            SnippetTemplate::AsymptoticRational,
            SnippetTemplate::SaddleSurface3D,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            SnippetTemplate::CobbDouglas => "Cobb-Douglas Isoquant & Isocost",
            SnippetTemplate::BudgetConstraint => "Consumer Utility Maximization",
            SnippetTemplate::IsLmMacro => "IS-LM Macroeconomic Equilibrium",
            SnippetTemplate::SolowGrowth => "Solow-Swan Growth Model",
            SnippetTemplate::MonopolistProfit => "Monopoly MR = MC & Profit",
            SnippetTemplate::AsymptoticRational => "Rational & Exponential Functions (1/x, e^x)",
            SnippetTemplate::SaddleSurface3D => "3D Surface Z = f(x, y) Wave & Production",
        }
    }

    fn description(self) -> &'static str {
        match self {
            SnippetTemplate::CobbDouglas => {
                "Isoquant curve K(L) for target output Y_bar tangent to linear isocost constraint."
            }
            SnippetTemplate::BudgetConstraint => {
                "Marshallian consumer choice with indifference curve, budget line, and utility."
            }
            SnippetTemplate::IsLmMacro => {
                "Goods market (IS) and money market (LM) interest rate & output equilibrium."
            }
            SnippetTemplate::SolowGrowth => {
                "Neoclassical capital accumulation: investment s*f(k) vs break-even (delta+n)*k."
            }
            SnippetTemplate::MonopolistProfit => {
                "Imperfect competition inverse demand, marginal revenue, marginal cost, and profit."
            }
            SnippetTemplate::AsymptoticRational => {
                "Demonstrates clean asymptote splitting on 1/x and memory-safe e^x rendering."
            }
            SnippetTemplate::SaddleSurface3D => {
                "Two-variable function f(x, y) ready for interactive 3D Surface exploration."
            }
        }
    }

    fn default_domains(self) -> (f64, f64, f64, f64) {
        match self {
            SnippetTemplate::CobbDouglas => (0.2_f64, 25.0_f64, 0.0_f64, 25.0_f64),
            SnippetTemplate::BudgetConstraint => (0.2_f64, 30.0_f64, 0.0_f64, 28.0_f64),
            SnippetTemplate::IsLmMacro => (0.0_f64, 20.0_f64, 0.0_f64, 35.0_f64),
            SnippetTemplate::SolowGrowth => (0.0_f64, 25.0_f64, 0.0_f64, 4.0_f64),
            SnippetTemplate::MonopolistProfit => (0.0_f64, 25.0_f64, -20.0_f64, 120.0_f64),
            SnippetTemplate::AsymptoticRational => (-6.0_f64, 6.0_f64, -10.0_f64, 10.0_f64),
            SnippetTemplate::SaddleSurface3D => (-5.0_f64, 5.0_f64, -5.0_f64, 5.0_f64),
        }
    }

    fn model_kind(self) -> ActiveModelKind {
        match self {
            SnippetTemplate::CobbDouglas => ActiveModelKind::CobbDouglas,
            SnippetTemplate::BudgetConstraint => ActiveModelKind::ConstrainedUtility,
            SnippetTemplate::IsLmMacro => ActiveModelKind::CustomExpression,
            SnippetTemplate::SolowGrowth => ActiveModelKind::CustomExpression,
            SnippetTemplate::MonopolistProfit => ActiveModelKind::GameTheory2x2,
            SnippetTemplate::AsymptoticRational => ActiveModelKind::CustomExpression,
            SnippetTemplate::SaddleSurface3D => ActiveModelKind::CustomExpression,
        }
    }

    fn script_text(self) -> &'static str {
        match self {
            SnippetTemplate::CobbDouglas => {
                r##"# Cobb-Douglas Production & Isoquant-Isocost Tangency
A = 2.0
\alpha = 0.45
\beta = 0.55
price_x = 4.0
price_y = 5.0
I = 100.0
Y_bar = 18.0

f(x) = (Y_bar / (A * x^\alpha))^(1 / \beta)
g(x) = (I - price_x * x) / price_y"##
            }
            SnippetTemplate::BudgetConstraint => {
                r##"# Consumer Constrained Utility Maximization
A = 1.5
\alpha = 0.50
\beta = 0.50
price_x = 4.0
price_y = 5.0
I = 120.0
Y_bar = 16.4317

f(x) = (Y_bar / (A * x^\alpha))^(1 / \beta)
g(x) = (I - price_x * x) / price_y
u(x) = A * x^\alpha * g(x)^\beta"##
            }
            SnippetTemplate::IsLmMacro => {
                r##"# Macroeconomics IS-LM Equilibrium
c_0 = 2.5
c_1 = 0.65
I_0 = 3.2
b = 22.0
G = 4.0
T = 3.0
M_s = 12.0
P = 1.0
k = 0.55
h = 18.0

f(x) = (c_0 - c_1 * T + I_0 + G - (1.0 - c_1) * x) / b * 100.0
g(x) = (k * x - M_s / P) / h * 100.0"##
            }
            SnippetTemplate::SolowGrowth => {
                r##"# Solow-Swan Neoclassical Growth Model
A = 2.2
\alpha = 0.35
s = 0.28
\delta = 0.07
n = 0.02

f(x) = s * A * x^\alpha
g(x) = (\delta + n) * x
y(x) = A * x^\alpha"##
            }
            SnippetTemplate::MonopolistProfit => {
                r##"# Monopoly MR = MC & Profit
a = 28.0
b = 0.85
c_0 = 12.0
c_1 = 2.0
c_2 = 0.25

f(x) = a - 2.0 * b * x
g(x) = c_1 + 2.0 * c_2 * x
demand(x) = a - b * x
profit(x) = demand(x) * x - (c_0 + c_1 * x + c_2 * x^2)"##
            }
            SnippetTemplate::AsymptoticRational => {
                r##"# Asymptotic & Exponential Functions
k = 1.0
f(x) = k / x
g(x) = e^x
h(x) = 1 / (x^2 - 4)"##
            }
            SnippetTemplate::SaddleSurface3D => {
                r##"# 3D Surface Function f(x, y)
a = 0.35
b = 0.35
f(x, y) = sin(x) * cos(y) + a * x^2 - b * y^2"##
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ConsoleLogEntry {
    timestamp_tag: String,
    source_line: String,
    message: String,
    is_error: bool,
}

/// Bridge exposing active Desmos functions & slider parameters to the embedded R engine.
struct WorkspaceDesmosBridge<'a> {
    parameters: &'a IndexMap<String, f64>,
    compiled_math: &'a CompiledWorkspace,
    diff_h: f64,
    integral_steps: usize,
}

impl DesmosBridge for WorkspaceDesmosBridge<'_> {
    fn has_desmos_fn(&self, name: &str) -> bool {
        self.compiled_math.functions.contains_key(name)
    }

    fn call_desmos_fn(&self, name: &str, args: &[f64]) -> Option<Result<f64, String>> {
        let user_fn = self.compiled_math.functions.get(name)?;
        if args.len() != user_fn.params.len() {
            return Some(Err(format!(
                "Desmos function `{}` expects {} argument(s), got {}",
                name,
                user_fn.params.len(),
                args.len()
            )));
        }
        let ctx = EvalContext::new(self.parameters, &self.compiled_math.functions)
            .with_diff_h(self.diff_h)
            .with_integral_steps(self.integral_steps);
        let mut overrides: Vec<(&str, f64)> = Vec::with_capacity(args.len() + 2);
        for (p, &v) in user_fn.params.iter().zip(args.iter()) {
            overrides.push((p.as_str(), v));
        }
        if let Some(&x0) = args.first() {
            overrides.push(("L", x0));
            overrides.push(("Q", x0));
            overrides.push(("Y", x0));
        }
        let mut scratch = EvalScratch::new();
        Some(
            user_fn
                .body
                .eval(&ctx, &overrides, &mut scratch)
                .map_err(|e| e.message),
        )
    }

    fn get_desmos_param(&self, name: &str) -> Option<f64> {
        self.parameters.get(name).copied()
    }
}

// =============================================================================
// MAIN APPLICATION STATE
// =============================================================================

pub struct EconCalcApp {
    pub workspace: Workspace,

    // Theme & panel layout visibility
    dark_mode: bool,
    show_left_panel: bool,
    show_right_panel: bool,
    show_bottom_panel: bool,
    maximize_graph: bool,

    // Bento card collapse & sub-section states
    symbolic_card_collapsed: bool,
    left_bottom_collapsed: bool,
    env_card_collapsed: bool,
    right_bottom_collapsed: bool,
    symbolic_functions_open: bool,
    symbolic_sliders_open: bool,
    dataset_show_raw_csv: bool,

    // Floating pop-out windows
    floating_datasets_window: bool,
    floating_notes_window: bool,
    floating_help_window: bool,
    floating_env_window: bool,
    floating_console_window: bool,

    // Navigation tabs
    active_model: ActiveModelKind,
    viewport_mode: ViewportMode,
    axis_zoom_mode: AxisZoomMode,
    left_bottom_tab: LeftBottomTab,
    file_filter: FileFilterKind,
    bottom_tab: BottomDockTab,
    right_tab: RightDockTab,
    selected_snippet: SnippetTemplate,
    report_format: ReportFormat,

    // Quick Console & Script Editor
    quick_command_input: String,
    new_fn_sig_input: String,
    new_fn_expr_input: String,
    new_param_name_input: String,
    new_param_val_input: String,
    smart_math_shortcuts: bool,
    editor_script: String,
    console_logs: Vec<ConsoleLogEntry>,
    eval_counter: usize,
    status_banner: String,

    // Domain, Home View & Numerical Settings
    domain_min: f64,
    domain_max: f64,
    domain_y_min: f64,
    domain_y_max: f64,
    home_x_min: f64,
    home_x_max: f64,
    home_y_min: f64,
    home_y_max: f64,
    needs_plot_bounds_sync: bool,
    pending_zoom_factor: Option<[f32; 2]>,
    lock_equal_aspect: bool,
    show_plot_grid: bool,
    show_plot_legend: bool,

    // Numerical & approximation settings
    curve_samples: usize,
    sim_grid_rows: usize,
    diff_step_h: f64,
    simpson_subintervals: usize,
    asymptote_jump_factor: f64,

    // Opt-in Calculus & Equilibrium Overlays
    probe_x: f64,
    show_probe_marker: bool,
    show_derivative_overlay: bool,
    show_integral_shade: bool,
    show_equilibrium_overlay: bool,
    enable_equilibrium_solver: bool,

    // 3D Surface controls
    selected_3d_function: String,
    yaw_deg: f32,
    pitch_deg: f32,
    surface_zoom: f32,
    surface_resolution: usize,
    surface_wireframe_only: bool,
    surface_colormap: SurfaceColorMap,

    // File & Project Management + Import/Export paths
    project_dir: PathBuf,
    project_files: Vec<ProjectFileEntry>,
    file_io_path_input: String,
    new_dataset_name_input: String,
    notes_preview_mode: bool,
}

impl EconCalcApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_bento_theme(&cc.egui_ctx, true);

        let workspace = Workspace::new_default();
        let initial_script = SnippetTemplate::CobbDouglas.script_text().to_string();
        let project_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let project_files = list_project_directory(&project_dir);

        let mut app = Self {
            workspace,
            dark_mode: true,
            show_left_panel: true,
            show_right_panel: true,
            show_bottom_panel: true,
            maximize_graph: false,
            symbolic_card_collapsed: false,
            left_bottom_collapsed: false,
            env_card_collapsed: false,
            right_bottom_collapsed: false,
            symbolic_functions_open: true,
            symbolic_sliders_open: true,
            dataset_show_raw_csv: false,
            floating_datasets_window: false,
            floating_notes_window: false,
            floating_help_window: false,
            floating_env_window: false,
            floating_console_window: false,
            active_model: ActiveModelKind::CobbDouglas,
            viewport_mode: ViewportMode::Plot2D,
            axis_zoom_mode: AxisZoomMode::BothXY,
            left_bottom_tab: LeftBottomTab::Files,
            file_filter: FileFilterKind::All,
            bottom_tab: BottomDockTab::QuickConsole,
            right_tab: RightDockTab::Datasets,
            selected_snippet: SnippetTemplate::CobbDouglas,
            report_format: ReportFormat::QuartoQmd,
            quick_command_input: String::new(),
            new_fn_sig_input: "h(x)".to_string(),
            new_fn_expr_input: "x^2".to_string(),
            new_param_name_input: "k".to_string(),
            new_param_val_input: "1.0".to_string(),
            smart_math_shortcuts: true,
            editor_script: initial_script,
            console_logs: Vec::new(),
            eval_counter: 0,
            status_banner: "Ready — Type `intro-help` or `general-help` in Console."
                .to_string(),
            domain_min: 0.2_f64,
            domain_max: 25.0_f64,
            domain_y_min: 0.0_f64,
            domain_y_max: 25.0_f64,
            home_x_min: 0.2_f64,
            home_x_max: 25.0_f64,
            home_y_min: 0.0_f64,
            home_y_max: 25.0_f64,
            needs_plot_bounds_sync: true,
            pending_zoom_factor: None,
            lock_equal_aspect: false,
            show_plot_grid: true,
            show_plot_legend: true,
            curve_samples: 400,
            sim_grid_rows: 50,
            diff_step_h: 1e-5_f64,
            simpson_subintervals: 200,
            asymptote_jump_factor: 4.5_f64,
            probe_x: 11.25_f64,
            show_probe_marker: false,
            show_derivative_overlay: false,
            show_integral_shade: false,
            show_equilibrium_overlay: false,
            enable_equilibrium_solver: false,
            selected_3d_function: "f(x)".to_string(),
            yaw_deg: 38.0_f32,
            pitch_deg: 28.0_f32,
            surface_zoom: 1.0_f32,
            surface_resolution: 28,
            surface_wireframe_only: false,
            surface_colormap: SurfaceColorMap::OceanThermal,
            project_dir,
            project_files,
            file_io_path_input: "exports/notes.qmd".to_string(),
            new_dataset_name_input: "dataset_2".to_string(),
            notes_preview_mode: false,
        };

        app.evaluate_editor_script();
        app
    }

    fn palette(&self) -> ThemePalette {
        ThemePalette::for_mode(self.dark_mode)
    }

    fn clear_entire_workspace(&mut self) {
        self.workspace.clear();
        self.editor_script.clear();
        self.show_equilibrium_overlay = false;
        self.enable_equilibrium_solver = false;
        self.status_banner = "Workspace cleared.".to_string();
    }

    fn load_snippet_template(&mut self, snip: SnippetTemplate) {
        self.selected_snippet = snip;
        self.active_model = snip.model_kind();
        self.editor_script = snip.script_text().to_string();
        let (x0, x1, y0, y1) = snip.default_domains();
        self.home_x_min = x0;
        self.home_x_max = x1;
        self.home_y_min = y0;
        self.home_y_max = y1;
        self.reset_graph_to_home();
        self.workspace.parameters.clear();
        self.workspace.slider_specs.clear();
        self.workspace.functions.clear();
        self.workspace.function_styles.clear();
        self.workspace.sync_math_cache();
        self.evaluate_editor_script();
        if snip == SnippetTemplate::SaddleSurface3D {
            self.viewport_mode = ViewportMode::Surface3D;
        } else {
            self.viewport_mode = ViewportMode::Plot2D;
        }
        self.status_banner = format!("Loaded model: {}", snip.label());
    }

    fn generate_dataset_from_primary_fn(&mut self) {
        let grid = self.build_simulation_grid(self.sim_grid_rows);
        let mut csv_s = String::from("x,y\n");
        for r in &grid {
            if r.primary_y.is_finite() {
                csv_s.push_str(&format!("{:.5},{:.5}\n", r.x, r.primary_y));
            }
        }
        self.workspace.datasets.push(DataTable::from_csv_str(
            format!("sim_{}_rows", grid.len()),
            csv_s,
            Color32::from_rgb(52, 211, 153),
        ));
        self.workspace.active_dataset_idx = self.workspace.datasets.len() - 1;
        self.right_tab = RightDockTab::Datasets;
        self.show_right_panel = true;
        self.right_bottom_collapsed = false;
        self.status_banner = format!("Generated {} sampled rows from primary f(x).", grid.len());
    }

    fn save_editor_script_default(&mut self) {
        let path = PathBuf::from("exports/workspace_script.R");
        match save_text_file(&path, &self.editor_script) {
            Ok(()) => {
                self.status_banner = "Saved script to exports/workspace_script.R".to_string();
                self.refresh_project_files();
            }
            Err(e) => self.status_banner = e,
        }
    }

    fn save_notes_default(&mut self) {
        let path = PathBuf::from(self.file_io_path_input.trim());
        match save_text_file(&path, &self.workspace.markdown_notes) {
            Ok(()) => {
                self.status_banner = format!("Saved notes to {:?}", path);
                self.refresh_project_files();
            }
            Err(e) => self.status_banner = e,
        }
    }

    pub fn refresh_project_files(&mut self) {
        self.project_files = list_project_directory(&self.project_dir);
    }

    pub fn reset_graph_to_home(&mut self) {
        self.domain_min = self.home_x_min;
        self.domain_max = self.home_x_max;
        self.domain_y_min = self.home_y_min;
        self.domain_y_max = self.home_y_max;
        self.needs_plot_bounds_sync = true;
    }

    // =========================================================================
    // AST-BASED SYMBOLIC & COMPILED NUMERICAL ENGINE (src/math/)
    // =========================================================================

    /// Normalizes a raw identifier or LaTeX token (`\alpha`, `\bar{Y}`, `P_x`) into a
    /// canonical workspace key (`alpha`, `Y_bar`, `P_x`).
    pub fn normalize_ident_name(raw: &str) -> String {
        let trimmed = raw.trim();
        if let Ok(ast) = math::parse_expr(trimmed) {
            if let Some(canon) = ast.as_canonical_ident() {
                return canon;
            }
        }
        trimmed
            .trim_start_matches('\\')
            .replace(['{', '}', ' '], "")
    }

    /// Normalizes a function signature (`f(x)`, `u(x, y)`, `\phi(t)`), preserving parameters.
    pub fn normalize_fn_signature(raw: &str) -> String {
        let trimmed = raw.trim();
        if let Some((name, params)) = math::parse_function_signature(trimmed) {
            if params.is_empty() {
                return format!("{}(x)", Self::normalize_ident_name(&name));
            }
            let norm_params: Vec<String> = params
                .iter()
                .map(|p| Self::normalize_ident_name(p))
                .collect();
            return format!(
                "{}({})",
                Self::normalize_ident_name(&name),
                norm_params.join(", ")
            );
        }
        Self::normalize_ident_name(trimmed)
    }

    fn build_eval_context<'a>(
        &'a self,
        ext: &'a REngineMathBridge<'a>,
    ) -> EvalContext<'a> {
        EvalContext::new(&self.workspace.parameters, &self.workspace.compiled_math.functions)
            .with_external(ext)
            .with_diff_h(self.diff_step_h)
            .with_integral_steps(self.simpson_subintervals)
    }

    fn eval_parameter_rhs(&self, expr_str: &str) -> Result<f64, String> {
        let ast = math::parse_expr_with_known_fns(
            expr_str,
            &self.workspace.compiled_math.known_fn_names,
        )
        .map_err(|e| e.format_with_source(expr_str))?;
        let compiled = CompiledExpr::compile(&ast);
        let ext = REngineMathBridge {
            r_objects: &self.workspace.r_objects,
        };
        let ctx = self.build_eval_context(&ext);
        let mut scratch = EvalScratch::new();
        compiled
            .eval(&ctx, &[], &mut scratch)
            .map_err(|e| e.message)
    }

    pub fn eval_expression_str(
        &self,
        expr_str: &str,
        x_val: f64,
        y_val: f64,
    ) -> Result<f64, String> {
        let ast = math::parse_expr_with_known_fns(
            expr_str,
            &self.workspace.compiled_math.known_fn_names,
        )
        .map_err(|e| e.format_with_source(expr_str))?;
        let compiled = CompiledExpr::compile(&ast);
        let ext = REngineMathBridge {
            r_objects: &self.workspace.r_objects,
        };
        let ctx = self.build_eval_context(&ext);
        let mut scratch = EvalScratch::new();
        compiled
            .eval(
                &ctx,
                &[
                    ("x", x_val),
                    ("y", y_val),
                    ("L", x_val),
                    ("K", y_val),
                    ("Q", x_val),
                    ("Y", x_val),
                ],
                &mut scratch,
            )
            .map_err(|e| e.message)
    }

    /// Samples a 1D function across `[sample_min, sample_max]` with asymptote & piecewise
    /// discontinuity splitting using the compiled bytecode evaluator.
    pub fn sample_function_branches(
        &self,
        expr_str: &str,
        sample_min: f64,
        sample_max: f64,
        vis_y_min: f64,
        vis_y_max: f64,
        samples: usize,
    ) -> Vec<Vec<[f64; 2]>> {
        let Ok(ast) = math::parse_expr_with_known_fns(
            expr_str,
            &self.workspace.compiled_math.known_fn_names,
        ) else {
            return Vec::new();
        };
        let compiled = CompiledExpr::compile(&ast);
        let ext = REngineMathBridge {
            r_objects: &self.workspace.r_objects,
        };
        let ctx = self.build_eval_context(&ext);
        math::sample_1d_branches(
            &compiled,
            &ctx,
            sample_min,
            sample_max,
            vis_y_min,
            vis_y_max,
            samples,
            self.asymptote_jump_factor,
        )
    }

    // =========================================================================
    // R-LITE, QUICK MATH & HELP COMMAND INTERPRETER
    // =========================================================================

    pub fn execute_console_command(&mut self, raw_cmd: &str) -> String {
        let trimmed = raw_cmd.trim();
        if trimmed.is_empty() {
            return String::new();
        }

        self.eval_counter += 1;
        let tag = format!("In[{}]", self.eval_counter);

        if trimmed.eq_ignore_ascii_case("intro-help") || trimmed.eq_ignore_ascii_case("intro_help")
        {
            let msg = Self::intro_help_text();
            self.right_tab = RightDockTab::HelpCenter;
            self.show_right_panel = true;
            self.console_logs.push(ConsoleLogEntry {
                timestamp_tag: tag,
                source_line: trimmed.to_string(),
                message: msg.to_string(),
                is_error: false,
            });
            self.status_banner = "Opened `intro-help` Application Guide.".to_string();
            return msg.to_string();
        }

        if trimmed.eq_ignore_ascii_case("general-help")
            || trimmed.eq_ignore_ascii_case("general_help")
            || trimmed.eq_ignore_ascii_case("help")
        {
            let msg = Self::general_help_text();
            self.right_tab = RightDockTab::HelpCenter;
            self.show_right_panel = true;
            self.console_logs.push(ConsoleLogEntry {
                timestamp_tag: tag,
                source_line: trimmed.to_string(),
                message: msg.to_string(),
                is_error: false,
            });
            self.status_banner = "Opened `general-help` LaTeX, R & Desmos Reference.".to_string();
            return msg.to_string();
        }

        if trimmed == "ls()" {
            let names: Vec<String> = self
                .workspace
                .r_objects
                .keys()
                .map(|k| format!("\"{}\"", k))
                .collect();
            let out = if names.is_empty() {
                "character(0)".to_string()
            } else {
                format!("[1] {}", names.join(" "))
            };
            self.console_logs.push(ConsoleLogEntry {
                timestamp_tag: tag,
                source_line: trimmed.to_string(),
                message: out.clone(),
                is_error: false,
            });
            return out;
        }

        if trimmed == "clear()" || trimmed == "rm(list=ls())" {
            self.workspace.clear();
            let out = "Workspace cleared.".to_string();
            self.console_logs.push(ConsoleLogEntry {
                timestamp_tag: tag,
                source_line: trimmed.to_string(),
                message: out.clone(),
                is_error: false,
            });
            return out;
        }

        if let Some(inner) = trimmed
            .strip_prefix("rm(")
            .and_then(|s| s.strip_suffix(')'))
        {
            let name = inner.trim();
            self.workspace.r_objects.remove(name);
            self.workspace.parameters.remove(name);
            self.workspace.slider_specs.remove(name);
            self.workspace.functions.remove(name);
            let out = format!("Removed object `{}`", name);
            self.console_logs.push(ConsoleLogEntry {
                timestamp_tag: tag,
                source_line: trimmed.to_string(),
                message: out.clone(),
                is_error: false,
            });
            return out;
        }

        if let Some(inner) = trimmed
            .strip_prefix("plot(")
            .and_then(|s| s.strip_suffix(')'))
        {
            return self.handle_r_plot_command(&tag, trimmed, inner);
        }

        // Handle explicit Desmos function definition `f(x) = ...` or `u(x, y) = ...`
        if let Some((lhs, rhs)) = Self::split_assignment(trimmed) {
            let lhs_trim = lhs.trim();
            let rhs_trim = rhs.trim();
            if Self::is_desmos_function_lhs(lhs_trim) && !rhs_trim.starts_with("function") {
                let lhs_clean = Self::normalize_fn_signature(lhs_trim);
                match self.eval_expression_str(rhs_trim, self.probe_x, 1.0_f64) {
                    Ok(v) => {
                        self.workspace
                            .set_function(lhs_clean.clone(), rhs_trim.to_string());
                        let out = format!(
                            "Function `{}` = `{}` (@ x={:.2} -> {:.4})",
                            lhs_clean, rhs_trim, self.probe_x, v
                        );
                        self.console_logs.push(ConsoleLogEntry {
                            timestamp_tag: tag,
                            source_line: trimmed.to_string(),
                            message: out.clone(),
                            is_error: false,
                        });
                        return out;
                    }
                    Err(e) => {
                        let out = format!("Error in `{}`: {}", lhs_clean, e);
                        self.console_logs.push(ConsoleLogEntry {
                            timestamp_tag: tag,
                            source_line: trimmed.to_string(),
                            message: out.clone(),
                            is_error: true,
                        });
                        return out;
                    }
                }
            }
            // Handle LaTeX parameter assignment `\alpha = 0.45`
            if lhs_trim.starts_with('\\') {
                let param_key = Self::normalize_ident_name(lhs_trim);
                if let Ok(v) = self.eval_parameter_rhs(rhs_trim) {
                    if v.is_finite() {
                        self.workspace.set_parameter(param_key.clone(), v);
                        let out = format!("{} = {:.6}", param_key, v);
                        self.console_logs.push(ConsoleLogEntry {
                            timestamp_tag: tag,
                            source_line: trimmed.to_string(),
                            message: out.clone(),
                            is_error: false,
                        });
                        return out;
                    }
                }
            }
        }

        // Execute using the embedded AST-based R engine with live Desmos bridge
        match self.execute_r_code_in_workspace(trimmed) {
            Ok((obj, stdout)) => {
                // Sync scalar numeric assignments (`a <- 5` or `a = 5`) to Desmos sliders
                if let Some((lhs, _)) = Self::split_assignment(trimmed) {
                    let lhs_t = lhs.trim();
                    if Self::is_plain_ident(lhs_t) {
                        if let Some(val) = obj.as_scalar_f64() {
                            if val.is_finite() {
                                self.workspace.set_parameter(lhs_t.to_string(), val);
                            }
                        }
                    }
                }
                let mut out = if stdout.is_empty() {
                    obj.preview_string()
                } else {
                    format!("{}\n{}", stdout.join("\n"), obj.preview_string())
                };
                if out.trim().is_empty() {
                    out = "NULL".to_string();
                }
                self.console_logs.push(ConsoleLogEntry {
                    timestamp_tag: tag,
                    source_line: trimmed.to_string(),
                    message: out.clone(),
                    is_error: false,
                });
                self.status_banner = format!("Result: {}", out.lines().next().unwrap_or("OK"));
                out
            }
            Err(r_err) => {
                // Fallback to Desmos math evaluator (for LaTeX, implicit multiplication, or `x` curves)
                if let Ok(val) = self.eval_parameter_rhs(trimmed) {
                    if val.is_finite() {
                        let out = format!("[1] {:.6}", val);
                        self.console_logs.push(ConsoleLogEntry {
                            timestamp_tag: tag,
                            source_line: trimmed.to_string(),
                            message: out.clone(),
                            is_error: false,
                        });
                        self.status_banner = format!("Result: {}", out);
                        return out;
                    }
                }
                if trimmed.contains('x') || trimmed.contains('y') || trimmed.contains("theta") {
                    if let Ok(sample) = self.eval_expression_str(trimmed, 1.0_f64, 1.0_f64) {
                        let auto_sig = format!("f{}(x)", self.workspace.functions.len() + 1);
                        self.workspace
                            .set_function(auto_sig.clone(), trimmed.to_string());
                        let out = format!(
                            "Plotted `{}` = `{}` (f(1) = {:.4})",
                            auto_sig, trimmed, sample
                        );
                        self.console_logs.push(ConsoleLogEntry {
                            timestamp_tag: tag,
                            source_line: trimmed.to_string(),
                            message: out.clone(),
                            is_error: false,
                        });
                        return out;
                    }
                }

                let err_msg = format!("Evaluation error: {}", r_err);
                self.console_logs.push(ConsoleLogEntry {
                    timestamp_tag: tag,
                    source_line: trimmed.to_string(),
                    message: err_msg.clone(),
                    is_error: true,
                });
                err_msg
            }
        }
    }

    fn is_desmos_function_lhs(lhs: &str) -> bool {
        let Some((name, rest)) = lhs.split_once('(') else {
            return false;
        };
        if !rest.ends_with(')') {
            return false;
        }
        let n = name.trim().trim_start_matches('\\');
        if n.is_empty()
            || matches!(
                n,
                "if" | "for" | "while" | "function" | "return" | "c" | "list" | "data.frame"
            )
        {
            return false;
        }
        n.chars().all(|c| c.is_alphanumeric() || c == '_')
    }

    fn is_plain_ident(s: &str) -> bool {
        !s.is_empty()
            && !s.contains(' ')
            && !s.contains('[')
            && !s.contains('$')
            && !s.contains('(')
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    }

    fn split_assignment(line: &str) -> Option<(&str, &str)> {
        let mut depth = 0usize;
        let bytes = line.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth = depth.saturating_sub(1),
                b'<' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'-' => {
                    return Some((&line[..i], &line[i + 2..]));
                }
                b'=' if depth == 0 => {
                    let prev = if i > 0 { bytes[i - 1] } else { 0 };
                    let next = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
                    if !matches!(prev, b'<' | b'>' | b'!' | b'=') && next != b'=' {
                        return Some((&line[..i], &line[i + 1..]));
                    }
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// Executes R code mutating `self.workspace.r_objects` with live access to Desmos functions & sliders.
    fn execute_r_code_in_workspace(&mut self, code: &str) -> Result<(RObject, Vec<String>), String> {
        let bridge = WorkspaceDesmosBridge {
            parameters: &self.workspace.parameters,
            compiled_math: &self.workspace.compiled_math,
            diff_h: self.diff_step_h,
            integral_steps: self.simpson_subintervals,
        };
        let res = EmbeddedREngine::eval_in_env(&mut self.workspace.r_objects, code, Some(&bridge))
            .map_err(|e| e.message)?;
        Ok((res.value, res.stdout))
    }

    /// Evaluates an R expression read-only against the current workspace environment and Desmos bridge.
    pub fn eval_r_expression(&self, expr: &str) -> Result<RObject, String> {
        let bridge = WorkspaceDesmosBridge {
            parameters: &self.workspace.parameters,
            compiled_math: &self.workspace.compiled_math,
            diff_h: self.diff_step_h,
            integral_steps: self.simpson_subintervals,
        };
        EmbeddedREngine::eval_readonly(&self.workspace.r_objects, expr, Some(&bridge))
            .map_err(|e| e.message)
    }

    fn split_top_level_args(inner: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut depth = 0usize;
        let mut start = 0usize;
        for (i, ch) in inner.char_indices() {
            match ch {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth = depth.saturating_sub(1),
                ',' if depth == 0 => {
                    out.push(inner[start..i].trim());
                    start = i + 1;
                }
                _ => {}
            }
        }
        let tail = inner[start..].trim();
        if !tail.is_empty() {
            out.push(tail);
        }
        out
    }

    fn handle_r_plot_command(&mut self, tag: &str, raw_cmd: &str, inner: &str) -> String {
        let args = Self::split_top_level_args(inner);
        let (xs, ys, label) = if args.len() == 1 {
            match self.eval_r_expression(args[0]) {
                Ok(obj) => {
                    let y = obj.as_f64_vec();
                    let x: Vec<f64> = (1..=y.len()).map(|i| i as f64).collect();
                    (x, y, format!("plot({})", args[0]))
                }
                Err(e) => return format!("plot error: {}", e),
            }
        } else if args.len() >= 2 {
            match (
                self.eval_r_expression(args[0]),
                self.eval_r_expression(args[1]),
            ) {
                (Ok(ox), Ok(oy)) => (
                    ox.as_f64_vec(),
                    oy.as_f64_vec(),
                    format!("plot({}, {})", args[0], args[1]),
                ),
                (Err(e), _) | (_, Err(e)) => return format!("plot error: {}", e),
            }
        } else {
            return "Usage: plot(x, y) or plot(y)".to_string();
        };

        let len = xs.len().min(ys.len());
        let mut pts = Vec::with_capacity(len);
        let mut raw_csv = String::from("x,y\n");
        for i in 0..len {
            pts.push([xs[i], ys[i]]);
            raw_csv.push_str(&format!("{:.6},{:.6}\n", xs[i], ys[i]));
        }

        let mut ds = DataTable {
            name: label.clone(),
            points: pts,
            raw_csv,
            show_on_plot: true,
            show_ols_line: false,
            color: Color32::from_rgb(16, 185, 129),
        };
        ds.sync_from_raw_csv();
        self.workspace.datasets.push(ds);
        self.workspace.active_dataset_idx = self.workspace.datasets.len() - 1;

        let msg = format!("Plotted {} points as dataset `{}` on 2D Graph.", len, label);
        self.console_logs.push(ConsoleLogEntry {
            timestamp_tag: tag.to_string(),
            source_line: raw_cmd.to_string(),
            message: msg.clone(),
            is_error: false,
        });
        msg
    }

    /// Groups multi-line R blocks (`{ ... }`, `( ... )`, trailing pipes) into complete chunks.
    fn split_script_chunks(script: &str) -> Vec<(usize, String)> {
        let mut chunks = Vec::new();
        let mut buf = String::new();
        let mut start_line = 1usize;
        let mut brace_depth = 0i32;
        let mut paren_depth = 0i32;
        let mut bracket_depth = 0i32;

        for (idx, raw_line) in script.lines().enumerate() {
            let line_num = idx + 1;
            let trimmed = raw_line.trim();
            if buf.is_empty() {
                if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
                    continue;
                }
                start_line = line_num;
            }
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(raw_line);

            let mut in_str: Option<char> = None;
            for ch in raw_line.chars() {
                if let Some(q) = in_str {
                    if ch == q {
                        in_str = None;
                    }
                    continue;
                }
                match ch {
                    '#' => break,
                    '"' | '\'' => in_str = Some(ch),
                    '{' => brace_depth += 1,
                    '}' => brace_depth -= 1,
                    '(' => paren_depth += 1,
                    ')' => paren_depth -= 1,
                    '[' => bracket_depth += 1,
                    ']' => bracket_depth -= 1,
                    _ => {}
                }
            }

            let code_before_comment = raw_line.split('#').next().unwrap_or("").trim_end();
            let continues = code_before_comment.ends_with("|>")
                || code_before_comment.ends_with("<-")
                || code_before_comment.ends_with('+')
                || code_before_comment.ends_with(',');

            if brace_depth <= 0 && paren_depth <= 0 && bracket_depth <= 0 && !continues {
                chunks.push((start_line, std::mem::take(&mut buf)));
                brace_depth = 0;
                paren_depth = 0;
                bracket_depth = 0;
            }
        }
        if !buf.trim().is_empty() {
            chunks.push((start_line, buf));
        }
        chunks
    }

    /// Evaluates the multiline Script Editor (`evaluate_editor_script`), supporting mixed
    /// Desmos symbolic models and full multi-line R scripts (`function`, `for`, `lm`, `|>`).
    pub fn evaluate_editor_script(&mut self) {
        self.eval_counter += 1;
        let run_id = self.eval_counter;
        let script_copy = self.editor_script.clone();
        let chunks = Self::split_script_chunks(&script_copy);

        let mut parsed_params = 0usize;
        let mut parsed_funcs = 0usize;
        let mut error_count = 0usize;

        for (line_num, chunk) in chunks {
            let trimmed = chunk.trim();
            if trimmed.eq_ignore_ascii_case("intro-help")
                || trimmed.eq_ignore_ascii_case("general-help")
            {
                self.execute_console_command(trimmed);
                continue;
            }

            // Single-line Desmos function definition or LaTeX parameter assignment
            if !trimmed.contains('\n') {
                if let Some((lhs_raw, rhs_raw)) = Self::split_assignment(trimmed) {
                    let lhs_trim = lhs_raw.trim();
                    let rhs_trim = rhs_raw.trim();

                    if Self::is_desmos_function_lhs(lhs_trim) && !rhs_trim.starts_with("function") {
                        let lhs_clean = Self::normalize_fn_signature(lhs_trim);
                        match self.eval_expression_str(rhs_trim, self.probe_x, 1.0_f64) {
                            Ok(sample_val) => {
                                self.workspace
                                    .set_function(lhs_clean.clone(), rhs_trim.to_string());
                                parsed_funcs += 1;
                                self.console_logs.push(ConsoleLogEntry {
                                    timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                                    source_line: trimmed.to_string(),
                                    message: format!(
                                        "Bound `{}` = `{}` (probe -> {:.4})",
                                        lhs_clean, rhs_trim, sample_val
                                    ),
                                    is_error: false,
                                });
                            }
                            Err(err) => {
                                error_count += 1;
                                self.console_logs.push(ConsoleLogEntry {
                                    timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                                    source_line: trimmed.to_string(),
                                    message: format!("Function `{}` error: {}", lhs_clean, err),
                                    is_error: true,
                                });
                            }
                        }
                        continue;
                    }

                    if lhs_trim.starts_with('\\') {
                        let param_key = Self::normalize_ident_name(lhs_trim);
                        match self.eval_parameter_rhs(rhs_trim) {
                            Ok(val) if val.is_finite() => {
                                self.workspace.set_parameter(param_key.clone(), val);
                                parsed_params += 1;
                                self.console_logs.push(ConsoleLogEntry {
                                    timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                                    source_line: trimmed.to_string(),
                                    message: format!("{} = {:.6}", param_key, val),
                                    is_error: false,
                                });
                            }
                            Ok(_) => {
                                error_count += 1;
                            }
                            Err(err) => {
                                error_count += 1;
                                self.console_logs.push(ConsoleLogEntry {
                                    timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                                    source_line: trimmed.to_string(),
                                    message: format!("Parameter `{}` error: {}", param_key, err),
                                    is_error: true,
                                });
                            }
                        }
                        continue;
                    }
                }
            }

            // Execute chunk via the embedded R engine
            match self.execute_r_code_in_workspace(trimmed) {
                Ok((obj, stdout)) => {
                    if let Some((lhs_raw, _)) = Self::split_assignment(trimmed) {
                        let lhs_t = lhs_raw.trim();
                        if Self::is_plain_ident(lhs_t) {
                            if let Some(val) = obj.as_scalar_f64() {
                                if val.is_finite() {
                                    self.workspace.set_parameter(lhs_t.to_string(), val);
                                }
                            }
                            parsed_params += 1;
                        }
                    }
                    let msg = if stdout.is_empty() {
                        obj.preview_string()
                    } else {
                        format!("{}\n{}", stdout.join("\n"), obj.preview_string())
                    };
                    self.console_logs.push(ConsoleLogEntry {
                        timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                        source_line: trimmed.lines().next().unwrap_or(trimmed).to_string(),
                        message: msg,
                        is_error: false,
                    });
                }
                Err(r_err) => {
                    // Fallback: try promoting `name = expr(x)` to a Desmos function
                    if !trimmed.contains('\n') {
                        if let Some((lhs_raw, rhs_raw)) = Self::split_assignment(trimmed) {
                            let param_key = Self::normalize_ident_name(lhs_raw);
                            let rhs_trim = rhs_raw.trim();
                            if let Ok(sample_val) =
                                self.eval_expression_str(rhs_trim, self.probe_x, 1.0_f64)
                            {
                                let promoted_sig = format!("{}(x)", param_key);
                                self.workspace
                                    .set_function(promoted_sig.clone(), rhs_trim.to_string());
                                parsed_funcs += 1;
                                self.console_logs.push(ConsoleLogEntry {
                                    timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                                    source_line: trimmed.to_string(),
                                    message: format!(
                                        "Promoted `{}` = `{}` (probe -> {:.4})",
                                        promoted_sig, rhs_trim, sample_val
                                    ),
                                    is_error: false,
                                });
                                continue;
                            }
                        }
                    }
                    error_count += 1;
                    self.console_logs.push(ConsoleLogEntry {
                        timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                        source_line: trimmed.lines().next().unwrap_or(trimmed).to_string(),
                        message: format!("Error: {}", r_err),
                        is_error: true,
                    });
                }
            }
        }

        let first_fn_opt = self.workspace.functions.keys().next().cloned();
        if let Some(first_fn) = first_fn_opt {
            self.selected_3d_function = first_fn;
        }

        self.status_banner = format!(
            "Run #{:02}: {} parameter/object(s), {} function(s), {} error(s)",
            run_id, parsed_params, parsed_funcs, error_count
        );
    }

    // =========================================================================
    // NUMERICAL CALCULUS, OLS & OPT-IN EQUILIBRIUM DIAGNOSTICS
    // =========================================================================

    fn numerical_derivative(&self, expr_str: &str, x: f64) -> Result<f64, String> {
        let base_h = self.diff_step_h.clamp(1e-9_f64, 1e-2_f64);
        let h = base_h.max(x.abs() * base_h * 0.1_f64);
        let f_plus = self.eval_expression_str(expr_str, x + h, 10.0_f64)?;
        let f_minus = self.eval_expression_str(expr_str, x - h, 10.0_f64)?;
        Ok((f_plus - f_minus) / (2.0_f64 * h))
    }

    fn numerical_second_derivative(&self, expr_str: &str, x: f64) -> Result<f64, String> {
        let base_h = (self.diff_step_h * 10.0_f64).clamp(1e-7_f64, 1e-2_f64);
        let h = base_h.max(x.abs() * base_h * 0.1_f64);
        let f_plus = self.eval_expression_str(expr_str, x + h, 10.0_f64)?;
        let f_mid = self.eval_expression_str(expr_str, x, 10.0_f64)?;
        let f_minus = self.eval_expression_str(expr_str, x - h, 10.0_f64)?;
        Ok((f_plus - 2.0_f64 * f_mid + f_minus) / (h * h))
    }

    fn composite_simpson(&self, expr_str: &str, a: f64, b: f64, n: usize) -> Result<f64, String> {
        let steps = if n.is_multiple_of(2) { n.max(4) } else { n + 1 };
        let h = (b - a) / (steps as f64);
        let mut sum = self.eval_expression_str(expr_str, a, 10.0_f64)?
            + self.eval_expression_str(expr_str, b, 10.0_f64)?;

        for i in 1..steps {
            let x = a + (i as f64) * h;
            let fx = self.eval_expression_str(expr_str, x, 10.0_f64)?;
            if !fx.is_finite() {
                return Err(format!("Non-finite integrand at x = {:.4}", x));
            }
            if i % 2 == 1 {
                sum += 4.0_f64 * fx;
            } else {
                sum += 2.0_f64 * fx;
            }
        }
        Ok(sum * h / 3.0_f64)
    }

    fn find_primary_secondary_intersection(&self) -> Option<(f64, f64)> {
        if self.workspace.functions.len() < 2 {
            return None;
        }
        let (f_expr, g_expr) = {
            let mut iter = self.workspace.functions.values();
            let f = iter.next()?.clone();
            let g = iter.next()?.clone();
            (f, g)
        };

        let samples = 240usize;
        let span = (self.domain_max - self.domain_min).max(1e-3_f64);
        let step = span / (samples as f64);

        let eval_diff = |x: f64| -> Option<f64> {
            let fy = self.eval_expression_str(&f_expr, x, 10.0_f64).ok()?;
            let gy = self.eval_expression_str(&g_expr, x, 10.0_f64).ok()?;
            let d = fy - gy;
            d.is_finite().then_some(d)
        };

        let mut prev_x = self.domain_min;
        let mut prev_d = eval_diff(prev_x);

        for i in 1..=samples {
            let curr_x = self.domain_min + (i as f64) * step;
            let curr_d = eval_diff(curr_x);
            if let (Some(pd), Some(cd)) = (prev_d, curr_d) {
                if pd * cd <= 0.0_f64 {
                    let mut lo = prev_x;
                    let mut hi = curr_x;
                    let mut d_lo = pd;
                    for _ in 0..48 {
                        let mid = 0.5_f64 * (lo + hi);
                        let Some(d_mid) = eval_diff(mid) else {
                            break;
                        };
                        if d_mid.abs() < 1e-9_f64 || (hi - lo).abs() < 1e-9_f64 {
                            let y_val = self.eval_expression_str(&f_expr, mid, 10.0_f64).ok()?;
                            return (y_val.is_finite()).then_some((mid, y_val));
                        }
                        if d_lo * d_mid <= 0.0_f64 {
                            hi = mid;
                        } else {
                            lo = mid;
                            d_lo = d_mid;
                        }
                    }
                    let mid = 0.5_f64 * (lo + hi);
                    if let Ok(y_val) = self.eval_expression_str(&f_expr, mid, 10.0_f64) {
                        if y_val.is_finite() {
                            return Some((mid, y_val));
                        }
                    }
                }
            }
            prev_x = curr_x;
            prev_d = curr_d;
        }
        None
    }

    /// Returns `None` if workspace is cleared or empty.
    pub fn solve_structural_equilibrium_opt(&self) -> Option<(f64, f64, f64, f64)> {
        if self.workspace.functions.is_empty() && self.workspace.parameters.is_empty() {
            return None;
        }

        let tfp_a = self
            .workspace
            .parameters
            .get("A")
            .copied()
            .unwrap_or(2.0_f64)
            .max(1e-6_f64);
        let alpha = self
            .workspace
            .parameters
            .get("alpha")
            .copied()
            .unwrap_or(0.45_f64)
            .max(1e-6_f64);
        let beta = self
            .workspace
            .parameters
            .get("beta")
            .copied()
            .unwrap_or(0.55_f64)
            .max(1e-6_f64);
        let px = self
            .workspace
            .parameters
            .get("price_x")
            .copied()
            .unwrap_or(4.0_f64)
            .max(1e-6_f64);
        let py = self
            .workspace
            .parameters
            .get("price_y")
            .copied()
            .unwrap_or(5.0_f64)
            .max(1e-6_f64);
        let income = self
            .workspace
            .parameters
            .get("I")
            .copied()
            .unwrap_or(100.0_f64)
            .max(1e-6_f64);

        if let Some((ix, iy)) = self.find_primary_secondary_intersection() {
            let indirect_u = tfp_a * ix.max(1e-6_f64).powf(alpha) * iy.max(1e-6_f64).powf(beta);
            let mu_x = tfp_a
                * alpha
                * ix.max(1e-6_f64).powf(alpha - 1.0_f64)
                * iy.max(1e-6_f64).powf(beta);
            let lambda_star = mu_x / px;
            return Some((ix, iy, lambda_star, indirect_u));
        }

        if self.workspace.parameters.get("alpha").is_some()
            && self.workspace.parameters.get("beta").is_some()
        {
            let sum_elast = (alpha + beta).max(1e-9_f64);
            let x_star = (alpha / sum_elast) * (income / px);
            let y_star = (beta / sum_elast) * (income / py);
            let indirect_u =
                tfp_a * x_star.max(1e-9_f64).powf(alpha) * y_star.max(1e-9_f64).powf(beta);
            let mu_x = tfp_a
                * alpha
                * x_star.max(1e-9_f64).powf(alpha - 1.0_f64)
                * y_star.max(1e-9_f64).powf(beta);
            let lambda_star = mu_x / px;
            return Some((x_star, y_star, lambda_star, indirect_u));
        }

        None
    }

    fn compute_hessian_diagnostics(&self, x: f64, y: f64) -> (Matrix2<f64>, Vector2<f64>, f64, f64) {
        let tfp_a = self.workspace.parameters.get("A").copied().unwrap_or(2.0_f64);
        let alpha = self
            .workspace
            .parameters
            .get("alpha")
            .copied()
            .unwrap_or(0.45_f64);
        let beta = self
            .workspace
            .parameters
            .get("beta")
            .copied()
            .unwrap_or(0.55_f64);

        let xs = x.max(1e-4_f64);
        let ys = y.max(1e-4_f64);
        let f_val = tfp_a * xs.powf(alpha) * ys.powf(beta);

        let grad = Vector2::new(alpha * f_val / xs, beta * f_val / ys);
        let f_xx = alpha * (alpha - 1.0_f64) * f_val / (xs * xs);
        let f_yy = beta * (beta - 1.0_f64) * f_val / (ys * ys);
        let f_xy = alpha * beta * f_val / (xs * ys);

        let hessian = Matrix2::new(f_xx, f_xy, f_xy, f_yy);
        (hessian, grad, hessian.determinant(), hessian.trace())
    }

    fn compute_points_ols(pts: &[[f64; 2]]) -> Option<(f64, f64, f64)> {
        if pts.len() < 2 {
            return None;
        }
        let n = pts.len() as f64;
        let mean_x = pts.iter().map(|p| p[0]).sum::<f64>() / n;
        let mean_y = pts.iter().map(|p| p[1]).sum::<f64>() / n;

        let mut cov_xy = 0.0_f64;
        let mut var_x = 0.0_f64;
        let mut var_y = 0.0_f64;

        for p in pts {
            let dx = p[0] - mean_x;
            let dy = p[1] - mean_y;
            cov_xy += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x.abs() < 1e-12_f64 {
            return None;
        }
        let beta1 = cov_xy / var_x;
        let beta0 = mean_y - beta1 * mean_x;
        let r2 = if var_y.abs() > 1e-12_f64 {
            (cov_xy * cov_xy) / (var_x * var_y)
        } else {
            1.0_f64
        };
        Some((beta1, beta0, r2))
    }

    fn build_simulation_grid(&self, rows_count: usize) -> Vec<GridRow> {
        let rows_count = rows_count.clamp(2, 2000);
        let mut rows = Vec::with_capacity(rows_count);
        let span = (self.domain_max - self.domain_min).max(1e-3_f64);

        let (primary_expr, secondary_expr) = {
            let mut fn_iter = self.workspace.functions.values();
            let p = fn_iter.next().cloned().unwrap_or_else(|| "0.0".to_string());
            let s = fn_iter.next().cloned().unwrap_or_else(|| "0.0".to_string());
            (p, s)
        };

        for i in 0..rows_count {
            let x = self.domain_min
                + span * (i as f64) / ((rows_count.saturating_sub(1)).max(1) as f64);
            let primary_y = self
                .eval_expression_str(&primary_expr, x, 10.0_f64)
                .unwrap_or(f64::NAN);
            let secondary_y = self
                .eval_expression_str(&secondary_expr, x, 10.0_f64)
                .unwrap_or(f64::NAN);
            let marginal_or_slope = self
                .numerical_derivative(&primary_expr, x)
                .unwrap_or(f64::NAN);

            rows.push(GridRow {
                index: i + 1,
                x,
                primary_y,
                secondary_y,
                marginal_or_slope,
            });
        }
        rows
    }

    // =========================================================================
    // FORMATTING, LATEX, R PREVIEW & HELP TEXTS
    // =========================================================================

    fn format_symbolic_math(raw: &str) -> String {
        raw.replace("\\alpha", "α")
            .replace("alpha", "α")
            .replace("\\beta", "β")
            .replace("beta", "β")
            .replace("\\gamma", "γ")
            .replace("gamma", "γ")
            .replace("\\delta", "δ")
            .replace("delta", "δ")
            .replace("\\lambda", "λ")
            .replace("lambda", "λ")
            .replace("\\sigma", "σ")
            .replace("sigma", "σ")
            .replace("\\theta", "θ")
            .replace("theta", "θ")
            .replace("\\rho", "ρ")
            .replace("rho", "ρ")
            .replace("price_x", "Pₓ")
            .replace("price_y", "Pᵧ")
            .replace("Y_bar", "Ȳ")
    }

    fn format_latex_math(raw: &str) -> String {
        math::to_latex_from_ast(raw)
    }

    fn substitute_params_for_tikz(&self, raw_expr: &str) -> String {
        math::format_for_tikz(raw_expr, &self.workspace.parameters)
    }

    fn generate_latex_derivation(&self) -> String {
        let (x_star, y_star, lambda_star, indirect_u) = self
            .solve_structural_equilibrium_opt()
            .unwrap_or((0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64));
        let (hessian, grad, det_h, trace_h) = self.compute_hessian_diagnostics(x_star, y_star);

        let (primary_sig, primary_expr, secondary_sig, secondary_expr) = {
            let mut fn_iter = self.workspace.functions.iter();
            let (ps, pe) = fn_iter
                .next()
                .map(|(k, v)| (k.clone(), v.clone()))
                .unwrap_or_else(|| ("f(x)".to_string(), "0".to_string()));
            let (ss, se) = fn_iter
                .next()
                .map(|(k, v)| (k.clone(), v.clone()))
                .unwrap_or_else(|| ("g(x)".to_string(), "0".to_string()));
            (ps, pe, ss, se)
        };

        let fx_probe = self
            .eval_expression_str(&primary_expr, self.probe_x, y_star)
            .unwrap_or(0.0_f64);
        let dfx_probe = self
            .numerical_derivative(&primary_expr, self.probe_x)
            .unwrap_or(0.0_f64);
        let d2fx_probe = self
            .numerical_second_derivative(&primary_expr, self.probe_x)
            .unwrap_or(0.0_f64);
        let definite_int = self
            .composite_simpson(
                &primary_expr,
                self.domain_min,
                self.domain_max,
                self.simpson_subintervals,
            )
            .unwrap_or(0.0_f64);

        let mut param_lines = String::new();
        for (k, v) in &self.workspace.parameters {
            param_lines.push_str(&format!(
                "  {} &= {:.4} \\\\\n",
                Self::format_latex_math(k),
                v
            ));
        }

        let mut func_lines = String::new();
        for (sig, expr) in &self.workspace.functions {
            func_lines.push_str(&format!(
                "  {} &= {} \\\\\n",
                Self::format_latex_math(sig),
                Self::format_latex_math(expr)
            ));
        }

        let tikz_primary = self.substitute_params_for_tikz(&primary_expr);
        let tikz_secondary = self.substitute_params_for_tikz(&secondary_expr);

        format!(
            r##"\begin{{align*}}
% --- Symbolic Functions ---
{func_lines}% --- Parameters ---
{param_lines}% --- Calculus Diagnostics ---
  {primary_sig}\big|_{{x={probe:.3}}} &= {fx_probe:.6}, \quad \frac{{d}}{{dx}}{primary_sig}\big|_{{x={probe:.3}}} = {dfx_probe:.6}, \quad \frac{{d^2}}{{dx^2}}{primary_sig}\big|_{{x={probe:.3}}} = {d2fx_probe:.6} \\
  \int_{{{dmin:.2}}}^{{{dmax:.2}}} {primary_sig}\,dx &\approx {definite_int:.6} \\
  \left(x^*, y^*\right) &= \left({x_star:.5}, {y_star:.5}\right), \quad \lambda^* = {lambda_star:.5}, \quad V^* = {indirect_u:.5} \\
  \nabla F(x^*, y^*) &= \begin{{bmatrix}} {gx:.5} \\ {gy:.5} \end{{bmatrix}}, \quad
  \mathbf{{H}}(x^*, y^*) = \begin{{bmatrix}} {h00:.5} & {h01:.5} \\ {h10:.5} & {h11:.5} \end{{bmatrix}}, \quad
  \det(\mathbf{{H}}) = {det_h:.5e}, \quad \operatorname{{tr}}(\mathbf{{H}}) = {trace_h:.5}
\end{{align*}}

\begin{{tikzpicture}}
\begin{{axis}}[
    title={{Equilibrium ({primary_sig} vs {secondary_sig})}},
    xlabel={{$x$}},
    ylabel={{$y$}},
    domain={dmin:.2}:{dmax:.2},
    samples=120,
    legend pos=north east,
    grid=major
]
\addplot[blue, thick] {{{tikz_primary}}};
\addlegendentry{{${primary_sig}$}}
\addplot[red, dashed, thick] {{{tikz_secondary}}};
\addlegendentry{{${secondary_sig}$}}
\end{{axis}}
\end{{tikzpicture}}
"##,
            func_lines = func_lines,
            param_lines = param_lines,
            primary_sig = primary_sig,
            secondary_sig = secondary_sig,
            tikz_primary = tikz_primary,
            tikz_secondary = tikz_secondary,
            probe = self.probe_x,
            fx_probe = fx_probe,
            dfx_probe = dfx_probe,
            d2fx_probe = d2fx_probe,
            dmin = self.domain_min,
            dmax = self.domain_max,
            definite_int = definite_int,
            x_star = x_star,
            y_star = y_star,
            lambda_star = lambda_star,
            indirect_u = indirect_u,
            gx = grad.x,
            gy = grad.y,
            h00 = hessian[(0, 0)],
            h01 = hessian[(0, 1)],
            h10 = hessian[(1, 0)],
            h11 = hessian[(1, 1)],
            det_h = det_h,
            trace_h = trace_h,
        )
    }

    fn generate_r_script_preview(&self) -> String {
        let (x_star, y_star, lambda_star, indirect_u) = self
            .solve_structural_equilibrium_opt()
            .unwrap_or((0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64));
        let mut custom_param_list = Vec::new();
        for (k, v) in &self.workspace.parameters {
            custom_param_list.push(format!("  {} = {:.6}", k, v));
        }
        let tibble_body = if custom_param_list.is_empty() {
            "  x_star = 0.0".to_string()
        } else {
            custom_param_list.join(",\n")
        };

        format!(
            r##"suppressPackageStartupMessages({{
  library(tidyverse)
}})

sim_data <- read_csv("simulation_data.csv", show_col_types = FALSE)

workspace_params <- tibble(
{tibble_body},
  x_star = {x_star:.6},
  y_star = {y_star:.6},
  lambda_star = {lambda_star:.6},
  indirect_utility = {indirect_u:.6}
)

print(workspace_params)
glimpse(sim_data)

p_main <- ggplot(sim_data, aes(x = x)) +
  geom_line(aes(y = primary_y, color = "Primary f(x)"), linewidth = 1.2) +
  geom_line(aes(y = secondary_y, color = "Secondary g(x)"), linewidth = 1.0, linetype = "dashed") +
  theme_minimal(base_size = 13)

print(p_main)
"##,
            tibble_body = tibble_body,
            x_star = x_star,
            y_star = y_star,
            lambda_star = lambda_star,
            indirect_u = indirect_u,
        )
    }

    pub fn intro_help_text() -> &'static str {
        "\
=== INTRO-HELP: EconCalc Studio Quickstart Guide ===
1. LAYOUT & PANELS:
   - Top Bar: Toggle Left Sidebar, Right Dock, Bottom Console, Maximize Graph, or Dark/Light Theme.
   - Left Sidebar:
     • Formulas & Sliders: Add/edit functions f(x), customize line style/color/role, and set Desmos slider min/max/step limits.
     • R Objects: Inspect scalars & vectors created in the Console/Script.
     • Models Library: Load ready-made economic & mathematical templates.
     • Project & Files: Browse project directory, open .R/.qmd/.Rmd/.md/.csv files.
2. GRAPH VIEWPORT (2D & 3D):
   - Click `🏠 Home` to restore your configured Home X & Y bounds.
   - Single-Axis Zoom: Select `X-Axis Only` or `Y-Axis Only` in the graph toolbar (or hold Shift for X-zoom, Ctrl for Y-zoom while scrolling).
   - 3D Surface: Switch to `3D Surface` to orbit and inspect any Z = f(x, y) function.
3. DATASETS & LITERATE NOTES (Right Dock):
   - Datasets Tab: Create/edit tables, Import/Export CSVs, or overlay scatter/OLS on the graph.
   - Notes Tab: Write Markdown/Quarto/RMarkdown notes, import/save .qmd/.Rmd/.md files, or export to Positron."
    }

    pub fn general_help_text() -> &'static str {
        "\
=== GENERAL-HELP: Quick Math, R Commands, Desmos & LaTeX Syntax ===
1. QUICK MATH (No formula required — type directly in Console):
   • `2 + 2 * 5`            -> 12
   • `2x`, `2(x + 1)`       Implicit multiplication: 2*x and 2*(x + 1)
   • `(x + 1)(x - 1)`       Adjacent parenthesized terms are multiplied
   • `sqrt(144) + e^2`      -> 19.389056
   • `sin(pi / 4)`          -> 0.707107
   • `log(100, 10)`         -> 2

2. R-LANGUAGE COMMANDS & OBJECTS:
   • `x <- c(1, 2, 3, 4, 5)`   Create numeric vector
   • `w <- rep(2.5, 4)`        Repeat value: [2.5, 2.5, 2.5, 2.5]
   • `s <- seq(0, 10, 2)`      Sequence: [0, 2, 4, 6, 8, 10]
   • `r <- 1:10`               Integer sequence 1 to 10
   • `y <- 2 * x + 3`          Vectorized arithmetic
   • `mean(x)`, `sd(x)`, `var(x)`, `sum(x)`, `min(x)`, `max(x)`, `median(x)`, `cumsum(x)`, `summary(x)`
   • `plot(x, y)`              Plot R vectors directly onto the 2D Graph
   • `data.frame(x, y)`        Create a Dataset table from R vectors
   • `ls()`, `rm(x)`, `clear()` Manage R workspace objects

3. DESMOS & LATEX SYMBOLIC FORMULAS:
   • `\\alpha = 0.45`           Creates parameter slider with Greek symbol α
   • `f(x) = 1 / x`            Asymptote-aware rational function
   • `g(x) = e^x`              Memory-safe exponential function
   • `h(x, y) = sin(x)*cos(y)` 3D surface function"
    }

    fn trigger_export_pipeline(&mut self) {
        let (x_star, y_star, lambda_star, indirect_utility) = self
            .solve_structural_equilibrium_opt()
            .unwrap_or((0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64));

        let (custom_expr, secondary_expr) = {
            let mut fn_iter = self.workspace.functions.values();
            let c = fn_iter.next().cloned().unwrap_or_else(|| "0.0".to_string());
            let s = fn_iter.next().cloned().unwrap_or_else(|| "0.0".to_string());
            (c, s)
        };

        let tfp_a = self.workspace.parameters.get("A").copied().unwrap_or(2.0_f64);
        let alpha = self
            .workspace
            .parameters
            .get("alpha")
            .copied()
            .unwrap_or(0.45_f64);
        let beta = self
            .workspace
            .parameters
            .get("beta")
            .copied()
            .unwrap_or(0.55_f64);
        let price_x = self
            .workspace
            .parameters
            .get("price_x")
            .copied()
            .unwrap_or(4.0_f64);
        let price_y = self
            .workspace
            .parameters
            .get("price_y")
            .copied()
            .unwrap_or(5.0_f64);
        let income = self
            .workspace
            .parameters
            .get("I")
            .copied()
            .unwrap_or(100.0_f64);
        let target_output = self
            .workspace
            .parameters
            .get("Y_bar")
            .copied()
            .unwrap_or(18.0_f64);

        let payload = ExportPayload {
            model_kind: self.active_model,
            report_format: self.report_format,
            custom_expr,
            secondary_expr,
            tfp_a,
            alpha,
            beta,
            price_x,
            price_y,
            income,
            target_output,
            x_star,
            y_star,
            lambda_star,
            indirect_utility,
            game_matrix: [
                [(x_star, y_star), (price_x, price_y)],
                [(alpha, beta), (lambda_star, indirect_utility)],
            ],
            game_summary: self.workspace.markdown_notes.clone(),
            latex_derivation: self.generate_latex_derivation(),
            grid_rows: self.build_simulation_grid(self.sim_grid_rows),
        };

        match export_and_launch_positron(&payload) {
            Ok(ExportOutcome {
                export_dir,
                launched_program,
                status_message,
            }) => {
                self.status_banner =
                    format!("Exported to {:?} ({})", export_dir, launched_program);
                self.console_logs.push(ConsoleLogEntry {
                    timestamp_tag: "EXPORT".to_string(),
                    source_line: "export_and_launch_positron".to_string(),
                    message: status_message,
                    is_error: false,
                });
                self.refresh_project_files();
            }
            Err(err) => {
                self.status_banner = format!("Export failed: {}", err);
            }
        }
    }

    // =========================================================================
    // 2-TIER TOP HEADER (MENU BAR + QUICK ACTION TOOLBAR)
    // =========================================================================

    fn render_graph_settings_popover(&mut self, ui: &mut Ui, pal: ThemePalette) {
        ui.label(
            RichText::new("Graph & Calculus Overlays")
                .strong()
                .color(pal.accent_blue),
        );
        ui.separator();
        ui.checkbox(&mut self.show_plot_grid, "Show Coordinate Grid");
        ui.checkbox(&mut self.show_plot_legend, "Show Series Legend");
        ui.checkbox(&mut self.lock_equal_aspect, "Lock 1:1 Equal Aspect Ratio");
        ui.separator();
        ui.checkbox(&mut self.show_derivative_overlay, "Derivative Overlay f'(x)");
        ui.checkbox(&mut self.show_integral_shade, "Integral Shade ∫f(x) dx");
        ui.checkbox(&mut self.show_equilibrium_overlay, "Intersection / Equilibrium Marker");
        ui.checkbox(&mut self.enable_equilibrium_solver, "Solved Equilibrium Panel");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_probe_marker, "Probe x₀:");
            ui.add(egui::DragValue::new(&mut self.probe_x).speed(0.1_f64));
        });
    }

    fn render_top_header(&mut self, ui: &mut Ui) {
        let pal = self.palette();

        // ---- Tier 1: Menu Bar ----
        egui::menu::bar(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 6.0_f32;
            ui.label(
                RichText::new("☰")
                    .font(FontId::proportional(14.0_f32))
                    .color(pal.text_secondary),
            );
            ui.label(
                RichText::new("EconCalc")
                    .font(FontId::proportional(13.5_f32))
                    .strong()
                    .color(pal.text_primary),
            );
            ui.add_space(4.0_f32);

            ui.menu_button(RichText::new("File ⌄").color(pal.text_secondary), |ui| {
                if ui.button("📄 New / Clear Workspace").clicked() {
                    self.clear_entire_workspace();
                    ui.close_menu();
                }
                if ui.button("📂 Open File from Project").clicked() {
                    self.maximize_graph = false;
                    self.show_left_panel = true;
                    self.left_bottom_collapsed = false;
                    self.left_bottom_tab = LeftBottomTab::Files;
                    self.refresh_project_files();
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("💾 Save R Script (exports/workspace_script.R)").clicked() {
                    self.save_editor_script_default();
                    ui.close_menu();
                }
                if ui.button("📝 Save Notes (.qmd / .Rmd / .md)").clicked() {
                    self.save_notes_default();
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .button(
                        RichText::new("🚀 Export Bundle & Launch Positron").color(pal.accent_green),
                    )
                    .clicked()
                {
                    self.trigger_export_pipeline();
                    ui.close_menu();
                }
            });

            ui.menu_button(RichText::new("Edit ⌄").color(pal.text_secondary), |ui| {
                if ui.button("📋 Copy Canonical LaTeX / TikZ").clicked() {
                    let latex = self.generate_latex_derivation();
                    ui.ctx().copy_text(latex);
                    self.status_banner = "Copied canonical LaTeX/TikZ to clipboard.".to_string();
                    ui.close_menu();
                }
                if ui.button("📋 Copy R / ggplot2 Script").clicked() {
                    let r_src = self.generate_r_script_preview();
                    ui.ctx().copy_text(r_src);
                    self.status_banner = "Copied R/ggplot2 script to clipboard.".to_string();
                    ui.close_menu();
                }
                if ui.button("🧹 Clear Console Logs").clicked() {
                    self.console_logs.clear();
                    self.status_banner = "Console cleared.".to_string();
                    ui.close_menu();
                }
                ui.separator();
                ui.checkbox(&mut self.smart_math_shortcuts, "Smart Math Shortcuts (α≥)");
            });

            ui.menu_button(RichText::new("View ⌄").color(pal.text_secondary), |ui| {
                if ui.checkbox(&mut self.show_left_panel, "Left Panel (Symbolic & Files)").changed()
                {
                    self.maximize_graph = false;
                }
                if ui.checkbox(&mut self.show_right_panel, "Right Panel (Env & Data)").changed() {
                    self.maximize_graph = false;
                }
                if ui.checkbox(&mut self.show_bottom_panel, "Bottom Dock (Console & Script)").changed()
                {
                    self.maximize_graph = false;
                }
                ui.checkbox(&mut self.maximize_graph, "⛶ Maximize Graph Viewport");
                ui.separator();
                if ui
                    .selectable_label(self.viewport_mode == ViewportMode::Plot2D, "📈 2D Graph")
                    .clicked()
                {
                    self.viewport_mode = ViewportMode::Plot2D;
                    ui.close_menu();
                }
                if ui
                    .selectable_label(self.viewport_mode == ViewportMode::Surface3D, "🧊 3D Surface")
                    .clicked()
                {
                    self.viewport_mode = ViewportMode::Surface3D;
                    ui.close_menu();
                }
                ui.separator();
                ui.checkbox(&mut self.floating_datasets_window, "🗗 Pop-Out Datasets");
                ui.checkbox(&mut self.floating_notes_window, "🗗 Pop-Out Notes");
                ui.checkbox(&mut self.floating_help_window, "🗗 Pop-Out Help");
                ui.checkbox(&mut self.floating_env_window, "🗗 Pop-Out Environment");
                ui.checkbox(&mut self.floating_console_window, "🗗 Pop-Out Console");
                ui.separator();
                let theme_txt = if self.dark_mode {
                    "☀ Switch to Light Theme"
                } else {
                    "🌙 Switch to Dark Slate Theme"
                };
                if ui.button(theme_txt).clicked() {
                    self.dark_mode = !self.dark_mode;
                    apply_bento_theme(ui.ctx(), self.dark_mode);
                    ui.close_menu();
                }
            });

            ui.menu_button(RichText::new("Run ⌄").color(pal.text_secondary), |ui| {
                if ui
                    .button(RichText::new("▶ Run Script Editor (Ctrl+Enter)").color(pal.accent_green))
                    .clicked()
                {
                    self.evaluate_editor_script();
                    ui.close_menu();
                }
                if ui.button("📊 Generate Dataset from f(x)").clicked() {
                    self.generate_dataset_from_primary_fn();
                    ui.close_menu();
                }
                ui.checkbox(&mut self.enable_equilibrium_solver, "⚖ Toggle Equilibrium Solver");
            });

            ui.menu_button(RichText::new("Tools ⌄").color(pal.text_secondary), |ui| {
                if ui.button("⚙ Numeric & Calculus Settings").clicked() {
                    self.maximize_graph = false;
                    self.show_bottom_panel = true;
                    self.bottom_tab = BottomDockTab::NumericAndGraphSettings;
                    ui.close_menu();
                }
                ui.checkbox(&mut self.show_derivative_overlay, "Derivative Overlay f'(x)");
                ui.checkbox(&mut self.show_integral_shade, "Integral Shade ∫f(x)");
                ui.separator();
                if ui.button("? Intro Help (intro-help)").clicked() {
                    self.execute_console_command("intro-help");
                    self.floating_help_window = true;
                    ui.close_menu();
                }
                if ui.button("? General Help (general-help)").clicked() {
                    self.execute_console_command("general-help");
                    self.floating_help_window = true;
                    ui.close_menu();
                }
            });

            // Right-aligned layout & theme pills
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 4.0_f32;
                if ui
                    .add(pill_button("📁", false, pal))
                    .on_hover_text("Focus Project Files Explorer")
                    .clicked()
                {
                    self.maximize_graph = false;
                    self.show_left_panel = true;
                    self.left_bottom_collapsed = false;
                    self.left_bottom_tab = LeftBottomTab::Files;
                    self.refresh_project_files();
                }
                if ui
                    .add(pill_button(
                        "⚙",
                        self.show_bottom_panel
                            && self.bottom_tab == BottomDockTab::NumericAndGraphSettings,
                        pal,
                    ))
                    .on_hover_text("Numeric, Calculus & Graph Settings")
                    .clicked()
                {
                    self.maximize_graph = false;
                    self.show_bottom_panel = true;
                    self.bottom_tab = BottomDockTab::NumericAndGraphSettings;
                }
                let theme_icon = if self.dark_mode { "🌙" } else { "☀" };
                if ui
                    .add(pill_button(theme_icon, false, pal))
                    .on_hover_text("Toggle Dark Slate / Light Bento Theme")
                    .clicked()
                {
                    self.dark_mode = !self.dark_mode;
                    apply_bento_theme(ui.ctx(), self.dark_mode);
                }
                if ui
                    .add(pill_button(
                        "◨",
                        self.show_right_panel && !self.maximize_graph,
                        pal,
                    ))
                    .on_hover_text("Toggle Right Column (Environment & Data Viewer)")
                    .clicked()
                {
                    self.maximize_graph = false;
                    self.show_right_panel = !self.show_right_panel;
                }
                if ui
                    .add(pill_button(
                        "◫",
                        self.show_bottom_panel && !self.maximize_graph,
                        pal,
                    ))
                    .on_hover_text("Toggle Bottom Center Card (Console & Script)")
                    .clicked()
                {
                    self.maximize_graph = false;
                    self.show_bottom_panel = !self.show_bottom_panel;
                }
                if ui
                    .add(pill_button(
                        "◧",
                        self.show_left_panel && !self.maximize_graph,
                        pal,
                    ))
                    .on_hover_text("Toggle Left Column (Symbolic & Files)")
                    .clicked()
                {
                    self.maximize_graph = false;
                    self.show_left_panel = !self.show_left_panel;
                }
            });
        });

        ui.add_space(2.0_f32);
        let sep_rect = ui.available_rect_before_wrap();
        ui.painter().line_segment(
            [
                Pos2::new(sep_rect.left() - 8.0_f32, sep_rect.top()),
                Pos2::new(sep_rect.right() + 8.0_f32, sep_rect.top()),
            ],
            Stroke::new(1.0_f32, pal.border.gamma_multiply(0.75_f32)),
        );
        ui.add_space(4.0_f32);

        // ---- Tier 2: Action Toolbar ----
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 5.0_f32;

            if ui
                .add(pill_button("📄", false, pal))
                .on_hover_text("New / Clear Workspace")
                .clicked()
            {
                self.clear_entire_workspace();
            }
            if ui
                .add(pill_button("📂", false, pal))
                .on_hover_text("Refresh & Open Project Files")
                .clicked()
            {
                self.maximize_graph = false;
                self.show_left_panel = true;
                self.left_bottom_collapsed = false;
                self.left_bottom_tab = LeftBottomTab::Files;
                self.refresh_project_files();
            }
            if ui
                .add(pill_button("💾", false, pal))
                .on_hover_text("Export Bundle & Launch Positron")
                .clicked()
            {
                self.trigger_export_pipeline();
            }

            ui.separator();

            if ui
                .add(pill_button("🏠", false, pal))
                .on_hover_text("Reset Graph to Home View")
                .clicked()
            {
                self.reset_graph_to_home();
            }
            if ui
                .add(
                    egui::Button::new(RichText::new("▶").strong().color(pal.accent_green))
                        .fill(pal.pill_bg)
                        .stroke(Stroke::new(1.0_f32, pal.border))
                        .rounding(Rounding::same(6.0_f32)),
                )
                .on_hover_text("Run Script Editor (Ctrl+Enter)")
                .clicked()
            {
                self.evaluate_editor_script();
            }
            if ui
                .add(pill_button("✂", false, pal))
                .on_hover_text("Clear Console Logs")
                .clicked()
            {
                self.console_logs.clear();
                self.status_banner = "Console logs cleared.".to_string();
            }
            if ui
                .add(pill_button("⚙", false, pal))
                .on_hover_text("Open Graph & Numerical Settings")
                .clicked()
            {
                self.maximize_graph = false;
                self.show_bottom_panel = true;
                self.bottom_tab = BottomDockTab::NumericAndGraphSettings;
            }

            ui.separator();

            // Quick Settings popover
            ui.menu_button(
                RichText::new("Settings ⌄")
                    .font(FontId::proportional(12.2_f32))
                    .color(pal.text_primary),
                |ui| {
                    self.render_graph_settings_popover(ui, pal);
                },
            );

            // Quick Mode switcher
            let mode_lbl = match self.viewport_mode {
                ViewportMode::Plot2D => "Mode: 2D Graph ⌄",
                ViewportMode::Surface3D => "Mode: 3D Surface ⌄",
            };
            ui.menu_button(
                RichText::new(mode_lbl)
                    .font(FontId::proportional(12.2_f32))
                    .color(pal.text_primary),
                |ui| {
                    if ui
                        .selectable_label(
                            self.viewport_mode == ViewportMode::Plot2D,
                            "📈 2D Symbolic Graph",
                        )
                        .clicked()
                    {
                        self.viewport_mode = ViewportMode::Plot2D;
                        ui.close_menu();
                    }
                    if ui
                        .selectable_label(
                            self.viewport_mode == ViewportMode::Surface3D,
                            "🧊 3D Surface Z = f(x, y)",
                        )
                        .clicked()
                    {
                        self.viewport_mode = ViewportMode::Surface3D;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("📝 R Script Editor").clicked() {
                        self.maximize_graph = false;
                        self.show_bottom_panel = true;
                        self.bottom_tab = BottomDockTab::ScriptEditor;
                        ui.close_menu();
                    }
                    if ui.button("📓 Quarto (.qmd) Notes").clicked() {
                        self.maximize_graph = false;
                        self.show_right_panel = true;
                        self.right_bottom_collapsed = false;
                        self.right_tab = RightDockTab::NotesAndReports;
                        self.report_format = ReportFormat::QuartoQmd;
                        ui.close_menu();
                    }
                },
            );

            // Quick Models Library dropdown
            ui.menu_button(
                RichText::new("Models Library ⌄")
                    .font(FontId::proportional(12.2_f32))
                    .color(pal.accent_blue),
                |ui| {
                    for snip in SnippetTemplate::all() {
                        let selected = self.selected_snippet == snip;
                        if ui
                            .selectable_label(selected, snip.label())
                            .on_hover_text(snip.description())
                            .clicked()
                        {
                            self.load_snippet_template(snip);
                            ui.close_menu();
                        }
                    }
                },
            );

            // Right-aligned live status banner
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(&self.status_banner)
                        .font(FontId::proportional(11.5_f32))
                        .color(pal.text_secondary),
                );
            });
        });
    }

    // =========================================================================
    // LEFT COLUMN BENTO STACK: CARD 1 (SYMBOLIC) + CARD 2 (FILES / MODELS / FORMAT)
    // =========================================================================

    fn render_left_column(&mut self, ui: &mut Ui) {
        let pal = self.palette();
        let avail = ui.available_size();
        let gap = 6.0_f32;
        let header_h = 34.0_f32;

        let (top_h, bot_h) = match (self.symbolic_card_collapsed, self.left_bottom_collapsed) {
            (true, true) => (header_h, header_h),
            (true, false) => (header_h, (avail.y - header_h - gap).max(110.0_f32)),
            (false, true) => ((avail.y - header_h - gap).max(160.0_f32), header_h),
            (false, false) => {
                let h1 = ((avail.y - gap) * 0.64_f32).clamp(180.0_f32, (avail.y - 130.0_f32).max(180.0_f32));
                let h2 = (avail.y - h1 - gap).max(100.0_f32);
                (h1, h2)
            }
        };

        render_bento_card_frame(ui, pal, Some(Vec2::new(avail.x, top_h)), |ui| {
            self.render_symbolic_card(ui, pal);
        });

        ui.add_space(gap);

        render_bento_card_frame(ui, pal, Some(Vec2::new(avail.x, bot_h)), |ui| {
            self.render_left_bottom_card(ui, pal);
        });
    }

    fn render_symbolic_card(&mut self, ui: &mut Ui, pal: ThemePalette) {
        let mut clear_clicked = false;
        let mut collapse_sidebar = false;
        let mut toggle_min = false;

        render_card_title_header(
            ui,
            pal,
            "Symbolic",
            self.symbolic_card_collapsed,
            |ui| {
                if ui
                    .add(header_icon_button("<", pal.text_secondary))
                    .on_hover_text("Hide Left Sidebar")
                    .clicked()
                {
                    collapse_sidebar = true;
                }
                if ui
                    .add(header_icon_button(
                        if self.symbolic_card_collapsed { "+" } else { "—" },
                        pal.text_secondary,
                    ))
                    .on_hover_text("Collapse Symbolic Card")
                    .clicked()
                {
                    toggle_min = true;
                }
                if ui
                    .add(header_icon_button("Clear", pal.accent_red))
                    .on_hover_text("Clear all functions, parameters, R objects, and overlays")
                    .clicked()
                {
                    clear_clicked = true;
                }
                ui.checkbox(&mut self.smart_math_shortcuts, "α≥")
                    .on_hover_text(
                        "Auto-convert `alpha ` -> `α `, `>=` -> `≥`, `<=` -> `≤`, `->` -> `→`",
                    );
            },
        );

        if clear_clicked {
            self.clear_entire_workspace();
        }
        if collapse_sidebar {
            self.show_left_panel = false;
        }
        if toggle_min {
            self.symbolic_card_collapsed = !self.symbolic_card_collapsed;
        }
        if self.symbolic_card_collapsed {
            return;
        }

        egui::Frame::none()
            .inner_margin(Margin::symmetric(8.0_f32, 6.0_f32))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_source("bento_symbolic_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // ---- Collapsible Section 1: Functions & Parameters ----
                        let fn_arrow = if self.symbolic_functions_open { "⌄" } else { "›" };
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(format!(
                                        "{} Functions & Parameters ({})",
                                        fn_arrow,
                                        self.workspace.functions.len()
                                    ))
                                    .font(FontId::proportional(12.5_f32))
                                    .strong()
                                    .color(pal.text_secondary),
                                )
                                .fill(Color32::TRANSPARENT)
                                .stroke(Stroke::NONE),
                            )
                            .clicked()
                        {
                            self.symbolic_functions_open = !self.symbolic_functions_open;
                        }

                        if self.symbolic_functions_open {
                            if let Some(ref cyc) = self.workspace.compiled_math.cycle_error {
                                egui::Frame::none()
                                    .fill(pal.accent_red.gamma_multiply(0.14_f32))
                                    .stroke(Stroke::new(1.0_f32, pal.accent_red))
                                    .rounding(Rounding::same(6.0_f32))
                                    .inner_margin(Margin::same(5.0_f32))
                                    .show(ui, |ui| {
                                        ui.label(
                                            RichText::new(format!("⚠ {}", cyc))
                                                .font(FontId::proportional(11.2_f32))
                                                .color(pal.accent_red),
                                        );
                                    });
                                ui.add_space(4.0_f32);
                            }

                            if self.workspace.functions.is_empty() {
                                ui.label(
                                    RichText::new("No active functions. Add one below or load a Model.")
                                        .italics()
                                        .color(pal.text_muted),
                                );
                            } else {
                                let fn_keys: Vec<String> =
                                    self.workspace.functions.keys().cloned().collect();
                                let existing_params: std::collections::HashSet<String> =
                                    self.workspace.parameters.keys().cloned().collect();
                                let existing_r_objs: std::collections::HashSet<String> =
                                    self.workspace.r_objects.keys().cloned().collect();
                                let mut remove_key: Option<String> = None;
                                let mut add_sliders: Vec<String> = Vec::new();
                                let mut latex_copied_msg: Option<String> = None;

                                for (idx, sig) in fn_keys.iter().enumerate() {
                                    let mut expr = self
                                        .workspace
                                        .functions
                                        .get(sig)
                                        .cloned()
                                        .unwrap_or_default();
                                    let mut style = self
                                        .workspace
                                        .function_styles
                                        .get(sig)
                                        .cloned()
                                        .unwrap_or_else(|| FunctionStyle::default_for_index(idx));

                                    capsule_frame(pal, style.color.gamma_multiply(0.55_f32))
                                        .show(ui, |ui| {
                                            // 1. Live native "Math-as-Written" preview
                                            let preview = egui::ScrollArea::horizontal()
                                                .id_source(format!("math_preview_scroll_{}", sig))
                                                .show(ui, |ui| {
                                                    render_math_preview_strip(
                                                        ui,
                                                        Some(sig.as_str()),
                                                        &expr,
                                                        &self.workspace.compiled_math.known_fn_names,
                                                        15.0_f32,
                                                        pal.text_primary,
                                                        Some(style.color),
                                                        pal.extreme_bg,
                                                        pal.border,
                                                        pal.accent_red,
                                                    )
                                                })
                                                .inner;

                                            ui.add_space(4.0_f32);

                                            // 2. Edit row + divided `×` action on right
                                            ui.horizontal(|ui| {
                                                ui.checkbox(&mut style.visible, "");
                                                ui.color_edit_button_srgba(&mut style.color);
                                                let edited = ui.add(
                                                    egui::TextEdit::singleline(&mut expr)
                                                        .font(FontId::monospace(11.6_f32))
                                                        .desired_width(
                                                            (ui.available_width() - 96.0_f32)
                                                                .max(68.0_f32),
                                                        ),
                                                );
                                                if edited.changed() {
                                                    if self.smart_math_shortcuts {
                                                        apply_smart_math_shortcuts(&mut expr);
                                                    }
                                                    self.workspace
                                                        .functions
                                                        .insert(sig.clone(), expr.clone());
                                                    self.workspace.sync_math_cache();
                                                }

                                                ui.with_layout(
                                                    egui::Layout::right_to_left(egui::Align::Center),
                                                    |ui| {
                                                        ui.spacing_mut().item_spacing.x = 3.0_f32;
                                                        if ui
                                                            .add(
                                                                egui::Button::new(
                                                                    RichText::new("×")
                                                                        .strong()
                                                                        .color(pal.accent_red),
                                                                )
                                                                .fill(pal.card_bg)
                                                                .stroke(Stroke::new(
                                                                    1.0_f32, pal.border,
                                                                ))
                                                                .rounding(Rounding::same(5.0_f32)),
                                                            )
                                                            .on_hover_text("Delete function")
                                                            .clicked()
                                                        {
                                                            remove_key = Some(sig.clone());
                                                        }
                                                        ui.separator();
                                                        if ui
                                                            .add(header_icon_button(
                                                                "⚙",
                                                                pal.text_secondary,
                                                            ))
                                                            .on_hover_text(
                                                                "Curve style, role & domain",
                                                            )
                                                            .clicked()
                                                        {
                                                            style.settings_open =
                                                                !style.settings_open;
                                                        }
                                                        if ui
                                                            .add(header_icon_button(
                                                                "TeX",
                                                                pal.accent_blue,
                                                            ))
                                                            .on_hover_text(
                                                                "Copy canonical LaTeX from AST",
                                                            )
                                                            .clicked()
                                                        {
                                                            ui.ctx()
                                                                .copy_text(preview.latex.clone());
                                                            latex_copied_msg = Some(format!(
                                                                "Copied LaTeX: {}",
                                                                preview.latex
                                                            ));
                                                        }
                                                    },
                                                );
                                            });

                                            // 3. Live evaluation / type pill or inline syntax error
                                            if let Some(ref err) = preview.error {
                                                let (_, col) = err.span.line_col(&expr);
                                                ui.label(
                                                    RichText::new(format!(
                                                        "⚠ col {}: {}",
                                                        col, err.message
                                                    ))
                                                    .font(FontId::monospace(10.8_f32))
                                                    .color(pal.accent_red),
                                                );
                                            } else if let Ok(plottable) =
                                                self.workspace.compiled_math.classify_entry(sig, &expr)
                                            {
                                                let ext_bridge = REngineMathBridge {
                                                    r_objects: &self.workspace.r_objects,
                                                };
                                                let eval_ctx = self.build_eval_context(&ext_bridge);
                                                let mut scratch = EvalScratch::new();

                                                ui.horizontal_wrapped(|ui| match plottable {
                                                    PlottableEntry::ScalarOnly { compiled } => {
                                                        if let Ok(v) = compiled
                                                            .eval(&eval_ctx, &[], &mut scratch)
                                                        {
                                                            if v.is_finite() {
                                                                ui.label(
                                                                    RichText::new(format!(
                                                                        "= {:.4}",
                                                                        v
                                                                    ))
                                                                    .font(FontId::monospace(11.2_f32))
                                                                    .strong()
                                                                    .color(pal.accent_green),
                                                                );
                                                            }
                                                        }
                                                    }
                                                    PlottableEntry::Explicit1D { compiled } => {
                                                        if let Ok(v) = compiled.eval(
                                                            &eval_ctx,
                                                            &[
                                                                ("x", self.probe_x),
                                                                ("y", 1.0_f64),
                                                                ("L", self.probe_x),
                                                                ("K", 1.0_f64),
                                                                ("Q", self.probe_x),
                                                                ("Y", self.probe_x),
                                                            ],
                                                            &mut scratch,
                                                        ) {
                                                            if v.is_finite() {
                                                                ui.label(
                                                                    RichText::new(format!(
                                                                        "@ x={:.2} → {:.4}",
                                                                        self.probe_x, v
                                                                    ))
                                                                    .font(FontId::monospace(10.5_f32))
                                                                    .color(pal.text_muted),
                                                                );
                                                            }
                                                        }
                                                    }
                                                    PlottableEntry::ImplicitEquation { .. } => {
                                                        ui.label(
                                                            RichText::new("Implicit Contour")
                                                                .font(FontId::monospace(10.5_f32))
                                                                .color(pal.accent_purple),
                                                        );
                                                    }
                                                    PlottableEntry::ExplicitYInequality { .. }
                                                    | PlottableEntry::ImplicitInequality { .. } => {
                                                        ui.label(
                                                            RichText::new("Shaded Inequality")
                                                                .font(FontId::monospace(10.5_f32))
                                                                .color(pal.accent_amber),
                                                        );
                                                    }
                                                    PlottableEntry::Parametric { .. } => {
                                                        ui.label(
                                                            RichText::new("Parametric Curve")
                                                                .font(FontId::monospace(10.5_f32))
                                                                .color(pal.accent_green),
                                                        );
                                                    }
                                                    PlottableEntry::Polar { .. } => {
                                                        ui.label(
                                                            RichText::new("Polar Curve r(θ)")
                                                                .font(FontId::monospace(10.5_f32))
                                                                .color(pal.accent_blue),
                                                        );
                                                    }
                                                });
                                            }

                                            // 4. Automatic unbound parameter discovery pills
                                            if let Some(ref stmt) = preview.parsed_statement {
                                                let mut bound_locals =
                                                    std::collections::HashSet::new();
                                                if let Some((_, params)) =
                                                    math::parse_function_signature(sig)
                                                {
                                                    for p in params {
                                                        bound_locals.insert(p);
                                                    }
                                                }
                                                let mut unbound = Vec::new();
                                                let mut collect_from = |e: &math::SpannedExpr| {
                                                    for sym in math::graph::discover_unbound_symbols(
                                                        e,
                                                        &bound_locals,
                                                        &self.workspace.compiled_math.known_fn_names,
                                                        &existing_params,
                                                        &existing_r_objs,
                                                    ) {
                                                        if !unbound.contains(&sym) {
                                                            unbound.push(sym);
                                                        }
                                                    }
                                                };
                                                match stmt {
                                                    math::MathStatement::Expr(e)
                                                    | math::MathStatement::FunctionDef {
                                                        body: e,
                                                        ..
                                                    }
                                                    | math::MathStatement::ParamAssign {
                                                        value: e,
                                                        ..
                                                    }
                                                    | math::MathStatement::Polar {
                                                        r_expr: e, ..
                                                    } => collect_from(e),
                                                    math::MathStatement::Equation { lhs, rhs }
                                                    | math::MathStatement::Inequality {
                                                        lhs,
                                                        rhs,
                                                        ..
                                                    } => {
                                                        collect_from(lhs);
                                                        collect_from(rhs);
                                                    }
                                                    math::MathStatement::Parametric {
                                                        x_expr,
                                                        y_expr,
                                                        ..
                                                    } => {
                                                        collect_from(x_expr);
                                                        collect_from(y_expr);
                                                    }
                                                }
                                                if !unbound.is_empty() {
                                                    ui.horizontal_wrapped(|ui| {
                                                        ui.label(
                                                            RichText::new("Add slider:")
                                                                .font(FontId::proportional(10.8_f32))
                                                                .color(pal.accent_amber),
                                                        );
                                                        for sym in &unbound {
                                                            let disp =
                                                                math::ast::ident_to_pretty_unicode(
                                                                    sym,
                                                                );
                                                            if ui
                                                                .small_button(format!("+ {}", disp))
                                                                .clicked()
                                                            {
                                                                add_sliders.push(sym.clone());
                                                            }
                                                        }
                                                    });
                                                }
                                            }

                                            // 5. Expandable curve settings
                                            if style.settings_open {
                                                ui.separator();
                                                ui.horizontal_wrapped(|ui| {
                                                    ui.label(
                                                        RichText::new("Style:")
                                                            .color(pal.text_secondary),
                                                    );
                                                    for ls in [
                                                        CurveLineStyle::Solid,
                                                        CurveLineStyle::Dashed,
                                                        CurveLineStyle::Dotted,
                                                    ] {
                                                        ui.selectable_value(
                                                            &mut style.line_style,
                                                            ls,
                                                            ls.label(),
                                                        );
                                                    }
                                                });
                                                ui.horizontal_wrapped(|ui| {
                                                    ui.label(
                                                        RichText::new("Role:")
                                                            .color(pal.text_secondary),
                                                    );
                                                    egui::ComboBox::from_id_source(format!(
                                                        "role_{}",
                                                        sig
                                                    ))
                                                    .selected_text(style.role.label())
                                                    .show_ui(ui, |ui| {
                                                        for r in [
                                                            FunctionPlotRole::StandardCurve,
                                                            FunctionPlotRole::AreaUnderCurve,
                                                            FunctionPlotRole::SampledPoints,
                                                            FunctionPlotRole::Surface3DOnly,
                                                        ] {
                                                            ui.selectable_value(
                                                                &mut style.role,
                                                                r,
                                                                r.label(),
                                                            );
                                                        }
                                                    });
                                                    ui.add(
                                                        egui::Slider::new(
                                                            &mut style.stroke_width,
                                                            1.0_f32..=5.0_f32,
                                                        )
                                                        .text("px"),
                                                    );
                                                });

                                                let mut has_custom_dom =
                                                    style.custom_domain.is_some();
                                                ui.horizontal_wrapped(|ui| {
                                                    if ui
                                                        .checkbox(
                                                            &mut has_custom_dom,
                                                            "Restrict Domain",
                                                        )
                                                        .changed()
                                                    {
                                                        if has_custom_dom {
                                                            style.custom_domain = Some((
                                                                self.domain_min,
                                                                self.domain_max,
                                                            ));
                                                        } else {
                                                            style.custom_domain = None;
                                                        }
                                                    }
                                                    if let Some((ref mut d0, ref mut d1)) =
                                                        style.custom_domain
                                                    {
                                                        ui.add(
                                                            egui::DragValue::new(d0)
                                                                .speed(0.1_f64)
                                                                .prefix("x_min: "),
                                                        );
                                                        ui.add(
                                                            egui::DragValue::new(d1)
                                                                .speed(0.1_f64)
                                                                .prefix("x_max: "),
                                                        );
                                                    }
                                                });
                                            }
                                        });

                                    self.workspace.function_styles.insert(sig.clone(), style);
                                    ui.add_space(4.0_f32);
                                }

                                if let Some(k) = remove_key {
                                    self.workspace.functions.remove(&k);
                                    self.workspace.function_styles.remove(&k);
                                    self.workspace.sync_math_cache();
                                }
                                for sym in add_sliders {
                                    self.workspace.set_parameter(sym, 1.0_f64);
                                }
                                if let Some(msg) = latex_copied_msg {
                                    self.status_banner = msg;
                                }
                            }

                            // ---- New Function Composer + Full-Width "Add Function" Button ----
                            capsule_frame(pal, pal.border).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.new_fn_sig_input)
                                            .font(FontId::monospace(11.8_f32))
                                            .desired_width(52.0_f32)
                                            .hint_text("f(x)"),
                                    );
                                    ui.label(RichText::new("=").color(pal.text_secondary));
                                    let expr_edit = ui.add(
                                        egui::TextEdit::singleline(&mut self.new_fn_expr_input)
                                            .font(FontId::monospace(11.8_f32))
                                            .desired_width((ui.available_width() - 4.0_f32).max(60.0_f32))
                                            .hint_text(r"\frac{1}{x}, y < x^2, (cos(t), sin(t))"),
                                    );
                                    if expr_edit.changed() && self.smart_math_shortcuts {
                                        apply_smart_math_shortcuts(&mut self.new_fn_expr_input);
                                    }
                                });
                                if !self.new_fn_expr_input.trim().is_empty() {
                                    ui.add_space(3.0_f32);
                                    let sig_opt = (!self.new_fn_sig_input.trim().is_empty())
                                        .then_some(self.new_fn_sig_input.as_str());
                                    egui::ScrollArea::horizontal()
                                        .id_source("new_fn_live_preview_scroll")
                                        .show(ui, |ui| {
                                            render_math_preview_strip(
                                                ui,
                                                sig_opt,
                                                &self.new_fn_expr_input,
                                                &self.workspace.compiled_math.known_fn_names,
                                                14.0_f32,
                                                pal.text_primary,
                                                Some(pal.accent_blue),
                                                pal.extreme_bg,
                                                pal.border,
                                                pal.accent_red,
                                            );
                                        });
                                }
                            });
                            ui.add_space(4.0_f32);

                            let add_fn_btn = egui::Button::new(
                                RichText::new("Add Function")
                                    .font(FontId::proportional(12.5_f32))
                                    .strong()
                                    .color(pal.text_primary),
                            )
                            .fill(pal.pill_bg)
                            .stroke(Stroke::new(1.0_f32, pal.border))
                            .rounding(Rounding::same(8.0_f32))
                            .min_size(Vec2::new(ui.available_width(), 28.0_f32));
                            if ui.add(add_fn_btn).clicked() {
                                let sig_raw = self.new_fn_sig_input.trim();
                                let expr = self.new_fn_expr_input.trim().to_string();
                                if !expr.is_empty() {
                                    let final_sig = if sig_raw.is_empty() {
                                        format!("e{}", self.workspace.functions.len() + 1)
                                    } else if sig_raw.contains('(') || sig_raw == "r" {
                                        Self::normalize_fn_signature(sig_raw)
                                    } else {
                                        format!("{}(x)", Self::normalize_ident_name(sig_raw))
                                    };
                                    self.workspace.set_function(final_sig, expr);
                                }
                            }
                        }

                        ui.add_space(8.0_f32);

                        // ---- Collapsible Section 2: Sliders ----
                        let sl_arrow = if self.symbolic_sliders_open { "⌄" } else { "›" };
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(format!(
                                        "{} Sliders ({})",
                                        sl_arrow,
                                        self.workspace.parameters.len()
                                    ))
                                    .font(FontId::proportional(12.5_f32))
                                    .strong()
                                    .color(pal.text_secondary),
                                )
                                .fill(Color32::TRANSPARENT)
                                .stroke(Stroke::NONE),
                            )
                            .clicked()
                        {
                            self.symbolic_sliders_open = !self.symbolic_sliders_open;
                        }

                        if self.symbolic_sliders_open {
                            if self.workspace.parameters.is_empty() {
                                ui.label(
                                    RichText::new("No active sliders.")
                                        .italics()
                                        .color(pal.text_muted),
                                );
                            } else {
                                let param_names: Vec<String> =
                                    self.workspace.parameters.keys().cloned().collect();
                                let mut remove_param: Option<String> = None;

                                for name in param_names {
                                    let mut val = self
                                        .workspace
                                        .parameters
                                        .get(&name)
                                        .copied()
                                        .unwrap_or(1.0_f64);
                                    let mut spec = self
                                        .workspace
                                        .slider_specs
                                        .get(&name)
                                        .cloned()
                                        .unwrap_or_else(|| SliderSpec::for_parameter(&name, val));

                                    capsule_frame(pal, pal.border).show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                RichText::new(Self::format_symbolic_math(&name))
                                                    .font(FontId::monospace(12.8_f32))
                                                    .strong()
                                                    .color(pal.text_primary),
                                            );
                                            ui.label(RichText::new("=").color(pal.text_secondary));
                                            if ui
                                                .add(
                                                    egui::DragValue::new(&mut val)
                                                        .speed(spec.step.max(1e-4_f64)),
                                                )
                                                .changed()
                                            {
                                                self.workspace
                                                    .parameters
                                                    .insert(name.clone(), val);
                                                self.workspace
                                                    .r_objects
                                                    .insert(name.clone(), RObject::scalar(val));
                                            }

                                            ui.with_layout(
                                                egui::Layout::right_to_left(egui::Align::Center),
                                                |ui| {
                                                    ui.spacing_mut().item_spacing.x = 3.0_f32;
                                                    if ui
                                                        .add(header_icon_button(
                                                            "×",
                                                            pal.accent_red,
                                                        ))
                                                        .on_hover_text("Remove parameter")
                                                        .clicked()
                                                    {
                                                        remove_param = Some(name.clone());
                                                    }
                                                    if ui
                                                        .add(header_icon_button(
                                                            &format!(
                                                                "[{:.0}..{:.0}]",
                                                                spec.min, spec.max
                                                            ),
                                                            pal.text_muted,
                                                        ))
                                                        .on_hover_text("Edit slider min, max, step")
                                                        .clicked()
                                                    {
                                                        spec.edit_bounds_open =
                                                            !spec.edit_bounds_open;
                                                    }
                                                },
                                            );
                                        });

                                        let safe_min = spec.min.min(spec.max - 1e-6_f64);
                                        let safe_max = spec.max.max(safe_min + 1e-6_f64);
                                        ui.spacing_mut().slider_width =
                                            (ui.available_width() - 6.0_f32).max(60.0_f32);
                                        let mut slider_control =
                                            egui::Slider::new(&mut val, safe_min..=safe_max)
                                                .show_value(false);
                                        if spec.step > 0.0_f64 {
                                            slider_control = slider_control.step_by(spec.step);
                                        }
                                        if ui.add(slider_control).changed() {
                                            self.workspace.parameters.insert(name.clone(), val);
                                            self.workspace
                                                .r_objects
                                                .insert(name.clone(), RObject::scalar(val));
                                        }

                                        if spec.edit_bounds_open {
                                            ui.horizontal_wrapped(|ui| {
                                                ui.add(
                                                    egui::DragValue::new(&mut spec.min)
                                                        .speed(0.1_f64)
                                                        .prefix("Min: "),
                                                );
                                                ui.add(
                                                    egui::DragValue::new(&mut spec.max)
                                                        .speed(0.1_f64)
                                                        .prefix("Max: "),
                                                );
                                                ui.add(
                                                    egui::DragValue::new(&mut spec.step)
                                                        .speed(0.01_f64)
                                                        .prefix("Step: "),
                                                );
                                                if spec.step < 0.0_f64 {
                                                    spec.step = 0.0_f64;
                                                }
                                            });
                                        }
                                    });

                                    self.workspace.slider_specs.insert(name, spec);
                                    ui.add_space(4.0_f32);
                                }

                                if let Some(p) = remove_param {
                                    self.workspace.parameters.remove(&p);
                                    self.workspace.slider_specs.remove(&p);
                                    self.workspace.r_objects.remove(&p);
                                }
                            }

                            // Compact `+ Add Slider` row
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.new_param_name_input)
                                        .font(FontId::monospace(11.6_f32))
                                        .desired_width(52.0_f32)
                                        .hint_text("alpha"),
                                );
                                ui.label("=");
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.new_param_val_input)
                                        .font(FontId::monospace(11.6_f32))
                                        .desired_width((ui.available_width() - 80.0_f32).max(52.0_f32))
                                        .hint_text("1.0"),
                                );
                                if ui.add(pill_button("+ Slider", false, pal)).clicked() {
                                    let name =
                                        Self::normalize_ident_name(&self.new_param_name_input);
                                    if let Ok(val) =
                                        self.eval_parameter_rhs(&self.new_param_val_input)
                                    {
                                        if !name.is_empty() && val.is_finite() {
                                            self.workspace.set_parameter(name, val);
                                        }
                                    }
                                }
                            });
                        }

                        if self.enable_equilibrium_solver {
                            if let Some((x_star, y_star, lambda_star, indirect_u)) =
                                self.solve_structural_equilibrium_opt()
                            {
                                ui.add_space(6.0_f32);
                                capsule_frame(pal, pal.accent_green).show(ui, |ui| {
                                    ui.label(
                                        RichText::new("Solved Equilibrium / Intersection")
                                            .strong()
                                            .color(pal.accent_green),
                                    );
                                    ui.monospace(format!(
                                        "x* = {:.4}  |  y* = {:.4}",
                                        x_star, y_star
                                    ));
                                    ui.monospace(format!(
                                        "λ* = {:.4}  |  V* = {:.4}",
                                        lambda_star, indirect_u
                                    ));
                                });
                            }
                        }
                    });
            });
    }

    fn render_left_bottom_card(&mut self, ui: &mut Ui, pal: ThemePalette) {
        let header_act = render_card_tab_header(
            ui,
            pal,
            &mut self.left_bottom_tab,
            &[
                (LeftBottomTab::Files, "Files"),
                (LeftBottomTab::Models, "Models"),
                (LeftBottomTab::ReportFormat, "Format"),
            ],
            self.left_bottom_collapsed,
            true,
            true,
            "Collapse Files & Models Card",
            "Toggle Expanded Files Card",
        );

        if header_act.minimize_clicked {
            self.left_bottom_collapsed = !self.left_bottom_collapsed;
        }
        if header_act.popout_clicked {
            self.symbolic_card_collapsed = !self.symbolic_card_collapsed;
            self.left_bottom_collapsed = false;
        }
        if self.left_bottom_collapsed {
            return;
        }

        match self.left_bottom_tab {
            LeftBottomTab::Files => self.render_project_files_tab(ui, pal),
            LeftBottomTab::Models => self.render_models_library_tab(ui, pal),
            LeftBottomTab::ReportFormat => {
                egui::Frame::none()
                    .inner_margin(Margin::same(8.0_f32))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new("Literate Export Format")
                                .strong()
                                .color(pal.accent_blue),
                        );
                        ui.add_space(4.0_f32);
                        for fmt in [
                            ReportFormat::QuartoQmd,
                            ReportFormat::RMarkdownRmd,
                            ReportFormat::MarkdownMd,
                        ] {
                            ui.selectable_value(&mut self.report_format, fmt, fmt.label());
                        }
                        ui.add_space(8.0_f32);
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("🚀 Export Bundle & Open Positron")
                                        .strong()
                                        .color(pal.accent_green),
                                )
                                .fill(pal.pill_bg)
                                .stroke(Stroke::new(1.0_f32, pal.border))
                                .rounding(Rounding::same(6.0_f32)),
                            )
                            .clicked()
                        {
                            self.trigger_export_pipeline();
                        }
                    });
            }
        }
    }

    fn render_environment_body(&mut self, ui: &mut Ui, pal: ThemePalette) {
        egui::Frame::none()
            .inner_margin(Margin::symmetric(8.0_f32, 6.0_f32))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Variables")
                            .font(FontId::proportional(12.2_f32))
                            .strong()
                            .color(pal.text_secondary),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{} obj", self.workspace.r_objects.len()))
                                .font(FontId::monospace(10.8_f32))
                                .color(pal.text_muted),
                        );
                    });
                });
                ui.add_space(3.0_f32);

                egui::ScrollArea::vertical()
                    .id_source("r_objects_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let keys: Vec<String> = self.workspace.r_objects.keys().cloned().collect();
                        if keys.is_empty() {
                            ui.label(
                                RichText::new("Environment is empty.")
                                    .italics()
                                    .color(pal.text_muted),
                            );
                            return;
                        }

                        let mut remove_obj: Option<String> = None;
                        let mut run_console_cmd: Option<String> = None;

                        for k in keys {
                            let Some(obj) = self.workspace.r_objects.get(&k).cloned() else {
                                continue;
                            };

                            // Clean one-line row for scalar numeric parameters (like `A = 50`, `α = 0.45`)
                            if let Some(scalar_val) = obj.as_scalar_f64() {
                                egui::Frame::none()
                                    .fill(pal.sub_card_bg.gamma_multiply(0.75_f32))
                                    .stroke(Stroke::new(1.0_f32, pal.border.gamma_multiply(0.6_f32)))
                                    .rounding(Rounding::same(6.0_f32))
                                    .inner_margin(Margin::symmetric(8.0_f32, 4.0_f32))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                RichText::new(Self::format_symbolic_math(&k))
                                                    .font(FontId::monospace(12.2_f32))
                                                    .strong()
                                                    .color(pal.text_primary),
                                            );
                                            ui.label(
                                                RichText::new("=")
                                                    .font(FontId::monospace(12.0_f32))
                                                    .color(pal.text_muted),
                                            );
                                            let val_fmt = if scalar_val.fract().abs() < 1e-9_f64
                                                && scalar_val.abs() < 1e9_f64
                                            {
                                                format!("{}", scalar_val.round() as i64)
                                            } else {
                                                let s = format!("{:.4}", scalar_val);
                                                s.trim_end_matches('0')
                                                    .trim_end_matches('.')
                                                    .to_string()
                                            };
                                            ui.label(
                                                RichText::new(val_fmt)
                                                    .font(FontId::monospace(12.2_f32))
                                                    .color(pal.accent_blue),
                                            );
                                            ui.with_layout(
                                                egui::Layout::right_to_left(egui::Align::Center),
                                                |ui| {
                                                    if ui
                                                        .add(header_icon_button(
                                                            "×",
                                                            pal.text_muted,
                                                        ))
                                                        .on_hover_text("Remove variable")
                                                        .clicked()
                                                    {
                                                        remove_obj = Some(k.clone());
                                                    }
                                                },
                                            );
                                        });
                                    });
                                ui.add_space(3.0_f32);
                                continue;
                            }

                            // Rich expandable cards for structured R objects (DataFrame, Matrix, List, lm, Vector, Closure)
                            capsule_frame(pal, pal.border).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(&k)
                                            .font(FontId::monospace(12.5_f32))
                                            .strong()
                                            .color(pal.accent_blue),
                                    );
                                    ui.label(
                                        RichText::new(obj.type_summary())
                                            .font(FontId::monospace(10.5_f32))
                                            .color(pal.text_muted),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.spacing_mut().item_spacing.x = 3.0_f32;
                                            if ui
                                                .add(header_icon_button("×", pal.accent_red))
                                                .clicked()
                                            {
                                                remove_obj = Some(k.clone());
                                            }
                                            if matches!(&obj, RObject::Numeric(v) if v.len() >= 2)
                                                && ui
                                                    .add(header_icon_button("📈", pal.accent_green))
                                                    .on_hover_text("Plot vector on 2D Graph")
                                                    .clicked()
                                            {
                                                run_console_cmd = Some(format!("plot({})", k));
                                            }
                                            if obj.is_lm_model() {
                                                if ui
                                                    .add(header_icon_button("📈", pal.accent_green))
                                                    .on_hover_text("Plot fitted values")
                                                    .clicked()
                                                {
                                                    run_console_cmd =
                                                        Some(format!("plot(fitted({}))", k));
                                                }
                                                if ui
                                                    .add(header_icon_button("Σ", pal.accent_purple))
                                                    .on_hover_text("Print summary(fit) to Console")
                                                    .clicked()
                                                {
                                                    run_console_cmd =
                                                        Some(format!("summary({})", k));
                                                }
                                            }
                                        },
                                    );
                                });

                                match &obj {
                                    RObject::DataFrame { columns, row_count } => {
                                        egui::CollapsingHeader::new(
                                            RichText::new(format!(
                                                "{} cols × {} rows",
                                                columns.len(),
                                                row_count
                                            ))
                                            .font(FontId::proportional(11.2_f32))
                                            .color(pal.text_secondary),
                                        )
                                        .id_source(format!("r_df_inspect_{}", k))
                                        .show(ui, |ui| {
                                            let col_names: Vec<String> =
                                                columns.keys().cloned().collect();
                                            if col_names.len() >= 2
                                                && ui
                                                    .small_button(format!(
                                                        "📈 Plot {}${} vs {}${}",
                                                        k, col_names[0], k, col_names[1]
                                                    ))
                                                    .clicked()
                                            {
                                                run_console_cmd = Some(format!(
                                                    "plot({}${}, {}${})",
                                                    k, col_names[0], k, col_names[1]
                                                ));
                                            }
                                            for (col_name, col_val) in columns {
                                                ui.horizontal_wrapped(|ui| {
                                                    ui.monospace(
                                                        RichText::new(format!("${}", col_name))
                                                            .strong()
                                                            .color(pal.accent_green),
                                                    );
                                                    ui.monospace(
                                                        RichText::new(col_val.type_summary())
                                                            .color(pal.text_muted),
                                                    );
                                                    if ui.small_button("📈").clicked() {
                                                        run_console_cmd = Some(format!(
                                                            "plot({}${})",
                                                            k, col_name
                                                        ));
                                                    }
                                                });
                                                ui.monospace(
                                                    RichText::new(format!(
                                                        "  {}",
                                                        col_val.preview_string()
                                                    ))
                                                    .font(FontId::monospace(10.8_f32))
                                                    .color(pal.text_primary),
                                                );
                                            }
                                        });
                                    }
                                    RObject::Matrix { nrow, ncol, .. } => {
                                        egui::CollapsingHeader::new(
                                            RichText::new(format!(
                                                "Matrix Grid [{} × {}]",
                                                nrow, ncol
                                            ))
                                            .font(FontId::proportional(11.2_f32))
                                            .color(pal.text_secondary),
                                        )
                                        .id_source(format!("r_mat_inspect_{}", k))
                                        .show(ui, |ui| {
                                            ui.monospace(
                                                RichText::new(obj.preview_string())
                                                    .font(FontId::monospace(10.8_f32))
                                                    .color(pal.text_primary),
                                            );
                                        });
                                    }
                                    RObject::List(map) => {
                                        let hdr = if obj.is_lm_model() {
                                            format!("lm() Model Fields ({})", map.len())
                                        } else {
                                            format!("List Fields ({})", map.len())
                                        };
                                        egui::CollapsingHeader::new(
                                            RichText::new(hdr)
                                                .font(FontId::proportional(11.2_f32))
                                                .color(pal.text_secondary),
                                        )
                                        .id_source(format!("r_list_inspect_{}", k))
                                        .show(ui, |ui| {
                                            for (field, fval) in map {
                                                ui.horizontal_wrapped(|ui| {
                                                    ui.monospace(
                                                        RichText::new(format!("${}", field))
                                                            .strong()
                                                            .color(pal.accent_purple),
                                                    );
                                                    ui.monospace(
                                                        RichText::new(fval.preview_string())
                                                            .font(FontId::monospace(10.8_f32))
                                                            .color(pal.text_primary),
                                                    );
                                                });
                                            }
                                        });
                                    }
                                    RObject::Closure { .. } => {
                                        ui.monospace(
                                            RichText::new(
                                                obj.closure_signature()
                                                    .unwrap_or_else(|| "function()".to_string()),
                                            )
                                            .font(FontId::monospace(11.2_f32))
                                            .color(pal.accent_amber),
                                        );
                                    }
                                    _ => {
                                        ui.monospace(
                                            RichText::new(obj.preview_string())
                                                .font(FontId::monospace(11.2_f32))
                                                .color(pal.text_primary),
                                        );
                                    }
                                }
                            });
                            ui.add_space(4.0_f32);
                        }

                        if let Some(k) = remove_obj {
                            self.workspace.r_objects.remove(&k);
                            self.workspace.parameters.remove(&k);
                            self.workspace.slider_specs.remove(&k);
                        }
                        if let Some(cmd) = run_console_cmd {
                            self.execute_console_command(&cmd);
                        }
                    });
            });
    }

    fn render_models_library_tab(&mut self, ui: &mut Ui, pal: ThemePalette) {
        egui::Frame::none()
            .inner_margin(Margin::symmetric(8.0_f32, 6.0_f32))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_source("models_library_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for snip in SnippetTemplate::all() {
                            let is_active = self.selected_snippet == snip;
                            capsule_frame(
                                pal,
                                if is_active {
                                    pal.accent_blue
                                } else {
                                    pal.border
                                },
                            )
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(snip.label())
                                            .font(FontId::proportional(12.2_f32))
                                            .strong()
                                            .color(pal.text_primary),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui
                                                .add(
                                                    egui::Button::new(
                                                        RichText::new("Load")
                                                            .font(FontId::proportional(11.2_f32))
                                                            .strong()
                                                            .color(pal.accent_green),
                                                    )
                                                    .fill(pal.pill_bg)
                                                    .stroke(Stroke::new(1.0_f32, pal.border))
                                                    .rounding(Rounding::same(5.0_f32)),
                                                )
                                                .clicked()
                                            {
                                                self.load_snippet_template(snip);
                                            }
                                        },
                                    );
                                });
                                ui.label(
                                    RichText::new(snip.description())
                                        .font(FontId::proportional(11.2_f32))
                                        .color(pal.text_secondary),
                                );
                            });
                            ui.add_space(4.0_f32);
                        }
                    });
            });
    }

    fn render_project_files_tab(&mut self, ui: &mut Ui, pal: ThemePalette) {
        render_card_sub_toolbar(ui, pal, |ui| {
            if ui
                .add(header_icon_button("←", pal.text_secondary))
                .on_hover_text("Parent directory")
                .clicked()
            {
                if let Some(parent) = self.project_dir.parent() {
                    self.project_dir = parent.to_path_buf();
                    self.refresh_project_files();
                }
            }
            egui::ComboBox::from_id_source("bento_files_filter_combo")
                .width(96.0_f32)
                .selected_text(
                    RichText::new(self.file_filter.label())
                        .font(FontId::proportional(11.5_f32))
                        .color(pal.text_primary),
                )
                .show_ui(ui, |ui| {
                    for f in [
                        FileFilterKind::All,
                        FileFilterKind::Scripts,
                        FileFilterKind::Data,
                        FileFilterKind::Notes,
                    ] {
                        ui.selectable_value(&mut self.file_filter, f, f.label());
                    }
                });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add(header_icon_button("🔄", pal.text_secondary))
                    .on_hover_text("Refresh directory listing")
                    .clicked()
                {
                    self.refresh_project_files();
                }
            });
        });

        egui::Frame::none()
            .inner_margin(Margin::symmetric(8.0_f32, 4.0_f32))
            .show(ui, |ui| {
                ui.monospace(
                    RichText::new(format!("{}", self.project_dir.display()))
                        .font(FontId::monospace(10.5_f32))
                        .color(pal.text_muted),
                );
                ui.add_space(3.0_f32);

                egui::ScrollArea::vertical()
                    .id_source("project_files_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let entries = self.project_files.clone();
                        for entry in entries {
                            if !self.file_filter.matches(&entry.name, entry.is_dir) {
                                continue;
                            }
                            ui.horizontal(|ui| {
                                let icon = if entry.is_dir { "📁" } else { "📄" };
                                if ui
                                    .selectable_label(
                                        false,
                                        RichText::new(format!("{} {}", icon, entry.name))
                                            .font(FontId::proportional(12.0_f32))
                                            .color(if entry.is_dir {
                                                pal.accent_blue
                                            } else {
                                                pal.text_primary
                                            }),
                                    )
                                    .clicked()
                                {
                                    if entry.is_dir {
                                        self.project_dir = entry.path;
                                        self.refresh_project_files();
                                    } else {
                                        self.open_project_file(&entry.path);
                                    }
                                }
                                if !entry.is_dir {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                RichText::new(format!("{} B", entry.size_bytes))
                                                    .font(FontId::monospace(10.2_f32))
                                                    .color(pal.text_muted),
                                            );
                                        },
                                    );
                                }
                            });
                        }
                    });
            });
    }

    fn open_project_file(&mut self, path: &std::path::Path) {
        let Ok(content) = load_text_file(path) else {
            self.status_banner = format!("Could not read {:?}", path);
            return;
        };
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported")
            .to_string();

        match ext.as_str() {
            "csv" | "tsv" => {
                let mut ds =
                    DataTable::from_csv_str(file_stem, content, Color32::from_rgb(14, 165, 233));
                ds.show_on_plot = true;
                self.workspace.datasets.push(ds);
                self.workspace.active_dataset_idx = self.workspace.datasets.len() - 1;
                self.right_tab = RightDockTab::Datasets;
                self.show_right_panel = true;
                self.status_banner = format!("Imported CSV dataset from {:?}", path);
            }
            "qmd" | "rmd" | "md" | "txt" => {
                self.workspace.markdown_notes = content;
                self.file_io_path_input = path.to_string_lossy().to_string();
                self.right_tab = RightDockTab::NotesAndReports;
                self.show_right_panel = true;
                self.status_banner = format!("Opened document {:?} in Notes editor", path);
            }
            "r" | "econ" => {
                self.editor_script = content;
                self.bottom_tab = BottomDockTab::ScriptEditor;
                self.show_bottom_panel = true;
                self.status_banner = format!("Loaded script {:?} into Script Editor", path);
            }
            _ => {
                self.workspace.markdown_notes = content;
                self.right_tab = RightDockTab::NotesAndReports;
                self.status_banner = format!("Opened {:?} in Notes", path);
            }
        }
    }

    // =========================================================================
    // CARD 3 — CENTER TOP: GRAPH VIEWPORT (2D DESMOS GLOW PLOT & 3D SURFACE)
    // =========================================================================

    fn render_center_viewport(&mut self, ui: &mut Ui) {
        let pal = self.palette();
        render_bento_card_frame(ui, pal, None, |ui| {
            // Card 3 Internal Top Bar
            let header_resp = egui::Frame::none()
                .fill(pal.card_header_bg)
                .rounding(Rounding {
                    nw: 8.0_f32,
                    ne: 8.0_f32,
                    sw: 0.0_f32,
                    se: 0.0_f32,
                })
                .inner_margin(Margin::symmetric(8.0_f32, 5.0_f32))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 5.0_f32;

                        // Left: Mode selector pill + `+` quick action
                        let mode_pill = match self.viewport_mode {
                            ViewportMode::Plot2D => "📈 2D Graph ⌄",
                            ViewportMode::Surface3D => "🧊 3D Surface ⌄",
                        };
                        ui.menu_button(
                            RichText::new(mode_pill)
                                .font(FontId::proportional(12.2_f32))
                                .strong()
                                .color(pal.accent_red),
                            |ui| {
                                if ui
                                    .selectable_label(
                                        self.viewport_mode == ViewportMode::Plot2D,
                                        "📈 2D Symbolic Graph",
                                    )
                                    .clicked()
                                {
                                    self.viewport_mode = ViewportMode::Plot2D;
                                    ui.close_menu();
                                }
                                if ui
                                    .selectable_label(
                                        self.viewport_mode == ViewportMode::Surface3D,
                                        "🧊 3D Isometric Surface",
                                    )
                                    .clicked()
                                {
                                    self.viewport_mode = ViewportMode::Surface3D;
                                    ui.close_menu();
                                }
                            },
                        );

                        if ui
                            .add(pill_button("+", false, pal))
                            .on_hover_text("Zoom in (or click Symbolic to add a new expression)")
                            .clicked()
                        {
                            if self.viewport_mode == ViewportMode::Plot2D {
                                self.pending_zoom_factor = Some([1.25_f32, 1.25_f32]);
                            } else {
                                self.surface_zoom = (self.surface_zoom * 1.15_f32).clamp(0.3, 2.5);
                            }
                        }

                        if ui
                            .add(pill_button("🏠", false, pal))
                            .on_hover_text(format!(
                                "Restore Home View: X[{:.1}, {:.1}], Y[{:.1}, {:.1}]",
                                self.home_x_min, self.home_x_max, self.home_y_min, self.home_y_max
                            ))
                            .clicked()
                        {
                            self.reset_graph_to_home();
                        }

                        if self.viewport_mode == ViewportMode::Plot2D {
                            for mode in [
                                AxisZoomMode::BothXY,
                                AxisZoomMode::XOnly,
                                AxisZoomMode::YOnly,
                            ] {
                                let selected = self.axis_zoom_mode == mode;
                                if ui
                                    .add(pill_button(mode.label(), selected, pal))
                                    .on_hover_text("Scroll-zoom axis lock")
                                    .clicked()
                                {
                                    self.axis_zoom_mode = mode;
                                }
                            }

                            if ui
                                .add(header_icon_button("X+", pal.text_secondary))
                                .on_hover_text("Zoom in X-axis")
                                .clicked()
                            {
                                self.pending_zoom_factor = Some([1.25_f32, 1.0_f32]);
                            }
                            if ui
                                .add(header_icon_button("X-", pal.text_secondary))
                                .on_hover_text("Zoom out X-axis")
                                .clicked()
                            {
                                self.pending_zoom_factor = Some([0.8_f32, 1.0_f32]);
                            }
                            if ui
                                .add(header_icon_button("Y+", pal.text_secondary))
                                .on_hover_text("Zoom in Y-axis")
                                .clicked()
                            {
                                self.pending_zoom_factor = Some([1.0_f32, 1.25_f32]);
                            }
                            if ui
                                .add(header_icon_button("Y-", pal.text_secondary))
                                .on_hover_text("Zoom out Y-axis")
                                .clicked()
                            {
                                self.pending_zoom_factor = Some([1.0_f32, 0.8_f32]);
                            }

                            let mut bounds_edited = false;
                            let drag_box = Vec2::new(46.0_f32, 19.0_f32);
                            ui.label(
                                RichText::new("X:")
                                    .font(FontId::monospace(11.2_f32))
                                    .strong()
                                    .color(pal.accent_blue),
                            );
                            bounds_edited |= ui
                                .add_sized(
                                    drag_box,
                                    egui::DragValue::new(&mut self.domain_min).speed(0.2_f64),
                                )
                                .changed();
                            ui.label(RichText::new("..").color(pal.text_muted));
                            bounds_edited |= ui
                                .add_sized(
                                    drag_box,
                                    egui::DragValue::new(&mut self.domain_max).speed(0.2_f64),
                                )
                                .changed();

                            ui.label(
                                RichText::new("Y:")
                                    .font(FontId::monospace(11.2_f32))
                                    .strong()
                                    .color(pal.accent_purple),
                            );
                            bounds_edited |= ui
                                .add_sized(
                                    drag_box,
                                    egui::DragValue::new(&mut self.domain_y_min).speed(0.2_f64),
                                )
                                .changed();
                            ui.label(RichText::new("..").color(pal.text_muted));
                            bounds_edited |= ui
                                .add_sized(
                                    drag_box,
                                    egui::DragValue::new(&mut self.domain_y_max).speed(0.2_f64),
                                )
                                .changed();

                            if bounds_edited {
                                if self.domain_max <= self.domain_min + 1e-3_f64 {
                                    self.domain_max = self.domain_min + 1.0_f64;
                                }
                                if self.domain_y_max <= self.domain_y_min + 1e-3_f64 {
                                    self.domain_y_max = self.domain_y_min + 1.0_f64;
                                }
                                self.needs_plot_bounds_sync = true;
                            }
                        }

                        // Right: `⋮` quick overlays popover + `⛶` Maximize Graph
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.spacing_mut().item_spacing.x = 4.0_f32;
                            if ui
                                .add(pill_button("⛶", self.maximize_graph, pal))
                                .on_hover_text("Maximize / Restore Graph Viewport")
                                .clicked()
                            {
                                self.maximize_graph = !self.maximize_graph;
                            }
                            ui.menu_button(
                                RichText::new("⋮")
                                    .font(FontId::proportional(14.0_f32))
                                    .strong()
                                    .color(pal.text_secondary),
                                |ui| {
                                    self.render_graph_settings_popover(ui, pal);
                                },
                            );
                        });
                    });
                })
                .response;

            let r = header_resp.rect;
            ui.painter().line_segment(
                [Pos2::new(r.left(), r.bottom()), Pos2::new(r.right(), r.bottom())],
                Stroke::new(1.0_f32, pal.border),
            );
            ui.add_space(-ui.spacing().item_spacing.y);

            egui::Frame::none()
                .fill(pal.plot_bg)
                .rounding(Rounding {
                    nw: 0.0_f32,
                    ne: 0.0_f32,
                    sw: 8.0_f32,
                    se: 8.0_f32,
                })
                .inner_margin(Margin::same(4.0_f32))
                .show(ui, |ui| match self.viewport_mode {
                    ViewportMode::Plot2D => self.render_2d_desmos_plot(ui, pal),
                    ViewportMode::Surface3D => self.render_3d_isometric_surface(ui, pal),
                });
        });
    }

    fn render_2d_desmos_plot(&mut self, ui: &mut Ui, pal: ThemePalette) {
        let modifiers = ui.input(|i| i.modifiers);
        let allow_zoom_xy = if modifiers.shift && !modifiers.ctrl {
            [true, false]
        } else if modifiers.ctrl && !modifiers.shift {
            [false, true]
        } else {
            match self.axis_zoom_mode {
                AxisZoomMode::BothXY => [true, true],
                AxisZoomMode::XOnly => [true, false],
                AxisZoomMode::YOnly => [false, true],
            }
        };

        let mut plot = Plot::new("desmos_2d_symbolic_plot")
            .allow_zoom(allow_zoom_xy)
            .allow_drag([true, true])
            .allow_scroll(
                !modifiers.shift
                    && !modifiers.ctrl
                    && self.axis_zoom_mode == AxisZoomMode::BothXY,
            )
            .allow_boxed_zoom(true)
            .show_grid(self.show_plot_grid);

        if self.show_plot_legend {
            plot = plot.legend(Legend::default());
        }
        if self.lock_equal_aspect {
            plot = plot.data_aspect(1.0_f32);
        }

        let needs_sync = self.needs_plot_bounds_sync;
        self.needs_plot_bounds_sync = false;
        let zoom_factor = self.pending_zoom_factor.take();

        ui.scope(|ui| {
            ui.visuals_mut().extreme_bg_color = pal.plot_bg;
            ui.visuals_mut().widgets.noninteractive.bg_stroke =
                Stroke::new(1.0_f32, pal.border.gamma_multiply(0.55_f32));

            plot.show(ui, |plot_ui| {
                if needs_sync {
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                        [self.domain_min, self.domain_y_min],
                        [self.domain_max, self.domain_y_max],
                    ));
                } else {
                    let b = plot_ui.plot_bounds();
                    let [bx0, by0] = b.min();
                    let [bx1, by1] = b.max();

                    let scroll_delta = plot_ui.ctx().input(|i| i.smooth_scroll_delta.y);
                    let mut custom_scale: Option<[f32; 2]> = zoom_factor;
                    if scroll_delta.abs() > 0.1_f32 && (!allow_zoom_xy[0] || !allow_zoom_xy[1]) {
                        let factor =
                            (1.0_f32 + scroll_delta * 0.002_f32).clamp(0.7_f32, 1.4_f32);
                        custom_scale = Some([
                            if allow_zoom_xy[0] { factor } else { 1.0_f32 },
                            if allow_zoom_xy[1] { factor } else { 1.0_f32 },
                        ]);
                    }

                    if let Some([zx, zy]) = custom_scale {
                        let cx = 0.5_f64 * (bx0 + bx1);
                        let cy = 0.5_f64 * (by0 + by1);
                        let half_w = 0.5_f64 * (bx1 - bx0) / (zx as f64);
                        let half_h = 0.5_f64 * (by1 - by0) / (zy as f64);
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [cx - half_w, cy - half_h],
                            [cx + half_w, cy + half_h],
                        ));
                    }

                    let updated = plot_ui.plot_bounds();
                    if updated.min()[0].is_finite() && updated.max()[0].is_finite() {
                        self.domain_min = updated.min()[0];
                        self.domain_max = updated.max()[0];
                        self.domain_y_min = updated.min()[1];
                        self.domain_y_max = updated.max()[1];
                    }
                }

                // Highlight principal x = 0 and y = 0 axes for crisp quadrant reading
                draw_principal_axes(plot_ui, pal);

                let vis_bounds = plot_ui.plot_bounds();
                let (vis_min_x, vis_min_y) = (vis_bounds.min()[0], vis_bounds.min()[1]);
                let (vis_max_x, vis_max_y) = (vis_bounds.max()[0], vis_bounds.max()[1]);

                let global_sample_min =
                    if vis_min_x.is_finite() && (vis_max_x - vis_min_x) > 1e-4_f64 {
                        vis_min_x
                    } else {
                        self.domain_min
                    };
                let global_sample_max =
                    if vis_max_x.is_finite() && (vis_max_x - vis_min_x) > 1e-4_f64 {
                        vis_max_x
                    } else {
                        self.domain_max
                    };

                let ext_bridge = REngineMathBridge {
                    r_objects: &self.workspace.r_objects,
                };
                let eval_ctx = self.build_eval_context(&ext_bridge);

                for (idx, (sig, expr)) in self.workspace.functions.iter().enumerate() {
                    let style = self
                        .workspace
                        .function_styles
                        .get(sig)
                        .cloned()
                        .unwrap_or_else(|| FunctionStyle::default_for_index(idx));

                    if !style.visible || style.role == FunctionPlotRole::Surface3DOnly {
                        continue;
                    }

                    let series_label = format!("{} = {}", Self::format_symbolic_math(sig), expr);

                    // Live R scatter/line layer when referencing R vectors, `df$col`, `(df$x, df$y)`, or `predict(fit)`
                    if let Some(r_pts) =
                        crate::r_engine::try_extract_r_plot_series(expr, &self.workspace.r_objects)
                    {
                        draw_glowing_plot_line(
                            plot_ui,
                            r_pts.clone(),
                            &series_label,
                            style.color,
                            style.stroke_width,
                            style.line_style.to_egui(),
                        );
                        plot_ui.points(
                            Points::new(PlotPoints::from(r_pts))
                                .radius(style.stroke_width + 2.0_f32)
                                .color(style.color),
                        );
                        continue;
                    }

                    let (s_min, s_max) = if let Some((cmin, cmax)) = style.custom_domain {
                        (cmin.max(global_sample_min), cmax.min(global_sample_max))
                    } else {
                        (global_sample_min, global_sample_max)
                    };

                    let Ok(plottable) = self.workspace.compiled_math.classify_entry(sig, expr)
                    else {
                        continue;
                    };

                    match plottable {
                        PlottableEntry::Explicit1D { compiled } => {
                            if s_max <= s_min {
                                continue;
                            }
                            let branches = math::sample_1d_branches(
                                &compiled,
                                &eval_ctx,
                                s_min,
                                s_max,
                                vis_min_y,
                                vis_max_y,
                                self.curve_samples,
                                self.asymptote_jump_factor,
                            );
                            match style.role {
                                FunctionPlotRole::StandardCurve => {
                                    for branch in branches {
                                        draw_glowing_plot_line(
                                            plot_ui,
                                            branch,
                                            &series_label,
                                            style.color,
                                            style.stroke_width,
                                            style.line_style.to_egui(),
                                        );
                                    }
                                }
                                FunctionPlotRole::AreaUnderCurve => {
                                    for branch in branches {
                                        if branch.len() >= 2 {
                                            let mut poly = Vec::with_capacity(branch.len() + 2);
                                            poly.push([branch[0][0], 0.0_f64]);
                                            poly.extend_from_slice(&branch);
                                            poly.push([branch[branch.len() - 1][0], 0.0_f64]);
                                            plot_ui.polygon(
                                                Polygon::new(PlotPoints::from(poly))
                                                    .name(&series_label)
                                                    .fill_color(
                                                        style.color.gamma_multiply(0.22_f32),
                                                    )
                                                    .stroke(Stroke::new(
                                                        style.stroke_width,
                                                        style.color,
                                                    )),
                                            );
                                        }
                                    }
                                }
                                FunctionPlotRole::SampledPoints => {
                                    for branch in branches {
                                        plot_ui.points(
                                            Points::new(PlotPoints::from(branch))
                                                .name(&series_label)
                                                .radius(style.stroke_width + 1.5_f32)
                                                .color(style.color),
                                        );
                                    }
                                }
                                FunctionPlotRole::Surface3DOnly => {}
                            }
                        }
                        PlottableEntry::ImplicitEquation { diff_compiled } => {
                            let contours = math::sample_implicit_contours(
                                &diff_compiled,
                                &eval_ctx,
                                global_sample_min,
                                global_sample_max,
                                vis_min_y,
                                vis_max_y,
                                72,
                                72,
                            );
                            for chain in contours {
                                draw_glowing_plot_line(
                                    plot_ui,
                                    chain,
                                    &series_label,
                                    style.color,
                                    style.stroke_width,
                                    style.line_style.to_egui(),
                                );
                            }
                        }
                        PlottableEntry::ExplicitYInequality { op, rhs_compiled } => {
                            if s_max <= s_min {
                                continue;
                            }
                            let region = math::sample_explicit_y_inequality(
                                &rhs_compiled,
                                op,
                                &eval_ctx,
                                s_min,
                                s_max,
                                vis_min_y,
                                vis_max_y,
                                self.curve_samples,
                                self.asymptote_jump_factor,
                            );
                            for poly in region.shaded_polygons {
                                plot_ui.polygon(
                                    Polygon::new(PlotPoints::from(poly))
                                        .name(&series_label)
                                        .fill_color(style.color.gamma_multiply(0.18_f32))
                                        .stroke(Stroke::NONE),
                                );
                            }
                            let bound_style = if region.strict {
                                LineStyle::Dashed { length: 7.0_f32 }
                            } else {
                                style.line_style.to_egui()
                            };
                            for branch in region.boundary_curves {
                                draw_glowing_plot_line(
                                    plot_ui,
                                    branch,
                                    &series_label,
                                    style.color,
                                    style.stroke_width,
                                    bound_style,
                                );
                            }
                        }
                        PlottableEntry::ImplicitInequality { op, diff_compiled } => {
                            let region = math::sample_implicit_2d_inequality(
                                &diff_compiled,
                                op,
                                &eval_ctx,
                                global_sample_min,
                                global_sample_max,
                                vis_min_y,
                                vis_max_y,
                                64,
                                64,
                            );
                            for poly in region.shaded_polygons {
                                plot_ui.polygon(
                                    Polygon::new(PlotPoints::from(poly))
                                        .fill_color(style.color.gamma_multiply(0.18_f32))
                                        .stroke(Stroke::NONE),
                                );
                            }
                            let bound_style = if region.strict {
                                LineStyle::Dashed { length: 7.0_f32 }
                            } else {
                                style.line_style.to_egui()
                            };
                            for chain in region.boundary_curves {
                                draw_glowing_plot_line(
                                    plot_ui,
                                    chain,
                                    &series_label,
                                    style.color,
                                    style.stroke_width,
                                    bound_style,
                                );
                            }
                        }
                        PlottableEntry::Parametric {
                            param,
                            x_compiled,
                            y_compiled,
                        } => {
                            let (t0, t1) = style
                                .custom_domain
                                .unwrap_or((0.0_f64, std::f64::consts::TAU));
                            let branches = math::sample_parametric_branches(
                                &x_compiled,
                                &y_compiled,
                                &param,
                                &eval_ctx,
                                t0,
                                t1,
                                self.curve_samples,
                            );
                            for branch in branches {
                                draw_glowing_plot_line(
                                    plot_ui,
                                    branch,
                                    &series_label,
                                    style.color,
                                    style.stroke_width,
                                    style.line_style.to_egui(),
                                );
                            }
                        }
                        PlottableEntry::Polar { param, r_compiled } => {
                            let (t0, t1) = style
                                .custom_domain
                                .unwrap_or((0.0_f64, std::f64::consts::TAU));
                            let branches = math::sample_polar_branches(
                                &r_compiled,
                                &param,
                                &eval_ctx,
                                t0,
                                t1,
                                self.curve_samples,
                            );
                            for branch in branches {
                                draw_glowing_plot_line(
                                    plot_ui,
                                    branch,
                                    &series_label,
                                    style.color,
                                    style.stroke_width,
                                    style.line_style.to_egui(),
                                );
                            }
                        }
                        PlottableEntry::ScalarOnly { .. } => {}
                    }
                }

                if self.show_derivative_overlay {
                    if let Some((sig, expr)) = self.workspace.functions.iter().next() {
                        let samples = (self.curve_samples / 2).clamp(64, 600);
                        let span = (global_sample_max - global_sample_min).max(1e-4_f64);
                        let deriv_pts: PlotPoints = (0..=samples)
                            .filter_map(|i| {
                                let x = global_sample_min + span * (i as f64) / (samples as f64);
                                let dy = self.numerical_derivative(expr, x).ok()?;
                                (dy.is_finite()
                                    && dy >= vis_min_y * 3.0_f64
                                    && dy <= vis_max_y * 3.0_f64)
                                    .then_some([x, dy])
                            })
                            .collect();

                        plot_ui.line(
                            Line::new(deriv_pts)
                                .name(format!("d/dx {}", sig))
                                .width(1.8_f32)
                                .style(LineStyle::Dotted { spacing: 5.0_f32 })
                                .color(pal.accent_purple),
                        );
                    }
                }

                if self.show_integral_shade {
                    if let Some((sig, expr)) = self.workspace.functions.iter().next() {
                        let branches = self.sample_function_branches(
                            expr,
                            self.domain_min,
                            self.domain_max,
                            vis_min_y,
                            vis_max_y,
                            160,
                        );
                        for branch in branches {
                            if branch.len() >= 2 {
                                let mut poly = Vec::with_capacity(branch.len() + 2);
                                poly.push([branch[0][0], 0.0_f64]);
                                poly.extend_from_slice(&branch);
                                poly.push([branch[branch.len() - 1][0], 0.0_f64]);
                                plot_ui.polygon(
                                    Polygon::new(PlotPoints::from(poly))
                                        .name(format!("∫ {} dx", sig))
                                        .fill_color(pal.accent_blue.gamma_multiply(0.18_f32))
                                        .stroke(Stroke::new(1.0_f32, pal.accent_blue)),
                                );
                            }
                        }
                    }
                }

                if self.show_probe_marker {
                    if let Some((sig, expr)) = self.workspace.functions.iter().next() {
                        if let Ok(y_probe) = self.eval_expression_str(expr, self.probe_x, 10.0_f64)
                        {
                            if y_probe.is_finite() {
                                plot_ui.vline(
                                    VLine::new(self.probe_x)
                                        .name(format!("Probe x₀ = {:.2}", self.probe_x))
                                        .style(LineStyle::Dotted { spacing: 6.0_f32 })
                                        .color(pal.text_muted),
                                );
                                plot_ui.points(
                                    Points::new(vec![[self.probe_x, y_probe]])
                                        .name(format!(
                                            "{}({:.2}) = {:.3}",
                                            sig, self.probe_x, y_probe
                                        ))
                                        .radius(5.5_f32)
                                        .shape(MarkerShape::Diamond)
                                        .color(pal.accent_amber),
                                );
                            }
                        }
                    }
                }

                if self.show_equilibrium_overlay {
                    if let Some((ix, iy)) = self.find_primary_secondary_intersection() {
                        plot_ui.points(
                            Points::new(vec![[ix, iy]])
                                .name(format!("Intersection ({:.3}, {:.3})", ix, iy))
                                .radius(6.5_f32)
                                .shape(MarkerShape::Circle)
                                .color(pal.accent_green),
                        );
                    }
                }

                for ds in &self.workspace.datasets {
                    if ds.show_on_plot && !ds.points.is_empty() {
                        plot_ui.points(
                            Points::new(ds.points.clone())
                                .name(format!("{} (n={})", ds.name, ds.points.len()))
                                .radius(4.8_f32)
                                .shape(MarkerShape::Cross)
                                .color(ds.color),
                        );

                        if ds.show_ols_line {
                            if let Some((b1, b0, r2)) = Self::compute_points_ols(&ds.points) {
                                let ols_pts = PlotPoints::from(vec![
                                    [global_sample_min, b1 * global_sample_min + b0],
                                    [global_sample_max, b1 * global_sample_max + b0],
                                ]);
                                plot_ui.line(
                                    Line::new(ols_pts)
                                        .name(format!(
                                            "{} OLS: y={:.2}x+{:.2} (R²={:.2})",
                                            ds.name, b1, b0, r2
                                        ))
                                        .width(1.6_f32)
                                        .style(LineStyle::Dashed { length: 6.0_f32 })
                                        .color(ds.color),
                                );
                            }
                        }
                    }
                }
            });
        });
    }

    fn render_3d_isometric_surface(&mut self, ui: &mut Ui, pal: ThemePalette) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new("Surface Function:")
                    .strong()
                    .color(pal.text_secondary),
            );
            let fn_keys: Vec<String> = self.workspace.functions.keys().cloned().collect();
            if !fn_keys.is_empty() && !fn_keys.contains(&self.selected_3d_function) {
                self.selected_3d_function = fn_keys[0].clone();
            }
            egui::ComboBox::from_id_source("surface_3d_fn_selector")
                .selected_text(&self.selected_3d_function)
                .show_ui(ui, |ui| {
                    for k in &fn_keys {
                        ui.selectable_value(&mut self.selected_3d_function, k.clone(), k);
                    }
                });

            ui.add(egui::Slider::new(&mut self.yaw_deg, -180.0_f32..=180.0_f32).text("Yaw"));
            ui.add(egui::Slider::new(&mut self.pitch_deg, 5.0_f32..=85.0_f32).text("Pitch"));
            ui.add(egui::Slider::new(&mut self.surface_zoom, 0.3_f32..=2.5_f32).text("Zoom"));
            ui.add(egui::Slider::new(&mut self.surface_resolution, 10..=45).text("Mesh"));

            egui::ComboBox::from_id_source("surface_colormap_combo")
                .selected_text(self.surface_colormap.label())
                .show_ui(ui, |ui| {
                    for cm in [
                        SurfaceColorMap::OceanThermal,
                        SurfaceColorMap::Viridis,
                        SurfaceColorMap::CoolWarm,
                    ] {
                        ui.selectable_value(&mut self.surface_colormap, cm, cm.label());
                    }
                });

            ui.checkbox(&mut self.surface_wireframe_only, "Wireframe");
            if ui.button("Reset 3D View").clicked() {
                self.yaw_deg = 38.0_f32;
                self.pitch_deg = 28.0_f32;
                self.surface_zoom = 1.0_f32;
            }
        });

        let available = ui.available_size().max(Vec2::new(100.0_f32, 100.0_f32));
        let (rect, response) = ui.allocate_exact_size(available, Sense::click_and_drag());

        if response.dragged() {
            let delta = response.drag_delta();
            self.yaw_deg = (self.yaw_deg + delta.x * 0.45_f32).clamp(-180.0_f32, 180.0_f32);
            self.pitch_deg = (self.pitch_deg + delta.y * 0.35_f32).clamp(5.0_f32, 85.0_f32);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1_f32 {
                self.surface_zoom =
                    (self.surface_zoom * (1.0_f32 + scroll * 0.0015_f32)).clamp(0.3_f32, 2.5_f32);
            }
        }

        let clipped_rect = rect.intersect(ui.clip_rect());
        let painter = ui.painter_at(clipped_rect);
        painter.rect_filled(rect, Rounding::same(6.0_f32), pal.plot_bg);
        painter.rect_stroke(
            rect,
            Rounding::same(6.0_f32),
            Stroke::new(1.0_f32, pal.border),
        );

        let expr_str = self
            .workspace
            .functions
            .get(&self.selected_3d_function)
            .cloned()
            .or_else(|| self.workspace.functions.values().next().cloned())
            .unwrap_or_else(|| "sin(x) * cos(y)".to_string());

        let compiled_3d = math::parse_expr_with_known_fns(
            &expr_str,
            &self.workspace.compiled_math.known_fn_names,
        )
        .ok()
        .map(|ast| CompiledExpr::compile(&ast));
        let ext_bridge = REngineMathBridge {
            r_objects: &self.workspace.r_objects,
        };
        let eval_ctx = self.build_eval_context(&ext_bridge);
        let mut scratch_3d = EvalScratch::new();

        let n = self.surface_resolution.clamp(8, 50);
        let x_min = self.domain_min;
        let x_max = self.domain_max.max(x_min + 1e-3_f64);
        let y_min = self.domain_y_min;
        let y_max = self.domain_y_max.max(y_min + 1e-3_f64);

        let mut z_grid = vec![vec![f64::NAN; n + 1]; n + 1];
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        if let Some(ref c_expr) = compiled_3d {
            c_expr.bind_slots(
                &eval_ctx,
                &[("x", x_min), ("y", y_min), ("L", x_min), ("K", y_min)],
                &mut scratch_3d,
            );
            let x_slots: Vec<usize> = ["x", "L"]
                .iter()
                .filter_map(|s| c_expr.slot_of(s))
                .collect();
            let y_slots: Vec<usize> = ["y", "K"]
                .iter()
                .filter_map(|s| c_expr.slot_of(s))
                .collect();

            for (i, row) in z_grid.iter_mut().enumerate() {
                let x = x_min + (x_max - x_min) * (i as f64) / (n as f64);
                for &s in &x_slots {
                    c_expr.set_slot(s, x, &mut scratch_3d);
                }
                for (j, cell) in row.iter_mut().enumerate() {
                    let y = y_min + (y_max - y_min) * (j as f64) / (n as f64);
                    for &s in &y_slots {
                        c_expr.set_slot(s, y, &mut scratch_3d);
                    }
                    if let Ok(z) = c_expr.eval_with_bound_slots(&eval_ctx, &mut scratch_3d) {
                        if z.is_finite() && z.abs() < 1e6_f64 {
                            *cell = z;
                            if z < z_min {
                                z_min = z;
                            }
                            if z > z_max {
                                z_max = z;
                            }
                        }
                    }
                }
            }
        }

        if !z_min.is_finite() || !z_max.is_finite() {
            z_min = -1.0_f64;
            z_max = 1.0_f64;
        }
        let z_span = (z_max - z_min).abs().max(1e-6_f64);

        let yaw = self.yaw_deg.to_radians();
        let pitch = self.pitch_deg.to_radians();
        let (cos_y, sin_y) = (yaw.cos(), yaw.sin());
        let (cos_p, sin_p) = (pitch.cos(), pitch.sin());

        let scale = (rect.width().min(rect.height()) * 0.34_f32) * self.surface_zoom;
        let center = rect.center() + Vec2::new(0.0_f32, rect.height() * 0.05_f32);

        let project = |nx: f32, ny: f32, nz: f32| -> (Pos2, f32) {
            let cx = nx * 2.0_f32 - 1.0_f32;
            let cy = ny * 2.0_f32 - 1.0_f32;
            let cz = nz * 1.4_f32 - 0.5_f32;

            let rx = cx * cos_y - cy * sin_y;
            let ry = cx * sin_y + cy * cos_y;
            let sx = rx;
            let sy = ry * sin_p - cz * cos_p;
            let depth = ry * cos_p + cz * sin_p;

            (
                Pos2::new(center.x + sx * scale, center.y + sy * scale),
                depth,
            )
        };

        let (origin_p, _) = project(0.0_f32, 0.0_f32, 0.0_f32);
        let (x_axis_p, _) = project(1.0_f32, 0.0_f32, 0.0_f32);
        let (y_axis_p, _) = project(0.0_f32, 1.0_f32, 0.0_f32);
        let (z_axis_p, _) = project(0.0_f32, 0.0_f32, 1.0_f32);

        painter.line_segment(
            [origin_p, x_axis_p],
            Stroke::new(1.5_f32, pal.accent_blue),
        );
        painter.line_segment(
            [origin_p, y_axis_p],
            Stroke::new(1.5_f32, pal.accent_green),
        );
        painter.line_segment(
            [origin_p, z_axis_p],
            Stroke::new(1.5_f32, pal.accent_amber),
        );

        painter.text(
            x_axis_p,
            egui::Align2::LEFT_CENTER,
            format!("X ({:.1})", x_max),
            FontId::monospace(11.0_f32),
            pal.accent_blue,
        );
        painter.text(
            y_axis_p,
            egui::Align2::LEFT_CENTER,
            format!("Y ({:.1})", y_max),
            FontId::monospace(11.0_f32),
            pal.accent_green,
        );
        painter.text(
            z_axis_p,
            egui::Align2::CENTER_BOTTOM,
            format!("Z ({:.2})", z_max),
            FontId::monospace(11.0_f32),
            pal.accent_amber,
        );

        struct SurfaceQuad {
            pts: [Pos2; 4],
            depth: f32,
            norm_z: f32,
        }

        let mut quads = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let (z00, z10, z11, z01) = (
                    z_grid[i][j],
                    z_grid[i + 1][j],
                    z_grid[i + 1][j + 1],
                    z_grid[i][j + 1],
                );
                if !z00.is_finite() || !z10.is_finite() || !z11.is_finite() || !z01.is_finite() {
                    continue;
                }

                let nx0 = (i as f32) / (n as f32);
                let nx1 = ((i + 1) as f32) / (n as f32);
                let ny0 = (j as f32) / (n as f32);
                let ny1 = ((j + 1) as f32) / (n as f32);

                let nz00 = ((z00 - z_min) / z_span) as f32;
                let nz10 = ((z10 - z_min) / z_span) as f32;
                let nz11 = ((z11 - z_min) / z_span) as f32;
                let nz01 = ((z01 - z_min) / z_span) as f32;

                let (p0, d0) = project(nx0, ny0, nz00);
                let (p1, d1) = project(nx1, ny0, nz10);
                let (p2, d2) = project(nx1, ny1, nz11);
                let (p3, d3) = project(nx0, ny1, nz01);

                quads.push(SurfaceQuad {
                    pts: [p0, p1, p2, p3],
                    depth: 0.25_f32 * (d0 + d1 + d2 + d3),
                    norm_z: 0.25_f32 * (nz00 + nz10 + nz11 + nz01),
                });
            }
        }

        quads.sort_by(|a, b| {
            b.depth
                .partial_cmp(&a.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for q in &quads {
            let (r, g, b) = self.surface_colormap.map_normalized(q.norm_z);
            if !self.surface_wireframe_only {
                painter.add(egui::Shape::convex_polygon(
                    q.pts.to_vec(),
                    Color32::from_rgba_unmultiplied(r, g, b, 215),
                    Stroke::new(
                        0.6_f32,
                        Color32::from_rgba_unmultiplied(15, 23, 42, 130),
                    ),
                ));
            } else {
                painter.add(egui::Shape::closed_line(
                    q.pts.to_vec(),
                    Stroke::new(1.1_f32, Color32::from_rgb(r, g, b)),
                ));
            }
        }

        painter.text(
            rect.left_top() + Vec2::new(12.0_f32, 8.0_f32),
            egui::Align2::LEFT_TOP,
            format!(
                "3D Surface: {} = {}  |  Z ∈ [{:.2}, {:.2}]",
                self.selected_3d_function, expr_str, z_min, z_max
            ),
            FontId::monospace(12.0_f32),
            pal.text_primary,
        );
    }

    // =========================================================================
    // CARD 4 — CENTER BOTTOM: CONSOLE | SCRIPT | SETTINGS
    // =========================================================================

    fn render_bottom_dock(&mut self, ui: &mut Ui) {
        let pal = self.palette();
        render_bento_card_frame(ui, pal, None, |ui| {
            let act = render_card_tab_header(
                ui,
                pal,
                &mut self.bottom_tab,
                &[
                    (BottomDockTab::QuickConsole, "Console"),
                    (BottomDockTab::ScriptEditor, "Script"),
                    (BottomDockTab::NumericAndGraphSettings, "Settings"),
                ],
                false,
                true,
                true,
                "Hide Bottom Console & Script Card",
                "Pop-Out Console / Script Window",
            );
            if act.minimize_clicked {
                self.show_bottom_panel = false;
            }
            if act.popout_clicked {
                self.floating_console_window = !self.floating_console_window;
            }
            self.render_bottom_dock_body(ui, pal);
        });
    }

    fn render_bottom_dock_body(&mut self, ui: &mut Ui, pal: ThemePalette) {
        // Sub-toolbar matching Bento IDE aesthetic
        render_card_sub_toolbar(ui, pal, |ui| {
            if ui
                .add(
                    egui::Button::new(
                        RichText::new("▶ Run")
                            .font(FontId::proportional(11.6_f32))
                            .strong()
                            .color(pal.accent_green),
                    )
                    .fill(pal.pill_bg)
                    .stroke(Stroke::new(1.0_f32, pal.border))
                    .rounding(Rounding::same(5.0_f32)),
                )
                .on_hover_text("Run Script Editor (Ctrl+Enter) or active Console input")
                .clicked()
            {
                if self.bottom_tab == BottomDockTab::QuickConsole
                    && !self.quick_command_input.trim().is_empty()
                {
                    let cmd = self.quick_command_input.clone();
                    self.quick_command_input.clear();
                    self.execute_console_command(&cmd);
                } else {
                    self.evaluate_editor_script();
                }
            }
            if ui
                .add(pill_button("🧹 Clear", false, pal))
                .on_hover_text("Clear console output")
                .clicked()
            {
                self.console_logs.clear();
                self.status_banner = "Console cleared.".to_string();
            }
            if ui
                .add(pill_button("📋 Copy", false, pal))
                .on_hover_text("Copy console transcript to clipboard")
                .clicked()
            {
                let transcript: String = self
                    .console_logs
                    .iter()
                    .map(|e| format!("> {}\n{}", e.source_line, e.message))
                    .collect::<Vec<_>>()
                    .join("\n");
                ui.ctx().copy_text(transcript);
                self.status_banner = "Copied console output to clipboard.".to_string();
            }
            if ui
                .add(pill_button("📂 Load .R", false, pal))
                .on_hover_text("Load exports/workspace_script.R into Script Editor")
                .clicked()
            {
                let path = PathBuf::from("exports/workspace_script.R");
                match load_text_file(&path) {
                    Ok(txt) => {
                        self.editor_script = txt;
                        self.bottom_tab = BottomDockTab::ScriptEditor;
                        self.status_banner = "Loaded exports/workspace_script.R".to_string();
                    }
                    Err(_) => {
                        self.show_left_panel = true;
                        self.left_bottom_collapsed = false;
                        self.left_bottom_tab = LeftBottomTab::Files;
                        self.file_filter = FileFilterKind::Scripts;
                        self.status_banner =
                            "Select a .R script from the Files explorer.".to_string();
                    }
                }
            }
            if ui
                .add(pill_button("💾 Save .R", false, pal))
                .on_hover_text("Save Script Editor to exports/workspace_script.R")
                .clicked()
            {
                self.save_editor_script_default();
            }
            if ui
                .add(pill_button("❓ Help", false, pal))
                .on_hover_text("Open interactive help (`general-help`)")
                .clicked()
            {
                self.execute_console_command("general-help");
                self.floating_help_window = true;
            }
        });

        match self.bottom_tab {
            BottomDockTab::QuickConsole => {
                egui::Frame::none()
                    .fill(pal.plot_bg)
                    .rounding(Rounding {
                        nw: 0.0_f32,
                        ne: 0.0_f32,
                        sw: 8.0_f32,
                        se: 8.0_f32,
                    })
                    .inner_margin(Margin::symmetric(10.0_f32, 6.0_f32))
                    .show(ui, |ui| {
                        let prompt_h = 28.0_f32;
                        let scroll_h = (ui.available_height() - prompt_h - 4.0_f32).max(40.0_f32);

                        egui::ScrollArea::vertical()
                            .id_source("quick_console_scroll")
                            .stick_to_bottom(true)
                            .auto_shrink([false, false])
                            .max_height(scroll_h)
                            .show(ui, |ui| {
                                if self.console_logs.is_empty() {
                                    ui.monospace(
                                        RichText::new("Interactive console ready.")
                                            .font(FontId::monospace(12.2_f32))
                                            .color(pal.text_muted),
                                    );
                                } else {
                                    for entry in &self.console_logs {
                                        // Command echo line: `> command`
                                        ui.horizontal_wrapped(|ui| {
                                            ui.monospace(
                                                RichText::new(">")
                                                    .font(FontId::monospace(12.2_f32))
                                                    .strong()
                                                    .color(pal.accent_blue),
                                            )
                                            .on_hover_text(format!(
                                                "Source: {}",
                                                entry.timestamp_tag
                                            ));
                                            ui.monospace(
                                                RichText::new(&entry.source_line)
                                                    .font(FontId::monospace(12.2_f32))
                                                    .color(pal.text_primary),
                                            )
                                            .on_hover_text(format!(
                                                "Source: {}",
                                                entry.timestamp_tag
                                            ));
                                        });
                                        // Output line(s) right below
                                        let msg_color = if entry.is_error {
                                            pal.accent_red
                                        } else {
                                            pal.text_secondary
                                        };
                                        ui.monospace(
                                            RichText::new(&entry.message)
                                                .font(FontId::monospace(12.0_f32))
                                                .color(msg_color),
                                        );
                                        ui.add_space(3.0_f32);
                                    }
                                }
                            });

                        ui.add_space(3.0_f32);
                        // Integrated terminal prompt at the bottom
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(">")
                                    .font(FontId::monospace(13.5_f32))
                                    .strong()
                                    .color(pal.accent_blue),
                            );
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut self.quick_command_input)
                                    .font(FontId::monospace(12.4_f32))
                                    .frame(false)
                                    .desired_width((ui.available_width() - 56.0_f32).max(80.0_f32))
                                    .hint_text(
                                        "Type R or Math: `f(seq(1,5))`, `lm(y ~ x, data = df)`, `2 + 2`, `general-help`",
                                    ),
                            );
                            let enter_pressed =
                                resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("↵")
                                            .strong()
                                            .color(pal.accent_green),
                                    )
                                    .fill(pal.pill_bg)
                                    .stroke(Stroke::new(1.0_f32, pal.border))
                                    .rounding(Rounding::same(4.0_f32)),
                                )
                                .on_hover_text("Execute command (Enter)")
                                .clicked()
                                || enter_pressed
                            {
                                let cmd = self.quick_command_input.clone();
                                self.quick_command_input.clear();
                                self.execute_console_command(&cmd);
                                resp.request_focus();
                            }
                        });
                    });
            }
            BottomDockTab::ScriptEditor => {
                if ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Enter)) {
                    self.evaluate_editor_script();
                }
                egui::Frame::none()
                    .fill(pal.plot_bg)
                    .rounding(Rounding {
                        nw: 0.0_f32,
                        ne: 0.0_f32,
                        sw: 8.0_f32,
                        se: 8.0_f32,
                    })
                    .inner_margin(Margin::same(6.0_f32))
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .id_source("script_editor_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.add_sized(
                                    ui.available_size().max(Vec2::new(120.0_f32, 90.0_f32)),
                                    egui::TextEdit::multiline(&mut self.editor_script)
                                        .code_editor()
                                        .desired_width(f32::INFINITY)
                                        .desired_rows(7),
                                );
                            });
                    });
            }
            BottomDockTab::NumericAndGraphSettings => {
                egui::Frame::none()
                    .inner_margin(Margin::symmetric(10.0_f32, 6.0_f32))
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .id_source("numeric_settings_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.columns(2, |cols| {
                                    cols[0].label(
                                        RichText::new("1. Home View & Numerical Approximation")
                                            .strong()
                                            .color(pal.accent_blue),
                                    );
                                    cols[0].horizontal_wrapped(|ui| {
                                        ui.label("Home X:");
                                        ui.add(
                                            egui::DragValue::new(&mut self.home_x_min)
                                                .speed(0.5_f64),
                                        );
                                        ui.label("to");
                                        ui.add(
                                            egui::DragValue::new(&mut self.home_x_max)
                                                .speed(0.5_f64),
                                        );
                                    });
                                    cols[0].horizontal_wrapped(|ui| {
                                        ui.label("Home Y:");
                                        ui.add(
                                            egui::DragValue::new(&mut self.home_y_min)
                                                .speed(0.5_f64),
                                        );
                                        ui.label("to");
                                        ui.add(
                                            egui::DragValue::new(&mut self.home_y_max)
                                                .speed(0.5_f64),
                                        );
                                        if ui.small_button("Use Current View").clicked() {
                                            self.home_x_min = self.domain_min;
                                            self.home_x_max = self.domain_max;
                                            self.home_y_min = self.domain_y_min;
                                            self.home_y_max = self.domain_y_max;
                                        }
                                    });
                                    cols[0].add(
                                        egui::Slider::new(&mut self.curve_samples, 64..=2000)
                                            .text("Curve Render Samples"),
                                    );
                                    cols[0].add(
                                        egui::Slider::new(&mut self.sim_grid_rows, 10..=500)
                                            .text("Simulation Grid Rows"),
                                    );
                                    cols[0].add(
                                        egui::Slider::new(
                                            &mut self.simpson_subintervals,
                                            20..=1000,
                                        )
                                        .text("Simpson Integral Steps N"),
                                    );
                                    cols[0].add(
                                        egui::Slider::new(
                                            &mut self.diff_step_h,
                                            1e-7_f64..=1e-2_f64,
                                        )
                                        .logarithmic(true)
                                        .text("Finite Diff Step h"),
                                    );

                                    cols[1].label(
                                        RichText::new("2. Opt-In Calculus & Graph Overlays")
                                            .strong()
                                            .color(pal.accent_purple),
                                    );
                                    cols[1]
                                        .checkbox(&mut self.show_plot_grid, "Show Coordinate Grid");
                                    cols[1]
                                        .checkbox(&mut self.show_plot_legend, "Show Series Legend");
                                    cols[1].checkbox(
                                        &mut self.lock_equal_aspect,
                                        "Lock 1:1 Equal Aspect Ratio",
                                    );
                                    cols[1].separator();
                                    cols[1].checkbox(
                                        &mut self.show_derivative_overlay,
                                        "Overlay Numerical Derivative f'(x)",
                                    );
                                    cols[1].checkbox(
                                        &mut self.show_integral_shade,
                                        "Shade Definite Integral ∫ f(x) dx",
                                    );
                                    cols[1].checkbox(
                                        &mut self.show_equilibrium_overlay,
                                        "Plot Curve Intersection / Equilibrium Marker",
                                    );
                                    cols[1].checkbox(
                                        &mut self.enable_equilibrium_solver,
                                        "Show Solved Equilibrium Panel in Symbolic",
                                    );
                                    cols[1].horizontal_wrapped(|ui| {
                                        ui.checkbox(
                                            &mut self.show_probe_marker,
                                            "Calculus Probe x₀:",
                                        );
                                        ui.add(
                                            egui::Slider::new(
                                                &mut self.probe_x,
                                                self.domain_min..=self.domain_max,
                                            )
                                            .show_value(true),
                                        );
                                    });
                                });
                            });
                    });
            }
        }
    }

    // =========================================================================
    // RIGHT COLUMN BENTO STACK: CARD 5A (ENVIRONMENT) + CARD 5B (DATA / NOTES / LATEX / HELP)
    // =========================================================================

    fn render_right_column(&mut self, ui: &mut Ui) {
        let pal = self.palette();
        let avail = ui.available_size();
        let gap = 6.0_f32;
        let header_h = 34.0_f32;

        let (top_h, bot_h) = match (self.env_card_collapsed, self.right_bottom_collapsed) {
            (true, true) => (header_h, header_h),
            (true, false) => (header_h, (avail.y - header_h - gap).max(120.0_f32)),
            (false, true) => ((avail.y - header_h - gap).max(140.0_f32), header_h),
            (false, false) => {
                let h1 = ((avail.y - gap) * 0.42_f32).clamp(130.0_f32, (avail.y - 160.0_f32).max(130.0_f32));
                let h2 = (avail.y - h1 - gap).max(120.0_f32);
                (h1, h2)
            }
        };

        render_bento_card_frame(ui, pal, Some(Vec2::new(avail.x, top_h)), |ui| {
            self.render_environment_card(ui, pal);
        });

        ui.add_space(gap);

        render_bento_card_frame(ui, pal, Some(Vec2::new(avail.x, bot_h)), |ui| {
            self.render_right_bottom_card(ui, pal);
        });
    }

    fn render_environment_card(&mut self, ui: &mut Ui, pal: ThemePalette) {
        let mut clear_env = false;
        let mut toggle_min = false;
        let mut toggle_popout = false;

        render_card_title_header(ui, pal, "Environment", self.env_card_collapsed, |ui| {
            if ui
                .add(header_icon_button("▢", pal.text_secondary))
                .on_hover_text("Pop-Out Environment Inspector")
                .clicked()
            {
                toggle_popout = true;
            }
            if ui
                .add(header_icon_button(
                    if self.env_card_collapsed { "+" } else { "—" },
                    pal.text_secondary,
                ))
                .on_hover_text("Collapse Environment Card")
                .clicked()
            {
                toggle_min = true;
            }
            if ui
                .add(header_icon_button("rm()", pal.text_muted))
                .on_hover_text("Clear R environment (`rm(list=ls())`)")
                .clicked()
            {
                clear_env = true;
            }
        });

        if clear_env {
            self.workspace.r_objects.clear();
            self.status_banner = "Cleared R environment objects.".to_string();
        }
        if toggle_min {
            self.env_card_collapsed = !self.env_card_collapsed;
        }
        if toggle_popout {
            self.floating_env_window = !self.floating_env_window;
        }
        if self.env_card_collapsed {
            return;
        }

        self.render_environment_body(ui, pal);
    }

    fn render_right_bottom_card(&mut self, ui: &mut Ui, pal: ThemePalette) {
        let act = render_card_tab_header(
            ui,
            pal,
            &mut self.right_tab,
            &[
                (RightDockTab::Datasets, "Data Viewer"),
                (RightDockTab::AnalysisAndLatex, "LaTeX & R"),
                (RightDockTab::NotesAndReports, "Notes"),
                (RightDockTab::HelpCenter, "Help"),
            ],
            self.right_bottom_collapsed,
            true,
            true,
            "Collapse Data / Notes Card",
            "Pop-Out Active Tab in Floating Window",
        );

        if act.minimize_clicked {
            self.right_bottom_collapsed = !self.right_bottom_collapsed;
        }
        if act.popout_clicked {
            match self.right_tab {
                RightDockTab::Datasets => {
                    self.floating_datasets_window = !self.floating_datasets_window;
                }
                RightDockTab::NotesAndReports | RightDockTab::AnalysisAndLatex => {
                    self.floating_notes_window = !self.floating_notes_window;
                }
                RightDockTab::HelpCenter => {
                    self.floating_help_window = !self.floating_help_window;
                }
            }
        }
        if self.right_bottom_collapsed {
            return;
        }

        match self.right_tab {
            RightDockTab::Datasets => self.render_datasets_manager_ui(ui, pal),
            RightDockTab::NotesAndReports => self.render_notes_and_reports_ui(ui, pal),
            RightDockTab::AnalysisAndLatex => self.render_analysis_and_latex_ui(ui, pal),
            RightDockTab::HelpCenter => self.render_help_center_ui(ui, pal),
        }
    }

    fn render_datasets_manager_ui(&mut self, ui: &mut Ui, pal: ThemePalette) {
        // Icon-driven sub-toolbar matching Bento Data Viewer
        let mut add_new_ds = false;
        let mut gen_from_fn = false;
        let mut add_row = false;
        let mut export_csv = false;

        if !self.workspace.datasets.is_empty()
            && self.workspace.active_dataset_idx >= self.workspace.datasets.len()
        {
            self.workspace.active_dataset_idx = 0;
        }

        render_card_sub_toolbar(ui, pal, |ui| {
            if let Some(ds) = self.workspace.datasets.get_mut(self.workspace.active_dataset_idx) {
                if ui
                    .add(pill_button("📈", ds.show_on_plot, pal))
                    .on_hover_text("Toggle Scatter Plot Overlay on 2D Graph")
                    .clicked()
                {
                    ds.show_on_plot = !ds.show_on_plot;
                }
                if ui
                    .add(pill_button("📉", ds.show_ols_line, pal))
                    .on_hover_text("Toggle OLS Regression Line Overlay")
                    .clicked()
                {
                    ds.show_ols_line = !ds.show_ols_line;
                }
                if ui
                    .add(pill_button(
                        if self.dataset_show_raw_csv { "CSV" } else { "⊞" },
                        self.dataset_show_raw_csv,
                        pal,
                    ))
                    .on_hover_text("Switch between Table Grid and Raw CSV Pasteboard")
                    .clicked()
                {
                    self.dataset_show_raw_csv = !self.dataset_show_raw_csv;
                }
                ui.color_edit_button_srgba(&mut ds.color);
            }

            if ui
                .add(pill_button("+ New", false, pal))
                .on_hover_text("Create a new empty dataset")
                .clicked()
            {
                add_new_ds = true;
            }
            if ui
                .add(pill_button("ƒ(x)→Tbl", false, pal))
                .on_hover_text("Generate Dataset sampled from primary f(x)")
                .clicked()
            {
                gen_from_fn = true;
            }
            if ui
                .add(pill_button("+ Row", false, pal))
                .on_hover_text("Append new (x, y) row")
                .clicked()
            {
                add_row = true;
            }
            if ui
                .add(pill_button("💾 CSV", false, pal))
                .on_hover_text("Export active dataset to exports/<name>.csv")
                .clicked()
            {
                export_csv = true;
            }
        });

        if add_new_ds {
            let name = if self.new_dataset_name_input.trim().is_empty() {
                format!("data_{}", self.workspace.datasets.len() + 1)
            } else {
                self.new_dataset_name_input.trim().to_string()
            };
            self.workspace.datasets.push(DataTable::from_csv_str(
                name,
                "x,y\n1.0,2.0\n2.0,4.0\n3.0,6.0",
                Color32::from_rgb(56, 189, 248),
            ));
            self.workspace.active_dataset_idx = self.workspace.datasets.len() - 1;
        }
        if gen_from_fn {
            self.generate_dataset_from_primary_fn();
        }

        egui::Frame::none()
            .inner_margin(Margin::symmetric(8.0_f32, 6.0_f32))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_dataset_name_input)
                            .font(FontId::monospace(11.2_f32))
                            .desired_width(84.0_f32)
                            .hint_text("dataset_name"),
                    );
                    for idx in 0..self.workspace.datasets.len() {
                        let name = self.workspace.datasets[idx].name.clone();
                        let sel = self.workspace.active_dataset_idx == idx;
                        if ui.add(pill_button(&name, sel, pal)).clicked() {
                            self.workspace.active_dataset_idx = idx;
                        }
                    }
                });

                if !self.workspace.datasets.is_empty() {
                    let active_idx = self.workspace.active_dataset_idx;
                    let ds = &mut self.workspace.datasets[active_idx];

                    if add_row {
                        ds.points.push([0.0_f64, 0.0_f64]);
                        ds.sync_raw_csv_from_points();
                    }
                    if export_csv {
                        let path = PathBuf::from(format!("exports/{}.csv", ds.name));
                        if save_text_file(&path, &ds.raw_csv).is_ok() {
                            self.status_banner = format!("Exported dataset to {:?}", path);
                        }
                    }

                    if let Some((b1, b0, r2)) = Self::compute_points_ols(&ds.points) {
                        ui.add_space(3.0_f32);
                        ui.label(
                            RichText::new(format!(
                                "n={} | y = {:.3}x + {:.3} (R²={:.3})",
                                ds.points.len(),
                                b1,
                                b0,
                                r2
                            ))
                            .font(FontId::monospace(11.0_f32))
                            .color(pal.accent_green),
                        );
                    }

                    ui.add_space(4.0_f32);

                    if self.dataset_show_raw_csv {
                        ui.label(
                            RichText::new("Raw CSV Pasteboard (Edit or Paste from Excel/R):")
                                .font(FontId::proportional(11.0_f32))
                                .color(pal.text_secondary),
                        );
                        egui::ScrollArea::vertical()
                            .id_source("raw_csv_editor_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                if ui
                                    .add_sized(
                                        ui.available_size().max(Vec2::new(100.0_f32, 90.0_f32)),
                                        egui::TextEdit::multiline(&mut ds.raw_csv)
                                            .code_editor()
                                            .desired_width(f32::INFINITY),
                                    )
                                    .changed()
                                {
                                    ds.sync_from_raw_csv();
                                }
                            });
                    } else {
                        egui::ScrollArea::vertical()
                            .id_source("active_dataset_table_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                let mut changed_cell = false;
                                let mut delete_row: Option<usize> = None;
                                egui::Grid::new("dataset_interactive_grid")
                                    .striped(true)
                                    .spacing([10.0_f32, 4.0_f32])
                                    .show(ui, |ui| {
                                        ui.label(RichText::new("#").strong().color(pal.text_muted));
                                        ui.label(
                                            RichText::new("x").strong().color(pal.accent_blue),
                                        );
                                        ui.label(
                                            RichText::new("y").strong().color(pal.accent_green),
                                        );
                                        ui.label("");
                                        ui.end_row();

                                        for (r_idx, pt) in ds.points.iter_mut().enumerate() {
                                            ui.monospace(format!("{:02}", r_idx + 1));
                                            changed_cell |= ui
                                                .add(
                                                    egui::DragValue::new(&mut pt[0]).speed(0.1_f64),
                                                )
                                                .changed();
                                            changed_cell |= ui
                                                .add(
                                                    egui::DragValue::new(&mut pt[1]).speed(0.1_f64),
                                                )
                                                .changed();
                                            if ui
                                                .add(header_icon_button("×", pal.accent_red))
                                                .clicked()
                                            {
                                                delete_row = Some(r_idx);
                                            }
                                            ui.end_row();
                                        }
                                    });
                                if let Some(r) = delete_row {
                                    ds.points.remove(r);
                                    ds.sync_raw_csv_from_points();
                                } else if changed_cell {
                                    ds.sync_raw_csv_from_points();
                                }
                            });
                    }
                }
            });
    }

    fn render_notes_and_reports_ui(&mut self, ui: &mut Ui, pal: ThemePalette) {
        render_card_sub_toolbar(ui, pal, |ui| {
            ui.selectable_value(&mut self.notes_preview_mode, false, "✏ Edit");
            ui.selectable_value(&mut self.notes_preview_mode, true, "👁 Reader");
            if ui.add(pill_button("📂", false, pal)).on_hover_text("Import file").clicked() {
                let path = PathBuf::from(self.file_io_path_input.trim());
                match load_text_file(&path) {
                    Ok(txt) => {
                        self.workspace.markdown_notes = txt;
                        self.status_banner = format!("Loaded notes from {:?}", path);
                    }
                    Err(e) => self.status_banner = e,
                }
            }
            if ui.add(pill_button("💾", false, pal)).on_hover_text("Save notes file").clicked() {
                self.save_notes_default();
            }
            if ui
                .add(pill_button("+ LaTeX", false, pal))
                .on_hover_text("Append current LaTeX derivation block to notes")
                .clicked()
            {
                let block = self.generate_latex_derivation();
                self.workspace.markdown_notes.push_str("\n\n");
                self.workspace.markdown_notes.push_str(&block);
            }
        });

        egui::Frame::none()
            .inner_margin(Margin::symmetric(8.0_f32, 6.0_f32))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.file_io_path_input)
                            .font(FontId::monospace(11.2_f32))
                            .desired_width((ui.available_width() - 8.0_f32).max(100.0_f32))
                            .hint_text("exports/notes.qmd"),
                    );
                });
                ui.add_space(4.0_f32);

                let editor_h = (ui.available_height() - 62.0_f32).max(80.0_f32);
                egui::ScrollArea::vertical()
                    .id_source("notes_markdown_scroll")
                    .max_height(editor_h)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.notes_preview_mode {
                            for line in self.workspace.markdown_notes.lines() {
                                if let Some(h1) = line.strip_prefix("# ") {
                                    ui.label(
                                        RichText::new(h1)
                                            .font(FontId::proportional(15.5_f32))
                                            .strong()
                                            .color(pal.accent_blue),
                                    );
                                } else if let Some(h2) = line.strip_prefix("## ") {
                                    ui.label(
                                        RichText::new(h2)
                                            .font(FontId::proportional(13.8_f32))
                                            .strong()
                                            .color(pal.accent_purple),
                                    );
                                } else if let Some(h3) = line.strip_prefix("### ") {
                                    ui.label(
                                        RichText::new(h3)
                                            .font(FontId::proportional(13.0_f32))
                                            .strong()
                                            .color(pal.text_primary),
                                    );
                                } else {
                                    ui.label(RichText::new(line).color(pal.text_primary));
                                }
                            }
                        } else {
                            ui.add_sized(
                                Vec2::new(ui.available_width(), editor_h),
                                egui::TextEdit::multiline(&mut self.workspace.markdown_notes)
                                    .desired_width(f32::INFINITY)
                                    .hint_text(
                                        "Write Markdown, Quarto (.qmd), or RMarkdown (.Rmd) notes...",
                                    ),
                            );
                        }
                    });

                ui.add_space(4.0_f32);
                ui.horizontal_wrapped(|ui| {
                    for fmt in [
                        ReportFormat::QuartoQmd,
                        ReportFormat::RMarkdownRmd,
                        ReportFormat::MarkdownMd,
                    ] {
                        ui.selectable_value(&mut self.report_format, fmt, fmt.label());
                    }
                });
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("🚀 Export Bundle & Open Positron")
                                .strong()
                                .color(pal.accent_green),
                        )
                        .fill(pal.pill_bg)
                        .stroke(Stroke::new(1.0_f32, pal.border))
                        .rounding(Rounding::same(6.0_f32)),
                    )
                    .clicked()
                {
                    self.trigger_export_pipeline();
                }
            });
    }

    fn render_analysis_and_latex_ui(&mut self, ui: &mut Ui, pal: ThemePalette) {
        let mut latex_str = self.generate_latex_derivation();
        let mut r_str = self.generate_r_script_preview();

        render_card_sub_toolbar(ui, pal, |ui| {
            if ui.add(pill_button("📋 Copy LaTeX / TikZ", false, pal)).clicked() {
                ui.ctx().copy_text(latex_str.clone());
                self.status_banner = "Copied LaTeX derivation to clipboard.".to_string();
            }
            if ui.add(pill_button("📋 Copy R Script", false, pal)).clicked() {
                ui.ctx().copy_text(r_str.clone());
                self.status_banner = "Copied R script to clipboard.".to_string();
            }
        });

        egui::Frame::none()
            .inner_margin(Margin::symmetric(8.0_f32, 6.0_f32))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_source("analysis_latex_r_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new("Auto-Derived LaTeX & TikZ")
                                .strong()
                                .color(pal.accent_blue),
                        );
                        ui.add_sized(
                            Vec2::new(ui.available_width(), 165.0_f32),
                            egui::TextEdit::multiline(&mut latex_str)
                                .code_editor()
                                .desired_width(f32::INFINITY),
                        );
                        ui.add_space(6.0_f32);
                        ui.label(
                            RichText::new("Reproducible R / ggplot2 Script")
                                .strong()
                                .color(pal.accent_green),
                        );
                        ui.add_sized(
                            Vec2::new(ui.available_width(), 165.0_f32),
                            egui::TextEdit::multiline(&mut r_str)
                                .code_editor()
                                .desired_width(f32::INFINITY),
                        );
                    });
            });
    }

    fn render_help_center_ui(&mut self, ui: &mut Ui, pal: ThemePalette) {
        egui::Frame::none()
            .inner_margin(Margin::symmetric(8.0_f32, 6.0_f32))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_source("help_center_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        capsule_frame(pal, pal.border).show(ui, |ui| {
                            ui.monospace(
                                RichText::new(Self::intro_help_text())
                                    .font(FontId::monospace(11.2_f32))
                                    .color(pal.text_primary),
                            );
                        });
                        ui.add_space(6.0_f32);
                        capsule_frame(pal, pal.border).show(ui, |ui| {
                            ui.monospace(
                                RichText::new(Self::general_help_text())
                                    .font(FontId::monospace(11.2_f32))
                                    .color(pal.text_primary),
                            );
                        });
                    });
            });
    }

    // =========================================================================
    // FLOATING POP-OUT WINDOWS (DATASETS, NOTES, HELP, ENVIRONMENT, CONSOLE)
    // =========================================================================

    fn render_floating_windows(&mut self, ctx: &egui::Context) {
        let pal = self.palette();

        if self.floating_datasets_window {
            let mut open = self.floating_datasets_window;
            egui::Window::new("📊 Data Viewer & Datasets Manager")
                .open(&mut open)
                .default_size([500.0_f32, 430.0_f32])
                .resizable(true)
                .show(ctx, |ui| {
                    self.render_datasets_manager_ui(ui, pal);
                });
            self.floating_datasets_window = open;
        }

        if self.floating_notes_window {
            let mut open = self.floating_notes_window;
            egui::Window::new("📓 Notes & Literate Reports (.qmd / .Rmd / .md)")
                .open(&mut open)
                .default_size([520.0_f32, 460.0_f32])
                .resizable(true)
                .show(ctx, |ui| {
                    self.render_notes_and_reports_ui(ui, pal);
                });
            self.floating_notes_window = open;
        }

        if self.floating_help_window {
            let mut open = self.floating_help_window;
            egui::Window::new("❓ EconCalc Studio Help (`intro-help` & `general-help`)")
                .open(&mut open)
                .default_size([560.0_f32, 460.0_f32])
                .resizable(true)
                .show(ctx, |ui| {
                    self.render_help_center_ui(ui, pal);
                });
            self.floating_help_window = open;
        }

        if self.floating_env_window {
            let mut open = self.floating_env_window;
            egui::Window::new("📦 R Environment Inspector")
                .open(&mut open)
                .default_size([360.0_f32, 420.0_f32])
                .resizable(true)
                .show(ctx, |ui| {
                    self.render_environment_body(ui, pal);
                });
            self.floating_env_window = open;
        }

        if self.floating_console_window {
            let mut open = self.floating_console_window;
            egui::Window::new("⚡ Console / Script / Settings")
                .open(&mut open)
                .default_size([560.0_f32, 340.0_f32])
                .resizable(true)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.bottom_tab,
                            BottomDockTab::QuickConsole,
                            "Console",
                        );
                        ui.selectable_value(
                            &mut self.bottom_tab,
                            BottomDockTab::ScriptEditor,
                            "Script",
                        );
                        ui.selectable_value(
                            &mut self.bottom_tab,
                            BottomDockTab::NumericAndGraphSettings,
                            "Settings",
                        );
                    });
                    ui.separator();
                    self.render_bottom_dock_body(ui, pal);
                });
            self.floating_console_window = open;
        }
    }
}

impl eframe::App for EconCalcApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let pal = self.palette();
        let screen_w = ctx.screen_rect().width().max(640.0_f32);
        let screen_h = ctx.screen_rect().height().max(480.0_f32);

        // ---- 2-Tier Top Header ----
        egui::TopBottomPanel::top("bento_top_header")
            .frame(
                egui::Frame::none()
                    .fill(pal.card_bg)
                    .stroke(Stroke::new(1.0_f32, pal.border))
                    .inner_margin(Margin::symmetric(8.0_f32, 4.0_f32)),
            )
            .show(ctx, |ui| {
                self.render_top_header(ui);
            });

        // ---- Left Bento Column (Card 1: Symbolic + Card 2: Files / Models) ----
        if self.show_left_panel && !self.maximize_graph {
            egui::SidePanel::left("bento_left_column")
                .resizable(true)
                .default_width((screen_w * 0.24_f32).clamp(245.0_f32, 340.0_f32))
                .width_range(210.0_f32..=(screen_w * 0.38_f32).max(260.0_f32))
                .frame(
                    egui::Frame::none()
                        .fill(pal.outer_bg)
                        .inner_margin(Margin {
                            left: 6.0_f32,
                            right: 3.0_f32,
                            top: 6.0_f32,
                            bottom: 6.0_f32,
                        }),
                )
                .show(ctx, |ui| {
                    self.render_left_column(ui);
                });
        }

        // ---- Right Bento Column (Card 5A: Environment + Card 5B: Data Viewer / Notes / LaTeX / Help) ----
        if self.show_right_panel && !self.maximize_graph {
            egui::SidePanel::right("bento_right_column")
                .resizable(true)
                .default_width((screen_w * 0.24_f32).clamp(245.0_f32, 350.0_f32))
                .width_range(215.0_f32..=(screen_w * 0.38_f32).max(260.0_f32))
                .frame(
                    egui::Frame::none()
                        .fill(pal.outer_bg)
                        .inner_margin(Margin {
                            left: 3.0_f32,
                            right: 6.0_f32,
                            top: 6.0_f32,
                            bottom: 6.0_f32,
                        }),
                )
                .show(ctx, |ui| {
                    self.render_right_column(ui);
                });
        }

        // ---- Center Bottom Bento Card (Card 4: Console | Script | Settings) ----
        // Declared AFTER Left and Right SidePanels so it occupies strictly the bottom of the Center column!
        if self.show_bottom_panel && !self.maximize_graph {
            egui::TopBottomPanel::bottom("bento_center_bottom_card")
                .resizable(true)
                .default_height((screen_h * 0.29_f32).clamp(155.0_f32, 260.0_f32))
                .height_range(115.0_f32..=(screen_h * 0.56_f32).max(200.0_f32))
                .frame(
                    egui::Frame::none()
                        .fill(pal.outer_bg)
                        .inner_margin(Margin {
                            left: 3.0_f32,
                            right: 3.0_f32,
                            top: 3.0_f32,
                            bottom: 6.0_f32,
                        }),
                )
                .show(ctx, |ui| {
                    self.render_bottom_dock(ui);
                });
        }

        // ---- Center Top Bento Card (Card 3: Graph Viewport 2D/3D) ----
        let center_margin = if self.maximize_graph {
            Margin::same(6.0_f32)
        } else {
            Margin {
                left: if self.show_left_panel { 3.0_f32 } else { 6.0_f32 },
                right: if self.show_right_panel { 3.0_f32 } else { 6.0_f32 },
                top: 6.0_f32,
                bottom: if self.show_bottom_panel { 3.0_f32 } else { 6.0_f32 },
            }
        };
        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(pal.outer_bg)
                    .inner_margin(center_margin),
            )
            .show(ctx, |ui| {
                self.render_center_viewport(ui);
            });

        self.render_floating_windows(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporter::{ARCH_FULL_LABEL, ARCH_SHORT_LABEL};

    #[test]
    fn test_ast_latex_and_ident_normalization() {
        let latex = math::to_latex_from_ast(r"\alpha \cdot x^{\beta} + \frac{a}{b}");
        assert_eq!(latex, r"\alpha \cdot x^{\beta} + \frac{a}{b}");
        assert_eq!(EconCalcApp::normalize_ident_name(r"\alpha"), "alpha");
        assert_eq!(EconCalcApp::normalize_ident_name(r"\bar{Y}"), "Y_bar");
    }

    #[test]
    fn test_architecture_labels_non_empty() {
        assert!(!ARCH_SHORT_LABEL.is_empty());
        assert!(!ARCH_FULL_LABEL.is_empty());
    }

    #[test]
    fn test_exponential_and_asymptote_branch_splitting() {
        let mut ws = Workspace::new_default();
        ws.clear();
        let app = EconCalcApp {
            workspace: ws,
            dark_mode: true,
            show_left_panel: true,
            show_right_panel: true,
            show_bottom_panel: true,
            maximize_graph: false,
            symbolic_card_collapsed: false,
            left_bottom_collapsed: false,
            env_card_collapsed: false,
            right_bottom_collapsed: false,
            symbolic_functions_open: true,
            symbolic_sliders_open: true,
            dataset_show_raw_csv: false,
            floating_datasets_window: false,
            floating_notes_window: false,
            floating_help_window: false,
            floating_env_window: false,
            floating_console_window: false,
            active_model: ActiveModelKind::CustomExpression,
            viewport_mode: ViewportMode::Plot2D,
            axis_zoom_mode: AxisZoomMode::BothXY,
            left_bottom_tab: LeftBottomTab::Files,
            file_filter: FileFilterKind::All,
            bottom_tab: BottomDockTab::QuickConsole,
            right_tab: RightDockTab::Datasets,
            selected_snippet: SnippetTemplate::AsymptoticRational,
            report_format: ReportFormat::QuartoQmd,
            quick_command_input: String::new(),
            new_fn_sig_input: String::new(),
            new_fn_expr_input: String::new(),
            new_param_name_input: String::new(),
            new_param_val_input: String::new(),
            smart_math_shortcuts: true,
            editor_script: String::new(),
            console_logs: Vec::new(),
            eval_counter: 0,
            status_banner: String::new(),
            domain_min: -5.0_f64,
            domain_max: 5.0_f64,
            domain_y_min: -10.0_f64,
            domain_y_max: 10.0_f64,
            home_x_min: -5.0_f64,
            home_x_max: 5.0_f64,
            home_y_min: -10.0_f64,
            home_y_max: 10.0_f64,
            needs_plot_bounds_sync: false,
            pending_zoom_factor: None,
            lock_equal_aspect: false,
            show_plot_grid: true,
            show_plot_legend: true,
            curve_samples: 200,
            sim_grid_rows: 50,
            diff_step_h: 1e-5_f64,
            simpson_subintervals: 100,
            asymptote_jump_factor: 4.0_f64,
            probe_x: 1.0_f64,
            show_probe_marker: false,
            show_derivative_overlay: false,
            show_integral_shade: false,
            show_equilibrium_overlay: false,
            enable_equilibrium_solver: false,
            selected_3d_function: String::new(),
            yaw_deg: 38.0_f32,
            pitch_deg: 28.0_f32,
            surface_zoom: 1.0_f32,
            surface_resolution: 20,
            surface_wireframe_only: false,
            surface_colormap: SurfaceColorMap::OceanThermal,
            project_dir: PathBuf::from("."),
            project_files: Vec::new(),
            file_io_path_input: String::new(),
            new_dataset_name_input: String::new(),
            notes_preview_mode: false,
        };

        let val_e = app.eval_expression_str("e^x", 1.0_f64, 0.0_f64).unwrap();
        assert!((val_e - std::f64::consts::E).abs() < 1e-9_f64);
        assert_eq!(
            app.eval_expression_str("2x", 3.0_f64, 0.0_f64).unwrap(),
            6.0_f64
        );
        assert_eq!(
            app.eval_expression_str("2(x + 1)", 3.0_f64, 0.0_f64)
                .unwrap(),
            8.0_f64
        );
        assert_eq!(
            app.eval_expression_str("(x + 1)(x - 1)", 3.0_f64, 0.0_f64)
                .unwrap(),
            8.0_f64
        );
        assert_eq!(
            app.eval_expression_str("x(x + 1)", 3.0_f64, 0.0_f64)
                .unwrap(),
            12.0_f64
        );
        assert_eq!(
            app.eval_expression_str("2e3x", 3.0_f64, 0.0_f64).unwrap(),
            6000.0_f64
        );
        assert!(
            (app.eval_expression_str("sin(x)", 1.0_f64, 0.0_f64).unwrap() - 1.0_f64.sin()).abs()
                < 1e-9_f64
        );

        let branches =
            app.sample_function_branches("1 / x", -5.0_f64, 5.0_f64, -10.0_f64, 10.0_f64, 201);
        assert!(branches.len() >= 2);

        assert!(app.solve_structural_equilibrium_opt().is_none());
    }

    #[test]
    fn test_r_commands_and_quick_math() {
        let ws = Workspace::new_default();
        let mut app = EconCalcApp {
            workspace: ws,
            dark_mode: true,
            show_left_panel: true,
            show_right_panel: true,
            show_bottom_panel: true,
            maximize_graph: false,
            symbolic_card_collapsed: false,
            left_bottom_collapsed: false,
            env_card_collapsed: false,
            right_bottom_collapsed: false,
            symbolic_functions_open: true,
            symbolic_sliders_open: true,
            dataset_show_raw_csv: false,
            floating_datasets_window: false,
            floating_notes_window: false,
            floating_help_window: false,
            floating_env_window: false,
            floating_console_window: false,
            active_model: ActiveModelKind::CustomExpression,
            viewport_mode: ViewportMode::Plot2D,
            axis_zoom_mode: AxisZoomMode::BothXY,
            left_bottom_tab: LeftBottomTab::Files,
            file_filter: FileFilterKind::All,
            bottom_tab: BottomDockTab::QuickConsole,
            right_tab: RightDockTab::Datasets,
            selected_snippet: SnippetTemplate::CobbDouglas,
            report_format: ReportFormat::QuartoQmd,
            quick_command_input: String::new(),
            new_fn_sig_input: String::new(),
            new_fn_expr_input: String::new(),
            new_param_name_input: String::new(),
            new_param_val_input: String::new(),
            smart_math_shortcuts: true,
            editor_script: String::new(),
            console_logs: Vec::new(),
            eval_counter: 0,
            status_banner: String::new(),
            domain_min: 0.0_f64,
            domain_max: 10.0_f64,
            domain_y_min: 0.0_f64,
            domain_y_max: 10.0_f64,
            home_x_min: 0.0_f64,
            home_x_max: 10.0_f64,
            home_y_min: 0.0_f64,
            home_y_max: 10.0_f64,
            needs_plot_bounds_sync: false,
            pending_zoom_factor: None,
            lock_equal_aspect: false,
            show_plot_grid: true,
            show_plot_legend: true,
            curve_samples: 200,
            sim_grid_rows: 50,
            diff_step_h: 1e-5_f64,
            simpson_subintervals: 100,
            asymptote_jump_factor: 4.0_f64,
            probe_x: 1.0_f64,
            show_probe_marker: false,
            show_derivative_overlay: false,
            show_integral_shade: false,
            show_equilibrium_overlay: false,
            enable_equilibrium_solver: false,
            selected_3d_function: String::new(),
            yaw_deg: 38.0_f32,
            pitch_deg: 28.0_f32,
            surface_zoom: 1.0_f32,
            surface_resolution: 20,
            surface_wireframe_only: false,
            surface_colormap: SurfaceColorMap::OceanThermal,
            project_dir: PathBuf::from("."),
            project_files: Vec::new(),
            file_io_path_input: String::new(),
            new_dataset_name_input: String::new(),
            notes_preview_mode: false,
        };

        assert_eq!(
            app.eval_r_expression("2 + 3 * 4").unwrap(),
            RObject::scalar(14.0_f64)
        );

        app.execute_console_command("v <- c(2, 4, 6, 8)");
        assert_eq!(
            app.workspace.r_objects.get("v"),
            Some(&RObject::numeric_vec(vec![2.0_f64, 4.0_f64, 6.0_f64, 8.0_f64]))
        );

        assert_eq!(
            app.eval_r_expression("rep(3, 4)").unwrap(),
            RObject::numeric_vec(vec![3.0_f64, 3.0_f64, 3.0_f64, 3.0_f64])
        );

        assert_eq!(
            app.eval_r_expression("mean(v)").unwrap(),
            RObject::scalar(5.0_f64)
        );

        // End-to-end Desmos <-> R bridge + lm() integration inside EconCalcApp
        app.workspace.set_parameter("k", 2.0_f64);
        app.workspace.set_function("q(x)", "k * x + 1");
        let bridged = app.eval_r_expression("q(seq(1, 3, by = 1))").unwrap();
        assert_eq!(bridged, RObject::numeric_vec(vec![3.0_f64, 5.0_f64, 7.0_f64]));

        app.execute_console_command("df <- data.frame(x = 1:4, y = c(3, 5, 7, 9))");
        app.execute_console_command("fit <- lm(y ~ x, data = df)");
        let pred_in_desmos = app.eval_expression_str("fit(10)", 0.0_f64, 0.0_f64).unwrap();
        assert!((pred_in_desmos - 21.0_f64).abs() < 1e-8_f64);

        let intro = app.execute_console_command("intro-help");
        assert!(intro.contains("INTRO-HELP"));
        let general = app.execute_console_command("general-help");
        assert!(general.contains("GENERAL-HELP"));
    }
}
