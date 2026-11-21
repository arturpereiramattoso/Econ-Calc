pub use crate::collections::IndexMap;
use crate::exporter::{
    ActiveModelKind, ExportOutcome, ExportPayload, GridRow, ProjectFileEntry, ReportFormat,
    export_and_launch_positron, list_project_directory, load_text_file, save_text_file,
};
use crate::math::{
    self, CompiledExpr, CompiledWorkspace, EvalContext, EvalScratch, PlottableEntry, PoiKind,
};
pub use crate::r_engine::RObject;
use crate::r_engine::{DesmosBridge, EmbeddedREngine, REngineMathBridge};
use crate::ui::bento::{
    ThemePalette, apply_bento_theme, capsule_frame, draw_glowing_plot_line_with_opacity,
    draw_principal_axes_highlighted, header_icon_button, pill_button, render_bento_card_frame,
    render_card_sub_toolbar, render_card_tab_header, render_card_title_header,
};
use crate::ui::math_preview::{
    apply_smart_math_shortcuts, render_math_ident_inline, render_math_preview_strip_interactive,
};
pub use crate::ui::symbolic::{
    AxisDisplayConfig, FrozenBaseline, FrozenCurve, FunctionPlotRole, FunctionStyle, PinnedPoi,
    ShiftDragState, ShiftDragZone, SliderSpec, SymbolicCell, SymbolicCellKind, UndoHistory,
    WorkspaceSnapshot, render_curve_appearance_controls, render_gutter_curve_badge,
    render_slider_anim_controls,
};
use eframe::egui::{
    self, Color32, FontId, Margin, Pos2, RichText, Rounding, Sense, Stroke, Ui, Vec2,
};
use egui_extras as _;
use egui_plot::{
    GridInput, GridMark, Legend, Line, LineStyle, MarkerShape, Plot, PlotBounds, PlotPoint,
    PlotPoints, Points, Polygon, Text as PlotText, VLine,
};
use nalgebra::{Matrix2, Vector2};
use std::path::PathBuf;

// =============================================================================
// DATASETS (RObject in src/r_engine/object.rs; Symbolic cells/styles in src/ui/symbolic.rs)
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
    pub cells: Vec<SymbolicCell>,
    pub next_cell_id: u64,
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
            cells: Vec::new(),
            next_cell_id: 1,
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

        ws.set_function("f(x)", "(Y_bar / (A * x^alpha))^(1 / beta)");
        ws.set_function("g(x)", "(I - price_x * x) / price_y");

        ws.set_parameter("A", 2.0_f64);
        ws.set_parameter("alpha", 0.45_f64);
        ws.set_parameter("beta", 0.55_f64);
        ws.set_parameter("price_x", 4.0_f64);
        ws.set_parameter("price_y", 5.0_f64);
        ws.set_parameter("I", 100.0_f64);
        ws.set_parameter("Y_bar", 18.0_f64);

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
            Color32::from_rgb(184, 74, 57),
        ));

        ws
    }

    pub fn alloc_cell_id(&mut self) -> u64 {
        let id = self.next_cell_id;
        self.next_cell_id = self.next_cell_id.saturating_add(1);
        id
    }

    pub fn snapshot(&self) -> WorkspaceSnapshot {
        WorkspaceSnapshot {
            cells: self.cells.clone(),
            next_cell_id: self.next_cell_id,
            parameters: self.parameters.clone(),
            slider_specs: self.slider_specs.clone(),
            functions: self.functions.clone(),
            function_styles: self.function_styles.clone(),
        }
    }

    pub fn restore_snapshot(&mut self, snap: WorkspaceSnapshot) {
        self.cells = snap.cells;
        self.next_cell_id = snap.next_cell_id;
        self.parameters = snap.parameters;
        self.slider_specs = snap.slider_specs;
        self.functions = snap.functions;
        self.function_styles = snap.function_styles;
        for (k, &v) in &self.parameters {
            self.r_objects.insert(k.clone(), RObject::scalar(v));
        }
        self.sync_math_cache();
    }

    pub fn set_parameter(&mut self, name: impl Into<String>, val: f64) {
        let key = name.into();
        if self.slider_specs.get(&key).is_none() {
            let spec = SliderSpec::for_parameter(&key, val);
            self.slider_specs.insert(key.clone(), spec);
        }
        self.parameters.insert(key.clone(), val);
        self.r_objects.insert(key.clone(), RObject::scalar(val));
        let already_in_cells = self.cells.iter().any(|c| match &c.kind {
            SymbolicCellKind::Slider { param_name } => param_name == &key,
            SymbolicCellKind::Expression { raw, .. } => {
                matches!(
                    math::parse_statement(raw),
                    Ok(math::MathStatement::ParamAssign { name: ref n, .. }) if n == &key
                )
            }
            _ => false,
        });
        if !already_in_cells {
            let id = self.alloc_cell_id();
            self.cells.push(SymbolicCell::new_slider(id, key));
        }
    }

    /// Inserts a slider cell right after `after_cell_id` (used by `Add slider: [a] [All]`).
    pub fn insert_parameter_after(&mut self, name: impl Into<String>, val: f64, after_cell_id: Option<u64>) {
        let key = name.into();
        if self.slider_specs.get(&key).is_none() {
            let spec = SliderSpec::for_parameter(&key, val);
            self.slider_specs.insert(key.clone(), spec);
        }
        self.parameters.insert(key.clone(), val);
        self.r_objects.insert(key.clone(), RObject::scalar(val));
        let already = self.cells.iter().any(|c| matches!(&c.kind, SymbolicCellKind::Slider { param_name } if param_name == &key));
        if !already {
            let id = self.alloc_cell_id();
            let mut cell = SymbolicCell::new_slider(id, key);
            if let Some(after_id) = after_cell_id {
                if let Some(pos) = self.cells.iter().position(|c| c.id == after_id) {
                    cell.folder_id = self.cells[pos].folder_id;
                    self.cells.insert(pos + 1, cell);
                    return;
                }
            }
            self.cells.push(cell);
        }
    }

    pub fn set_function(&mut self, sig: impl Into<String>, expr: impl Into<String>) {
        let key = sig.into();
        let expr_s = expr.into();
        let idx = self.functions.len();
        if self.function_styles.get(&key).is_none() {
            self.function_styles
                .insert(key.clone(), FunctionStyle::default_for_index(idx));
        }
        self.functions.insert(key.clone(), expr_s.clone());

        let unified_raw = if key.starts_with("__cell_")
            || expr_s.contains('=')
            || expr_s.contains('<')
            || expr_s.contains('>')
            || expr_s.contains('≤')
            || expr_s.contains('≥')
            || expr_s.trim_start().starts_with('(')
        {
            expr_s.clone()
        } else {
            format!("{} = {}", key, expr_s)
        };

        if let Some(existing) = self.cells.iter_mut().find(|c| {
            matches!(&c.kind, SymbolicCellKind::Expression { bound_key: Some(bk), .. } if bk == &key)
        }) {
            if let SymbolicCellKind::Expression {
                raw,
                latex_draft,
                latex_error,
                ..
            } = &mut existing.kind
            {
                *raw = unified_raw;
                *latex_draft = math::to_latex_from_ast(raw);
                *latex_error = None;
            }
        } else {
            let id = self.alloc_cell_id();
            let cell = SymbolicCell::new_expression(id, unified_raw, Some(key));
            // Insert functions ahead of trailing slider cells so formulas stay grouped neatly at top
            let insert_pos = self
                .cells
                .iter()
                .position(|c| matches!(c.kind, SymbolicCellKind::Slider { .. }))
                .unwrap_or(self.cells.len());
            self.cells.insert(insert_pos, cell);
        }

        self.sync_math_cache();
    }

    pub fn is_folder_visible(&self, folder_id: Option<u64>) -> bool {
        let Some(fid) = folder_id else {
            return true;
        };
        for c in &self.cells {
            if c.id == fid {
                if let SymbolicCellKind::Folder { visible, .. } = c.kind {
                    return visible;
                }
            }
        }
        true
    }

    pub fn is_folder_collapsed(&self, folder_id: Option<u64>) -> bool {
        let Some(fid) = folder_id else {
            return false;
        };
        for c in &self.cells {
            if c.id == fid {
                if let SymbolicCellKind::Folder { collapsed, .. } = c.kind {
                    return collapsed;
                }
            }
        }
        false
    }

    /// Returns `true` if `fn_key` is visible both on its own style AND within its parent folder.
    pub fn key_is_effectively_visible(&self, fn_key: &str) -> bool {
        let own_vis = self
            .function_styles
            .get(fn_key)
            .map(|s| s.visible)
            .unwrap_or(true);
        if !own_vis {
            return false;
        }
        for c in &self.cells {
            if let SymbolicCellKind::Expression {
                bound_key: Some(bk),
                ..
            } = &c.kind
            {
                if bk == fn_key {
                    return self.is_folder_visible(c.folder_id);
                }
            }
        }
        true
    }

    pub fn remove_cell(&mut self, cell_id: u64) {
        let Some(pos) = self.cells.iter().position(|c| c.id == cell_id) else {
            return;
        };
        let removed = self.cells.remove(pos);
        match removed.kind {
            SymbolicCellKind::Expression { bound_key, .. } => {
                if let Some(bk) = bound_key {
                    self.functions.remove(&bk);
                    self.function_styles.remove(&bk);
                    self.sync_math_cache();
                }
            }
            SymbolicCellKind::Slider { param_name } => {
                self.parameters.remove(&param_name);
                self.slider_specs.remove(&param_name);
                self.r_objects.remove(&param_name);
            }
            SymbolicCellKind::Folder { .. } => {
                // Unparent children instead of deleting them unexpectedly
                for c in &mut self.cells {
                    if c.folder_id == Some(cell_id) {
                        c.folder_id = None;
                    }
                }
            }
            SymbolicCellKind::Note { .. } | SymbolicCellKind::Table { .. } => {}
        }
    }

    pub fn duplicate_cell(&mut self, cell_id: u64) -> Option<u64> {
        let pos = self.cells.iter().position(|c| c.id == cell_id)?;
        let new_id = self.alloc_cell_id();
        let mut cloned = self.cells[pos].clone();
        cloned.id = new_id;
        match &mut cloned.kind {
            SymbolicCellKind::Expression { raw, bound_key, .. } => {
                let new_key = format!("__cell_{}", new_id);
                if let Some(old_k) = bound_key.as_ref() {
                    if let Some(st) = self.function_styles.get(old_k).cloned() {
                        let mut new_st = st;
                        new_st.color = FunctionStyle::default_for_index(self.functions.len()).color;
                        self.function_styles.insert(new_key.clone(), new_st);
                    }
                }
                // Avoid overwriting same function name when duplicating `f(x) = ...`
                if let Ok(math::MathStatement::FunctionDef { name, params, body }) =
                    math::parse_statement(raw)
                {
                    let alt_name = format!("{}_2", name);
                    let alt_sig = format!("{}({})", alt_name, params.join(", "));
                    let body_s = body.to_symbolic();
                    *raw = format!("{} = {}", alt_sig, body_s);
                    *bound_key = Some(alt_sig.clone());
                    let style = FunctionStyle::default_for_index(self.functions.len());
                    self.function_styles.insert(alt_sig.clone(), style);
                    self.functions.insert(alt_sig, body_s);
                } else {
                    *bound_key = Some(new_key.clone());
                    self.functions.insert(new_key, raw.clone());
                }
                self.sync_math_cache();
            }
            SymbolicCellKind::Slider { param_name } => {
                let alt = format!("{}_2", param_name);
                let val = self.parameters.get(param_name).copied().unwrap_or(1.0_f64);
                let spec = self
                    .slider_specs
                    .get(param_name)
                    .cloned()
                    .unwrap_or_else(|| SliderSpec::for_parameter(&alt, val));
                self.parameters.insert(alt.clone(), val);
                self.slider_specs.insert(alt.clone(), spec);
                self.r_objects.insert(alt.clone(), RObject::scalar(val));
                *param_name = alt;
            }
            SymbolicCellKind::Folder { title, .. } => {
                title.push_str(" (Copy)");
            }
            SymbolicCellKind::Note { .. } | SymbolicCellKind::Table { .. } => {}
        }
        self.cells.insert(pos + 1, cloned);
        Some(new_id)
    }

    pub fn sync_math_cache(&mut self) {
        self.compiled_math = CompiledWorkspace::build(&self.functions);
    }

    pub fn clear(&mut self) {
        self.cells.clear();
        self.next_cell_id = 1;
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
            SurfaceColorMap::OceanThermal => "Lapis & Ochre",
            SurfaceColorMap::Viridis => "Malachite & Gold",
            SurfaceColorMap::CoolWarm => "Lapis & Cinnabar",
        }
    }

    fn map_normalized(self, t: f32) -> (u8, u8, u8) {
        let t = t.clamp(0.0_f32, 1.0_f32);
        match self {
            SurfaceColorMap::OceanThermal => {
                // Deep Lapis Lazuli (#2E527E) -> Warm Limestone -> Ochre Gold (#C69C4E)
                let r = (46.0_f32 + 152.0_f32 * t) as u8;
                let g = (82.0_f32 + 78.0_f32 * t + 24.0_f32 * (1.0_f32 - (2.0_f32 * t - 1.0_f32).abs())) as u8;
                let b = (126.0_f32 - 48.0_f32 * t) as u8;
                (r, g, b)
            }
            SurfaceColorMap::Viridis => {
                // Deep Monastic Umber/Malachite (#264E3A) -> Verdigris -> Ochre Gold (#CEA24E)
                let r = (38.0_f32 + 168.0_f32 * t * t) as u8;
                let g = (78.0_f32 + 86.0_f32 * t) as u8;
                let b = (58.0_f32 + 30.0_f32 * (1.0_f32 - (2.0_f32 * t - 1.0_f32).abs())
                    - 10.0_f32 * t)
                    .clamp(28.0_f32, 220.0_f32) as u8;
                (r, g, b)
            }
            SurfaceColorMap::CoolWarm => {
                // Lapis Lazuli (#34567E) -> Parchment -> Cinnabar Rubric (#B84A39)
                let r = (52.0_f32 + 132.0_f32 * t) as u8;
                let g = (86.0_f32 - 12.0_f32 * t + 42.0_f32 * (1.0_f32 - (2.0_f32 * t - 1.0_f32).abs())) as u8;
                let b = (126.0_f32 - 69.0_f32 * t) as u8;
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
    dataset_show_raw_csv: bool,

    // Unified Symbolic List, Undo/Redo, Comparative Statics & Graph Interaction
    undo_history: UndoHistory,
    symbolic_list_edit_mode: bool,
    selected_cell_id: Option<u64>,
    request_focus_cell_id: Option<u64>,
    frozen_baseline: Option<FrozenBaseline>,
    pinned_pois: Vec<PinnedPoi>,
    shift_drag: ShiftDragState,
    axis_config: AxisDisplayConfig,
    last_frame_time: Option<f64>,

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
            dataset_show_raw_csv: false,
            undo_history: UndoHistory::default(),
            symbolic_list_edit_mode: false,
            selected_cell_id: None,
            request_focus_cell_id: None,
            frozen_baseline: None,
            pinned_pois: Vec::new(),
            shift_drag: ShiftDragState::default(),
            axis_config: AxisDisplayConfig::default(),
            last_frame_time: None,
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
            smart_math_shortcuts: true,
            editor_script: initial_script,
            console_logs: Vec::new(),
            eval_counter: 0,
            status_banner: "Oikonomos (Οἰκονόμος) — Ready. Type `intro-help` or `general-help` in Console."
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
        self.push_undo_snapshot();
        self.workspace.clear();
        self.editor_script.clear();
        self.frozen_baseline = None;
        self.pinned_pois.clear();
        self.selected_cell_id = None;
        self.show_equilibrium_overlay = false;
        self.enable_equilibrium_solver = false;
        self.status_banner = "Workspace cleared.".to_string();
    }

    fn load_snippet_template(&mut self, snip: SnippetTemplate) {
        self.push_undo_snapshot();
        self.selected_snippet = snip;
        self.active_model = snip.model_kind();
        self.editor_script = snip.script_text().to_string();
        let (x0, x1, y0, y1) = snip.default_domains();
        self.home_x_min = x0;
        self.home_x_max = x1;
        self.home_y_min = y0;
        self.home_y_max = y1;
        self.reset_graph_to_home();
        self.workspace.cells.clear();
        self.workspace.next_cell_id = 1;
        self.workspace.parameters.clear();
        self.workspace.slider_specs.clear();
        self.workspace.functions.clear();
        self.workspace.function_styles.clear();
        self.workspace.sync_math_cache();
        self.frozen_baseline = None;
        self.pinned_pois.clear();
        self.evaluate_editor_script();
        if snip == SnippetTemplate::SaddleSurface3D {
            self.viewport_mode = ViewportMode::Surface3D;
        } else {
            self.viewport_mode = ViewportMode::Plot2D;
        }
        self.status_banner = format!("Loaded model: {}", snip.label());
    }

    // =========================================================================
    // UNDO/REDO, SLIDER ANIMATION, UNIFIED CELL SYNC, BASELINE FREEZE & TABLES
    // =========================================================================

    fn push_undo_snapshot(&mut self) {
        let snap = self.workspace.snapshot();
        self.undo_history.push(snap);
    }

    fn perform_undo(&mut self) {
        let cur = self.workspace.snapshot();
        if let Some(prev) = self.undo_history.undo(cur) {
            self.workspace.restore_snapshot(prev);
            self.status_banner = "Undo symbolic edit.".to_string();
        }
    }

    fn perform_redo(&mut self) {
        let cur = self.workspace.snapshot();
        if let Some(next) = self.undo_history.redo(cur) {
            self.workspace.restore_snapshot(next);
            self.status_banner = "Redo symbolic edit.".to_string();
        }
    }

    fn step_slider_animations(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        let dt = self
            .last_frame_time
            .map(|prev| (now - prev).clamp(0.0_f64, 0.08_f64))
            .unwrap_or(1.0_f64 / 60.0_f64);
        self.last_frame_time = Some(now);

        let mut any_playing = false;
        let param_keys: Vec<String> = self.workspace.slider_specs.keys().cloned().collect();
        for name in param_keys {
            if let Some(spec) = self.workspace.slider_specs.get_mut(&name) {
                if spec.playing {
                    any_playing = true;
                    let cur = self
                        .workspace
                        .parameters
                        .get(&name)
                        .copied()
                        .unwrap_or(spec.min);
                    let next = spec.advance_value(cur, dt);
                    self.workspace.parameters.insert(name.clone(), next);
                    self.workspace
                        .r_objects
                        .insert(name.clone(), RObject::scalar(next));
                }
            }
        }
        if any_playing {
            ctx.request_repaint();
        }
    }

    fn focus_first_quadrant(&mut self) {
        let x_span = (self.domain_max - self.domain_min).abs().max(10.0_f64);
        let y_span = (self.domain_y_max - self.domain_y_min).abs().max(10.0_f64);
        self.domain_min = 0.0_f64;
        self.domain_max = x_span.max(12.0_f64);
        self.domain_y_min = 0.0_f64;
        self.domain_y_max = y_span.max(12.0_f64);
        self.needs_plot_bounds_sync = true;
        self.status_banner = "Focused 1st Economic Quadrant (x ≥ 0, y ≥ 0).".to_string();
    }

    /// Evaluates a per-function domain/range bound expression (e.g. `"-2*pi"`, `"I / price_x"`).
    fn eval_bound_expr_opt(&self, raw: &str) -> Option<f64> {
        let t = raw.trim();
        if t.is_empty() {
            return None;
        }
        let ext = REngineMathBridge {
            r_objects: &self.workspace.r_objects,
        };
        let ctx = self.build_eval_context(&ext);
        let ast = math::parse_expr_with_known_fns(t, &self.workspace.compiled_math.known_fn_names)
            .ok()?;
        let compiled = CompiledExpr::compile(&ast);
        let mut scratch = EvalScratch::new();
        compiled
            .eval(&ctx, &[], &mut scratch)
            .ok()
            .filter(|v| v.is_finite())
    }

    /// Resolves effective `(x_min, x_max)`, `(y_min, y_max)`, or `(t_min, t_max)` for a function style.
    fn resolve_style_x_domain(&self, style: &FunctionStyle) -> Option<(f64, f64)> {
        let lo = self.eval_bound_expr_opt(&style.custom_x_min_expr);
        let hi = self.eval_bound_expr_opt(&style.custom_x_max_expr);
        match (lo, hi) {
            (Some(a), Some(b)) => Some((a, b)),
            (Some(a), None) => Some((a, self.domain_max)),
            (None, Some(b)) => Some((self.domain_min, b)),
            (None, None) => style.custom_domain,
        }
    }

    fn resolve_style_y_range(&self, style: &FunctionStyle) -> Option<(f64, f64)> {
        let lo = self.eval_bound_expr_opt(&style.custom_y_min_expr);
        let hi = self.eval_bound_expr_opt(&style.custom_y_max_expr);
        match (lo, hi) {
            (Some(a), Some(b)) => Some((a, b)),
            (Some(a), None) => Some((a, f64::INFINITY)),
            (None, Some(b)) => Some((f64::NEG_INFINITY, b)),
            (None, None) => style.custom_y_range,
        }
    }

    fn resolve_style_param_domain(&self, style: &FunctionStyle) -> (f64, f64) {
        let lo = self.eval_bound_expr_opt(&style.custom_t_min_expr);
        let hi = self.eval_bound_expr_opt(&style.custom_t_max_expr);
        match (lo, hi) {
            (Some(a), Some(b)) if b > a => (a, b),
            _ => style
                .custom_param_domain
                .unwrap_or((0.0_f64, std::f64::consts::TAU)),
        }
    }

    /// Synchronizes an edited unified `SymbolicCellKind::Expression` into `workspace.functions`,
    /// `workspace.parameters`, `workspace.function_styles`, and `workspace.compiled_math`.
    fn sync_expression_cell(&mut self, cell_idx: usize) {
        let Some(cell) = self.workspace.cells.get(cell_idx).cloned() else {
            return;
        };
        let SymbolicCellKind::Expression {
            raw,
            bound_key: old_key,
            ..
        } = cell.kind
        else {
            return;
        };

        let trimmed = raw.trim();
        if trimmed.is_empty() {
            if let Some(old_k) = old_key {
                self.workspace.functions.remove(&old_k);
                self.workspace.sync_math_cache();
            }
            if let Some(c) = self.workspace.cells.get_mut(cell_idx) {
                if let SymbolicCellKind::Expression { bound_key, .. } = &mut c.kind {
                    *bound_key = None;
                }
            }
            return;
        }

        // Parse statement to classify FunctionDef vs ParamAssign vs general equation/expression
        let known = self.workspace.compiled_math.known_fn_names.clone();
        let parsed = math::parse_statement_with_known_fns(trimmed, Some(&known));

        match parsed {
            Ok(math::MathStatement::FunctionDef { name, params, .. }) => {
                let new_key = format!("{}({})", name, params.join(", "));
                // Extract RHS slice after first `=` so user formatting/restrictions stay intact
                let body_raw = trimmed
                    .split_once('=')
                    .map(|(_, rhs)| rhs.trim().to_string())
                    .unwrap_or_else(|| trimmed.to_string());

                let style = old_key
                    .as_ref()
                    .and_then(|ok| self.workspace.function_styles.get(ok).cloned())
                    .or_else(|| self.workspace.function_styles.get(&new_key).cloned())
                    .unwrap_or_else(|| {
                        FunctionStyle::default_for_index(self.workspace.functions.len())
                    });

                if let Some(ref ok) = old_key {
                    if ok != &new_key {
                        self.workspace.functions.remove(ok);
                        self.workspace.function_styles.remove(ok);
                    }
                }
                self.workspace
                    .function_styles
                    .insert(new_key.clone(), style);
                self.workspace.functions.insert(new_key.clone(), body_raw);
                self.workspace.sync_math_cache();

                if let Some(c) = self.workspace.cells.get_mut(cell_idx) {
                    if let SymbolicCellKind::Expression { bound_key, .. } = &mut c.kind {
                        *bound_key = Some(new_key);
                    }
                }
            }
            Ok(math::MathStatement::ParamAssign { name, value }) => {
                if let Some(ref ok) = old_key {
                    self.workspace.functions.remove(ok);
                    self.workspace.function_styles.remove(ok);
                    self.workspace.sync_math_cache();
                }
                let ext = REngineMathBridge {
                    r_objects: &self.workspace.r_objects,
                };
                let ctx = self.build_eval_context(&ext);
                let compiled = CompiledExpr::compile(&value);
                let mut scratch = EvalScratch::new();
                if let Ok(val) = compiled.eval(&ctx, &[], &mut scratch) {
                    if val.is_finite() {
                        if self.workspace.slider_specs.get(&name).is_none() {
                            self.workspace
                                .slider_specs
                                .insert(name.clone(), SliderSpec::for_parameter(&name, val));
                        }
                        self.workspace.parameters.insert(name.clone(), val);
                        self.workspace
                            .r_objects
                            .insert(name.clone(), RObject::scalar(val));
                    }
                }
                if let Some(c) = self.workspace.cells.get_mut(cell_idx) {
                    if let SymbolicCellKind::Expression { bound_key, .. } = &mut c.kind {
                        *bound_key = None;
                    }
                }
            }
            _ => {
                let cell_key = format!("__cell_{}", cell.id);
                let style = old_key
                    .as_ref()
                    .and_then(|ok| self.workspace.function_styles.get(ok).cloned())
                    .or_else(|| self.workspace.function_styles.get(&cell_key).cloned())
                    .unwrap_or_else(|| {
                        FunctionStyle::default_for_index(self.workspace.functions.len())
                    });
                if let Some(ref ok) = old_key {
                    if ok != &cell_key {
                        self.workspace.functions.remove(ok);
                        self.workspace.function_styles.remove(ok);
                    }
                }
                self.workspace
                    .function_styles
                    .insert(cell_key.clone(), style);
                self.workspace
                    .functions
                    .insert(cell_key.clone(), trimmed.to_string());
                self.workspace.sync_math_cache();

                if let Some(c) = self.workspace.cells.get_mut(cell_idx) {
                    if let SymbolicCellKind::Expression { bound_key, .. } = &mut c.kind {
                        *bound_key = Some(cell_key);
                    }
                }
            }
        }
    }

    /// Formats a scalar value with thousands spacing (e.g. `15 625`) and optional rational fraction (`⅓`).
    fn format_scalar_result(val: f64, as_fraction: bool) -> (String, Option<(i64, i64)>) {
        if !val.is_finite() {
            return ("undefined".to_string(), None);
        }
        let frac = math::float_to_fraction(val, 10_000, 1e-8_f64);
        if as_fraction {
            if let Some((p, q)) = frac {
                return (format!("{}/{}", p, q), frac);
            }
        }
        if val.fract().abs() < 1e-9_f64 && val.abs() < 1e12_f64 {
            let n = val.round() as i64;
            return (Self::format_int_spaced(n), frac);
        }
        let s = format!("{:.6}", val);
        let trimmed = s.trim_end_matches('0').trim_end_matches('.').to_string();
        (trimmed, frac)
    }

    fn format_int_spaced(n: i64) -> String {
        let neg = n < 0;
        let digits = n.unsigned_abs().to_string();
        if digits.len() <= 4 {
            return if neg {
                format!("-{}", digits)
            } else {
                digits
            };
        }
        let mut out = String::with_capacity(digits.len() + digits.len() / 3 + 1);
        if neg {
            out.push('-');
        }
        for (i, ch) in digits.chars().enumerate() {
            let rem = digits.len() - i;
            if i > 0 && rem.is_multiple_of(3) {
                out.push(' ');
            }
            out.push(ch);
        }
        out
    }

    /// Evaluates vectorized R-bridged expressions inside Symbolic cells (e.g. `f(c(1, 2, 3, 4))` -> `[1, 4, 9, 16]`).
    fn try_eval_vectorized_r_cell(&self, raw: &str) -> Option<String> {
        let t = raw.trim();
        if t.is_empty() || t.contains('=') || t.contains('<') || t.contains('>') {
            return None;
        }
        // Trigger R bridge when expression contains `c(`, `seq(`, `rep(`, `:`, `$`, or references an R vector
        let likely_vec = t.contains("c(")
            || t.contains("seq(")
            || t.contains("rep(")
            || t.contains('$')
            || self.workspace.r_objects.iter().any(|(k, v)| {
                matches!(v, RObject::Numeric(vec) if vec.len() > 1) && t.contains(k.as_str())
            });
        if !likely_vec {
            return None;
        }
        let bridge = WorkspaceDesmosBridge {
            parameters: &self.workspace.parameters,
            compiled_math: &self.workspace.compiled_math,
            diff_h: self.diff_step_h,
            integral_steps: self.simpson_subintervals,
        };
        let val = EmbeddedREngine::eval_readonly(&self.workspace.r_objects, t, Some(&bridge)).ok()?;
        match val {
            RObject::Numeric(ref nums) if nums.len() > 1 => {
                let items: Vec<String> = nums
                    .iter()
                    .take(12)
                    .map(|opt| match opt {
                        Some(v) if v.is_finite() => Self::format_scalar_result(*v, false).0,
                        _ => "NA".to_string(),
                    })
                    .collect();
                let suffix = if nums.len() > 12 { ", …" } else { "" };
                Some(format!("[{}{}]", items.join(", "), suffix))
            }
            RObject::Numeric(ref nums) if nums.len() == 1 => {
                nums[0].map(|v| Self::format_scalar_result(v, false).0)
            }
            _ => None,
        }
    }

    /// Freezes all currently visible 2D curves + current equilibrium into `self.frozen_baseline`
    /// for side-by-side comparative statics (`📌`).
    fn freeze_current_baseline(&mut self) {
        let ext = REngineMathBridge {
            r_objects: &self.workspace.r_objects,
        };
        let eval_ctx = self.build_eval_context(&ext);
        let mut frozen_curves = Vec::new();

        for (idx, (sig, expr)) in self.workspace.functions.iter().enumerate() {
            if !self.workspace.key_is_effectively_visible(sig) {
                continue;
            }
            let style = self
                .workspace
                .function_styles
                .get(sig)
                .cloned()
                .unwrap_or_else(|| FunctionStyle::default_for_index(idx));
            if style.role == FunctionPlotRole::Surface3DOnly {
                continue;
            }
            let Ok(plottable) = self.workspace.compiled_math.classify_entry(sig, expr) else {
                continue;
            };
            let (s_min, s_max) = self
                .resolve_style_x_domain(&style)
                .unwrap_or((self.domain_min, self.domain_max));
            let branches = match plottable {
                PlottableEntry::Explicit1D { compiled }
                | PlottableEntry::TangentLine {
                    line_compiled: compiled,
                    ..
                } => math::sample_1d_branches(
                    &compiled,
                    &eval_ctx,
                    s_min.max(self.domain_min),
                    s_max.min(self.domain_max),
                    self.domain_y_min,
                    self.domain_y_max,
                    self.curve_samples.min(300),
                    self.asymptote_jump_factor,
                ),
                PlottableEntry::ImplicitEquation { diff_compiled } => {
                    math::sample_implicit_contours(
                        &diff_compiled,
                        &eval_ctx,
                        self.domain_min,
                        self.domain_max,
                        self.domain_y_min,
                        self.domain_y_max,
                        64,
                        64,
                    )
                }
                PlottableEntry::Parametric {
                    param,
                    x_compiled,
                    y_compiled,
                } => {
                    let (t0, t1) = self.resolve_style_param_domain(&style);
                    math::sample_parametric_branches(
                        &x_compiled,
                        &y_compiled,
                        &param,
                        &eval_ctx,
                        t0,
                        t1,
                        self.curve_samples.min(300),
                    )
                }
                PlottableEntry::Polar { param, r_compiled } => {
                    let (t0, t1) = self.resolve_style_param_domain(&style);
                    math::sample_polar_branches(
                        &r_compiled,
                        &param,
                        &eval_ctx,
                        t0,
                        t1,
                        self.curve_samples.min(300),
                    )
                }
                _ => Vec::new(),
            };
            if !branches.is_empty() {
                let lbl = if sig.starts_with("__cell_") {
                    format!("{} (baseline)", expr)
                } else {
                    format!("{}₀", sig)
                };
                frozen_curves.push(FrozenCurve {
                    label: lbl,
                    color: style.color,
                    branches,
                });
            }
        }

        let eq = self.find_primary_secondary_intersection();
        self.frozen_baseline = Some(FrozenBaseline {
            label: "Baseline Snapshot".to_string(),
            curves: frozen_curves,
            equilibrium: eq,
        });
        self.status_banner =
            "† Frozen comparative-statics baseline — move sliders to compare Δx*, Δy*."
                .to_string();
    }

    /// Creates an interactive `SymbolicCellKind::Table` for `fn_sig` synced to Data Viewer & R `data.frame`.
    fn create_table_for_function(&mut self, fn_sig: &str, after_cell_id: Option<u64>) {
        self.push_undo_snapshot();
        let x0 = self.domain_min.max(0.0_f64).min(self.domain_max - 1.0_f64);
        let step = ((self.domain_max - x0) / 5.0_f64).max(1.0_f64).round().max(1.0_f64);
        let x_values: Vec<f64> = (0..6).map(|i| x0 + (i as f64) * step).collect();

        let base_name = fn_sig
            .split('(')
            .next()
            .unwrap_or("fn")
            .trim()
            .replace(|c: char| !c.is_ascii_alphanumeric(), "_");
        let ds_name = format!("{}_table", base_name.trim_matches('_'));

        let id = self.workspace.alloc_cell_id();
        let mut cell = SymbolicCell::new_table(id, fn_sig.to_string(), x_values.clone(), ds_name.clone());
        if let Some(after_id) = after_cell_id {
            if let Some(pos) = self.workspace.cells.iter().position(|c| c.id == after_id) {
                cell.folder_id = self.workspace.cells[pos].folder_id;
                self.workspace.cells.insert(pos + 1, cell);
            } else {
                self.workspace.cells.push(cell);
            }
        } else {
            self.workspace.cells.push(cell);
        }

        self.sync_table_cell_to_data_and_r(fn_sig, &x_values, &ds_name);
        self.status_banner = format!(
            "Created synced table `{}` for {} (Data Viewer + R data.frame).",
            ds_name, fn_sig
        );
    }

    /// Synchronizes a `SymbolicCellKind::Table` into `workspace.datasets` (Data Viewer)
    /// and `workspace.r_objects` (R `data.frame` with columns `x`, `y`, `dy_dx`).
    fn sync_table_cell_to_data_and_r(&mut self, fn_sig: &str, x_values: &[f64], dataset_name: &str) {
        let expr_opt = self.workspace.functions.get(fn_sig).cloned();
        let Some(expr) = expr_opt else {
            return;
        };

        let mut pts: Vec<[f64; 2]> = Vec::new();
        let mut r_x: Vec<Option<f64>> = Vec::new();
        let mut r_y: Vec<Option<f64>> = Vec::new();
        let mut r_dy: Vec<Option<f64>> = Vec::new();

        for &xv in x_values {
            let yv = self
                .eval_expression_str(&expr, xv, 1.0_f64)
                .ok()
                .filter(|v| v.is_finite());
            let dyv = self
                .numerical_derivative(&expr, xv)
                .ok()
                .filter(|v| v.is_finite());
            r_x.push(Some(xv));
            r_y.push(yv);
            r_dy.push(dyv);
            if let Some(y) = yv {
                pts.push([xv, y]);
            }
        }

        // 1. Sync to Data Viewer (`workspace.datasets`)
        if let Some(ds) = self
            .workspace
            .datasets
            .iter_mut()
            .find(|d| d.name == dataset_name)
        {
            ds.points = pts;
            ds.sync_raw_csv_from_points();
        } else {
            let mut ds = DataTable::from_csv_str(
                dataset_name.to_string(),
                "x,y\n",
                Color32::from_rgb(72, 118, 168),
            );
            ds.points = pts;
            ds.sync_raw_csv_from_points();
            ds.show_on_plot = true;
            self.workspace.datasets.push(ds);
            self.workspace.active_dataset_idx = self.workspace.datasets.len() - 1;
        }

        // 2. Sync to R Environment as a real `data.frame(x, y, dy_dx)`
        let mut cols = IndexMap::new();
        cols.insert("x".to_string(), RObject::Numeric(r_x));
        cols.insert("y".to_string(), RObject::Numeric(r_y));
        cols.insert("dy_dx".to_string(), RObject::Numeric(r_dy));
        self.workspace.r_objects.insert(
            dataset_name.to_string(),
            RObject::DataFrame {
                columns: cols,
                row_count: x_values.len(),
            },
        );
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
            Color32::from_rgb(76, 142, 104),
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
            color: Color32::from_rgb(76, 142, 104),
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
            .replace("price_x", "P_x")
            .replace("price_y", "P_y")
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
=== INTRO-HELP: Oikonomos (Οἰκονόμος) Quickstart Guide ===
1. LAYOUT & PANELS:
   - Top Bar: Toggle Left Sidebar, Right Dock, Bottom Console, Maximize Graph, or Vigília (Dark) / Pergaminho (Light) Theme.
   - Left Sidebar:
     • Formulas & Sliders: Add/edit functions f(x), customize line style/color/role, and set slider min/max/step limits.
     • R Objects: Inspect scalars & vectors created in the Console/Script.
     • Models Library: Load ready-made economic & mathematical templates.
     • Project & Files: Browse project directory, open .R/.qmd/.Rmd/.md/.csv files.
2. GRAPH VIEWPORT (2D & 3D):
   - Click `⌂ Home` to restore your configured Home X & Y bounds.
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
        ui.set_min_width(250.0_f32);
        ui.label(
            RichText::new("Axes & Domain Configuration")
                .strong()
                .color(pal.accent_blue),
        );
        ui.separator();

        let mut bounds_changed = false;
        // X-Axis row
        ui.horizontal(|ui| {
            ui.label(RichText::new("X-Axis").strong().color(pal.accent_blue));
            ui.add(
                egui::TextEdit::singleline(&mut self.axis_config.x_label)
                    .desired_width(110.0_f32)
                    .hint_text("Label (e.g. Labor L)"),
            );
        });
        ui.horizontal(|ui| {
            bounds_changed |= ui
                .add(egui::DragValue::new(&mut self.domain_min).speed(0.2_f64).prefix("Min: "))
                .changed();
            bounds_changed |= ui
                .add(egui::DragValue::new(&mut self.domain_max).speed(0.2_f64).prefix("Max: "))
                .changed();
            let mut x_step_val = self.axis_config.x_step.unwrap_or(0.0_f64);
            if ui
                .add(
                    egui::DragValue::new(&mut x_step_val)
                        .speed(0.1_f64)
                        .range(0.0_f64..=1000.0_f64)
                        .prefix("Step: "),
                )
                .on_hover_text("0 = automatic grid step")
                .changed()
            {
                self.axis_config.x_step = (x_step_val > 1e-6_f64).then_some(x_step_val);
            }
        });

        ui.add_space(4.0_f32);
        // Y-Axis row
        ui.horizontal(|ui| {
            ui.label(RichText::new("Y-Axis").strong().color(pal.accent_purple));
            ui.add(
                egui::TextEdit::singleline(&mut self.axis_config.y_label)
                    .desired_width(110.0_f32)
                    .hint_text("Label (e.g. Capital K)"),
            );
        });
        ui.horizontal(|ui| {
            bounds_changed |= ui
                .add(egui::DragValue::new(&mut self.domain_y_min).speed(0.2_f64).prefix("Min: "))
                .changed();
            bounds_changed |= ui
                .add(egui::DragValue::new(&mut self.domain_y_max).speed(0.2_f64).prefix("Max: "))
                .changed();
            let mut y_step_val = self.axis_config.y_step.unwrap_or(0.0_f64);
            if ui
                .add(
                    egui::DragValue::new(&mut y_step_val)
                        .speed(0.1_f64)
                        .range(0.0_f64..=1000.0_f64)
                        .prefix("Step: "),
                )
                .on_hover_text("0 = automatic grid step")
                .changed()
            {
                self.axis_config.y_step = (y_step_val > 1e-6_f64).then_some(y_step_val);
            }
        });

        if bounds_changed {
            if self.domain_max <= self.domain_min + 1e-3_f64 {
                self.domain_max = self.domain_min + 1.0_f64;
            }
            if self.domain_y_max <= self.domain_y_min + 1e-3_f64 {
                self.domain_y_max = self.domain_y_min + 1.0_f64;
            }
            self.needs_plot_bounds_sync = true;
        }

        ui.add_space(4.0_f32);
        ui.horizontal_wrapped(|ui| {
            if ui
                .add(pill_button("Focus 1st Quadrant (x ≥ 0, y ≥ 0)", false, pal))
                .clicked()
            {
                self.focus_first_quadrant();
            }
            if ui
                .add(pill_button("[-10, 10]", false, pal))
                .clicked()
            {
                self.domain_min = -10.0_f64;
                self.domain_max = 10.0_f64;
                self.domain_y_min = -10.0_f64;
                self.domain_y_max = 10.0_f64;
                self.needs_plot_bounds_sync = true;
            }
        });

        ui.separator();
        ui.label(
            RichText::new("Graph & Calculus Overlays")
                .strong()
                .color(pal.accent_purple),
        );
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
        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add(pill_button("† Freeze Baseline", self.frozen_baseline.is_some(), pal))
                .clicked()
            {
                self.freeze_current_baseline();
            }
            if self.frozen_baseline.is_some()
                && ui.add(pill_button("Clear Baseline", false, pal)).clicked()
            {
                self.frozen_baseline = None;
                self.status_banner = "Cleared frozen comparative-statics baseline.".to_string();
            }
            if !self.pinned_pois.is_empty()
                && ui.add(pill_button("Clear Pins", false, pal)).clicked()
            {
                self.pinned_pois.clear();
            }
        });
        ui.label(
            RichText::new("Tip: Shift + Drag on X axis, Y axis, or canvas to scale axes.")
                .font(FontId::proportional(10.5_f32))
                .color(pal.text_muted),
        );
    }

    fn render_top_header(&mut self, ui: &mut Ui) {
        let pal = self.palette();

        // ---- Tier 1: Menu Bar ----
        egui::menu::bar(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 6.0_f32;
            ui.label(
                RichText::new("◈")
                    .font(FontId::proportional(14.0_f32))
                    .color(pal.accent_amber),
            );
            ui.label(
                RichText::new("OIKONOMOS")
                    .font(FontId::proportional(13.2_f32))
                    .strong()
                    .color(pal.text_primary),
            );
            ui.label(
                RichText::new("Οἰκονόμος")
                    .font(FontId::proportional(11.8_f32))
                    .color(pal.text_muted),
            );
            ui.add_space(4.0_f32);

            ui.menu_button(RichText::new("File ⌄").color(pal.text_secondary), |ui| {
                if ui.button("§ New / Clear Workspace").clicked() {
                    self.clear_entire_workspace();
                    ui.close_menu();
                }
                if ui.button("⊞ Open File from Project").clicked() {
                    self.maximize_graph = false;
                    self.show_left_panel = true;
                    self.left_bottom_collapsed = false;
                    self.left_bottom_tab = LeftBottomTab::Files;
                    self.refresh_project_files();
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("‡ Save R Script (exports/workspace_script.R)").clicked() {
                    self.save_editor_script_default();
                    ui.close_menu();
                }
                if ui.button("¶ Save Notes (.qmd / .Rmd / .md)").clicked() {
                    self.save_notes_default();
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .button(
                        RichText::new("⊳ Export Bundle & Open Positron").color(pal.accent_green),
                    )
                    .clicked()
                {
                    self.trigger_export_pipeline();
                    ui.close_menu();
                }
            });

            ui.menu_button(RichText::new("Edit ⌄").color(pal.text_secondary), |ui| {
                if ui.button("⎘ Copy Canonical LaTeX / TikZ").clicked() {
                    let latex = self.generate_latex_derivation();
                    ui.ctx().copy_text(latex);
                    self.status_banner = "Copied canonical LaTeX/TikZ to clipboard.".to_string();
                    ui.close_menu();
                }
                if ui.button("⎘ Copy R / ggplot2 Script").clicked() {
                    let r_src = self.generate_r_script_preview();
                    ui.ctx().copy_text(r_src);
                    self.status_banner = "Copied R/ggplot2 script to clipboard.".to_string();
                    ui.close_menu();
                }
                if ui.button("⌫ Clear Console Logs").clicked() {
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
                ui.checkbox(&mut self.maximize_graph, "⊞ Maximize Graph Viewport");
                ui.separator();
                if ui
                    .selectable_label(self.viewport_mode == ViewportMode::Plot2D, "◿ 2D Graph")
                    .clicked()
                {
                    self.viewport_mode = ViewportMode::Plot2D;
                    ui.close_menu();
                }
                if ui
                    .selectable_label(self.viewport_mode == ViewportMode::Surface3D, "◈ 3D Surface")
                    .clicked()
                {
                    self.viewport_mode = ViewportMode::Surface3D;
                    ui.close_menu();
                }
                ui.separator();
                ui.checkbox(&mut self.floating_datasets_window, "▢ Pop-Out Datasets");
                ui.checkbox(&mut self.floating_notes_window, "▢ Pop-Out Notes");
                ui.checkbox(&mut self.floating_help_window, "▢ Pop-Out Help");
                ui.checkbox(&mut self.floating_env_window, "▢ Pop-Out Environment");
                ui.checkbox(&mut self.floating_console_window, "▢ Pop-Out Console");
                ui.separator();
                let theme_txt = if self.dark_mode {
                    "☼ Switch to Pergaminho (Light)"
                } else {
                    "☾ Switch to Vigília (Dark)"
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
                if ui.button("⊞ Generate Dataset from f(x)").clicked() {
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
                    .add(pill_button("▣", false, pal))
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
                let theme_icon = if self.dark_mode { "☾" } else { "☼" };
                if ui
                    .add(pill_button(theme_icon, false, pal))
                    .on_hover_text("Toggle Vigília (Dark) / Pergaminho (Light) Theme")
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
                .add(pill_button("§", false, pal))
                .on_hover_text("New / Clear Workspace")
                .clicked()
            {
                self.clear_entire_workspace();
            }
            if ui
                .add(pill_button("▣", false, pal))
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
                .add(pill_button("‡", false, pal))
                .on_hover_text("Export Bundle & Open Positron")
                .clicked()
            {
                self.trigger_export_pipeline();
            }

            ui.separator();

            if ui
                .add(pill_button("⌂", false, pal))
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
                .add(pill_button("⌫", false, pal))
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
                            "◿ 2D Symbolic Graph",
                        )
                        .clicked()
                    {
                        self.viewport_mode = ViewportMode::Plot2D;
                        ui.close_menu();
                    }
                    if ui
                        .selectable_label(
                            self.viewport_mode == ViewportMode::Surface3D,
                            "◈ 3D Surface Z = f(x, y)",
                        )
                        .clicked()
                    {
                        self.viewport_mode = ViewportMode::Surface3D;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("¶ R Script Editor").clicked() {
                        self.maximize_graph = false;
                        self.show_bottom_panel = true;
                        self.bottom_tab = BottomDockTab::ScriptEditor;
                        ui.close_menu();
                    }
                    if ui.button("§ Quarto (.qmd) Notes").clicked() {
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
        let mut collapse_sidebar = false;
        let mut toggle_min = false;
        let mut do_undo = false;
        let mut do_redo = false;
        let mut add_kind: Option<u8> = None; // 0=Expr, 1=Slider, 2=Folder, 3=Note, 4=Table

        // ---- Card 1 Top Toolbar: `+` Add Menu | Undo/Redo | `α≥` | `⚙` Edit Mode | `<` ----
        let header_rounding = if self.symbolic_card_collapsed {
            Rounding::same(8.0_f32)
        } else {
            Rounding {
                nw: 8.0_f32,
                ne: 8.0_f32,
                sw: 0.0_f32,
                se: 0.0_f32,
            }
        };
        let header_resp = egui::Frame::none()
            .fill(pal.card_header_bg)
            .rounding(header_rounding)
            .inner_margin(Margin::symmetric(8.0_f32, 5.0_f32))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0_f32;
                    // Left: `+` Add Item dropdown menu
                    ui.menu_button(
                        RichText::new("+")
                            .font(FontId::proportional(14.5_f32))
                            .strong()
                            .color(pal.accent_blue),
                        |ui| {
                            ui.label(
                                RichText::new("Add Symbolic Item")
                                    .strong()
                                    .color(pal.text_secondary),
                            );
                            ui.separator();
                            if ui.button("ƒ(x)  Expression / Equation").clicked() {
                                add_kind = Some(0);
                                ui.close_menu();
                            }
                            if ui.button("÷  Parameter Slider").clicked() {
                                add_kind = Some(1);
                                ui.close_menu();
                            }
                            if ui.button("▣  Collapsible Folder").clicked() {
                                add_kind = Some(2);
                                ui.close_menu();
                            }
                            if ui.button("¶  Text / Economic Note").clicked() {
                                add_kind = Some(3);
                                ui.close_menu();
                            }
                            if ui.button("⊞  Function Value Table").clicked() {
                                add_kind = Some(4);
                                ui.close_menu();
                            }
                        },
                    )
                    .response
                    .on_hover_text("Add Expression, Slider, Folder, Note, or Table");

                    ui.label(
                        RichText::new("Symbolic")
                            .font(FontId::proportional(12.5_f32))
                            .strong()
                            .color(pal.text_primary),
                    );

                    // Center: Undo (`↶`) & Redo (`↷`)
                    ui.add_space(4.0_f32);
                    let undo_col = if self.undo_history.can_undo() {
                        pal.text_primary
                    } else {
                        pal.text_muted.gamma_multiply(0.45_f32)
                    };
                    if ui
                        .add_enabled(
                            self.undo_history.can_undo(),
                            header_icon_button("↶", undo_col),
                        )
                        .on_hover_text("Undo symbolic edit")
                        .clicked()
                    {
                        do_undo = true;
                    }
                    let redo_col = if self.undo_history.can_redo() {
                        pal.text_primary
                    } else {
                        pal.text_muted.gamma_multiply(0.45_f32)
                    };
                    if ui
                        .add_enabled(
                            self.undo_history.can_redo(),
                            header_icon_button("↷", redo_col),
                        )
                        .on_hover_text("Redo symbolic edit")
                        .clicked()
                    {
                        do_redo = true;
                    }

                    // Right: `α≥`, `⚙` List Edit Mode, `—`, `<`
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 3.0_f32;
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
                            .add(pill_button("⚙", self.symbolic_list_edit_mode, pal))
                            .on_hover_text("Toggle List Edit Mode (Duplicate, Table, Freeze Baseline, Clear All)")
                            .clicked()
                        {
                            self.symbolic_list_edit_mode = !self.symbolic_list_edit_mode;
                        }
                        if ui
                            .add(pill_button("α≥", self.smart_math_shortcuts, pal))
                            .on_hover_text(
                                "Smart Math Shortcuts (`alpha `→`α `, `<=`→`≤`, `>=`→`≥`, `!=`→`≠`, `sqrt `→`√`, `int `→`∫`)",
                            )
                            .clicked()
                        {
                            self.smart_math_shortcuts = !self.smart_math_shortcuts;
                        }
                    });
                });
            })
            .response;

        if !self.symbolic_card_collapsed {
            let r = header_resp.rect;
            ui.painter().line_segment(
                [Pos2::new(r.left(), r.bottom()), Pos2::new(r.right(), r.bottom())],
                Stroke::new(1.0_f32, pal.border),
            );
            ui.add_space(-ui.spacing().item_spacing.y);
        }

        if do_undo {
            self.perform_undo();
        }
        if do_redo {
            self.perform_redo();
        }
        if let Some(kind_code) = add_kind {
            self.symbolic_card_collapsed = false;
            match kind_code {
                0 => {
                    self.push_undo_snapshot();
                    let id = self.workspace.alloc_cell_id();
                    let mut cell = SymbolicCell::new_expression(id, "", None);
                    // Attach inside selected folder if currently selecting a folder or child
                    if let Some(sel_id) = self.selected_cell_id {
                        if let Some(sel_cell) = self.workspace.cells.iter().find(|c| c.id == sel_id)
                        {
                            cell.folder_id = match sel_cell.kind {
                                SymbolicCellKind::Folder { .. } => Some(sel_cell.id),
                                _ => sel_cell.folder_id,
                            };
                        }
                    }
                    self.workspace.cells.push(cell);
                    self.selected_cell_id = Some(id);
                    self.request_focus_cell_id = Some(id);
                }
                1 => {
                    self.push_undo_snapshot();
                    let mut idx = 1usize;
                    let mut name = "k".to_string();
                    while self.workspace.parameters.contains_key(&name) {
                        idx += 1;
                        name = format!("k_{}", idx);
                    }
                    self.workspace.set_parameter(name, 1.0_f64);
                }
                2 => {
                    self.push_undo_snapshot();
                    let id = self.workspace.alloc_cell_id();
                    let folder = SymbolicCell::new_folder(id, "Model Group");
                    self.workspace.cells.push(folder);
                    self.selected_cell_id = Some(id);
                }
                3 => {
                    self.push_undo_snapshot();
                    let id = self.workspace.alloc_cell_id();
                    let note = SymbolicCell::new_note(
                        id,
                        "Economic Note: describe equilibrium or comparative statics...",
                    );
                    self.workspace.cells.push(note);
                    self.selected_cell_id = Some(id);
                }
                4 => {
                    let target_sig = self
                        .workspace
                        .functions
                        .keys()
                        .find(|k| !k.starts_with("__cell_"))
                        .cloned()
                        .or_else(|| self.workspace.functions.keys().next().cloned())
                        .unwrap_or_else(|| "f(x)".to_string());
                    self.create_table_for_function(&target_sig, self.selected_cell_id);
                }
                _ => {}
            }
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

        // ---- Optional List Edit Mode Action Bar (`† Freeze Baseline`, `Clear All`, `Done`) ----
        if self.symbolic_list_edit_mode {
            render_card_sub_toolbar(ui, pal, |ui| {
                if ui
                    .add(pill_button(
                        "† Freeze Baseline",
                        self.frozen_baseline.is_some(),
                        pal,
                    ))
                    .on_hover_text(
                        "Freeze current curves & equilibrium as ghost baseline for comparative statics",
                    )
                    .clicked()
                {
                    self.freeze_current_baseline();
                }
                if self.frozen_baseline.is_some()
                    && ui
                        .add(pill_button("Unfreeze", false, pal))
                        .on_hover_text("Clear frozen baseline")
                        .clicked()
                {
                    self.frozen_baseline = None;
                }
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("Clear All")
                                .font(FontId::proportional(11.5_f32))
                                .color(pal.accent_red),
                        )
                        .fill(pal.pill_bg)
                        .stroke(Stroke::new(1.0_f32, pal.accent_red.gamma_multiply(0.5_f32)))
                        .rounding(Rounding::same(5.0_f32)),
                    )
                    .clicked()
                {
                    self.clear_entire_workspace();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(pill_button("Done", true, pal)).clicked() {
                        self.symbolic_list_edit_mode = false;
                    }
                });
            });
        }

        // ---- Cycle warning banner if cyclic function definitions exist ----
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
        }

        // ---- Deferred Mutations Collected While Iterating Unified Cell List ----
        let mut cells_to_sync: Vec<usize> = Vec::new();
        let mut remove_cell_id: Option<u64> = None;
        let mut duplicate_cell_id: Option<u64> = None;
        let mut insert_after_cell_id: Option<u64> = None;
        let mut move_cell_swap: Option<(usize, usize)> = None;
        let mut table_for_fn: Option<(String, u64)> = None;
        let mut add_sliders: Vec<(String, u64)> = Vec::new();
        let mut status_msg: Option<String> = None;
        let mut sync_tables: Vec<(String, Vec<f64>, String)> = Vec::new();
        let mut need_undo_snapshot = false;

        let existing_params: std::collections::HashSet<String> =
            self.workspace.parameters.keys().cloned().collect();
        let existing_r_objs: std::collections::HashSet<String> =
            self.workspace.r_objects.keys().cloned().collect();
        let cell_count = self.workspace.cells.len();

        egui::Frame::none()
            .inner_margin(Margin::symmetric(6.0_f32, 6.0_f32))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_source("bento_symbolic_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let mut display_index = 0usize;

                        for idx in 0..cell_count {
                            let cell_id = self.workspace.cells[idx].id;
                            let folder_id = self.workspace.cells[idx].folder_id;

                            // Skip children of collapsed folders
                            if self.workspace.is_folder_collapsed(folder_id) {
                                continue;
                            }
                            display_index += 1;

                            let is_selected = self.selected_cell_id == Some(cell_id);
                            let in_folder = folder_id.is_some();
                            let folder_vis = self.workspace.is_folder_visible(folder_id);

                            let cell_border = if is_selected {
                                pal.accent_blue.gamma_multiply(0.85_f32)
                            } else {
                                pal.border
                            };

                            let card_resp = egui::Frame::none()
                                .fill(if is_selected {
                                    pal.sub_card_bg
                                } else {
                                    pal.sub_card_bg.gamma_multiply(0.88_f32)
                                })
                                .stroke(Stroke::new(
                                    if is_selected { 1.3_f32 } else { 1.0_f32 },
                                    cell_border,
                                ))
                                .rounding(Rounding::same(8.0_f32))
                                .inner_margin(Margin {
                                    left: if in_folder { 8.0_f32 } else { 4.0_f32 },
                                    right: 6.0_f32,
                                    top: 5.0_f32,
                                    bottom: 5.0_f32,
                                })
                                .show(ui, |ui| {
                                    ui.set_min_width(ui.available_width());
                                    let kind_tag = match &self.workspace.cells[idx].kind {
                                        SymbolicCellKind::Expression { .. } => 0u8,
                                        SymbolicCellKind::Slider { .. } => 1u8,
                                        SymbolicCellKind::Folder { .. } => 2u8,
                                        SymbolicCellKind::Note { .. } => 3u8,
                                        SymbolicCellKind::Table { .. } => 4u8,
                                    };

                                    match kind_tag {
                                        0 => {
                                            self.render_symbolic_expression_cell(
                                                ui,
                                                idx,
                                                display_index,
                                                cell_id,
                                                is_selected,
                                                folder_vis,
                                                pal,
                                                &existing_params,
                                                &existing_r_objs,
                                                &mut cells_to_sync,
                                                &mut remove_cell_id,
                                                &mut duplicate_cell_id,
                                                &mut insert_after_cell_id,
                                                &mut table_for_fn,
                                                &mut add_sliders,
                                                &mut status_msg,
                                                &mut need_undo_snapshot,
                                            );
                                        }
                                        1 => {
                                            self.render_symbolic_slider_cell(
                                                ui,
                                                idx,
                                                display_index,
                                                cell_id,
                                                pal,
                                                &mut remove_cell_id,
                                                &mut duplicate_cell_id,
                                            );
                                        }
                                        2 => {
                                            self.render_symbolic_folder_cell(
                                                ui,
                                                idx,
                                                display_index,
                                                cell_id,
                                                pal,
                                                &mut remove_cell_id,
                                                &mut duplicate_cell_id,
                                            );
                                        }
                                        3 => {
                                            self.render_symbolic_note_cell(
                                                ui,
                                                idx,
                                                display_index,
                                                cell_id,
                                                pal,
                                                &mut remove_cell_id,
                                                &mut duplicate_cell_id,
                                            );
                                        }
                                        _ => {
                                            self.render_symbolic_table_cell(
                                                ui,
                                                idx,
                                                display_index,
                                                cell_id,
                                                pal,
                                                &mut remove_cell_id,
                                                &mut sync_tables,
                                            );
                                        }
                                    }

                                    // List Edit Mode quick reorder controls (`↑`, `↓`, folder nest toggle)
                                    if self.symbolic_list_edit_mode {
                                        ui.add_space(2.0_f32);
                                        ui.horizontal(|ui| {
                                            ui.add_space(38.0_f32);
                                            if idx > 0
                                                && ui
                                                    .add(header_icon_button("↑", pal.text_secondary))
                                                    .on_hover_text("Move up")
                                                    .clicked()
                                            {
                                                move_cell_swap = Some((idx, idx - 1));
                                            }
                                            if idx + 1 < cell_count
                                                && ui
                                                    .add(header_icon_button("↓", pal.text_secondary))
                                                    .on_hover_text("Move down")
                                                    .clicked()
                                            {
                                                move_cell_swap = Some((idx, idx + 1));
                                            }
                                            if ui
                                                .add(header_icon_button("❐", pal.accent_blue))
                                                .on_hover_text("Duplicate cell")
                                                .clicked()
                                            {
                                                duplicate_cell_id = Some(cell_id);
                                            }
                                            if kind_tag != 2 {
                                                // Attach to previous folder or unnest
                                                let prev_folder = (0..idx).rev().find_map(|j| {
                                                    matches!(
                                                        self.workspace.cells[j].kind,
                                                        SymbolicCellKind::Folder { .. }
                                                    )
                                                    .then_some(self.workspace.cells[j].id)
                                                });
                                                if let Some(pfid) = prev_folder {
                                                    let nested =
                                                        self.workspace.cells[idx].folder_id == Some(pfid);
                                                    if ui
                                                        .add(header_icon_button(
                                                            if nested { "⇤ Out" } else { "⇥ Folder" },
                                                            pal.text_muted,
                                                        ))
                                                        .on_hover_text("Toggle folder nesting")
                                                        .clicked()
                                                    {
                                                        self.workspace.cells[idx].folder_id =
                                                            if nested { None } else { Some(pfid) };
                                                    }
                                                }
                                            }
                                        });
                                    }
                                })
                                .response;

                            // Draw vertical connector bar for cells nested inside a collapsible Folder
                            if in_folder {
                                let r = card_resp.rect;
                                ui.painter().line_segment(
                                    [
                                        Pos2::new(r.left() + 3.0_f32, r.top() + 3.0_f32),
                                        Pos2::new(r.left() + 3.0_f32, r.bottom() - 3.0_f32),
                                    ],
                                    Stroke::new(2.0_f32, pal.accent_blue.gamma_multiply(0.55_f32)),
                                );
                            }

                            ui.add_space(4.0_f32);
                        }

                        // ---- Ghost Row at Bottom (`+ Click or press Enter to add expression...`) ----
                        let ghost_resp = egui::Frame::none()
                            .fill(pal.extreme_bg.gamma_multiply(0.55_f32))
                            .stroke(Stroke::new(1.0_f32, pal.border.gamma_multiply(0.6_f32)))
                            .rounding(Rounding::same(8.0_f32))
                            .inner_margin(Margin::symmetric(10.0_f32, 7.0_f32))
                            .show(ui, |ui| {
                                ui.set_min_width(ui.available_width());
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(format!("{}", display_index + 1))
                                            .font(FontId::monospace(10.5_f32))
                                            .color(pal.text_muted),
                                    );
                                    ui.add_space(8.0_f32);
                                    ui.label(
                                        RichText::new("+ Click or press Enter to add expression...")
                                            .font(FontId::proportional(12.0_f32))
                                            .italics()
                                            .color(pal.text_muted),
                                    );
                                });
                            })
                            .response
                            .interact(Sense::click());
                        if ghost_resp.clicked() {
                            self.push_undo_snapshot();
                            let id = self.workspace.alloc_cell_id();
                            self.workspace
                                .cells
                                .push(SymbolicCell::new_expression(id, "", None));
                            self.selected_cell_id = Some(id);
                            self.request_focus_cell_id = Some(id);
                        }

                        // ---- Optional Equilibrium Summary Card ----
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

        // ---- Apply Deferred Structural Edits ----
        if need_undo_snapshot {
            self.push_undo_snapshot();
        }
        for c_idx in cells_to_sync {
            self.sync_expression_cell(c_idx);
        }
        if let Some((a, b)) = move_cell_swap {
            self.push_undo_snapshot();
            self.workspace.cells.swap(a, b);
        }
        if let Some(after_id) = insert_after_cell_id {
            self.push_undo_snapshot();
            let id = self.workspace.alloc_cell_id();
            if let Some(pos) = self.workspace.cells.iter().position(|c| c.id == after_id) {
                let mut cell = SymbolicCell::new_expression(id, "", None);
                cell.folder_id = self.workspace.cells[pos].folder_id;
                self.workspace.cells.insert(pos + 1, cell);
            } else {
                self.workspace
                    .cells
                    .push(SymbolicCell::new_expression(id, "", None));
            }
            self.selected_cell_id = Some(id);
            self.request_focus_cell_id = Some(id);
        }
        if let Some(dup_id) = duplicate_cell_id {
            self.push_undo_snapshot();
            if let Some(new_id) = self.workspace.duplicate_cell(dup_id) {
                self.selected_cell_id = Some(new_id);
            }
        }
        if let Some(rem_id) = remove_cell_id {
            self.push_undo_snapshot();
            let prev_focus = self
                .workspace
                .cells
                .iter()
                .position(|c| c.id == rem_id)
                .and_then(|p| p.checked_sub(1))
                .and_then(|p| self.workspace.cells.get(p))
                .map(|c| c.id);
            self.workspace.remove_cell(rem_id);
            if self.selected_cell_id == Some(rem_id) {
                self.selected_cell_id = prev_focus;
                self.request_focus_cell_id = prev_focus;
            }
        }
        if !add_sliders.is_empty() {
            self.push_undo_snapshot();
            for (sym, after_id) in add_sliders {
                self.workspace
                    .insert_parameter_after(sym, 1.0_f64, Some(after_id));
            }
        }
        if let Some((sig, after_id)) = table_for_fn {
            self.create_table_for_function(&sig, Some(after_id));
        }
        for (sig, xs, ds_name) in sync_tables {
            self.sync_table_cell_to_data_and_r(&sig, &xs, &ds_name);
        }
        if let Some(msg) = status_msg {
            self.status_banner = msg;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_symbolic_expression_cell(
        &mut self,
        ui: &mut Ui,
        idx: usize,
        display_index: usize,
        cell_id: u64,
        is_selected: bool,
        folder_vis: bool,
        pal: ThemePalette,
        existing_params: &std::collections::HashSet<String>,
        existing_r_objs: &std::collections::HashSet<String>,
        cells_to_sync: &mut Vec<usize>,
        remove_cell_id: &mut Option<u64>,
        duplicate_cell_id: &mut Option<u64>,
        insert_after_cell_id: &mut Option<u64>,
        table_for_fn: &mut Option<(String, u64)>,
        add_sliders: &mut Vec<(String, u64)>,
        status_msg: &mut Option<String>,
        need_undo_snapshot: &mut bool,
    ) {
        let (
            mut raw,
            bound_key,
            mut latex_drawer_open,
            mut latex_draft,
            mut latex_error,
            mut show_fraction,
            mut plot_eval_point,
            mut highlight_roots,
        ) = match &self.workspace.cells[idx].kind {
            SymbolicCellKind::Expression {
                raw,
                bound_key,
                latex_drawer_open,
                latex_draft,
                latex_error,
                show_fraction,
                plot_eval_point,
                highlight_roots,
            } => (
                raw.clone(),
                bound_key.clone(),
                *latex_drawer_open,
                latex_draft.clone(),
                latex_error.clone(),
                *show_fraction,
                *plot_eval_point,
                *highlight_roots,
            ),
            _ => return,
        };

        let style_key = bound_key
            .clone()
            .unwrap_or_else(|| format!("__cell_{}", cell_id));
        let mut style = self
            .workspace
            .function_styles
            .get(&style_key)
            .cloned()
            .unwrap_or_else(|| FunctionStyle::default_for_index(idx));

        // Pre-classify expression for gutter badge & result pill
        let known_fns = self.workspace.compiled_math.known_fn_names.clone();
        let trimmed_non_empty = !raw.trim().is_empty();
        let (inline_param_assign, classified, gutter_syntax_err, gutter_has_unbound) = {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                (None, None, false, false)
            } else {
                let strict_stmt = math::parse_statement_with_known_fns(trimmed, Some(&known_fns));
                let is_structural_placeholder = strict_stmt.is_err()
                    && math::parser::parse_statement_for_preview(trimmed, Some(&known_fns))
                        .map(|s| s.has_placeholder())
                        .unwrap_or(false);
                let syntax_err = strict_stmt.is_err() && !is_structural_placeholder;

                let mut has_unbound = false;
                if let Ok(ref st) = strict_stmt {
                    let mut locals = std::collections::HashSet::new();
                    if let math::MathStatement::FunctionDef { ref params, .. } = st {
                        for p in params {
                            locals.insert(p.clone());
                        }
                    }
                    let check_e = |e: &math::SpannedExpr| {
                        !math::graph::discover_unbound_symbols(
                            e,
                            &locals,
                            &known_fns,
                            existing_params,
                            existing_r_objs,
                        )
                        .is_empty()
                    };
                    has_unbound = match st {
                        math::MathStatement::Expr(e)
                        | math::MathStatement::FunctionDef { body: e, .. }
                        | math::MathStatement::ParamAssign { value: e, .. }
                        | math::MathStatement::Polar { r_expr: e, .. } => check_e(e),
                        math::MathStatement::Equation { lhs, rhs }
                        | math::MathStatement::Inequality { lhs, rhs, .. } => {
                            check_e(lhs) || check_e(rhs)
                        }
                        math::MathStatement::Parametric { x_expr, y_expr, .. } => {
                            check_e(x_expr) || check_e(y_expr)
                        }
                    };
                }

                let param_assign = strict_stmt.ok().and_then(|st| match st {
                    math::MathStatement::ParamAssign { name, .. } => Some(name),
                    _ => None,
                });
                let cls = if param_assign.is_some() {
                    None
                } else {
                    let sig_for_cls = bound_key.as_deref().unwrap_or("");
                    let expr_for_cls = bound_key
                        .as_deref()
                        .and_then(|k| self.workspace.functions.get(k).map(|s| s.as_str()))
                        .unwrap_or(trimmed);
                    self.workspace
                        .compiled_math
                        .classify_entry(sig_for_cls, expr_for_cls)
                        .ok()
                };
                (param_assign, cls, syntax_err, has_unbound)
            }
        };
        let is_plottable_curve = matches!(
            classified,
            Some(
                PlottableEntry::Explicit1D { .. }
                    | PlottableEntry::ImplicitEquation { .. }
                    | PlottableEntry::ExplicitYInequality { .. }
                    | PlottableEntry::ImplicitInequality { .. }
                    | PlottableEntry::Parametric { .. }
                    | PlottableEntry::Polar { .. }
                    | PlottableEntry::TangentLine { .. }
                    | PlottableEntry::AreaBetween { .. }
                    | PlottableEntry::Equation1D { .. }
            )
        );

        let mut raw_changed = false;

        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0_f32;

            // ---- 36px Left Gutter (Index + Interactive Type/Visibility/Anim Badge) ----
            ui.allocate_ui_with_layout(
                Vec2::new(36.0_f32, 38.0_f32),
                egui::Layout::top_down(egui::Align::Center),
                |ui| {
                    ui.label(
                        RichText::new(format!("{}", display_index))
                            .font(FontId::monospace(9.8_f32))
                            .color(pal.text_muted),
                    );
                    if gutter_syntax_err || gutter_has_unbound {
                        let warn_col = if gutter_syntax_err {
                            pal.accent_red
                        } else {
                            pal.accent_amber
                        };
                        let (r, resp) = ui.allocate_exact_size(Vec2::splat(22.0_f32), Sense::hover());
                        ui.painter()
                            .circle_filled(r.center(), 9.5_f32, warn_col.gamma_multiply(0.18_f32));
                        ui.painter()
                            .circle_stroke(r.center(), 9.5_f32, Stroke::new(1.2_f32, warn_col));
                        ui.painter().text(
                            r.center(),
                            egui::Align2::CENTER_CENTER,
                            "⚠",
                            FontId::proportional(11.5_f32),
                            warn_col,
                        );
                        resp.on_hover_text(if gutter_syntax_err {
                            "Syntax error in expression"
                        } else {
                            "Missing slider(s) for free parameters — click Add slider below"
                        });
                    } else if let Some(ref pname) = inline_param_assign {
                        // Inline parameter assignment (`a = 2.5`) gets Play/Pause in gutter!
                        let mut spec = self
                            .workspace
                            .slider_specs
                            .get(pname)
                            .cloned()
                            .unwrap_or_else(|| SliderSpec::for_parameter(pname, 1.0_f64));
                        let play_icon = if spec.playing { "⏸" } else { "▶" };
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(play_icon)
                                        .font(FontId::proportional(11.0_f32))
                                        .color(if spec.playing {
                                            pal.accent_green
                                        } else {
                                            pal.accent_blue
                                        }),
                                )
                                .fill(pal.pill_bg)
                                .stroke(Stroke::new(1.0_f32, pal.border))
                                .rounding(Rounding::same(11.0_f32))
                                .min_size(Vec2::splat(22.0_f32)),
                            )
                            .on_hover_text("Play / Pause slider animation")
                            .clicked()
                        {
                            spec.playing = !spec.playing;
                        }
                        self.workspace.slider_specs.insert(pname.clone(), spec);
                    } else if is_plottable_curve {
                        let badge_resp = render_gutter_curve_badge(
                            ui,
                            style.color,
                            style.visible && folder_vis,
                            style.line_style,
                            style.show_points,
                            pal,
                        )
                        .on_hover_text(
                            "Left-click: toggle curve visibility | Right-click: appearance popover",
                        );
                        if badge_resp.clicked() {
                            style.visible = !style.visible;
                        }
                        if badge_resp.secondary_clicked() {
                            style.appearance_popover_open = !style.appearance_popover_open;
                        }
                        // Tiny caret button for appearance popover
                        let pop_id = ui.make_persistent_id(format!("appear_pop_{}", cell_id));
                        let caret_resp = ui
                            .add(header_icon_button("▾", pal.text_muted))
                            .on_hover_text("Curve appearance & color");
                        if caret_resp.clicked() || style.appearance_popover_open {
                            if caret_resp.clicked() {
                                ui.memory_mut(|m| m.toggle_popup(pop_id));
                            } else if style.appearance_popover_open {
                                ui.memory_mut(|m| m.open_popup(pop_id));
                                style.appearance_popover_open = false;
                            }
                        }
                        egui::popup_below_widget(
                            ui,
                            pop_id,
                            &caret_resp,
                            egui::PopupCloseBehavior::CloseOnClickOutside,
                            |ui| {
                                render_curve_appearance_controls(
                                    ui,
                                    &style_key,
                                    &mut style,
                                    pal,
                                );
                            },
                        );
                    } else if trimmed_non_empty {
                        // Scalar calculator badge
                        let (r, _) = ui.allocate_exact_size(Vec2::splat(22.0_f32), Sense::hover());
                        ui.painter()
                            .circle_filled(r.center(), 9.5_f32, pal.pill_bg);
                        ui.painter()
                            .circle_stroke(r.center(), 9.5_f32, Stroke::new(1.0_f32, pal.border));
                        ui.painter().text(
                            r.center(),
                            egui::Align2::CENTER_CENTER,
                            "=",
                            FontId::monospace(11.5_f32),
                            pal.accent_blue,
                        );
                    }
                },
            );

            // ---- Right Cell Body ----
            ui.vertical(|ui| {
                ui.set_min_width((ui.available_width() - 4.0_f32).max(120.0_f32));

                // 1. Live 2D Structural "Math-as-Written" Area (click directly to edit!)
                let (preview, preview_resp) = egui::ScrollArea::horizontal()
                    .id_source(format!("cell_math_scroll_{}", cell_id))
                    .show(ui, |ui| {
                        render_math_preview_strip_interactive(
                            ui,
                            None,
                            &raw,
                            &known_fns,
                            15.5_f32,
                            pal.text_primary,
                            Some(style.color),
                            pal.extreme_bg,
                            pal.border,
                            pal.accent_red,
                            is_selected,
                            pal.accent_blue,
                        )
                    })
                    .inner;
                if preview_resp.clicked() {
                    self.selected_cell_id = Some(cell_id);
                    self.request_focus_cell_id = Some(cell_id);
                }

                ui.add_space(3.0_f32);

                // 2. Unified Single-Input Editor + Compact Action Strip (`TeX`, `⚙`, `⊞`, `×`)
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0_f32;
                    let edit_w = (ui.available_width() - 102.0_f32).max(64.0_f32);
                    let edit_id = ui.make_persistent_id(format!("sym_cell_edit_{}", cell_id));
                    let was_empty = raw.is_empty();
                    let edit_resp = ui.add(
                        egui::TextEdit::singleline(&mut raw)
                            .id(edit_id)
                            .font(FontId::monospace(11.6_f32))
                            .desired_width(edit_w)
                            .hint_text("f(x) = x^2, 1/x, f(5), x^2 - 5x + 6 = 0, tangent(f, 2)"),
                    );

                    if self.request_focus_cell_id == Some(cell_id) {
                        edit_resp.request_focus();
                        self.request_focus_cell_id = None;
                    }
                    if edit_resp.gained_focus() || edit_resp.clicked() {
                        self.selected_cell_id = Some(cell_id);
                    }
                    if edit_resp.changed() {
                        if self.smart_math_shortcuts {
                            apply_smart_math_shortcuts(&mut raw);
                        }
                        raw_changed = true;
                        // Two-way sync: keep LaTeX drawer synced live with symbolic input
                        latex_draft = math::to_latex_from_ast(&raw);
                        latex_error = None;
                    }
                    if edit_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        *insert_after_cell_id = Some(cell_id);
                    }
                    if edit_resp.has_focus()
                        && was_empty
                        && raw.is_empty()
                        && ui.input(|i| i.key_pressed(egui::Key::Backspace))
                    {
                        *remove_cell_id = Some(cell_id);
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 2.5_f32;
                        if ui
                            .add(header_icon_button("×", pal.accent_red))
                            .on_hover_text("Delete cell")
                            .clicked()
                        {
                            *remove_cell_id = Some(cell_id);
                        }
                        if is_plottable_curve {
                            if ui
                                .add(header_icon_button(
                                    "⚙",
                                    if style.settings_open {
                                        pal.accent_blue
                                    } else {
                                        pal.text_secondary
                                    },
                                ))
                                .on_hover_text("Curve appearance & X/Y/t domain configuration")
                                .clicked()
                            {
                                style.settings_open = !style.settings_open;
                            }
                            if let Some(ref bk) = bound_key {
                                if !bk.starts_with("__cell_")
                                    && ui
                                        .add(header_icon_button("⊞", pal.text_secondary))
                                        .on_hover_text(
                                            "Create synced Function Value Table (Data Viewer + R data.frame)",
                                        )
                                        .clicked()
                                {
                                    *table_for_fn = Some((bk.clone(), cell_id));
                                }
                            }
                        }
                        if ui
                            .add(pill_button("TeX", latex_drawer_open, pal))
                            .on_hover_text("Toggle Two-Way Live LaTeX Drawer")
                            .clicked()
                        {
                            latex_drawer_open = !latex_drawer_open;
                            if latex_drawer_open {
                                latex_draft = if !preview.latex.is_empty() {
                                    preview.latex.clone()
                                } else {
                                    math::to_latex_from_ast(&raw)
                                };
                                latex_error = None;
                            }
                        }
                    });
                });

                // 3. Inline Parameter Slider if cell is `a = 2.5`
                if let Some(ref pname) = inline_param_assign {
                    let mut val = self
                        .workspace
                        .parameters
                        .get(pname)
                        .copied()
                        .unwrap_or(1.0_f64);
                    let mut spec = self
                        .workspace
                        .slider_specs
                        .get(pname)
                        .cloned()
                        .unwrap_or_else(|| SliderSpec::for_parameter(pname, val));
                    let safe_min = spec.min.min(spec.max - 1e-6_f64);
                    let safe_max = spec.max.max(safe_min + 1e-6_f64);
                    ui.add_space(2.0_f32);
                    ui.horizontal(|ui| {
                        if ui
                            .add(header_icon_button(&format!("{:.1}", spec.min), pal.text_muted))
                            .on_hover_text("Click to edit slider bounds")
                            .clicked()
                        {
                            spec.edit_bounds_open = !spec.edit_bounds_open;
                        }
                        ui.spacing_mut().slider_width =
                            (ui.available_width() - 62.0_f32).max(50.0_f32);
                        let mut sl = egui::Slider::new(&mut val, safe_min..=safe_max)
                            .show_value(false);
                        if spec.step > 0.0_f64 {
                            sl = sl.step_by(spec.step);
                        }
                        if ui.add(sl).changed() {
                            self.workspace.parameters.insert(pname.clone(), val);
                            self.workspace
                                .r_objects
                                .insert(pname.clone(), RObject::scalar(val));
                            raw = format!("{} = {}", pname, Self::format_scalar_result(val, false).0);
                            latex_draft = math::to_latex_from_ast(&raw);
                        }
                        if ui
                            .add(header_icon_button(&format!("{:.1}", spec.max), pal.text_muted))
                            .on_hover_text("Click to edit slider bounds")
                            .clicked()
                        {
                            spec.edit_bounds_open = !spec.edit_bounds_open;
                        }
                    });
                    if spec.edit_bounds_open {
                        ui.horizontal_wrapped(|ui| {
                            ui.add(egui::DragValue::new(&mut spec.min).speed(0.1).prefix("Min:"));
                            ui.add(egui::DragValue::new(&mut spec.max).speed(0.1).prefix("Max:"));
                            ui.add(egui::DragValue::new(&mut spec.step).speed(0.01).prefix("Step:"));
                        });
                    }
                    self.workspace.slider_specs.insert(pname.clone(), spec);
                }

                // 4. Right-Aligned Result Pill (`= valor`, roots `x = 2, 3`, tangent `y = mx+b`,
                //    `area`/`surplus` integral, R vector `[1, 4, 9, 16]`, or inline syntax diagnostic)
                if let Some(ref err) = preview.error {
                    let (_, col) = err.span.line_col(&raw);
                    ui.label(
                        RichText::new(format!("⚠ col {}: {}", col, err.message))
                            .font(FontId::monospace(10.6_f32))
                            .color(pal.accent_red),
                    );
                } else if preview.has_placeholders {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new("□ awaiting input…")
                                .font(FontId::monospace(10.2_f32))
                                .color(pal.accent_blue.gamma_multiply(0.8_f32)),
                        );
                    });
                } else if let Some(vec_str) = self.try_eval_vectorized_r_cell(&raw) {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        egui::Frame::none()
                            .fill(pal.extreme_bg)
                            .stroke(Stroke::new(1.0_f32, pal.accent_green.gamma_multiply(0.55_f32)))
                            .rounding(Rounding::same(5.0_f32))
                            .inner_margin(Margin::symmetric(7.0_f32, 2.0_f32))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(format!("= {}", vec_str))
                                        .font(FontId::monospace(11.6_f32))
                                        .strong()
                                        .color(pal.accent_green),
                                );
                            });
                    });
                } else if let Some(ref plottable) = classified {
                    let ext_bridge = REngineMathBridge {
                        r_objects: &self.workspace.r_objects,
                    };
                    let eval_ctx = self.build_eval_context(&ext_bridge);
                    let mut scratch = EvalScratch::new();

                    match plottable {
                        PlottableEntry::ScalarOnly {
                            compiled,
                            eval_point_x,
                        } => {
                            if let Ok(v) = compiled.eval(&eval_ctx, &[], &mut scratch) {
                                if v.is_finite() {
                                    let (disp, frac_opt) =
                                        Self::format_scalar_result(v, show_fraction);
                                    ui.add_space(2.0_f32);
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            egui::Frame::none()
                                                .fill(pal.extreme_bg)
                                                .stroke(Stroke::new(
                                                    1.0_f32,
                                                    pal.accent_green.gamma_multiply(0.6_f32),
                                                ))
                                                .rounding(Rounding::same(5.0_f32))
                                                .inner_margin(Margin::symmetric(7.0_f32, 2.0_f32))
                                                .show(ui, |ui| {
                                                    ui.label(
                                                        RichText::new(format!("= {}", disp))
                                                            .font(FontId::monospace(12.2_f32))
                                                            .strong()
                                                            .color(pal.accent_green),
                                                    );
                                                });
                                            if frac_opt.is_some()
                                                && ui
                                                    .add(pill_button(
                                                        if show_fraction { "0.3" } else { "⅓" },
                                                        show_fraction,
                                                        pal,
                                                    ))
                                                    .on_hover_text(
                                                        "Toggle Exact Fraction ↔ Decimal",
                                                    )
                                                    .clicked()
                                            {
                                                show_fraction = !show_fraction;
                                            }
                                            if let Some(px_comp) = eval_point_x {
                                                let mut sx = EvalScratch::new();
                                                if let Ok(x0) = px_comp.eval(&eval_ctx, &[], &mut sx)
                                                {
                                                    if x0.is_finite()
                                                        && ui
                                                            .add(pill_button(
                                                                "◈ Plot Point",
                                                                plot_eval_point,
                                                                pal,
                                                            ))
                                                            .on_hover_text(format!(
                                                                "Mark point ({:.3}, {:.3}) on 2D graph",
                                                                x0, v
                                                            ))
                                                            .clicked()
                                                    {
                                                        plot_eval_point = !plot_eval_point;
                                                    }
                                                }
                                            }
                                        },
                                    );
                                }
                            }
                        }
                        PlottableEntry::Equation1D {
                            diff_compiled,
                            lhs_compiled,
                            is_fn_intersection,
                        } => {
                            let roots = math::solve_compiled_roots_1d(
                                diff_compiled,
                                &eval_ctx,
                                self.domain_min,
                                self.domain_max,
                                320,
                            );
                            ui.add_space(2.0_f32);
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if roots.is_empty() {
                                    ui.label(
                                        RichText::new("No real roots in domain")
                                            .font(FontId::monospace(10.5_f32))
                                            .color(pal.text_muted),
                                    );
                                } else {
                                    let pill_text = if *is_fn_intersection && roots.len() == 1 {
                                        let rx = roots[0];
                                        let ry = math::eval_compiled_at_x(
                                            lhs_compiled,
                                            &eval_ctx,
                                            rx,
                                            &mut scratch,
                                        );
                                        if let Some(y_val) = ry {
                                            format!("x* ≈ {:.4}, y* ≈ {:.4}", rx, y_val)
                                        } else {
                                            format!("x* ≈ {:.4}", rx)
                                        }
                                    } else {
                                        let r_list: Vec<String> = roots
                                            .iter()
                                            .take(5)
                                            .map(|&r| Self::format_scalar_result(r, show_fraction).0)
                                            .collect();
                                        format!("x = {}", r_list.join(", "))
                                    };
                                    egui::Frame::none()
                                        .fill(pal.extreme_bg)
                                        .stroke(Stroke::new(
                                            1.0_f32,
                                            pal.accent_blue.gamma_multiply(0.65_f32),
                                        ))
                                        .rounding(Rounding::same(5.0_f32))
                                        .inner_margin(Margin::symmetric(7.0_f32, 2.0_f32))
                                        .show(ui, |ui| {
                                            ui.label(
                                                RichText::new(pill_text)
                                                    .font(FontId::monospace(11.6_f32))
                                                    .strong()
                                                    .color(pal.accent_blue),
                                            );
                                        });
                                    if ui
                                        .add(pill_button("◈ Roots", highlight_roots, pal))
                                        .on_hover_text("Highlight roots / intersections on graph")
                                        .clicked()
                                    {
                                        highlight_roots = !highlight_roots;
                                    }
                                }
                            });
                        }
                        PlottableEntry::TangentLine {
                            fn_name,
                            x0_compiled,
                            ..
                        } => {
                            if let Ok(x0) = x0_compiled.eval(&eval_ctx, &[], &mut scratch) {
                                if let Ok((y0, m)) = math::eval_fn_val_and_deriv(
                                    fn_name,
                                    x0,
                                    math::Span::default(),
                                    &eval_ctx,
                                    0,
                                ) {
                                    let b = y0 - m * x0;
                                    let sign = if b >= 0.0_f64 { "+" } else { "−" };
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            egui::Frame::none()
                                                .fill(pal.extreme_bg)
                                                .stroke(Stroke::new(1.0_f32, pal.accent_amber.gamma_multiply(0.6_f32)))
                                                .rounding(Rounding::same(5.0_f32))
                                                .inner_margin(Margin::symmetric(7.0_f32, 2.0_f32))
                                                .show(ui, |ui| {
                                                    ui.label(
                                                        RichText::new(format!(
                                                            "y = {:.3}x {} {:.3}",
                                                            m,
                                                            sign,
                                                            b.abs()
                                                        ))
                                                        .font(FontId::monospace(11.3_f32))
                                                        .strong()
                                                        .color(pal.accent_amber),
                                                    );
                                                });
                                        },
                                    );
                                }
                            }
                        }
                        PlottableEntry::AreaBetween { area_compiled, .. } => {
                            if let Ok(val) = area_compiled.eval(&eval_ctx, &[], &mut scratch) {
                                if val.is_finite() {
                                    let (disp, _) = Self::format_scalar_result(val, show_fraction);
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            egui::Frame::none()
                                                .fill(pal.extreme_bg)
                                                .stroke(Stroke::new(
                                                    1.0_f32,
                                                    pal.accent_green.gamma_multiply(0.6_f32),
                                                ))
                                                .rounding(Rounding::same(5.0_f32))
                                                .inner_margin(Margin::symmetric(7.0_f32, 2.0_f32))
                                                .show(ui, |ui| {
                                                    ui.label(
                                                        RichText::new(format!("∫ = {}", disp))
                                                            .font(FontId::monospace(11.6_f32))
                                                            .strong()
                                                            .color(pal.accent_green),
                                                    );
                                                });
                                        },
                                    );
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // 5. Automatic Unbound Parameter Discovery Bar (`Add slider: [a] [b] [c] [All]`)
                if let Some(ref stmt) = preview.parsed_statement {
                    let mut bound_locals = std::collections::HashSet::new();
                    if let math::MathStatement::FunctionDef { ref params, .. } = stmt {
                        for p in params {
                            bound_locals.insert(p.clone());
                        }
                    }
                    let mut unbound = Vec::new();
                    let mut collect_from = |e: &math::SpannedExpr| {
                        for sym in math::graph::discover_unbound_symbols(
                            e,
                            &bound_locals,
                            &known_fns,
                            existing_params,
                            existing_r_objs,
                        ) {
                            if !unbound.contains(&sym) {
                                unbound.push(sym);
                            }
                        }
                    };
                    match stmt {
                        math::MathStatement::Expr(e)
                        | math::MathStatement::FunctionDef { body: e, .. }
                        | math::MathStatement::ParamAssign { value: e, .. }
                        | math::MathStatement::Polar { r_expr: e, .. } => collect_from(e),
                        math::MathStatement::Equation { lhs, rhs }
                        | math::MathStatement::Inequality { lhs, rhs, .. } => {
                            collect_from(lhs);
                            collect_from(rhs);
                        }
                        math::MathStatement::Parametric { x_expr, y_expr, .. } => {
                            collect_from(x_expr);
                            collect_from(y_expr);
                        }
                    }
                    if !unbound.is_empty() {
                        ui.add_space(3.0_f32);
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                RichText::new("Add slider:")
                                    .font(FontId::proportional(10.8_f32))
                                    .color(pal.accent_amber),
                            );
                            for sym in &unbound {
                                let disp = math::ast::ident_to_pretty_unicode(sym);
                                if ui.add(pill_button(&disp, false, pal)).clicked() {
                                    add_sliders.push((sym.clone(), cell_id));
                                }
                            }
                            if ui
                                .add(pill_button("All", true, pal))
                                .on_hover_text("Create sliders for all free parameters at once")
                                .clicked()
                            {
                                for sym in &unbound {
                                    add_sliders.push((sym.clone(), cell_id));
                                }
                            }
                        });
                    }
                }

                // 6. Interactive Two-Way LaTeX Drawer (`TeX`)
                if latex_drawer_open {
                    ui.add_space(4.0_f32);
                    egui::Frame::none()
                        .fill(pal.extreme_bg)
                        .stroke(Stroke::new(1.0_f32, pal.accent_blue.gamma_multiply(0.55_f32)))
                        .rounding(Rounding::same(6.0_f32))
                        .inner_margin(Margin::same(6.0_f32))
                        .show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new("LaTeX (Two-Way Live Sync)")
                                        .font(FontId::proportional(10.8_f32))
                                        .strong()
                                        .color(pal.accent_blue),
                                );
                                if ui
                                    .add(header_icon_button("⎘ Copy", pal.text_primary))
                                    .on_hover_text("Copy canonical LaTeX to clipboard")
                                    .clicked()
                                {
                                    ui.ctx().copy_text(latex_draft.clone());
                                    *status_msg = Some(format!("Copied LaTeX: {}", latex_draft));
                                }
                                let tex_edit_id =
                                    ui.make_persistent_id(format!("tex_drawer_edit_{}", cell_id));
                                if ui
                                    .add(header_icon_button("⎗ Paste", pal.text_secondary))
                                    .on_hover_text("Focus LaTeX box to paste (Ctrl+V)")
                                    .clicked()
                                {
                                    ui.memory_mut(|m| m.request_focus(tex_edit_id));
                                }
                                if ui
                                    .add(header_icon_button("≡ Format", pal.accent_green))
                                    .on_hover_text("Canonicalize LaTeX from AST")
                                    .clicked()
                                {
                                    latex_draft = math::to_latex_from_ast(&raw);
                                    latex_error = None;
                                }
                                if ui
                                    .add(header_icon_button("◿ TikZ", pal.accent_purple))
                                    .on_hover_text("Copy TikZ/pgfplots \\addplot snippet")
                                    .clicked()
                                {
                                    let body_for_tikz = raw
                                        .split_once('=')
                                        .map(|(_, r)| r.trim())
                                        .unwrap_or(raw.as_str());
                                    let tikz_expr =
                                        self.substitute_params_for_tikz(body_for_tikz);
                                    let tikz = format!(
                                        "\\addplot[domain={:.2}:{:.2}, samples=200, thick] {{{}}};",
                                        self.domain_min, self.domain_max, tikz_expr
                                    );
                                    ui.ctx().copy_text(tikz);
                                    *status_msg = Some("Copied TikZ/pgfplots snippet.".to_string());
                                }
                            });
                            ui.add_space(3.0_f32);
                            let tex_edit_id =
                                ui.make_persistent_id(format!("tex_drawer_edit_{}", cell_id));
                            let tex_resp = ui.add(
                                egui::TextEdit::multiline(&mut latex_draft)
                                    .id(tex_edit_id)
                                    .font(FontId::monospace(11.3_f32))
                                    .desired_rows(2)
                                    .desired_width(f32::INFINITY)
                                    .hint_text(r"Type or paste LaTeX, e.g. \frac{-b + \sqrt{b^2 - 4ac}}{2a}"),
                            );
                            if tex_resp.changed() {
                                // Reverse-sync: parse LaTeX -> update symbolic expression & 2D graph immediately!
                                match math::latex_to_symbolic_from_ast(&latex_draft, &known_fns) {
                                    Ok(sym_str) => {
                                        raw = sym_str;
                                        raw_changed = true;
                                        latex_error = None;
                                    }
                                    Err(e) => {
                                        latex_error = Some(e.message);
                                    }
                                }
                            }
                            if let Some(ref le) = latex_error {
                                ui.label(
                                    RichText::new(format!("⚠ LaTeX: {}", le))
                                        .font(FontId::monospace(10.2_f32))
                                        .color(pal.accent_amber),
                                );
                            }
                        });
                }

                // 7. Expandable Full Curve Settings (Appearance + X/Y/t Domain Expressions + Presets)
                if style.settings_open && is_plottable_curve {
                    ui.add_space(4.0_f32);
                    egui::Frame::none()
                        .fill(pal.extreme_bg)
                        .stroke(Stroke::new(1.0_f32, pal.border))
                        .rounding(Rounding::same(6.0_f32))
                        .inner_margin(Margin::same(6.0_f32))
                        .show(ui, |ui| {
                            render_curve_appearance_controls(ui, &style_key, &mut style, pal);
                            ui.separator();
                            ui.label(
                                RichText::new("Domain X / Image Y / Parameter Restrictions")
                                    .font(FontId::proportional(11.2_f32))
                                    .strong()
                                    .color(pal.accent_blue),
                            );
                            ui.label(
                                RichText::new("Accepts math expressions: e.g. -2*pi, 2*pi, I / price_x")
                                    .font(FontId::proportional(10.0_f32))
                                    .color(pal.text_muted),
                            );
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("X:").strong().color(pal.accent_blue));
                                ui.add(
                                    egui::TextEdit::singleline(&mut style.custom_x_min_expr)
                                        .font(FontId::monospace(11.0_f32))
                                        .desired_width(64.0_f32)
                                        .hint_text("x_min"),
                                );
                                ui.label("≤ x ≤");
                                ui.add(
                                    egui::TextEdit::singleline(&mut style.custom_x_max_expr)
                                        .font(FontId::monospace(11.0_f32))
                                        .desired_width(64.0_f32)
                                        .hint_text("x_max"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Y:").strong().color(pal.accent_purple));
                                ui.add(
                                    egui::TextEdit::singleline(&mut style.custom_y_min_expr)
                                        .font(FontId::monospace(11.0_f32))
                                        .desired_width(64.0_f32)
                                        .hint_text("y_min"),
                                );
                                ui.label("≤ y ≤");
                                ui.add(
                                    egui::TextEdit::singleline(&mut style.custom_y_max_expr)
                                        .font(FontId::monospace(11.0_f32))
                                        .desired_width(64.0_f32)
                                        .hint_text("y_max"),
                                );
                            });
                            if matches!(
                                classified,
                                Some(PlottableEntry::Parametric { .. } | PlottableEntry::Polar { .. })
                            ) {
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new("t/θ:").strong().color(pal.accent_green));
                                    ui.add(
                                        egui::TextEdit::singleline(&mut style.custom_t_min_expr)
                                            .font(FontId::monospace(11.0_f32))
                                            .desired_width(64.0_f32)
                                            .hint_text("0"),
                                    );
                                    ui.label("..");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut style.custom_t_max_expr)
                                            .font(FontId::monospace(11.0_f32))
                                            .desired_width(64.0_f32)
                                            .hint_text("2*pi"),
                                    );
                                });
                            }
                            ui.add_space(3.0_f32);
                            ui.horizontal_wrapped(|ui| {
                                if ui.small_button("1st Quad (x≥0, y≥0)").clicked() {
                                    style.custom_x_min_expr = "0".to_string();
                                    style.custom_x_max_expr.clear();
                                    style.custom_y_min_expr = "0".to_string();
                                    style.custom_y_max_expr.clear();
                                }
                                if ui.small_button("[-10, 10]").clicked() {
                                    style.custom_x_min_expr = "-10".to_string();
                                    style.custom_x_max_expr = "10".to_string();
                                }
                                if ui.small_button("[0, 2π]").clicked() {
                                    style.custom_x_min_expr = "0".to_string();
                                    style.custom_x_max_expr = "2*pi".to_string();
                                    style.custom_t_min_expr = "0".to_string();
                                    style.custom_t_max_expr = "2*pi".to_string();
                                }
                                if ui.small_button("Auto-Fit Y").clicked() {
                                    if let Some(PlottableEntry::Explicit1D { ref compiled }) =
                                        classified
                                    {
                                        let ext = REngineMathBridge {
                                            r_objects: &self.workspace.r_objects,
                                        };
                                        let ctx = self.build_eval_context(&ext);
                                        let mut sc = EvalScratch::new();
                                        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                                        let (sx0, sx1) = self
                                            .resolve_style_x_domain(&style)
                                            .unwrap_or((self.domain_min, self.domain_max));
                                        for k in 0..=80 {
                                            let xv = sx0 + (sx1 - sx0) * (k as f64) / 80.0_f64;
                                            if let Some(yv) =
                                                math::eval_compiled_at_x(compiled, &ctx, xv, &mut sc)
                                            {
                                                if yv.abs() < 1e5_f64 {
                                                    lo = lo.min(yv);
                                                    hi = hi.max(yv);
                                                }
                                            }
                                        }
                                        if lo.is_finite() && hi.is_finite() {
                                            let pad = ((hi - lo) * 0.12_f64).max(1.0_f64);
                                            self.domain_y_min = lo - pad;
                                            self.domain_y_max = hi + pad;
                                            self.needs_plot_bounds_sync = true;
                                        }
                                    }
                                }
                                if ui.small_button("Clear Bounds").clicked() {
                                    style.custom_domain = None;
                                    style.custom_x_min_expr.clear();
                                    style.custom_x_max_expr.clear();
                                    style.custom_y_range = None;
                                    style.custom_y_min_expr.clear();
                                    style.custom_y_max_expr.clear();
                                }
                            });
                        });
                }
            });
        });

        let _ = duplicate_cell_id;
        // Write back updated state
        self.workspace.function_styles.insert(style_key, style);
        if let Some(cell) = self.workspace.cells.get_mut(idx) {
            if let SymbolicCellKind::Expression {
                raw: ref mut r_slot,
                latex_drawer_open: ref mut l_open,
                latex_draft: ref mut l_draft,
                latex_error: ref mut l_err,
                show_fraction: ref mut s_frac,
                plot_eval_point: ref mut p_pt,
                highlight_roots: ref mut h_rt,
                ..
            } = cell.kind
            {
                if raw_changed {
                    *need_undo_snapshot = true;
                    *r_slot = raw;
                    cells_to_sync.push(idx);
                }
                *l_open = latex_drawer_open;
                *l_draft = latex_draft;
                *l_err = latex_error;
                *s_frac = show_fraction;
                *p_pt = plot_eval_point;
                *h_rt = highlight_roots;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_symbolic_slider_cell(
        &mut self,
        ui: &mut Ui,
        idx: usize,
        display_index: usize,
        cell_id: u64,
        pal: ThemePalette,
        remove_cell_id: &mut Option<u64>,
        _duplicate_cell_id: &mut Option<u64>,
    ) {
        let param_name = match &self.workspace.cells[idx].kind {
            SymbolicCellKind::Slider { param_name } => param_name.clone(),
            _ => return,
        };
        let mut val = self
            .workspace
            .parameters
            .get(&param_name)
            .copied()
            .unwrap_or(1.0_f64);
        let mut spec = self
            .workspace
            .slider_specs
            .get(&param_name)
            .cloned()
            .unwrap_or_else(|| SliderSpec::for_parameter(&param_name, val));

        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0_f32;

            // ---- 36px Left Gutter: Index + Circular Play/Pause (`▶`/`⏸`) + Anim Mode (`⇄`) ----
            ui.allocate_ui_with_layout(
                Vec2::new(36.0_f32, 46.0_f32),
                egui::Layout::top_down(egui::Align::Center),
                |ui| {
                    ui.label(
                        RichText::new(format!("{}", display_index))
                            .font(FontId::monospace(9.8_f32))
                            .color(pal.text_muted),
                    );
                    let play_icon = if spec.playing { "⏸" } else { "▶" };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(play_icon)
                                    .font(FontId::proportional(11.0_f32))
                                    .color(if spec.playing {
                                        pal.accent_green
                                    } else {
                                        pal.accent_blue
                                    }),
                            )
                            .fill(pal.pill_bg)
                            .stroke(Stroke::new(
                                1.0_f32,
                                if spec.playing {
                                    pal.accent_green
                                } else {
                                    pal.border
                                },
                            ))
                            .rounding(Rounding::same(11.0_f32))
                            .min_size(Vec2::splat(22.0_f32)),
                        )
                        .on_hover_text("Play / Pause parameter animation (60 FPS)")
                        .clicked()
                    {
                        spec.playing = !spec.playing;
                    }
                    let anim_pop_id = ui.make_persistent_id(format!("slider_anim_pop_{}", cell_id));
                    let anim_btn = ui
                        .add(header_icon_button(spec.anim_mode.icon(), pal.text_secondary))
                        .on_hover_text(format!(
                            "Animation Mode: {} ({:.2}x)",
                            spec.anim_mode.label(),
                            spec.anim_speed
                        ));
                    if anim_btn.clicked() {
                        ui.memory_mut(|m| m.toggle_popup(anim_pop_id));
                    }
                    egui::popup_below_widget(
                        ui,
                        anim_pop_id,
                        &anim_btn,
                        egui::PopupCloseBehavior::CloseOnClickOutside,
                        |ui| {
                            render_slider_anim_controls(ui, &mut spec, pal);
                        },
                    );
                },
            );

            // ---- Right Slider Body ----
            ui.vertical(|ui| {
                ui.set_min_width((ui.available_width() - 4.0_f32).max(120.0_f32));

                // Top row: Typographic identifier (`P_x`, `\bar{Y}`, `α`) + `=` + DragValue + `×`
                ui.horizontal(|ui| {
                    render_math_ident_inline(ui, &param_name, 14.2_f32, pal.text_primary);
                    ui.label(RichText::new("=").color(pal.text_secondary));
                    if ui
                        .add(
                            egui::DragValue::new(&mut val)
                                .speed(spec.step.max(1e-4_f64)),
                        )
                        .changed()
                    {
                        self.workspace.parameters.insert(param_name.clone(), val);
                        self.workspace
                            .r_objects
                            .insert(param_name.clone(), RObject::scalar(val));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(header_icon_button("×", pal.accent_red))
                            .on_hover_text("Remove parameter slider")
                            .clicked()
                        {
                            *remove_cell_id = Some(cell_id);
                        }
                        if spec.playing {
                            ui.label(
                                RichText::new(format!("{:.2}x", spec.anim_speed).trim_end_matches('0').trim_end_matches('.'))
                                    .font(FontId::monospace(10.0_f32))
                                    .color(pal.accent_green),
                            );
                        }
                    });
                });

                // Slider bar with clickable `min` and `max` limits directly at left & right ends
                let safe_min = spec.min.min(spec.max - 1e-6_f64);
                let safe_max = spec.max.max(safe_min + 1e-6_f64);
                let min_lbl = Self::format_scalar_result(spec.min, false).0;
                let max_lbl = Self::format_scalar_result(spec.max, false).0;

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0_f32;
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(min_lbl)
                                    .font(FontId::monospace(10.5_f32))
                                    .color(pal.text_muted),
                            )
                            .fill(Color32::TRANSPARENT)
                            .stroke(Stroke::NONE),
                        )
                        .on_hover_text("Click to edit slider min, max & step")
                        .clicked()
                    {
                        spec.edit_bounds_open = !spec.edit_bounds_open;
                    }

                    ui.spacing_mut().slider_width = (ui.available_width() - 44.0_f32).max(52.0_f32);
                    let mut slider_ctrl =
                        egui::Slider::new(&mut val, safe_min..=safe_max).show_value(false);
                    if spec.step > 0.0_f64 {
                        slider_ctrl = slider_ctrl.step_by(spec.step);
                    }
                    if ui.add(slider_ctrl).changed() {
                        self.workspace.parameters.insert(param_name.clone(), val);
                        self.workspace
                            .r_objects
                            .insert(param_name.clone(), RObject::scalar(val));
                    }

                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(max_lbl)
                                    .font(FontId::monospace(10.5_f32))
                                    .color(pal.text_muted),
                            )
                            .fill(Color32::TRANSPARENT)
                            .stroke(Stroke::NONE),
                        )
                        .on_hover_text("Click to edit slider min, max & step")
                        .clicked()
                    {
                        spec.edit_bounds_open = !spec.edit_bounds_open;
                    }
                });

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
        });

        self.workspace.slider_specs.insert(param_name, spec);
    }

    #[allow(clippy::too_many_arguments)]
    fn render_symbolic_folder_cell(
        &mut self,
        ui: &mut Ui,
        idx: usize,
        display_index: usize,
        cell_id: u64,
        pal: ThemePalette,
        remove_cell_id: &mut Option<u64>,
        _duplicate_cell_id: &mut Option<u64>,
    ) {
        let SymbolicCellKind::Folder {
            ref mut title,
            ref mut collapsed,
            ref mut visible,
        } = self.workspace.cells[idx].kind
        else {
            return;
        };

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0_f32;
            ui.allocate_ui_with_layout(
                Vec2::new(36.0_f32, 26.0_f32),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.label(
                        RichText::new(format!("{}", display_index))
                            .font(FontId::monospace(9.8_f32))
                            .color(pal.text_muted),
                    );
                    if ui
                        .add(header_icon_button(
                            if *collapsed { "▶" } else { "▼" },
                            pal.text_secondary,
                        ))
                        .on_hover_text("Collapse / expand folder contents")
                        .clicked()
                    {
                        *collapsed = !*collapsed;
                    }
                },
            );

            if ui
                .add(header_icon_button(
                    if *visible { "▣" } else { "▢" },
                    if *visible {
                        pal.accent_blue
                    } else {
                        pal.text_muted
                    },
                ))
                .on_hover_text("Toggle visibility of all curves in this folder")
                .clicked()
            {
                *visible = !*visible;
            }

            let resp = ui.add(
                egui::TextEdit::singleline(title)
                    .font(FontId::proportional(12.6_f32))
                    .frame(false)
                    .desired_width((ui.available_width() - 32.0_f32).max(80.0_f32))
                    .hint_text("Folder name..."),
            );
            if resp.clicked() || resp.gained_focus() {
                self.selected_cell_id = Some(cell_id);
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add(header_icon_button("×", pal.accent_red))
                    .on_hover_text("Delete folder (keeps children)")
                    .clicked()
                {
                    *remove_cell_id = Some(cell_id);
                }
            });
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn render_symbolic_note_cell(
        &mut self,
        ui: &mut Ui,
        idx: usize,
        display_index: usize,
        cell_id: u64,
        pal: ThemePalette,
        remove_cell_id: &mut Option<u64>,
        _duplicate_cell_id: &mut Option<u64>,
    ) {
        let SymbolicCellKind::Note { ref mut text } = self.workspace.cells[idx].kind else {
            return;
        };
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                Vec2::new(36.0_f32, 28.0_f32),
                egui::Layout::top_down(egui::Align::Center),
                |ui| {
                    ui.label(
                        RichText::new(format!("{}", display_index))
                            .font(FontId::monospace(9.8_f32))
                            .color(pal.text_muted),
                    );
                    ui.label(
                        RichText::new("\"\"")
                            .font(FontId::monospace(12.0_f32))
                            .strong()
                            .color(pal.accent_amber),
                    );
                },
            );
            ui.add(
                egui::TextEdit::multiline(text)
                    .font(FontId::proportional(12.0_f32))
                    .desired_rows(2)
                    .desired_width((ui.available_width() - 28.0_f32).max(80.0_f32))
                    .hint_text("Add economic commentary or derivation note..."),
            );
            if ui
                .add(header_icon_button("×", pal.accent_red))
                .on_hover_text("Delete note")
                .clicked()
            {
                *remove_cell_id = Some(cell_id);
            }
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn render_symbolic_table_cell(
        &mut self,
        ui: &mut Ui,
        idx: usize,
        display_index: usize,
        cell_id: u64,
        pal: ThemePalette,
        remove_cell_id: &mut Option<u64>,
        sync_tables: &mut Vec<(String, Vec<f64>, String)>,
    ) {
        let (fn_sig, mut x_values, mut show_derivative, dataset_name) =
            match &self.workspace.cells[idx].kind {
                SymbolicCellKind::Table {
                    fn_sig,
                    x_values,
                    show_derivative,
                    dataset_name,
                } => (
                    fn_sig.clone(),
                    x_values.clone(),
                    *show_derivative,
                    dataset_name.clone(),
                ),
                _ => return,
            };

        let expr_opt = self.workspace.functions.get(&fn_sig).cloned();
        let mut changed = false;

        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                Vec2::new(36.0_f32, 34.0_f32),
                egui::Layout::top_down(egui::Align::Center),
                |ui| {
                    ui.label(
                        RichText::new(format!("{}", display_index))
                            .font(FontId::monospace(9.8_f32))
                            .color(pal.text_muted),
                    );
                    ui.label(
                        RichText::new("⊞")
                            .font(FontId::proportional(14.0_f32))
                            .color(pal.accent_blue),
                    );
                },
            );

            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("Table: {} → {}", fn_sig, dataset_name))
                            .font(FontId::monospace(11.2_f32))
                            .strong()
                            .color(pal.accent_blue),
                    );
                    ui.checkbox(&mut show_derivative, "f'(x)");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(header_icon_button("×", pal.accent_red))
                            .on_hover_text("Delete table cell")
                            .clicked()
                        {
                            *remove_cell_id = Some(cell_id);
                        }
                        if ui
                            .add(header_icon_button("+ Row", pal.accent_green))
                            .on_hover_text("Append x value row")
                            .clicked()
                        {
                            let next_x = x_values
                                .last()
                                .copied()
                                .map(|v| v + 1.0_f64)
                                .unwrap_or(0.0_f64);
                            x_values.push(next_x);
                            changed = true;
                        }
                    });
                });

                egui::Grid::new(format!("sym_tbl_grid_{}", cell_id))
                    .striped(true)
                    .spacing([10.0_f32, 3.0_f32])
                    .show(ui, |ui| {
                        ui.label(RichText::new("x").strong().color(pal.text_secondary));
                        ui.label(RichText::new(&fn_sig).strong().color(pal.accent_green));
                        if show_derivative {
                            ui.label(RichText::new("f'(x)").strong().color(pal.accent_purple));
                        }
                        ui.end_row();

                        for xv in &mut x_values {
                            if ui.add(egui::DragValue::new(xv).speed(0.1_f64)).changed() {
                                changed = true;
                            }
                            if let Some(ref expr) = expr_opt {
                                let y_txt = self
                                    .eval_expression_str(expr, *xv, 1.0_f64)
                                    .ok()
                                    .filter(|v| v.is_finite())
                                    .map(|v| format!("{:.4}", v))
                                    .unwrap_or_else(|| "—".to_string());
                                ui.monospace(RichText::new(y_txt).color(pal.text_primary));
                                if show_derivative {
                                    let dy_txt = self
                                        .numerical_derivative(expr, *xv)
                                        .ok()
                                        .filter(|v| v.is_finite())
                                        .map(|v| format!("{:.4}", v))
                                        .unwrap_or_else(|| "—".to_string());
                                    ui.monospace(RichText::new(dy_txt).color(pal.accent_purple));
                                }
                            } else {
                                ui.monospace("—");
                                if show_derivative {
                                    ui.monospace("—");
                                }
                            }
                            ui.end_row();
                        }
                    });
            });
        });

        if let Some(cell) = self.workspace.cells.get_mut(idx) {
            if let SymbolicCellKind::Table {
                x_values: ref mut xs,
                show_derivative: ref mut sd,
                ..
            } = cell.kind
            {
                *xs = x_values.clone();
                *sd = show_derivative;
            }
        }
        if changed {
            sync_tables.push((fn_sig, x_values, dataset_name));
        }
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
                                    RichText::new("⊳ Export Bundle & Open Positron")
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
                                            render_math_ident_inline(
                                                ui,
                                                &k,
                                                13.0_f32,
                                                pal.text_primary,
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
                                                    .add(header_icon_button("◿", pal.accent_green))
                                                    .on_hover_text("Plot vector on 2D Graph")
                                                    .clicked()
                                            {
                                                run_console_cmd = Some(format!("plot({})", k));
                                            }
                                            if obj.is_lm_model() {
                                                if ui
                                                    .add(header_icon_button("◿", pal.accent_green))
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
                                                        "◿ Plot {}${} vs {}${}",
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
                                                    if ui.small_button("◿").clicked() {
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
                    .add(header_icon_button("↺", pal.text_secondary))
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
                                let icon = if entry.is_dir { "▣" } else { "§" };
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
                    DataTable::from_csv_str(file_stem, content, Color32::from_rgb(72, 118, 168));
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
                            ViewportMode::Plot2D => "◿ 2D Graph ⌄",
                            ViewportMode::Surface3D => "◈ 3D Surface ⌄",
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
                                        "◿ 2D Symbolic Graph",
                                    )
                                    .clicked()
                                {
                                    self.viewport_mode = ViewportMode::Plot2D;
                                    ui.close_menu();
                                }
                                if ui
                                    .selectable_label(
                                        self.viewport_mode == ViewportMode::Surface3D,
                                        "◈ 3D Isometric Surface",
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
                            .add(pill_button("⌂", false, pal))
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

                        // Right: `†` Freeze Baseline + `⋮` quick overlays popover + `⊞` Maximize Graph
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.spacing_mut().item_spacing.x = 4.0_f32;
                            if ui
                                .add(pill_button("⊞", self.maximize_graph, pal))
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
                            )
                            .response
                            .on_hover_text("Graph Axes, Grid Steps & Overlays");
                            if self.viewport_mode == ViewportMode::Plot2D
                                && ui
                                    .add(pill_button("†", self.frozen_baseline.is_some(), pal))
                                    .on_hover_text(if self.frozen_baseline.is_some() {
                                        "Click to clear frozen comparative-statics baseline"
                                    } else {
                                        "Freeze Baseline curves & equilibrium for comparative statics (Δx*, Δy*)"
                                    })
                                    .clicked()
                            {
                                if self.frozen_baseline.is_some() {
                                    self.frozen_baseline = None;
                                    self.status_banner =
                                        "Cleared comparative-statics baseline.".to_string();
                                } else {
                                    self.freeze_current_baseline();
                                }
                            }
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

    fn clip_branches_to_y_range(
        branches: Vec<Vec<[f64; 2]>>,
        y_range: Option<(f64, f64)>,
    ) -> Vec<Vec<[f64; 2]>> {
        let Some((y_min, y_max)) = y_range else {
            return branches;
        };
        let mut out = Vec::new();
        for branch in branches {
            let mut cur = Vec::new();
            for pt in branch {
                if pt[1] >= y_min && pt[1] <= y_max {
                    cur.push(pt);
                } else if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            if !cur.is_empty() {
                out.push(cur);
            }
        }
        out
    }

    fn uniform_grid_marks(input: GridInput, step: f64) -> Vec<GridMark> {
        let mut marks = Vec::new();
        if !step.is_finite() || step <= 1e-9_f64 {
            return marks;
        }
        let (lo, hi) = input.bounds;
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            return marks;
        }
        // Prevent excessive mark counts when zoomed very far out
        let mut eff_step = step;
        while (hi - lo) / eff_step > 80.0_f64 {
            eff_step *= 2.0_f64;
        }
        let first = (lo / eff_step).floor() as i64;
        let last = (hi / eff_step).ceil() as i64;
        for k in first..=last {
            marks.push(GridMark {
                value: (k as f64) * eff_step,
                step_size: eff_step,
            });
        }
        marks
    }

    fn render_2d_desmos_plot(&mut self, ui: &mut Ui, pal: ThemePalette) {
        let modifiers = ui.input(|i| i.modifiers);
        let shift_held = modifiers.shift;
        let allow_zoom_xy = if shift_held && !modifiers.ctrl {
            [true, false]
        } else if modifiers.ctrl && !shift_held {
            [false, true]
        } else {
            match self.axis_zoom_mode {
                AxisZoomMode::BothXY => [true, true],
                AxisZoomMode::XOnly => [true, false],
                AxisZoomMode::YOnly => [false, true],
            }
        };

        // Disable default plot pan while Shift is held so Shift + Click & Drag scales axes cleanly
        let allow_drag = if shift_held || self.shift_drag.active_zone.is_some() {
            [false, false]
        } else {
            [true, true]
        };

        let mut plot = Plot::new("desmos_2d_symbolic_plot")
            .allow_zoom(allow_zoom_xy)
            .allow_drag(allow_drag)
            .allow_scroll(
                !shift_held
                    && !modifiers.ctrl
                    && self.axis_zoom_mode == AxisZoomMode::BothXY,
            )
            .allow_boxed_zoom(!shift_held)
            .show_grid(self.show_plot_grid);

        if !self.axis_config.x_label.trim().is_empty() {
            plot = plot.x_axis_label(self.axis_config.x_label.clone());
        }
        if !self.axis_config.y_label.trim().is_empty() {
            plot = plot.y_axis_label(self.axis_config.y_label.clone());
        }
        if let Some(xs) = self.axis_config.x_step {
            plot = plot.x_grid_spacer(move |inp| Self::uniform_grid_marks(inp, xs));
        }
        if let Some(ys) = self.axis_config.y_step {
            plot = plot.y_grid_spacer(move |inp| Self::uniform_grid_marks(inp, ys));
        }
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
                // ---- 1. Spatial Zone Detection & Shift + Click & Drag Axis Scaling ----
                let frame_rect = *plot_ui.transform().frame();
                let origin_screen = plot_ui.screen_from_plot(PlotPoint::new(0.0_f64, 0.0_f64));
                let (hover_pos, primary_pressed, primary_down, primary_released, pointer_delta) =
                    plot_ui.ctx().input(|i| {
                        (
                            i.pointer.hover_pos(),
                            i.pointer.primary_pressed(),
                            i.pointer.primary_down(),
                            i.pointer.primary_released(),
                            i.pointer.delta(),
                        )
                    });

                let detected_zone = hover_pos.and_then(|pos| {
                    if !frame_rect.expand(20.0_f32).contains(pos) {
                        return None;
                    }
                    let dist_origin_y = if frame_rect.y_range().contains(origin_screen.y) {
                        (pos.y - origin_screen.y).abs()
                    } else {
                        f32::INFINITY
                    };
                    let dist_bottom_edge = (frame_rect.bottom() - pos.y).abs();
                    let on_x_axis = dist_origin_y <= 18.0_f32 || dist_bottom_edge <= 24.0_f32;

                    let dist_origin_x = if frame_rect.x_range().contains(origin_screen.x) {
                        (pos.x - origin_screen.x).abs()
                    } else {
                        f32::INFINITY
                    };
                    let dist_left_edge = (pos.x - frame_rect.left()).abs();
                    let on_y_axis = dist_origin_x <= 18.0_f32 || dist_left_edge <= 24.0_f32;

                    if on_x_axis && on_y_axis {
                        if dist_origin_x <= 8.0_f32 && dist_origin_y <= 8.0_f32 {
                            Some(ShiftDragZone::BothXY)
                        } else if dist_origin_y.min(dist_bottom_edge)
                            <= dist_origin_x.min(dist_left_edge)
                        {
                            Some(ShiftDragZone::XAxis)
                        } else {
                            Some(ShiftDragZone::YAxis)
                        }
                    } else if on_x_axis {
                        Some(ShiftDragZone::XAxis)
                    } else if on_y_axis {
                        Some(ShiftDragZone::YAxis)
                    } else if frame_rect.contains(pos) {
                        Some(ShiftDragZone::BothXY)
                    } else {
                        None
                    }
                });

                if !shift_held || primary_released {
                    self.shift_drag.active_zone = None;
                }
                if shift_held {
                    self.shift_drag.hovered_zone = detected_zone;
                    if primary_pressed {
                        if let Some(zone) = detected_zone {
                            self.shift_drag.active_zone = Some(zone);
                        }
                    }
                    let visual_zone = self.shift_drag.active_zone.or(self.shift_drag.hovered_zone);
                    if let Some(z) = visual_zone {
                        let cursor = match z {
                            ShiftDragZone::XAxis => egui::CursorIcon::ResizeHorizontal,
                            ShiftDragZone::YAxis => egui::CursorIcon::ResizeVertical,
                            ShiftDragZone::BothXY => egui::CursorIcon::ResizeNwSe,
                        };
                        plot_ui.ctx().set_cursor_icon(cursor);
                    }
                } else {
                    self.shift_drag.hovered_zone = None;
                }

                // ---- 2. Bounds Synchronization, Button Zoom & Shift+Drag Axis Zoom ----
                if needs_sync {
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                        [self.domain_min, self.domain_y_min],
                        [self.domain_max, self.domain_y_max],
                    ));
                } else if shift_held && primary_down && self.shift_drag.active_zone.is_some() {
                    if let Some(zone) = self.shift_drag.active_zone {
                        let b = plot_ui.plot_bounds();
                        let [mut bx0, mut by0] = b.min();
                        let [mut bx1, mut by1] = b.max();
                        let dx = pointer_delta.x as f64;
                        let dy = pointer_delta.y as f64;

                        let scale_axis = |a0: f64, a1: f64, factor: f64| -> (f64, f64) {
                            let anchor = if a0 < 0.0_f64 && a1 > 0.0_f64 {
                                0.0_f64
                            } else {
                                0.5_f64 * (a0 + a1)
                            };
                            let n0 = anchor + (a0 - anchor) * factor;
                            let n1 = anchor + (a1 - anchor) * factor;
                            if (n1 - n0).abs() > 1e-5_f64 && (n1 - n0).abs() < 1e9_f64 {
                                (n0, n1)
                            } else {
                                (a0, a1)
                            }
                        };

                        match zone {
                            ShiftDragZone::XAxis => {
                                if dx.abs() > 0.01_f64 {
                                    let fx = (-dx * 0.006_f64).exp();
                                    (bx0, bx1) = scale_axis(bx0, bx1, fx);
                                }
                            }
                            ShiftDragZone::YAxis => {
                                if dy.abs() > 0.01_f64 {
                                    let fy = (dy * 0.006_f64).exp();
                                    (by0, by1) = scale_axis(by0, by1, fy);
                                }
                            }
                            ShiftDragZone::BothXY => {
                                if dx.abs() > 0.01_f64 {
                                    let fx = (-dx * 0.0055_f64).exp();
                                    (bx0, bx1) = scale_axis(bx0, bx1, fx);
                                }
                                if dy.abs() > 0.01_f64 {
                                    let fy = (dy * 0.0055_f64).exp();
                                    (by0, by1) = scale_axis(by0, by1, fy);
                                }
                            }
                        }

                        plot_ui.set_plot_bounds(PlotBounds::from_min_max([bx0, by0], [bx1, by1]));
                        self.domain_min = bx0;
                        self.domain_max = bx1;
                        self.domain_y_min = by0;
                        self.domain_y_max = by1;
                    }
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

                // Highlight principal x = 0 and y = 0 axes (illuminating cyan during Shift axis hover/drag)
                let active_or_hover_zone =
                    self.shift_drag.active_zone.or(self.shift_drag.hovered_zone);
                let hi_x = matches!(
                    active_or_hover_zone,
                    Some(ShiftDragZone::XAxis | ShiftDragZone::BothXY)
                );
                let hi_y = matches!(
                    active_or_hover_zone,
                    Some(ShiftDragZone::YAxis | ShiftDragZone::BothXY)
                );
                draw_principal_axes_highlighted(plot_ui, pal, hi_x, hi_y);

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

                // ---- 3. Render Frozen Comparative-Statics Ghost Baseline (`📌`) ----
                if let Some(ref base) = self.frozen_baseline {
                    for fc in &base.curves {
                        for branch in &fc.branches {
                            plot_ui.line(
                                Line::new(PlotPoints::from(branch.clone()))
                                    .name(&fc.label)
                                    .width(1.6_f32)
                                    .style(LineStyle::Dashed { length: 6.0_f32 })
                                    .color(fc.color.gamma_multiply(0.42_f32)),
                            );
                        }
                    }
                    if let Some((bx, by)) = base.equilibrium {
                        plot_ui.points(
                            Points::new(vec![[bx, by]])
                                .name(format!("Baseline E₀ ({:.2}, {:.2})", bx, by))
                                .radius(5.5_f32)
                                .shape(MarkerShape::Diamond)
                                .color(pal.text_muted),
                        );
                        if let Some((nx, ny)) = self.find_primary_secondary_intersection() {
                            let dx_eq = nx - bx;
                            let dy_eq = ny - by;
                            if dx_eq.hypot(dy_eq) > 1e-4_f64 {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(vec![[bx, by], [nx, ny]]))
                                        .width(1.5_f32)
                                        .style(LineStyle::Dotted { spacing: 4.0_f32 })
                                        .color(pal.accent_amber),
                                );
                                plot_ui.text(PlotText::new(
                                    PlotPoint::new(
                                        0.5_f64 * (bx + nx),
                                        0.5_f64 * (by + ny),
                                    ),
                                    RichText::new(format!(
                                        "Δx* = {:+.3}, Δy* = {:+.3}",
                                        dx_eq, dy_eq
                                    ))
                                    .font(FontId::monospace(11.0_f32))
                                    .strong()
                                    .color(pal.accent_amber),
                                ));
                            }
                        }
                    }
                }

                // ---- 4. Render Active Symbolic Curves, Inequalities, Contours, Tangents, Surplus & Roots ----
                let ext_bridge = REngineMathBridge {
                    r_objects: &self.workspace.r_objects,
                };
                let eval_ctx = self.build_eval_context(&ext_bridge);

                // Collect compiled 1D curves for automatic POI detection on focused curve
                let mut visible_1d_curves: Vec<(String, CompiledExpr, Color32, bool)> = Vec::new();
                let selected_fn_key: Option<String> = self.selected_cell_id.and_then(|sid| {
                    self.workspace.cells.iter().find_map(|c| {
                        if c.id == sid {
                            if let SymbolicCellKind::Expression { ref bound_key, .. } = c.kind {
                                return bound_key.clone();
                            }
                        }
                        None
                    })
                });

                for (idx, (sig, expr)) in self.workspace.functions.iter().enumerate() {
                    if !self.workspace.key_is_effectively_visible(sig) {
                        continue;
                    }
                    let style = self
                        .workspace
                        .function_styles
                        .get(sig)
                        .cloned()
                        .unwrap_or_else(|| FunctionStyle::default_for_index(idx));
                    if style.role == FunctionPlotRole::Surface3DOnly {
                        continue;
                    }

                    // Look up cell flags (`plot_eval_point`, `highlight_roots`)
                    let (plot_eval_pt, highlight_rts) = self
                        .workspace
                        .cells
                        .iter()
                        .find_map(|c| {
                            if let SymbolicCellKind::Expression {
                                bound_key: Some(ref bk),
                                plot_eval_point,
                                highlight_roots,
                                ..
                            } = c.kind
                            {
                                if bk == sig {
                                    return Some((plot_eval_point, highlight_roots));
                                }
                            }
                            None
                        })
                        .unwrap_or((false, true));

                    let series_label = if sig.starts_with("__cell_") {
                        Self::format_symbolic_math(expr)
                    } else {
                        format!("{} = {}", Self::format_symbolic_math(sig), expr)
                    };

                    // Live R scatter/line layer when referencing R vectors, `df$col`, `(df$x, df$y)`, or `predict(fit)`
                    if let Some(r_pts) =
                        crate::r_engine::try_extract_r_plot_series(expr, &self.workspace.r_objects)
                    {
                        if style.show_lines {
                            draw_glowing_plot_line_with_opacity(
                                plot_ui,
                                r_pts.clone(),
                                &series_label,
                                style.color,
                                style.stroke_width,
                                style.line_style.to_egui(),
                                style.opacity,
                            );
                        }
                        plot_ui.points(
                            Points::new(PlotPoints::from(r_pts))
                                .radius(style.stroke_width + 2.0_f32)
                                .color(style.color.gamma_multiply(style.opacity)),
                        );
                        continue;
                    }

                    let (s_min, s_max) = if let Some((cmin, cmax)) = self.resolve_style_x_domain(&style) {
                        (cmin.max(global_sample_min), cmax.min(global_sample_max))
                    } else {
                        (global_sample_min, global_sample_max)
                    };
                    let y_range = self.resolve_style_y_range(&style);

                    let Ok(plottable) = self.workspace.compiled_math.classify_entry(sig, expr)
                    else {
                        continue;
                    };

                    match plottable {
                        PlottableEntry::Explicit1D { compiled } => {
                            let is_focus = selected_fn_key.as_deref() == Some(sig.as_str());
                            let short_name = if sig.starts_with("__cell_") {
                                expr.clone()
                            } else {
                                sig.clone()
                            };
                            visible_1d_curves.push((short_name, compiled.clone(), style.color, is_focus));

                            if s_max <= s_min {
                                continue;
                            }
                            let raw_branches = math::sample_1d_branches(
                                &compiled,
                                &eval_ctx,
                                s_min,
                                s_max,
                                vis_min_y,
                                vis_max_y,
                                self.curve_samples,
                                self.asymptote_jump_factor,
                            );
                            let branches = Self::clip_branches_to_y_range(raw_branches, y_range);
                            match style.role {
                                FunctionPlotRole::StandardCurve => {
                                    for branch in branches {
                                        if style.show_lines {
                                            draw_glowing_plot_line_with_opacity(
                                                plot_ui,
                                                branch.clone(),
                                                &series_label,
                                                style.color,
                                                style.stroke_width,
                                                style.line_style.to_egui(),
                                                style.opacity,
                                            );
                                        }
                                        if style.show_points {
                                            let stride = (branch.len() / 24).max(1);
                                            let pts: Vec<[f64; 2]> =
                                                branch.into_iter().step_by(stride).collect();
                                            plot_ui.points(
                                                Points::new(PlotPoints::from(pts))
                                                    .radius(style.stroke_width + 1.4_f32)
                                                    .color(style.color.gamma_multiply(style.opacity)),
                                            );
                                        }
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
                                                        style.color.gamma_multiply(0.22_f32 * style.opacity),
                                                    )
                                                    .stroke(Stroke::new(
                                                        style.stroke_width,
                                                        style.color.gamma_multiply(style.opacity),
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
                                                .color(style.color.gamma_multiply(style.opacity)),
                                        );
                                    }
                                }
                                FunctionPlotRole::Surface3DOnly => {}
                            }
                        }
                        PlottableEntry::Equation1D {
                            diff_compiled,
                            lhs_compiled,
                            is_fn_intersection,
                        } => {
                            if highlight_rts {
                                let roots = math::solve_compiled_roots_1d(
                                    &diff_compiled,
                                    &eval_ctx,
                                    global_sample_min,
                                    global_sample_max,
                                    320,
                                );
                                let mut sc = EvalScratch::new();
                                for rx in roots {
                                    plot_ui.vline(
                                        VLine::new(rx)
                                            .name(format!("{} (x={:.3})", series_label, rx))
                                            .width(style.stroke_width.min(2.2_f32))
                                            .style(LineStyle::Dashed { length: 7.0_f32 })
                                            .color(style.color.gamma_multiply(style.opacity)),
                                    );
                                    let ry = if is_fn_intersection {
                                        math::eval_compiled_at_x(&lhs_compiled, &eval_ctx, rx, &mut sc)
                                            .unwrap_or(0.0_f64)
                                    } else {
                                        0.0_f64
                                    };
                                    plot_ui.points(
                                        Points::new(vec![[rx, ry]])
                                            .radius(5.8_f32)
                                            .shape(MarkerShape::Circle)
                                            .color(style.color),
                                    );
                                    plot_ui.text(PlotText::new(
                                        PlotPoint::new(rx, ry),
                                        RichText::new(format!("({:.3}, {:.3})", rx, ry))
                                            .font(FontId::monospace(10.8_f32))
                                            .color(style.color),
                                    ));
                                }
                            }
                        }
                        PlottableEntry::TangentLine {
                            fn_name,
                            x0_compiled,
                            line_compiled,
                        } => {
                            let branches = math::sample_1d_branches(
                                &line_compiled,
                                &eval_ctx,
                                global_sample_min,
                                global_sample_max,
                                vis_min_y,
                                vis_max_y,
                                120,
                                self.asymptote_jump_factor,
                            );
                            for branch in branches {
                                if style.show_lines {
                                    draw_glowing_plot_line_with_opacity(
                                        plot_ui,
                                        branch,
                                        &series_label,
                                        style.color,
                                        style.stroke_width,
                                        LineStyle::Dashed { length: 7.0_f32 },
                                        style.opacity,
                                    );
                                }
                            }
                            let mut sc = EvalScratch::new();
                            if let Ok(x0) = x0_compiled.eval(&eval_ctx, &[], &mut sc) {
                                if let Ok((y0, _)) = math::eval_fn_val_and_deriv(
                                    &fn_name,
                                    x0,
                                    math::Span::default(),
                                    &eval_ctx,
                                    0,
                                ) {
                                    plot_ui.points(
                                        Points::new(vec![[x0, y0]])
                                            .name(format!("Tangency ({:.2}, {:.2})", x0, y0))
                                            .radius(5.5_f32)
                                            .shape(MarkerShape::Circle)
                                            .color(style.color),
                                    );
                                }
                            }
                        }
                        PlottableEntry::AreaBetween {
                            f_name,
                            g_name,
                            a_compiled,
                            b_compiled,
                            area_compiled,
                        } => {
                            let mut sc = EvalScratch::new();
                            let bounds_opt = if let (Some(ac), Some(bc)) = (a_compiled, b_compiled)
                            {
                                match (
                                    ac.eval(&eval_ctx, &[], &mut sc),
                                    bc.eval(&eval_ctx, &[], &mut sc),
                                ) {
                                    (Ok(a), Ok(b)) if a.is_finite() && b.is_finite() => {
                                        Some((a.min(b), a.max(b)))
                                    }
                                    _ => None,
                                }
                            } else {
                                math::find_surplus_equilibrium_x(
                                    &f_name,
                                    &g_name,
                                    &eval_ctx,
                                    math::Span::default(),
                                    0,
                                )
                                .ok()
                                .map(|eq_x| (0.0_f64, eq_x))
                            };

                            if let Some((a, b)) = bounds_opt {
                                if b > a + 1e-6_f64 {
                                    let n = 80usize;
                                    let mut upper_pts = Vec::with_capacity(n + 1);
                                    let mut lower_pts = Vec::with_capacity(n + 1);
                                    for i in 0..=n {
                                        let x = a + (b - a) * (i as f64) / (n as f64);
                                        if let (Ok((yf, _)), Ok((yg, _))) = (
                                            math::eval_fn_val_and_deriv(
                                                &f_name,
                                                x,
                                                math::Span::default(),
                                                &eval_ctx,
                                                0,
                                            ),
                                            math::eval_fn_val_and_deriv(
                                                &g_name,
                                                x,
                                                math::Span::default(),
                                                &eval_ctx,
                                                0,
                                            ),
                                        ) {
                                            if yf.is_finite() && yg.is_finite() {
                                                upper_pts.push([x, yf]);
                                                lower_pts.push([x, yg]);
                                            }
                                        }
                                    }
                                    if upper_pts.len() >= 2 && lower_pts.len() >= 2 {
                                        let mut poly = upper_pts;
                                        lower_pts.reverse();
                                        poly.extend(lower_pts);
                                        let area_val = area_compiled
                                            .eval(&eval_ctx, &[], &mut sc)
                                            .unwrap_or(f64::NAN);
                                        let lbl = if area_val.is_finite() {
                                            format!("{} = {:.3}", series_label, area_val)
                                        } else {
                                            series_label.clone()
                                        };
                                        plot_ui.polygon(
                                            Polygon::new(PlotPoints::from(poly))
                                                .name(lbl)
                                                .fill_color(
                                                    style.color.gamma_multiply(0.24_f32 * style.opacity),
                                                )
                                                .stroke(Stroke::new(
                                                    1.4_f32,
                                                    style.color.gamma_multiply(style.opacity),
                                                )),
                                        );
                                    }
                                }
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
                                if style.show_lines {
                                    draw_glowing_plot_line_with_opacity(
                                        plot_ui,
                                        chain,
                                        &series_label,
                                        style.color,
                                        style.stroke_width,
                                        style.line_style.to_egui(),
                                        style.opacity,
                                    );
                                }
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
                                        .fill_color(style.color.gamma_multiply(0.18_f32 * style.opacity))
                                        .stroke(Stroke::NONE),
                                );
                            }
                            let bound_style = if region.strict {
                                LineStyle::Dashed { length: 7.0_f32 }
                            } else {
                                style.line_style.to_egui()
                            };
                            for branch in region.boundary_curves {
                                if style.show_lines {
                                    draw_glowing_plot_line_with_opacity(
                                        plot_ui,
                                        branch,
                                        &series_label,
                                        style.color,
                                        style.stroke_width,
                                        bound_style,
                                        style.opacity,
                                    );
                                }
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
                                        .fill_color(style.color.gamma_multiply(0.18_f32 * style.opacity))
                                        .stroke(Stroke::NONE),
                                );
                            }
                            let bound_style = if region.strict {
                                LineStyle::Dashed { length: 7.0_f32 }
                            } else {
                                style.line_style.to_egui()
                            };
                            for chain in region.boundary_curves {
                                if style.show_lines {
                                    draw_glowing_plot_line_with_opacity(
                                        plot_ui,
                                        chain,
                                        &series_label,
                                        style.color,
                                        style.stroke_width,
                                        bound_style,
                                        style.opacity,
                                    );
                                }
                            }
                        }
                        PlottableEntry::Parametric {
                            param,
                            x_compiled,
                            y_compiled,
                        } => {
                            let (t0, t1) = self.resolve_style_param_domain(&style);
                            let raw_branches = math::sample_parametric_branches(
                                &x_compiled,
                                &y_compiled,
                                &param,
                                &eval_ctx,
                                t0,
                                t1,
                                self.curve_samples,
                            );
                            let branches = Self::clip_branches_to_y_range(raw_branches, y_range);
                            for branch in branches {
                                if style.show_lines {
                                    draw_glowing_plot_line_with_opacity(
                                        plot_ui,
                                        branch.clone(),
                                        &series_label,
                                        style.color,
                                        style.stroke_width,
                                        style.line_style.to_egui(),
                                        style.opacity,
                                    );
                                }
                                if style.show_points {
                                    let stride = (branch.len() / 24).max(1);
                                    let pts: Vec<[f64; 2]> =
                                        branch.into_iter().step_by(stride).collect();
                                    plot_ui.points(
                                        Points::new(PlotPoints::from(pts))
                                            .radius(style.stroke_width + 1.4_f32)
                                            .color(style.color.gamma_multiply(style.opacity)),
                                    );
                                }
                            }
                        }
                        PlottableEntry::Polar { param, r_compiled } => {
                            let (t0, t1) = self.resolve_style_param_domain(&style);
                            let raw_branches = math::sample_polar_branches(
                                &r_compiled,
                                &param,
                                &eval_ctx,
                                t0,
                                t1,
                                self.curve_samples,
                            );
                            let branches = Self::clip_branches_to_y_range(raw_branches, y_range);
                            for branch in branches {
                                if style.show_lines {
                                    draw_glowing_plot_line_with_opacity(
                                        plot_ui,
                                        branch,
                                        &series_label,
                                        style.color,
                                        style.stroke_width,
                                        style.line_style.to_egui(),
                                        style.opacity,
                                    );
                                }
                            }
                        }
                        PlottableEntry::ScalarOnly {
                            compiled,
                            eval_point_x,
                        } => {
                            if plot_eval_pt {
                                if let Some(px_comp) = eval_point_x {
                                    let mut sc = EvalScratch::new();
                                    if let (Ok(x0), Ok(y0)) = (
                                        px_comp.eval(&eval_ctx, &[], &mut sc),
                                        compiled.eval(&eval_ctx, &[], &mut sc),
                                    ) {
                                        if x0.is_finite() && y0.is_finite() {
                                            plot_ui.points(
                                                Points::new(vec![[x0, y0]])
                                                    .name(format!("{} = {:.3}", expr, y0))
                                                    .radius(6.0_f32)
                                                    .shape(MarkerShape::Circle)
                                                    .color(pal.accent_green),
                                            );
                                            plot_ui.text(PlotText::new(
                                                PlotPoint::new(x0, y0),
                                                RichText::new(format!("({:.2}, {:.3})", x0, y0))
                                                    .font(FontId::monospace(11.0_f32))
                                                    .strong()
                                                    .color(pal.accent_green),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ---- 5. Interactive Automatic Points of Interest (Roots, Y-Intercept, Extrema, Intersections) ----
                let focus_idx = visible_1d_curves
                    .iter()
                    .position(|(_, _, _, is_f)| *is_f)
                    .or_else(|| (!visible_1d_curves.is_empty()).then_some(0));
                if let Some(fidx) = focus_idx {
                    let (ref fname, ref fcomp, fcol, _) = visible_1d_curves[fidx];
                    let others: Vec<(&str, &CompiledExpr)> = visible_1d_curves
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != fidx)
                        .map(|(_, (n, c, _, _))| (n.as_str(), c))
                        .collect();
                    let pois = math::detect_curve_pois(
                        fname,
                        fcomp,
                        &others,
                        &eval_ctx,
                        global_sample_min,
                        global_sample_max,
                        vis_min_y,
                        vis_max_y,
                    );

                    let clicked_plot = !shift_held && primary_pressed && hover_pos.map(|p| frame_rect.contains(p)).unwrap_or(false);
                    for poi in pois {
                        let dot_color = match poi.kind {
                            PoiKind::Intersection => pal.accent_green,
                            PoiKind::LocalMax | PoiKind::LocalMin => pal.accent_amber,
                            PoiKind::Root | PoiKind::YIntercept => pal.text_secondary,
                            PoiKind::Tangency => pal.accent_blue,
                        };
                        let screen_pt = plot_ui.screen_from_plot(PlotPoint::new(poi.x, poi.y));
                        let hovered_dot = hover_pos
                            .map(|hp| hp.distance(screen_pt) <= 12.0_f32)
                            .unwrap_or(false);

                        // Subtle grey/accent interactive dot (glows brighter on hover)
                        plot_ui.points(
                            Points::new(vec![[poi.x, poi.y]])
                                .radius(if hovered_dot { 6.2_f32 } else { 4.2_f32 })
                                .shape(MarkerShape::Circle)
                                .color(if hovered_dot { fcol } else { dot_color }),
                        );

                        if hovered_dot {
                            plot_ui.text(PlotText::new(
                                PlotPoint::new(poi.x, poi.y),
                                RichText::new(&poi.label)
                                    .font(FontId::monospace(11.0_f32))
                                    .strong()
                                    .color(pal.text_primary),
                            ));
                            if clicked_plot {
                                // Toggle pin on click
                                if let Some(pos) = self.pinned_pois.iter().position(|p| {
                                    (p.x - poi.x).abs() < 1e-3_f64 && (p.y - poi.y).abs() < 1e-3_f64
                                }) {
                                    self.pinned_pois.remove(pos);
                                } else {
                                    self.pinned_pois.push(PinnedPoi {
                                        x: poi.x,
                                        y: poi.y,
                                        label: poi.label.clone(),
                                        color: dot_color,
                                    });
                                }
                            }
                        }
                    }
                }

                // Render user-pinned POIs permanently
                for pin in &self.pinned_pois {
                    plot_ui.points(
                        Points::new(vec![[pin.x, pin.y]])
                            .radius(5.8_f32)
                            .shape(MarkerShape::Circle)
                            .color(pin.color),
                    );
                    plot_ui.text(PlotText::new(
                        PlotPoint::new(pin.x, pin.y),
                        RichText::new(&pin.label)
                            .font(FontId::monospace(10.8_f32))
                            .strong()
                            .color(pin.color),
                    ));
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
                .add(pill_button("⌫ Clear", false, pal))
                .on_hover_text("Clear console output")
                .clicked()
            {
                self.console_logs.clear();
                self.status_banner = "Console cleared.".to_string();
            }
            if ui
                .add(pill_button("⎘ Copy", false, pal))
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
                .add(pill_button("▣ Load .R", false, pal))
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
                .add(pill_button("‡ Save .R", false, pal))
                .on_hover_text("Save Script Editor to exports/workspace_script.R")
                .clicked()
            {
                self.save_editor_script_default();
            }
            if ui
                .add(pill_button("? Help", false, pal))
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
                                        if ui.small_button("Focus 1st Quadrant (x ≥ 0, y ≥ 0)").clicked() {
                                            self.focus_first_quadrant();
                                        }
                                    });
                                    cols[0].horizontal_wrapped(|ui| {
                                        ui.label("X Label:");
                                        ui.add(
                                            egui::TextEdit::singleline(&mut self.axis_config.x_label)
                                                .desired_width(82.0_f32),
                                        );
                                        let mut xs = self.axis_config.x_step.unwrap_or(0.0_f64);
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut xs)
                                                    .speed(0.1_f64)
                                                    .range(0.0_f64..=1000.0_f64)
                                                    .prefix("Step: "),
                                            )
                                            .changed()
                                        {
                                            self.axis_config.x_step = (xs > 1e-6_f64).then_some(xs);
                                        }
                                    });
                                    cols[0].horizontal_wrapped(|ui| {
                                        ui.label("Y Label:");
                                        ui.add(
                                            egui::TextEdit::singleline(&mut self.axis_config.y_label)
                                                .desired_width(82.0_f32),
                                        );
                                        let mut ys = self.axis_config.y_step.unwrap_or(0.0_f64);
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut ys)
                                                    .speed(0.1_f64)
                                                    .range(0.0_f64..=1000.0_f64)
                                                    .prefix("Step: "),
                                            )
                                            .changed()
                                        {
                                            self.axis_config.y_step = (ys > 1e-6_f64).then_some(ys);
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
                    .add(pill_button("◿", ds.show_on_plot, pal))
                    .on_hover_text("Toggle Scatter Plot Overlay on 2D Graph")
                    .clicked()
                {
                    ds.show_on_plot = !ds.show_on_plot;
                }
                if ui
                    .add(pill_button("⟋", ds.show_ols_line, pal))
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
                .add(pill_button("‡ CSV", false, pal))
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
                Color32::from_rgb(72, 118, 168),
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
            ui.selectable_value(&mut self.notes_preview_mode, false, "¶ Edit");
            ui.selectable_value(&mut self.notes_preview_mode, true, "§ Reader");
            if ui.add(pill_button("▣", false, pal)).on_hover_text("Import file").clicked() {
                let path = PathBuf::from(self.file_io_path_input.trim());
                match load_text_file(&path) {
                    Ok(txt) => {
                        self.workspace.markdown_notes = txt;
                        self.status_banner = format!("Loaded notes from {:?}", path);
                    }
                    Err(e) => self.status_banner = e,
                }
            }
            if ui.add(pill_button("‡", false, pal)).on_hover_text("Save notes file").clicked() {
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
                            RichText::new("⊳ Export Bundle & Open Positron")
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
            if ui.add(pill_button("⎘ Copy LaTeX / TikZ", false, pal)).clicked() {
                ui.ctx().copy_text(latex_str.clone());
                self.status_banner = "Copied LaTeX derivation to clipboard.".to_string();
            }
            if ui.add(pill_button("⎘ Copy R Script", false, pal)).clicked() {
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
            egui::Window::new("⊞ Data Viewer & Datasets Manager")
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
            egui::Window::new("§ Notes & Literate Reports (.qmd / .Rmd / .md)")
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
            egui::Window::new("? Oikonomos (Οἰκονόμος) Help (`intro-help` & `general-help`)")
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
            egui::Window::new("◈ R Environment Inspector")
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
            egui::Window::new("⊳ Console / Script / Settings")
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
        self.step_slider_animations(ctx);
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
            dataset_show_raw_csv: false,
            undo_history: UndoHistory::default(),
            symbolic_list_edit_mode: false,
            selected_cell_id: None,
            request_focus_cell_id: None,
            frozen_baseline: None,
            pinned_pois: Vec::new(),
            shift_drag: ShiftDragState::default(),
            axis_config: AxisDisplayConfig::default(),
            last_frame_time: None,
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
            dataset_show_raw_csv: false,
            undo_history: UndoHistory::default(),
            symbolic_list_edit_mode: false,
            selected_cell_id: None,
            request_focus_cell_id: None,
            frozen_baseline: None,
            pinned_pois: Vec::new(),
            shift_drag: ShiftDragState::default(),
            axis_config: AxisDisplayConfig::default(),
            last_frame_time: None,
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

        // Vectorized Desmos->R evaluation in Symbolic cell: q(c(1, 2, 3, 4)) -> "[3, 5, 7, 9]"
        assert_eq!(
            app.try_eval_vectorized_r_cell("q(c(1, 2, 3, 4))"),
            Some("[3, 5, 7, 9]".to_string())
        );

        // Synced Function Value Table creates both Data Viewer dataset AND R data.frame(x, y, dy_dx)
        app.create_table_for_function("q(x)", None);
        assert!(app.workspace.datasets.iter().any(|d| d.name == "q_table"));
        assert!(matches!(
            app.workspace.r_objects.get("q_table"),
            Some(RObject::DataFrame { row_count: 6, .. })
        ));

        // Freeze Baseline captures visible curves snapshot
        app.freeze_current_baseline();
        assert!(app.frozen_baseline.is_some());
        assert!(!app.frozen_baseline.as_ref().unwrap().curves.is_empty());
    }
}
