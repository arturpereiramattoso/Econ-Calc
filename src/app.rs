use crate::exporter::{
    ActiveModelKind, ExportOutcome, ExportPayload, GridRow, ReportFormat,
    export_and_launch_positron,
};
use eframe::egui::{
    self, Color32, FontFamily, FontId, Margin, Pos2, RichText, Rounding, Sense, Stroke, TextStyle,
    Ui, Vec2,
};
use egui_extras as _;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, PlotPoints, Points, VLine};
use nalgebra::{Matrix2, Vector2};
use std::borrow::Borrow;

/// Deterministic insertion-ordered map (`IndexMap<K, V>`) preserving declaration sequence
/// for symbolic parameters and functions without requiring external crate changes.
#[derive(Debug, Clone, Default)]
pub struct IndexMap<K, V> {
    entries: Vec<(K, V)>,
}

impl<K, V> IndexMap<K, V>
where
    K: PartialEq,
{
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        for (existing_k, existing_v) in &mut self.entries {
            if *existing_k == key {
                return Some(std::mem::replace(existing_v, value));
            }
        }
        self.entries.push((key, value));
        None
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: PartialEq + ?Sized,
    {
        self.entries
            .iter()
            .find(|(k, _)| k.borrow() == key)
            .map(|(_, v)| v)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: PartialEq + ?Sized,
    {
        if let Some(idx) = self.entries.iter().position(|(k, _)| k.borrow() == key) {
            Some(self.entries.remove(idx).1)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.entries.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }
}

impl<'a, K, V> IntoIterator for &'a IndexMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::iter::Map<std::slice::Iter<'a, (K, V)>, fn(&'a (K, V)) -> (&'a K, &'a V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter().map(|(k, v)| (k, v))
    }
}

impl<'a, K, V> IntoIterator for &'a mut IndexMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter =
        std::iter::Map<std::slice::IterMut<'a, (K, V)>, fn(&'a mut (K, V)) -> (&'a K, &'a mut V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter_mut().map(|(k, v)| (&*k, v))
    }
}

/// Core Desmos-style dynamic symbolic state for the computational economics IDE.
#[derive(Debug, Clone)]
pub struct Workspace {
    pub parameters: IndexMap<String, f64>,
    pub functions: IndexMap<String, String>,
    pub markdown_notes: String,
    pub csv_scratchpad: String,
}

impl Workspace {
    pub fn new_default() -> Self {
        let mut parameters = IndexMap::new();
        parameters.insert("A".to_string(), 2.0_f64);
        parameters.insert("alpha".to_string(), 0.45_f64);
        parameters.insert("beta".to_string(), 0.55_f64);
        parameters.insert("price_x".to_string(), 4.0_f64);
        parameters.insert("price_y".to_string(), 5.0_f64);
        parameters.insert("I".to_string(), 100.0_f64);
        parameters.insert("Y_bar".to_string(), 18.0_f64);

        let mut functions = IndexMap::new();
        functions.insert(
            "f(x)".to_string(),
            "(Y_bar / (A * x^alpha))^(1 / beta)".to_string(),
        );
        functions.insert("g(x)".to_string(), "(I - price_x * x) / price_y".to_string());
        functions.insert(
            "u(x)".to_string(),
            "A * x^alpha * ((I - price_x * x) / price_y)^beta".to_string(),
        );

        let markdown_notes = "\
### Microeconomic Tangency & Constrained Optimization Notes
- **Indifference / Isoquant Curve** $f(x)$ traces combinations of $(x, y)$ yielding constant output $\\bar{Y}$.
- **Budget / Isocost Constraint** $g(x) = \\frac{I - P_x x}{P_y}$ has constant slope $-\\frac{P_x}{P_y}$.
- **Optimality Condition:** Interior optimum satisfies $\\text{MRS}_{x,y} = \\frac{\\alpha y}{\\beta x} = \\frac{P_x}{P_y}$."
            .to_string();

        let csv_scratchpad = "\
x,y
1.5,16.8
3.0,13.5
5.0,11.2
8.0,8.6
11.25,11.0
14.0,6.2
18.0,4.5"
            .to_string();

        Self {
            parameters,
            functions,
            markdown_notes,
            csv_scratchpad,
        }
    }

    pub fn clear(&mut self) {
        self.parameters.clear();
        self.functions.clear();
        self.markdown_notes.clear();
        self.csv_scratchpad.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewportMode {
    Plot2D,
    Surface3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BottomEditorTab {
    CommandEditor,
    ConsoleLog,
    LatexDerivation,
    RScriptPreview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightDataTab {
    SimulationGrid,
    CsvPasteboard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnippetTemplate {
    CobbDouglas,
    BudgetConstraint,
    IsLmMacro,
    SolowGrowth,
    MonopolistProfit,
}

impl SnippetTemplate {
    fn all() -> [SnippetTemplate; 5] {
        [
            SnippetTemplate::CobbDouglas,
            SnippetTemplate::BudgetConstraint,
            SnippetTemplate::IsLmMacro,
            SnippetTemplate::SolowGrowth,
            SnippetTemplate::MonopolistProfit,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            SnippetTemplate::CobbDouglas => "Cobb-Douglas",
            SnippetTemplate::BudgetConstraint => "Budget Constraint",
            SnippetTemplate::IsLmMacro => "IS-LM",
            SnippetTemplate::SolowGrowth => "Solow-Swan Growth",
            SnippetTemplate::MonopolistProfit => "Monopoly MR = MC",
        }
    }

    fn model_kind(self) -> ActiveModelKind {
        match self {
            SnippetTemplate::CobbDouglas => ActiveModelKind::CobbDouglas,
            SnippetTemplate::BudgetConstraint => ActiveModelKind::ConstrainedUtility,
            SnippetTemplate::IsLmMacro => ActiveModelKind::CustomExpression,
            SnippetTemplate::SolowGrowth => ActiveModelKind::CustomExpression,
            SnippetTemplate::MonopolistProfit => ActiveModelKind::GameTheory2x2,
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

# Isoquant K(L) for target output Y_bar and Isocost Line
f(x) = (Y_bar / (A * x^\alpha))^(1 / \beta)
g(x) = (I - price_x * x) / price_y
mp_l(x) = A * \alpha * x^(\alpha - 1) * (10.0^\beta)"##
            }
            SnippetTemplate::BudgetConstraint => {
                r##"# Consumer Constrained Utility Maximization (Marshallian Demand)
A = 1.5
\alpha = 0.50
\beta = 0.50
price_x = 4.0
price_y = 5.0
I = 120.0
Y_bar = 16.4317

# Indifference Curve f(x), Budget Line g(x), and Substituted Utility u(x)
f(x) = (Y_bar / (A * x^\alpha))^(1 / \beta)
g(x) = (I - price_x * x) / price_y
u(x) = A * x^\alpha * g(x)^\beta"##
            }
            SnippetTemplate::IsLmMacro => {
                r##"# Macroeconomics IS-LM Goods & Money Market Equilibrium (x = Output Y, y = Interest Rate r)
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

# IS Curve r(Y) and LM Curve r(Y)
f(x) = (c_0 - c_1 * T + I_0 + G - (1.0 - c_1) * x) / b * 100.0
g(x) = (k * x - M_s / P) / h * 100.0"##
            }
            SnippetTemplate::SolowGrowth => {
                r##"# Solow-Swan Neoclassical Growth Model (x = Capital per Worker k)
A = 2.2
\alpha = 0.35
s = 0.28
\delta = 0.07
n = 0.02

# Actual Investment s*f(k), Break-Even Investment (delta + n)*k, Output y(k)
f(x) = s * A * x^\alpha
g(x) = (\delta + n) * x
y(x) = A * x^\alpha"##
            }
            SnippetTemplate::MonopolistProfit => {
                r##"# Imperfect Competition: Inverse Demand, Marginal Revenue, Marginal Cost (x = Quantity Q)
a = 28.0
b = 0.85
c_0 = 12.0
c_1 = 2.0
c_2 = 0.25

# Marginal Revenue f(x), Marginal Cost g(x), Demand, and Total Profit composing g(x)
f(x) = a - 2.0 * b * x
g(x) = c_1 + 2.0 * c_2 * x
demand(x) = a - b * x
profit(x) = demand(x) * x - (c_0 + c_1 * x + c_2 * x^2)"##
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

pub struct EconCalcApp {
    pub workspace: Workspace,

    // UI state & navigation
    dark_mode: bool,
    active_model: ActiveModelKind,
    viewport_mode: ViewportMode,
    bottom_tab: BottomEditorTab,
    right_data_tab: RightDataTab,
    selected_snippet: SnippetTemplate,
    report_format: ReportFormat,

    // Editor buffer & logs
    editor_script: String,
    console_logs: Vec<ConsoleLogEntry>,
    eval_counter: usize,
    status_banner: String,

    // Numerical evaluation & plot controls
    domain_min: f64,
    domain_max: f64,
    probe_x: f64,
    show_derivative_overlay: bool,
    show_csv_ols_line: bool,

    // 3D Isometric surface controls
    yaw_deg: f32,
    pitch_deg: f32,
    surface_zoom: f32,
    surface_resolution: usize,
    surface_wireframe_only: bool,
}

impl EconCalcApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::apply_positron_theme(&cc.egui_ctx, true);

        let workspace = Workspace::new_default();
        let initial_script = SnippetTemplate::CobbDouglas.script_text().to_string();

        let mut app = Self {
            workspace,
            dark_mode: true,
            active_model: ActiveModelKind::CobbDouglas,
            viewport_mode: ViewportMode::Plot2D,
            bottom_tab: BottomEditorTab::CommandEditor,
            right_data_tab: RightDataTab::SimulationGrid,
            selected_snippet: SnippetTemplate::CobbDouglas,
            report_format: ReportFormat::QuartoQmd,
            editor_script: initial_script,
            console_logs: Vec::new(),
            eval_counter: 0,
            status_banner:
                "Positron Symbolic Engine Ready — Native Windows 11 ARM64 (aarch64-pc-windows-msvc)"
                    .to_string(),
            domain_min: 0.25_f64,
            domain_max: 25.0_f64,
            probe_x: 11.25_f64,
            show_derivative_overlay: false,
            show_csv_ols_line: true,
            yaw_deg: 38.0_f32,
            pitch_deg: 28.0_f32,
            surface_zoom: 1.0_f32,
            surface_resolution: 28,
            surface_wireframe_only: false,
        };

        app.evaluate_editor_script();
        app
    }

    fn apply_positron_theme(ctx: &egui::Context, dark: bool) {
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(8.0_f32, 6.0_f32);
        style.spacing.window_margin = Margin::same(8.0_f32);
        style.spacing.button_padding = Vec2::new(10.0_f32, 5.0_f32);

        style.text_styles.insert(
            TextStyle::Heading,
            FontId::new(16.5_f32, FontFamily::Proportional),
        );
        style.text_styles.insert(
            TextStyle::Body,
            FontId::new(13.5_f32, FontFamily::Proportional),
        );
        style.text_styles.insert(
            TextStyle::Monospace,
            FontId::new(13.0_f32, FontFamily::Monospace),
        );
        style.text_styles.insert(
            TextStyle::Button,
            FontId::new(13.0_f32, FontFamily::Proportional),
        );

        if dark {
            let mut visuals = egui::Visuals::dark();
            visuals.panel_fill = Color32::from_rgb(15, 23, 42);
            visuals.window_fill = Color32::from_rgb(15, 23, 42);
            visuals.extreme_bg_color = Color32::from_rgb(9, 14, 26);
            visuals.faint_bg_color = Color32::from_rgb(23, 33, 56);
            visuals.code_bg_color = Color32::from_rgb(11, 17, 32);
            visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(20, 29, 49);
            visuals.widgets.noninteractive.bg_stroke =
                Stroke::new(1.0_f32, Color32::from_rgb(38, 52, 82));
            visuals.widgets.inactive.bg_fill = Color32::from_rgb(30, 41, 69);
            visuals.widgets.hovered.bg_fill = Color32::from_rgb(43, 61, 102);
            visuals.widgets.active.bg_fill = Color32::from_rgb(37, 99, 235);
            visuals.selection.bg_fill = Color32::from_rgb(30, 64, 175);
            visuals.selection.stroke = Stroke::new(1.0_f32, Color32::from_rgb(147, 197, 253));
            style.visuals = visuals;
        } else {
            style.visuals = egui::Visuals::light();
        }

        ctx.set_style(style);
    }

    /// Strips LaTeX slashes (`\alpha` -> `alpha`) and normalizes braces (`{` -> `(`, `}` -> `)`)
    /// so user expressions like `\alpha * x^{\beta}` evaluate cleanly in `meval`.
    fn sanitize_for_meval(raw: &str) -> String {
        let mut s = raw
            .replace("\\cdot", "*")
            .replace("\\times", "*")
            .replace('{', "(")
            .replace('}', ")");
        s.retain(|ch| ch != '\\');
        s.trim().to_string()
    }

    /// Expands references to other workspace functions (e.g., `g(x)` inside `u(x) = A * x^alpha * g(x)^beta`)
    /// so `meval` can evaluate composed functions cleanly.
    fn expand_workspace_functions(&self, expr_str: &str) -> String {
        let mut expanded = Self::sanitize_for_meval(expr_str);
        for _ in 0..4 {
            let mut changed = false;
            for (sig, body) in &self.workspace.functions {
                if expanded.contains(sig.as_str()) && sig != &expanded {
                    let replacement = format!("({})", body);
                    expanded = expanded.replace(sig.as_str(), &replacement);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        expanded
    }

    /// Replaces word-bounded identifier occurrences in an expression string.
    fn replace_identifier_token(expr: &str, ident: &str, replacement: &str) -> String {
        if ident.is_empty() || !expr.contains(ident) {
            return expr.to_string();
        }
        let is_ident_char = |c: char| c.is_ascii_alphanumeric() || c == '_';
        let mut result = String::with_capacity(expr.len() + 16);
        let mut rest = expr;
        let mut prev_char: Option<char> = None;

        while let Some(pos) = rest.find(ident) {
            let prefix = &rest[..pos];
            if let Some(last_ch) = prefix.chars().last() {
                prev_char = Some(last_ch);
            }
            let valid_start = !prev_char.map(is_ident_char).unwrap_or(false);
            let after_idx = pos + ident.len();
            let after_str = &rest[after_idx..];
            let next_char = after_str.chars().next();
            let valid_end = !next_char.map(is_ident_char).unwrap_or(false);

            result.push_str(prefix);
            if valid_start && valid_end {
                result.push_str(replacement);
            } else {
                result.push_str(ident);
            }
            prev_char = ident.chars().last();
            rest = after_str;
        }
        result.push_str(rest);
        result
    }

    /// Converts an internal `meval` expression into valid LaTeX math syntax for `\begin{align*}`.
    fn format_latex_math(raw: &str) -> String {
        let mut s = raw.to_string();
        for (ident, latex_sym) in [
            ("alpha", "\\alpha"),
            ("beta", "\\beta"),
            ("gamma", "\\gamma"),
            ("delta", "\\delta"),
            ("lambda", "\\lambda"),
            ("sigma", "\\sigma"),
            ("theta", "\\theta"),
            ("rho", "\\rho"),
            ("price_x", "P_x"),
            ("price_y", "P_y"),
            ("Y_bar", "\\bar{Y}"),
            ("c_0", "c_0"),
            ("c_1", "c_1"),
            ("c_2", "c_2"),
            ("I_0", "I_0"),
            ("M_s", "M_s"),
        ] {
            s = Self::replace_identifier_token(&s, ident, latex_sym);
        }
        s.replace('*', " \\cdot ")
    }

    /// Substitutes numeric parameter values into an expression so PGFPlots (`\addplot`) compiles without unknown symbols.
    fn substitute_params_for_tikz(&self, raw_expr: &str) -> String {
        let mut s = self.expand_workspace_functions(raw_expr);
        let mut pairs: Vec<(&String, &f64)> = self.workspace.parameters.iter().collect();
        pairs.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()));
        for (name, val) in pairs {
            s = Self::replace_identifier_token(&s, name, &format!("({:.4})", val));
        }
        s
    }

    /// Formats symbolic identifiers with Unicode math symbols for the Symbolic Environment display.
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

    /// Parses and evaluates the multiline Command Editor script, splitting lines by `=`
    /// into parameters (`name = value`) and functions (`f(x) = expression`).
    pub fn evaluate_editor_script(&mut self) {
        self.eval_counter += 1;
        let run_id = self.eval_counter;

        let lines: Vec<String> = self
            .editor_script
            .lines()
            .map(|l| l.trim().to_string())
            .collect();

        let mut parsed_params = 0_usize;
        let mut parsed_funcs = 0_usize;
        let mut error_count = 0_usize;

        for (idx, raw_line) in lines.iter().enumerate() {
            let line_num = idx + 1;
            if raw_line.is_empty() || raw_line.starts_with('#') || raw_line.starts_with("//") {
                continue;
            }

            let Some((lhs_raw, rhs_raw)) = raw_line.split_once('=') else {
                error_count += 1;
                self.console_logs.push(ConsoleLogEntry {
                    timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                    source_line: raw_line.clone(),
                    message: "Syntax error: expected `=` assignment (`param = val` or `f(x) = expr`)."
                        .to_string(),
                    is_error: true,
                });
                continue;
            };

            let lhs_clean = Self::sanitize_for_meval(lhs_raw);
            let rhs_clean = Self::sanitize_for_meval(rhs_raw);

            if lhs_clean.is_empty() || rhs_clean.is_empty() {
                error_count += 1;
                self.console_logs.push(ConsoleLogEntry {
                    timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                    source_line: raw_line.clone(),
                    message: "Empty left-hand side or right-hand side in assignment.".to_string(),
                    is_error: true,
                });
                continue;
            }

            let is_explicit_fn = lhs_clean.contains('(') && lhs_clean.ends_with(')');

            if is_explicit_fn {
                let sig = lhs_clean.replace(' ', "");
                match self.eval_expression_str(&rhs_clean, self.probe_x, 10.0_f64) {
                    Ok(sample_val) => {
                        self.workspace.functions.insert(sig.clone(), rhs_clean.clone());
                        parsed_funcs += 1;
                        self.console_logs.push(ConsoleLogEntry {
                            timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                            source_line: raw_line.clone(),
                            message: format!(
                                "Bound function `{}` = `{}` (probe @ x={:.2} -> {:.4})",
                                sig, rhs_clean, self.probe_x, sample_val
                            ),
                            is_error: false,
                        });
                    }
                    Err(err) => {
                        error_count += 1;
                        self.console_logs.push(ConsoleLogEntry {
                            timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                            source_line: raw_line.clone(),
                            message: format!("Function `{}` failed meval validation: {}", sig, err),
                            is_error: true,
                        });
                    }
                }
            } else {
                let param_name = lhs_clean.replace(' ', "");
                match self.eval_parameter_rhs(&rhs_clean) {
                    Ok(val) if val.is_finite() => {
                        self.workspace.parameters.insert(param_name.clone(), val);
                        parsed_params += 1;
                        self.console_logs.push(ConsoleLogEntry {
                            timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                            source_line: raw_line.clone(),
                            message: format!("Set parameter `{}` = {:.6}", param_name, val),
                            is_error: false,
                        });
                    }
                    _ => {
                        let promoted_sig = format!("{}(x)", param_name);
                        match self.eval_expression_str(&rhs_clean, self.probe_x, 10.0_f64) {
                            Ok(sample_val) => {
                                self.workspace
                                    .functions
                                    .insert(promoted_sig.clone(), rhs_clean.clone());
                                parsed_funcs += 1;
                                self.console_logs.push(ConsoleLogEntry {
                                    timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                                    source_line: raw_line.clone(),
                                    message: format!(
                                        "Promoted `x`-dependent expression to `{}` = `{}` (probe -> {:.4})",
                                        promoted_sig, rhs_clean, sample_val
                                    ),
                                    is_error: false,
                                });
                            }
                            Err(err) => {
                                error_count += 1;
                                self.console_logs.push(ConsoleLogEntry {
                                    timestamp_tag: format!("Run #{:02} L{:02}", run_id, line_num),
                                    source_line: raw_line.clone(),
                                    message: format!(
                                        "Could not evaluate `{}` as parameter or function: {}",
                                        param_name, err
                                    ),
                                    is_error: true,
                                });
                            }
                        }
                    }
                }
            }
        }

        self.status_banner = format!(
            "Evaluated Run #{:02}: {} parameter(s), {} function(s) active, {} error(s)",
            run_id,
            parsed_params,
            parsed_funcs,
            error_count
        );
    }

    fn eval_parameter_rhs(&self, expr_str: &str) -> Result<f64, String> {
        let mut ctx = meval::Context::new();
        for (k, &v) in &self.workspace.parameters {
            ctx.var(k.as_str(), v);
        }
        meval::eval_str_with_context(expr_str, &ctx).map_err(|e| e.to_string())
    }

    /// Evaluates an arbitrary expression string with all workspace parameters, composed functions, and `(x, y)` injected.
    pub fn eval_expression_str(&self, expr_str: &str, x_val: f64, y_val: f64) -> Result<f64, String> {
        let mut ctx = meval::Context::new();
        for (k, &v) in &self.workspace.parameters {
            ctx.var(k.as_str(), v);
        }
        ctx.var("x", x_val);
        ctx.var("y", y_val);
        ctx.var("L", x_val);
        ctx.var("K", y_val);
        ctx.var("Q", x_val);
        ctx.var("Y", x_val);

        let expanded = self.expand_workspace_functions(expr_str);
        meval::eval_str_with_context(&expanded, &ctx).map_err(|e| e.to_string())
    }

    /// Central finite-difference first derivative $f'(x)$.
    fn numerical_derivative(&self, expr_str: &str, x: f64) -> Result<f64, String> {
        let h = 1e-5_f64.max(x.abs() * 1e-6_f64);
        let f_plus = self.eval_expression_str(expr_str, x + h, 10.0_f64)?;
        let f_minus = self.eval_expression_str(expr_str, x - h, 10.0_f64)?;
        Ok((f_plus - f_minus) / (2.0_f64 * h))
    }

    /// Central finite-difference second derivative $f''(x)$.
    fn numerical_second_derivative(&self, expr_str: &str, x: f64) -> Result<f64, String> {
        let h = 1e-4_f64.max(x.abs() * 1e-5_f64);
        let f_plus = self.eval_expression_str(expr_str, x + h, 10.0_f64)?;
        let f_mid = self.eval_expression_str(expr_str, x, 10.0_f64)?;
        let f_minus = self.eval_expression_str(expr_str, x - h, 10.0_f64)?;
        Ok((f_plus - 2.0_f64 * f_mid + f_minus) / (h * h))
    }

    /// Composite Simpson's Rule definite integral on $[a, b]$.
    fn composite_simpson(&self, expr_str: &str, a: f64, b: f64, n: usize) -> Result<f64, String> {
        let steps = if n % 2 == 0 { n.max(4) } else { n + 1 };
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

    /// Finds the intersection of the first two functions in `workspace.functions` on `[domain_min, domain_max]`.
    fn find_primary_secondary_intersection(&self) -> Option<(f64, f64)> {
        let mut iter = self.workspace.functions.values();
        let f_expr = iter.next()?;
        let g_expr = iter.next()?;

        let samples = 240_usize;
        let span = (self.domain_max - self.domain_min).max(1e-3_f64);
        let step = span / (samples as f64);

        let eval_diff = |x: f64| -> Option<f64> {
            let fy = self.eval_expression_str(f_expr, x, 10.0_f64).ok()?;
            let gy = self.eval_expression_str(g_expr, x, 10.0_f64).ok()?;
            let d = fy - gy;
            d.is_finite().then_some(d)
        };

        let mut prev_x = self.domain_min;
        let mut prev_d = eval_diff(prev_x)?;

        for i in 1..=samples {
            let curr_x = self.domain_min + (i as f64) * step;
            if let Some(curr_d) = eval_diff(curr_x) {
                if prev_d * curr_d <= 0.0_f64 {
                    let mut lo = prev_x;
                    let mut hi = curr_x;
                    let mut d_lo = prev_d;
                    for _ in 0..50 {
                        let mid = 0.5_f64 * (lo + hi);
                        let Some(d_mid) = eval_diff(mid) else {
                            break;
                        };
                        if d_mid.abs() < 1e-9_f64 || (hi - lo).abs() < 1e-9_f64 {
                            let y_val = self.eval_expression_str(f_expr, mid, 10.0_f64).ok()?;
                            return Some((mid, y_val));
                        }
                        if d_lo * d_mid <= 0.0_f64 {
                            hi = mid;
                        } else {
                            lo = mid;
                            d_lo = d_mid;
                        }
                    }
                    let mid = 0.5_f64 * (lo + hi);
                    if let Ok(y_val) = self.eval_expression_str(f_expr, mid, 10.0_f64) {
                        if y_val.is_finite() {
                            return Some((mid, y_val));
                        }
                    }
                }
                prev_x = curr_x;
                prev_d = curr_d;
            }
        }
        None
    }

    /// Parses `workspace.csv_scratchpad` into `(x, y)` coordinate pairs for scatter plot overlay.
    fn parse_csv_scratchpad_points(&self) -> Vec<[f64; 2]> {
        let mut pts = Vec::new();
        for raw_line in self.workspace.csv_scratchpad.lines() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed
                .split(|c| c == ',' || c == ';' || c == '\t')
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

    /// Computes Ordinary Least Squares `(slope_beta1, intercept_beta0, r_squared)` on CSV points.
    fn compute_csv_ols(&self, pts: &[[f64; 2]]) -> Option<(f64, f64, f64)> {
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

    /// Extracts structural parameters from `workspace.parameters` (with fallbacks) and solves
    /// the analytical constrained optimization equilibrium `(x_star, y_star, lambda_star, indirect_u)`.
    fn solve_structural_equilibrium(&self) -> (f64, f64, f64, f64) {
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
            let mu_x =
                tfp_a * alpha * ix.max(1e-6_f64).powf(alpha - 1.0_f64) * iy.max(1e-6_f64).powf(beta);
            let lambda_star = mu_x / px;
            return (ix, iy, lambda_star, indirect_u);
        }

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
        (x_star, y_star, lambda_star, indirect_u)
    }

    /// Computes second-order Hessian matrix diagnostics via `nalgebra::Matrix2` and `Vector2`.
    fn compute_hessian_diagnostics(&self, x: f64, y: f64) -> (Matrix2<f64>, Vector2<f64>, f64, f64) {
        let tfp_a = self
            .workspace
            .parameters
            .get("A")
            .copied()
            .unwrap_or(2.0_f64);
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

    /// Dynamically evaluates the primary and secondary workspace functions into a `rows_count`-row `GridRow` vector.
    fn build_simulation_grid(&self, rows_count: usize) -> Vec<GridRow> {
        let mut rows = Vec::with_capacity(rows_count);
        let span = (self.domain_max - self.domain_min).max(1e-3_f64);

        let mut fn_iter = self.workspace.functions.values();
        let primary_expr = fn_iter.next().cloned().unwrap_or_else(|| "x".to_string());
        let secondary_expr = fn_iter.next().cloned().unwrap_or_else(|| "0.0".to_string());

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

    /// Generates auto-derived LaTeX mathematical output bundled with the user's Markdown Notes.
    fn generate_latex_derivation(&self) -> String {
        let (x_star, y_star, lambda_star, indirect_u) = self.solve_structural_equilibrium();
        let (hessian, grad, det_h, trace_h) = self.compute_hessian_diagnostics(x_star, y_star);

        let mut fn_iter = self.workspace.functions.iter();
        let (primary_sig, primary_expr) = fn_iter
            .next()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .unwrap_or(("f(x)", "x"));
        let (secondary_sig, secondary_expr) = fn_iter
            .next()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .unwrap_or(("g(x)", "0"));

        let fx_probe = self
            .eval_expression_str(primary_expr, self.probe_x, y_star)
            .unwrap_or(0.0_f64);
        let dfx_probe = self
            .numerical_derivative(primary_expr, self.probe_x)
            .unwrap_or(0.0_f64);
        let d2fx_probe = self
            .numerical_second_derivative(primary_expr, self.probe_x)
            .unwrap_or(0.0_f64);
        let definite_int = self
            .composite_simpson(primary_expr, self.domain_min, self.domain_max, 200)
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

        let tikz_primary = self.substitute_params_for_tikz(primary_expr);
        let tikz_secondary = self.substitute_params_for_tikz(secondary_expr);

        format!(
            r##"% Auto-Generated Symbolic Derivation — econ-calc (Windows 11 ARM64)
\begin{{align*}}
% --- Active Symbolic Functions ---
{func_lines}% --- Structural Parameters ---
{param_lines}% --- Numerical Calculus & Tangency Diagnostics ---
  {primary_sig}\big|_{{x={probe:.3}}} &= {fx_probe:.6}, \quad \frac{{d}}{{dx}}{primary_sig}\big|_{{x={probe:.3}}} = {dfx_probe:.6}, \quad \frac{{d^2}}{{dx^2}}{primary_sig}\big|_{{x={probe:.3}}} = {d2fx_probe:.6} \\
  \int_{{{dmin:.2}}}^{{{dmax:.2}}} {primary_sig}\,dx &\approx {definite_int:.6} \quad (\text{{Composite Simpson }} N=200) \\
  \left(x^*, y^*\right) &= \left({x_star:.5}, {y_star:.5}\right), \quad \lambda^* = {lambda_star:.5}, \quad V^* = {indirect_u:.5} \\
  \nabla F(x^*, y^*) &= \begin{{bmatrix}} {gx:.5} \\ {gy:.5} \end{{bmatrix}}, \quad
  \mathbf{{H}}(x^*, y^*) = \begin{{bmatrix}} {h00:.5} & {h01:.5} \\ {h10:.5} & {h11:.5} \end{{bmatrix}}, \quad
  \det(\mathbf{{H}}) = {det_h:.5e}, \quad \operatorname{{tr}}(\mathbf{{H}}) = {trace_h:.5}
\end{{align*}}

% --- TikZ / pgfplots Reproducible Figure ---
\begin{{tikzpicture}}
\begin{{axis}}[
    title={{Desmos-Positron Economic Equilibrium ({primary_sig} vs {secondary_sig})}},
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
\addplot[only marks, mark=*, mark size=3pt, green!60!black] coordinates {{({x_star:.4}, {y_star:.4})}};
\addlegendentry{{Equilibrium $(x^*, y^*)$}}
\end{{axis}}
\end{{tikzpicture}}

## Theoretical Interpretation & User Markdown Notes

{markdown_notes}
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
            markdown_notes = if self.workspace.markdown_notes.trim().is_empty() {
                "_No custom Markdown notes entered._"
            } else {
                self.workspace.markdown_notes.trim()
            }
        )
    }

    /// Generates the live `model_analysis.R` preview script reflecting the active `Workspace`.
    fn generate_r_script_preview(&self) -> String {
        let (x_star, y_star, lambda_star, indirect_u) = self.solve_structural_equilibrium();
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
        let px = self
            .workspace
            .parameters
            .get("price_x")
            .copied()
            .unwrap_or(4.0_f64);
        let py = self
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
            r##"# ==============================================================================
# econ-calc Reproducible Analysis Script for Posit Positron IDE
# Active Model: {model_title}
# Target Architecture: Windows 11 ARM64 (aarch64-pc-windows-msvc)
# ==============================================================================

suppressPackageStartupMessages({{
  library(tidyverse)
}})

# Load exported numerical simulation grid (50 rows)
sim_data <- read_csv("simulation_data.csv", show_col_types = FALSE)

# Dynamic Symbolic Workspace Parameters
workspace_params <- tibble(
{tibble_body},
  A = {tfp_a:.6},
  alpha = {alpha:.6},
  beta = {beta:.6},
  price_x = {price_x:.6},
  price_y = {price_y:.6},
  income = {income:.6},
  target_output = {target_output:.6},
  x_star = {x_star:.6},
  y_star = {y_star:.6},
  lambda_star = {lambda_star:.6},
  indirect_utility = {indirect_u:.6}
)

print(workspace_params)
glimpse(sim_data)

# Publication-quality ggplot2 visualization
p_main <- ggplot(sim_data, aes(x = x)) +
  geom_line(aes(y = primary_y, color = "Primary Function f(x)"), linewidth = 1.2) +
  geom_line(aes(y = secondary_y, color = "Secondary Function g(x)"), linewidth = 1.0, linetype = "dashed") +
  geom_point(
    data = workspace_params,
    aes(x = x_star, y = y_star, color = "Equilibrium (x*, y*)"),
    size = 3.8
  ) +
  scale_color_manual(
    name = "Series",
    values = c(
      "Primary Function f(x)" = "#38bdf8",
      "Secondary Function g(x)" = "#f97316",
      "Equilibrium (x*, y*)" = "#22c55e"
    )
  ) +
  labs(
    title = "{model_title}",
    subtitle = sprintf("Equilibrium: (x* = %.4f, y* = %.4f) | lambda* = %.4f", workspace_params$x_star, workspace_params$y_star, workspace_params$lambda_star),
    x = "Domain Variable x",
    y = "Evaluated Response y",
    caption = "Exported from econ-calc Dynamic Mathematical IDE (ARM64)"
  ) +
  theme_minimal(base_size = 13)

print(p_main)
ggsave("equilibrium_plot.png", plot = p_main, width = 8.5, height = 5.5, dpi = 300)
"##,
            model_title = self.active_model.title(),
            tibble_body = tibble_body,
            tfp_a = tfp_a,
            alpha = alpha,
            beta = beta,
            price_x = px,
            price_y = py,
            income = income,
            target_output = target_output,
            x_star = x_star,
            y_star = y_star,
            lambda_star = lambda_star,
            indirect_u = indirect_u,
        )
    }

    /// Bundles the active dynamic `Workspace` (including `markdown_notes` and `csv_scratchpad` diagnostics)
    /// into an `ExportPayload` and invokes `export_and_launch_positron`.
    fn trigger_export_pipeline(&mut self) {
        let (x_star, y_star, lambda_star, indirect_utility) = self.solve_structural_equilibrium();
        let mut fn_iter = self.workspace.functions.values();
        let custom_expr = fn_iter.next().cloned().unwrap_or_else(|| "x".to_string());
        let secondary_expr = fn_iter.next().cloned().unwrap_or_else(|| "0.0".to_string());

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

        let csv_pts = self.parse_csv_scratchpad_points();
        let ols_summary = if let Some((b1, b0, r2)) = self.compute_csv_ols(&csv_pts) {
            format!(
                "CSV Scratchpad ({} obs): OLS y = {:.4}*x + {:.4} (R^2 = {:.4})",
                csv_pts.len(),
                b1,
                b0,
                r2
            )
        } else {
            format!("CSV Scratchpad: {} valid (x, y) observation(s)", csv_pts.len())
        };

        let summary_block = format!(
            "Markdown Notes:\n{}\n\nEmpirical Scratchpad Summary:\n{}",
            self.workspace.markdown_notes.trim(),
            ols_summary
        );

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
            game_summary: summary_block,
            latex_derivation: self.generate_latex_derivation(),
            grid_rows: self.build_simulation_grid(50),
        };

        match export_and_launch_positron(&payload) {
            Ok(ExportOutcome {
                export_dir,
                launched_program,
                status_message,
            }) => {
                self.status_banner = format!(
                    "Exported bundle to {:?} via `{}`",
                    export_dir, launched_program
                );
                self.console_logs.push(ConsoleLogEntry {
                    timestamp_tag: "EXPORT".to_string(),
                    source_line: format!(
                        "export_and_launch_positron({})",
                        self.report_format.label()
                    ),
                    message: status_message,
                    is_error: false,
                });
            }
            Err(err) => {
                self.status_banner = format!("Export failed: {}", err);
                self.console_logs.push(ConsoleLogEntry {
                    timestamp_tag: "EXPORT".to_string(),
                    source_line: "export_and_launch_positron".to_string(),
                    message: err,
                    is_error: true,
                });
            }
        }
    }

    // =========================================================================
    // TOP BAR: Workspace Branding, Theme, & Status
    // =========================================================================
    fn render_top_header(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new("econ-calc")
                    .font(FontId::proportional(17.5_f32))
                    .strong()
                    .color(Color32::from_rgb(56, 189, 248)),
            );
            ui.label(
                RichText::new("Desmos × RStudio × Positron Symbolic IDE")
                    .font(FontId::proportional(12.5_f32))
                    .color(Color32::from_rgb(148, 163, 184)),
            );
            ui.separator();

            if ui
                .button(if self.dark_mode {
                    RichText::new("Theme: Positron Dark").color(Color32::from_rgb(125, 211, 252))
                } else {
                    RichText::new("Theme: Positron Light").color(Color32::from_rgb(30, 64, 175))
                })
                .clicked()
            {
                self.dark_mode = !self.dark_mode;
                Self::apply_positron_theme(ui.ctx(), self.dark_mode);
            }

            ui.separator();
            ui.label(RichText::new("Active Template:").color(Color32::from_rgb(148, 163, 184)));
            egui::ComboBox::from_id_source("header_model_kind_combo")
                .selected_text(self.active_model.title())
                .show_ui(ui, |ui| {
                    for kind in [
                        ActiveModelKind::CustomExpression,
                        ActiveModelKind::CobbDouglas,
                        ActiveModelKind::ConstrainedUtility,
                        ActiveModelKind::GameTheory2x2,
                    ] {
                        ui.selectable_value(&mut self.active_model, kind, kind.title());
                    }
                });

            ui.separator();
            ui.label(
                RichText::new(&self.status_banner)
                    .italics()
                    .color(Color32::from_rgb(52, 211, 153)),
            );
        });
    }

    // =========================================================================
    // 1. LEFT PANEL: "The Symbolic Environment"
    // =========================================================================
    fn render_left_symbolic_panel(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("1. Symbolic Environment")
                    .font(FontId::proportional(15.5_f32))
                    .strong()
                    .color(Color32::from_rgb(56, 189, 248)),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(
                        RichText::new("Clear Workspace")
                            .strong()
                            .color(Color32::from_rgb(248, 113, 113)),
                    )
                    .on_hover_text("Clears all symbolic functions, parameters, notes, and CSV data.")
                    .clicked()
                {
                    self.workspace.clear();
                    self.editor_script.clear();
                    self.status_banner = "Cleared symbolic workspace environment.".to_string();
                    self.console_logs.push(ConsoleLogEntry {
                        timestamp_tag: "CLEAR".to_string(),
                        source_line: "workspace.clear()".to_string(),
                        message: "Removed all symbolic functions, parameters, and scratchpad data."
                            .to_string(),
                        is_error: false,
                    });
                }
            });
        });

        ui.separator();

        egui::ScrollArea::vertical()
            .id_source("left_symbolic_env_scroll")
            .show(ui, |ui| {
                // --- Section A: Active Symbolic Functions ---
                ui.label(
                    RichText::new(format!(
                        "Symbolic Functions ({})",
                        self.workspace.functions.len()
                    ))
                    .font(FontId::proportional(13.5_f32))
                    .strong()
                    .color(Color32::from_rgb(165, 180, 252)),
                );
                ui.add_space(4.0_f32);

                if self.workspace.functions.is_empty() {
                    ui.label(
                        RichText::new(
                            "No active functions. Define `f(x) = ...` in the Command Editor below.",
                        )
                        .italics()
                        .color(Color32::from_rgb(148, 163, 184)),
                    );
                } else {
                    let palette = Self::series_palette();
                    let mut remove_fn_key: Option<String> = None;

                    let fn_snapshot: Vec<(String, String)> = self
                        .workspace
                        .functions
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();

                    let card_bg = if self.dark_mode {
                        Color32::from_rgb(20, 29, 49)
                    } else {
                        Color32::from_rgb(241, 245, 249)
                    };

                    for (idx, (sig, expr)) in fn_snapshot.iter().enumerate() {
                        let color = palette[idx % palette.len()];
                        let math_sig = Self::format_symbolic_math(sig);
                        let math_expr = Self::format_symbolic_math(expr);
                        let probe_val = self
                            .eval_expression_str(expr, self.probe_x, 10.0_f64)
                            .ok();

                        egui::Frame::group(ui.style())
                            .fill(card_bg)
                            .stroke(Stroke::new(1.0_f32, color.gamma_multiply(0.55_f32)))
                            .rounding(Rounding::same(6.0_f32))
                            .inner_margin(Margin::same(8.0_f32))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new("ƒ")
                                            .font(FontId::proportional(16.0_f32))
                                            .strong()
                                            .italics()
                                            .color(color),
                                    );
                                    ui.label(
                                        RichText::new(format!("{} =", math_sig))
                                            .font(FontId::monospace(13.5_f32))
                                            .strong()
                                            .color(color),
                                    );
                                    ui.label(
                                        RichText::new(&math_expr)
                                            .font(FontId::proportional(14.0_f32))
                                            .italics()
                                            .color(if self.dark_mode {
                                                Color32::from_rgb(226, 232, 240)
                                            } else {
                                                Color32::from_rgb(15, 23, 42)
                                            }),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui
                                                .small_button(
                                                    RichText::new("×")
                                                        .color(Color32::from_rgb(248, 113, 113)),
                                                )
                                                .on_hover_text("Remove function")
                                                .clicked()
                                            {
                                                remove_fn_key = Some(sig.clone());
                                            }
                                        },
                                    );
                                });

                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(format!("meval: `{}`", expr))
                                            .font(FontId::monospace(11.0_f32))
                                            .color(Color32::from_rgb(148, 163, 184)),
                                    );
                                    if let Some(v) = probe_val {
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    RichText::new(format!(
                                                        "@ x={:.2} → {:.4}",
                                                        self.probe_x, v
                                                    ))
                                                    .font(FontId::monospace(11.5_f32))
                                                    .color(Color32::from_rgb(52, 211, 153)),
                                                );
                                            },
                                        );
                                    }
                                });
                            });
                        ui.add_space(4.0_f32);
                    }

                    if let Some(k) = remove_fn_key {
                        self.workspace.functions.remove(&k);
                    }
                }

                ui.add_space(8.0_f32);
                ui.separator();

                // --- Section B: Dynamic Parameter Sliders ---
                ui.label(
                    RichText::new(format!(
                        "Dynamic Parameters ({} Live Sliders)",
                        self.workspace.parameters.len()
                    ))
                    .font(FontId::proportional(13.5_f32))
                    .strong()
                    .color(Color32::from_rgb(56, 189, 248)),
                );
                ui.add_space(4.0_f32);

                if self.workspace.parameters.is_empty() {
                    ui.label(
                        RichText::new(
                            "No active parameters. Enter `\\alpha = 0.5` in the Command Editor.",
                        )
                        .italics()
                        .color(Color32::from_rgb(148, 163, 184)),
                    );
                } else {
                    let param_bg = if self.dark_mode {
                        Color32::from_rgb(18, 26, 44)
                    } else {
                        Color32::from_rgb(241, 245, 249)
                    };
                    let mut any_slider_changed = false;

                    for (name, val) in &mut self.workspace.parameters {
                        let pretty_name = Self::format_symbolic_math(name);
                        let (min_bound, max_bound) = Self::infer_slider_bounds(name, *val);

                        egui::Frame::group(ui.style())
                            .fill(param_bg)
                            .rounding(Rounding::same(5.0_f32))
                            .inner_margin(Margin::symmetric(8.0_f32, 5.0_f32))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(&pretty_name)
                                            .font(FontId::monospace(13.5_f32))
                                            .strong()
                                            .color(Color32::from_rgb(125, 211, 252)),
                                    );
                                    ui.label(
                                        RichText::new(format!("({})", name))
                                            .font(FontId::monospace(11.0_f32))
                                            .color(Color32::from_rgb(100, 116, 139)),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                RichText::new(format!("{:.4}", *val))
                                                    .font(FontId::monospace(12.5_f32))
                                                    .strong()
                                                    .color(Color32::from_rgb(250, 204, 21)),
                                            );
                                        },
                                    );
                                });
                                let slider_resp = ui.add(
                                    egui::Slider::new(val, min_bound..=max_bound)
                                        .show_value(false)
                                        .step_by(0.005_f64),
                                );
                                if slider_resp.changed() {
                                    any_slider_changed = true;
                                }
                            });
                        ui.add_space(3.0_f32);
                    }

                    if any_slider_changed {
                        ui.ctx().request_repaint();
                    }
                }

                ui.add_space(8.0_f32);
                ui.separator();

                // --- Section C: Domain & Probe Controls ---
                ui.label(
                    RichText::new("Domain & Calculus Probe Controls")
                        .font(FontId::proportional(13.5_f32))
                        .strong()
                        .color(Color32::from_rgb(148, 163, 184)),
                );
                ui.add(
                    egui::Slider::new(&mut self.domain_min, -25.0_f64..=20.0_f64)
                        .text("Domain x_min"),
                );
                ui.add(
                    egui::Slider::new(&mut self.domain_max, 1.0_f64..=150.0_f64)
                        .text("Domain x_max"),
                );
                if self.domain_max <= self.domain_min + 0.25_f64 {
                    self.domain_max = self.domain_min + 0.25_f64;
                }
                ui.add(
                    egui::Slider::new(&mut self.probe_x, self.domain_min..=self.domain_max)
                        .text("Calculus Probe x₀"),
                );
                ui.checkbox(
                    &mut self.show_derivative_overlay,
                    "Overlay Numerical Derivative f'(x)",
                );
                ui.checkbox(
                    &mut self.show_csv_ols_line,
                    "Overlay CSV OLS Regression Line",
                );

                ui.add_space(6.0_f32);
                let (x_star, y_star, lambda_star, indirect_u) = self.solve_structural_equilibrium();
                egui::Frame::group(ui.style())
                    .fill(if self.dark_mode {
                        Color32::from_rgb(14, 34, 48)
                    } else {
                        Color32::from_rgb(220, 252, 231)
                    })
                    .stroke(Stroke::new(1.0_f32, Color32::from_rgb(34, 197, 94)))
                    .rounding(Rounding::same(6.0_f32))
                    .inner_margin(Margin::same(8.0_f32))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new("Solved Equilibrium / Tangency")
                                .strong()
                                .color(Color32::from_rgb(74, 222, 128)),
                        );
                        ui.monospace(format!("x* = {:.5}   |   y* = {:.5}", x_star, y_star));
                        ui.monospace(format!(
                            "λ* = {:.5}   |   V* = {:.5}",
                            lambda_star, indirect_u
                        ));
                    });
            });
    }

    fn infer_slider_bounds(name: &str, current: f64) -> (f64, f64) {
        let lower_name = name.to_lowercase();
        if lower_name.contains("alpha")
            || lower_name.contains("beta")
            || lower_name.contains("delta")
            || lower_name == "s"
            || lower_name == "c_1"
            || lower_name == "n"
        {
            let lo = 0.01_f64.min(current - 0.5_f64);
            let hi = 1.50_f64.max(current + 0.5_f64);
            (lo, hi)
        } else if lower_name == "i" || lower_name.contains("income") || lower_name.contains("y_bar")
        {
            let lo = 1.0_f64.min(current * 0.2_f64);
            let hi = 300.0_f64.max(current * 2.5_f64);
            (lo, hi)
        } else if current.abs() <= 2.0_f64 {
            (-5.0_f64, 10.0_f64)
        } else {
            let span = current.abs() * 2.5_f64;
            ((current - span).min(0.0_f64), current + span)
        }
    }

    fn series_palette() -> [Color32; 6] {
        [
            Color32::from_rgb(56, 189, 248),  // Sky Blue
            Color32::from_rgb(249, 115, 22),  // Orange
            Color32::from_rgb(168, 85, 247),  // Purple
            Color32::from_rgb(52, 211, 153),  // Emerald
            Color32::from_rgb(250, 204, 21),  // Yellow
            Color32::from_rgb(244, 114, 182), // Pink
        ]
    }

    // =========================================================================
    // 2. CENTER PANEL: "The Editor & Viewport" (Split Top/Bottom)
    // =========================================================================
    fn render_center_panel(&mut self, ui: &mut Ui) {
        let total_height = ui.available_height();
        let top_viewport_height = (total_height * 0.54_f32).max(240.0_f32);

        // --- TOP HALF: 2D Desmos Plot & 3D Isometric Surface ---
        ui.allocate_ui(Vec2::new(ui.available_width(), top_viewport_height), |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("2A. Interactive Mathematical Viewport")
                        .font(FontId::proportional(15.5_f32))
                        .strong()
                        .color(Color32::from_rgb(56, 189, 248)),
                );
                ui.separator();
                ui.selectable_value(
                    &mut self.viewport_mode,
                    ViewportMode::Plot2D,
                    RichText::new("2D Desmos Plot (Independent X/Y Zoom)").strong(),
                );
                ui.selectable_value(
                    &mut self.viewport_mode,
                    ViewportMode::Surface3D,
                    RichText::new("3D Isometric Surface").strong(),
                );
                ui.separator();
                ui.label(
                    RichText::new(
                        "Tip: Drag to pan, Scroll to zoom, Ctrl+Scroll (X-axis), Shift+Scroll (Y-axis), Right-drag box zoom",
                    )
                    .font(FontId::proportional(11.5_f32))
                    .color(Color32::from_rgb(148, 163, 184)),
                );
            });
            ui.separator();

            match self.viewport_mode {
                ViewportMode::Plot2D => self.render_2d_desmos_plot(ui),
                ViewportMode::Surface3D => self.render_3d_isometric_surface(ui),
            }
        });

        ui.add_space(4.0_f32);
        ui.separator();

        // --- BOTTOM HALF: Full-Width Code Editor & Output Tabs ---
        self.render_bottom_code_editor(ui);
    }

    fn render_2d_desmos_plot(&mut self, ui: &mut Ui) {
        let palette = Self::series_palette();
        let csv_pts = self.parse_csv_scratchpad_points();
        let ols_fit = if self.show_csv_ols_line {
            self.compute_csv_ols(&csv_pts)
        } else {
            None
        };
        let intersection = self.find_primary_secondary_intersection();

        // Do NOT set view_aspect or data_aspect so X and Y axes stretch/zoom independently like Desmos.
        Plot::new("desmos_2d_symbolic_plot")
            .legend(Legend::default())
            .allow_zoom([true, true])
            .allow_drag([true, true])
            .allow_scroll(true)
            .allow_boxed_zoom(true)
            .show(ui, |plot_ui| {
                // Combine configured domain with visible viewport bounds so panning/zooming never clips curves
                let visible_bounds = plot_ui.plot_bounds();
                let (vis_min_x, vis_max_x) = (visible_bounds.min()[0], visible_bounds.max()[0]);
                let sample_min = if vis_min_x.is_finite() && (vis_max_x - vis_min_x) > 1e-3_f64 {
                    self.domain_min.min(vis_min_x).max(-1e4_f64)
                } else {
                    self.domain_min
                };
                let sample_max = if vis_max_x.is_finite() && (vis_max_x - vis_min_x) > 1e-3_f64 {
                    self.domain_max.max(vis_max_x).min(1e4_f64)
                } else {
                    self.domain_max
                };
                let span = (sample_max - sample_min).max(1e-3_f64);
                let samples = 300_usize;

                // 1. Plot all symbolic functions in workspace.functions
                for (idx, (sig, expr)) in self.workspace.functions.iter().enumerate() {
                    let color = palette[idx % palette.len()];
                    let pts: PlotPoints = (0..=samples)
                        .filter_map(|i| {
                            let x = sample_min + span * (i as f64) / (samples as f64);
                            let y = self.eval_expression_str(expr, x, 10.0_f64).ok()?;
                            (y.is_finite() && y.abs() < 1e6_f64).then_some([x, y])
                        })
                        .collect();

                    let line_style = if idx == 1 {
                        LineStyle::Dashed { length: 8.0_f32 }
                    } else {
                        LineStyle::Solid
                    };

                    plot_ui.line(
                        Line::new(pts)
                            .name(format!("{} = {}", Self::format_symbolic_math(sig), expr))
                            .width(2.4_f32)
                            .style(line_style)
                            .color(color),
                    );
                }

                // 2. Optional numerical derivative f'(x) of primary function
                if self.show_derivative_overlay {
                    if let Some((sig, expr)) = self.workspace.functions.iter().next() {
                        let deriv_pts: PlotPoints = (0..=samples)
                            .filter_map(|i| {
                                let x = sample_min + span * (i as f64) / (samples as f64);
                                let dy = self.numerical_derivative(expr, x).ok()?;
                                (dy.is_finite() && dy.abs() < 1e6_f64).then_some([x, dy])
                            })
                            .collect();

                        plot_ui.line(
                            Line::new(deriv_pts)
                                .name(format!("d/dx {}", sig))
                                .width(1.6_f32)
                                .style(LineStyle::Dotted { spacing: 5.0_f32 })
                                .color(Color32::from_rgb(244, 114, 182)),
                        );
                    }
                }

                // 3. Probe point marker on primary function
                if let Some((sig, expr)) = self.workspace.functions.iter().next() {
                    if let Ok(y_probe) = self.eval_expression_str(expr, self.probe_x, 10.0_f64) {
                        if y_probe.is_finite() {
                            plot_ui.vline(
                                VLine::new(self.probe_x)
                                    .name(format!("Probe x₀ = {:.2}", self.probe_x))
                                    .style(LineStyle::Dotted { spacing: 6.0_f32 })
                                    .color(Color32::from_rgb(148, 163, 184)),
                            );
                            plot_ui.points(
                                Points::new(vec![[self.probe_x, y_probe]])
                                    .name(format!(
                                        "Probe {}({:.2}) = {:.3}",
                                        sig, self.probe_x, y_probe
                                    ))
                                    .radius(5.5_f32)
                                    .shape(MarkerShape::Diamond)
                                    .color(Color32::from_rgb(250, 204, 21)),
                            );
                        }
                    }
                }

                // 4. Intersection / Equilibrium marker between primary & secondary curves
                if let Some((ix, iy)) = intersection {
                    plot_ui.points(
                        Points::new(vec![[ix, iy]])
                            .name(format!("Equilibrium (x*={:.3}, y*={:.3})", ix, iy))
                            .radius(6.5_f32)
                            .shape(MarkerShape::Circle)
                            .color(Color32::from_rgb(34, 197, 94)),
                    );
                }

                // 5. CSV Scratchpad Scatter Plot Overlay + Optional OLS Fit
                if !csv_pts.is_empty() {
                    plot_ui.points(
                        Points::new(csv_pts.clone())
                            .name(format!("CSV Scratchpad Data (n={})", csv_pts.len()))
                            .radius(5.0_f32)
                            .shape(MarkerShape::Cross)
                            .color(Color32::from_rgb(248, 113, 113)),
                    );

                    if let Some((beta1, beta0, r2)) = ols_fit {
                        let x0 = sample_min;
                        let x1 = sample_max;
                        let ols_pts = PlotPoints::from(vec![
                            [x0, beta1 * x0 + beta0],
                            [x1, beta1 * x1 + beta0],
                        ]);
                        plot_ui.line(
                            Line::new(ols_pts)
                                .name(format!(
                                    "CSV OLS Fit: y = {:.3}x + {:.3} (R²={:.3})",
                                    beta1, beta0, r2
                                ))
                                .width(1.6_f32)
                                .style(LineStyle::Dashed { length: 5.0_f32 })
                                .color(Color32::from_rgb(248, 113, 113)),
                        );
                    }
                }
            });
    }

    fn render_3d_isometric_surface(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::Slider::new(&mut self.yaw_deg, -180.0_f32..=180.0_f32).text("Yaw (°)"),
            );
            ui.add(
                egui::Slider::new(&mut self.pitch_deg, 5.0_f32..=85.0_f32).text("Pitch (°)"),
            );
            ui.add(
                egui::Slider::new(&mut self.surface_zoom, 0.4_f32..=2.5_f32).text("Zoom"),
            );
            ui.add(egui::Slider::new(&mut self.surface_resolution, 12..=40).text("Mesh N"));
            ui.checkbox(&mut self.surface_wireframe_only, "Wireframe Only");
            if ui.button("Reset View").clicked() {
                self.yaw_deg = 38.0_f32;
                self.pitch_deg = 28.0_f32;
                self.surface_zoom = 1.0_f32;
            }
        });

        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, Sense::drag());

        if response.dragged() {
            let delta = response.drag_delta();
            self.yaw_deg = (self.yaw_deg + delta.x * 0.45_f32).clamp(-180.0_f32, 180.0_f32);
            self.pitch_deg = (self.pitch_deg + delta.y * 0.35_f32).clamp(5.0_f32, 85.0_f32);
        }

        let painter = ui.painter_at(rect);
        let canvas_bg = if self.dark_mode {
            Color32::from_rgb(10, 15, 29)
        } else {
            Color32::from_rgb(241, 245, 249)
        };
        painter.rect_filled(rect, Rounding::same(6.0_f32), canvas_bg);

        let n = self.surface_resolution.max(8);
        let (x_star, y_star, _, _) = self.solve_structural_equilibrium();
        let max_dom = self.domain_max.max(10.0_f64);

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

        let mut z_grid = vec![vec![0.0_f64; n + 1]; n + 1];
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        for (i, row) in z_grid.iter_mut().enumerate() {
            let x = 0.2_f64 + (max_dom - 0.2_f64) * (i as f64) / (n as f64);
            for (j, cell) in row.iter_mut().enumerate() {
                let y = 0.2_f64 + (max_dom - 0.2_f64) * (j as f64) / (n as f64);
                let z = tfp_a * x.powf(alpha) * y.powf(beta);
                *cell = z;
                if z < z_min {
                    z_min = z;
                }
                if z > z_max {
                    z_max = z;
                }
            }
        }
        let z_span = (z_max - z_min).max(1e-6_f64);

        let yaw = self.yaw_deg.to_radians();
        let pitch = self.pitch_deg.to_radians();
        let (cos_y, sin_y) = (yaw.cos(), yaw.sin());
        let (cos_p, sin_p) = (pitch.cos(), pitch.sin());

        let scale = (rect.width().min(rect.height()) * 0.36_f32) * self.surface_zoom;
        let center = rect.center() + Vec2::new(0.0_f32, rect.height() * 0.08_f32);

        let project = |nx: f32, ny: f32, nz: f32| -> (Pos2, f32) {
            let cx = nx * 2.0_f32 - 1.0_f32;
            let cy = ny * 2.0_f32 - 1.0_f32;
            let cz = nz * 1.5_f32 - 0.5_f32;

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

        struct SurfaceQuad {
            pts: [Pos2; 4],
            depth: f32,
            norm_z: f32,
        }

        let mut quads = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let nx0 = (i as f32) / (n as f32);
                let nx1 = ((i + 1) as f32) / (n as f32);
                let ny0 = (j as f32) / (n as f32);
                let ny1 = ((j + 1) as f32) / (n as f32);

                let nz00 = ((z_grid[i][j] - z_min) / z_span) as f32;
                let nz10 = ((z_grid[i + 1][j] - z_min) / z_span) as f32;
                let nz11 = ((z_grid[i + 1][j + 1] - z_min) / z_span) as f32;
                let nz01 = ((z_grid[i][j + 1] - z_min) / z_span) as f32;

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
            let t = q.norm_z.clamp(0.0_f32, 1.0_f32);
            let r = (30.0_f32 + 205.0_f32 * t) as u8;
            let g = (105.0_f32 + 120.0_f32 * (1.0_f32 - (2.0_f32 * t - 1.0_f32).abs())) as u8;
            let b = (240.0_f32 - 180.0_f32 * t) as u8;

            if !self.surface_wireframe_only {
                painter.add(egui::Shape::convex_polygon(
                    q.pts.to_vec(),
                    Color32::from_rgba_unmultiplied(r, g, b, 215),
                    Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(15, 23, 42, 150)),
                ));
            } else {
                painter.add(egui::Shape::closed_line(
                    q.pts.to_vec(),
                    Stroke::new(1.1_f32, Color32::from_rgb(r, g, b)),
                ));
            }
        }

        let nx_opt = ((x_star / max_dom) as f32).clamp(0.0_f32, 1.0_f32);
        let ny_opt = ((y_star / max_dom) as f32).clamp(0.0_f32, 1.0_f32);
        let z_opt = tfp_a * x_star.max(1e-4_f64).powf(alpha) * y_star.max(1e-4_f64).powf(beta);
        let nz_opt = (((z_opt - z_min) / z_span) as f32).clamp(0.0_f32, 1.0_f32);
        let (opt_pos, _) = project(nx_opt, ny_opt, nz_opt);
        painter.circle_filled(opt_pos, 6.0_f32, Color32::from_rgb(34, 197, 94));
        painter.circle_stroke(opt_pos, 6.0_f32, Stroke::new(2.0_f32, Color32::WHITE));

        painter.text(
            rect.left_top() + Vec2::new(14.0_f32, 10.0_f32),
            egui::Align2::LEFT_TOP,
            format!(
                "3D Surface Z(x,y) = {:.2} * x^{:.2} * y^{:.2} | Tangency ({:.2}, {:.2}, Z*={:.2})",
                tfp_a, alpha, beta, x_star, y_star, z_opt
            ),
            FontId::monospace(12.5_f32),
            Color32::from_rgb(125, 211, 252),
        );
    }

    fn render_bottom_code_editor(&mut self, ui: &mut Ui) {
        // Tab Selector Bar
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new("2B. IDE Editor & Diagnostics")
                    .font(FontId::proportional(15.0_f32))
                    .strong()
                    .color(Color32::from_rgb(56, 189, 248)),
            );
            ui.separator();
            ui.selectable_value(
                &mut self.bottom_tab,
                BottomEditorTab::CommandEditor,
                RichText::new("1. Command Editor").strong(),
            );
            ui.selectable_value(
                &mut self.bottom_tab,
                BottomEditorTab::ConsoleLog,
                RichText::new(format!("2. Console Log ({})", self.console_logs.len())).strong(),
            );
            ui.selectable_value(
                &mut self.bottom_tab,
                BottomEditorTab::LatexDerivation,
                RichText::new("3. LaTeX Derivation").strong(),
            );
            ui.selectable_value(
                &mut self.bottom_tab,
                BottomEditorTab::RScriptPreview,
                RichText::new("4. R Script Preview").strong(),
            );
        });

        ui.separator();

        match self.bottom_tab {
            BottomEditorTab::CommandEditor => {
                // Action Bar: Run / Evaluate button + Insert Snippet dropdown
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button(
                            RichText::new("▶ Run / Evaluate")
                                .font(FontId::proportional(13.5_f32))
                                .strong()
                                .color(Color32::from_rgb(74, 222, 128)),
                        )
                        .on_hover_text(
                            "Parses `name = value` parameters and `f(x) = expr` functions, stripping LaTeX slashes for meval.",
                        )
                        .clicked()
                    {
                        self.evaluate_editor_script();
                    }

                    ui.separator();
                    ui.label(
                        RichText::new("Insert Snippet:")
                            .strong()
                            .color(Color32::from_rgb(148, 163, 184)),
                    );

                    let prev_snippet = self.selected_snippet;
                    egui::ComboBox::from_id_source("insert_snippet_dropdown")
                        .selected_text(self.selected_snippet.label())
                        .show_ui(ui, |ui| {
                            for snip in SnippetTemplate::all() {
                                ui.selectable_value(&mut self.selected_snippet, snip, snip.label());
                            }
                        });

                    if self.selected_snippet != prev_snippet
                        || ui.button("Paste Snippet").clicked()
                    {
                        self.active_model = self.selected_snippet.model_kind();
                        self.editor_script = self.selected_snippet.script_text().to_string();
                        self.workspace.parameters.clear();
                        self.workspace.functions.clear();
                        self.evaluate_editor_script();
                    }

                    ui.separator();
                    ui.label(
                        RichText::new("Syntax: `\\alpha = 0.5`, `f(x) = \\alpha * x^2 + 3`")
                            .font(FontId::monospace(11.5_f32))
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                });

                ui.add_space(4.0_f32);

                egui::ScrollArea::vertical()
                    .id_source("command_editor_scroll")
                    .show(ui, |ui| {
                        ui.add_sized(
                            ui.available_size(),
                            egui::TextEdit::multiline(&mut self.editor_script)
                                .code_editor()
                                .desired_width(f32::INFINITY)
                                .desired_rows(10)
                                .hint_text(
                                    "# Type symbolic parameters and functions:\n\\alpha = 0.5\nf(x) = \\alpha * x^2 + 3",
                                ),
                        );
                    });
            }
            BottomEditorTab::ConsoleLog => {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Execution & Parser Logs")
                            .strong()
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    if ui.small_button("Clear Log").clicked() {
                        self.console_logs.clear();
                    }
                });
                ui.add_space(4.0_f32);

                egui::ScrollArea::vertical()
                    .id_source("console_log_scroll")
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for entry in &self.console_logs {
                            let tag_color = if entry.is_error {
                                Color32::from_rgb(248, 113, 113)
                            } else {
                                Color32::from_rgb(56, 189, 248)
                            };
                            let msg_color = if entry.is_error {
                                Color32::from_rgb(252, 165, 165)
                            } else {
                                Color32::from_rgb(203, 213, 225)
                            };

                            ui.horizontal_wrapped(|ui| {
                                ui.monospace(
                                    RichText::new(format!("[{}]", entry.timestamp_tag))
                                        .strong()
                                        .color(tag_color),
                                );
                                ui.monospace(
                                    RichText::new(format!("`{}` →", entry.source_line))
                                        .color(Color32::from_rgb(148, 163, 184)),
                                );
                                ui.monospace(RichText::new(&entry.message).color(msg_color));
                            });
                        }
                    });
            }
            BottomEditorTab::LatexDerivation => {
                let mut latex_str = self.generate_latex_derivation();
                ui.horizontal(|ui| {
                    if ui
                        .button(
                            RichText::new("Copy LaTeX / TikZ to Clipboard")
                                .strong()
                                .color(Color32::from_rgb(125, 211, 252)),
                        )
                        .clicked()
                    {
                        ui.ctx().copy_text(latex_str.clone());
                        self.status_banner =
                            "Copied auto-generated LaTeX & TikZ derivation to clipboard."
                                .to_string();
                    }
                });
                ui.add_space(4.0_f32);
                egui::ScrollArea::vertical()
                    .id_source("latex_derivation_scroll")
                    .show(ui, |ui| {
                        ui.add_sized(
                            ui.available_size(),
                            egui::TextEdit::multiline(&mut latex_str)
                                .code_editor()
                                .desired_width(f32::INFINITY),
                        );
                    });
            }
            BottomEditorTab::RScriptPreview => {
                let mut r_script_str = self.generate_r_script_preview();
                ui.horizontal(|ui| {
                    if ui
                        .button(
                            RichText::new("Copy model_analysis.R to Clipboard")
                                .strong()
                                .color(Color32::from_rgb(125, 211, 252)),
                        )
                        .clicked()
                    {
                        ui.ctx().copy_text(r_script_str.clone());
                        self.status_banner =
                            "Copied model_analysis.R tidyverse script to clipboard.".to_string();
                    }
                });
                ui.add_space(4.0_f32);
                egui::ScrollArea::vertical()
                    .id_source("r_script_preview_scroll")
                    .show(ui, |ui| {
                        ui.add_sized(
                            ui.available_size(),
                            egui::TextEdit::multiline(&mut r_script_str)
                                .code_editor()
                                .desired_width(f32::INFINITY),
                        );
                    });
            }
        }
    }

    // =========================================================================
    // 3. RIGHT PANEL: "Data & Docs" (Split Top/Bottom)
    // =========================================================================
    fn render_right_data_docs_panel(&mut self, ui: &mut Ui) {
        let total_height = ui.available_height();
        let top_data_height = (total_height * 0.54_f32).max(220.0_f32);

        // --- TOP RIGHT: Tabbed Data Viewer (Simulation Grid & CSV Pasteboard) ---
        ui.allocate_ui(Vec2::new(ui.available_width(), top_data_height), |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("3A. Data Viewer")
                        .font(FontId::proportional(15.0_f32))
                        .strong()
                        .color(Color32::from_rgb(56, 189, 248)),
                );
                ui.separator();
                ui.selectable_value(
                    &mut self.right_data_tab,
                    RightDataTab::SimulationGrid,
                    RichText::new("Tab A: Simulation Grid (50 rows)").strong(),
                );
                ui.selectable_value(
                    &mut self.right_data_tab,
                    RightDataTab::CsvPasteboard,
                    RichText::new("Tab B: CSV Pasteboard").strong(),
                );
            });
            ui.separator();

            match self.right_data_tab {
                RightDataTab::SimulationGrid => {
                    let rows = self.build_simulation_grid(50);
                    let mut fn_keys = self.workspace.functions.keys();
                    let col1_name = fn_keys.next().map(|s| s.as_str()).unwrap_or("f(x)");
                    let col2_name = fn_keys.next().map(|s| s.as_str()).unwrap_or("g(x)");

                    egui::ScrollArea::vertical()
                        .id_source("right_sim_grid_scroll")
                        .show(ui, |ui| {
                            egui::Grid::new("right_50_row_sim_grid")
                                .striped(true)
                                .spacing([12.0_f32, 4.0_f32])
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new("#")
                                            .strong()
                                            .color(Color32::from_rgb(148, 163, 184)),
                                    );
                                    ui.label(
                                        RichText::new("x")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    );
                                    ui.label(
                                        RichText::new(col1_name)
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    );
                                    ui.label(
                                        RichText::new(col2_name)
                                            .strong()
                                            .color(Color32::from_rgb(249, 115, 22)),
                                    );
                                    ui.label(
                                        RichText::new("f'(x)")
                                            .strong()
                                            .color(Color32::from_rgb(168, 85, 247)),
                                    );
                                    ui.end_row();

                                    for r in &rows {
                                        ui.monospace(format!("{:02}", r.index));
                                        ui.monospace(format!("{:.3}", r.x));
                                        ui.monospace(format!("{:.4}", r.primary_y));
                                        ui.monospace(format!("{:.4}", r.secondary_y));
                                        ui.monospace(format!("{:.4}", r.marginal_or_slope));
                                        ui.end_row();
                                    }
                                });
                        });
                }
                RightDataTab::CsvPasteboard => {
                    let pts = self.parse_csv_scratchpad_points();
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("Parsed {} (x, y) rows", pts.len()))
                                .font(FontId::monospace(11.5_f32))
                                .color(Color32::from_rgb(52, 211, 153)),
                        );
                        if let Some((b1, b0, r2)) = self.compute_csv_ols(&pts) {
                            ui.label(
                                RichText::new(format!(
                                    "| OLS: y={:.2}x+{:.2} (R²={:.3})",
                                    b1, b0, r2
                                ))
                                .font(FontId::monospace(11.5_f32))
                                .color(Color32::from_rgb(250, 204, 21)),
                            );
                        }
                    });
                    ui.add_space(4.0_f32);
                    egui::ScrollArea::vertical()
                        .id_source("csv_pasteboard_scroll")
                        .show(ui, |ui| {
                            ui.add_sized(
                                ui.available_size(),
                                egui::TextEdit::multiline(&mut self.workspace.csv_scratchpad)
                                    .code_editor()
                                    .desired_width(f32::INFINITY)
                                    .hint_text("Paste raw `x,y` CSV data from Excel or R..."),
                            );
                        });
                }
            }
        });

        ui.add_space(4.0_f32);
        ui.separator();

        // --- BOTTOM RIGHT: Markdown Notes & Positron / Quarto Export Pipeline ---
        ui.label(
            RichText::new("3B. Theoretical Markdown Notes & Positron Export")
                .font(FontId::proportional(15.0_f32))
                .strong()
                .color(Color32::from_rgb(56, 189, 248)),
        );
        ui.add_space(4.0_f32);

        let notes_height = (ui.available_height() - 78.0_f32).max(80.0_f32);
        egui::ScrollArea::vertical()
            .id_source("markdown_notes_scroll")
            .max_height(notes_height)
            .show(ui, |ui| {
                ui.add_sized(
                    Vec2::new(ui.available_width(), notes_height),
                    egui::TextEdit::multiline(&mut self.workspace.markdown_notes)
                        .desired_width(f32::INFINITY)
                        .hint_text(
                            "Write theoretical interpretations, intuition, and derivations here (bundled into Quarto/RMarkdown report)...",
                        ),
                );
            });

        ui.add_space(6.0_f32);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Format:").strong());
            ui.selectable_value(
                &mut self.report_format,
                ReportFormat::QuartoQmd,
                ReportFormat::QuartoQmd.label(),
            );
            ui.selectable_value(
                &mut self.report_format,
                ReportFormat::RMarkdownRmd,
                ReportFormat::RMarkdownRmd.label(),
            );
        });

        ui.horizontal_wrapped(|ui| {
            if ui
                .button(
                    RichText::new("🚀 Export Quarto/R & Launch Positron")
                        .font(FontId::proportional(13.5_f32))
                        .strong()
                        .color(Color32::from_rgb(74, 222, 128)),
                )
                .on_hover_text(
                    "Exports project.Rproj, simulation_data.csv, model_analysis.R, and report.qmd/.Rmd (with Markdown Notes) and opens Positron IDE.",
                )
                .clicked()
            {
                self.trigger_export_pipeline();
            }
        });
    }
}

impl eframe::App for EconCalcApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("positron_top_header").show(ctx, |ui| {
            self.render_top_header(ui);
        });

        egui::SidePanel::left("symbolic_environment_left_panel")
            .resizable(true)
            .default_width(320.0_f32)
            .min_width(250.0_f32)
            .show(ctx, |ui| {
                self.render_left_symbolic_panel(ui);
            });

        egui::SidePanel::right("data_docs_right_panel")
            .resizable(true)
            .default_width(370.0_f32)
            .min_width(280.0_f32)
            .show(ctx, |ui| {
                self.render_right_data_docs_panel(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_center_panel(ui);
        });
    }
}
