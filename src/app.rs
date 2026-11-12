use crate::exporter::{
    ActiveModelKind, ExportPayload, GridRow, ReportFormat, export_and_launch_positron,
};
use eframe::egui::{
    self, Color32, FontId, Pos2, RichText, Rounding, Sense, Stroke, TextEdit, Ui, Vec2,
};
use egui_plot::{Legend, Line, Plot, PlotPoints, Points, VLine};
use nalgebra::{Matrix2, Vector2};

#[derive(Debug, Clone)]
pub struct ModelParameter {
    pub name: String,
    pub symbol: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
}

impl ModelParameter {
    pub fn new(name: &str, symbol: &str, value: f64, min: f64, max: f64) -> Self {
        Self {
            name: name.to_string(),
            symbol: symbol.to_string(),
            value,
            min,
            max,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveModel {
    CustomCAS,
    CobbDouglas,
    ConstrainedUtility,
    GameTheory,
}

impl ActiveModel {
    pub fn to_exporter_kind(self) -> ActiveModelKind {
        match self {
            ActiveModel::CustomCAS => ActiveModelKind::CustomExpression,
            ActiveModel::CobbDouglas => ActiveModelKind::CobbDouglas,
            ActiveModel::ConstrainedUtility => ActiveModelKind::ConstrainedUtility,
            ActiveModel::GameTheory => ActiveModelKind::GameTheory2x2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ActiveModel::CustomCAS => "Custom CAS Workspace",
            ActiveModel::CobbDouglas => "Cobb-Douglas Production & Isoquant",
            ActiveModel::ConstrainedUtility => "Lagrangian Constrained Utility",
            ActiveModel::GameTheory => "2x2 Normal-Form Game Theory",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SidebarTab {
    ModelExplorer,
    Settings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewportTab {
    Plot2D,
    Surface3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodeConsoleTab {
    CasMathInput,
    LatexOutput,
    RScriptPreview,
}

#[derive(Debug, Clone)]
struct ConsoleEntry {
    command: String,
    output: String,
    is_error: bool,
}

#[derive(Debug, Clone)]
struct GameSolution {
    pure_equilibria: Vec<(usize, usize, String)>,
    mixed_p: Option<f64>,
    mixed_q: Option<f64>,
    expected_u1: Option<f64>,
    expected_u2: Option<f64>,
    summary_text: String,
}

pub struct EconCalcApp {
    // Global IDE & Theme state
    dark_mode: bool,
    ui_scale: f32,
    sidebar_open: bool,
    sidebar_tab: SidebarTab,
    viewport_tab: ViewportTab,
    code_console_tab: CodeConsoleTab,
    report_format: ReportFormat,

    // Generalized Model State
    active_model: ActiveModel,
    active_preset_name: String,
    parameters: Vec<ModelParameter>,

    // New custom parameter builder fields
    new_param_name: String,
    new_param_symbol: String,
    new_param_val: f64,
    new_param_min: f64,
    new_param_max: f64,

    // Symbolic / CAS Editor & Calculus state
    custom_expr: String,
    secondary_expr: String,
    cas_editor_buffer: String,
    show_derivative_curve: bool,
    simpson_intervals: usize,

    // 2x2 Normal-Form Game Matrix state: [row][col] -> (Player 1, Player 2)
    payoff_matrix: [[(f64, f64); 2]; 2],
    row_labels: [String; 2],
    col_labels: [String; 2],

    // 3D Isometric Surface controls
    yaw_deg: f32,
    pitch_deg: f32,
    surface_zoom: f32,
    surface_resolution: usize,
    surface_wireframe_only: bool,

    // Evaluation log & status
    console_history: Vec<ConsoleEntry>,
    status_banner: String,
}

impl EconCalcApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(8.0_f32, 6.0_f32);
        style.visuals = egui::Visuals::dark();
        cc.egui_ctx.set_style(style);

        let mut app = Self {
            dark_mode: true,
            ui_scale: 1.0_f32,
            sidebar_open: true,
            sidebar_tab: SidebarTab::ModelExplorer,
            viewport_tab: ViewportTab::Plot2D,
            code_console_tab: CodeConsoleTab::CasMathInput,
            report_format: ReportFormat::QuartoQmd,

            active_model: ActiveModel::ConstrainedUtility,
            active_preset_name: "Constrained Utility Maximization".to_string(),
            parameters: Vec::new(),

            new_param_name: "Tax Rate".to_string(),
            new_param_symbol: "tau".to_string(),
            new_param_val: 0.15,
            new_param_min: 0.0,
            new_param_max: 1.0,

            custom_expr: "A * x^alpha - price_x * x".to_string(),
            secondary_expr: "0.5 * x^1.4".to_string(),
            cas_editor_buffer: String::new(),
            show_derivative_curve: true,
            simpson_intervals: 200,

            payoff_matrix: [[(4.0, 4.0), (1.0, 6.0)], [(6.0, 1.0), (2.0, 2.0)]],
            row_labels: ["Cooperate (U)".to_string(), "Defect (D)".to_string()],
            col_labels: ["Cooperate (L)".to_string(), "Defect (R)".to_string()],

            yaw_deg: 38.0_f32,
            pitch_deg: 28.0_f32,
            surface_zoom: 1.0_f32,
            surface_resolution: 26,
            surface_wireframe_only: false,

            console_history: Vec::new(),
            status_banner:
                "Ready — Native Windows 11 ARM64 (aarch64-pc-windows-msvc) Computational IDE"
                    .to_string(),
        };

        app.load_model_workspace(
            ActiveModel::ConstrainedUtility,
            "Standard Marshallian Consumer",
        );

        app.console_history.push(ConsoleEntry {
            command: ":init".to_string(),
            output: "Initialized generalized parameter environment. Edit formulas in the CAS tab and click 'Run / Evaluate'.".to_string(),
            is_error: false,
        });

        app
    }

    /// Initializes the dynamic `Vec<ModelParameter>` and default CAS buffer for a chosen model preset.
    fn load_model_workspace(&mut self, model: ActiveModel, preset_label: &str) {
        self.active_model = model;
        self.active_preset_name = preset_label.to_string();
        self.parameters.clear();

        match model {
            ActiveModel::ConstrainedUtility => {
                self.parameters.push(ModelParameter::new(
                    "Utility Scale (TFP)",
                    "A",
                    2.0,
                    0.2,
                    10.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Preference Weight X",
                    "alpha",
                    0.45,
                    0.05,
                    0.95,
                ));
                self.parameters.push(ModelParameter::new(
                    "Preference Weight Y",
                    "beta",
                    0.55,
                    0.05,
                    0.95,
                ));
                self.parameters.push(ModelParameter::new(
                    "Price of Good X",
                    "price_x",
                    4.0,
                    0.5,
                    25.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Price of Good Y",
                    "price_y",
                    5.0,
                    0.5,
                    25.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Budget Income",
                    "I",
                    100.0,
                    10.0,
                    500.0,
                ));

                self.custom_expr = "A * x^alpha * ((I - price_x * x) / price_y)^beta".to_string();
                self.secondary_expr = "(I - price_x * x) / price_y".to_string();
                self.cas_editor_buffer = format!(
                    "# Constrained Utility CAS Script\n# Variables bound from Environment: A, alpha, beta, price_x, price_y, I, x\nf(x) = {}\ng(x) = {}\neval A * x^alpha at 8.0\ndiff {} at 8.0\nhessian",
                    self.custom_expr, self.secondary_expr, self.custom_expr
                );
            }
            ActiveModel::CobbDouglas => {
                self.parameters.push(ModelParameter::new(
                    "Total Factor Productivity",
                    "A",
                    2.2,
                    0.2,
                    10.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Labor Elasticity",
                    "alpha",
                    0.40,
                    0.05,
                    0.95,
                ));
                self.parameters.push(ModelParameter::new(
                    "Capital Elasticity",
                    "beta",
                    0.60,
                    0.05,
                    0.95,
                ));
                self.parameters.push(ModelParameter::new(
                    "Wage Rate (w)",
                    "price_x",
                    4.0,
                    0.5,
                    25.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Capital Rental (r)",
                    "price_y",
                    5.0,
                    0.5,
                    25.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Target Isoquant Output",
                    "Y_bar",
                    22.0,
                    2.0,
                    120.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Probe Labor Input",
                    "L",
                    10.0,
                    0.5,
                    40.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Probe Capital Input",
                    "K",
                    12.0,
                    0.5,
                    40.0,
                ));

                self.custom_expr = "(Y_bar / (A * x^alpha))^(1 / beta)".to_string();
                self.secondary_expr = "A * x^alpha * K^beta".to_string();
                self.cas_editor_buffer = format!(
                    "# Cobb-Douglas Production & Isoquant CAS Script\nf(x) = {}\ng(x) = {}\neval A * L^alpha * K^beta at 10.0\ndiff A * x^alpha * K^beta at 10.0\nhessian",
                    self.custom_expr, self.secondary_expr
                );
            }
            ActiveModel::GameTheory => {
                self.parameters.push(ModelParameter::new(
                    "Row Belief (Col Plays L)",
                    "q_probe",
                    0.50,
                    0.0,
                    1.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Col Belief (Row Plays U)",
                    "p_probe",
                    0.50,
                    0.0,
                    1.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Discount Factor",
                    "delta",
                    0.90,
                    0.0,
                    0.99,
                ));

                let m = &self.payoff_matrix;
                self.custom_expr = format!("x * ({:.2}) + (1 - x) * ({:.2})", m[0][0].0, m[0][1].0);
                self.secondary_expr =
                    format!("x * ({:.2}) + (1 - x) * ({:.2})", m[1][0].0, m[1][1].0);
                self.cas_editor_buffer = format!(
                    "# 2x2 Normal-Form Game Expected Payoff Analyzer\nf(x) = {}\ng(x) = {}\nroot ({}) - ({}) near 0.5",
                    self.custom_expr,
                    self.secondary_expr,
                    self.custom_expr,
                    self.secondary_expr
                );
            }
            ActiveModel::CustomCAS => {
                self.parameters.push(ModelParameter::new(
                    "Scale Parameter A",
                    "A",
                    2.5,
                    0.1,
                    15.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Exponent Alpha",
                    "alpha",
                    0.65,
                    0.05,
                    2.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Exponent Beta",
                    "beta",
                    0.35,
                    0.05,
                    2.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Unit Cost / Price",
                    "price_x",
                    1.8,
                    0.1,
                    20.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Domain Min (x_min)",
                    "x_min",
                    0.1,
                    -20.0,
                    10.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Domain Max (x_max)",
                    "x_max",
                    25.0,
                    2.0,
                    100.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Evaluation Probe (x0)",
                    "x0",
                    6.0,
                    0.1,
                    25.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Integral Lower (a)",
                    "int_a",
                    1.0,
                    0.1,
                    25.0,
                ));
                self.parameters.push(ModelParameter::new(
                    "Integral Upper (b)",
                    "int_b",
                    12.0,
                    0.5,
                    25.0,
                ));

                self.custom_expr = "A * x^alpha - price_x * x".to_string();
                self.secondary_expr = "0.4 * x^1.35".to_string();
                self.cas_editor_buffer = format!(
                    "# Custom CAS Mathematical Workspace\n# Define primary f(x) and secondary g(x), or run calculus queries:\nf(x) = {}\ng(x) = {}\neval {} at 6.0\ndiff {} at 6.0\nint {} from 1.0 to 12.0\nroot {} near 8.0",
                    self.custom_expr,
                    self.secondary_expr,
                    self.custom_expr,
                    self.custom_expr,
                    self.custom_expr,
                    self.custom_expr
                );
            }
        }
    }

    /// Looks up a parameter value by symbol from the generalized parameter vector, returning `default` if absent.
    fn get_param(&self, symbol: &str, default: f64) -> f64 {
        for p in &self.parameters {
            if p.symbol == symbol {
                return p.value;
            }
        }
        default
    }

    /// Sets a parameter value by symbol if present, or inserts a new parameter if absent.
    fn set_param(&mut self, symbol: &str, value: f64) {
        for p in &mut self.parameters {
            if p.symbol == symbol {
                p.value = value.clamp(p.min, p.max);
                return;
            }
        }
    }

    /// Evaluates a single-variable expression `f(x)` with all active `ModelParameter` symbols dynamically bound into `meval::Context`.
    fn eval_expr_at(&self, expr_str: &str, x_val: f64) -> Result<f64, String> {
        let mut ctx = meval::Context::new();
        ctx.var("x", x_val);

        // Provide fallback standard economic aliases before overriding with active model parameters
        ctx.var("A", 2.0);
        ctx.var("alpha", 0.5);
        ctx.var("beta", 0.5);
        ctx.var("price_x", 4.0);
        ctx.var("price_y", 5.0);
        ctx.var("I", 100.0);
        ctx.var("Y_bar", 20.0);
        ctx.var("K", 10.0);
        ctx.var("L", 10.0);

        for param in &self.parameters {
            ctx.var(&param.symbol, param.value);
        }

        meval::eval_str_with_context(expr_str, &ctx).map_err(|e| e.to_string())
    }

    /// Central finite difference first derivative f'(x).
    fn numerical_derivative(&self, expr_str: &str, x: f64) -> Result<f64, String> {
        let h = 1e-5_f64.max(x.abs() * 1e-6);
        let f_plus = self.eval_expr_at(expr_str, x + h)?;
        let f_minus = self.eval_expr_at(expr_str, x - h)?;
        Ok((f_plus - f_minus) / (2.0 * h))
    }

    /// Central finite difference second derivative f''(x).
    fn numerical_second_derivative(&self, expr_str: &str, x: f64) -> Result<f64, String> {
        let h = 1e-4_f64.max(x.abs() * 1e-5);
        let f_plus = self.eval_expr_at(expr_str, x + h)?;
        let f_mid = self.eval_expr_at(expr_str, x)?;
        let f_minus = self.eval_expr_at(expr_str, x - h)?;
        Ok((f_plus - 2.0 * f_mid + f_minus) / (h * h))
    }

    /// Composite Simpson's Rule definite integral on [a, b].
    fn composite_simpson(&self, expr_str: &str, a: f64, b: f64, n: usize) -> Result<f64, String> {
        let steps = if n % 2 == 0 { n.max(4) } else { n + 1 };
        let h = (b - a) / (steps as f64);
        let mut sum = self.eval_expr_at(expr_str, a)? + self.eval_expr_at(expr_str, b)?;

        for i in 1..steps {
            let x = a + (i as f64) * h;
            let fx = self.eval_expr_at(expr_str, x)?;
            if !fx.is_finite() {
                return Err(format!("Non-finite integrand evaluation at x = {:.4}", x));
            }
            if i % 2 == 1 {
                sum += 4.0 * fx;
            } else {
                sum += 2.0 * fx;
            }
        }
        Ok(sum * h / 3.0)
    }

    /// Hybrid Newton-Raphson root solver with bracketed bisection fallback.
    fn solve_root(&self, expr_str: &str, initial_guess: f64) -> Result<f64, String> {
        let mut x = initial_guess;
        for _ in 0..60 {
            let fx = self.eval_expr_at(expr_str, x)?;
            if fx.abs() < 1e-9 {
                return Ok(x);
            }
            let dfx = self.numerical_derivative(expr_str, x)?;
            if dfx.abs() < 1e-12 || !dfx.is_finite() {
                break;
            }
            let x_next = x - fx / dfx;
            if !x_next.is_finite() {
                break;
            }
            if (x_next - x).abs() < 1e-9 {
                return Ok(x_next);
            }
            x = x_next;
        }

        let dom_min = self.get_param("x_min", 0.05);
        let dom_max = self.get_param("x_max", 25.0).max(dom_min + 1.0);
        let samples = 200;
        let step = (dom_max - dom_min) / (samples as f64);
        let mut prev_x = dom_min;
        let mut prev_f = self.eval_expr_at(expr_str, prev_x)?;

        for i in 1..=samples {
            let curr_x = dom_min + (i as f64) * step;
            if let Ok(curr_f) = self.eval_expr_at(expr_str, curr_x) {
                if prev_f.is_finite() && curr_f.is_finite() && prev_f * curr_f <= 0.0 {
                    let mut lo = prev_x;
                    let mut hi = curr_x;
                    let mut f_lo = prev_f;
                    for _ in 0..60 {
                        let mid = 0.5 * (lo + hi);
                        let f_mid = self.eval_expr_at(expr_str, mid)?;
                        if f_mid.abs() < 1e-9 || (hi - lo).abs() < 1e-9 {
                            return Ok(mid);
                        }
                        if f_lo * f_mid <= 0.0 {
                            hi = mid;
                        } else {
                            lo = mid;
                            f_lo = f_mid;
                        }
                    }
                }
                prev_x = curr_x;
                prev_f = curr_f;
            }
        }

        Err("No root converged via Newton-Raphson or bracketed bisection on domain.".to_string())
    }

    /// Computes analytical Marshallian demand and Lagrangian equilibrium from dynamic parameters.
    fn solve_constrained_equilibrium(&self) -> (f64, f64, f64, f64) {
        let tfp_a = self.get_param("A", 2.0).max(1e-6);
        let alpha = self.get_param("alpha", 0.45).max(1e-6);
        let beta = self.get_param("beta", 0.55).max(1e-6);
        let sum_elast = (alpha + beta).max(1e-9);
        let px = self.get_param("price_x", 4.0).max(1e-6);
        let py = self.get_param("price_y", 5.0).max(1e-6);
        let income = self.get_param("I", 100.0).max(1e-6);
        let target_output = self.get_param("Y_bar", 20.0).max(1e-6);

        match self.active_model {
            ActiveModel::CobbDouglas => {
                let ratio = (alpha / beta).max(1e-6) * (py / px);
                let l_star = ((target_output / tfp_a) * ratio.powf(beta)).powf(1.0 / sum_elast);
                let k_star = ((target_output / tfp_a) * ratio.powf(-alpha)).powf(1.0 / sum_elast);
                let min_cost = px * l_star + py * k_star;
                let mp_l =
                    tfp_a * alpha * l_star.max(1e-6).powf(alpha - 1.0) * k_star.max(1e-6).powf(beta);
                let lambda_mc = if mp_l.abs() > 1e-9 { px / mp_l } else { 0.0 };
                (l_star, k_star, lambda_mc, min_cost)
            }
            _ => {
                let x_star = (alpha / sum_elast) * (income / px);
                let y_star = (beta / sum_elast) * (income / py);
                let indirect_u =
                    tfp_a * x_star.max(1e-9).powf(alpha) * y_star.max(1e-9).powf(beta);
                let mu_x = tfp_a
                    * alpha
                    * x_star.max(1e-9).powf(alpha - 1.0)
                    * y_star.max(1e-9).powf(beta);
                let lambda_star = mu_x / px;
                (x_star, y_star, lambda_star, indirect_u)
            }
        }
    }

    /// Uses `nalgebra::Matrix2` to compute the Hessian matrix, determinant, and trace.
    fn compute_cobb_douglas_hessian(&self, x: f64, y: f64) -> (Matrix2<f64>, f64, f64) {
        let tfp_a = self.get_param("A", 2.0);
        let alpha = self.get_param("alpha", 0.45);
        let beta = self.get_param("beta", 0.55);
        let x_safe = x.max(1e-4);
        let y_safe = y.max(1e-4);
        let f_val = tfp_a * x_safe.powf(alpha) * y_safe.powf(beta);
        let f_xx = alpha * (alpha - 1.0) * f_val / (x_safe * x_safe);
        let f_yy = beta * (beta - 1.0) * f_val / (y_safe * y_safe);
        let f_xy = alpha * beta * f_val / (x_safe * y_safe);

        let hessian = Matrix2::new(f_xx, f_xy, f_xy, f_yy);
        let det = hessian.determinant();
        let trace = hessian.trace();
        (hessian, det, trace)
    }

    /// Solves the 2x2 Normal-Form Game for Pure Strategy and Mixed Strategy Nash Equilibria.
    fn solve_game_2x2(&self) -> GameSolution {
        let m = &self.payoff_matrix;
        let mut pure_equilibria = Vec::new();
        let eps = 1e-9;

        for (r, row_label) in self.row_labels.iter().enumerate() {
            for (c, col_label) in self.col_labels.iter().enumerate() {
                let u1 = m[r][c].0;
                let u1_alt = m[1 - r][c].0;
                let u2 = m[r][c].1;
                let u2_alt = m[r][1 - c].1;

                if u1 + eps >= u1_alt && u2 + eps >= u2_alt {
                    pure_equilibria.push((
                        r,
                        c,
                        format!(
                            "({}, {}) -> Payoffs ({:.2}, {:.2})",
                            row_label, col_label, u1, u2
                        ),
                    ));
                }
            }
        }

        let p1_vec = Vector2::new(m[0][0].0 - m[1][0].0, m[1][1].0 - m[0][1].0);
        let p2_vec = Vector2::new(m[0][0].1 - m[0][1].1, m[1][1].1 - m[1][0].1);

        let denom_q = p1_vec.x + p1_vec.y;
        let denom_p = p2_vec.x + p2_vec.y;

        let mixed_q = if denom_q.abs() > 1e-9 {
            let q = p1_vec.y / denom_q;
            if (0.0..=1.0).contains(&q) {
                Some(q)
            } else {
                None
            }
        } else {
            None
        };

        let mixed_p = if denom_p.abs() > 1e-9 {
            let p = p2_vec.y / denom_p;
            if (0.0..=1.0).contains(&p) {
                Some(p)
            } else {
                None
            }
        } else {
            None
        };

        let (expected_u1, expected_u2) = match (mixed_p, mixed_q) {
            (Some(p), Some(q)) => {
                let eu1 = p * q * m[0][0].0
                    + p * (1.0 - q) * m[0][1].0
                    + (1.0 - p) * q * m[1][0].0
                    + (1.0 - p) * (1.0 - q) * m[1][1].0;
                let eu2 = p * q * m[0][0].1
                    + p * (1.0 - q) * m[0][1].1
                    + (1.0 - p) * q * m[1][0].1
                    + (1.0 - p) * (1.0 - q) * m[1][1].1;
                (Some(eu1), Some(eu2))
            }
            _ => (None, None),
        };

        let psne_str = if pure_equilibria.is_empty() {
            "None (No pure strategy saddle point)".to_string()
        } else {
            pure_equilibria
                .iter()
                .map(|(_, _, label)| label.clone())
                .collect::<Vec<_>>()
                .join("; ")
        };

        let msne_str = match (mixed_p, mixed_q, expected_u1, expected_u2) {
            (Some(p), Some(q), Some(eu1), Some(eu2)) => format!(
                "p* (Row U) = {:.4}, q* (Col L) = {:.4} | E[U1] = {:.4}, E[U2] = {:.4}",
                p, q, eu1, eu2
            ),
            _ => "No interior Mixed Strategy Nash Equilibrium in (0, 1)".to_string(),
        };

        let summary_text = format!("PSNE: {}\nMSNE: {}", psne_str, msne_str);

        GameSolution {
            pure_equilibria,
            mixed_p,
            mixed_q,
            expected_u1,
            expected_u2,
            summary_text,
        }
    }

    /// Generates the numerical simulation table for the RStudio-style dataframe viewer and CSV exporter.
    fn build_simulation_grid(&self, rows_count: usize) -> Vec<GridRow> {
        let mut grid = Vec::with_capacity(rows_count);
        let (x_star, y_star, _, indirect_u) = self.solve_constrained_equilibrium();
        let tfp_a = self.get_param("A", 2.0).max(1e-6);
        let alpha = self.get_param("alpha", 0.45).max(1e-6);
        let beta = self.get_param("beta", 0.55).max(1e-6);
        let px = self.get_param("price_x", 4.0).max(1e-6);
        let py = self.get_param("price_y", 5.0).max(1e-6);
        let income = self.get_param("I", 100.0).max(1e-6);
        let target_output = self.get_param("Y_bar", 20.0).max(1e-6);

        match self.active_model {
            ActiveModel::CustomCAS => {
                let dom_min = self.get_param("x_min", 0.1);
                let dom_max = self.get_param("x_max", 25.0).max(dom_min + 0.5);
                let span = (dom_max - dom_min).max(1e-3);
                for i in 0..rows_count {
                    let x = dom_min + span * (i as f64) / ((rows_count - 1).max(1) as f64);
                    let y1 = self.eval_expr_at(&self.custom_expr, x).unwrap_or(f64::NAN);
                    let y2 = self
                        .eval_expr_at(&self.secondary_expr, x)
                        .unwrap_or(f64::NAN);
                    let dy = self
                        .numerical_derivative(&self.custom_expr, x)
                        .unwrap_or(f64::NAN);
                    grid.push(GridRow {
                        index: i + 1,
                        x,
                        primary_y: y1,
                        secondary_y: y2,
                        marginal_or_slope: dy,
                    });
                }
            }
            ActiveModel::CobbDouglas => {
                let max_l = (x_star * 2.4).max(15.0);
                let min_cost = px * x_star + py * y_star;

                for i in 0..rows_count {
                    let l = 0.25 + (max_l - 0.25) * (i as f64) / ((rows_count - 1).max(1) as f64);
                    let k_isoquant = (target_output / (tfp_a * l.powf(alpha)))
                        .max(0.0)
                        .powf(1.0 / beta);
                    let k_isocost = ((min_cost - px * l) / py).max(0.0);
                    let mrts = (alpha / beta) * (k_isoquant / l);
                    grid.push(GridRow {
                        index: i + 1,
                        x: l,
                        primary_y: k_isoquant,
                        secondary_y: k_isocost,
                        marginal_or_slope: -mrts,
                    });
                }
            }
            ActiveModel::ConstrainedUtility => {
                let max_x = (income / px) * 1.15;
                let u_bar = indirect_u.max(1e-6);

                for i in 0..rows_count {
                    let x = 0.15 + (max_x - 0.15) * (i as f64) / ((rows_count - 1).max(1) as f64);
                    let y_indiff = (u_bar / (tfp_a * x.powf(alpha)))
                        .max(0.0)
                        .powf(1.0 / beta);
                    let y_budget = ((income - px * x) / py).max(0.0);
                    let mrs = (alpha / beta) * (y_indiff / x);
                    grid.push(GridRow {
                        index: i + 1,
                        x,
                        primary_y: y_indiff,
                        secondary_y: y_budget,
                        marginal_or_slope: -mrs,
                    });
                }
            }
            ActiveModel::GameTheory => {
                let m = &self.payoff_matrix;
                for i in 0..rows_count {
                    let q = (i as f64) / ((rows_count - 1).max(1) as f64);
                    let eu1_up = q * m[0][0].0 + (1.0 - q) * m[0][1].0;
                    let eu1_down = q * m[1][0].0 + (1.0 - q) * m[1][1].0;
                    grid.push(GridRow {
                        index: i + 1,
                        x: q,
                        primary_y: eu1_up,
                        secondary_y: eu1_down,
                        marginal_or_slope: eu1_up - eu1_down,
                    });
                }
            }
        }

        grid
    }

    /// Generates publication-grade LaTeX & TikZ derivation for the LaTeX Output tab and exporter.
    fn generate_latex_and_tikz(&self) -> String {
        let (x_star, y_star, lambda_star, obj_star) = self.solve_constrained_equilibrium();
        let tfp_a = self.get_param("A", 2.0);
        let alpha = self.get_param("alpha", 0.45);
        let beta = self.get_param("beta", 0.55);
        let px = self.get_param("price_x", 4.0);
        let py = self.get_param("price_y", 5.0);
        let income = self.get_param("I", 100.0);
        let target_output = self.get_param("Y_bar", 20.0);

        match self.active_model {
            ActiveModel::CustomCAS => {
                let x0 = self.get_param("x0", 6.0);
                let int_a = self.get_param("int_a", 1.0);
                let int_b = self.get_param("int_b", 12.0);
                let fx = self.eval_expr_at(&self.custom_expr, x0).unwrap_or(0.0);
                let dfx = self
                    .numerical_derivative(&self.custom_expr, x0)
                    .unwrap_or(0.0);
                let d2fx = self
                    .numerical_second_derivative(&self.custom_expr, x0)
                    .unwrap_or(0.0);
                let integral = self
                    .composite_simpson(&self.custom_expr, int_a, int_b, self.simpson_intervals)
                    .unwrap_or(0.0);

                format!(
                    r#"% Generated by econ-calc (Windows 11 ARM64)
\begin{{align*}}
f(x) &= \text{{{expr}}} \\
g(x) &= \text{{{gexpr}}} \\
f({x0:.4}) &= {fx:.6}, \quad f'({x0:.4}) = {dfx:.6}, \quad f''({x0:.4}) = {d2fx:.6} \\
\int_{{{a:.2}}}^{{{b:.2}}} f(x)\,dx &\approx {integral:.6} \quad (\text{{Composite Simpson }} N={n})
\end{{align*}}
"#,
                    expr = self.custom_expr,
                    gexpr = self.secondary_expr,
                    x0 = x0,
                    fx = fx,
                    dfx = dfx,
                    d2fx = d2fx,
                    a = int_a,
                    b = int_b,
                    integral = integral,
                    n = self.simpson_intervals
                )
            }
            ActiveModel::CobbDouglas => {
                let rts = alpha + beta;
                format!(
                    r#"% Cobb-Douglas Production & Isoquant Cost Minimization (econ-calc)
\begin{{align*}}
Y(L, K) &= {a:.3} L^{{{alpha:.3}}} K^{{{beta:.3}}}, \quad \text{{Returns to Scale: }} \alpha + \beta = {rts:.3} \\
\text{{MRTS}}_{{L,K}} &= \frac{{MP_L}}{{MP_K}} = \frac{{\alpha K}}{{\beta L}} = \frac{{w}}{{r}} = \frac{{{px:.2}}}{{{py:.2}}} \\
\left(L^*, K^*\right) &= \left({x_star:.4}, {y_star:.4}\right), \quad C^*(\bar{{Y}} = {ybar:.2}) = {obj_star:.4}, \quad \lambda^* = {lambda_star:.4}
\end{{align*}}

\begin{{tikzpicture}}
\begin{{axis}}[
    title={{Cobb-Douglas Isoquant \& Isocost Tangency ($\bar{{Y}} = {ybar:.2}$)}},
    xlabel={{Labor ($L$)}},
    ylabel={{Capital ($K$)}},
    domain=0.5:{xmax:.2},
    samples=100,
    ymin=0, ymax={ymax:.2},
    legend pos=north east,
    grid=major
]
\addplot[blue, thick] {{ ({ybar:.4} / ({a:.4} * x^({alpha:.4})))^(1 / {beta:.4}) }};
\addlegendentry{{Isoquant $\bar{{Y}} = {ybar:.2}$}}
\addplot[red, dashed, thick] {{ ({obj_star:.4} - {px:.4}*x) / {py:.4} }};
\addlegendentry{{Isocost $C^* = {obj_star:.2}$}}
\addplot[only marks, mark=*, mark size=2.8pt, green!60!black] coordinates {{({x_star:.4}, {y_star:.4})}};
\addlegendentry{{Cost-Minimizing Bundle $(L^*, K^*)$}}
\end{{axis}}
\end{{tikzpicture}}
"#,
                    a = tfp_a,
                    alpha = alpha,
                    beta = beta,
                    rts = rts,
                    px = px,
                    py = py,
                    x_star = x_star,
                    y_star = y_star,
                    ybar = target_output,
                    obj_star = obj_star,
                    lambda_star = lambda_star,
                    xmax = (x_star * 2.2).max(10.0),
                    ymax = (y_star * 2.2).max(10.0)
                )
            }
            ActiveModel::ConstrainedUtility => {
                format!(
                    r#"% Constrained Utility Maximization Lagrangian (econ-calc)
\begin{{align*}}
\mathcal{{L}}(x, y, \lambda) &= {a:.3} x^{{{alpha:.3}}} y^{{{beta:.3}}} + \lambda \left( {income:.2} - {px:.2}x - {py:.2}y \right) \\
\frac{{\partial \mathcal{{L}}}}{{\partial x}} &= {a:.3} ({alpha:.3}) x^{{{alpha_m1:.3}}} y^{{{beta:.3}}} - \lambda ({px:.2}) = 0 \\
\frac{{\partial \mathcal{{L}}}}{{\partial y}} &= {a:.3} ({beta:.3}) x^{{{alpha:.3}}} y^{{{beta_m1:.3}}} - \lambda ({py:.2}) = 0 \\
\text{{MRS}}_{{x,y}} &= \frac{{\alpha y}}{{\beta x}} = \frac{{P_x}}{{P_y}} \implies x^* = \frac{{\alpha}}{{\alpha+\beta}}\frac{{I}}{{P_x}} = {x_star:.4}, \quad y^* = \frac{{\beta}}{{\alpha+\beta}}\frac{{I}}{{P_y}} = {y_star:.4} \\
V(P_x, P_y, I) &= {obj_star:.4}, \quad \lambda^* = {lambda_star:.4}
\end{{align*}}

\begin{{tikzpicture}}
\begin{{axis}}[
    title={{Marshallian Demand \& Indifference Tangency}},
    xlabel={{Good $x$}},
    ylabel={{Good $y$}},
    domain=0.5:{xmax:.2},
    samples=100,
    ymin=0, ymax={ymax:.2},
    legend pos=north east,
    grid=major
]
\addplot[blue, thick] {{ ({obj_star:.4} / ({a:.4} * x^({alpha:.4})))^(1 / {beta:.4}) }};
\addlegendentry{{Indifference Curve $U^* = {obj_star:.2}$}}
\addplot[red, dashed, thick] {{ ({income:.4} - {px:.4}*x) / {py:.4} }};
\addlegendentry{{Budget Line $P_x x + P_y y = {income:.1}$}}
\addplot[only marks, mark=*, mark size=2.8pt, green!60!black] coordinates {{({x_star:.4}, {y_star:.4})}};
\addlegendentry{{Equilibrium $(x^*, y^*)$}}
\end{{axis}}
\end{{tikzpicture}}
"#,
                    a = tfp_a,
                    alpha = alpha,
                    beta = beta,
                    alpha_m1 = alpha - 1.0,
                    beta_m1 = beta - 1.0,
                    income = income,
                    px = px,
                    py = py,
                    x_star = x_star,
                    y_star = y_star,
                    obj_star = obj_star,
                    lambda_star = lambda_star,
                    xmax = (income / px.max(0.1)) * 1.1,
                    ymax = (income / py.max(0.1)) * 1.2
                )
            }
            ActiveModel::GameTheory => {
                let sol = self.solve_game_2x2();
                let m = &self.payoff_matrix;
                format!(
                    r#"% 2x2 Normal-Form Game Payoff Matrix & Nash Equilibria (econ-calc)
\begin{{table}}[htbp]
\centering
\begin{{tabular}}{{l|cc}}
\toprule
\textbf{{Player 1 \textbackslash\ Player 2}} & \textbf{{{col0}}} & \textbf{{{col1}}} \\
\midrule
\textbf{{{row0}}} & $({u00_1:.2}, {u00_2:.2})$ & $({u01_1:.2}, {u01_2:.2})$ \\
\textbf{{{row1}}} & $({u10_1:.2}, {u10_2:.2})$ & $({u11_1:.2}, {u11_2:.2})$ \\
\bottomrule
\end{{tabular}}
\caption{{{summary}}}
\end{{table}}
"#,
                    col0 = self.col_labels[0],
                    col1 = self.col_labels[1],
                    row0 = self.row_labels[0],
                    row1 = self.row_labels[1],
                    u00_1 = m[0][0].0,
                    u00_2 = m[0][0].1,
                    u01_1 = m[0][1].0,
                    u01_2 = m[0][1].1,
                    u10_1 = m[1][0].0,
                    u10_2 = m[1][0].1,
                    u11_1 = m[1][1].0,
                    u11_2 = m[1][1].1,
                    summary = sol.summary_text.replace('\n', " | ")
                )
            }
        }
    }

    /// Generates a live preview of `model_analysis.R` for the R Script Preview tab.
    fn generate_r_script_preview(&self) -> String {
        let (x_star, y_star, lambda_star, indirect_utility) = self.solve_constrained_equilibrium();
        let tfp_a = self.get_param("A", 2.0);
        let alpha = self.get_param("alpha", 0.45);
        let beta = self.get_param("beta", 0.55);
        let price_x = self.get_param("price_x", 4.0);
        let price_y = self.get_param("price_y", 5.0);
        let income = self.get_param("I", 100.0);
        let target_output = self.get_param("Y_bar", 20.0);

        format!(
            r##"# ==============================================================================
# econ-calc Reproducible Analysis Script for Posit Positron IDE
# Active Model: {model_title}
# Preset: {preset}
# Target Architecture: Windows 11 ARM64 (aarch64-pc-windows-msvc)
# ==============================================================================

suppressPackageStartupMessages({{
  library(tidyverse)
}})

# Load exported numerical simulation grid
sim_data <- read_csv("simulation_data.csv", show_col_types = FALSE)

# Active structural parameters exported from econ-calc Environment
params <- tibble(
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
  indirect_utility = {indirect_utility:.6}
)

print(params)
glimpse(sim_data)

# Generate primary publication-quality ggplot2 figure
p_main <- ggplot(sim_data, aes(x = x)) +
  geom_line(aes(y = primary_y, color = "Primary Curve"), linewidth = 1.15) +
  geom_line(aes(y = secondary_y, color = "Secondary / Constraint"), linewidth = 0.95, linetype = "dashed") +
  geom_point(
    data = params,
    aes(x = x_star, y = y_star, color = "Solved Equilibrium (x*, y*)"),
    size = 3.6
  ) +
  scale_color_manual(
    name = "Series",
    values = c(
      "Primary Curve" = "#2563eb",
      "Secondary / Constraint" = "#dc2626",
      "Solved Equilibrium (x*, y*)" = "#16a34a"
    )
  ) +
  labs(
    title = "{model_title}",
    subtitle = sprintf(
      "Equilibrium: x* = %.4f, y* = %.4f | alpha = %.3f, beta = %.3f",
      params$x_star, params$y_star, params$alpha, params$beta
    ),
    x = "x (Input L / Good X)",
    y = "y (Input K / Good Y / f(x))",
    caption = "Generated natively by econ-calc (Windows 11 ARM64)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

print(p_main)
ggsave("equilibrium_plot.png", plot = p_main, width = 8.5, height = 5.5, dpi = 300)
"##,
            model_title = self.active_model.to_exporter_kind().title(),
            preset = self.active_preset_name,
            tfp_a = tfp_a,
            alpha = alpha,
            beta = beta,
            price_x = price_x,
            price_y = price_y,
            income = income,
            target_output = target_output,
            x_star = x_star,
            y_star = y_star,
            lambda_star = lambda_star,
            indirect_utility = indirect_utility,
        )
    }

    /// Evaluates the multiline CAS / Math Input block line by line.
    fn evaluate_cas_buffer(&mut self) {
        let lines: Vec<String> = self
            .cas_editor_buffer
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();

        if lines.is_empty() {
            return;
        }

        for line in lines {
            if line == ":clear" {
                self.console_history.clear();
                continue;
            }

            if let Some(rhs) = line.strip_prefix("f(x) =") {
                self.custom_expr = rhs.trim().to_string();
                self.console_history.push(ConsoleEntry {
                    command: line,
                    output: format!("Updated primary curve f(x) = {}", self.custom_expr),
                    is_error: false,
                });
                continue;
            }

            if let Some(rhs) = line.strip_prefix("g(x) =") {
                self.secondary_expr = rhs.trim().to_string();
                self.console_history.push(ConsoleEntry {
                    command: line,
                    output: format!("Updated secondary curve g(x) = {}", self.secondary_expr),
                    is_error: false,
                });
                continue;
            }

            // Support direct parameter assignment `param = value` in CAS editor
            if let Some((lhs, rhs)) = line.split_once('=') {
                let sym = lhs.trim();
                let is_known_param = self.parameters.iter().any(|p| p.symbol == sym);
                if is_known_param {
                    let probe_x = self.get_param("x0", 6.0);
                    match self.eval_expr_at(rhs.trim(), probe_x) {
                        Ok(val) => {
                            self.set_param(sym, val);
                            self.console_history.push(ConsoleEntry {
                                command: line,
                                output: format!("Updated parameter {} = {:.6}", sym, val),
                                is_error: false,
                            });
                        }
                        Err(err) => {
                            self.console_history.push(ConsoleEntry {
                                command: line,
                                output: err,
                                is_error: true,
                            });
                        }
                    }
                    continue;
                }
            }

            let res = self.parse_and_run_repl(&line);
            match res {
                Ok(out) => self.console_history.push(ConsoleEntry {
                    command: line,
                    output: out,
                    is_error: false,
                }),
                Err(err) => self.console_history.push(ConsoleEntry {
                    command: line,
                    output: err,
                    is_error: true,
                }),
            }
        }

        self.status_banner =
            "Evaluated CAS / Math buffer against active Environment parameters.".to_string();
    }

    fn parse_and_run_repl(&self, cmd: &str) -> Result<String, String> {
        if cmd == ":help" {
            return Ok(
                "Commands: `f(x) = <expr>`, `g(x) = <expr>`, `eval <expr> at <x>`, `diff <expr> at <x>`, `diff2 <expr> at <x>`, `int <expr> from <a> to <b>`, `root <expr> near <x0>`, `hessian`, or any math expression."
                    .to_string(),
            );
        }

        if cmd == "hessian" {
            let (x_star, y_star, _, _) = self.solve_constrained_equilibrium();
            let (h, det, trace) = self.compute_cobb_douglas_hessian(x_star, y_star);
            return Ok(format!(
                "Hessian H(x*={:.3}, y*={:.3}) = [[{:.5}, {:.5}], [{:.5}, {:.5}]] | det(H) = {:.6}, trace(H) = {:.6}",
                x_star,
                y_star,
                h[(0, 0)],
                h[(0, 1)],
                h[(1, 0)],
                h[(1, 1)],
                det,
                trace
            ));
        }

        if let Some(rest) = cmd.strip_prefix("eval ") {
            if let Some((expr, x_str)) = rest.rsplit_once(" at ") {
                let x: f64 = x_str
                    .trim()
                    .parse()
                    .map_err(|_| format!("Invalid number `{}`", x_str.trim()))?;
                let val = self.eval_expr_at(expr.trim(), x)?;
                return Ok(format!("f({:.6}) = {:.8}", x, val));
            }
        }

        if let Some(rest) = cmd.strip_prefix("diff2 ") {
            if let Some((expr, x_str)) = rest.rsplit_once(" at ") {
                let x: f64 = x_str
                    .trim()
                    .parse()
                    .map_err(|_| format!("Invalid number `{}`", x_str.trim()))?;
                let val = self.numerical_second_derivative(expr.trim(), x)?;
                return Ok(format!("f''({:.6}) = {:.8} (central difference)", x, val));
            }
        }

        if let Some(rest) = cmd.strip_prefix("diff ") {
            if let Some((expr, x_str)) = rest.rsplit_once(" at ") {
                let x: f64 = x_str
                    .trim()
                    .parse()
                    .map_err(|_| format!("Invalid number `{}`", x_str.trim()))?;
                let val = self.numerical_derivative(expr.trim(), x)?;
                return Ok(format!("f'({:.6}) = {:.8} (central difference)", x, val));
            }
        }

        if let Some(rest) = cmd.strip_prefix("int ") {
            if let Some((expr, bounds)) = rest.rsplit_once(" from ") {
                if let Some((a_str, b_str)) = bounds.split_once(" to ") {
                    let a: f64 = a_str
                        .trim()
                        .parse()
                        .map_err(|_| format!("Invalid lower bound `{}`", a_str.trim()))?;
                    let b: f64 = b_str
                        .trim()
                        .parse()
                        .map_err(|_| format!("Invalid upper bound `{}`", b_str.trim()))?;
                    let area =
                        self.composite_simpson(expr.trim(), a, b, self.simpson_intervals)?;
                    return Ok(format!(
                        "Integral of `{}` on [{:.4}, {:.4}] = {:.8} (Simpson N={})",
                        expr.trim(),
                        a,
                        b,
                        area,
                        self.simpson_intervals
                    ));
                }
            }
        }

        if let Some(rest) = cmd.strip_prefix("root ") {
            if let Some((expr, x0_str)) = rest.rsplit_once(" near ") {
                let x0: f64 = x0_str
                    .trim()
                    .parse()
                    .map_err(|_| format!("Invalid initial guess `{}`", x0_str.trim()))?;
                let root = self.solve_root(expr.trim(), x0)?;
                let residual = self.eval_expr_at(expr.trim(), root)?;
                return Ok(format!(
                    "Root x* = {:.8} (residual f(x*) = {:.3e})",
                    root, residual
                ));
            }
        }

        let probe_x = self.get_param("x0", 6.0);
        let val = self.eval_expr_at(cmd, probe_x)?;
        Ok(format!("= {:.8} (evaluated at x = {:.4})", val, probe_x))
    }

    fn build_export_payload(&self) -> ExportPayload {
        let (x_star, y_star, lambda_star, indirect_utility) = self.solve_constrained_equilibrium();
        let game_sol = self.solve_game_2x2();

        ExportPayload {
            model_kind: self.active_model.to_exporter_kind(),
            report_format: self.report_format,
            custom_expr: self.custom_expr.clone(),
            secondary_expr: self.secondary_expr.clone(),
            tfp_a: self.get_param("A", 2.0),
            alpha: self.get_param("alpha", 0.45),
            beta: self.get_param("beta", 0.55),
            price_x: self.get_param("price_x", 4.0),
            price_y: self.get_param("price_y", 5.0),
            income: self.get_param("I", 100.0),
            target_output: self.get_param("Y_bar", 20.0),
            x_star,
            y_star,
            lambda_star,
            indirect_utility,
            game_matrix: self.payoff_matrix,
            game_summary: game_sol.summary_text,
            latex_derivation: self.generate_latex_and_tikz(),
            grid_rows: self.build_simulation_grid(100),
        }
    }

    fn trigger_positron_export(&mut self) {
        let payload = self.build_export_payload();
        match export_and_launch_positron(&payload) {
            Ok(outcome) => {
                self.status_banner = format!(
                    "Exported to {:?} via {}",
                    outcome.export_dir, outcome.launched_program
                );
                self.console_history.push(ConsoleEntry {
                    command: ":export positron".to_string(),
                    output: outcome.status_message,
                    is_error: false,
                });
            }
            Err(err) => {
                self.status_banner = format!("Export error: {}", err);
                self.console_history.push(ConsoleEntry {
                    command: ":export positron".to_string(),
                    output: err,
                    is_error: true,
                });
            }
        }
    }

    // =========================================================================
    // UI RENDERING: TOP HEADER BAR
    // =========================================================================
    fn render_top_menu_bar(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new("econ-calc IDE")
                    .font(FontId::proportional(16.5_f32))
                    .strong()
                    .color(Color32::from_rgb(56, 189, 248)),
            );
            ui.separator();

            ui.label(
                RichText::new(format!(
                    "[{} — {}]",
                    self.active_model.label(),
                    self.active_preset_name
                ))
                .strong()
                .color(Color32::from_rgb(129, 140, 248)),
            );

            ui.separator();

            if ui
                .button(RichText::new("Copy LaTeX / TikZ").strong())
                .on_hover_text("Copy analytical LaTeX & TikZ derivation to clipboard")
                .clicked()
            {
                let latex = self.generate_latex_and_tikz();
                ui.ctx().copy_text(latex);
                self.status_banner =
                    "Copied publication-ready LaTeX & TikZ derivation to clipboard.".to_string();
            }

            ui.separator();

            ui.label(RichText::new("Report:").strong());
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

            if ui
                .button(
                    RichText::new("Export & Open in Positron")
                        .strong()
                        .color(Color32::from_rgb(34, 197, 94)),
                )
                .on_hover_text(
                    "Export project.Rproj, simulation_data.csv, model_analysis.R, and report.qmd/.Rmd",
                )
                .clicked()
            {
                self.trigger_positron_export();
            }

            ui.separator();
            ui.label(
                RichText::new(&self.status_banner)
                    .italics()
                    .color(Color32::from_rgb(148, 163, 184)),
            );
        });
    }

    // =========================================================================
    // 1. LEFT ACTIVITY BAR & SIDEBAR (VS CODE HYBRID)
    // =========================================================================
    fn render_left_activity_and_sidebar(&mut self, ui: &mut Ui) {
        // Activity Bar strip at the top of the Left Sidebar
        ui.horizontal(|ui| {
            let explorer_selected =
                self.sidebar_open && self.sidebar_tab == SidebarTab::ModelExplorer;
            if ui
                .selectable_label(
                    explorer_selected,
                    RichText::new("📚 Explorer").font(FontId::proportional(13.5_f32)).strong(),
                )
                .on_hover_text("Model Explorer & Preset Library")
                .clicked()
            {
                if explorer_selected {
                    self.sidebar_open = false;
                } else {
                    self.sidebar_open = true;
                    self.sidebar_tab = SidebarTab::ModelExplorer;
                }
            }

            let settings_selected = self.sidebar_open && self.sidebar_tab == SidebarTab::Settings;
            if ui
                .selectable_label(
                    settings_selected,
                    RichText::new("⚙ Settings").font(FontId::proportional(13.5_f32)).strong(),
                )
                .on_hover_text("Global IDE Settings (Theme, 3D Resolution, UI Scale)")
                .clicked()
            {
                if settings_selected {
                    self.sidebar_open = false;
                } else {
                    self.sidebar_open = true;
                    self.sidebar_tab = SidebarTab::Settings;
                }
            }
        });

        ui.separator();

        if !self.sidebar_open {
            ui.vertical_centered(|ui| {
                ui.add_space(12.0_f32);
                ui.label(
                    RichText::new("Sidebar Collapsed\nClick Activity Tab to Expand")
                        .italics()
                        .color(Color32::from_rgb(148, 163, 184)),
                );
            });
            return;
        }

        egui::ScrollArea::vertical()
            .id_source("left_sidebar_content_scroll")
            .show(ui, |ui| match self.sidebar_tab {
                SidebarTab::ModelExplorer => self.render_model_explorer_tab(ui),
                SidebarTab::Settings => self.render_settings_tab(ui),
            });
    }

    fn render_model_explorer_tab(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("MODEL EXPLORER")
                .font(FontId::proportional(12.0_f32))
                .strong()
                .color(Color32::from_rgb(148, 163, 184)),
        );
        ui.add_space(4.0_f32);

        if ui
            .button(
                RichText::new("➕ Blank Custom Workspace")
                    .strong()
                    .color(Color32::from_rgb(56, 189, 248)),
            )
            .on_hover_text("Initialize a clean Custom CAS workspace with generic parameters")
            .clicked()
        {
            self.load_model_workspace(ActiveModel::CustomCAS, "Blank Custom Workspace");
            self.custom_expr = "A * x^alpha".to_string();
            self.secondary_expr = "price_x * x".to_string();
            self.cas_editor_buffer =
                "# Blank Custom Workspace\nf(x) = A * x^alpha\ng(x) = price_x * x\neval A * x^alpha at 5.0"
                    .to_string();
            self.status_banner = "Initialized Blank Custom CAS Workspace.".to_string();
        }

        ui.add_space(6.0_f32);

        egui::CollapsingHeader::new(RichText::new("Consumer & Demand Theory").strong())
            .default_open(true)
            .show(ui, |ui| {
                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::ConstrainedUtility
                            && self.active_preset_name == "Standard Marshallian Consumer",
                        "• Constrained Utility (Marshallian)",
                    )
                    .clicked()
                {
                    self.load_model_workspace(
                        ActiveModel::ConstrainedUtility,
                        "Standard Marshallian Consumer",
                    );
                }

                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::ConstrainedUtility
                            && self.active_preset_name == "Good-X Intensive Preferences",
                        "• Good-X Intensive Preferences (0.75 / 0.25)",
                    )
                    .clicked()
                {
                    self.load_model_workspace(
                        ActiveModel::ConstrainedUtility,
                        "Good-X Intensive Preferences",
                    );
                    self.set_param("alpha", 0.75);
                    self.set_param("beta", 0.25);
                    self.set_param("price_x", 3.0);
                    self.set_param("price_y", 6.0);
                    self.set_param("I", 120.0);
                }

                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::ConstrainedUtility
                            && self.active_preset_name == "High-Income Luxury Allocation",
                        "• High-Income Luxury Allocation",
                    )
                    .clicked()
                {
                    self.load_model_workspace(
                        ActiveModel::ConstrainedUtility,
                        "High-Income Luxury Allocation",
                    );
                    self.set_param("A", 3.5);
                    self.set_param("alpha", 0.50);
                    self.set_param("beta", 0.50);
                    self.set_param("price_x", 8.0);
                    self.set_param("price_y", 4.0);
                    self.set_param("I", 320.0);
                }
            });

        egui::CollapsingHeader::new(RichText::new("Production & Firm Theory").strong())
            .default_open(true)
            .show(ui, |ui| {
                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::CobbDouglas
                            && self.active_preset_name == "CRS Cobb-Douglas Isoquant",
                        "• Cobb-Douglas CRS (alpha+beta = 1.0)",
                    )
                    .clicked()
                {
                    self.load_model_workspace(
                        ActiveModel::CobbDouglas,
                        "CRS Cobb-Douglas Isoquant",
                    );
                }

                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::CobbDouglas
                            && self.active_preset_name == "Increasing Returns (IRS) Tech",
                        "• Increasing Returns (IRS, alpha+beta = 1.3)",
                    )
                    .clicked()
                {
                    self.load_model_workspace(
                        ActiveModel::CobbDouglas,
                        "Increasing Returns (IRS) Tech",
                    );
                    self.set_param("alpha", 0.65);
                    self.set_param("beta", 0.65);
                    self.set_param("Y_bar", 40.0);
                }

                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::CobbDouglas
                            && self.active_preset_name == "Capital-Intensive Automation",
                        "• Capital-Intensive Automation",
                    )
                    .clicked()
                {
                    self.load_model_workspace(
                        ActiveModel::CobbDouglas,
                        "Capital-Intensive Automation",
                    );
                    self.set_param("alpha", 0.25);
                    self.set_param("beta", 0.75);
                    self.set_param("price_x", 10.0);
                    self.set_param("price_y", 3.0);
                }
            });

        egui::CollapsingHeader::new(RichText::new("Strategic Game Theory (2x2)").strong())
            .default_open(true)
            .show(ui, |ui| {
                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::GameTheory
                            && self.active_preset_name == "Prisoner's Dilemma",
                        "• Prisoner's Dilemma",
                    )
                    .clicked()
                {
                    self.row_labels = ["Cooperate".to_string(), "Defect".to_string()];
                    self.col_labels = ["Cooperate".to_string(), "Defect".to_string()];
                    self.payoff_matrix = [[(3.0, 3.0), (0.0, 5.0)], [(5.0, 0.0), (1.0, 1.0)]];
                    self.load_model_workspace(ActiveModel::GameTheory, "Prisoner's Dilemma");
                }

                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::GameTheory
                            && self.active_preset_name == "Matching Pennies (MSNE)",
                        "• Matching Pennies (Zero-Sum MSNE)",
                    )
                    .clicked()
                {
                    self.row_labels = ["Heads".to_string(), "Tails".to_string()];
                    self.col_labels = ["Heads".to_string(), "Tails".to_string()];
                    self.payoff_matrix = [[(1.0, -1.0), (-1.0, 1.0)], [(-1.0, 1.0), (1.0, -1.0)]];
                    self.load_model_workspace(ActiveModel::GameTheory, "Matching Pennies (MSNE)");
                }

                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::GameTheory
                            && self.active_preset_name == "Battle of the Sexes",
                        "• Battle of the Sexes (Coordination)",
                    )
                    .clicked()
                {
                    self.row_labels = ["Opera".to_string(), "Football".to_string()];
                    self.col_labels = ["Opera".to_string(), "Football".to_string()];
                    self.payoff_matrix = [[(3.0, 2.0), (0.0, 0.0)], [(0.0, 0.0), (2.0, 3.0)]];
                    self.load_model_workspace(ActiveModel::GameTheory, "Battle of the Sexes");
                }

                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::GameTheory
                            && self.active_preset_name == "Stag Hunt",
                        "• Stag Hunt (Payoff vs Risk Dominance)",
                    )
                    .clicked()
                {
                    self.row_labels = ["Stag".to_string(), "Hare".to_string()];
                    self.col_labels = ["Stag".to_string(), "Hare".to_string()];
                    self.payoff_matrix = [[(4.0, 4.0), (1.0, 3.0)], [(3.0, 1.0), (2.0, 2.0)]];
                    self.load_model_workspace(ActiveModel::GameTheory, "Stag Hunt");
                }
            });

        egui::CollapsingHeader::new(RichText::new("Quantitative CAS & Macroeconomics").strong())
            .default_open(true)
            .show(ui, |ui| {
                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::CustomCAS
                            && self.active_preset_name == "Firm Profit & Marginal Cost",
                        "• Firm Profit & Marginal Cost Curve",
                    )
                    .clicked()
                {
                    self.load_model_workspace(
                        ActiveModel::CustomCAS,
                        "Firm Profit & Marginal Cost",
                    );
                    self.custom_expr = "A * x^alpha - price_x * x".to_string();
                    self.secondary_expr = "price_x * x".to_string();
                }

                if ui
                    .selectable_label(
                        self.active_model == ActiveModel::CustomCAS
                            && self.active_preset_name == "Solow-Swan Steady State",
                        "• Solow-Swan Steady State Growth",
                    )
                    .clicked()
                {
                    self.load_model_workspace(ActiveModel::CustomCAS, "Solow-Swan Steady State");
                    self.set_param("A", 1.8);
                    self.set_param("alpha", 0.35);
                    self.set_param("price_x", 0.12);
                    self.custom_expr = "0.30 * A * x^alpha".to_string();
                    self.secondary_expr = "price_x * x".to_string();
                    self.cas_editor_buffer = "# Solow-Swan Capital Accumulation\n# f(x) = s * A * k^alpha, g(x) = (delta + n) * k\nf(x) = 0.30 * A * x^alpha\ng(x) = price_x * x\nroot (0.30 * A * x^alpha) - (price_x * x) near 8.0".to_string();
                }
            });
    }

    fn render_settings_tab(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("GLOBAL IDE SETTINGS")
                .font(FontId::proportional(12.0_f32))
                .strong()
                .color(Color32::from_rgb(148, 163, 184)),
        );
        ui.add_space(6.0_f32);

        ui.label(RichText::new("Appearance & Scaling").strong());
        if ui
            .checkbox(&mut self.dark_mode, "Positron Dark Theme")
            .changed()
        {
            ui.ctx().set_visuals(if self.dark_mode {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            });
        }

        if ui
            .add(
                egui::Slider::new(&mut self.ui_scale, 0.85_f32..=1.35_f32)
                    .text("UI Scale (DPI)"),
            )
            .changed()
        {
            ui.ctx().set_pixels_per_point(self.ui_scale);
        }

        ui.separator();
        ui.label(RichText::new("3D Surface Renderer").strong());
        ui.add(
            egui::Slider::new(&mut self.surface_resolution, 12..=48).text("Mesh Resolution (N)"),
        );
        ui.add(egui::Slider::new(&mut self.yaw_deg, -180.0_f32..=180.0_f32).text("Camera Yaw (°)"));
        ui.add(egui::Slider::new(&mut self.pitch_deg, 5.0_f32..=85.0_f32).text("Camera Pitch (°)"));
        ui.add(egui::Slider::new(&mut self.surface_zoom, 0.4_f32..=2.5_f32).text("Surface Zoom"));
        ui.checkbox(&mut self.surface_wireframe_only, "Wireframe Mesh Only");

        ui.separator();
        ui.label(RichText::new("Numerical Calculus Engine").strong());
        ui.checkbox(
            &mut self.show_derivative_curve,
            "Overlay Derivative f'(x) on 2D Plot",
        );
        ui.add(
            egui::Slider::new(&mut self.simpson_intervals, 20..=600)
                .text("Simpson Subintervals (N)"),
        );
    }

    // =========================================================================
    // 2. RIGHT ENVIRONMENT & DATA GRID PANE (RSTUDIO HYBRID)
    // =========================================================================
    fn render_right_environment_pane(&mut self, ui: &mut Ui) {
        let total_height = ui.available_height();
        let top_env_height = (total_height * 0.48_f32).max(200.0_f32);

        // TOP HALF: Dynamic ModelParameter Sliders & Computed Values
        ui.vertical(|ui| {
            ui.set_min_height(top_env_height);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("ENVIRONMENT PARAMETERS")
                        .font(FontId::proportional(12.5_f32))
                        .strong()
                        .color(Color32::from_rgb(56, 189, 248)),
                );
                ui.label(
                    RichText::new(format!("({} active)", self.parameters.len()))
                        .color(Color32::from_rgb(148, 163, 184)),
                );
            });
            ui.separator();

            egui::ScrollArea::vertical()
                .id_source("right_env_parameters_scroll")
                .max_height(top_env_height - 36.0_f32)
                .show(ui, |ui| {
                    for param in &mut self.parameters {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("{}", param.symbol))
                                    .monospace()
                                    .strong()
                                    .color(Color32::from_rgb(125, 211, 252)),
                            );
                            ui.label(
                                RichText::new(format!("— {}", param.name))
                                    .font(FontId::proportional(12.0_f32)),
                            );
                        });
                        ui.add(
                            egui::Slider::new(&mut param.value, param.min..=param.max)
                                .clamping(egui::SliderClamping::Always),
                        );
                        ui.add_space(2.0_f32);
                    }

                    ui.separator();
                    egui::CollapsingHeader::new(
                        RichText::new("➕ Add Custom Environment Variable").font(FontId::proportional(12.0_f32)),
                    )
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Symbol:");
                            ui.text_edit_singleline(&mut self.new_param_symbol);
                        });
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            ui.text_edit_singleline(&mut self.new_param_name);
                        });
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.new_param_val)
                                    .speed(0.1)
                                    .prefix("val: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.new_param_min)
                                    .speed(0.5)
                                    .prefix("min: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.new_param_max)
                                    .speed(0.5)
                                    .prefix("max: "),
                            );
                        });
                        if ui.button("Bind Variable to Environment").clicked() {
                            let sym = self.new_param_symbol.trim().to_string();
                            if !sym.is_empty() {
                                let max_v = self.new_param_max.max(self.new_param_min + 0.01);
                                let clamped = self.new_param_val.clamp(self.new_param_min, max_v);
                                self.parameters.push(ModelParameter::new(
                                    self.new_param_name.trim(),
                                    &sym,
                                    clamped,
                                    self.new_param_min,
                                    max_v,
                                ));
                            }
                        }
                    });

                    ui.separator();
                    let (x_star, y_star, lambda_star, obj_star) =
                        self.solve_constrained_equilibrium();
                    ui.label(
                        RichText::new("Computed Equilibrium Bindings")
                            .font(FontId::proportional(12.0_f32))
                            .strong()
                            .color(Color32::from_rgb(34, 197, 94)),
                    );
                    egui::Grid::new("env_computed_summary_grid")
                        .striped(true)
                        .spacing([10.0_f32, 4.0_f32])
                        .show(ui, |ui| {
                            ui.monospace("x* / L*");
                            ui.monospace(format!("{:.5}", x_star));
                            ui.monospace("y* / K*");
                            ui.monospace(format!("{:.5}", y_star));
                            ui.end_row();

                            ui.monospace("lambda*");
                            ui.monospace(format!("{:.5}", lambda_star));
                            ui.monospace("V* / C*");
                            ui.monospace(format!("{:.5}", obj_star));
                            ui.end_row();
                        });
                });
        });

        ui.separator();

        // BOTTOM HALF: RStudio-style Numerical Simulation Data Grid
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("DATA VIEWER: simulation_data (50 rows)")
                    .font(FontId::proportional(12.5_f32))
                    .strong()
                    .color(Color32::from_rgb(56, 189, 248)),
            );
        });
        ui.add_space(2.0_f32);

        let grid_rows = self.build_simulation_grid(50);
        let (col1, col2, col3, col4) = match self.active_model {
            ActiveModel::CustomCAS => ("x", "f(x)", "g(x)", "f'(x)"),
            ActiveModel::CobbDouglas => ("L", "K_isoquant", "K_isocost", "-MRTS"),
            ActiveModel::ConstrainedUtility => ("x", "y_indiff", "y_budget", "-MRS"),
            ActiveModel::GameTheory => ("q (Col L)", "E[U1|Up]", "E[U1|Down]", "Diff"),
        };

        egui::ScrollArea::vertical()
            .id_source("right_data_viewer_grid_scroll")
            .show(ui, |ui| {
                egui::Grid::new("rstudio_dataframe_grid")
                    .striped(true)
                    .spacing([12.0_f32, 4.0_f32])
                    .show(ui, |ui| {
                        ui.label(RichText::new("#").strong());
                        ui.label(RichText::new(col1).strong());
                        ui.label(RichText::new(col2).strong());
                        ui.label(RichText::new(col3).strong());
                        ui.label(RichText::new(col4).strong());
                        ui.end_row();

                        for row in &grid_rows {
                            ui.monospace(format!("{:02}", row.index));
                            ui.monospace(format!("{:.3}", row.x));
                            ui.monospace(format!("{:.4}", row.primary_y));
                            ui.monospace(format!("{:.4}", row.secondary_y));
                            ui.monospace(format!("{:.4}", row.marginal_or_slope));
                            ui.end_row();
                        }
                    });
            });
    }

    // =========================================================================
    // 3. CENTER WORKSPACE: TOP VISUAL VIEWPORT & BOTTOM CODE/CONSOLE BLOCK
    // =========================================================================
    fn render_center_workspace(&mut self, ui: &mut Ui) {
        // Split the CentralPanel vertically by placing a resizable bottom panel inside it
        egui::TopBottomPanel::bottom("center_bottom_code_console_panel")
            .resizable(true)
            .default_height(275.0_f32)
            .min_height(165.0_f32)
            .show_inside(ui, |ui| {
                self.render_center_bottom_code_block(ui);
            });

        // Remaining Top Area of CentralPanel is the Visual Viewport (2D Plot / 3D Surface)
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_center_top_visual_viewport(ui);
        });
    }

    fn render_center_top_visual_viewport(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("VISUAL VIEWPORT")
                    .font(FontId::proportional(13.0_f32))
                    .strong()
                    .color(Color32::from_rgb(56, 189, 248)),
            );
            ui.separator();
            ui.selectable_value(
                &mut self.viewport_tab,
                ViewportTab::Plot2D,
                "📈 2D Economic Plot (egui_plot)",
            );
            ui.selectable_value(
                &mut self.viewport_tab,
                ViewportTab::Surface3D,
                "🧊 3D Isometric Surface",
            );

            ui.separator();
            let (x_star, y_star, lambda_star, obj_star) = self.solve_constrained_equilibrium();
            ui.label(
                RichText::new(format!(
                    "Equilibrium: (x*={:.3}, y*={:.3}) | obj*={:.3} | λ*={:.3}",
                    x_star, y_star, obj_star, lambda_star
                ))
                .monospace()
                .color(Color32::from_rgb(34, 197, 94)),
            );
        });
        ui.separator();

        match self.viewport_tab {
            ViewportTab::Plot2D => self.render_2d_plot_tab(ui),
            ViewportTab::Surface3D => self.render_3d_surface_tab(ui),
        }
    }

    fn render_center_bottom_code_block(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.code_console_tab,
                CodeConsoleTab::CasMathInput,
                RichText::new("⌨ CAS / Math Input").strong(),
            );
            ui.selectable_value(
                &mut self.code_console_tab,
                CodeConsoleTab::LatexOutput,
                RichText::new("📄 LaTeX Output").strong(),
            );
            ui.selectable_value(
                &mut self.code_console_tab,
                CodeConsoleTab::RScriptPreview,
                RichText::new("📊 R Script Preview (model_analysis.R)").strong(),
            );

            ui.separator();

            if ui
                .button(
                    RichText::new("▶ Run / Evaluate")
                        .strong()
                        .color(Color32::from_rgb(34, 197, 94)),
                )
                .on_hover_text(
                    "Evaluate CAS / Math Input using current Environment variables via meval",
                )
                .clicked()
            {
                self.evaluate_cas_buffer();
            }

            if ui
                .button("Solve f(x) = g(x)")
                .on_hover_text("Find numerical intersection of f(x) and g(x)")
                .clicked()
            {
                let diff_expr = format!("({}) - ({})", self.custom_expr, self.secondary_expr);
                let guess = self.get_param("x0", 5.0);
                match self.solve_root(&diff_expr, guess) {
                    Ok(r) => {
                        self.set_param("x0", r);
                        let y_val = self.eval_expr_at(&self.custom_expr, r).unwrap_or(0.0);
                        self.console_history.push(ConsoleEntry {
                            command: format!("intersect f(x) == g(x) near {:.2}", guess),
                            output: format!("Intersection converged at (x* = {:.6}, y* = {:.6})", r, y_val),
                            is_error: false,
                        });
                    }
                    Err(e) => {
                        self.console_history.push(ConsoleEntry {
                            command: "intersect f(x) == g(x)".to_string(),
                            output: e,
                            is_error: true,
                        });
                    }
                }
            }

            if ui.button("Inspect Hessian").clicked() {
                if let Ok(out) = self.parse_and_run_repl("hessian") {
                    self.console_history.push(ConsoleEntry {
                        command: "hessian".to_string(),
                        output: out,
                        is_error: false,
                    });
                }
            }

            if ui.button("Clear Log").clicked() {
                self.console_history.clear();
            }
        });

        ui.separator();

        match self.code_console_tab {
            CodeConsoleTab::CasMathInput => self.render_cas_math_input_tab(ui),
            CodeConsoleTab::LatexOutput => self.render_latex_output_tab(ui),
            CodeConsoleTab::RScriptPreview => self.render_r_script_preview_tab(ui),
        }
    }

    fn render_cas_math_input_tab(&mut self, ui: &mut Ui) {
        ui.columns(2, |cols| {
            // Left Column: Formula Bindings, Multiline CAS Editor, and optional 2x2 Game Matrix
            cols[0].vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("f(x) =").monospace().strong());
                    ui.text_edit_singleline(&mut self.custom_expr);
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("g(x) =").monospace().strong());
                    ui.text_edit_singleline(&mut self.secondary_expr);
                });

                if self.active_model == ActiveModel::GameTheory {
                    ui.add_space(2.0_f32);
                    ui.label(
                        RichText::new("2x2 Normal-Form Payoff Matrix (u1, u2):")
                            .font(FontId::proportional(12.0_f32))
                            .strong(),
                    );
                    egui::Grid::new("center_game_matrix_grid")
                        .striped(true)
                        .spacing([8.0_f32, 4.0_f32])
                        .show(ui, |ui| {
                            ui.label(RichText::new("P1 \\ P2").strong());
                            ui.text_edit_singleline(&mut self.col_labels[0]);
                            ui.text_edit_singleline(&mut self.col_labels[1]);
                            ui.end_row();

                            for r in 0..2 {
                                ui.text_edit_singleline(&mut self.row_labels[r]);
                                for c in 0..2 {
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::DragValue::new(&mut self.payoff_matrix[r][c].0)
                                                .speed(0.25)
                                                .prefix("u1:"),
                                        );
                                        ui.add(
                                            egui::DragValue::new(&mut self.payoff_matrix[r][c].1)
                                                .speed(0.25)
                                                .prefix("u2:"),
                                        );
                                    });
                                }
                                ui.end_row();
                            }
                        });
                } else {
                    ui.label(
                        RichText::new("CAS Script Buffer (supports meval expressions & commands):")
                            .font(FontId::proportional(11.5_f32))
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    egui::ScrollArea::vertical()
                        .id_source("cas_multiline_editor_scroll")
                        .show(ui, |ui| {
                            ui.add_sized(
                                ui.available_size(),
                                TextEdit::multiline(&mut self.cas_editor_buffer)
                                    .code_editor()
                                    .desired_rows(6),
                            );
                        });
                }
            });

            // Right Column: Live CAS Evaluation & Equilibrium Output Console
            cols[1].vertical(|ui| {
                ui.label(
                    RichText::new("CONSOLE / EVALUATION OUTPUT")
                        .font(FontId::proportional(11.5_f32))
                        .strong()
                        .color(Color32::from_rgb(56, 189, 248)),
                );

                if self.active_model == ActiveModel::GameTheory {
                    let sol = self.solve_game_2x2();
                    ui.monospace(
                        RichText::new(&sol.summary_text).color(Color32::from_rgb(34, 197, 94)),
                    );
                    ui.separator();
                }

                egui::ScrollArea::vertical()
                    .id_source("cas_output_log_scroll")
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for entry in &self.console_history {
                            ui.monospace(
                                RichText::new(format!("> {}", entry.command))
                                    .color(Color32::from_rgb(56, 189, 248))
                                    .strong(),
                            );
                            let out_color = if entry.is_error {
                                Color32::from_rgb(248, 113, 113)
                            } else {
                                ui.visuals().text_color()
                            };
                            ui.monospace(
                                RichText::new(format!("  {}", entry.output)).color(out_color),
                            );
                        }
                    });
            });
        });
    }

    fn render_latex_output_tab(&mut self, ui: &mut Ui) {
        let mut latex_code = self.generate_latex_and_tikz();
        egui::ScrollArea::vertical()
            .id_source("latex_readonly_scroll")
            .show(ui, |ui| {
                ui.add_sized(
                    ui.available_size(),
                    TextEdit::multiline(&mut latex_code)
                        .code_editor()
                        .interactive(false),
                );
            });
    }

    fn render_r_script_preview_tab(&mut self, ui: &mut Ui) {
        let mut r_script = self.generate_r_script_preview();
        egui::ScrollArea::vertical()
            .id_source("r_script_readonly_scroll")
            .show(ui, |ui| {
                ui.add_sized(
                    ui.available_size(),
                    TextEdit::multiline(&mut r_script)
                        .code_editor()
                        .interactive(false),
                );
            });
    }

    // =========================================================================
    // 2D & 3D VIEWPORT RENDERERS
    // =========================================================================
    fn render_2d_plot_tab(&mut self, ui: &mut Ui) {
        let (x_star, y_star, _, indirect_u) = self.solve_constrained_equilibrium();
        let tfp_a = self.get_param("A", 2.0).max(1e-6);
        let alpha = self.get_param("alpha", 0.45).max(1e-6);
        let beta = self.get_param("beta", 0.55).max(1e-6);
        let px = self.get_param("price_x", 4.0).max(1e-6);
        let py = self.get_param("price_y", 5.0).max(1e-6);
        let income = self.get_param("I", 100.0).max(1e-6);
        let target_output = self.get_param("Y_bar", 20.0).max(1e-6);

        Plot::new("main_econ_plot")
            .legend(Legend::default())
            .show(ui, |plot_ui| match self.active_model {
                ActiveModel::CustomCAS => {
                    let dom_min = self.get_param("x_min", 0.1);
                    let dom_max = self.get_param("x_max", 25.0).max(dom_min + 0.5);
                    let probe_x = self.get_param("x0", 6.0);
                    let span = (dom_max - dom_min).max(1e-3);

                    let primary_pts: PlotPoints = (0..=240)
                        .filter_map(|i| {
                            let x = dom_min + span * (i as f64) / 240.0;
                            let y = self.eval_expr_at(&self.custom_expr, x).ok()?;
                            y.is_finite().then_some([x, y])
                        })
                        .collect();

                    let secondary_pts: PlotPoints = (0..=240)
                        .filter_map(|i| {
                            let x = dom_min + span * (i as f64) / 240.0;
                            let y = self.eval_expr_at(&self.secondary_expr, x).ok()?;
                            y.is_finite().then_some([x, y])
                        })
                        .collect();

                    plot_ui.line(
                        Line::new(primary_pts)
                            .name(format!("f(x) = {}", self.custom_expr))
                            .width(2.4_f32)
                            .color(Color32::from_rgb(59, 130, 246)),
                    );
                    plot_ui.line(
                        Line::new(secondary_pts)
                            .name(format!("g(x) = {}", self.secondary_expr))
                            .width(1.8_f32)
                            .color(Color32::from_rgb(249, 115, 22)),
                    );

                    if self.show_derivative_curve {
                        let deriv_pts: PlotPoints = (0..=240)
                            .filter_map(|i| {
                                let x = dom_min + span * (i as f64) / 240.0;
                                let dy = self.numerical_derivative(&self.custom_expr, x).ok()?;
                                dy.is_finite().then_some([x, dy])
                            })
                            .collect();
                        plot_ui.line(
                            Line::new(deriv_pts)
                                .name("Derivative f'(x)")
                                .width(1.5_f32)
                                .color(Color32::from_rgb(168, 85, 247)),
                        );
                    }

                    if let Ok(y0) = self.eval_expr_at(&self.custom_expr, probe_x) {
                        if y0.is_finite() {
                            plot_ui.points(
                                Points::new(vec![[probe_x, y0]])
                                    .name(format!("Probe x0 ({:.3}, {:.3})", probe_x, y0))
                                    .radius(5.5_f32)
                                    .color(Color32::from_rgb(34, 197, 94)),
                            );
                        }
                    }
                }
                ActiveModel::CobbDouglas => {
                    let max_l = (x_star * 2.5).max(15.0);
                    let max_k = (y_star * 2.5).max(15.0);
                    let min_cost = px * x_star + py * y_star;

                    for (mult, label_suffix, width) in [
                        (0.75, "0.75 * Y_bar", 1.3_f32),
                        (1.0, "Active Isoquant Y_bar", 2.6_f32),
                        (1.30, "1.30 * Y_bar", 1.3_f32),
                    ] {
                        let y_target = target_output * mult;
                        let iso_pts: PlotPoints = (1..=220)
                            .filter_map(|i| {
                                let l = 0.2 + (max_l - 0.2) * (i as f64) / 220.0;
                                let k = (y_target / (tfp_a * l.powf(alpha)))
                                    .max(0.0)
                                    .powf(1.0 / beta);
                                (k.is_finite() && k <= max_k * 1.4).then_some([l, k])
                            })
                            .collect();
                        plot_ui.line(
                            Line::new(iso_pts)
                                .name(format!("{} ({:.1})", label_suffix, y_target))
                                .width(width),
                        );
                    }

                    let isocost_pts: PlotPoints = (0..=100)
                        .filter_map(|i| {
                            let l = max_l * (i as f64) / 100.0;
                            let k = (min_cost - px * l) / py;
                            (k >= 0.0).then_some([l, k])
                        })
                        .collect();

                    plot_ui.line(
                        Line::new(isocost_pts)
                            .name(format!("Isocost Line (C* = {:.2})", min_cost))
                            .width(2.0_f32)
                            .color(Color32::from_rgb(239, 68, 68)),
                    );

                    plot_ui.points(
                        Points::new(vec![[x_star, y_star]])
                            .name(format!("Cost-Min Bundle (L*={:.2}, K*={:.2})", x_star, y_star))
                            .radius(6.0_f32)
                            .color(Color32::from_rgb(34, 197, 94)),
                    );
                }
                ActiveModel::ConstrainedUtility => {
                    let max_x = (income / px) * 1.25;
                    let max_y = (income / py) * 1.25;

                    for (mult, label_prefix, width) in [
                        (0.75, "Lower Indifference (0.75 U*)", 1.3_f32),
                        (1.0, "Optimal Indifference U*", 2.6_f32),
                        (1.25, "Higher Indifference (1.25 U*)", 1.3_f32),
                    ] {
                        let u_level = indirect_u.max(1e-6) * mult;
                        let indiff_pts: PlotPoints = (1..=220)
                            .filter_map(|i| {
                                let x = 0.15 + (max_x - 0.15) * (i as f64) / 220.0;
                                let y = (u_level / (tfp_a * x.powf(alpha)))
                                    .max(0.0)
                                    .powf(1.0 / beta);
                                (y.is_finite() && y <= max_y * 1.5).then_some([x, y])
                            })
                            .collect();
                        plot_ui.line(
                            Line::new(indiff_pts)
                                .name(format!("{} = {:.2}", label_prefix, u_level))
                                .width(width),
                        );
                    }

                    let budget_pts: PlotPoints = (0..=100)
                        .filter_map(|i| {
                            let x = max_x * (i as f64) / 100.0;
                            let y = (income - px * x) / py;
                            (y >= 0.0).then_some([x, y])
                        })
                        .collect();

                    plot_ui.line(
                        Line::new(budget_pts)
                            .name(format!(
                                "Budget Constraint ({:.1}x + {:.1}y = {:.1})",
                                px, py, income
                            ))
                            .width(2.2_f32)
                            .color(Color32::from_rgb(239, 68, 68)),
                    );

                    plot_ui.points(
                        Points::new(vec![[x_star, y_star]])
                            .name(format!("Marshallian Optimum (x*={:.2}, y*={:.2})", x_star, y_star))
                            .radius(6.0_f32)
                            .color(Color32::from_rgb(34, 197, 94)),
                    );
                }
                ActiveModel::GameTheory => {
                    let m = &self.payoff_matrix;
                    let sol = self.solve_game_2x2();

                    let eu_up_pts: PlotPoints = (0..=100)
                        .map(|i| {
                            let q = (i as f64) / 100.0;
                            let u = q * m[0][0].0 + (1.0 - q) * m[0][1].0;
                            [q, u]
                        })
                        .collect();

                    let eu_down_pts: PlotPoints = (0..=100)
                        .map(|i| {
                            let q = (i as f64) / 100.0;
                            let u = q * m[1][0].0 + (1.0 - q) * m[1][1].0;
                            [q, u]
                        })
                        .collect();

                    plot_ui.line(
                        Line::new(eu_up_pts)
                            .name(format!("E[U1 | {}] vs q", self.row_labels[0]))
                            .width(2.4_f32)
                            .color(Color32::from_rgb(59, 130, 246)),
                    );
                    plot_ui.line(
                        Line::new(eu_down_pts)
                            .name(format!("E[U1 | {}] vs q", self.row_labels[1]))
                            .width(2.4_f32)
                            .color(Color32::from_rgb(249, 115, 22)),
                    );

                    if let (Some(q_star), Some(eu1)) = (sol.mixed_q, sol.expected_u1) {
                        plot_ui.vline(
                            VLine::new(q_star)
                                .name(format!("MSNE q* = {:.3}", q_star))
                                .color(Color32::from_rgb(168, 85, 247)),
                        );
                        plot_ui.points(
                            Points::new(vec![[q_star, eu1]])
                                .name(format!(
                                    "Indifference Point (q*={:.3}, E[U1]={:.3})",
                                    q_star, eu1
                                ))
                                .radius(6.0_f32)
                                .color(Color32::from_rgb(34, 197, 94)),
                        );
                    }
                }
            });
    }

    fn render_3d_surface_tab(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.add(egui::Slider::new(&mut self.yaw_deg, -180.0_f32..=180.0_f32).text("Yaw (°)"));
            ui.add(egui::Slider::new(&mut self.pitch_deg, 5.0_f32..=85.0_f32).text("Pitch (°)"));
            ui.add(egui::Slider::new(&mut self.surface_zoom, 0.4_f32..=2.5_f32).text("Zoom"));
            ui.add(egui::Slider::new(&mut self.surface_resolution, 12..=44).text("Grid N"));
            ui.checkbox(&mut self.surface_wireframe_only, "Wireframe Only");
            if ui.button("Reset Camera").clicked() {
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
        let bg_color = if self.dark_mode {
            Color32::from_rgb(15, 23, 42)
        } else {
            Color32::from_rgb(241, 245, 249)
        };
        painter.rect_filled(rect, Rounding::same(6.0_f32), bg_color);

        let n = self.surface_resolution.max(8);
        let (x_star, y_star, _, _) = self.solve_constrained_equilibrium();
        let tfp_a = self.get_param("A", 2.0);
        let alpha = self.get_param("alpha", 0.45);
        let beta = self.get_param("beta", 0.55);
        let max_domain = (x_star.max(y_star) * 2.0).clamp(10.0, 40.0);

        let mut z_grid = vec![vec![0.0_f64; n + 1]; n + 1];
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        for (i, row) in z_grid.iter_mut().enumerate() {
            let x = 0.2 + (max_domain - 0.2) * (i as f64) / (n as f64);
            for (j, cell) in row.iter_mut().enumerate() {
                let y = 0.2 + (max_domain - 0.2) * (j as f64) / (n as f64);
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
        let z_span = (z_max - z_min).max(1e-6);

        let yaw = self.yaw_deg.to_radians();
        let pitch = self.pitch_deg.to_radians();
        let cos_y = yaw.cos();
        let sin_y = yaw.sin();
        let cos_p = pitch.cos();
        let sin_p = pitch.sin();

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

            let screen_pos = Pos2::new(center.x + sx * scale, center.y + sy * scale);
            (screen_pos, depth)
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

                let avg_depth = 0.25_f32 * (d0 + d1 + d2 + d3);
                let avg_nz = 0.25_f32 * (nz00 + nz10 + nz11 + nz01);

                quads.push(SurfaceQuad {
                    pts: [p0, p1, p2, p3],
                    depth: avg_depth,
                    norm_z: avg_nz,
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
            let r = (30.0_f32 + 210.0_f32 * t) as u8;
            let g = (100.0_f32 + 120.0_f32 * (1.0_f32 - (2.0_f32 * t - 1.0_f32).abs())) as u8;
            let b = (235.0_f32 - 180.0_f32 * t) as u8;

            if !self.surface_wireframe_only {
                let fill = Color32::from_rgba_unmultiplied(r, g, b, 210);
                painter.add(egui::Shape::convex_polygon(
                    q.pts.to_vec(),
                    fill,
                    Stroke::new(
                        0.65_f32,
                        Color32::from_rgba_unmultiplied(15, 23, 42, 140),
                    ),
                ));
            } else {
                let stroke_col = Color32::from_rgb(r, g, b);
                painter.add(egui::Shape::closed_line(
                    q.pts.to_vec(),
                    Stroke::new(1.1_f32, stroke_col),
                ));
            }
        }

        let nx_opt = ((x_star / max_domain) as f32).clamp(0.0_f32, 1.0_f32);
        let ny_opt = ((y_star / max_domain) as f32).clamp(0.0_f32, 1.0_f32);
        let z_opt = tfp_a * x_star.powf(alpha) * y_star.powf(beta);
        let nz_opt = (((z_opt - z_min) / z_span) as f32).clamp(0.0_f32, 1.0_f32);
        let (opt_pos, _) = project(nx_opt, ny_opt, nz_opt);
        painter.circle_filled(opt_pos, 6.0_f32, Color32::from_rgb(34, 197, 94));
        painter.circle_stroke(opt_pos, 6.0_f32, Stroke::new(2.0_f32, Color32::WHITE));

        painter.text(
            rect.left_top() + Vec2::new(14.0_f32, 12.0_f32),
            egui::Align2::LEFT_TOP,
            format!(
                "Surface Z(x,y) = {:.2} * x^{:.2} * y^{:.2} | Equilibrium ({:.2}, {:.2}, Z*={:.2}) — Drag to Rotate",
                tfp_a, alpha, beta, x_star, y_star, z_opt
            ),
            FontId::monospace(12.5_f32),
            Color32::from_rgb(125, 211, 252),
        );
    }
}

impl eframe::App for EconCalcApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("ide_top_menu_bar").show(ctx, |ui| {
            self.render_top_menu_bar(ui);
        });

        egui::SidePanel::left("vscode_activity_and_sidebar")
            .resizable(true)
            .default_width(265.0_f32)
            .min_width(185.0_f32)
            .show(ctx, |ui| {
                self.render_left_activity_and_sidebar(ui);
            });

        egui::SidePanel::right("rstudio_environment_and_grid_pane")
            .resizable(true)
            .default_width(370.0_f32)
            .min_width(280.0_f32)
            .show(ctx, |ui| {
                self.render_right_environment_pane(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_center_workspace(ui);
        });
    }
}
