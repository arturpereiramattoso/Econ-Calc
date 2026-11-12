use crate::exporter::{
    ActiveModelKind, ExportPayload, GridRow, ReportFormat, export_and_launch_positron,
};
use eframe::egui::{
    self, Color32, FontId, Pos2, RichText, Rounding, Sense, Stroke, Ui, Vec2,
};
use egui_plot::{Legend, Line, Plot, PlotPoints, Points, VLine};
use nalgebra::{Matrix2, Vector2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewportTab {
    Plot2D,
    Surface3D,
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
    // Theme & UI state
    dark_mode: bool,
    active_model: ActiveModelKind,
    viewport_tab: ViewportTab,
    report_format: ReportFormat,

    // Custom Expression tab state
    custom_expr: String,
    secondary_expr: String,
    domain_min: f64,
    domain_max: f64,
    eval_point_x: f64,
    integral_a: f64,
    integral_b: f64,
    show_derivative_curve: bool,

    // Cobb-Douglas & Constrained Utility state
    tfp_a: f64,
    alpha: f64,
    beta: f64,
    price_x: f64,
    price_y: f64,
    income: f64,
    target_output: f64,
    input_k_probe: f64,
    input_l_probe: f64,

    // 2x2 Normal-Form Game Payoff Matrix: [row][col] -> (Player 1, Player 2)
    payoff_matrix: [[(f64, f64); 2]; 2],
    row_labels: [String; 2],
    col_labels: [String; 2],

    // 3D Isometric Surface controls
    yaw_deg: f32,
    pitch_deg: f32,
    surface_zoom: f32,
    surface_resolution: usize,
    surface_wireframe_only: bool,

    // Console REPL state
    console_input: String,
    console_history: Vec<ConsoleEntry>,
    status_banner: String,
}

impl EconCalcApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(8.0, 6.0);
        style.visuals = egui::Visuals::dark();
        cc.egui_ctx.set_style(style);

        let mut app = Self {
            dark_mode: true,
            active_model: ActiveModelKind::ConstrainedUtility,
            viewport_tab: ViewportTab::Plot2D,
            report_format: ReportFormat::QuartoQmd,

            custom_expr: "A * x^alpha - price_x * x".to_string(),
            secondary_expr: "0.5 * x^1.4".to_string(),
            domain_min: 0.1,
            domain_max: 25.0,
            eval_point_x: 6.0,
            integral_a: 1.0,
            integral_b: 12.0,
            show_derivative_curve: true,

            tfp_a: 2.0,
            alpha: 0.45,
            beta: 0.55,
            price_x: 4.0,
            price_y: 5.0,
            income: 100.0,
            target_output: 18.0,
            input_k_probe: 10.0,
            input_l_probe: 12.0,

            payoff_matrix: [
                [(4.0, 4.0), (1.0, 6.0)],
                [(6.0, 1.0), (2.0, 2.0)],
            ],
            row_labels: ["Cooperate (U)".to_string(), "Defect (D)".to_string()],
            col_labels: ["Cooperate (L)".to_string(), "Defect (R)".to_string()],

            yaw_deg: 38.0,
            pitch_deg: 28.0,
            surface_zoom: 1.0,
            surface_resolution: 26,
            surface_wireframe_only: false,

            console_input: String::new(),
            console_history: Vec::new(),
            status_banner: "Ready — Native Windows 11 ARM64 (aarch64-pc-windows-msvc) Workspace"
                .to_string(),
        };

        app.console_history.push(ConsoleEntry {
            command: ":help".to_string(),
            output: "Commands: `eval <expr> at <x>`, `diff <expr> at <x>`, `diff2 <expr> at <x>`, `int <expr> from <a> to <b>`, `root <expr> near <x0>`, `hessian`, or enter any math expression using A, alpha, beta, price_x, price_y, I, x.".to_string(),
            is_error: false,
        });

        app
    }

    /// Evaluates a single-variable expression `f(x)` with active economic parameters bound into the context.
    fn eval_expr_at(&self, expr_str: &str, x_val: f64) -> Result<f64, String> {
        let mut ctx = meval::Context::new();
        ctx.var("x", x_val);
        ctx.var("A", self.tfp_a);
        ctx.var("alpha", self.alpha);
        ctx.var("beta", self.beta);
        ctx.var("price_x", self.price_x);
        ctx.var("price_y", self.price_y);
        ctx.var("I", self.income);
        ctx.var("K", self.input_k_probe);
        ctx.var("L", self.input_l_probe);
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

    /// Composite Simpson's Rule definite integral on [a, b] with N = 200 subintervals.
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

    /// Hybrid Newton-Raphson solver with bracketed bisection fallback.
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

        // Fallback bisection scan across active domain
        let samples = 200;
        let step = (self.domain_max - self.domain_min) / (samples as f64);
        let mut prev_x = self.domain_min;
        let mut prev_f = self.eval_expr_at(expr_str, prev_x)?;

        for i in 1..=samples {
            let curr_x = self.domain_min + (i as f64) * step;
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

    /// Computes analytical Marshallian demand and Lagrangian equilibrium for Constrained Utility / Cost Minimization.
    fn solve_constrained_equilibrium(&self) -> (f64, f64, f64, f64) {
        let sum_elast = (self.alpha + self.beta).max(1e-9);
        let px = self.price_x.max(1e-6);
        let py = self.price_y.max(1e-6);

        match self.active_model {
            ActiveModelKind::CobbDouglas => {
                // Cost-minimizing input demand for target isoquant Y_bar = A * L^alpha * K^beta
                let a = self.tfp_a.max(1e-6);
                let y_bar = self.target_output.max(1e-6);
                let ratio = (self.alpha / self.beta).max(1e-6) * (py / px);
                let l_star = ((y_bar / a) * ratio.powf(self.beta)).powf(1.0 / sum_elast);
                let k_star = ((y_bar / a) * ratio.powf(-self.alpha)).powf(1.0 / sum_elast);
                let min_cost = px * l_star + py * k_star;
                let mp_l = a * self.alpha * l_star.max(1e-6).powf(self.alpha - 1.0)
                    * k_star.max(1e-6).powf(self.beta);
                let lambda_mc = if mp_l.abs() > 1e-9 { px / mp_l } else { 0.0 };
                (l_star, k_star, lambda_mc, min_cost)
            }
            _ => {
                // Marshallian demand for U(x, y) = A * x^alpha * y^beta s.t. Px*x + Py*y = I
                let x_star = (self.alpha / sum_elast) * (self.income / px);
                let y_star = (self.beta / sum_elast) * (self.income / py);
                let indirect_u = self.tfp_a
                    * x_star.max(1e-9).powf(self.alpha)
                    * y_star.max(1e-9).powf(self.beta);
                let mu_x = self.tfp_a
                    * self.alpha
                    * x_star.max(1e-9).powf(self.alpha - 1.0)
                    * y_star.max(1e-9).powf(self.beta);
                let lambda_star = mu_x / px;
                (x_star, y_star, lambda_star, indirect_u)
            }
        }
    }

    /// Uses `nalgebra::Matrix2` to compute the Hessian matrix and determinant at the probe/equilibrium point.
    fn compute_cobb_douglas_hessian(&self, x: f64, y: f64) -> (Matrix2<f64>, f64, f64) {
        let x_safe = x.max(1e-4);
        let y_safe = y.max(1e-4);
        let f_val = self.tfp_a * x_safe.powf(self.alpha) * y_safe.powf(self.beta);
        let f_xx = self.alpha * (self.alpha - 1.0) * f_val / (x_safe * x_safe);
        let f_yy = self.beta * (self.beta - 1.0) * f_val / (y_safe * y_safe);
        let f_xy = self.alpha * self.beta * f_val / (x_safe * y_safe);

        let hessian = Matrix2::new(f_xx, f_xy, f_xy, f_yy);
        let det = hessian.determinant();
        let trace = hessian.trace();
        (hessian, det, trace)
    }

    /// Solves the 2x2 Normal-Form Game for all Pure Strategy Nash Equilibria (PSNE) and Mixed Strategy Nash Equilibria (MSNE).
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

        // Mixed strategy denominators using nalgebra Vector2 operations
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
                "p* (Row plays U) = {:.4}, q* (Col plays L) = {:.4} | E[U1] = {:.4}, E[U2] = {:.4}",
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

    /// Generates the numerical simulation table for the right-hand Environment Data Grid and CSV export.
    fn build_simulation_grid(&self, rows_count: usize) -> Vec<GridRow> {
        let mut grid = Vec::with_capacity(rows_count);
        let (x_star, y_star, _, indirect_u) = self.solve_constrained_equilibrium();

        match self.active_model {
            ActiveModelKind::CustomExpression => {
                let span = (self.domain_max - self.domain_min).max(1e-3);
                for i in 0..rows_count {
                    let x = self.domain_min + span * (i as f64) / ((rows_count - 1).max(1) as f64);
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
            ActiveModelKind::CobbDouglas => {
                let max_l = (x_star * 2.4).max(15.0);
                let a = self.tfp_a.max(1e-6);
                let beta = self.beta.max(1e-6);
                let y_bar = self.target_output.max(1e-6);
                let min_cost = self.price_x * x_star + self.price_y * y_star;

                for i in 0..rows_count {
                    let l = 0.25 + (max_l - 0.25) * (i as f64) / ((rows_count - 1).max(1) as f64);
                    let k_isoquant = (y_bar / (a * l.powf(self.alpha)))
                        .max(0.0)
                        .powf(1.0 / beta);
                    let k_isocost = ((min_cost - self.price_x * l) / self.price_y).max(0.0);
                    let mrts = (self.alpha / beta) * (k_isoquant / l);
                    grid.push(GridRow {
                        index: i + 1,
                        x: l,
                        primary_y: k_isoquant,
                        secondary_y: k_isocost,
                        marginal_or_slope: -mrts,
                    });
                }
            }
            ActiveModelKind::ConstrainedUtility => {
                let max_x = (self.income / self.price_x.max(1e-6)) * 1.15;
                let a = self.tfp_a.max(1e-6);
                let beta = self.beta.max(1e-6);
                let u_bar = indirect_u.max(1e-6);

                for i in 0..rows_count {
                    let x = 0.15 + (max_x - 0.15) * (i as f64) / ((rows_count - 1).max(1) as f64);
                    let y_indiff = (u_bar / (a * x.powf(self.alpha)))
                        .max(0.0)
                        .powf(1.0 / beta);
                    let y_budget = ((self.income - self.price_x * x) / self.price_y).max(0.0);
                    let mrs = (self.alpha / beta) * (y_indiff / x);
                    grid.push(GridRow {
                        index: i + 1,
                        x,
                        primary_y: y_indiff,
                        secondary_y: y_budget,
                        marginal_or_slope: -mrs,
                    });
                }
            }
            ActiveModelKind::GameTheory2x2 => {
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

    /// Generates publication-ready LaTeX (`align*`, `tabular`, and `pgfplots` TikZ code) for the active model.
    fn generate_latex_and_tikz(&self) -> String {
        let (x_star, y_star, lambda_star, obj_star) = self.solve_constrained_equilibrium();

        match self.active_model {
            ActiveModelKind::CustomExpression => {
                let fx = self
                    .eval_expr_at(&self.custom_expr, self.eval_point_x)
                    .unwrap_or(0.0);
                let dfx = self
                    .numerical_derivative(&self.custom_expr, self.eval_point_x)
                    .unwrap_or(0.0);
                let d2fx = self
                    .numerical_second_derivative(&self.custom_expr, self.eval_point_x)
                    .unwrap_or(0.0);
                let integral = self
                    .composite_simpson(&self.custom_expr, self.integral_a, self.integral_b, 200)
                    .unwrap_or(0.0);

                format!(
                    r#"% Generated by econ-calc (Windows 11 ARM64)
\begin{{align*}}
f(x) &= \text{{{expr}}} \\
f({x0:.4}) &= {fx:.6}, \quad f'({x0:.4}) = {dfx:.6}, \quad f''({x0:.4}) = {d2fx:.6} \\
\int_{{{a:.2}}}^{{{b:.2}}} f(x)\,dx &\approx {integral:.6}
\end{{align*}}
"#,
                    expr = self.custom_expr,
                    x0 = self.eval_point_x,
                    fx = fx,
                    dfx = dfx,
                    d2fx = d2fx,
                    a = self.integral_a,
                    b = self.integral_b,
                    integral = integral
                )
            }
            ActiveModelKind::CobbDouglas => {
                let rts = self.alpha + self.beta;
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
                    a = self.tfp_a,
                    alpha = self.alpha,
                    beta = self.beta,
                    rts = rts,
                    px = self.price_x,
                    py = self.price_y,
                    x_star = x_star,
                    y_star = y_star,
                    ybar = self.target_output,
                    obj_star = obj_star,
                    lambda_star = lambda_star,
                    xmax = (x_star * 2.2).max(10.0),
                    ymax = (y_star * 2.2).max(10.0)
                )
            }
            ActiveModelKind::ConstrainedUtility => {
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
                    a = self.tfp_a,
                    alpha = self.alpha,
                    beta = self.beta,
                    alpha_m1 = self.alpha - 1.0,
                    beta_m1 = self.beta - 1.0,
                    income = self.income,
                    px = self.price_x,
                    py = self.price_y,
                    x_star = x_star,
                    y_star = y_star,
                    obj_star = obj_star,
                    lambda_star = lambda_star,
                    xmax = (self.income / self.price_x.max(0.1)) * 1.1,
                    ymax = (self.income / self.price_y.max(0.1)) * 1.2
                )
            }
            ActiveModelKind::GameTheory2x2 => {
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

    /// Executes a command entered into the bottom Numerical CAS REPL.
    fn execute_console_command(&mut self) {
        let cmd = self.console_input.trim().to_string();
        if cmd.is_empty() {
            return;
        }
        self.console_input.clear();

        if cmd == ":clear" {
            self.console_history.clear();
            return;
        }

        let response = self.parse_and_run_repl(&cmd);
        match response {
            Ok(output) => self.console_history.push(ConsoleEntry {
                command: cmd,
                output,
                is_error: false,
            }),
            Err(err_msg) => self.console_history.push(ConsoleEntry {
                command: cmd,
                output: err_msg,
                is_error: true,
            }),
        }
    }

    fn parse_and_run_repl(&self, cmd: &str) -> Result<String, String> {
        if cmd == ":help" {
            return Ok(
                "Available commands: `eval <expr> at <x>`, `diff <expr> at <x>`, `diff2 <expr> at <x>`, `int <expr> from <a> to <b>`, `root <expr> near <x0>`, `hessian`, or any mathematical expression."
                    .to_string(),
            );
        }

        if cmd == "hessian" {
            let (x_star, y_star, _, _) = self.solve_constrained_equilibrium();
            let (h, det, trace) = self.compute_cobb_douglas_hessian(x_star, y_star);
            return Ok(format!(
                "Hessian H(x*={:.3}, y*={:.3}) = [[{:.5}, {:.5}], [{:.5}, {:.5}]] | det(H) = {:.6}, trace(H) = {:.6}",
                x_star, y_star, h[(0, 0)], h[(0, 1)], h[(1, 0)], h[(1, 1)], det, trace
            ));
        }

        if let Some(rest) = cmd.strip_prefix("eval ") {
            if let Some((expr, x_str)) = rest.rsplit_once(" at ") {
                let x: f64 = x_str.trim().parse().map_err(|_| format!("Invalid number `{}`", x_str.trim()))?;
                let val = self.eval_expr_at(expr.trim(), x)?;
                return Ok(format!("f({:.6}) = {:.8}", x, val));
            }
        }

        if let Some(rest) = cmd.strip_prefix("diff2 ") {
            if let Some((expr, x_str)) = rest.rsplit_once(" at ") {
                let x: f64 = x_str.trim().parse().map_err(|_| format!("Invalid number `{}`", x_str.trim()))?;
                let val = self.numerical_second_derivative(expr.trim(), x)?;
                return Ok(format!("f''({:.6}) = {:.8} (central difference)", x, val));
            }
        }

        if let Some(rest) = cmd.strip_prefix("diff ") {
            if let Some((expr, x_str)) = rest.rsplit_once(" at ") {
                let x: f64 = x_str.trim().parse().map_err(|_| format!("Invalid number `{}`", x_str.trim()))?;
                let val = self.numerical_derivative(expr.trim(), x)?;
                return Ok(format!("f'({:.6}) = {:.8} (central difference)", x, val));
            }
        }

        if let Some(rest) = cmd.strip_prefix("int ") {
            if let Some((expr, bounds)) = rest.rsplit_once(" from ") {
                if let Some((a_str, b_str)) = bounds.split_once(" to ") {
                    let a: f64 = a_str.trim().parse().map_err(|_| format!("Invalid lower bound `{}`", a_str.trim()))?;
                    let b: f64 = b_str.trim().parse().map_err(|_| format!("Invalid upper bound `{}`", b_str.trim()))?;
                    let area = self.composite_simpson(expr.trim(), a, b, 200)?;
                    return Ok(format!(
                        "Integral of `{}` on [{:.4}, {:.4}] = {:.8} (Composite Simpson N=200)",
                        expr.trim(), a, b, area
                    ));
                }
            }
        }

        if let Some(rest) = cmd.strip_prefix("root ") {
            if let Some((expr, x0_str)) = rest.rsplit_once(" near ") {
                let x0: f64 = x0_str.trim().parse().map_err(|_| format!("Invalid initial guess `{}`", x0_str.trim()))?;
                let root = self.solve_root(expr.trim(), x0)?;
                let residual = self.eval_expr_at(expr.trim(), root)?;
                return Ok(format!(
                    "Root x* = {:.8} (residual f(x*) = {:.3e})",
                    root, residual
                ));
            }
        }

        // Default: evaluate expression at `eval_point_x`
        let val = self.eval_expr_at(cmd, self.eval_point_x)?;
        Ok(format!("= {:.8} (with x = {:.4})", val, self.eval_point_x))
    }

    fn trigger_positron_export(&mut self) {
        let (x_star, y_star, lambda_star, indirect_utility) = self.solve_constrained_equilibrium();
        let game_sol = self.solve_game_2x2();
        let payload = ExportPayload {
            model_kind: self.active_model,
            report_format: self.report_format,
            custom_expr: self.custom_expr.clone(),
            secondary_expr: self.secondary_expr.clone(),
            tfp_a: self.tfp_a,
            alpha: self.alpha,
            beta: self.beta,
            price_x: self.price_x,
            price_y: self.price_y,
            income: self.income,
            target_output: self.target_output,
            x_star,
            y_star,
            lambda_star,
            indirect_utility,
            game_matrix: self.payoff_matrix,
            game_summary: game_sol.summary_text,
            latex_derivation: self.generate_latex_and_tikz(),
            grid_rows: self.build_simulation_grid(100),
        };

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

    fn render_top_menu_bar(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new("econ-calc")
                    .font(FontId::proportional(17.0))
                    .strong()
                    .color(Color32::from_rgb(56, 189, 248)),
            );
            ui.separator();

            if ui
                .button(if self.dark_mode {
                    "Theme: Positron Dark"
                } else {
                    "Theme: Positron Light"
                })
                .clicked()
            {
                self.dark_mode = !self.dark_mode;
                ui.ctx().set_visuals(if self.dark_mode {
                    egui::Visuals::dark()
                } else {
                    egui::Visuals::light()
                });
            }

            ui.separator();

            if ui
                .button(RichText::new("Copy as LaTeX / TikZ").strong())
                .on_hover_text(
                    "Formats the active model, Lagrangian FOCs, equilibrium, or 2x2 table into publication-ready LaTeX/TikZ and copies to clipboard.",
                )
                .clicked()
            {
                let latex = self.generate_latex_and_tikz();
                ui.ctx().copy_text(latex);
                self.status_banner =
                    "Copied publication-ready LaTeX & TikZ code to system clipboard.".to_string();
                self.console_history.push(ConsoleEntry {
                    command: ":copy latex".to_string(),
                    output: "Copied active model LaTeX/TikZ snippet to clipboard.".to_string(),
                    is_error: false,
                });
            }

            ui.separator();

            ui.label("Report Format:");
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
                    "Writes project.Rproj, model_analysis.R, report.qmd/.Rmd, and simulation_data.csv, then spawns Positron IDE.",
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

    fn render_left_editor_pane(&mut self, ui: &mut Ui) {
        ui.heading("1. Source & Model Editor");
        ui.add_space(4.0);

        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.active_model,
                ActiveModelKind::ConstrainedUtility,
                "Lagrangian Utility",
            );
            ui.selectable_value(
                &mut self.active_model,
                ActiveModelKind::CobbDouglas,
                "Cobb-Douglas",
            );
            ui.selectable_value(
                &mut self.active_model,
                ActiveModelKind::GameTheory2x2,
                "2x2 Game Theory",
            );
            ui.selectable_value(
                &mut self.active_model,
                ActiveModelKind::CustomExpression,
                "Custom CAS",
            );
        });

        ui.separator();

        egui::ScrollArea::vertical()
            .id_source("left_editor_scroll")
            .show(ui, |ui| match self.active_model {
                ActiveModelKind::CustomExpression => self.render_custom_expr_editor(ui),
                ActiveModelKind::CobbDouglas => self.render_cobb_douglas_editor(ui),
                ActiveModelKind::ConstrainedUtility => self.render_constrained_utility_editor(ui),
                ActiveModelKind::GameTheory2x2 => self.render_game_theory_editor(ui),
            });
    }

    fn render_custom_expr_editor(&mut self, ui: &mut Ui) {
        ui.label(RichText::new("Primary Function f(x):").strong());
        ui.text_edit_singleline(&mut self.custom_expr);

        ui.label(RichText::new("Secondary / Benchmark Function g(x):").strong());
        ui.text_edit_singleline(&mut self.secondary_expr);

        ui.checkbox(
            &mut self.show_derivative_curve,
            "Overlay Numerical Derivative f'(x) on 2D Plot",
        );

        ui.separator();
        ui.label(RichText::new("Domain & Numerical Calculus Controls").strong());
        ui.add(
            egui::Slider::new(&mut self.domain_min, -20.0..=10.0).text("Domain x_min"),
        );
        ui.add(
            egui::Slider::new(&mut self.domain_max, 1.0..=100.0).text("Domain x_max"),
        );
        if self.domain_max <= self.domain_min + 0.5 {
            self.domain_max = self.domain_min + 0.5;
        }
        ui.add(
            egui::Slider::new(&mut self.eval_point_x, self.domain_min..=self.domain_max)
                .text("Probe Point x0"),
        );
        ui.add(
            egui::Slider::new(&mut self.integral_a, self.domain_min..=self.domain_max)
                .text("Integral Lower a"),
        );
        ui.add(
            egui::Slider::new(&mut self.integral_b, self.domain_min..=self.domain_max)
                .text("Integral Upper b"),
        );

        ui.separator();
        ui.label(RichText::new("Live Numerical Diagnostics").strong());
        let fx = self.eval_expr_at(&self.custom_expr, self.eval_point_x);
        let dfx = self.numerical_derivative(&self.custom_expr, self.eval_point_x);
        let d2fx = self.numerical_second_derivative(&self.custom_expr, self.eval_point_x);
        let area = self.composite_simpson(&self.custom_expr, self.integral_a, self.integral_b, 200);

        ui.monospace(format!(
            "f({:.4})   = {}",
            self.eval_point_x,
            fx.map(|v| format!("{:.6}", v))
                .unwrap_or_else(|e| e)
        ));
        ui.monospace(format!(
            "f'({:.4})  = {}",
            self.eval_point_x,
            dfx.map(|v| format!("{:.6}", v))
                .unwrap_or_else(|e| e)
        ));
        ui.monospace(format!(
            "f''({:.4}) = {}",
            self.eval_point_x,
            d2fx.map(|v| format!("{:.6}", v))
                .unwrap_or_else(|e| e)
        ));
        ui.monospace(format!(
            "Integral [{:.2}, {:.2}] = {}",
            self.integral_a,
            self.integral_b,
            area.map(|v| format!("{:.6}", v))
                .unwrap_or_else(|e| e)
        ));

        ui.add_space(6.0);
        if ui
            .button("Solve Root f(x) = 0 near x0")
            .clicked()
        {
            match self.solve_root(&self.custom_expr, self.eval_point_x) {
                Ok(r) => {
                    self.eval_point_x = r;
                    self.console_history.push(ConsoleEntry {
                        command: format!("root {} near {:.4}", self.custom_expr, self.eval_point_x),
                        output: format!("Converged to root x* = {:.8}", r),
                        is_error: false,
                    });
                }
                Err(e) => {
                    self.console_history.push(ConsoleEntry {
                        command: format!("root {}", self.custom_expr),
                        output: e,
                        is_error: true,
                    });
                }
            }
        }

        if ui
            .button("Solve Intersection f(x) = g(x)")
            .clicked()
        {
            let diff_expr = format!("({}) - ({})", self.custom_expr, self.secondary_expr);
            match self.solve_root(&diff_expr, self.eval_point_x) {
                Ok(r) => {
                    self.eval_point_x = r;
                    let y_val = self.eval_expr_at(&self.custom_expr, r).unwrap_or(0.0);
                    self.console_history.push(ConsoleEntry {
                        command: format!("intersect ({}) == ({})", self.custom_expr, self.secondary_expr),
                        output: format!("Intersection at (x* = {:.6}, y* = {:.6})", r, y_val),
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
    }

    fn render_cobb_douglas_editor(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Production Specification: Y = A * L^alpha * K^beta")
                .strong()
                .color(Color32::from_rgb(96, 165, 250)),
        );
        ui.add_space(4.0);

        let rts = self.alpha + self.beta;
        let rts_class = if (rts - 1.0).abs() < 1e-3 {
            "Constant Returns to Scale (CRS)"
        } else if rts > 1.0 {
            "Increasing Returns to Scale (IRS)"
        } else {
            "Decreasing Returns to Scale (DRS)"
        };

        let l = self.input_l_probe.max(1e-4);
        let k = self.input_k_probe.max(1e-4);
        let y_out = self.tfp_a * l.powf(self.alpha) * k.powf(self.beta);
        let mp_l = self.alpha * y_out / l;
        let mp_k = self.beta * y_out / k;
        let mrts_lk = mp_l / mp_k.max(1e-9);
        let (l_star, k_star, lambda_mc, min_cost) = self.solve_constrained_equilibrium();
        let (_, det_h, trace_h) = self.compute_cobb_douglas_hessian(l_star, k_star);

        ui.label(format!("Returns to Scale (alpha + beta = {:.3}):", rts));
        ui.label(RichText::new(rts_class).strong());
        ui.separator();

        ui.label(RichText::new("Probe Input Bundle (L, K) Diagnostics:").strong());
        ui.add(egui::Slider::new(&mut self.input_l_probe, 0.5..=40.0).text("Probe Labor (L)"));
        ui.add(egui::Slider::new(&mut self.input_k_probe, 0.5..=40.0).text("Probe Capital (K)"));

        ui.monospace(format!("Output Y(L, K)         = {:.6}", y_out));
        ui.monospace(format!("Marginal Product MP_L  = {:.6}", mp_l));
        ui.monospace(format!("Marginal Product MP_K  = {:.6}", mp_k));
        ui.monospace(format!("MRTS_(L,K) = MP_L/MP_K = {:.6}", mrts_lk));
        ui.monospace(format!(
            "Factor Price Ratio w/r = {:.6}",
            self.price_x / self.price_y.max(1e-6)
        ));

        ui.separator();
        ui.label(RichText::new("Cost-Minimizing Bundle for Isoquant Y_bar:").strong());
        ui.monospace(format!("Optimal Labor L*       = {:.6}", l_star));
        ui.monospace(format!("Optimal Capital K*     = {:.6}", k_star));
        ui.monospace(format!("Minimum Cost C*(Y_bar) = {:.6}", min_cost));
        ui.monospace(format!("Marginal Cost lambda*  = {:.6}", lambda_mc));
        ui.monospace(format!(
            "Hessian det(H) = {:.4e}, tr(H) = {:.4}",
            det_h, trace_h
        ));
    }

    fn render_constrained_utility_editor(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Max U(x,y) = A * x^alpha * y^beta  s.t.  Px*x + Py*y = I")
                .strong()
                .color(Color32::from_rgb(96, 165, 250)),
        );
        ui.add_space(4.0);

        let (x_star, y_star, lambda_star, indirect_u) = self.solve_constrained_equilibrium();
        let share_x = (self.price_x * x_star) / self.income.max(1e-6);
        let share_y = (self.price_y * y_star) / self.income.max(1e-6);
        let mrs_at_opt = (self.alpha / self.beta.max(1e-6)) * (y_star / x_star.max(1e-6));
        let price_ratio = self.price_x / self.price_y.max(1e-6);

        ui.label(RichText::new("Analytical Lagrangian Solution:").strong());
        ui.monospace("L(x,y,lambda) = A*x^alpha*y^beta + lambda*(I - Px*x - Py*y)");
        ui.add_space(4.0);
        ui.monospace(format!("Marshallian Demand x*    = {:.6}", x_star));
        ui.monospace(format!("Marshallian Demand y*    = {:.6}", y_star));
        ui.monospace(format!("Indirect Utility V*      = {:.6}", indirect_u));
        ui.monospace(format!("Shadow Price lambda*     = {:.6}", lambda_star));
        ui.monospace(format!("Tangency MRS(x*, y*)     = {:.6}", mrs_at_opt));
        ui.monospace(format!("Relative Price Px / Py   = {:.6}", price_ratio));
        ui.monospace(format!(
            "Expenditure Shares       = ({:.1}%, {:.1}%)",
            share_x * 100.0,
            share_y * 100.0
        ));

        ui.separator();
        ui.label(RichText::new("Quick Preset Scenarios:").strong());
        ui.horizontal_wrapped(|ui| {
            if ui.button("Balanced (alpha=0.5, beta=0.5)").clicked() {
                self.alpha = 0.5;
                self.beta = 0.5;
                self.price_x = 4.0;
                self.price_y = 4.0;
                self.income = 100.0;
            }
            if ui.button("Good X Intensive (0.7 / 0.3)").clicked() {
                self.alpha = 0.7;
                self.beta = 0.3;
                self.price_x = 3.0;
                self.price_y = 6.0;
                self.income = 120.0;
            }
        });
    }

    fn render_game_theory_editor(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("2x2 Normal-Form Strategic Game Solver")
                .strong()
                .color(Color32::from_rgb(96, 165, 250)),
        );
        ui.add_space(4.0);

        ui.label("Preset Classic Games:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Prisoner's Dilemma").clicked() {
                self.row_labels = ["Cooperate".to_string(), "Defect".to_string()];
                self.col_labels = ["Cooperate".to_string(), "Defect".to_string()];
                self.payoff_matrix = [[(3.0, 3.0), (0.0, 5.0)], [(5.0, 0.0), (1.0, 1.0)]];
            }
            if ui.button("Matching Pennies").clicked() {
                self.row_labels = ["Heads".to_string(), "Tails".to_string()];
                self.col_labels = ["Heads".to_string(), "Tails".to_string()];
                self.payoff_matrix = [[(1.0, -1.0), (-1.0, 1.0)], [(-1.0, 1.0), (1.0, -1.0)]];
            }
            if ui.button("Battle of the Sexes").clicked() {
                self.row_labels = ["Opera".to_string(), "Football".to_string()];
                self.col_labels = ["Opera".to_string(), "Football".to_string()];
                self.payoff_matrix = [[(3.0, 2.0), (0.0, 0.0)], [(0.0, 0.0), (2.0, 3.0)]];
            }
            if ui.button("Stag Hunt").clicked() {
                self.row_labels = ["Stag".to_string(), "Hare".to_string()];
                self.col_labels = ["Stag".to_string(), "Hare".to_string()];
                self.payoff_matrix = [[(4.0, 4.0), (1.0, 3.0)], [(3.0, 1.0), (2.0, 2.0)]];
            }
        });

        ui.separator();
        ui.label(RichText::new("Payoff Matrix (Player 1, Player 2):").strong());

        egui::Grid::new("payoff_matrix_grid")
            .striped(true)
            .spacing([10.0, 8.0])
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
                                    .prefix("u1: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.payoff_matrix[r][c].1)
                                    .speed(0.25)
                                    .prefix("u2: "),
                            );
                        });
                    }
                    ui.end_row();
                }
            });

        ui.separator();
        let sol = self.solve_game_2x2();
        ui.label(RichText::new("Pure Strategy Nash Equilibria (PSNE):").strong());
        if sol.pure_equilibria.is_empty() {
            ui.label("None (strictly cyclic best responses)");
        } else {
            for (_, _, desc) in &sol.pure_equilibria {
                ui.label(
                    RichText::new(format!("• {}", desc))
                        .color(Color32::from_rgb(34, 197, 94))
                        .strong(),
                );
            }
        }

        ui.add_space(4.0);
        ui.label(RichText::new("Mixed Strategy Nash Equilibrium (MSNE):").strong());
        match (sol.mixed_p, sol.mixed_q, sol.expected_u1, sol.expected_u2) {
            (Some(p), Some(q), Some(eu1), Some(eu2)) => {
                ui.monospace(format!(
                    "p* (P1 plays `{}`) = {:.4} (1 - p* = {:.4})",
                    self.row_labels[0],
                    p,
                    1.0 - p
                ));
                ui.monospace(format!(
                    "q* (P2 plays `{}`) = {:.4} (1 - q* = {:.4})",
                    self.col_labels[0],
                    q,
                    1.0 - q
                ));
                ui.monospace(format!("Expected Payoff E[U1] = {:.4}", eu1));
                ui.monospace(format!("Expected Payoff E[U2] = {:.4}", eu2));
            }
            _ => {
                ui.label("No interior mixed strategy equilibrium in (0, 1).");
            }
        }
    }

    fn render_right_environment_pane(&mut self, ui: &mut Ui) {
        ui.heading("3. Environment & Data Grid");
        ui.add_space(4.0);

        ui.label(RichText::new("Structural Parameters (Live Sliders)").strong());
        ui.add(egui::Slider::new(&mut self.tfp_a, 0.2..=10.0).text("TFP / Scale (A)"));
        ui.add(egui::Slider::new(&mut self.alpha, 0.05..=0.95).text("Elasticity alpha"));
        ui.add(egui::Slider::new(&mut self.beta, 0.05..=0.95).text("Elasticity beta"));
        ui.add(egui::Slider::new(&mut self.price_x, 0.5..=25.0).text("Price Px / Wage w"));
        ui.add(egui::Slider::new(&mut self.price_y, 0.5..=25.0).text("Price Py / Rent r"));
        ui.add(egui::Slider::new(&mut self.income, 10.0..=500.0).text("Budget Income I"));
        ui.add(
            egui::Slider::new(&mut self.target_output, 2.0..=120.0).text("Target Isoquant Y_bar"),
        );

        ui.separator();
        let (x_star, y_star, lambda_star, obj_star) = self.solve_constrained_equilibrium();
        ui.label(RichText::new("Active Equilibrium Summary").strong());
        egui::Grid::new("env_metrics_summary")
            .striped(true)
            .show(ui, |ui| {
                ui.label("x* (or L*)");
                ui.monospace(format!("{:.5}", x_star));
                ui.label("y* (or K*)");
                ui.monospace(format!("{:.5}", y_star));
                ui.end_row();

                ui.label("lambda*");
                ui.monospace(format!("{:.5}", lambda_star));
                ui.label("V* / C*");
                ui.monospace(format!("{:.5}", obj_star));
                ui.end_row();
            });

        ui.separator();
        ui.label(RichText::new("Evaluated Numerical Simulation Grid (50 rows)").strong());

        let grid_rows = self.build_simulation_grid(50);
        let (col1, col2, col3, col4) = match self.active_model {
            ActiveModelKind::CustomExpression => ("x", "f(x)", "g(x)", "f'(x)"),
            ActiveModelKind::CobbDouglas => ("L", "K_isoquant", "K_isocost", "-MRTS"),
            ActiveModelKind::ConstrainedUtility => ("x", "y_indiff", "y_budget", "-MRS"),
            ActiveModelKind::GameTheory2x2 => ("q (Col L)", "E[U1|Up]", "E[U1|Down]", "Diff"),
        };

        egui::ScrollArea::vertical()
            .id_source("right_env_grid_scroll")
            .show(ui, |ui| {
                egui::Grid::new("numerical_simulation_table")
                    .striped(true)
                    .spacing([12.0, 4.0])
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

    fn render_bottom_console_pane(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("2. Console & Numerical CAS REPL");
            ui.separator();
            if ui.small_button("Eval f(x0)").clicked() {
                self.console_input =
                    format!("eval {} at {:.4}", self.custom_expr, self.eval_point_x);
                self.execute_console_command();
            }
            if ui.small_button("Diff f'(x0)").clicked() {
                self.console_input =
                    format!("diff {} at {:.4}", self.custom_expr, self.eval_point_x);
                self.execute_console_command();
            }
            if ui.small_button("Integrate [a,b]").clicked() {
                self.console_input = format!(
                    "int {} from {:.2} to {:.2}",
                    self.custom_expr, self.integral_a, self.integral_b
                );
                self.execute_console_command();
            }
            if ui.small_button("Inspect Hessian").clicked() {
                self.console_input = "hessian".to_string();
                self.execute_console_command();
            }
            if ui.small_button("Clear").clicked() {
                self.console_history.clear();
            }
        });

        ui.separator();

        let available_height = (ui.available_height() - 34.0).max(60.0);
        egui::ScrollArea::vertical()
            .id_source("console_repl_scroll")
            .max_height(available_height)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for entry in &self.console_history {
                    ui.horizontal_wrapped(|ui| {
                        ui.monospace(
                            RichText::new(format!("> {}", entry.command))
                                .color(Color32::from_rgb(56, 189, 248))
                                .strong(),
                        );
                    });
                    let out_color = if entry.is_error {
                        Color32::from_rgb(248, 113, 113)
                    } else {
                        ui.visuals().text_color()
                    };
                    ui.monospace(RichText::new(format!("  {}", entry.output)).color(out_color));
                }
            });

        ui.horizontal(|ui| {
            ui.monospace(RichText::new("cas>").strong());
            let response = ui.add_sized(
                [ui.available_width() - 80.0, 24.0],
                egui::TextEdit::singleline(&mut self.console_input).hint_text(
                    "Type expression or command (`eval x^2 at 3`, `diff A*x^alpha at 4`, `int x^2 from 0 to 5`, `root x^2-9 near 2`)",
                ),
            );
            if (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                || ui.button("Run").clicked()
            {
                self.execute_console_command();
            }
        });
    }

    fn render_central_viewport(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("4. Interactive Plot & 3D Isometric Surface");
            ui.separator();
            ui.selectable_value(
                &mut self.viewport_tab,
                ViewportTab::Plot2D,
                "2D Economic Plot (egui_plot 0.38)",
            );
            ui.selectable_value(
                &mut self.viewport_tab,
                ViewportTab::Surface3D,
                "3D Isometric Surface (Adreno 60 FPS Painter)",
            );
        });
        ui.separator();

        match self.viewport_tab {
            ViewportTab::Plot2D => self.render_2d_plot_tab(ui),
            ViewportTab::Surface3D => self.render_3d_surface_tab(ui),
        }
    }

    fn render_2d_plot_tab(&mut self, ui: &mut Ui) {
        let (x_star, y_star, _, indirect_u) = self.solve_constrained_equilibrium();

        Plot::new("main_econ_plot")
            .legend(Legend::default())
            .show(ui, |plot_ui| match self.active_model {
                ActiveModelKind::CustomExpression => {
                    let span = (self.domain_max - self.domain_min).max(1e-3);
                    let primary_pts: PlotPoints = (0..=240)
                        .filter_map(|i| {
                            let x = self.domain_min + span * (i as f64) / 240.0;
                            let y = self.eval_expr_at(&self.custom_expr, x).ok()?;
                            y.is_finite().then_some([x, y])
                        })
                        .collect();

                    let secondary_pts: PlotPoints = (0..=240)
                        .filter_map(|i| {
                            let x = self.domain_min + span * (i as f64) / 240.0;
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
                                let x = self.domain_min + span * (i as f64) / 240.0;
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

                    if let Ok(y0) = self.eval_expr_at(&self.custom_expr, self.eval_point_x) {
                        if y0.is_finite() {
                            plot_ui.points(
                                Points::new(vec![[self.eval_point_x, y0]])
                                    .name(format!("Probe ({:.3}, {:.3})", self.eval_point_x, y0))
                                    .radius(5.5_f32)
                                    .color(Color32::from_rgb(34, 197, 94)),
                            );
                        }
                    }
                }
                ActiveModelKind::CobbDouglas => {
                    let a = self.tfp_a.max(1e-6);
                    let beta = self.beta.max(1e-6);
                    let max_l = (x_star * 2.5).max(15.0);
                    let max_k = (y_star * 2.5).max(15.0);
                    let min_cost = self.price_x * x_star + self.price_y * y_star;

                    for (mult, label_suffix, width) in [
                        (0.75, "0.75 * Y_bar", 1.3_f32),
                        (1.0, "Active Isoquant Y_bar", 2.6_f32),
                        (1.30, "1.30 * Y_bar", 1.3_f32),
                    ] {
                        let y_target = self.target_output * mult;
                        let iso_pts: PlotPoints = (1..=220)
                            .filter_map(|i| {
                                let l = 0.2 + (max_l - 0.2) * (i as f64) / 220.0;
                                let k = (y_target / (a * l.powf(self.alpha)))
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
                            let k = (min_cost - self.price_x * l) / self.price_y.max(1e-6);
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
                            .name(format!("Cost-Min Tangency (L*={:.2}, K*={:.2})", x_star, y_star))
                            .radius(6.0_f32)
                            .color(Color32::from_rgb(34, 197, 94)),
                    );
                }
                ActiveModelKind::ConstrainedUtility => {
                    let a = self.tfp_a.max(1e-6);
                    let beta = self.beta.max(1e-6);
                    let max_x = (self.income / self.price_x.max(1e-6)) * 1.25;
                    let max_y = (self.income / self.price_y.max(1e-6)) * 1.25;

                    for (mult, label_prefix, width) in [
                        (0.75, "Lower Indifference (0.75 U*)", 1.3_f32),
                        (1.0, "Optimal Indifference U*", 2.6_f32),
                        (1.25, "Higher Indifference (1.25 U*)", 1.3_f32),
                    ] {
                        let u_level = indirect_u.max(1e-6) * mult;
                        let indiff_pts: PlotPoints = (1..=220)
                            .filter_map(|i| {
                                let x = 0.15 + (max_x - 0.15) * (i as f64) / 220.0;
                                let y = (u_level / (a * x.powf(self.alpha)))
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
                            let y = (self.income - self.price_x * x) / self.price_y.max(1e-6);
                            (y >= 0.0).then_some([x, y])
                        })
                        .collect();

                    plot_ui.line(
                        Line::new(budget_pts)
                            .name(format!(
                                "Budget Constraint ({:.1}x + {:.1}y = {:.1})",
                                self.price_x, self.price_y, self.income
                            ))
                            .width(2.2_f32)
                            .color(Color32::from_rgb(239, 68, 68)),
                    );

                    plot_ui.points(
                        Points::new(vec![[x_star, y_star]])
                            .name(format!("Equilibrium (x*={:.2}, y*={:.2})", x_star, y_star))
                            .radius(6.0_f32)
                            .color(Color32::from_rgb(34, 197, 94)),
                    );
                }
                ActiveModelKind::GameTheory2x2 => {
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
                                .name(format!("Indifference Point (q*={:.3}, E[U1]={:.3})", q_star, eu1))
                                .radius(6.0_f32)
                                .color(Color32::from_rgb(34, 197, 94)),
                        );
                    }
                }
            });
    }

    fn render_3d_surface_tab(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.add(egui::Slider::new(&mut self.yaw_deg, -180.0..=180.0).text("Yaw (°)"));
            ui.add(egui::Slider::new(&mut self.pitch_deg, 5.0..=85.0).text("Pitch (°)"));
            ui.add(egui::Slider::new(&mut self.surface_zoom, 0.4..=2.5).text("Zoom"));
            ui.add(egui::Slider::new(&mut self.surface_resolution, 12..=42).text("Grid N"));
            ui.checkbox(&mut self.surface_wireframe_only, "Wireframe Only");
            if ui.button("Reset Camera").clicked() {
                self.yaw_deg = 38.0;
                self.pitch_deg = 28.0;
                self.surface_zoom = 1.0;
            }
        });

        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, Sense::drag());

        if response.dragged() {
            let delta = response.drag_delta();
            self.yaw_deg = (self.yaw_deg + delta.x * 0.45).clamp(-180.0, 180.0);
            self.pitch_deg = (self.pitch_deg + delta.y * 0.35).clamp(5.0, 85.0);
        }

        let painter = ui.painter_at(rect);
        let bg_color = if self.dark_mode {
            Color32::from_rgb(15, 23, 42)
        } else {
            Color32::from_rgb(241, 245, 249)
        };
        painter.rect_filled(rect, Rounding::same(6.0), bg_color);

        let n = self.surface_resolution.max(8);
        let (x_star, y_star, _, _) = self.solve_constrained_equilibrium();
        let max_domain = (x_star.max(y_star) * 2.0).clamp(10.0, 40.0);

        let mut z_grid = vec![vec![0.0_f64; n + 1]; n + 1];
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        for (i, row) in z_grid.iter_mut().enumerate() {
            let x = 0.2 + (max_domain - 0.2) * (i as f64) / (n as f64);
            for (j, cell) in row.iter_mut().enumerate() {
                let y = 0.2 + (max_domain - 0.2) * (j as f64) / (n as f64);
                let z = self.tfp_a * x.powf(self.alpha) * y.powf(self.beta);
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

        let scale = (rect.width().min(rect.height()) * 0.36) * self.surface_zoom;
        let center = rect.center() + Vec2::new(0.0, rect.height() * 0.08);

        let project = |nx: f32, ny: f32, nz: f32| -> (Pos2, f32) {
            let cx = nx * 2.0 - 1.0;
            let cy = ny * 2.0 - 1.0;
            let cz = nz * 1.5 - 0.5;

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

                let avg_depth = 0.25 * (d0 + d1 + d2 + d3);
                let avg_nz = 0.25 * (nz00 + nz10 + nz11 + nz01);

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
            let t = q.norm_z.clamp(0.0, 1.0);
            let r = (30.0 + 210.0 * t) as u8;
            let g = (100.0 + 120.0 * (1.0 - (2.0 * t - 1.0).abs())) as u8;
            let b = (235.0 - 180.0 * t) as u8;

            if !self.surface_wireframe_only {
                let fill = Color32::from_rgba_unmultiplied(r, g, b, 210);
                painter.add(egui::Shape::convex_polygon(
                    q.pts.to_vec(),
                    fill,
                    Stroke::new(0.65_f32, Color32::from_rgba_unmultiplied(15, 23, 42, 140)),
                ));
            } else {
                let stroke_col = Color32::from_rgb(r, g, b);
                painter.add(egui::Shape::closed_line(
                    q.pts.to_vec(),
                    Stroke::new(1.1_f32, stroke_col),
                ));
            }
        }

        let nx_opt = ((x_star / max_domain) as f32).clamp(0.0, 1.0);
        let ny_opt = ((y_star / max_domain) as f32).clamp(0.0, 1.0);
        let z_opt = self.tfp_a * x_star.powf(self.alpha) * y_star.powf(self.beta);
        let nz_opt = (((z_opt - z_min) / z_span) as f32).clamp(0.0, 1.0);
        let (opt_pos, _) = project(nx_opt, ny_opt, nz_opt);
        painter.circle_filled(opt_pos, 6.0, Color32::from_rgb(34, 197, 94));
        painter.circle_stroke(opt_pos, 6.0, Stroke::new(2.0_f32, Color32::WHITE));

        painter.text(
            rect.left_top() + Vec2::new(14.0, 12.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Surface Z(x,y) = {:.2} * x^{:.2} * y^{:.2} | Equilibrium ({:.2}, {:.2}, Z*={:.2}) — Drag to Rotate",
                self.tfp_a, self.alpha, self.beta, x_star, y_star, z_opt
            ),
            FontId::monospace(12.5),
            Color32::from_rgb(125, 211, 252),
        );
    }
}

impl eframe::App for EconCalcApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_menu_bar").show(ctx, |ui| {
            self.render_top_menu_bar(ui);
        });

        egui::TopBottomPanel::bottom("console_repl_pane")
            .resizable(true)
            .show(ctx, |ui| {
                self.render_bottom_console_pane(ui);
            });

        egui::SidePanel::left("editor_pane")
            .resizable(true)
            .show(ctx, |ui| {
                self.render_left_editor_pane(ui);
            });

        egui::SidePanel::right("environment_grid_pane")
            .resizable(true)
            .show(ctx, |ui| {
                self.render_right_environment_pane(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_central_viewport(ui);
        });
    }
}
