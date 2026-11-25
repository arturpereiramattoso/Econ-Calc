pub mod ast;
pub mod compiler;
pub mod graph;
pub mod lexer;
pub mod parser;
pub mod sampling;
pub mod solver;

use crate::collections::IndexMap;
pub use ast::{
    ident_to_latex, BinOp, CmpOp, MathError, MathExpr, MathStatement, Span, SpannedExpr, UnaryOp,
};
pub use compiler::{
    eval_fn_val_and_deriv, extract_fn_arg_name, find_surplus_equilibrium_x, CompiledExpr,
    CompiledUserFn, EvalContext, EvalScratch, MathExternalEnv,
};
pub use graph::{parse_function_signature, topological_sort_functions, CycleError, ParsedFunction};
pub use parser::{
    expr_mentions_any, parse_expr, parse_expr_with_known_fns, parse_statement,
    parse_statement_with_known_fns,
};
pub use sampling::{
    sample_1d_branches, sample_explicit_y_inequality, sample_implicit_2d_inequality,
    sample_implicit_contours, sample_parametric_branches, sample_polar_branches,
};
pub use solver::{
    detect_curve_pois, eval_compiled_at_x, float_to_fraction, solve_compiled_roots_1d, PoiKind,
};
use std::collections::{HashMap, HashSet};

/// Classified plot kind for a Desmos workspace entry.
#[derive(Debug, Clone)]
pub enum PlottableEntry {
    Explicit1D {
        compiled: CompiledExpr,
    },
    /// 1-variable equation in `x` (e.g. `x^2 - 5x + 6 = 0`, `f(x) = g(x)`, `f'(x) = 0`)
    /// solved automatically for real roots `x*` in the active domain!
    Equation1D {
        diff_compiled: CompiledExpr,
        lhs_compiled: CompiledExpr,
        is_fn_intersection: bool,
    },
    ImplicitEquation {
        /// Compiled `lhs - rhs`
        diff_compiled: CompiledExpr,
    },
    ExplicitYInequality {
        op: CmpOp,
        rhs_compiled: CompiledExpr,
    },
    ImplicitInequality {
        op: CmpOp,
        diff_compiled: CompiledExpr,
    },
    Parametric {
        param: String,
        x_compiled: CompiledExpr,
        y_compiled: CompiledExpr,
    },
    Polar {
        param: String,
        r_compiled: CompiledExpr,
    },
    /// Native `tangent(f, x0)` operator: plots tangent line + tangency point & shows `y = mx + b`
    TangentLine {
        fn_name: String,
        x0_compiled: CompiledExpr,
        line_compiled: CompiledExpr,
    },
    /// Native `area(f, g, a, b)` / `surplus(f, g)` operator: shades region between `f` and `g` & shows integral value
    AreaBetween {
        f_name: String,
        g_name: String,
        a_compiled: Option<CompiledExpr>,
        b_compiled: Option<CompiledExpr>,
        area_compiled: CompiledExpr,
    },
    /// Constant scalar expression or point function evaluation (`f(5)`, `f(g(5))`, `f'(5)`, `2^10 + 45/9`, `\int_0^1 x^2 dx`)
    ScalarOnly {
        compiled: CompiledExpr,
        /// When expression is a 1-arg call like `f(5)` or `f'(5)`, stores compiled argument `x0`
        /// so the UI can offer a one-click `📍 Plot Point` toggle for `(x0, val)`.
        eval_point_x: Option<CompiledExpr>,
    },
}

/// Result of compiling all functions in a workspace, ready for 60 FPS evaluation & plotting.
#[derive(Debug, Clone, Default)]
pub struct CompiledWorkspace {
    pub functions: HashMap<String, CompiledUserFn>,
    pub known_fn_names: HashSet<String>,
    #[allow(dead_code)]
    pub topo_order: Vec<String>,
    pub cycle_error: Option<CycleError>,
}

impl CompiledWorkspace {
    /// Compiles all workspace functions `(sig, expr_body)` into bytecode + dependency order.
    pub fn build(workspace_functions: &IndexMap<String, String>) -> Self {
        let mut known_fn_names = HashSet::new();
        for sig in workspace_functions.keys() {
            if let Some((name, _)) = parse_function_signature(sig) {
                known_fn_names.insert(name);
            }
        }

        let mut parsed_fns = Vec::new();
        for (sig, body_str) in workspace_functions {
            if let Some((name, params)) = parse_function_signature(sig) {
                if let Ok(body) = parse_expr_with_known_fns(body_str, &known_fn_names) {
                    parsed_fns.push(ParsedFunction {
                        sig: sig.clone(),
                        name,
                        params,
                        body,
                    });
                }
            }
        }

        let (topo_order, cycle_error) = match topological_sort_functions(&parsed_fns) {
            Ok(order) => (order, None),
            Err(err) => (Vec::new(), Some(err)),
        };

        let mut functions = HashMap::with_capacity(parsed_fns.len());
        for pf in &parsed_fns {
            functions.insert(pf.name.clone(), CompiledUserFn::from_parsed(pf));
        }

        Self {
            functions,
            known_fn_names,
            topo_order,
            cycle_error,
        }
    }

    /// Parses and classifies a workspace row `(sig, expr_str)` into a `PlottableEntry`.
    pub fn classify_entry(&self, sig: &str, expr_str: &str) -> Result<PlottableEntry, MathError> {
        let trimmed = expr_str.trim();
        let trimmed_sig = sig.trim();

        // Check if `sig` itself is an explicit polar signature `r(theta)` or `r`
        if let Some((fn_name, params)) = parse_function_signature(trimmed_sig) {
            if fn_name == "r" && params.len() == 1 && matches!(params[0].as_str(), "theta" | "t") {
                let r_ast = parse_expr_with_known_fns(trimmed, &self.known_fn_names)?;
                return Ok(PlottableEntry::Polar {
                    param: params[0].clone(),
                    r_compiled: CompiledExpr::compile(&r_ast),
                });
            }
        }

        // Parse the expression body as a full statement so inequalities, implicit eqs,
        // polar, and parametric tuples work directly inside the formula input box!
        let stmt = parse_statement_with_known_fns(trimmed, Some(&self.known_fn_names))?;
        match stmt {
            MathStatement::Parametric {
                param,
                x_expr,
                y_expr,
                ..
            } => Ok(PlottableEntry::Parametric {
                param,
                x_compiled: CompiledExpr::compile(&x_expr),
                y_compiled: CompiledExpr::compile(&y_expr),
            }),
            MathStatement::Polar { param, r_expr } => Ok(PlottableEntry::Polar {
                param,
                r_compiled: CompiledExpr::compile(&r_expr),
            }),
            MathStatement::Equation { lhs, rhs } => {
                // `y = f(x)` -> Explicit1D when RHS does not depend on `y`
                if lhs.as_canonical_ident().as_deref() == Some("y")
                    && !expr_mentions_any(&rhs, &["y"])
                {
                    return Ok(PlottableEntry::Explicit1D {
                        compiled: CompiledExpr::compile(&rhs),
                    });
                }
                let span = lhs.span.merge(rhs.span);
                let diff = SpannedExpr::new(
                    MathExpr::Binary {
                        op: BinOp::Sub,
                        lhs: Box::new(lhs.clone()),
                        rhs: Box::new(rhs.clone()),
                    },
                    span,
                );
                // Check if 1-variable equation in `x` (no `y` present, `x` IS present):
                // e.g. `x^2 - 5x + 6 = 0`, `2x + 4 = 10`, `f(x) = g(x)`, `f'(x) = 0`
                let mentions_y = expr_mentions_any(&lhs, &["y"]) || expr_mentions_any(&rhs, &["y"]);
                let mentions_x = expr_mentions_any(&lhs, &["x", "L", "Q", "Y"])
                    || expr_mentions_any(&rhs, &["x", "L", "Q", "Y"]);
                if !mentions_y && mentions_x {
                    let rhs_is_zero =
                        matches!(rhs.expr, MathExpr::Number(n) if n.abs() < 1e-12_f64);
                    let lhs_is_zero =
                        matches!(lhs.expr, MathExpr::Number(n) if n.abs() < 1e-12_f64);
                    let is_fn_intersection = !rhs_is_zero
                        && !lhs_is_zero
                        && (matches!(lhs.expr, MathExpr::Call { .. })
                            || matches!(rhs.expr, MathExpr::Call { .. }));
                    return Ok(PlottableEntry::Equation1D {
                        diff_compiled: CompiledExpr::compile(&diff),
                        lhs_compiled: CompiledExpr::compile(&lhs),
                        is_fn_intersection,
                    });
                }
                Ok(PlottableEntry::ImplicitEquation {
                    diff_compiled: CompiledExpr::compile(&diff),
                })
            }
            MathStatement::Inequality { lhs, op, rhs } => {
                if lhs.as_canonical_ident().as_deref() == Some("y")
                    && !expr_mentions_any(&rhs, &["y"])
                {
                    return Ok(PlottableEntry::ExplicitYInequality {
                        op,
                        rhs_compiled: CompiledExpr::compile(&rhs),
                    });
                }
                if rhs.as_canonical_ident().as_deref() == Some("y")
                    && !expr_mentions_any(&lhs, &["y"])
                {
                    let flipped = match op {
                        CmpOp::Lt => CmpOp::Gt,
                        CmpOp::Le => CmpOp::Ge,
                        CmpOp::Gt => CmpOp::Lt,
                        CmpOp::Ge => CmpOp::Le,
                        other => other,
                    };
                    return Ok(PlottableEntry::ExplicitYInequality {
                        op: flipped,
                        rhs_compiled: CompiledExpr::compile(&lhs),
                    });
                }
                let span = lhs.span.merge(rhs.span);
                let diff = SpannedExpr::new(
                    MathExpr::Binary {
                        op: BinOp::Sub,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    span,
                );
                Ok(PlottableEntry::ImplicitInequality {
                    op,
                    diff_compiled: CompiledExpr::compile(&diff),
                })
            }
            MathStatement::FunctionDef { body, .. } => {
                if let Some(special) = classify_econ_special_expr(&body) {
                    return Ok(special);
                }
                Ok(PlottableEntry::Explicit1D {
                    compiled: CompiledExpr::compile(&body),
                })
            }
            MathStatement::ParamAssign { value, .. } => Ok(PlottableEntry::ScalarOnly {
                compiled: CompiledExpr::compile(&value),
                eval_point_x: None,
            }),
            MathStatement::Expr(expr) => {
                if let Some(special) = classify_econ_special_expr(&expr) {
                    return Ok(special);
                }
                if let MathExpr::Tuple(mut items) = expr.expr.clone() {
                    if items.len() == 2 {
                        let y_expr = items.remove(1);
                        let x_expr = items.remove(0);
                        let param = if expr_mentions_any(&x_expr, &["theta"])
                            || expr_mentions_any(&y_expr, &["theta"])
                        {
                            "theta".to_string()
                        } else {
                            "t".to_string()
                        };
                        return Ok(PlottableEntry::Parametric {
                            param,
                            x_compiled: CompiledExpr::compile(&x_expr),
                            y_compiled: CompiledExpr::compile(&y_expr),
                        });
                    }
                }
                // Check if `sig` is `r` (without parens) and expr mentions `theta`
                if trimmed_sig == "r" && expr_mentions_any(&expr, &["theta"]) {
                    return Ok(PlottableEntry::Polar {
                        param: "theta".to_string(),
                        r_compiled: CompiledExpr::compile(&expr),
                    });
                }
                let is_curve_sig = parse_function_signature(trimmed_sig).is_some()
                    || expr_mentions_any(&expr, &["x", "y", "L", "K", "Q", "Y"]);
                if is_curve_sig {
                    Ok(PlottableEntry::Explicit1D {
                        compiled: CompiledExpr::compile(&expr),
                    })
                } else {
                    // Check if constant single-arg function call `f(5)` or `f'(5)` for `📍 Plot Point`
                    let eval_point_x = match &expr.expr {
                        MathExpr::Call { name, args, .. }
                            if args.len() == 1
                                && (self.known_fn_names.contains(name)
                                    || parser::is_builtin_function(name)) =>
                        {
                            Some(CompiledExpr::compile(&args[0]))
                        }
                        _ => None,
                    };
                    Ok(PlottableEntry::ScalarOnly {
                        compiled: CompiledExpr::compile(&expr),
                        eval_point_x,
                    })
                }
            }
        }
    }
}

/// Detects native computational economics / calculus plotting expressions (`tangent`, `area`, `surplus`).
fn classify_econ_special_expr(expr: &SpannedExpr) -> Option<PlottableEntry> {
    if let MathExpr::Call {
        name,
        prime_order: 0,
        args,
    } = &expr.expr
    {
        if name == "tangent" && args.len() == 2 {
            if let Some(fn_name) = extract_fn_arg_name(&args[0]) {
                return Some(PlottableEntry::TangentLine {
                    fn_name,
                    x0_compiled: CompiledExpr::compile(&args[1]),
                    line_compiled: CompiledExpr::compile(expr),
                });
            }
        }
        if matches!(name.as_str(), "area" | "surplus") && (args.len() == 2 || args.len() == 4) {
            if let (Some(f_name), Some(g_name)) =
                (extract_fn_arg_name(&args[0]), extract_fn_arg_name(&args[1]))
            {
                let (a_compiled, b_compiled) = if args.len() == 4 {
                    (
                        Some(CompiledExpr::compile(&args[2])),
                        Some(CompiledExpr::compile(&args[3])),
                    )
                } else {
                    (None, None)
                };
                return Some(PlottableEntry::AreaBetween {
                    f_name,
                    g_name,
                    a_compiled,
                    b_compiled,
                    area_compiled: CompiledExpr::compile(expr),
                });
            }
        }
    }
    None
}

/// Converts a math expression or statement into canonical LaTeX directly from the parsed AST.
pub fn to_latex_from_ast(raw: &str) -> String {
    if let Ok(stmt) = parse_statement(raw) {
        return stmt.to_latex();
    }
    if let Ok(expr) = parse_expr(raw) {
        return expr.to_latex();
    }
    // Graceful fallback for incomplete/partial text
    ident_to_latex(raw.trim())
}

/// Parses LaTeX or math syntax and emits clean canonical EconCalc symbolic notation
/// (for two-way LaTeX <-> Symbolic synchronization).
pub fn latex_to_symbolic_from_ast(
    raw: &str,
    known_fns: &HashSet<String>,
) -> Result<String, MathError> {
    let stmt = parse_statement_with_known_fns(raw.trim(), Some(known_fns))?;
    Ok(stmt.to_symbolic())
}

/// Substitutes parameter values into an AST-formatted expression for TikZ/pgfplots export.
pub fn format_for_tikz(raw: &str, params: &IndexMap<String, f64>) -> String {
    fn emit_tikz(expr: &MathExpr, params: &IndexMap<String, f64>) -> String {
        match expr {
            MathExpr::Number(n) => format!("{:.6}", n)
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
            MathExpr::Ident(name) => {
                if let Some(&v) = params.get(name) {
                    format!("({:.4})", v)
                } else {
                    name.clone()
                }
            }
            MathExpr::Accent { .. } | MathExpr::Subscript { .. } => {
                let dummy = SpannedExpr::new(expr.clone(), Span::default());
                if let Some(canon) = dummy.as_canonical_ident() {
                    if let Some(&v) = params.get(&canon) {
                        return format!("({:.4})", v);
                    }
                    return canon;
                }
                "0".to_string()
            }
            MathExpr::Group(inner) => format!("({})", emit_tikz(&inner.expr, params)),
            MathExpr::Abs(inner) => format!("abs({})", emit_tikz(&inner.expr, params)),
            MathExpr::Unary { op, expr } => match op {
                UnaryOp::Neg => format!("(-({}))", emit_tikz(&expr.expr, params)),
                UnaryOp::Pos => emit_tikz(&expr.expr, params),
            },
            MathExpr::Binary { op, lhs, rhs } => {
                let l = emit_tikz(&lhs.expr, params);
                let r = emit_tikz(&rhs.expr, params);
                match op {
                    BinOp::Add => format!("({} + {})", l, r),
                    BinOp::Sub => format!("({} - {})", l, r),
                    BinOp::Mul | BinOp::ImplicitMul => format!("({} * {})", l, r),
                    BinOp::Pow => format!("(({})^({}))", l, r),
                    BinOp::Mod => format!("mod({}, {})", l, r),
                }
            }
            MathExpr::Fraction { num, den } => {
                format!(
                    "(({}) / ({}))",
                    emit_tikz(&num.expr, params),
                    emit_tikz(&den.expr, params)
                )
            }
            MathExpr::Sqrt { index, radicand } => {
                let r = emit_tikz(&radicand.expr, params);
                if let Some(idx) = index {
                    format!("(({})^(1/({})))", r, emit_tikz(&idx.expr, params))
                } else {
                    format!("sqrt({})", r)
                }
            }
            MathExpr::Call { name, args, .. } => {
                let a: Vec<String> = args.iter().map(|e| emit_tikz(&e.expr, params)).collect();
                format!("{}({})", name, a.join(", "))
            }
            MathExpr::Restricted { inner, .. } => emit_tikz(&inner.expr, params),
            _ => "0".to_string(),
        }
    }

    match parse_expr(raw) {
        Ok(ast) => emit_tikz(&ast.expr, params),
        Err(_) => raw.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_simple(expr: &str, x: f64, y: f64, params: &[(&str, f64)]) -> f64 {
        let mut pmap = IndexMap::new();
        for &(k, v) in params {
            pmap.insert(k.to_string(), v);
        }
        let fns = HashMap::new();
        let ctx = EvalContext::new(&pmap, &fns);
        let ast = parse_expr(expr).unwrap_or_else(|e| panic!("parse `{}` failed: {}", expr, e));
        let compiled = CompiledExpr::compile(&ast);
        let mut scratch = EvalScratch::new();
        compiled
            .eval(&ctx, &[("x", x), ("y", y)], &mut scratch)
            .unwrap_or_else(|e| panic!("eval `{}` failed: {}", expr, e))
    }

    #[test]
    fn test_precedence_and_unary_power() {
        assert!((eval_simple("2 + 3 * 4", 0.0, 0.0, &[]) - 14.0).abs() < 1e-12);
        assert!((eval_simple("-x^2", 3.0, 0.0, &[]) - (-9.0)).abs() < 1e-12);
        assert!((eval_simple("2^3^2", 0.0, 0.0, &[]) - 512.0).abs() < 1e-12);
        assert!((eval_simple("e^-x", 1.0, 0.0, &[]) - (-1.0_f64).exp()).abs() < 1e-12);
        assert!((eval_simple("2x^2", 3.0, 0.0, &[]) - 18.0).abs() < 1e-12);
    }

    #[test]
    fn test_implicit_multiplication_and_latex() {
        assert!((eval_simple("2x", 3.0, 0.0, &[]) - 6.0).abs() < 1e-12);
        assert!((eval_simple("3(x + 1)", 3.0, 0.0, &[]) - 12.0).abs() < 1e-12);
        assert!(
            (eval_simple(
                "(a + b)(c + d)",
                0.0,
                0.0,
                &[("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]
            ) - 21.0)
                .abs()
                < 1e-12
        );
        assert!(
            (eval_simple(r"x \sin(y)", 2.0, std::f64::consts::FRAC_PI_2, &[]) - 2.0).abs() < 1e-12
        );
        assert!((eval_simple("x(x + 1)", 3.0, 0.0, &[]) - 12.0).abs() < 1e-12);
        assert!((eval_simple("2e3x", 3.0, 0.0, &[]) - 6000.0).abs() < 1e-12);
        assert!(
            (eval_simple(r"\frac{a + b}{2}", 0.0, 0.0, &[("a", 4.0), ("b", 6.0)]) - 5.0).abs()
                < 1e-12
        );
        assert!((eval_simple(r"\sqrt[3]{27} + \sqrt{16}", 0.0, 0.0, &[]) - 7.0).abs() < 1e-12);
        assert!(
            (eval_simple(
                r"\alpha \cdot x^{\beta}",
                4.0,
                0.0,
                &[("alpha", 2.0), ("beta", 0.5)]
            ) - 4.0)
                .abs()
                < 1e-12
        );
        assert!(
            (eval_simple(r"\bar{Y} / P_x", 0.0, 0.0, &[("Y_bar", 20.0), ("P_x", 4.0)]) - 5.0).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_chained_comparisons_and_piecewise() {
        assert_eq!(eval_simple("0 <= x <= 10", 5.0, 0.0, &[]), 1.0);
        assert_eq!(eval_simple("0 <= x <= 10", 12.0, 0.0, &[]), 0.0);
        assert_eq!(eval_simple("0 < x < 10", 0.0, 0.0, &[]), 0.0);

        let pw = "{ x < 0: -x, x >= 0: x^2 }";
        assert!((eval_simple(pw, -3.0, 0.0, &[]) - 3.0).abs() < 1e-12);
        assert!((eval_simple(pw, 3.0, 0.0, &[]) - 9.0).abs() < 1e-12);

        let pw_fallback = "{ x < 0: -1, x < 2: 0, 1 }";
        assert_eq!(eval_simple(pw_fallback, -2.0, 0.0, &[]), -1.0);
        assert_eq!(eval_simple(pw_fallback, 1.0, 0.0, &[]), 0.0);
        assert_eq!(eval_simple(pw_fallback, 5.0, 0.0, &[]), 1.0);
    }

    #[test]
    fn test_calculus_and_aggregations() {
        // Numerical derivative `\frac{d}{dx}`
        let deriv = eval_simple(r"\frac{d}{dx}(x^3)", 2.0, 0.0, &[]);
        assert!((deriv - 12.0).abs() < 1e-5, "got {}", deriv);

        // Definite integral `\int_0^1 x^2 dx = 1/3`
        let integral = eval_simple(r"\int_0^1 x^2 dx", 0.0, 0.0, &[]);
        assert!((integral - (1.0 / 3.0)).abs() < 1e-6, "got {}", integral);

        // Summation `\sum_{i=1}^{5} i^2 = 55`
        let sum_val = eval_simple(r"\sum_{i=1}^{5} i^2", 0.0, 0.0, &[]);
        assert!((sum_val - 55.0).abs() < 1e-12);

        // Product `\prod_{i=1}^{5} i = 120`
        let prod_val = eval_simple(r"\prod_{i=1}^{5} i", 0.0, 0.0, &[]);
        assert!((prod_val - 120.0).abs() < 1e-12);
    }

    #[test]
    fn test_nary_functions_topo_sort_and_cycle_detection() {
        let mut ws_fns = IndexMap::new();
        ws_fns.insert("g(x, y)".to_string(), "x^2 + y^2".to_string());
        ws_fns.insert("f(x)".to_string(), "g(x, 3) + 1".to_string());
        ws_fns.insert("df(x)".to_string(), "f'(x)".to_string());

        let compiled_ws = CompiledWorkspace::build(&ws_fns);
        assert!(compiled_ws.cycle_error.is_none());
        let g_pos = compiled_ws
            .topo_order
            .iter()
            .position(|n| n == "g")
            .unwrap();
        let f_pos = compiled_ws
            .topo_order
            .iter()
            .position(|n| n == "f")
            .unwrap();
        assert!(g_pos < f_pos);

        let params = IndexMap::new();
        let ctx = EvalContext::new(&params, &compiled_ws.functions);
        let ast = parse_expr_with_known_fns("f(4)", &compiled_ws.known_fn_names).unwrap();
        let ce = CompiledExpr::compile(&ast);
        let mut scratch = EvalScratch::new();
        // f(4) = g(4, 3) + 1 = 16 + 9 + 1 = 26
        assert!((ce.eval(&ctx, &[], &mut scratch).unwrap() - 26.0).abs() < 1e-12);

        // f'(4) = d/dx(x^2 + 10) @ 4 = 8
        let ast_d = parse_expr_with_known_fns("f'(4)", &compiled_ws.known_fn_names).unwrap();
        let ce_d = CompiledExpr::compile(&ast_d);
        let dv = ce_d.eval(&ctx, &[], &mut scratch).unwrap();
        assert!((dv - 8.0).abs() < 1e-5, "got {}", dv);

        // Cycle detection
        let mut cyc_fns = IndexMap::new();
        cyc_fns.insert("a(x)".to_string(), "b(x) + 1".to_string());
        cyc_fns.insert("b(x)".to_string(), "a(x) - 1".to_string());
        let cyc_ws = CompiledWorkspace::build(&cyc_fns);
        assert!(cyc_ws.cycle_error.is_some());
    }

    #[test]
    fn test_unbound_symbol_discovery() {
        let ast = parse_expr(r"a * x^\alpha + \bar{Y} + \sum_{i=1}^{n} i").unwrap();
        let bound = HashSet::new();
        let known_fns = HashSet::new();
        let mut existing = HashSet::new();
        existing.insert("alpha".to_string());
        let r_objs = HashSet::new();

        let unbound = graph::discover_unbound_symbols(&ast, &bound, &known_fns, &existing, &r_objs);
        assert_eq!(unbound, vec!["a", "Y_bar", "n"]);
    }

    #[test]
    fn test_implicit_circle_marching_squares_and_polar_parametric() {
        let params = IndexMap::new();
        let fns = HashMap::new();
        let ctx = EvalContext::new(&params, &fns);

        // Circle x^2 + y^2 - 25 = 0
        let diff_ast = parse_expr("x^2 + y^2 - 25").unwrap();
        let diff_compiled = CompiledExpr::compile(&diff_ast);
        let contours = sample_implicit_contours(&diff_compiled, &ctx, -6.0, 6.0, -6.0, 6.0, 48, 48);
        assert!(!contours.is_empty());
        // Verify every extracted point lies close to radius 5
        for chain in &contours {
            for &[px, py] in chain {
                let r = (px * px + py * py).sqrt();
                assert!((r - 5.0).abs() < 0.25, "point ({}, {}) has r={}", px, py, r);
            }
        }

        // Parametric circle (cos(t), sin(t))
        let stmt = parse_statement("(cos(t), sin(t))").unwrap();
        let MathStatement::Parametric {
            param,
            x_expr,
            y_expr,
            ..
        } = stmt
        else {
            panic!("Expected parametric statement");
        };
        let branches = sample_parametric_branches(
            &CompiledExpr::compile(&x_expr),
            &CompiledExpr::compile(&y_expr),
            &param,
            &ctx,
            0.0,
            std::f64::consts::TAU,
            64,
        );
        assert_eq!(branches.len(), 1);
        for &[px, py] in &branches[0] {
            assert!(((px * px + py * py).sqrt() - 1.0).abs() < 1e-9);
        }

        // Polar cardioid r = 1 + cos(theta)
        let polar_stmt = parse_statement(r"r = 1 + \cos(\theta)").unwrap();
        let MathStatement::Polar { param, r_expr } = polar_stmt else {
            panic!("Expected polar statement");
        };
        let polar_branches = sample_polar_branches(
            &CompiledExpr::compile(&r_expr),
            &param,
            &ctx,
            0.0,
            std::f64::consts::TAU,
            64,
        );
        assert_eq!(polar_branches.len(), 1);
        // At theta = 0, r = 2 -> (2, 0)
        assert!((polar_branches[0][0][0] - 2.0).abs() < 1e-9);
        assert!(polar_branches[0][0][1].abs() < 1e-9);
    }

    #[test]
    fn test_piecewise_discontinuity_splitting_and_inequality() {
        let params = IndexMap::new();
        let fns = HashMap::new();
        let ctx = EvalContext::new(&params, &fns);

        let step_ast = parse_expr("{ x < 0: -4, 4 }").unwrap();
        let step_compiled = CompiledExpr::compile(&step_ast);
        let branches = sample_1d_branches(&step_compiled, &ctx, -5.0, 5.0, -6.0, 6.0, 200, 4.0);
        assert!(
            branches.len() >= 2,
            "expected step split, got {} branches",
            branches.len()
        );

        let ineq_ast = parse_expr("x^2").unwrap();
        let region = sample_explicit_y_inequality(
            &CompiledExpr::compile(&ineq_ast),
            CmpOp::Lt,
            &ctx,
            -3.0,
            3.0,
            -2.0,
            10.0,
            100,
            4.0,
        );
        assert!(region.strict);
        assert!(!region.shaded_polygons.is_empty());
    }

    #[test]
    fn test_span_accurate_diagnostics_and_ast_latex() {
        let err = parse_expr("2 + * 3").unwrap_err();
        assert_eq!(err.span.start, 4);
        assert_eq!(err.span.end, 5);

        let latex = to_latex_from_ast(r"\frac{I - price_x * x}{price_y}");
        assert_eq!(latex, r"\frac{I - P_x \cdot x}{P_y}");
    }

    #[test]
    fn test_structural_placeholder_parsing_for_live_typing() {
        let samples = [
            "/",
            "1/",
            "(x+1)/",
            "g(x,y) = ln(5x)/",
            "x^",
            "P_",
            "sqrt()",
            "nthroot()",
            r"\int_0^1",
            r"\sum_{i=1}",
            "d/dx",
            "f(x) = ",
        ];
        for s in samples {
            let st = parser::parse_statement_for_preview(s, None)
                .unwrap_or_else(|e| panic!("Lenient preview parse failed on `{}`: {}", s, e));
            assert!(
                st.has_placeholder(),
                "Expected placeholder slot in `{}` -> {:?}",
                s,
                st
            );
        }
    }

    #[test]
    fn test_inline_domain_and_range_restrictions() {
        assert!((eval_simple("x^2 {0 <= x <= 5}", 3.0, 0.0, &[]) - 9.0).abs() < 1e-12);
        assert!(eval_simple("x^2 {0 <= x <= 5}", -1.0, 0.0, &[]).is_nan());
        assert!(eval_simple("x^2 {0 <= x <= 5}", 6.0, 0.0, &[]).is_nan());

        // Range restriction `{x >= 0, y >= 0}` restricts output `y` directly
        assert!((eval_simple("10 - x {x >= 0, y >= 0}", 4.0, 0.0, &[]) - 6.0).abs() < 1e-12);
        assert!(eval_simple("10 - x {x >= 0, y >= 0}", 12.0, 0.0, &[]).is_nan());
        assert!(eval_simple("10 - x {x >= 0, y >= 0}", -2.0, 0.0, &[]).is_nan());
    }

    #[test]
    fn test_function_argument_eval_compositions_derivatives_and_1d_equations() {
        let mut fns_raw = IndexMap::new();
        fns_raw.insert("f(x)".to_string(), "x^2".to_string());
        fns_raw.insert("g(x)".to_string(), "x^3".to_string());
        fns_raw.insert("u(x, y)".to_string(), "x^0.5 * y^0.5".to_string());
        let cw = CompiledWorkspace::build(&fns_raw);
        let params = IndexMap::new();
        let ctx = EvalContext::new(&params, &cw.functions);
        let mut scratch = EvalScratch::new();

        // f(5) = 25, classified as ScalarOnly with plot-point x0 = 5
        let cls_f5 = cw.classify_entry("", "f(5)").unwrap();
        let PlottableEntry::ScalarOnly {
            compiled,
            eval_point_x,
        } = cls_f5
        else {
            panic!("Expected ScalarOnly for `f(5)`");
        };
        assert!((compiled.eval(&ctx, &[], &mut scratch).unwrap() - 25.0).abs() < 1e-9);
        assert!(eval_point_x.is_some());

        // Compositions & derivatives at a point: f(g(5)) = 15625, f'(5) = 10, f''(5) = 2, u(4, 9) = 6
        let eval_c = |expr: &str| -> f64 {
            let ast = parse_expr_with_known_fns(expr, &cw.known_fn_names).unwrap();
            let comp = CompiledExpr::compile(&ast);
            let mut s = EvalScratch::new();
            comp.eval(&ctx, &[], &mut s).unwrap()
        };
        assert!((eval_c("f(g(5))") - 15_625.0).abs() < 1e-6);
        assert!((eval_c("f'(5)") - 10.0).abs() < 1e-4);
        assert!((eval_c("f''(5)") - 2.0).abs() < 1e-3);
        assert!((eval_c("u(4, 9)") - 6.0).abs() < 1e-9);

        // 1-variable equation auto-solver: x^2 - 5x + 6 = 0 -> x = 2, 3
        let eq_cls = cw.classify_entry("", "x^2 - 5x + 6 = 0").unwrap();
        let PlottableEntry::Equation1D { diff_compiled, .. } = eq_cls else {
            panic!("Expected Equation1D for `x^2 - 5x + 6 = 0`");
        };
        let roots = solve_compiled_roots_1d(&diff_compiled, &ctx, -5.0, 10.0, 300);
        assert_eq!(roots.len(), 2);
        assert!((roots[0] - 2.0).abs() < 1e-6);
        assert!((roots[1] - 3.0).abs() < 1e-6);

        // Even-multiplicity tangent root: (x - 2)^2 = 0 -> x = 2
        let tang_eq = cw.classify_entry("", "(x - 2)^2 = 0").unwrap();
        let PlottableEntry::Equation1D { diff_compiled, .. } = tang_eq else {
            panic!("Expected Equation1D");
        };
        let tang_roots = solve_compiled_roots_1d(&diff_compiled, &ctx, -5.0, 10.0, 300);
        assert_eq!(tang_roots.len(), 1);
        assert!((tang_roots[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_native_economics_operators_tangent_elasticity_marginal_average_surplus() {
        let mut fns_raw = IndexMap::new();
        fns_raw.insert("f(x)".to_string(), "x^2".to_string());
        fns_raw.insert("q(x)".to_string(), "2 * x^0.45".to_string());
        fns_raw.insert("tc(x)".to_string(), "x^2 + 3 * x".to_string());
        fns_raw.insert("d(x)".to_string(), "20 - 2 * x".to_string());
        fns_raw.insert("s(x)".to_string(), "2 * x".to_string());
        let cw = CompiledWorkspace::build(&fns_raw);
        let params = IndexMap::new();
        let ctx = EvalContext::new(&params, &cw.functions);
        let mut scratch = EvalScratch::new();

        // tangent(f, 3) for f(x)=x^2 -> y = 6x - 9 => at x=4, y=15
        let tang = cw.classify_entry("", "tangent(f, 3)").unwrap();
        let PlottableEntry::TangentLine { line_compiled, .. } = tang else {
            panic!("Expected TangentLine");
        };
        let y_at_4 = line_compiled
            .eval(&ctx, &[("x", 4.0)], &mut scratch)
            .unwrap();
        assert!((y_at_4 - 15.0).abs() < 1e-4);

        // elasticity(q, 5) & \varepsilon(q, 5) for 2*x^0.45 -> 0.45
        let el_ast = parse_expr_with_known_fns("elasticity(q, 5)", &cw.known_fn_names).unwrap();
        let el_val = CompiledExpr::compile(&el_ast)
            .eval(&ctx, &[], &mut scratch)
            .unwrap();
        assert!((el_val - 0.45).abs() < 1e-4);

        let eps_ast = parse_expr_with_known_fns(r"\varepsilon(q, 5)", &cw.known_fn_names).unwrap();
        let eps_val = CompiledExpr::compile(&eps_ast)
            .eval(&ctx, &[], &mut scratch)
            .unwrap();
        assert!((eps_val - 0.45).abs() < 1e-4);

        // marginal(tc, 4) = 2(4)+3 = 11; average(tc, 4) = (16+12)/4 = 7
        let mc_ast = parse_expr_with_known_fns("marginal(tc, 4)", &cw.known_fn_names).unwrap();
        let mc_val = CompiledExpr::compile(&mc_ast)
            .eval(&ctx, &[], &mut scratch)
            .unwrap();
        assert!((mc_val - 11.0).abs() < 1e-4);

        let ac_ast = parse_expr_with_known_fns("average(tc, 4)", &cw.known_fn_names).unwrap();
        let ac_val = CompiledExpr::compile(&ac_ast)
            .eval(&ctx, &[], &mut scratch)
            .unwrap();
        assert!((ac_val - 7.0).abs() < 1e-9);

        // surplus(d, s) between d(x)=20-2x and s(x)=2x -> eq x*=5, ∫_0^5 (20 - 4x) dx = 50
        let surp = cw.classify_entry("", "surplus(d, s)").unwrap();
        let PlottableEntry::AreaBetween { area_compiled, .. } = surp else {
            panic!("Expected AreaBetween");
        };
        let surp_val = area_compiled.eval(&ctx, &[], &mut scratch).unwrap();
        assert!((surp_val - 50.0).abs() < 1e-3);
    }

    #[test]
    fn test_continued_fractions_poi_detection_and_two_way_latex() {
        assert_eq!(float_to_fraction(1.0 / 3.0, 10_000, 1e-8), Some((1, 3)));
        assert_eq!(float_to_fraction(-1.75, 10_000, 1e-8), Some((-7, 4)));
        assert_eq!(float_to_fraction(std::f64::consts::PI, 1_000, 1e-8), None);

        // POI detection on y = x^2 - 4 against y = 5
        let params = IndexMap::new();
        let fns = HashMap::new();
        let ctx = EvalContext::new(&params, &fns);
        let parabola = CompiledExpr::compile(&parse_expr("x^2 - 4").unwrap());
        let line5 = CompiledExpr::compile(&parse_expr("5").unwrap());
        let pois = detect_curve_pois("f", &parabola, &[("g", &line5)], &ctx, -5.0, 5.0, -6.0, 8.0);
        assert!(pois
            .iter()
            .any(|p| p.kind == PoiKind::Root && (p.x - 2.0).abs() < 1e-4));
        assert!(pois
            .iter()
            .any(|p| p.kind == PoiKind::Root && (p.x + 2.0).abs() < 1e-4));
        assert!(pois
            .iter()
            .any(|p| p.kind == PoiKind::LocalMin && p.x.abs() < 1e-3 && (p.y + 4.0).abs() < 1e-3));
        assert!(pois
            .iter()
            .any(|p| p.kind == PoiKind::Intersection && (p.x - 3.0).abs() < 1e-3));

        // Two-way LaTeX -> Symbolic round-trip
        let sym = latex_to_symbolic_from_ast(r"\frac{-b + \sqrt{b^2 - 4ac}}{2a}", &HashSet::new())
            .unwrap();
        assert!(sym.contains("sqrt("));
        assert!(sym.contains('/'));
    }

    #[test]
    fn test_security_depth_and_dos_guards() {
        let deep_parens = format!("{}1{}", "(".repeat(400), ")".repeat(400));
        assert!(parse_expr(&deep_parens).is_err());

        let deep_unary = format!("{}1", "-".repeat(400));
        assert!(parse_expr(&deep_unary).is_err());

        let mut fns_raw = IndexMap::new();
        fns_raw.insert("f(x)".to_string(), "tangent(f, x)".to_string());
        let cw = CompiledWorkspace::build(&fns_raw);
        let params = IndexMap::new();
        let ctx = EvalContext::new(&params, &cw.functions);
        let mut scratch = EvalScratch::new();
        let self_tan = parse_expr_with_known_fns("f(2)", &cw.known_fn_names).unwrap();
        assert!(CompiledExpr::compile(&self_tan)
            .eval(&ctx, &[], &mut scratch)
            .is_err());

        let high_deriv = parse_expr_with_known_fns("f'''''(2)", &cw.known_fn_names).unwrap();
        assert!(CompiledExpr::compile(&high_deriv)
            .eval(&ctx, &[], &mut scratch)
            .is_err());

        let huge_sum = parse_expr(r"\sum_{i=-1e19}^{1e19} 1").unwrap();
        let sum_val = CompiledExpr::compile(&huge_sum)
            .eval(&ctx, &[], &mut scratch)
            .unwrap();
        assert!((sum_val - 100_000.0).abs() < 1e-6);
    }
}
