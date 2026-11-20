use super::ast::RError;
use super::object::{RObject, format_r_num};
use crate::collections::IndexMap;
use nalgebra::{DMatrix, DVector};

/// Parsed linear model formula specification.
struct FormulaSpec {
    lhs: String,
    terms: Vec<String>,
    has_intercept: bool,
}

fn parse_formula_terms(
    lhs_opt: Option<&str>,
    rhs_str: &str,
    available_cols: &[String],
) -> Result<FormulaSpec, RError> {
    let lhs = lhs_opt
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| RError::new("`lm()` formula requires a response variable on the LHS (e.g. `y ~ x1 + x2`)"))?;

    let mut has_intercept = true;
    let mut terms: Vec<String> = Vec::new();

    // Tokenize top-level `+` and `-` terms in RHS
    let mut current = String::new();
    let mut sign = '+';
    let mut depth = 0usize;
    let mut raw_pieces: Vec<(char, String)> = Vec::new();

    for ch in rhs_str.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' => {
                depth = depth.saturating_sub(1);
                current.push(ch);
            }
            '+' | '-' if depth == 0 => {
                let t = current.trim().to_string();
                if !t.is_empty() {
                    raw_pieces.push((sign, t));
                }
                current.clear();
                sign = ch;
            }
            _ => current.push(ch),
        }
    }
    let tail = current.trim().to_string();
    if !tail.is_empty() {
        raw_pieces.push((sign, tail));
    }

    for (sgn, piece) in raw_pieces {
        if piece == "0" || (sgn == '-' && piece == "1") {
            has_intercept = false;
            continue;
        }
        if piece == "1" {
            has_intercept = true;
            continue;
        }
        if sgn == '-' {
            terms.retain(|t| t != &piece);
            continue;
        }
        if piece == "." {
            for col in available_cols {
                if col != &lhs && !terms.contains(col) {
                    terms.push(col.clone());
                }
            }
            continue;
        }
        // Expand `a * b` -> `a`, `b`, `a:b`
        if piece.contains('*') && !piece.starts_with("I(") {
            let factors: Vec<String> = piece
                .split('*')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            for f in &factors {
                if !terms.contains(f) {
                    terms.push(f.clone());
                }
            }
            if factors.len() >= 2 {
                let inter = factors.join(":");
                if !terms.contains(&inter) {
                    terms.push(inter);
                }
            }
            continue;
        }
        if !terms.contains(&piece) {
            terms.push(piece);
        }
    }

    Ok(FormulaSpec {
        lhs,
        terms,
        has_intercept,
    })
}

fn lookup_column(
    name: &str,
    data: Option<&IndexMap<String, RObject>>,
    env: &IndexMap<String, RObject>,
) -> Result<Vec<Option<f64>>, RError> {
    if let Some(df_cols) = data {
        if let Some(col) = df_cols.get(name) {
            return Ok(col.as_numeric_options());
        }
    }
    if let Some(obj) = env.get(name) {
        return Ok(obj.as_numeric_options());
    }
    Err(RError::new(format!("`lm()`: variable `{}` not found", name)))
}

fn evaluate_term_column(
    term: &str,
    data: Option<&IndexMap<String, RObject>>,
    env: &IndexMap<String, RObject>,
) -> Result<Vec<Option<f64>>, RError> {
    // Interaction term `a:b`
    if term.contains(':') && !term.starts_with("I(") {
        let parts: Vec<&str> = term.split(':').map(|s| s.trim()).collect();
        let mut acc = lookup_column(parts[0], data, env)?;
        for part in &parts[1..] {
            let col = lookup_column(part, data, env)?;
            let len = acc.len().min(col.len());
            acc.truncate(len);
            for i in 0..len {
                acc[i] = match (acc[i], col[i]) {
                    (Some(a), Some(b)) => Some(a * b),
                    _ => None,
                };
            }
        }
        return Ok(acc);
    }

    // Power / transform shorthand `I(x^2)` or `log(x)`
    if let Some(inner) = term.strip_prefix("I(").and_then(|s| s.strip_suffix(')')) {
        if let Some((base, exp_s)) = inner.split_once('^') {
            let col = lookup_column(base.trim(), data, env)?;
            let exp: f64 = exp_s
                .trim()
                .parse()
                .map_err(|_| RError::new(format!("Invalid exponent in `{}`", term)))?;
            return Ok(col
                .into_iter()
                .map(|opt| opt.map(|v| v.powf(exp)))
                .collect());
        }
    }
    if let Some(inner) = term.strip_prefix("log(").and_then(|s| s.strip_suffix(')')) {
        let col = lookup_column(inner.trim(), data, env)?;
        return Ok(col
            .into_iter()
            .map(|opt| opt.and_then(|v| (v > 0.0_f64).then_some(v.ln())))
            .collect());
    }

    lookup_column(term, data, env)
}

/// Fits an Ordinary Least Squares model via `nalgebra` QR (with SVD fallback).
pub fn fit_lm(
    lhs_opt: Option<&str>,
    rhs_str: &str,
    data_obj: Option<&RObject>,
    env: &IndexMap<String, RObject>,
) -> Result<RObject, RError> {
    let data_map: Option<&IndexMap<String, RObject>> = match data_obj {
        Some(RObject::DataFrame { columns, .. }) => Some(columns),
        Some(RObject::List(map)) => Some(map),
        Some(RObject::Null) | None => None,
        Some(_) => {
            return Err(RError::new(
                "`lm()`: `data` must be a data.frame or list",
            ));
        }
    };

    let available_cols: Vec<String> = data_map
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();

    let spec = parse_formula_terms(lhs_opt, rhs_str, &available_cols)?;
    let y_raw = lookup_column(&spec.lhs, data_map, env)?;
    let n_total = y_raw.len();
    if n_total == 0 {
        return Err(RError::new("`lm()`: response variable has 0 observations"));
    }

    let mut x_raw_cols: Vec<Vec<Option<f64>>> = Vec::with_capacity(spec.terms.len());
    for term in &spec.terms {
        let col = evaluate_term_column(term, data_map, env)?;
        if col.len() != n_total {
            return Err(RError::new(format!(
                "`lm()`: variable `{}` length ({}) does not match `{}` length ({})",
                term,
                col.len(),
                spec.lhs,
                n_total
            )));
        }
        x_raw_cols.push(col);
    }

    // Listwise deletion of rows with NA / non-finite values
    let mut valid_rows = Vec::with_capacity(n_total);
    for i in 0..n_total {
        let Some(yv) = y_raw[i] else {
            continue;
        };
        if !yv.is_finite() {
            continue;
        }
        let all_x_ok = x_raw_cols
            .iter()
            .all(|col| col[i].map(|v| v.is_finite()).unwrap_or(false));
        if all_x_ok {
            valid_rows.push(i);
        }
    }

    let n = valid_rows.len();
    let p = spec.terms.len() + if spec.has_intercept { 1 } else { 0 };
    if p == 0 {
        return Err(RError::new("`lm()`: model has 0 parameters to estimate"));
    }
    if n < p {
        return Err(RError::new(format!(
            "`lm()`: insufficient complete observations ({}) for {} parameters",
            n, p
        )));
    }

    let mut x_data = Vec::with_capacity(n * p);
    let mut y_data = Vec::with_capacity(n);

    // DMatrix::from_column_slice expects column-major order
    if spec.has_intercept {
        x_data.extend(std::iter::repeat_n(1.0_f64, n));
    }
    for col in &x_raw_cols {
        for &r in &valid_rows {
            x_data.push(col[r].unwrap_or(0.0_f64));
        }
    }
    for &r in &valid_rows {
        y_data.push(y_raw[r].unwrap_or(0.0_f64));
    }

    let x_mat = DMatrix::from_column_slice(n, p, &x_data);
    let y_vec = DVector::from_column_slice(&y_data);

    // Solve via QR decomposition first, fallback to SVD for rank-deficient designs
    let beta = solve_ols_qr_or_svd(&x_mat, &y_vec)?;

    let fitted_vec = &x_mat * &beta;
    let resid_vec = &y_vec - &fitted_vec;

    let rss: f64 = resid_vec.iter().map(|e| e * e).sum();
    let y_mean: f64 = y_vec.iter().sum::<f64>() / (n as f64);
    let tss: f64 = if spec.has_intercept {
        y_vec.iter().map(|y| (y - y_mean).powi(2)).sum()
    } else {
        y_vec.iter().map(|y| y * y).sum()
    };

    let df_resid = n.saturating_sub(p);
    let sigma = if df_resid > 0 {
        (rss / (df_resid as f64)).sqrt()
    } else {
        0.0_f64
    };
    let r_squared = if tss.abs() > 1e-15_f64 {
        (1.0_f64 - rss / tss).clamp(0.0_f64, 1.0_f64)
    } else {
        1.0_f64
    };
    let adj_r_squared = if df_resid > 0 && n > 1 {
        let df_tot = if spec.has_intercept { n - 1 } else { n };
        1.0_f64 - (1.0_f64 - r_squared) * ((df_tot as f64) / (df_resid as f64))
    } else {
        r_squared
    };

    // Standard errors via pseudo-inverse of (X^T X)
    let xtx = x_mat.transpose() * &x_mat;
    let xtx_inv = xtx
        .svd(true, true)
        .pseudo_inverse(1e-12_f64)
        .unwrap_or_else(|_| DMatrix::zeros(p, p));

    let mut std_errors = Vec::with_capacity(p);
    let mut t_values = Vec::with_capacity(p);
    for j in 0..p {
        let var_j = (sigma * sigma * xtx_inv[(j, j)]).max(0.0_f64);
        let se = var_j.sqrt();
        std_errors.push(Some(se));
        let tv = if se > 1e-15_f64 {
            Some(beta[j] / se)
        } else {
            None
        };
        t_values.push(tv);
    }

    let mut coef_names = Vec::with_capacity(p);
    if spec.has_intercept {
        coef_names.push(Some("(Intercept)".to_string()));
    }
    for t in &spec.terms {
        coef_names.push(Some(t.clone()));
    }

    let mut model = IndexMap::new();
    model.insert(
        "coefficients".to_string(),
        RObject::Numeric(beta.iter().map(|&v| Some(v)).collect()),
    );
    model.insert("coef.names".to_string(), RObject::Character(coef_names));
    model.insert(
        "residuals".to_string(),
        RObject::Numeric(resid_vec.iter().map(|&v| Some(v)).collect()),
    );
    model.insert(
        "fitted.values".to_string(),
        RObject::Numeric(fitted_vec.iter().map(|&v| Some(v)).collect()),
    );
    model.insert("r.squared".to_string(), RObject::scalar(r_squared));
    model.insert("adj.r.squared".to_string(), RObject::scalar(adj_r_squared));
    model.insert("sigma".to_string(), RObject::scalar(sigma));
    model.insert("df.residual".to_string(), RObject::scalar(df_resid as f64));
    model.insert("std.error".to_string(), RObject::Numeric(std_errors));
    model.insert("t.value".to_string(), RObject::Numeric(t_values));
    model.insert(
        "terms".to_string(),
        RObject::Character(spec.terms.iter().map(|t| Some(t.clone())).collect()),
    );
    model.insert(
        "intercept".to_string(),
        RObject::bool_scalar(spec.has_intercept),
    );
    model.insert("response".to_string(), RObject::char_scalar(spec.lhs.clone()));
    model.insert(
        "call".to_string(),
        RObject::char_scalar(format!("lm({} ~ {})", spec.lhs, rhs_str)),
    );
    model.insert("class".to_string(), RObject::char_scalar("lm"));

    Ok(RObject::List(model))
}

fn solve_ols_qr_or_svd(x_mat: &DMatrix<f64>, y_vec: &DVector<f64>) -> Result<DVector<f64>, RError> {
    let p = x_mat.ncols();
    let qr = x_mat.clone().qr();
    let r = qr.r();
    let mut full_rank = true;
    for j in 0..p {
        if r[(j, j)].abs() < 1e-11_f64 {
            full_rank = false;
            break;
        }
    }

    if full_rank {
        let q = qr.q();
        let qty = q.transpose() * y_vec;
        let r_top = r.rows(0, p).into_owned();
        let rhs = qty.rows(0, p).into_owned();
        if let Some(sol) = r_top.solve_upper_triangular(&rhs) {
            if sol.iter().all(|v| v.is_finite()) {
                return Ok(sol);
            }
        }
    }

    x_mat
        .clone()
        .svd(true, true)
        .solve(y_vec, 1e-12_f64)
        .map_err(|e| RError::new(format!("`lm()` SVD solver failed: {}", e)))
}

/// Evaluates `predict(fit, newdata)` on a fitted `lm` model list.
pub fn predict_lm(
    model: &RObject,
    newdata: Option<&RObject>,
    env: &IndexMap<String, RObject>,
) -> Result<RObject, RError> {
    let RObject::List(map) = model else {
        return Err(RError::new("`predict()` expects an `lm` model object"));
    };
    if !model.is_lm_model() {
        return Err(RError::new("`predict()` currently supports `lm` objects"));
    }

    let Some(new_obj) = newdata else {
        return Ok(map
            .get("fitted.values")
            .cloned()
            .unwrap_or(RObject::Null));
    };

    let coefs = map
        .get("coefficients")
        .map(|c| c.as_numeric_options())
        .unwrap_or_default();
    let has_intercept = map
        .get("intercept")
        .and_then(|o| o.is_truthy().ok())
        .unwrap_or(true);
    let terms: Vec<String> = match map.get("terms") {
        Some(RObject::Character(v)) => v.iter().filter_map(|s| s.clone()).collect(),
        _ => Vec::new(),
    };

    // Build predictor columns from `newdata` (supports DataFrame, List, or 1D Numeric vector for single-predictor models!)
    let mut pred_cols: Vec<Vec<Option<f64>>> = Vec::with_capacity(terms.len());
    match new_obj {
        RObject::DataFrame { columns, .. } | RObject::List(columns) => {
            for t in &terms {
                pred_cols.push(evaluate_term_column(t, Some(columns), env)?);
            }
        }
        RObject::Numeric(_) | RObject::Logical(_) => {
            if terms.len() == 1 {
                pred_cols.push(new_obj.as_numeric_options());
            } else {
                return Err(RError::new(format!(
                    "`predict()`: model has {} predictors; pass a `data.frame` as `newdata`",
                    terms.len()
                )));
            }
        }
        _ => {
            return Err(RError::new(
                "`predict()`: `newdata` must be a data.frame, list, or numeric vector",
            ));
        }
    }

    let n_rows = pred_cols.iter().map(|c| c.len()).max().unwrap_or(1);
    let mut out = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let mut y_hat = if has_intercept {
            match coefs.first().copied().flatten() {
                Some(b0) => b0,
                None => {
                    out.push(None);
                    continue;
                }
            }
        } else {
            0.0_f64
        };
        let coef_offset = usize::from(has_intercept);
        let mut valid = true;
        for (j, col) in pred_cols.iter().enumerate() {
            let x_val = if col.is_empty() {
                None
            } else {
                col[i % col.len()]
            };
            let b_val = coefs.get(coef_offset + j).copied().flatten();
            match (x_val, b_val) {
                (Some(x), Some(b)) => y_hat += b * x,
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        out.push(valid.then_some(y_hat));
    }

    Ok(RObject::Numeric(out))
}

/// Formats an R-style `summary(fit)` report for an `lm` object.
pub fn format_lm_summary(model: &RObject) -> Result<String, RError> {
    let RObject::List(map) = model else {
        return Err(RError::new("Not an `lm` object"));
    };
    let call_str = match map.get("call") {
        Some(RObject::Character(c)) => c
            .first()
            .and_then(|s| s.clone())
            .unwrap_or_else(|| "lm(...)".to_string()),
        _ => "lm(...)".to_string(),
    };
    let coef_names: Vec<String> = match map.get("coef.names") {
        Some(RObject::Character(v)) => v
            .iter()
            .map(|s| s.clone().unwrap_or_else(|| "?".to_string()))
            .collect(),
        _ => Vec::new(),
    };
    let coefs = map
        .get("coefficients")
        .map(|o| o.as_numeric_options())
        .unwrap_or_default();
    let ses = map
        .get("std.error")
        .map(|o| o.as_numeric_options())
        .unwrap_or_default();
    let tvals = map
        .get("t.value")
        .map(|o| o.as_numeric_options())
        .unwrap_or_default();
    let sigma = map
        .get("sigma")
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(f64::NAN);
    let df_resid = map
        .get("df.residual")
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(0.0_f64);
    let r2 = map
        .get("r.squared")
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(f64::NAN);
    let adj_r2 = map
        .get("adj.r.squared")
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(f64::NAN);

    let mut lines = vec![
        format!("Call: {}", call_str),
        "Coefficients:".to_string(),
        format!(
            "  {:<14} {:>11} {:>11} {:>10}",
            "Term", "Estimate", "Std. Error", "t value"
        ),
    ];

    for (i, coef_opt) in coefs.iter().enumerate() {
        let name = coef_names
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or("?");
        let est = coef_opt
            .map(format_r_num)
            .unwrap_or_else(|| "NA".to_string());
        let se = ses
            .get(i)
            .copied()
            .flatten()
            .map(format_r_num)
            .unwrap_or_else(|| "NA".to_string());
        let tv = tvals
            .get(i)
            .copied()
            .flatten()
            .map(format_r_num)
            .unwrap_or_else(|| "NA".to_string());
        lines.push(format!("  {:<14} {:>11} {:>11} {:>10}", name, est, se, tv));
    }

    lines.push(format!(
        "Residual standard error: {:.5} on {} df",
        sigma, df_resid as usize
    ));
    lines.push(format!(
        "Multiple R-squared: {:.5},  Adjusted R-squared: {:.5}",
        r2, adj_r2
    ));

    Ok(lines.join("\n"))
}
