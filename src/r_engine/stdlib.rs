use super::ast::RError;
use super::lm::{fit_lm, format_lm_summary, predict_lm};
use super::object::{RObject, format_r_num};
use crate::collections::IndexMap;

/// Evaluated function call argument `(optional_name, value)`.
#[derive(Debug, Clone)]
pub struct EvalArg {
    pub name: Option<String>,
    pub value: RObject,
}

fn find_arg<'a>(args: &'a [EvalArg], name: &str, pos_idx: usize) -> Option<&'a RObject> {
    if let Some(a) = args.iter().find(|a| a.name.as_deref() == Some(name)) {
        return Some(&a.value);
    }
    let mut positional = 0usize;
    for a in args {
        if a.name.is_none() {
            if positional == pos_idx {
                return Some(&a.value);
            }
            positional += 1;
        }
    }
    None
}

fn find_named<'a>(args: &'a [EvalArg], name: &str) -> Option<&'a RObject> {
    args.iter()
        .find(|a| a.name.as_deref() == Some(name))
        .map(|a| &a.value)
}

fn na_rm_flag(args: &[EvalArg]) -> bool {
    find_named(args, "na.rm")
        .and_then(|o| o.is_truthy().ok())
        .unwrap_or(false)
}

/// Dispatches built-in R standard library & econometrics functions.
/// Returns `Ok(Some(obj))` if `name` is a known built-in, or `Ok(None)` if not built-in.
pub fn call_builtin(
    name: &str,
    args: &[EvalArg],
    env: &IndexMap<String, RObject>,
    stdout: &mut Vec<String>,
) -> Result<Option<RObject>, RError> {
    let res = match name {
        // ---- Constructors ----
        "c" => Some(builtin_c(args)),
        "seq" => Some(builtin_seq(args)?),
        "rep" => Some(builtin_rep(args)?),
        "matrix" => Some(builtin_matrix(args)?),
        "data.frame" | "tibble" => Some(builtin_data_frame(args)?),
        "list" => Some(builtin_list(args)),
        "cbind" => Some(builtin_cbind(args)?),
        "rbind" => Some(builtin_rbind(args)?),

        // ---- Inspection & Coercion ----
        "length" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`length(x)` requires 1 argument"))?;
            Some(RObject::scalar(x.value.r_len() as f64))
        }
        "dim" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`dim(x)` requires 1 argument"))?;
            Some(match &x.value {
                RObject::Matrix { nrow, ncol, .. } => {
                    RObject::numeric_vec(vec![*nrow as f64, *ncol as f64])
                }
                RObject::DataFrame { columns, row_count } => {
                    RObject::numeric_vec(vec![*row_count as f64, columns.len() as f64])
                }
                _ => RObject::Null,
            })
        }
        "nrow" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`nrow(x)` requires 1 argument"))?;
            Some(match &x.value {
                RObject::Matrix { nrow, .. } => RObject::scalar(*nrow as f64),
                RObject::DataFrame { row_count, .. } => RObject::scalar(*row_count as f64),
                _ => RObject::Null,
            })
        }
        "ncol" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`ncol(x)` requires 1 argument"))?;
            Some(match &x.value {
                RObject::Matrix { ncol, .. } => RObject::scalar(*ncol as f64),
                RObject::DataFrame { columns, .. } => RObject::scalar(columns.len() as f64),
                _ => RObject::Null,
            })
        }
        "names" | "colnames" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new(format!("`{}(x)` requires 1 argument", name)))?;
            Some(match &x.value {
                RObject::DataFrame { columns, .. } => RObject::Character(
                    columns.keys().map(|k| Some(k.clone())).collect(),
                ),
                RObject::List(map) => RObject::Character(
                    map.keys().map(|k| Some(k.clone())).collect(),
                ),
                RObject::Matrix {
                    colnames: Some(c), ..
                } => RObject::Character(c.iter().map(|s| Some(s.clone())).collect()),
                RObject::Matrix { colnames: None, .. } => RObject::Null,
                _ => RObject::Null,
            })
        }
        "rownames" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`rownames(x)` requires 1 argument"))?;
            Some(match &x.value {
                RObject::Matrix {
                    rownames: Some(r), ..
                } => RObject::Character(r.iter().map(|s| Some(s.clone())).collect()),
                RObject::Matrix { rownames: None, .. } => RObject::Null,
                RObject::DataFrame { row_count, .. } => RObject::Character(
                    (1..=*row_count).map(|i| Some(i.to_string())).collect(),
                ),
                _ => RObject::Null,
            })
        }
        "head" => Some(builtin_head_tail(args, true)?),
        "tail" => Some(builtin_head_tail(args, false)?),
        "str" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`str(x)` requires 1 argument"))?;
            let text = x.value.format_str();
            stdout.push(text.clone());
            Some(RObject::char_scalar(text))
        }
        "summary" => Some(builtin_summary(args)?),
        "is.na" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`is.na(x)` requires 1 argument"))?;
            Some(match &x.value {
                RObject::Numeric(v) => {
                    RObject::Logical(v.iter().map(|o| Some(o.is_none() || o.map(|f| f.is_nan()).unwrap_or(false))).collect())
                }
                RObject::Logical(v) => {
                    RObject::Logical(v.iter().map(|o| Some(o.is_none())).collect())
                }
                RObject::Character(v) => {
                    RObject::Logical(v.iter().map(|o| Some(o.is_none())).collect())
                }
                RObject::Null => RObject::Logical(Vec::new()),
                _ => RObject::bool_scalar(false),
            })
        }
        "is.null" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`is.null(x)` requires 1 argument"))?;
            Some(RObject::bool_scalar(matches!(x.value, RObject::Null)))
        }
        "which" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`which(x)` requires 1 argument"))?;
            let mask = x.value.as_logical_options();
            let idxs: Vec<Option<f64>> = mask
                .iter()
                .enumerate()
                .filter_map(|(i, opt)| (*opt == Some(true)).then_some(Some((i + 1) as f64)))
                .collect();
            Some(RObject::Numeric(idxs))
        }
        "as.numeric" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`as.numeric(x)` requires 1 argument"))?;
            Some(RObject::Numeric(x.value.as_numeric_options()))
        }
        "as.character" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`as.character(x)` requires 1 argument"))?;
            Some(RObject::Character(x.value.as_character_options()))
        }
        "as.logical" => {
            let x = args
                .first()
                .ok_or_else(|| RError::new("`as.logical(x)` requires 1 argument"))?;
            Some(RObject::Logical(x.value.as_logical_options()))
        }
        "paste" | "paste0" => {
            let default_sep = if name == "paste0" { "" } else { " " };
            let sep = find_named(args, "sep")
                .and_then(|o| o.as_character_options().into_iter().next().flatten())
                .unwrap_or_else(|| default_sep.to_string());
            let parts: Vec<Vec<Option<String>>> = args
                .iter()
                .filter(|a| a.name.is_none())
                .map(|a| a.value.as_character_options())
                .collect();
            let max_len = parts.iter().map(|p| p.len()).max().unwrap_or(0);
            let mut out = Vec::with_capacity(max_len);
            for i in 0..max_len {
                let joined: Vec<String> = parts
                    .iter()
                    .map(|p| {
                        if p.is_empty() {
                            String::new()
                        } else {
                            p[i % p.len()].clone().unwrap_or_else(|| "NA".to_string())
                        }
                    })
                    .collect();
                out.push(Some(joined.join(&sep)));
            }
            Some(RObject::Character(out))
        }
        "print" | "cat" => {
            if let Some(first) = args.first() {
                stdout.push(first.value.preview_string());
                Some(first.value.clone())
            } else {
                Some(RObject::Null)
            }
        }

        // ---- Vectorized Unary Math ----
        "sin" => Some(map_numeric_unary(args, "sin", f64::sin)?),
        "cos" => Some(map_numeric_unary(args, "cos", f64::cos)?),
        "tan" => Some(map_numeric_unary(args, "tan", f64::tan)?),
        "asin" => Some(map_numeric_unary(args, "asin", f64::asin)?),
        "acos" => Some(map_numeric_unary(args, "acos", f64::acos)?),
        "atan" => Some(map_numeric_unary(args, "atan", f64::atan)?),
        "exp" => Some(map_numeric_unary(args, "exp", f64::exp)?),
        "log" => Some(builtin_log(args)?),
        "log10" => Some(map_numeric_unary(args, "log10", f64::log10)?),
        "log2" => Some(map_numeric_unary(args, "log2", f64::log2)?),
        "sqrt" => Some(map_numeric_unary(args, "sqrt", f64::sqrt)?),
        "abs" => Some(map_numeric_unary(args, "abs", f64::abs)?),
        "floor" => Some(map_numeric_unary(args, "floor", f64::floor)?),
        "ceiling" | "ceil" => Some(map_numeric_unary(args, name, f64::ceil)?),
        "sign" => Some(map_numeric_unary(args, "sign", |v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })?),
        "round" => Some(builtin_round(args)?),
        "ifelse" => Some(builtin_ifelse(args)?),

        // ---- Stats & Aggregations ----
        "sum" => Some(builtin_sum_prod(args, true)?),
        "prod" => Some(builtin_sum_prod(args, false)?),
        "mean" => Some(builtin_mean(args)?),
        "median" => Some(builtin_median(args)?),
        "var" => Some(builtin_var_sd(args, false)?),
        "sd" => Some(builtin_var_sd(args, true)?),
        "min" => Some(builtin_min_max(args, true)?),
        "max" => Some(builtin_min_max(args, false)?),
        "quantile" => Some(builtin_quantile(args)?),
        "cumsum" => Some(builtin_cum(args, true)?),
        "cumprod" => Some(builtin_cum(args, false)?),
        "diff" => Some(builtin_diff(args)?),
        "sort" => Some(builtin_sort(args)?),
        "rev" => Some(builtin_rev(args)?),
        "cor" => Some(builtin_cov_cor(args, true)?),
        "cov" => Some(builtin_cov_cor(args, false)?),

        // ---- Econometrics & Linear Models ----
        "lm" => {
            let formula = find_arg(args, "formula", 0)
                .ok_or_else(|| RError::new("`lm()` requires a formula (e.g. `lm(y ~ x1 + x2, data = df)`)"))?;
            let data = find_arg(args, "data", 1);
            match formula {
                RObject::Formula { lhs, rhs } => Some(fit_lm(lhs.as_deref(), rhs, data, env)?),
                _ => {
                    return Err(RError::new(
                        "`lm()` first argument must be a formula `y ~ x1 + x2`",
                    ));
                }
            }
        }
        "coef" | "coefficients" => {
            let fit = args
                .first()
                .ok_or_else(|| RError::new("`coef(fit)` requires a model argument"))?;
            if let RObject::List(map) = &fit.value {
                if let Some(c) = map.get("coefficients") {
                    return Ok(Some(c.clone()));
                }
            }
            return Err(RError::new("`coef()` expects a fitted `lm` model"));
        }
        "residuals" | "resid" => {
            let fit = args
                .first()
                .ok_or_else(|| RError::new("`residuals(fit)` requires a model argument"))?;
            if let RObject::List(map) = &fit.value {
                if let Some(r) = map.get("residuals") {
                    return Ok(Some(r.clone()));
                }
            }
            return Err(RError::new("`residuals()` expects a fitted `lm` model"));
        }
        "fitted" | "fitted.values" => {
            let fit = args
                .first()
                .ok_or_else(|| RError::new("`fitted(fit)` requires a model argument"))?;
            if let RObject::List(map) = &fit.value {
                if let Some(f) = map.get("fitted.values") {
                    return Ok(Some(f.clone()));
                }
            }
            return Err(RError::new("`fitted()` expects a fitted `lm` model"));
        }
        "predict" => {
            let fit = find_arg(args, "object", 0)
                .ok_or_else(|| RError::new("`predict(fit, [newdata])` requires a model"))?;
            let newdata = find_arg(args, "newdata", 1);
            Some(predict_lm(fit, newdata, env)?)
        }

        _ => None,
    };

    Ok(res)
}

fn builtin_c(args: &[EvalArg]) -> RObject {
    let mut has_char = false;
    let mut has_num = false;
    let mut has_log = false;

    for a in args {
        match &a.value {
            RObject::Character(_) => has_char = true,
            RObject::Numeric(_) | RObject::Matrix { .. } => has_num = true,
            RObject::Logical(_) => has_log = true,
            RObject::Null => {}
            _ => has_char = true,
        }
    }

    if has_char {
        let mut out = Vec::new();
        for a in args {
            if !matches!(a.value, RObject::Null) {
                out.extend(a.value.as_character_options());
            }
        }
        RObject::Character(out)
    } else if has_num {
        let mut out = Vec::new();
        for a in args {
            if !matches!(a.value, RObject::Null) {
                out.extend(a.value.as_numeric_options());
            }
        }
        RObject::Numeric(out)
    } else if has_log {
        let mut out = Vec::new();
        for a in args {
            if !matches!(a.value, RObject::Null) {
                out.extend(a.value.as_logical_options());
            }
        }
        RObject::Logical(out)
    } else {
        RObject::Null
    }
}

fn builtin_seq(args: &[EvalArg]) -> Result<RObject, RError> {
    if args.len() == 1 && args[0].name.is_none() {
        let n = args[0]
            .value
            .as_scalar_f64()
            .ok_or_else(|| RError::new("`seq(n)` requires a finite numeric scalar"))?;
        let end = n.floor().clamp(0.0_f64, 100_000.0_f64) as usize;
        return Ok(RObject::Numeric((1..=end).map(|i| Some(i as f64)).collect()));
    }

    let from = find_arg(args, "from", 0)
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(1.0_f64);
    let to = find_arg(args, "to", 1)
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(1.0_f64);
    let length_out = find_named(args, "length.out")
        .or_else(|| find_named(args, "len"))
        .and_then(|o| o.as_scalar_f64());

    if let Some(len_f) = length_out {
        let n = len_f.round().clamp(0.0_f64, 100_000.0_f64) as usize;
        if n == 0 {
            return Ok(RObject::Numeric(Vec::new()));
        }
        if n == 1 {
            return Ok(RObject::Numeric(vec![Some(from)]));
        }
        let step = (to - from) / ((n - 1) as f64);
        let out = (0..n).map(|i| Some(from + step * (i as f64))).collect();
        return Ok(RObject::Numeric(out));
    }

    let by = find_arg(args, "by", 2)
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or_else(|| if to >= from { 1.0_f64 } else { -1.0_f64 });

    if by.abs() < 1e-15_f64 {
        return Err(RError::new("`seq()` step `by` cannot be 0"));
    }
    if (to > from && by < 0.0_f64) || (to < from && by > 0.0_f64) {
        return Err(RError::new("wrong sign in `by` argument to `seq()`"));
    }

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < 100_000 {
        let v = from + (i as f64) * by;
        if (by > 0.0_f64 && v > to + 1e-11_f64) || (by < 0.0_f64 && v < to - 1e-11_f64) {
            break;
        }
        out.push(Some(v));
        i += 1;
    }
    Ok(RObject::Numeric(out))
}

fn builtin_rep(args: &[EvalArg]) -> Result<RObject, RError> {
    let x = find_arg(args, "x", 0)
        .ok_or_else(|| RError::new("`rep()` requires at least 1 argument"))?;
    let times = find_arg(args, "times", 1)
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(1.0_f64)
        .clamp(0.0_f64, 50_000.0_f64) as usize;
    let each = find_named(args, "each")
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(1.0_f64)
        .clamp(0.0_f64, 50_000.0_f64) as usize;

    fn repeat_vec<T: Clone>(src: &[T], times: usize, each: usize) -> Vec<T> {
        let mut expanded = Vec::with_capacity((src.len() * each).min(100_000));
        for item in src {
            for _ in 0..each {
                if expanded.len() >= 100_000 {
                    break;
                }
                expanded.push(item.clone());
            }
        }
        let mut out = Vec::with_capacity((expanded.len() * times).min(100_000));
        for _ in 0..times {
            for item in &expanded {
                if out.len() >= 100_000 {
                    break;
                }
                out.push(item.clone());
            }
        }
        out
    }

    Ok(match x {
        RObject::Numeric(v) => RObject::Numeric(repeat_vec(v, times, each)),
        RObject::Logical(v) => RObject::Logical(repeat_vec(v, times, each)),
        RObject::Character(v) => RObject::Character(repeat_vec(v, times, each)),
        _ => RObject::Numeric(repeat_vec(&x.as_numeric_options(), times, each)),
    })
}

fn builtin_matrix(args: &[EvalArg]) -> Result<RObject, RError> {
    let data = find_arg(args, "data", 0)
        .map(|o| o.as_numeric_options())
        .unwrap_or_else(|| vec![None]);
    let byrow = find_arg(args, "byrow", 3)
        .and_then(|o| o.is_truthy().ok())
        .unwrap_or(false);

    let nrow_opt = find_arg(args, "nrow", 1)
        .and_then(|o| o.as_scalar_f64())
        .map(|v| v.max(1.0_f64) as usize);
    let ncol_opt = find_arg(args, "ncol", 2)
        .and_then(|o| o.as_scalar_f64())
        .map(|v| v.max(1.0_f64) as usize);

    let (nrow, ncol) = match (nrow_opt, ncol_opt) {
        (Some(r), Some(c)) => (r, c),
        (Some(r), None) => (r, data.len().div_ceil(r).max(1)),
        (None, Some(c)) => (data.len().div_ceil(c).max(1), c),
        (None, None) => (data.len().max(1), 1),
    };

    let total = (nrow * ncol).min(250_000);
    let mut out = vec![None; total];
    if !data.is_empty() {
        if byrow {
            for r in 0..nrow {
                for c in 0..ncol {
                    let src_idx = (r * ncol + c) % data.len();
                    let dst_idx = c * nrow + r;
                    if dst_idx < out.len() {
                        out[dst_idx] = data[src_idx];
                    }
                }
            }
        } else {
            for (i, cell) in out.iter_mut().enumerate() {
                *cell = data[i % data.len()];
            }
        }
    }

    Ok(RObject::Matrix {
        data: out,
        nrow,
        ncol,
        rownames: None,
        colnames: None,
    })
}

fn builtin_data_frame(args: &[EvalArg]) -> Result<RObject, RError> {
    if args.is_empty() {
        return Ok(RObject::DataFrame {
            columns: IndexMap::new(),
            row_count: 0,
        });
    }

    let mut max_rows = 0usize;
    for a in args {
        max_rows = max_rows.max(a.value.r_len());
    }

    let mut columns = IndexMap::new();
    for (idx, a) in args.iter().enumerate() {
        let col_name = a
            .name
            .clone()
            .unwrap_or_else(|| format!("V{}", idx + 1));
        let col_len = a.value.r_len();
        if col_len != max_rows && col_len != 1 && max_rows > 0 {
            return Err(RError::new(format!(
                "`data.frame()`: column `{}` has length {}, expected {} or 1",
                col_name, col_len, max_rows
            )));
        }

        let col_val = if col_len == 1 && max_rows > 1 {
            match &a.value {
                RObject::Numeric(v) => RObject::Numeric(vec![v[0]; max_rows]),
                RObject::Logical(v) => RObject::Logical(vec![v[0]; max_rows]),
                RObject::Character(v) => RObject::Character(vec![v[0].clone(); max_rows]),
                other => other.clone(),
            }
        } else {
            a.value.clone()
        };
        columns.insert(col_name, col_val);
    }

    Ok(RObject::DataFrame {
        columns,
        row_count: max_rows,
    })
}

fn builtin_list(args: &[EvalArg]) -> RObject {
    let mut map = IndexMap::new();
    for (idx, a) in args.iter().enumerate() {
        let key = a
            .name
            .clone()
            .unwrap_or_else(|| format!("[[{}]]", idx + 1));
        map.insert(key, a.value.clone());
    }
    RObject::List(map)
}

fn builtin_cbind(args: &[EvalArg]) -> Result<RObject, RError> {
    if args.is_empty() {
        return Ok(RObject::Null);
    }
    if args
        .iter()
        .any(|a| matches!(a.value, RObject::DataFrame { .. }))
    {
        let mut columns = IndexMap::new();
        let mut row_count = 0usize;
        for (i, a) in args.iter().enumerate() {
            match &a.value {
                RObject::DataFrame {
                    columns: cols,
                    row_count: rc,
                } => {
                    row_count = row_count.max(*rc);
                    for (k, v) in cols {
                        columns.insert(k.clone(), v.clone());
                    }
                }
                other => {
                    row_count = row_count.max(other.r_len());
                    let key = a.name.clone().unwrap_or_else(|| format!("V{}", i + 1));
                    columns.insert(key, other.clone());
                }
            }
        }
        return Ok(RObject::DataFrame { columns, row_count });
    }

    let mut cols: Vec<Vec<Option<f64>>> = Vec::new();
    let mut nrow = 0usize;
    for a in args {
        match &a.value {
            RObject::Matrix {
                data,
                nrow: r,
                ncol: c,
                ..
            } => {
                nrow = nrow.max(*r);
                for j in 0..*c {
                    cols.push(data[j * r..(j + 1) * r].to_vec());
                }
            }
            other => {
                let v = other.as_numeric_options();
                nrow = nrow.max(v.len());
                cols.push(v);
            }
        }
    }
    let ncol = cols.len();
    let mut data = Vec::with_capacity(nrow * ncol);
    for col in cols {
        for r in 0..nrow {
            data.push(if col.is_empty() {
                None
            } else {
                col[r % col.len()]
            });
        }
    }
    Ok(RObject::Matrix {
        data,
        nrow,
        ncol,
        rownames: None,
        colnames: None,
    })
}

fn builtin_rbind(args: &[EvalArg]) -> Result<RObject, RError> {
    if args.is_empty() {
        return Ok(RObject::Null);
    }
    if let RObject::DataFrame { .. } = &args[0].value {
        let mut out_cols: IndexMap<String, Vec<Option<f64>>> = IndexMap::new();
        let mut col_names: Vec<String> = Vec::new();
        let mut total_rows = 0usize;
        for a in args {
            if let RObject::DataFrame { columns, row_count } = &a.value {
                if col_names.is_empty() {
                    col_names = columns.keys().cloned().collect();
                    for k in &col_names {
                        out_cols.insert(k.clone(), Vec::new());
                    }
                }
                for k in &col_names {
                    let vals = columns
                        .get(k)
                        .map(|c| c.as_numeric_options())
                        .unwrap_or_else(|| vec![None; *row_count]);
                    if let Some(acc) = out_cols.get_mut(k) {
                        acc.extend(vals);
                    }
                }
                total_rows += row_count;
            }
        }
        let mut df_cols = IndexMap::new();
        for (k, v) in out_cols.iter() {
            df_cols.insert(k.clone(), RObject::Numeric(v.clone()));
        }
        return Ok(RObject::DataFrame {
            columns: df_cols,
            row_count: total_rows,
        });
    }

    let mut rows: Vec<Vec<Option<f64>>> = Vec::new();
    let mut ncol = 0usize;
    for a in args {
        match &a.value {
            RObject::Matrix {
                data,
                nrow: r,
                ncol: c,
                ..
            } => {
                ncol = ncol.max(*c);
                for ri in 0..*r {
                    let row: Vec<Option<f64>> = (0..*c).map(|ci| data[ci * r + ri]).collect();
                    rows.push(row);
                }
            }
            other => {
                let v = other.as_numeric_options();
                ncol = ncol.max(v.len());
                rows.push(v);
            }
        }
    }
    let nrow = rows.len();
    let mut data = vec![None; nrow * ncol];
    for (r, row) in rows.iter().enumerate() {
        for c in 0..ncol {
            data[c * nrow + r] = if row.is_empty() {
                None
            } else {
                row[c % row.len()]
            };
        }
    }
    Ok(RObject::Matrix {
        data,
        nrow,
        ncol,
        rownames: None,
        colnames: None,
    })
}

fn builtin_head_tail(args: &[EvalArg], is_head: bool) -> Result<RObject, RError> {
    let x = find_arg(args, "x", 0)
        .ok_or_else(|| RError::new("`head`/`tail` requires an object argument"))?;
    let n = find_arg(args, "n", 1)
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(6.0_f64)
        .max(0.0_f64) as usize;

    fn slice_vec<T: Clone>(v: &[T], n: usize, is_head: bool) -> Vec<T> {
        let take = n.min(v.len());
        if is_head {
            v[..take].to_vec()
        } else {
            v[v.len() - take..].to_vec()
        }
    }

    Ok(match x {
        RObject::Numeric(v) => RObject::Numeric(slice_vec(v, n, is_head)),
        RObject::Logical(v) => RObject::Logical(slice_vec(v, n, is_head)),
        RObject::Character(v) => RObject::Character(slice_vec(v, n, is_head)),
        RObject::DataFrame { columns, row_count } => {
            let take = n.min(*row_count);
            let mut new_cols = IndexMap::new();
            for (k, col) in columns {
                let sliced = match col {
                    RObject::Numeric(v) => RObject::Numeric(slice_vec(v, take, is_head)),
                    RObject::Logical(v) => RObject::Logical(slice_vec(v, take, is_head)),
                    RObject::Character(v) => RObject::Character(slice_vec(v, take, is_head)),
                    other => other.clone(),
                };
                new_cols.insert(k.clone(), sliced);
            }
            RObject::DataFrame {
                columns: new_cols,
                row_count: take,
            }
        }
        other => other.clone(),
    })
}

fn builtin_summary(args: &[EvalArg]) -> Result<RObject, RError> {
    let x = args
        .first()
        .ok_or_else(|| RError::new("`summary(x)` requires 1 argument"))?;
    if x.value.is_lm_model() {
        return Ok(RObject::char_scalar(format_lm_summary(&x.value)?));
    }
    match &x.value {
        RObject::DataFrame { columns, row_count } => {
            let mut lines = vec![format!(
                "DataFrame Summary ({} rows, {} cols):",
                row_count,
                columns.len()
            )];
            for (name, col) in columns {
                lines.push(format!("  {}: {}", name, summarize_vector(col)));
            }
            Ok(RObject::char_scalar(lines.join("\n")))
        }
        other => Ok(RObject::char_scalar(summarize_vector(other))),
    }
}

fn summarize_vector(obj: &RObject) -> String {
    let opts = obj.as_numeric_options();
    let na_count = opts.iter().filter(|o| o.is_none()).count();
    let mut vals: Vec<f64> = opts
        .into_iter()
        .flatten()
        .filter(|v| v.is_finite())
        .collect();
    if vals.is_empty() {
        return format!("{} (no numeric values)", obj.type_summary());
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let q0 = vals[0];
    let q25 = quantile_sorted(&vals, 0.25_f64);
    let q50 = quantile_sorted(&vals, 0.5_f64);
    let q75 = quantile_sorted(&vals, 0.75_f64);
    let q100 = vals[vals.len() - 1];
    let na_str = if na_count > 0 {
        format!(" | NA's={}", na_count)
    } else {
        String::new()
    };
    format!(
        "Min={:.4} | 1st Qu={:.4} | Median={:.4} | Mean={:.4} | 3rd Qu={:.4} | Max={:.4}{}",
        q0, q25, q50, mean, q75, q100, na_str
    )
}

fn map_numeric_unary(
    args: &[EvalArg],
    fn_name: &str,
    f: impl Fn(f64) -> f64,
) -> Result<RObject, RError> {
    let x = args
        .first()
        .ok_or_else(|| RError::new(format!("`{}(x)` requires 1 argument", fn_name)))?;
    let vals = x.value.as_numeric_options();
    let out: Vec<Option<f64>> = vals
        .into_iter()
        .map(|opt| opt.map(&f).and_then(|v| (!v.is_nan()).then_some(v)))
        .collect();
    Ok(RObject::Numeric(out))
}

fn builtin_log(args: &[EvalArg]) -> Result<RObject, RError> {
    let x = find_arg(args, "x", 0)
        .ok_or_else(|| RError::new("`log(x, [base])` requires at least 1 argument"))?;
    let base = find_arg(args, "base", 1).and_then(|o| o.as_scalar_f64());
    let vals = x.as_numeric_options();
    let out = vals
        .into_iter()
        .map(|opt| {
            opt.and_then(|v| {
                let r = match base {
                    Some(b) => v.ln() / b.ln(),
                    None => v.ln(),
                };
                (!r.is_nan()).then_some(r)
            })
        })
        .collect();
    Ok(RObject::Numeric(out))
}

fn builtin_round(args: &[EvalArg]) -> Result<RObject, RError> {
    let x = find_arg(args, "x", 0)
        .ok_or_else(|| RError::new("`round(x, digits = 0)` requires at least 1 argument"))?;
    let digits = find_arg(args, "digits", 1)
        .and_then(|o| o.as_scalar_f64())
        .unwrap_or(0.0_f64)
        .round() as i32;
    let factor = 10.0_f64.powi(digits);
    let out = x
        .as_numeric_options()
        .into_iter()
        .map(|opt| opt.map(|v| (v * factor).round() / factor))
        .collect();
    Ok(RObject::Numeric(out))
}

fn builtin_ifelse(args: &[EvalArg]) -> Result<RObject, RError> {
    let test = find_arg(args, "test", 0)
        .ok_or_else(|| RError::new("`ifelse(test, yes, no)` requires 3 arguments"))?;
    let yes = find_arg(args, "yes", 1)
        .ok_or_else(|| RError::new("`ifelse(test, yes, no)` requires 3 arguments"))?;
    let no = find_arg(args, "no", 2)
        .ok_or_else(|| RError::new("`ifelse(test, yes, no)` requires 3 arguments"))?;

    let conds = test.as_logical_options();
    if matches!(yes, RObject::Character(_)) || matches!(no, RObject::Character(_)) {
        let yv = yes.as_character_options();
        let nv = no.as_character_options();
        let mut out = Vec::with_capacity(conds.len());
        for (i, c) in conds.iter().enumerate() {
            match c {
                Some(true) => out.push(if yv.is_empty() { None } else { yv[i % yv.len()].clone() }),
                Some(false) => out.push(if nv.is_empty() { None } else { nv[i % nv.len()].clone() }),
                None => out.push(None),
            }
        }
        return Ok(RObject::Character(out));
    }

    let yv = yes.as_numeric_options();
    let nv = no.as_numeric_options();
    let mut out = Vec::with_capacity(conds.len());
    for (i, c) in conds.iter().enumerate() {
        match c {
            Some(true) => out.push(if yv.is_empty() { None } else { yv[i % yv.len()] }),
            Some(false) => out.push(if nv.is_empty() { None } else { nv[i % nv.len()] }),
            None => out.push(None),
        }
    }
    Ok(RObject::Numeric(out))
}

fn collect_stat_values(args: &[EvalArg], na_rm: bool) -> (Vec<f64>, bool) {
    let mut vals = Vec::new();
    let mut saw_na = false;
    for a in args {
        if a.name.as_deref() == Some("na.rm") {
            continue;
        }
        for opt in a.value.as_numeric_options() {
            match opt {
                Some(v) if !v.is_nan() => vals.push(v),
                _ => {
                    if !na_rm {
                        saw_na = true;
                    }
                }
            }
        }
    }
    (vals, saw_na)
}

fn builtin_sum_prod(args: &[EvalArg], is_sum: bool) -> Result<RObject, RError> {
    let na_rm = na_rm_flag(args);
    let (vals, saw_na) = collect_stat_values(args, na_rm);
    if saw_na {
        return Ok(RObject::Numeric(vec![None]));
    }
    let res = if is_sum {
        vals.iter().sum()
    } else {
        vals.iter().product()
    };
    Ok(RObject::scalar(res))
}

fn builtin_mean(args: &[EvalArg]) -> Result<RObject, RError> {
    let na_rm = na_rm_flag(args);
    let (vals, saw_na) = collect_stat_values(args, na_rm);
    if saw_na || vals.is_empty() {
        return Ok(RObject::Numeric(vec![None]));
    }
    Ok(RObject::scalar(vals.iter().sum::<f64>() / (vals.len() as f64)))
}

fn builtin_median(args: &[EvalArg]) -> Result<RObject, RError> {
    let na_rm = na_rm_flag(args);
    let (mut vals, saw_na) = collect_stat_values(args, na_rm);
    if saw_na || vals.is_empty() {
        return Ok(RObject::Numeric(vec![None]));
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(RObject::scalar(quantile_sorted(&vals, 0.5_f64)))
}

fn builtin_var_sd(args: &[EvalArg], is_sd: bool) -> Result<RObject, RError> {
    let na_rm = na_rm_flag(args);
    let (vals, saw_na) = collect_stat_values(args, na_rm);
    if saw_na || vals.len() < 2 {
        return Ok(RObject::Numeric(vec![None]));
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0_f64);
    Ok(RObject::scalar(if is_sd { var.sqrt() } else { var }))
}

fn builtin_min_max(args: &[EvalArg], is_min: bool) -> Result<RObject, RError> {
    let na_rm = na_rm_flag(args);
    let (vals, saw_na) = collect_stat_values(args, na_rm);
    if saw_na {
        return Ok(RObject::Numeric(vec![None]));
    }
    if vals.is_empty() {
        return Ok(RObject::scalar(if is_min {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }));
    }
    let res = if is_min {
        vals.iter().copied().fold(f64::INFINITY, f64::min)
    } else {
        vals.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    };
    Ok(RObject::scalar(res))
}

fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let p = p.clamp(0.0_f64, 1.0_f64);
    if sorted.len() == 1 {
        return sorted[0];
    }
    // R Type 7 quantile interpolation
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    let frac = h - (lo as f64);
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

fn builtin_quantile(args: &[EvalArg]) -> Result<RObject, RError> {
    let x = find_arg(args, "x", 0)
        .ok_or_else(|| RError::new("`quantile(x, probs)` requires a numeric vector"))?;
    let na_rm = na_rm_flag(args);
    let probs = find_arg(args, "probs", 1)
        .map(|o| o.as_f64_vec())
        .unwrap_or_else(|| vec![0.0_f64, 0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]);

    let opts = x.as_numeric_options();
    if !na_rm && opts.iter().any(|o| o.is_none()) {
        return Err(RError::new("missing values not allowed in `quantile()` unless `na.rm = TRUE`"));
    }
    let mut vals: Vec<f64> = opts.into_iter().flatten().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return Ok(RObject::Numeric(vec![None; probs.len()]));
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let out = probs
        .into_iter()
        .map(|p| Some(quantile_sorted(&vals, p)))
        .collect();
    Ok(RObject::Numeric(out))
}

fn builtin_cum(args: &[EvalArg], is_sum: bool) -> Result<RObject, RError> {
    let x = args
        .first()
        .ok_or_else(|| RError::new("`cumsum`/`cumprod` requires 1 argument"))?;
    let vals = x.value.as_numeric_options();
    let mut acc = if is_sum { 0.0_f64 } else { 1.0_f64 };
    let mut poisoned = false;
    let mut out = Vec::with_capacity(vals.len());
    for opt in vals {
        if poisoned {
            out.push(None);
            continue;
        }
        match opt {
            Some(v) => {
                if is_sum {
                    acc += v;
                } else {
                    acc *= v;
                }
                out.push(Some(acc));
            }
            None => {
                poisoned = true;
                out.push(None);
            }
        }
    }
    Ok(RObject::Numeric(out))
}

fn builtin_diff(args: &[EvalArg]) -> Result<RObject, RError> {
    let x = args
        .first()
        .ok_or_else(|| RError::new("`diff(x)` requires 1 argument"))?;
    let vals = x.value.as_numeric_options();
    if vals.len() < 2 {
        return Ok(RObject::Numeric(Vec::new()));
    }
    let mut out = Vec::with_capacity(vals.len() - 1);
    for w in vals.windows(2) {
        out.push(match (w[0], w[1]) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        });
    }
    Ok(RObject::Numeric(out))
}

fn builtin_sort(args: &[EvalArg]) -> Result<RObject, RError> {
    let x = find_arg(args, "x", 0)
        .ok_or_else(|| RError::new("`sort(x)` requires 1 argument"))?;
    let decreasing = find_named(args, "decreasing")
        .and_then(|o| o.is_truthy().ok())
        .unwrap_or(false);
    let mut vals: Vec<f64> = x
        .as_numeric_options()
        .into_iter()
        .flatten()
        .filter(|v| v.is_finite())
        .collect();
    vals.sort_by(|a, b| {
        if decreasing {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
    Ok(RObject::numeric_vec(vals))
}

fn builtin_rev(args: &[EvalArg]) -> Result<RObject, RError> {
    let x = args
        .first()
        .ok_or_else(|| RError::new("`rev(x)` requires 1 argument"))?;
    Ok(match &x.value {
        RObject::Numeric(v) => RObject::Numeric(v.iter().rev().copied().collect()),
        RObject::Logical(v) => RObject::Logical(v.iter().rev().copied().collect()),
        RObject::Character(v) => RObject::Character(v.iter().rev().cloned().collect()),
        other => other.clone(),
    })
}

fn builtin_cov_cor(args: &[EvalArg], is_cor: bool) -> Result<RObject, RError> {
    let x = find_arg(args, "x", 0)
        .ok_or_else(|| RError::new("`cor`/`cov` requires `x` and `y` arguments"))?;
    let y = find_arg(args, "y", 1)
        .ok_or_else(|| RError::new("`cor`/`cov` requires `x` and `y` arguments"))?;

    let xv = x.as_numeric_options();
    let yv = y.as_numeric_options();
    if xv.len() != yv.len() {
        return Err(RError::new("`cor`/`cov`: incompatible vector lengths"));
    }
    let mut pairs = Vec::with_capacity(xv.len());
    for (a, b) in xv.into_iter().zip(yv) {
        match (a, b) {
            (Some(xa), Some(yb)) if xa.is_finite() && yb.is_finite() => pairs.push((xa, yb)),
            _ => return Ok(RObject::Numeric(vec![None])),
        }
    }
    if pairs.len() < 2 {
        return Ok(RObject::Numeric(vec![None]));
    }
    let n = pairs.len() as f64;
    let mx = pairs.iter().map(|(a, _)| *a).sum::<f64>() / n;
    let my = pairs.iter().map(|(_, b)| *b).sum::<f64>() / n;
    let mut sxy = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    for (a, b) in &pairs {
        let dx = a - mx;
        let dy = b - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if is_cor {
        let denom = (sxx * syy).sqrt();
        if denom < 1e-15_f64 {
            Ok(RObject::Numeric(vec![None]))
        } else {
            Ok(RObject::scalar(sxy / denom))
        }
    } else {
        Ok(RObject::scalar(sxy / (n - 1.0_f64)))
    }
}

#[allow(dead_code)]
pub fn format_vector_preview_compact(vals: &[Option<f64>]) -> String {
    let head: Vec<String> = vals
        .iter()
        .take(5)
        .map(|o| o.map(format_r_num).unwrap_or_else(|| "NA".to_string()))
        .collect();
    if vals.len() > 5 {
        format!("{} ...", head.join(", "))
    } else {
        head.join(", ")
    }
}
