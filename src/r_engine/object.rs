use super::ast::{RExpr, RParam};
use crate::collections::IndexMap;

/// Rich R-subset runtime value supporting vectors with `NA` propagation, matrices, lists,
/// data frames, model formulas, and lexical closures.
#[derive(Debug, Clone, PartialEq)]
pub enum RObject {
    Null,
    Logical(Vec<Option<bool>>),
    Numeric(Vec<Option<f64>>),
    Character(Vec<Option<String>>),
    Matrix {
        /// Column-major storage matching R's `matrix()` semantics.
        data: Vec<Option<f64>>,
        nrow: usize,
        ncol: usize,
        rownames: Option<Vec<String>>,
        colnames: Option<Vec<String>>,
    },
    List(IndexMap<String, RObject>),
    DataFrame {
        columns: IndexMap<String, RObject>,
        row_count: usize,
    },
    Formula {
        lhs: Option<String>,
        rhs: String,
    },
    Closure {
        params: Vec<RParam>,
        body: Box<RExpr>,
        env: IndexMap<String, RObject>,
    },
}

impl RObject {
    pub fn scalar(v: f64) -> Self {
        if v.is_nan() {
            RObject::Numeric(vec![None])
        } else {
            RObject::Numeric(vec![Some(v)])
        }
    }

    pub fn numeric_vec(vals: Vec<f64>) -> Self {
        RObject::Numeric(
            vals.into_iter()
                .map(|v| if v.is_nan() { None } else { Some(v) })
                .collect(),
        )
    }

    pub fn bool_scalar(b: bool) -> Self {
        RObject::Logical(vec![Some(b)])
    }

    pub fn char_scalar(s: impl Into<String>) -> Self {
        RObject::Character(vec![Some(s.into())])
    }

    pub fn as_scalar_f64(&self) -> Option<f64> {
        match self {
            RObject::Numeric(v) if v.len() == 1 => v[0].filter(|x| x.is_finite()),
            RObject::Logical(v) if v.len() == 1 => v[0].map(|b| if b { 1.0_f64 } else { 0.0_f64 }),
            _ => None,
        }
    }

    pub fn as_numeric_options(&self) -> Vec<Option<f64>> {
        match self {
            RObject::Numeric(v) => v.clone(),
            RObject::Logical(v) => v
                .iter()
                .map(|opt| opt.map(|b| if b { 1.0_f64 } else { 0.0_f64 }))
                .collect(),
            RObject::Matrix { data, .. } => data.clone(),
            RObject::Character(v) => v
                .iter()
                .map(|opt| opt.as_deref().and_then(|s| s.trim().parse::<f64>().ok()))
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn as_f64_vec(&self) -> Vec<f64> {
        self.as_numeric_options()
            .into_iter()
            .flatten()
            .filter(|v| v.is_finite())
            .collect()
    }

    pub fn as_logical_options(&self) -> Vec<Option<bool>> {
        match self {
            RObject::Logical(v) => v.clone(),
            RObject::Numeric(v) => v.iter().map(|opt| opt.map(|x| x != 0.0_f64)).collect(),
            RObject::Character(v) => v
                .iter()
                .map(|opt| {
                    opt.as_deref().and_then(|s| match s.trim() {
                        "TRUE" | "T" | "true" => Some(true),
                        "FALSE" | "F" | "false" => Some(false),
                        _ => None,
                    })
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn as_character_options(&self) -> Vec<Option<String>> {
        match self {
            RObject::Character(v) => v.clone(),
            RObject::Numeric(v) => v.iter().map(|opt| opt.map(format_r_num)).collect(),
            RObject::Logical(v) => v
                .iter()
                .map(|opt| opt.map(|b| if b { "TRUE" } else { "FALSE" }.to_string()))
                .collect(),
            RObject::Matrix { data, .. } => data.iter().map(|opt| opt.map(format_r_num)).collect(),
            _ => Vec::new(),
        }
    }

    pub fn is_truthy(&self) -> Result<bool, String> {
        match self {
            RObject::Logical(v) => match v.first().copied().flatten() {
                Some(b) => Ok(b),
                None => Err("missing value where TRUE/FALSE needed".to_string()),
            },
            RObject::Numeric(v) => match v.first().copied().flatten() {
                Some(x) => Ok(x != 0.0_f64),
                None => Err("missing value where TRUE/FALSE needed".to_string()),
            },
            _ => Err("argument is not interpretable as logical".to_string()),
        }
    }

    pub fn r_len(&self) -> usize {
        match self {
            RObject::Null => 0,
            RObject::Logical(v) => v.len(),
            RObject::Numeric(v) => v.len(),
            RObject::Character(v) => v.len(),
            RObject::Matrix { data, .. } => data.len(),
            RObject::List(m) => m.len(),
            RObject::DataFrame { columns, .. } => columns.len(),
            RObject::Formula { lhs, .. } => {
                if lhs.is_some() {
                    3
                } else {
                    2
                }
            }
            RObject::Closure { .. } => 1,
        }
    }

    pub fn is_lm_model(&self) -> bool {
        if let RObject::List(map) = self {
            if let Some(RObject::Character(cls)) = map.get("class") {
                return cls.first().and_then(|c| c.as_deref()) == Some("lm");
            }
        }
        false
    }

    pub fn closure_signature(&self) -> Option<String> {
        if let RObject::Closure { params, .. } = self {
            let ps: Vec<String> = params
                .iter()
                .map(|p| {
                    if let Some(d) = &p.default {
                        format!("{} = {}", p.name, d.to_source_string())
                    } else {
                        p.name.clone()
                    }
                })
                .collect();
            Some(format!("function({})", ps.join(", ")))
        } else {
            None
        }
    }

    pub fn type_summary(&self) -> String {
        match self {
            RObject::Null => "NULL".to_string(),
            RObject::Logical(v) => format!("logical [1:{}]", v.len()),
            RObject::Numeric(v) => {
                if v.len() == 1 {
                    "numeric [1]".to_string()
                } else {
                    format!("numeric [1:{}]", v.len())
                }
            }
            RObject::Character(v) => format!("character [1:{}]", v.len()),
            RObject::Matrix { nrow, ncol, .. } => format!("matrix [{} x {}]", nrow, ncol),
            RObject::List(m) => {
                if self.is_lm_model() {
                    format!("lm model ({} fields)", m.len())
                } else {
                    format!("list ({} items)", m.len())
                }
            }
            RObject::DataFrame { columns, row_count } => {
                format!("data.frame [{} obs. of {} var.]", row_count, columns.len())
            }
            RObject::Formula { .. } => "formula".to_string(),
            RObject::Closure { params, .. } => {
                let ps: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
                format!("function({})", ps.join(", "))
            }
        }
    }

    pub fn preview_string(&self) -> String {
        match self {
            RObject::Null => "NULL".to_string(),
            RObject::Logical(v) => {
                if v.is_empty() {
                    return "logical(0)".to_string();
                }
                let head: Vec<&str> = v
                    .iter()
                    .take(10)
                    .map(|x| match x {
                        Some(true) => "TRUE",
                        Some(false) => "FALSE",
                        None => "NA",
                    })
                    .collect();
                if v.len() > 10 {
                    format!("[1] {} ... ({} items)", head.join(" "), v.len())
                } else {
                    format!("[1] {}", head.join(" "))
                }
            }
            RObject::Numeric(v) => {
                if v.is_empty() {
                    return "numeric(0)".to_string();
                }
                let head: Vec<String> = v
                    .iter()
                    .take(8)
                    .map(|x| match x {
                        Some(val) => format_r_num(*val),
                        None => "NA".to_string(),
                    })
                    .collect();
                if v.len() > 8 {
                    format!("[1] {} ... ({} items)", head.join(" "), v.len())
                } else {
                    format!("[1] {}", head.join(" "))
                }
            }
            RObject::Character(v) => {
                if v.is_empty() {
                    return "character(0)".to_string();
                }
                let head: Vec<String> = v
                    .iter()
                    .take(6)
                    .map(|x| match x {
                        Some(s) => format!("\"{}\"", s),
                        None => "NA".to_string(),
                    })
                    .collect();
                if v.len() > 6 {
                    format!("[1] {} ... ({} items)", head.join(" "), v.len())
                } else {
                    format!("[1] {}", head.join(" "))
                }
            }
            RObject::Matrix {
                data, nrow, ncol, ..
            } => {
                let mut lines = vec![format!("Matrix [{} x {}]:", nrow, ncol)];
                let max_r = (*nrow).min(5);
                let max_c = (*ncol).min(6);
                for r in 0..max_r {
                    let mut row_items = Vec::with_capacity(max_c);
                    for c in 0..max_c {
                        let val = data.get(c * nrow + r).copied().flatten();
                        row_items.push(val.map(format_r_num).unwrap_or_else(|| "NA".to_string()));
                    }
                    let ellipsis = if *ncol > max_c { " ..." } else { "" };
                    lines.push(format!(
                        "  [{},] {}{}",
                        r + 1,
                        row_items.join("  "),
                        ellipsis
                    ));
                }
                if *nrow > max_r {
                    lines.push(format!("  ... ({} more rows)", nrow - max_r));
                }
                lines.join("\n")
            }
            RObject::List(map) => {
                if self.is_lm_model() {
                    let call_str = map
                        .get("call")
                        .and_then(|o| match o {
                            RObject::Character(c) => c.first().and_then(|s| s.clone()),
                            _ => None,
                        })
                        .unwrap_or_else(|| "lm(...)".to_string());
                    let r2 = map
                        .get("r.squared")
                        .and_then(|o| o.as_scalar_f64())
                        .unwrap_or(f64::NAN);
                    let coef_str = map
                        .get("coefficients")
                        .map(|c| c.preview_string())
                        .unwrap_or_default();
                    return format!(
                        "Call: {} | R² = {:.4}\nCoefficients: {}",
                        call_str, r2, coef_str
                    );
                }
                let mut lines = Vec::new();
                for (i, (k, v)) in map.iter().take(6).enumerate() {
                    let label = if k.is_empty() {
                        format!("[[{}]]", i + 1)
                    } else {
                        format!("${}", k)
                    };
                    lines.push(format!("{} : {}", label, v.preview_string()));
                }
                if map.len() > 6 {
                    lines.push(format!("... ({} more entries)", map.len() - 6));
                }
                if lines.is_empty() {
                    "list()".to_string()
                } else {
                    lines.join("\n")
                }
            }
            RObject::DataFrame { columns, row_count } => {
                let col_names: Vec<&str> = columns.keys().map(|s| s.as_str()).collect();
                let mut lines = vec![format!(
                    "data.frame [{} x {}]: {}",
                    row_count,
                    columns.len(),
                    col_names.join(", ")
                )];
                let max_r = (*row_count).min(5);
                for r in 0..max_r {
                    let mut cells = Vec::with_capacity(columns.len());
                    for (_, col) in columns.iter().take(6) {
                        let cell = format_column_cell(col, r);
                        cells.push(cell);
                    }
                    lines.push(format!("  {} | {}", r + 1, cells.join("  ")));
                }
                if *row_count > max_r {
                    lines.push(format!("  ... ({} more rows)", row_count - max_r));
                }
                lines.join("\n")
            }
            RObject::Formula { lhs, rhs } => {
                if let Some(l) = lhs {
                    format!("{} ~ {}", l, rhs)
                } else {
                    format!("~ {}", rhs)
                }
            }
            RObject::Closure { .. } => self
                .closure_signature()
                .unwrap_or_else(|| "function()".to_string()),
        }
    }

    /// Detailed `str(x)` output matching R's structure summary.
    pub fn format_str(&self) -> String {
        match self {
            RObject::DataFrame { columns, row_count } => {
                let mut out = vec![format!(
                    "'data.frame': {} obs. of {} variables:",
                    row_count,
                    columns.len()
                )];
                for (name, col) in columns {
                    out.push(format!(
                        " $ {}: {} {}",
                        name,
                        col.type_summary(),
                        col.preview_string()
                    ));
                }
                out.join("\n")
            }
            RObject::List(map) => {
                let mut out = vec![format!("List of {}", map.len())];
                for (name, val) in map {
                    out.push(format!(" $ {}: {}", name, val.type_summary()));
                }
                out.join("\n")
            }
            other => format!("{} {}", other.type_summary(), other.preview_string()),
        }
    }
}

pub fn format_r_num(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_positive() {
            "Inf".to_string()
        } else {
            "-Inf".to_string()
        };
    }
    if v.fract().abs() < 1e-10_f64 && v.abs() < 1e12_f64 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{:.5}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub fn format_column_cell(col: &RObject, row: usize) -> String {
    match col {
        RObject::Numeric(v) => v
            .get(row)
            .copied()
            .flatten()
            .map(format_r_num)
            .unwrap_or_else(|| "NA".to_string()),
        RObject::Logical(v) => match v.get(row).copied().flatten() {
            Some(true) => "TRUE".to_string(),
            Some(false) => "FALSE".to_string(),
            None => "NA".to_string(),
        },
        RObject::Character(v) => v
            .get(row)
            .and_then(|s| s.as_deref())
            .map(|s| format!("\"{}\"", s))
            .unwrap_or_else(|| "NA".to_string()),
        _ => "?".to_string(),
    }
}
