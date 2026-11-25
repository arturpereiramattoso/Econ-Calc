use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RSpan {
    pub start: usize,
    pub end: usize,
}

impl RSpan {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RError {
    pub message: String,
    pub span: Option<RSpan>,
}

impl RError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
        }
    }

    pub fn with_span(message: impl Into<String>, span: RSpan) -> Self {
        Self {
            message: message.into(),
            span: Some(span),
        }
    }
}

impl fmt::Display for RError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RError {}

#[derive(Debug, Clone, PartialEq)]
pub struct RParam {
    pub name: String,
    pub default: Option<RExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RCallArg {
    pub name: Option<String>,
    pub value: Option<RExpr>, // None supports empty slice slots like `m[1, ]`
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RUnaryOp {
    Neg,
    Pos,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    IntDiv,
    MatMul,
    Colon,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    ShortAnd,
    ShortOr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RExpr {
    Null,
    Na,
    Bool(bool),
    Number(f64),
    String(String),
    Ident(String),
    Unary {
        op: RUnaryOp,
        expr: Box<RExpr>,
    },
    Binary {
        op: RBinOp,
        lhs: Box<RExpr>,
        rhs: Box<RExpr>,
    },
    Assign {
        target: Box<RExpr>,
        value: Box<RExpr>,
    },
    Formula {
        lhs: Option<Box<RExpr>>,
        rhs: Box<RExpr>,
    },
    Block(Vec<RExpr>),
    If {
        cond: Box<RExpr>,
        then_branch: Box<RExpr>,
        else_branch: Option<Box<RExpr>>,
    },
    For {
        var: String,
        iter: Box<RExpr>,
        body: Box<RExpr>,
    },
    While {
        cond: Box<RExpr>,
        body: Box<RExpr>,
    },
    FunctionDef {
        params: Vec<RParam>,
        body: Box<RExpr>,
    },
    Return(Option<Box<RExpr>>),
    Break,
    Next,
    Call {
        callee: Box<RExpr>,
        args: Vec<RCallArg>,
    },
    /// `x[i]` or `m[r, c]` (entries may be `None` for missing slice dimension e.g. `m[, 2]`)
    Index {
        target: Box<RExpr>,
        indices: Vec<Option<RExpr>>,
    },
    /// `x[[k]]`
    DoubleIndex {
        target: Box<RExpr>,
        index: Box<RExpr>,
    },
    /// `df$col`
    Member {
        target: Box<RExpr>,
        field: String,
    },
}

impl RExpr {
    /// Renders a compact textual representation (used for formula terms and closure display).
    pub fn to_source_string(&self) -> String {
        match self {
            RExpr::Null => "NULL".to_string(),
            RExpr::Na => "NA".to_string(),
            RExpr::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
            RExpr::Number(n) => {
                if n.fract().abs() < 1e-12 && n.abs() < 1e12 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            RExpr::String(s) => format!("\"{}\"", s),
            RExpr::Ident(name) => name.clone(),
            RExpr::Unary { op, expr } => {
                let sym = match op {
                    RUnaryOp::Neg => "-",
                    RUnaryOp::Pos => "+",
                    RUnaryOp::Not => "!",
                };
                format!("{}{}", sym, expr.to_source_string())
            }
            RExpr::Binary { op, lhs, rhs } => {
                let sym = match op {
                    RBinOp::Add => " + ",
                    RBinOp::Sub => " - ",
                    RBinOp::Mul => " * ",
                    RBinOp::Div => " / ",
                    RBinOp::Pow => "^",
                    RBinOp::Mod => " %% ",
                    RBinOp::IntDiv => " %/% ",
                    RBinOp::MatMul => " %*% ",
                    RBinOp::Colon => ":",
                    RBinOp::Eq => " == ",
                    RBinOp::Ne => " != ",
                    RBinOp::Lt => " < ",
                    RBinOp::Le => " <= ",
                    RBinOp::Gt => " > ",
                    RBinOp::Ge => " >= ",
                    RBinOp::And => " & ",
                    RBinOp::Or => " | ",
                    RBinOp::ShortAnd => " && ",
                    RBinOp::ShortOr => " || ",
                };
                format!(
                    "{}{}{}",
                    lhs.to_source_string(),
                    sym,
                    rhs.to_source_string()
                )
            }
            RExpr::Assign { target, value } => {
                format!(
                    "{} <- {}",
                    target.to_source_string(),
                    value.to_source_string()
                )
            }
            RExpr::Formula { lhs, rhs } => {
                if let Some(l) = lhs {
                    format!("{} ~ {}", l.to_source_string(), rhs.to_source_string())
                } else {
                    format!("~ {}", rhs.to_source_string())
                }
            }
            RExpr::Call { callee, args } => {
                let a: Vec<String> = args
                    .iter()
                    .map(|arg| match (&arg.name, &arg.value) {
                        (Some(n), Some(v)) => format!("{} = {}", n, v.to_source_string()),
                        (None, Some(v)) => v.to_source_string(),
                        (Some(n), None) => format!("{} =", n),
                        (None, None) => String::new(),
                    })
                    .collect();
                format!("{}({})", callee.to_source_string(), a.join(", "))
            }
            RExpr::Member { target, field } => format!("{}${}", target.to_source_string(), field),
            RExpr::Index { target, indices } => {
                let idxs: Vec<String> = indices
                    .iter()
                    .map(|i| i.as_ref().map(|e| e.to_source_string()).unwrap_or_default())
                    .collect();
                format!("{}[{}]", target.to_source_string(), idxs.join(", "))
            }
            RExpr::DoubleIndex { target, index } => {
                format!(
                    "{}[[{}]]",
                    target.to_source_string(),
                    index.to_source_string()
                )
            }
            RExpr::FunctionDef { params, .. } => {
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
                format!("function({}) {{ ... }}", ps.join(", "))
            }
            RExpr::Block(stmts) => {
                let s: Vec<String> = stmts.iter().map(|e| e.to_source_string()).collect();
                format!("{{ {} }}", s.join("; "))
            }
            RExpr::If { cond, .. } => format!("if ({}) ...", cond.to_source_string()),
            RExpr::For { var, iter, .. } => {
                format!("for ({} in {}) ...", var, iter.to_source_string())
            }
            RExpr::While { cond, .. } => format!("while ({}) ...", cond.to_source_string()),
            RExpr::Return(v) => match v {
                Some(e) => format!("return({})", e.to_source_string()),
                None => "return()".to_string(),
            },
            RExpr::Break => "break".to_string(),
            RExpr::Next => "next".to_string(),
        }
    }
}
