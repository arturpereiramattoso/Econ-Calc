use std::fmt;

/// Byte span in the original input string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub const fn merge(self, other: Self) -> Self {
        let start = if self.start < other.start {
            self.start
        } else {
            other.start
        };
        let end = if self.end > other.end {
            self.end
        } else {
            other.end
        };
        Self { start, end }
    }

    /// Returns 1-based `(line, column)` for the start of this span in `src`.
    pub fn line_col(self, src: &str) -> (usize, usize) {
        let clamped = self.start.min(src.len());
        let prefix = &src[..clamped];
        let mut line = 1usize;
        let mut col = 1usize;
        for ch in prefix.chars() {
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }
}

/// Span-accurate math parse or evaluation diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathError {
    pub message: String,
    pub span: Span,
}

impl MathError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }

    pub fn format_with_source(&self, src: &str) -> String {
        let (line, col) = self.span.line_col(src);
        format!("line {}, col {}: {}", line, col, self.message)
    }
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (bytes {}..{})",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for MathError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccentKind {
    Bar,
    Hat,
    Tilde,
    Vec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Pos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    ImplicitMul,
    Pow,
    Mod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CmpOp {
    pub fn as_unicode(self) -> &'static str {
        match self {
            CmpOp::Lt => "<",
            CmpOp::Le => "≤",
            CmpOp::Gt => ">",
            CmpOp::Ge => "≥",
            CmpOp::Eq => "=",
            CmpOp::Ne => "≠",
        }
    }

    pub fn as_latex(self) -> &'static str {
        match self {
            CmpOp::Lt => "<",
            CmpOp::Le => "\\le",
            CmpOp::Gt => ">",
            CmpOp::Ge => "\\ge",
            CmpOp::Eq => "=",
            CmpOp::Ne => "\\neq",
        }
    }

    pub fn eval(self, lhs: f64, rhs: f64) -> bool {
        if !lhs.is_finite() || !rhs.is_finite() {
            return false;
        }
        match self {
            CmpOp::Lt => lhs < rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Gt => lhs > rhs,
            CmpOp::Ge => lhs >= rhs,
            CmpOp::Eq => (lhs - rhs).abs() <= 1e-12_f64,
            CmpOp::Ne => (lhs - rhs).abs() > 1e-12_f64,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseBranch {
    pub condition: Option<SpannedExpr>,
    pub value: SpannedExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpannedExpr {
    pub expr: MathExpr,
    pub span: Span,
}

impl SpannedExpr {
    pub fn new(expr: MathExpr, span: Span) -> Self {
        Self { expr, span }
    }

    pub fn to_latex(&self) -> String {
        expr_to_latex(&self.expr, 0)
    }

    /// Returns a canonical identifier string if this expression is an identifier or `\bar{Y}`-style accent.
    pub fn as_canonical_ident(&self) -> Option<String> {
        match &self.expr {
            MathExpr::Ident(name) => Some(name.clone()),
            MathExpr::Accent {
                kind: AccentKind::Bar,
                inner,
            } => {
                if let MathExpr::Ident(base) = &inner.expr {
                    Some(format!("{}_bar", base))
                } else {
                    None
                }
            }
            MathExpr::Accent {
                kind: AccentKind::Hat,
                inner,
            } => {
                if let MathExpr::Ident(base) = &inner.expr {
                    Some(format!("{}_hat", base))
                } else {
                    None
                }
            }
            MathExpr::Accent {
                kind: AccentKind::Tilde,
                inner,
            } => {
                if let MathExpr::Ident(base) = &inner.expr {
                    Some(format!("{}_tilde", base))
                } else {
                    None
                }
            }
            MathExpr::Subscript { base, sub } => {
                if let (Some(b), Some(s)) = (base.as_canonical_ident(), sub.as_canonical_ident()) {
                    Some(format!("{}_{}", b, s))
                } else if let (Some(b), MathExpr::Number(n)) =
                    (base.as_canonical_ident(), &sub.expr)
                {
                    if n.fract().abs() < 1e-12_f64 {
                        Some(format!("{}_{}", b, *n as i64))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MathExpr {
    Number(f64),
    Ident(String),
    Subscript {
        base: Box<SpannedExpr>,
        sub: Box<SpannedExpr>,
    },
    Accent {
        kind: AccentKind,
        inner: Box<SpannedExpr>,
    },
    Group(Box<SpannedExpr>),
    Abs(Box<SpannedExpr>),
    Unary {
        op: UnaryOp,
        expr: Box<SpannedExpr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<SpannedExpr>,
        rhs: Box<SpannedExpr>,
    },
    Fraction {
        num: Box<SpannedExpr>,
        den: Box<SpannedExpr>,
    },
    Sqrt {
        index: Option<Box<SpannedExpr>>,
        radicand: Box<SpannedExpr>,
    },
    Call {
        name: String,
        prime_order: u8,
        args: Vec<SpannedExpr>,
    },
    Derivative {
        var: String,
        order: u8,
        body: Box<SpannedExpr>,
    },
    Integral {
        var: String,
        lower: Box<SpannedExpr>,
        upper: Box<SpannedExpr>,
        body: Box<SpannedExpr>,
    },
    Sum {
        var: String,
        lower: Box<SpannedExpr>,
        upper: Box<SpannedExpr>,
        body: Box<SpannedExpr>,
    },
    Prod {
        var: String,
        lower: Box<SpannedExpr>,
        upper: Box<SpannedExpr>,
        body: Box<SpannedExpr>,
    },
    ComparisonChain {
        first: Box<SpannedExpr>,
        rest: Vec<(CmpOp, SpannedExpr)>,
    },
    Piecewise(Vec<PiecewiseBranch>),
    Tuple(Vec<SpannedExpr>),
    MemberAccess {
        object: String,
        field: String,
    },
    Index {
        target: Box<SpannedExpr>,
        index: Box<SpannedExpr>,
    },
}

/// Top-level Desmos-style mathematical statement.
#[derive(Debug, Clone, PartialEq)]
pub enum MathStatement {
    FunctionDef {
        name: String,
        params: Vec<String>,
        body: SpannedExpr,
    },
    ParamAssign {
        name: String,
        value: SpannedExpr,
    },
    Equation {
        lhs: SpannedExpr,
        rhs: SpannedExpr,
    },
    Inequality {
        lhs: SpannedExpr,
        op: CmpOp,
        rhs: SpannedExpr,
    },
    Parametric {
        name: Option<String>,
        param: String,
        x_expr: SpannedExpr,
        y_expr: SpannedExpr,
    },
    Polar {
        param: String,
        r_expr: SpannedExpr,
    },
    Expr(SpannedExpr),
}

impl MathStatement {
    pub fn to_latex(&self) -> String {
        match self {
            MathStatement::FunctionDef { name, params, body } => {
                let p_str: Vec<String> = params.iter().map(|p| ident_to_latex(p)).collect();
                format!(
                    "{}({}) = {}",
                    ident_to_latex(name),
                    p_str.join(", "),
                    body.to_latex()
                )
            }
            MathStatement::ParamAssign { name, value } => {
                format!("{} = {}", ident_to_latex(name), value.to_latex())
            }
            MathStatement::Equation { lhs, rhs } => {
                format!("{} = {}", lhs.to_latex(), rhs.to_latex())
            }
            MathStatement::Inequality { lhs, op, rhs } => {
                format!("{} {} {}", lhs.to_latex(), op.as_latex(), rhs.to_latex())
            }
            MathStatement::Parametric {
                name,
                param,
                x_expr,
                y_expr,
            } => {
                let tuple = format!("\\left({}, {}\\right)", x_expr.to_latex(), y_expr.to_latex());
                if let Some(n) = name {
                    format!("{}({}) = {}", ident_to_latex(n), ident_to_latex(param), tuple)
                } else {
                    tuple
                }
            }
            MathStatement::Polar { param, r_expr } => {
                if param == "theta" {
                    format!("r = {}", r_expr.to_latex())
                } else {
                    format!("r({}) = {}", ident_to_latex(param), r_expr.to_latex())
                }
            }
            MathStatement::Expr(expr) => expr.to_latex(),
        }
    }
}

/// Greek letters mapping `(ascii_name, unicode_glyph, latex_cmd)`.
pub const GREEK_TABLE: &[(&str, &str, &str)] = &[
    ("alpha", "α", "\\alpha"),
    ("beta", "β", "\\beta"),
    ("gamma", "γ", "\\gamma"),
    ("delta", "δ", "\\delta"),
    ("epsilon", "ε", "\\epsilon"),
    ("varepsilon", "ε", "\\varepsilon"),
    ("zeta", "ζ", "\\zeta"),
    ("eta", "η", "\\eta"),
    ("theta", "θ", "\\theta"),
    ("vartheta", "ϑ", "\\vartheta"),
    ("iota", "ι", "\\iota"),
    ("kappa", "κ", "\\kappa"),
    ("lambda", "λ", "\\lambda"),
    ("mu", "μ", "\\mu"),
    ("nu", "ν", "\\nu"),
    ("xi", "ξ", "\\xi"),
    ("pi", "π", "\\pi"),
    ("rho", "ρ", "\\rho"),
    ("sigma", "σ", "\\sigma"),
    ("tau", "τ", "\\tau"),
    ("upsilon", "υ", "\\upsilon"),
    ("phi", "φ", "\\phi"),
    ("varphi", "φ", "\\varphi"),
    ("chi", "χ", "\\chi"),
    ("psi", "ψ", "\\psi"),
    ("omega", "ω", "\\omega"),
    ("Gamma", "Γ", "\\Gamma"),
    ("Delta", "Δ", "\\Delta"),
    ("Theta", "Θ", "\\Theta"),
    ("Lambda", "Λ", "\\Lambda"),
    ("Xi", "Ξ", "\\Xi"),
    ("Pi", "Π", "\\Pi"),
    ("Sigma", "Σ", "\\Sigma"),
    ("Phi", "Φ", "\\Phi"),
    ("Psi", "Ψ", "\\Psi"),
    ("Omega", "Ω", "\\Omega"),
    ("infty", "∞", "\\infty"),
];

pub fn greek_name_to_unicode(name: &str) -> Option<&'static str> {
    GREEK_TABLE
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, u, _)| *u)
}

pub fn unicode_char_to_greek_name(ch: char) -> Option<&'static str> {
    let mut buf = [0u8; 4];
    let s = ch.encode_utf8(&mut buf);
    GREEK_TABLE
        .iter()
        .find(|(_, u, _)| *u == s)
        .map(|(n, _, _)| *n)
}

pub fn greek_name_to_latex(name: &str) -> Option<&'static str> {
    GREEK_TABLE
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, _, l)| *l)
}

pub fn ident_to_latex(name: &str) -> String {
    if let Some(cmd) = greek_name_to_latex(name) {
        return cmd.to_string();
    }
    if name == "price_x" {
        return "P_x".to_string();
    }
    if name == "price_y" {
        return "P_y".to_string();
    }
    if let Some(base) = name.strip_suffix("_bar") {
        return format!("\\bar{{{}}}", ident_to_latex(base));
    }
    if let Some(base) = name.strip_suffix("_hat") {
        return format!("\\hat{{{}}}", ident_to_latex(base));
    }
    if let Some(base) = name.strip_suffix("_tilde") {
        return format!("\\tilde{{{}}}", ident_to_latex(base));
    }
    if let Some((base, sub)) = name.split_once('_') {
        if !base.is_empty() && !sub.is_empty() {
            let b = ident_to_latex(base);
            let s = if sub.len() == 1 {
                ident_to_latex(sub)
            } else {
                format!("{{{}}}", ident_to_latex(sub))
            };
            return format!("{}_{}", b, s);
        }
    }
    name.to_string()
}

pub fn ident_to_pretty_unicode(name: &str) -> String {
    if let Some(u) = greek_name_to_unicode(name) {
        return u.to_string();
    }
    if name == "price_x" {
        return "Pₓ".to_string();
    }
    if name == "price_y" {
        return "Pᵧ".to_string();
    }
    if let Some(base) = name.strip_suffix("_bar") {
        return format!("{}\u{0304}", ident_to_pretty_unicode(base));
    }
    if let Some((base, sub)) = name.split_once('_') {
        return format!("{}_{}", ident_to_pretty_unicode(base), sub);
    }
    name.to_string()
}

fn format_number_latex(v: f64) -> String {
    if !v.is_finite() {
        return "\\text{NaN}".to_string();
    }
    if v.fract().abs() < 1e-12_f64 && v.abs() < 1e12_f64 {
        format!("{}", v as i64)
    } else {
        let s = format!("{:.6}", v);
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    }
}

fn is_standard_latex_func(name: &str) -> bool {
    matches!(
        name,
        "sin"
            | "cos"
            | "tan"
            | "sec"
            | "csc"
            | "cot"
            | "arcsin"
            | "arccos"
            | "arctan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "ln"
            | "log"
            | "exp"
            | "min"
            | "max"
    )
}

fn expr_to_latex(expr: &MathExpr, parent_prec: u8) -> String {
    match expr {
        MathExpr::Number(n) => format_number_latex(*n),
        MathExpr::Ident(name) => ident_to_latex(name),
        MathExpr::Subscript { base, sub } => {
            format!(
                "{{{}}}_{{{}}}",
                expr_to_latex(&base.expr, 5),
                expr_to_latex(&sub.expr, 0)
            )
        }
        MathExpr::Accent { kind, inner } => {
            let cmd = match kind {
                AccentKind::Bar => "\\bar",
                AccentKind::Hat => "\\hat",
                AccentKind::Tilde => "\\tilde",
                AccentKind::Vec => "\\vec",
            };
            format!("{}{{{}}}", cmd, expr_to_latex(&inner.expr, 0))
        }
        MathExpr::Group(inner) => format!("\\left({}\\right)", expr_to_latex(&inner.expr, 0)),
        MathExpr::Abs(inner) => format!("\\left|{}\\right|", expr_to_latex(&inner.expr, 0)),
        MathExpr::Unary { op, expr } => {
            let inner = expr_to_latex(&expr.expr, 4);
            match op {
                UnaryOp::Neg => format!("-{}", inner),
                UnaryOp::Pos => format!("+{}", inner),
            }
        }
        MathExpr::Fraction { num, den } => {
            let n = strip_outer_group(&num.expr);
            let d = strip_outer_group(&den.expr);
            format!(
                "\\frac{{{}}}{{{}}}",
                expr_to_latex(n, 0),
                expr_to_latex(d, 0)
            )
        }
        MathExpr::Sqrt { index, radicand } => {
            let r = expr_to_latex(strip_outer_group(&radicand.expr), 0);
            if let Some(idx) = index {
                format!("\\sqrt[{}]{{{}}}", expr_to_latex(&idx.expr, 0), r)
            } else {
                format!("\\sqrt{{{}}}", r)
            }
        }
        MathExpr::Binary { op, lhs, rhs } => {
            let prec = match op {
                BinOp::Add | BinOp::Sub => 1,
                BinOp::Mul | BinOp::ImplicitMul | BinOp::Mod => 2,
                BinOp::Pow => 4,
            };
            let s = match op {
                BinOp::Add => format!(
                    "{} + {}",
                    expr_to_latex(&lhs.expr, 1),
                    expr_to_latex(&rhs.expr, 1)
                ),
                BinOp::Sub => format!(
                    "{} - {}",
                    expr_to_latex(&lhs.expr, 1),
                    expr_to_latex(&rhs.expr, 2)
                ),
                BinOp::Mul => format!(
                    "{} \\cdot {}",
                    expr_to_latex(&lhs.expr, 2),
                    expr_to_latex(&rhs.expr, 2)
                ),
                BinOp::ImplicitMul => format!(
                    "{} {}",
                    expr_to_latex(&lhs.expr, 2),
                    expr_to_latex(&rhs.expr, 2)
                ),
                BinOp::Mod => format!(
                    "{} \\bmod {}",
                    expr_to_latex(&lhs.expr, 2),
                    expr_to_latex(&rhs.expr, 2)
                ),
                BinOp::Pow => {
                    let base = expr_to_latex(&lhs.expr, 5);
                    let exp = expr_to_latex(strip_outer_group(&rhs.expr), 0);
                    format!("{}^{{{}}}", base, exp)
                }
            };
            if prec < parent_prec {
                format!("\\left({}\\right)", s)
            } else {
                s
            }
        }
        MathExpr::Call {
            name,
            prime_order,
            args,
        } => {
            let primes = "'".repeat(*prime_order as usize);
            let args_str: Vec<String> = args.iter().map(|a| expr_to_latex(&a.expr, 0)).collect();
            if *prime_order == 0 && name == "abs" && args.len() == 1 {
                return format!("\\left|{}\\right|", args_str[0]);
            }
            if *prime_order == 0 && name == "sqrt" && args.len() == 1 {
                return format!("\\sqrt{{{}}}", args_str[0]);
            }
            if *prime_order == 0 && is_standard_latex_func(name) {
                format!("\\{}\\left({}\\right)", name, args_str.join(", "))
            } else if *prime_order == 0 && matches!(name.as_str(), "asin" | "acos" | "atan") {
                format!(
                    "\\arc{}\\left({}\\right)",
                    &name[1..],
                    args_str.join(", ")
                )
            } else {
                format!(
                    "{}{}\\left({}\\right)",
                    ident_to_latex(name),
                    primes,
                    args_str.join(", ")
                )
            }
        }
        MathExpr::Derivative { var, order, body } => {
            let v = ident_to_latex(var);
            let b = expr_to_latex(&body.expr, 0);
            if *order <= 1 {
                format!("\\frac{{d}}{{d{}}}\\left({}\\right)", v, b)
            } else {
                format!(
                    "\\frac{{d^{{{}}}}}{{d{}^{{{}}}}}\\left({}\\right)",
                    order, v, order, b
                )
            }
        }
        MathExpr::Integral {
            var,
            lower,
            upper,
            body,
        } => {
            format!(
                "\\int_{{{}}}^{{{}}} {} \\, d{}",
                expr_to_latex(&lower.expr, 0),
                expr_to_latex(&upper.expr, 0),
                expr_to_latex(&body.expr, 0),
                ident_to_latex(var)
            )
        }
        MathExpr::Sum {
            var,
            lower,
            upper,
            body,
        } => {
            format!(
                "\\sum_{{{}={}}}^{{{}}} {}",
                ident_to_latex(var),
                expr_to_latex(&lower.expr, 0),
                expr_to_latex(&upper.expr, 0),
                expr_to_latex(&body.expr, 2)
            )
        }
        MathExpr::Prod {
            var,
            lower,
            upper,
            body,
        } => {
            format!(
                "\\prod_{{{}={}}}^{{{}}} {}",
                ident_to_latex(var),
                expr_to_latex(&lower.expr, 0),
                expr_to_latex(&upper.expr, 0),
                expr_to_latex(&body.expr, 2)
            )
        }
        MathExpr::ComparisonChain { first, rest } => {
            let mut out = expr_to_latex(&first.expr, 0);
            for (op, rhs) in rest {
                out.push(' ');
                out.push_str(op.as_latex());
                out.push(' ');
                out.push_str(&expr_to_latex(&rhs.expr, 0));
            }
            out
        }
        MathExpr::Piecewise(branches) => {
            let rows: Vec<String> = branches
                .iter()
                .map(|b| {
                    let val = expr_to_latex(&b.value.expr, 0);
                    if let Some(cond) = &b.condition {
                        format!("{} & \\text{{if }} {}", val, expr_to_latex(&cond.expr, 0))
                    } else {
                        format!("{} & \\text{{otherwise}}", val)
                    }
                })
                .collect();
            format!("\\begin{{cases}} {} \\end{{cases}}", rows.join(" \\\\ "))
        }
        MathExpr::Tuple(items) => {
            let parts: Vec<String> = items.iter().map(|e| expr_to_latex(&e.expr, 0)).collect();
            format!("\\left({}\\right)", parts.join(", "))
        }
        MathExpr::MemberAccess { object, field } => {
            format!("\\text{{{}}}\\$\\text{{{}}}", object, field)
        }
        MathExpr::Index { target, index } => {
            format!(
                "{}\\left[{}\\right]",
                expr_to_latex(&target.expr, 5),
                expr_to_latex(&index.expr, 0)
            )
        }
    }
}

fn strip_outer_group(expr: &MathExpr) -> &MathExpr {
    match expr {
        MathExpr::Group(inner) => strip_outer_group(&inner.expr),
        other => other,
    }
}
