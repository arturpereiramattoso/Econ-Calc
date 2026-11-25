use super::ast::{
    AccentKind, BinOp, CmpOp, MathError, MathExpr, MathStatement, PiecewiseBranch, Span,
    SpannedExpr, UnaryOp,
};
use super::lexer::{tokenize, tokenize_lenient, Token, TokenKind};
use std::collections::HashSet;

/// Built-in math function names recognized by the parser and evaluator.
pub fn is_builtin_function(name: &str) -> bool {
    matches!(
        name,
        "abs"
            | "acos"
            | "acosh"
            | "asin"
            | "asinh"
            | "atan"
            | "atanh"
            | "atan2"
            | "cbrt"
            | "ceil"
            | "cos"
            | "cosh"
            | "cot"
            | "csc"
            | "exp"
            | "floor"
            | "ln"
            | "log"
            | "log10"
            | "log2"
            | "max"
            | "min"
            | "nthroot"
            | "round"
            | "sec"
            | "sign"
            | "signum"
            | "sin"
            | "sinh"
            | "sqrt"
            | "tan"
            | "tanh"
            | "sum"
            | "prod"
            | "int"
            | "diff"
            | "tangent"
            | "elasticity"
            | "marginal"
            | "average"
            | "area"
            | "surplus"
    )
}

/// Built-in unary functions that may omit parentheses in LaTeX (e.g. `\sin x`, `\ln x`).
fn is_prefix_math_func(name: &str) -> bool {
    matches!(
        name,
        "sin"
            | "cos"
            | "tan"
            | "sec"
            | "csc"
            | "cot"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asin"
            | "acos"
            | "atan"
            | "ln"
            | "log"
            | "exp"
            | "sqrt"
    )
}

/// Variables that should prefer implicit multiplication over function-call syntax (`x(x+1)` -> `x*(x+1)`).
fn is_coordinate_or_const_ident(name: &str) -> bool {
    matches!(
        name,
        "x" | "y" | "z" | "t" | "theta" | "pi" | "PI" | "e" | "phi" | "infty"
    )
}

pub fn parse_expr(input: &str) -> Result<SpannedExpr, MathError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens, None, false);
    let expr = parser.parse_comparison()?;
    parser.expect_eof()?;
    Ok(expr)
}

pub fn parse_expr_with_known_fns(
    input: &str,
    known_fns: &HashSet<String>,
) -> Result<SpannedExpr, MathError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens, Some(known_fns), false);
    let expr = parser.parse_comparison()?;
    parser.expect_eof()?;
    Ok(expr)
}

pub fn parse_statement(input: &str) -> Result<MathStatement, MathError> {
    parse_statement_with_known_fns(input, None)
}

pub fn parse_statement_with_known_fns(
    input: &str,
    known_fns: Option<&HashSet<String>>,
) -> Result<MathStatement, MathError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens, known_fns, false);
    let stmt = parser.parse_top_level_statement()?;
    parser.expect_eof()?;
    Ok(stmt)
}

/// Structural lenient statement parser for live "Math-as-Written" typing previews.
/// Produces `MathExpr::Placeholder` slots (`□`) for missing structural operands (`/`, `^`, `_`,
/// `\frac`, `sqrt()`, `nthroot()`, `\int`, `\sum`, `\prod`, `d/dx`, `=`) without failing.
pub fn parse_statement_for_preview(
    input: &str,
    known_fns: Option<&HashSet<String>>,
) -> Result<MathStatement, MathError> {
    let tokens = tokenize_lenient(input)?;
    let mut parser = Parser::new(tokens, known_fns, true);
    let stmt = parser.parse_top_level_statement()?;
    parser.expect_eof()?;
    Ok(stmt)
}

const MAX_PARSE_DEPTH: usize = 128;

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    known_fns: Option<&'a HashSet<String>>,
    in_abs_depth: usize,
    in_integral_depth: usize,
    lenient: bool,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token>, known_fns: Option<&'a HashSet<String>>, lenient: bool) -> Self {
        Self {
            tokens,
            pos: 0,
            known_fns,
            in_abs_depth: 0,
            in_integral_depth: 0,
            lenient,
            depth: 0,
        }
    }

    fn with_depth<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, MathError>,
    ) -> Result<T, MathError> {
        if self.depth >= MAX_PARSE_DEPTH {
            return Err(MathError::new(
                "Expression exceeds maximum nesting depth",
                self.peek().span,
            ));
        }
        self.depth += 1;
        let res = f(self);
        self.depth -= 1;
        res
    }

    fn is_structural_end(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::Eof
                | TokenKind::RParen
                | TokenKind::RBrace
                | TokenKind::RBracket
                | TokenKind::Comma
                | TokenKind::Colon
                | TokenKind::Semicolon
                | TokenKind::Eq
                | TokenKind::EqEq
                | TokenKind::Ne
                | TokenKind::Lt
                | TokenKind::Le
                | TokenKind::Gt
                | TokenKind::Ge
        )
    }

    fn placeholder_here(&self) -> SpannedExpr {
        SpannedExpr::new(MathExpr::Placeholder, self.peek().span)
    }

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .unwrap_or_else(|| &self.tokens[self.tokens.len() - 1])
    }

    fn peek_ahead(&self, offset: usize) -> &Token {
        let idx = (self.pos + offset).min(self.tokens.len().saturating_sub(1));
        &self.tokens[idx]
    }

    fn advance(&mut self) -> Token {
        let tok = self.peek().clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn is_eof(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    fn expect_eof(&self) -> Result<(), MathError> {
        if self.is_eof() {
            Ok(())
        } else {
            Err(MathError::new(
                format!("Unexpected token `{:?}`", self.peek().kind),
                self.peek().span,
            ))
        }
    }

    fn expect(&mut self, expected: TokenKind, msg: &str) -> Result<Token, MathError> {
        if self.peek().kind == expected {
            Ok(self.advance())
        } else {
            Err(MathError::new(msg, self.peek().span))
        }
    }

    fn is_known_function(&self, name: &str) -> bool {
        is_builtin_function(name)
            || self
                .known_fns
                .map(|set| set.contains(name))
                .unwrap_or(false)
    }

    fn parse_top_level_statement(&mut self) -> Result<MathStatement, MathError> {
        if self.is_eof() {
            return Err(MathError::new("Empty expression", Span::new(0, 0)));
        }

        // Parse LHS additive/expression first without consuming top-level `=` or `<`/`<=`/`>`/`>=`
        let lhs = self.parse_additive()?;

        match self.peek().kind {
            TokenKind::Eq => {
                self.advance();
                let rhs = if self.lenient && self.is_eof() {
                    self.placeholder_here()
                } else {
                    self.parse_comparison()?
                };

                // 1. Function definition: `f(x) = ...`, `f(x, y) = ...`, `r(theta) = ...`, `p(t) = (x(t), y(t))`
                if let MathExpr::Call {
                    ref name,
                    prime_order: 0,
                    ref args,
                } = lhs.expr
                {
                    let param_names: Option<Vec<String>> =
                        args.iter().map(|a| a.as_canonical_ident()).collect();
                    if let Some(params) = param_names {
                        if !params.is_empty() {
                            if name == "r"
                                && params.len() == 1
                                && (params[0] == "theta" || params[0] == "t")
                            {
                                return Ok(MathStatement::Polar {
                                    param: params[0].clone(),
                                    r_expr: rhs,
                                });
                            }
                            if let MathExpr::Tuple(mut items) = rhs.expr.clone() {
                                if items.len() == 2 && params.len() == 1 {
                                    let y_expr = items.remove(1);
                                    let x_expr = items.remove(0);
                                    return Ok(MathStatement::Parametric {
                                        name: Some(name.clone()),
                                        param: params[0].clone(),
                                        x_expr,
                                        y_expr,
                                    });
                                }
                            }
                            return Ok(MathStatement::FunctionDef {
                                name: name.clone(),
                                params,
                                body: rhs,
                            });
                        }
                    }
                }

                // 2. Polar curve: `r = f(theta)`
                if let Some(ident) = lhs.as_canonical_ident() {
                    if ident == "r" && expr_mentions_any(&rhs, &["theta"]) {
                        return Ok(MathStatement::Polar {
                            param: "theta".to_string(),
                            r_expr: rhs,
                        });
                    }
                    // 3. Explicit `y = f(x)` -> Equation (so plotter renders 1D curve and LaTeX shows `y = ...`)
                    if ident == "y" || ident == "x" {
                        return Ok(MathStatement::Equation { lhs, rhs });
                    }
                    // 4. Scalar/parameter assignment if RHS has no free coordinate variables
                    if !expr_mentions_any(&rhs, &["x", "y", "t", "theta"])
                        && !matches!(rhs.expr, MathExpr::Restricted { .. })
                    {
                        return Ok(MathStatement::ParamAssign {
                            name: ident,
                            value: rhs,
                        });
                    }
                }

                // 5. General implicit equation `F(x, y) = G(x, y)`
                Ok(MathStatement::Equation { lhs, rhs })
            }
            TokenKind::Lt | TokenKind::Le | TokenKind::Gt | TokenKind::Ge => {
                let op = self.consume_cmp_op()?;
                let rhs = if self.lenient && self.is_eof() {
                    self.placeholder_here()
                } else {
                    self.parse_additive()?
                };
                // Check if chained comparison (e.g. `0 <= x <= 10`) -> return as Expr(ComparisonChain)
                if self.peek_cmp_op().is_some() {
                    let mut rest = vec![(op, rhs)];
                    while self.peek_cmp_op().is_some() {
                        let next_op = self.consume_cmp_op()?;
                        let next_rhs = if self.lenient && self.is_eof() {
                            self.placeholder_here()
                        } else {
                            self.parse_additive()?
                        };
                        rest.push((next_op, next_rhs));
                    }
                    let span = lhs
                        .span
                        .merge(rest.last().map(|(_, e)| e.span).unwrap_or(lhs.span));
                    let expr = SpannedExpr::new(
                        MathExpr::ComparisonChain {
                            first: Box::new(lhs),
                            rest,
                        },
                        span,
                    );
                    let expr = self.maybe_parse_trailing_restrictions(expr)?;
                    return Ok(MathStatement::Expr(expr));
                }
                let rhs = self.maybe_parse_trailing_restrictions(rhs)?;
                Ok(MathStatement::Inequality { lhs, op, rhs })
            }
            TokenKind::EqEq | TokenKind::Ne => {
                let op = self.consume_cmp_op()?;
                let rhs = self.parse_additive()?;
                let mut rest = vec![(op, rhs)];
                while self.peek_cmp_op().is_some() {
                    let next_op = self.consume_cmp_op()?;
                    let next_rhs = self.parse_additive()?;
                    rest.push((next_op, next_rhs));
                }
                let span = lhs
                    .span
                    .merge(rest.last().map(|(_, e)| e.span).unwrap_or(lhs.span));
                Ok(MathStatement::Expr(SpannedExpr::new(
                    MathExpr::ComparisonChain {
                        first: Box::new(lhs),
                        rest,
                    },
                    span,
                )))
            }
            _ => {
                let lhs = self.maybe_parse_trailing_restrictions(lhs)?;
                if let MathExpr::Tuple(mut items) = lhs.expr.clone() {
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
                        return Ok(MathStatement::Parametric {
                            name: None,
                            param,
                            x_expr,
                            y_expr,
                        });
                    }
                }
                Ok(MathStatement::Expr(lhs))
            }
        }
    }

    fn peek_cmp_op(&self) -> Option<CmpOp> {
        match self.peek().kind {
            TokenKind::Lt => Some(CmpOp::Lt),
            TokenKind::Le => Some(CmpOp::Le),
            TokenKind::Gt => Some(CmpOp::Gt),
            TokenKind::Ge => Some(CmpOp::Ge),
            TokenKind::EqEq | TokenKind::Eq => Some(CmpOp::Eq),
            TokenKind::Ne => Some(CmpOp::Ne),
            _ => None,
        }
    }

    fn consume_cmp_op(&mut self) -> Result<CmpOp, MathError> {
        let tok = self.advance();
        match tok.kind {
            TokenKind::Lt => Ok(CmpOp::Lt),
            TokenKind::Le => Ok(CmpOp::Le),
            TokenKind::Gt => Ok(CmpOp::Gt),
            TokenKind::Ge => Ok(CmpOp::Ge),
            TokenKind::EqEq | TokenKind::Eq => Ok(CmpOp::Eq),
            TokenKind::Ne => Ok(CmpOp::Ne),
            _ => Err(MathError::new("Expected comparison operator", tok.span)),
        }
    }

    fn parse_comparison(&mut self) -> Result<SpannedExpr, MathError> {
        let first = self.parse_additive()?;
        let base = if self.peek_cmp_op().is_none() {
            first
        } else {
            let mut rest = Vec::new();
            while self.peek_cmp_op().is_some() {
                let op = self.consume_cmp_op()?;
                let rhs = if self.lenient && self.is_structural_end() {
                    self.placeholder_here()
                } else {
                    self.parse_additive()?
                };
                rest.push((op, rhs));
            }
            let end_span = rest.last().map(|(_, e)| e.span).unwrap_or(first.span);
            let span = first.span.merge(end_span);
            SpannedExpr::new(
                MathExpr::ComparisonChain {
                    first: Box::new(first),
                    rest,
                },
                span,
            )
        };
        self.maybe_parse_trailing_restrictions(base)
    }

    fn brace_is_restriction(&self) -> bool {
        if !matches!(self.peek().kind, TokenKind::LBrace) {
            return false;
        }
        let mut depth = 0usize;
        let mut idx = self.pos;
        let mut saw_cmp = false;
        while idx < self.tokens.len() {
            match self.tokens[idx].kind {
                TokenKind::LBrace | TokenKind::LParen | TokenKind::LBracket => depth += 1,
                TokenKind::RBrace | TokenKind::RParen | TokenKind::RBracket => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return saw_cmp;
                    }
                }
                TokenKind::Colon if depth == 1 => return false,
                TokenKind::Lt
                | TokenKind::Le
                | TokenKind::Gt
                | TokenKind::Ge
                | TokenKind::EqEq
                | TokenKind::Ne
                    if depth == 1 =>
                {
                    saw_cmp = true;
                }
                TokenKind::Eof => return false,
                _ => {}
            }
            idx += 1;
        }
        false
    }

    fn maybe_parse_trailing_restrictions(
        &mut self,
        expr: SpannedExpr,
    ) -> Result<SpannedExpr, MathError> {
        let mut current = expr;
        while matches!(self.peek().kind, TokenKind::LBrace) && self.brace_is_restriction() {
            self.advance(); // `{`
            let mut restrictions = Vec::new();
            while !matches!(self.peek().kind, TokenKind::RBrace | TokenKind::Eof) {
                restrictions.push(self.parse_comparison()?);
                if matches!(self.peek().kind, TokenKind::Comma | TokenKind::Semicolon) {
                    self.advance();
                } else {
                    break;
                }
            }
            let close_span = if matches!(self.peek().kind, TokenKind::RBrace) {
                self.advance().span
            } else if self.lenient {
                self.peek().span
            } else {
                return Err(MathError::new(
                    "Expected `}` to close domain/range restriction",
                    self.peek().span,
                ));
            };
            if !restrictions.is_empty() {
                let span = current.span.merge(close_span);
                current = SpannedExpr::new(
                    MathExpr::Restricted {
                        inner: Box::new(current),
                        restrictions,
                    },
                    span,
                );
            }
        }
        Ok(current)
    }

    fn parse_additive(&mut self) -> Result<SpannedExpr, MathError> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.peek().kind {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_multiplicative()?;
            let span = lhs.span.merge(rhs.span);
            lhs = SpannedExpr::new(
                MathExpr::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn can_start_implicit_mul(&self) -> bool {
        // Inside an integral, stop before differential `dx`, `dt`, `d theta`, etc.
        if self.in_integral_depth > 0 && self.peek_is_differential() {
            return false;
        }
        match &self.peek().kind {
            TokenKind::Number(_) | TokenKind::Ident(_) | TokenKind::LParen => true,
            TokenKind::LBrace => !self.brace_is_restriction(),
            TokenKind::Pipe => self.in_abs_depth == 0,
            TokenKind::Command(cmd) => matches!(
                cmd.as_str(),
                "frac" | "sqrt" | "sum" | "prod" | "int" | "bar" | "hat" | "tilde" | "vec"
            ),
            _ => false,
        }
    }

    fn peek_is_differential(&self) -> bool {
        match &self.peek().kind {
            TokenKind::Ident(s) if s == "d" => {
                matches!(self.peek_ahead(1).kind, TokenKind::Ident(_))
            }
            TokenKind::Ident(s)
                if s.starts_with('d')
                    && s.len() >= 2
                    && !is_builtin_function(s)
                    && matches!(
                        &s[1..],
                        "x" | "y" | "z" | "t" | "u" | "v" | "s" | "r" | "theta" | "k" | "q"
                    ) =>
            {
                true
            }
            _ => false,
        }
    }

    fn parse_multiplicative(&mut self) -> Result<SpannedExpr, MathError> {
        let mut lhs = self.parse_unary()?;
        loop {
            match self.peek().kind {
                TokenKind::Star => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    let span = lhs.span.merge(rhs.span);
                    lhs = SpannedExpr::new(
                        MathExpr::Binary {
                            op: BinOp::Mul,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                        span,
                    );
                }
                TokenKind::Slash => {
                    // Check `d/dx(expr)` shorthand!
                    if let MathExpr::Ident(ref name) = lhs.expr {
                        if name == "d" {
                            if let TokenKind::Ident(ref den_name) = self.peek_ahead(1).kind {
                                if let Some(var) = den_name.strip_prefix('d') {
                                    if !var.is_empty() && !is_builtin_function(den_name) {
                                        let var = var.to_string();
                                        self.advance(); // `/`
                                        self.advance(); // `dx`
                                        let body = if self.lenient && self.is_structural_end() {
                                            self.placeholder_here()
                                        } else {
                                            self.parse_unary()?
                                        };
                                        let span = lhs.span.merge(body.span);
                                        lhs = SpannedExpr::new(
                                            MathExpr::Derivative {
                                                var,
                                                order: 1,
                                                body: Box::new(body),
                                            },
                                            span,
                                        );
                                        continue;
                                    }
                                }
                            }
                        }
                    }

                    let slash_tok = self.advance();
                    let rhs = if self.lenient
                        && (self.is_structural_end()
                            || matches!(
                                self.peek().kind,
                                TokenKind::Plus
                                    | TokenKind::Minus
                                    | TokenKind::Star
                                    | TokenKind::Slash
                            )) {
                        SpannedExpr::new(MathExpr::Placeholder, slash_tok.span)
                    } else {
                        self.parse_unary()?
                    };
                    let span = lhs.span.merge(rhs.span);
                    lhs = SpannedExpr::new(
                        MathExpr::Fraction {
                            num: Box::new(lhs),
                            den: Box::new(rhs),
                        },
                        span,
                    );
                }
                TokenKind::Percent => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    let span = lhs.span.merge(rhs.span);
                    lhs = SpannedExpr::new(
                        MathExpr::Binary {
                            op: BinOp::Mod,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                        span,
                    );
                }
                _ if self.can_start_implicit_mul() => {
                    let rhs = self.parse_power()?;
                    let span = lhs.span.merge(rhs.span);
                    lhs = SpannedExpr::new(
                        MathExpr::Binary {
                            op: BinOp::ImplicitMul,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                        span,
                    );
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<SpannedExpr, MathError> {
        self.with_depth(Self::parse_unary_inner)
    }

    fn parse_unary_inner(&mut self) -> Result<SpannedExpr, MathError> {
        match self.peek().kind {
            TokenKind::Minus => {
                let tok = self.advance();
                let operand = self.parse_unary()?;
                let span = tok.span.merge(operand.span);
                Ok(SpannedExpr::new(
                    MathExpr::Unary {
                        op: UnaryOp::Neg,
                        expr: Box::new(operand),
                    },
                    span,
                ))
            }
            TokenKind::Plus => {
                let tok = self.advance();
                let operand = self.parse_unary()?;
                let span = tok.span.merge(operand.span);
                Ok(SpannedExpr::new(
                    MathExpr::Unary {
                        op: UnaryOp::Pos,
                        expr: Box::new(operand),
                    },
                    span,
                ))
            }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> Result<SpannedExpr, MathError> {
        let base = self.parse_postfix()?;
        if matches!(self.peek().kind, TokenKind::Caret) {
            let caret_tok = self.advance();
            let exp = if self.lenient
                && (self.is_structural_end()
                    || matches!(
                        self.peek().kind,
                        TokenKind::Plus | TokenKind::Star | TokenKind::Slash
                    )) {
                SpannedExpr::new(MathExpr::Placeholder, caret_tok.span)
            } else {
                self.parse_unary()?
            };
            let span = base.span.merge(exp.span);
            Ok(SpannedExpr::new(
                MathExpr::Binary {
                    op: BinOp::Pow,
                    lhs: Box::new(base),
                    rhs: Box::new(exp),
                },
                span,
            ))
        } else {
            Ok(base)
        }
    }

    fn parse_postfix(&mut self) -> Result<SpannedExpr, MathError> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.peek().kind {
                TokenKind::Underscore => {
                    let us_tok = self.advance();
                    let sub = if self.lenient
                        && (self.is_structural_end()
                            || matches!(
                                self.peek().kind,
                                TokenKind::Plus
                                    | TokenKind::Minus
                                    | TokenKind::Star
                                    | TokenKind::Slash
                                    | TokenKind::Caret
                            )) {
                        SpannedExpr::new(MathExpr::Placeholder, us_tok.span)
                    } else {
                        self.parse_script_atom()?
                    };
                    let span = expr.span.merge(sub.span);
                    expr = SpannedExpr::new(
                        MathExpr::Subscript {
                            base: Box::new(expr),
                            sub: Box::new(sub),
                        },
                        span,
                    );
                }
                TokenKind::Dollar => {
                    if let Some(obj_name) = expr.as_canonical_ident() {
                        self.advance();
                        let field_tok = self.advance();
                        match field_tok.kind {
                            TokenKind::Ident(field) => {
                                let span = expr.span.merge(field_tok.span);
                                expr = SpannedExpr::new(
                                    MathExpr::MemberAccess {
                                        object: obj_name,
                                        field,
                                    },
                                    span,
                                );
                            }
                            _ => {
                                return Err(MathError::new(
                                    "Expected column/field name after `$`",
                                    field_tok.span,
                                ));
                            }
                        }
                    } else {
                        return Err(MathError::new(
                            "`$` member access requires an object identifier",
                            self.peek().span,
                        ));
                    }
                }
                TokenKind::LBracket => {
                    self.advance();
                    let idx = self.parse_comparison()?;
                    let end_tok = self.expect(TokenKind::RBracket, "Expected `]` after index")?;
                    let span = expr.span.merge(end_tok.span);
                    expr = SpannedExpr::new(
                        MathExpr::Index {
                            target: Box::new(expr),
                            index: Box::new(idx),
                        },
                        span,
                    );
                }
                TokenKind::Prime => {
                    if let Some(fn_name) = expr.as_canonical_ident() {
                        let mut prime_order = 0u8;
                        let mut last_span = expr.span;
                        while matches!(self.peek().kind, TokenKind::Prime) {
                            last_span = self.advance().span;
                            prime_order = prime_order.saturating_add(1);
                        }
                        if matches!(self.peek().kind, TokenKind::LParen) {
                            let (args, close_span) = self.parse_arg_list()?;
                            let span = expr.span.merge(close_span);
                            expr = SpannedExpr::new(
                                MathExpr::Call {
                                    name: fn_name,
                                    prime_order,
                                    args,
                                },
                                span,
                            );
                        } else {
                            // `f'` without parens -> derivative w.r.t. x applied to x
                            let span = expr.span.merge(last_span);
                            expr = SpannedExpr::new(
                                MathExpr::Call {
                                    name: fn_name,
                                    prime_order,
                                    args: vec![SpannedExpr::new(
                                        MathExpr::Ident("x".to_string()),
                                        last_span,
                                    )],
                                },
                                span,
                            );
                        }
                    } else {
                        return Err(MathError::new(
                            "Prime `'` can only be applied to a function name",
                            self.peek().span,
                        ));
                    }
                }
                TokenKind::LParen => {
                    // Decide whether `ident(...)` is a Call or ImplicitMul
                    if let Some(fn_name) = expr.as_canonical_ident() {
                        // Look ahead: if multi-arg `(a, b)`, always Call; if coordinate/const (`x(x+1)`), ImplicitMul!
                        let is_coord = is_coordinate_or_const_ident(&fn_name)
                            && !self.is_known_function(&fn_name);
                        let is_multi_arg = self.paren_has_top_level_comma();
                        if !is_coord || is_multi_arg {
                            let (args, close_span) = self.parse_arg_list()?;
                            let span = expr.span.merge(close_span);
                            // Normalize `sqrt()` / `nthroot(n, x)` into `MathExpr::Sqrt`
                            if fn_name == "sqrt" && args.len() <= 1 {
                                let radicand = args.into_iter().next().unwrap_or_else(|| {
                                    SpannedExpr::new(MathExpr::Placeholder, span)
                                });
                                if matches!(radicand.expr, MathExpr::Placeholder) && !self.lenient {
                                    return Err(MathError::new(
                                        "sqrt() requires an argument",
                                        span,
                                    ));
                                }
                                expr = SpannedExpr::new(
                                    MathExpr::Sqrt {
                                        index: None,
                                        radicand: Box::new(radicand),
                                    },
                                    span,
                                );
                                continue;
                            }
                            if fn_name == "nthroot" {
                                let mut it = args.into_iter();
                                let idx = it.next().unwrap_or_else(|| {
                                    SpannedExpr::new(MathExpr::Placeholder, span)
                                });
                                let rad = it.next().unwrap_or_else(|| {
                                    SpannedExpr::new(MathExpr::Placeholder, span)
                                });
                                if (matches!(idx.expr, MathExpr::Placeholder)
                                    || matches!(rad.expr, MathExpr::Placeholder))
                                    && !self.lenient
                                {
                                    return Err(MathError::new(
                                        "nthroot(n, x) requires 2 arguments",
                                        span,
                                    ));
                                }
                                expr = SpannedExpr::new(
                                    MathExpr::Sqrt {
                                        index: Some(Box::new(idx)),
                                        radicand: Box::new(rad),
                                    },
                                    span,
                                );
                                continue;
                            }
                            expr = SpannedExpr::new(
                                MathExpr::Call {
                                    name: fn_name,
                                    prime_order: 0,
                                    args,
                                },
                                span,
                            );
                            continue;
                        }
                    }
                    break;
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn paren_has_top_level_comma(&self) -> bool {
        if !matches!(self.peek().kind, TokenKind::LParen) {
            return false;
        }
        let mut depth = 0usize;
        let mut idx = self.pos;
        while idx < self.tokens.len() {
            match self.tokens[idx].kind {
                TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => depth += 1,
                TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return false;
                    }
                }
                TokenKind::Comma if depth == 1 => return true,
                TokenKind::Eof => return false,
                _ => {}
            }
            idx += 1;
        }
        false
    }

    fn parse_arg_list(&mut self) -> Result<(Vec<SpannedExpr>, Span), MathError> {
        let open = self.expect(TokenKind::LParen, "Expected `(`")?;
        let mut args = Vec::new();
        if matches!(self.peek().kind, TokenKind::RParen) {
            let close = self.advance();
            return Ok((args, open.span.merge(close.span)));
        }
        if self.lenient && self.is_eof() {
            return Ok((args, open.span));
        }
        loop {
            if self.lenient && matches!(self.peek().kind, TokenKind::RParen | TokenKind::Eof) {
                break;
            }
            args.push(self.parse_comparison()?);
            if matches!(self.peek().kind, TokenKind::Comma) {
                self.advance();
                if self.lenient && matches!(self.peek().kind, TokenKind::RParen | TokenKind::Eof) {
                    args.push(self.placeholder_here());
                    break;
                }
            } else {
                break;
            }
        }
        if matches!(self.peek().kind, TokenKind::RParen) {
            let close = self.advance();
            Ok((args, close.span))
        } else if self.lenient {
            Ok((args, self.peek().span))
        } else {
            Err(MathError::new(
                "Expected `)` to close argument list",
                self.peek().span,
            ))
        }
    }

    fn parse_script_atom(&mut self) -> Result<SpannedExpr, MathError> {
        if matches!(self.peek().kind, TokenKind::LBrace) {
            self.parse_braced_group()
        } else if self.lenient && self.is_structural_end() {
            Ok(self.placeholder_here())
        } else {
            self.parse_primary()
        }
    }

    fn parse_braced_group(&mut self) -> Result<SpannedExpr, MathError> {
        let open = self.expect(TokenKind::LBrace, "Expected `{`")?;
        if self.lenient && matches!(self.peek().kind, TokenKind::RBrace | TokenKind::Eof) {
            let end_span = if matches!(self.peek().kind, TokenKind::RBrace) {
                self.advance().span
            } else {
                open.span
            };
            return Ok(SpannedExpr::new(
                MathExpr::Placeholder,
                open.span.merge(end_span),
            ));
        }
        let inner = self.parse_comparison()?;
        let close_span = if matches!(self.peek().kind, TokenKind::RBrace) {
            self.advance().span
        } else if self.lenient {
            self.peek().span
        } else {
            return Err(MathError::new("Expected `}`", self.peek().span));
        };
        Ok(SpannedExpr::new(inner.expr, open.span.merge(close_span)))
    }

    fn brace_has_top_level_colon(&self) -> bool {
        if !matches!(self.peek().kind, TokenKind::LBrace) {
            return false;
        }
        let mut depth = 0usize;
        let mut idx = self.pos;
        while idx < self.tokens.len() {
            match self.tokens[idx].kind {
                TokenKind::LBrace | TokenKind::LParen | TokenKind::LBracket => depth += 1,
                TokenKind::RBrace | TokenKind::RParen | TokenKind::RBracket => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return false;
                    }
                }
                TokenKind::Colon if depth == 1 => return true,
                TokenKind::Eof => return false,
                _ => {}
            }
            idx += 1;
        }
        false
    }

    fn parse_piecewise(&mut self) -> Result<SpannedExpr, MathError> {
        let open = self.expect(TokenKind::LBrace, "Expected `{` for piecewise expression")?;
        let mut branches = Vec::new();

        while !matches!(self.peek().kind, TokenKind::RBrace | TokenKind::Eof) {
            let first = self.parse_comparison()?;
            if matches!(self.peek().kind, TokenKind::Colon) {
                self.advance();
                let value = self.parse_comparison()?;
                branches.push(PiecewiseBranch {
                    condition: Some(first),
                    value,
                });
            } else {
                branches.push(PiecewiseBranch {
                    condition: None,
                    value: first,
                });
            }

            if matches!(self.peek().kind, TokenKind::Comma | TokenKind::Semicolon) {
                self.advance();
            } else {
                break;
            }
        }

        let close = self.expect(
            TokenKind::RBrace,
            "Expected `}` to close piecewise expression",
        )?;
        if branches.is_empty() {
            return Err(MathError::new(
                "Piecewise expression cannot be empty",
                open.span.merge(close.span),
            ));
        }
        Ok(SpannedExpr::new(
            MathExpr::Piecewise(branches),
            open.span.merge(close.span),
        ))
    }

    fn parse_primary(&mut self) -> Result<SpannedExpr, MathError> {
        self.with_depth(Self::parse_primary_inner)
    }

    fn parse_primary_inner(&mut self) -> Result<SpannedExpr, MathError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Number(n) => {
                self.advance();
                Ok(SpannedExpr::new(MathExpr::Number(n), tok.span))
            }
            TokenKind::Ident(ref name) => {
                let name = name.clone();
                self.advance();

                // Support non-backslash big operators `int_0^1`, `sum_{i=1}^n`, `prod_{i=1}^n`
                if matches!(name.as_str(), "int" | "sum" | "prod")
                    && (matches!(self.peek().kind, TokenKind::Underscore | TokenKind::Caret)
                        || (self.lenient && self.is_eof()))
                {
                    return self.parse_latex_command(&name, tok.span);
                }

                // Lenient bare `sqrt` / `nthroot` at end of input -> immediate radical with slot
                if self.lenient && name == "sqrt" && self.is_eof() {
                    return Ok(SpannedExpr::new(
                        MathExpr::Sqrt {
                            index: None,
                            radicand: Box::new(SpannedExpr::new(MathExpr::Placeholder, tok.span)),
                        },
                        tok.span,
                    ));
                }
                if self.lenient && name == "nthroot" && self.is_eof() {
                    return Ok(SpannedExpr::new(
                        MathExpr::Sqrt {
                            index: Some(Box::new(SpannedExpr::new(
                                MathExpr::Placeholder,
                                tok.span,
                            ))),
                            radicand: Box::new(SpannedExpr::new(MathExpr::Placeholder, tok.span)),
                        },
                        tok.span,
                    ));
                }

                // Support `\sin^2(x)` / `sin^2(x)` -> `(sin(x))^2`
                if is_prefix_math_func(&name) && matches!(self.peek().kind, TokenKind::Caret) {
                    let saved = self.pos;
                    self.advance(); // `^`
                    if let Ok(exp) = self.parse_script_atom() {
                        if matches!(self.peek().kind, TokenKind::LParen) {
                            let (args, close_span) = self.parse_arg_list()?;
                            let call_span = tok.span.merge(close_span);
                            let call_expr = SpannedExpr::new(
                                MathExpr::Call {
                                    name,
                                    prime_order: 0,
                                    args,
                                },
                                call_span,
                            );
                            let full_span = call_span.merge(exp.span);
                            return Ok(SpannedExpr::new(
                                MathExpr::Binary {
                                    op: BinOp::Pow,
                                    lhs: Box::new(call_expr),
                                    rhs: Box::new(exp),
                                },
                                full_span,
                            ));
                        }
                    }
                    self.pos = saved;
                }

                // Support `\sin x`, `\ln x`, `\cos 2x` without parens if next token is an atom
                if is_prefix_math_func(&name)
                    && !matches!(
                        self.peek().kind,
                        TokenKind::LParen
                            | TokenKind::Underscore
                            | TokenKind::Caret
                            | TokenKind::Prime
                    )
                    && self.can_start_implicit_mul()
                {
                    let arg = self.parse_power()?;
                    let span = tok.span.merge(arg.span);
                    return Ok(SpannedExpr::new(
                        MathExpr::Call {
                            name,
                            prime_order: 0,
                            args: vec![arg],
                        },
                        span,
                    ));
                }

                Ok(SpannedExpr::new(MathExpr::Ident(name), tok.span))
            }
            TokenKind::Command(ref cmd) => {
                let cmd = cmd.clone();
                self.advance();
                self.parse_latex_command(&cmd, tok.span)
            }
            TokenKind::LParen => {
                let open = self.advance();
                if self.lenient && matches!(self.peek().kind, TokenKind::RParen | TokenKind::Eof) {
                    let close_span = if matches!(self.peek().kind, TokenKind::RParen) {
                        self.advance().span
                    } else {
                        open.span
                    };
                    return Ok(SpannedExpr::new(
                        MathExpr::Group(Box::new(SpannedExpr::new(
                            MathExpr::Placeholder,
                            open.span,
                        ))),
                        open.span.merge(close_span),
                    ));
                }
                let first = self.parse_comparison()?;
                if matches!(self.peek().kind, TokenKind::Comma) {
                    let mut items = vec![first];
                    while matches!(self.peek().kind, TokenKind::Comma) {
                        self.advance();
                        if self.lenient
                            && matches!(self.peek().kind, TokenKind::RParen | TokenKind::Eof)
                        {
                            items.push(self.placeholder_here());
                            break;
                        }
                        items.push(self.parse_comparison()?);
                    }
                    let close_span = if matches!(self.peek().kind, TokenKind::RParen) {
                        self.advance().span
                    } else if self.lenient {
                        self.peek().span
                    } else {
                        return Err(MathError::new(
                            "Expected `)` to close tuple",
                            self.peek().span,
                        ));
                    };
                    Ok(SpannedExpr::new(
                        MathExpr::Tuple(items),
                        open.span.merge(close_span),
                    ))
                } else {
                    let close_span = if matches!(self.peek().kind, TokenKind::RParen) {
                        self.advance().span
                    } else if self.lenient {
                        self.peek().span
                    } else {
                        return Err(MathError::new(
                            "Expected `)` to close parenthesized group",
                            self.peek().span,
                        ));
                    };
                    Ok(SpannedExpr::new(
                        MathExpr::Group(Box::new(first)),
                        open.span.merge(close_span),
                    ))
                }
            }
            TokenKind::Slash if self.lenient => {
                // Leading `/` alone or `/den` -> immediate fraction with empty numerator slot `□`
                let slash = self.advance();
                let den = if self.is_structural_end()
                    || matches!(self.peek().kind, TokenKind::Plus | TokenKind::Minus)
                {
                    SpannedExpr::new(MathExpr::Placeholder, slash.span)
                } else {
                    self.parse_unary()?
                };
                let span = slash.span.merge(den.span);
                Ok(SpannedExpr::new(
                    MathExpr::Fraction {
                        num: Box::new(SpannedExpr::new(MathExpr::Placeholder, slash.span)),
                        den: Box::new(den),
                    },
                    span,
                ))
            }
            TokenKind::Pipe => {
                let open = self.advance();
                if self.lenient && matches!(self.peek().kind, TokenKind::Pipe | TokenKind::Eof) {
                    let close_span = if matches!(self.peek().kind, TokenKind::Pipe) {
                        self.advance().span
                    } else {
                        open.span
                    };
                    return Ok(SpannedExpr::new(
                        MathExpr::Abs(Box::new(SpannedExpr::new(MathExpr::Placeholder, open.span))),
                        open.span.merge(close_span),
                    ));
                }
                self.in_abs_depth += 1;
                let inner_res = self.parse_comparison();
                self.in_abs_depth -= 1;
                let inner = inner_res?;
                let close_span = if matches!(self.peek().kind, TokenKind::Pipe) {
                    self.advance().span
                } else if self.lenient {
                    self.peek().span
                } else {
                    return Err(MathError::new(
                        "Expected closing `|` for absolute value",
                        self.peek().span,
                    ));
                };
                Ok(SpannedExpr::new(
                    MathExpr::Abs(Box::new(inner)),
                    open.span.merge(close_span),
                ))
            }
            TokenKind::LBrace => {
                if self.brace_has_top_level_colon() {
                    self.parse_piecewise()
                } else {
                    self.parse_braced_group()
                }
            }
            TokenKind::Eof => Err(MathError::new("Unexpected end of expression", tok.span)),
            _ => Err(MathError::new(
                format!("Unexpected token `{:?}` while parsing expression", tok.kind),
                tok.span,
            )),
        }
    }

    fn parse_latex_command(&mut self, cmd: &str, cmd_span: Span) -> Result<SpannedExpr, MathError> {
        match cmd {
            "square" if self.lenient => Ok(SpannedExpr::new(MathExpr::Placeholder, cmd_span)),
            "frac" => {
                let num = self.parse_script_atom()?;
                let den = self.parse_script_atom()?;
                // Detect `\frac{d}{dx}` or `\frac{d^2}{dx^2}`
                if let Some((var, order)) = detect_derivative_frac(&num, &den) {
                    let body = if self.lenient && self.is_structural_end() {
                        self.placeholder_here()
                    } else {
                        self.parse_unary()?
                    };
                    let span = cmd_span.merge(body.span);
                    return Ok(SpannedExpr::new(
                        MathExpr::Derivative {
                            var,
                            order,
                            body: Box::new(body),
                        },
                        span,
                    ));
                }
                let span = cmd_span.merge(den.span);
                Ok(SpannedExpr::new(
                    MathExpr::Fraction {
                        num: Box::new(num),
                        den: Box::new(den),
                    },
                    span,
                ))
            }
            "sqrt" => {
                let index = if matches!(self.peek().kind, TokenKind::LBracket) {
                    self.advance();
                    let idx_expr = if self.lenient
                        && matches!(self.peek().kind, TokenKind::RBracket | TokenKind::Eof)
                    {
                        self.placeholder_here()
                    } else {
                        self.parse_comparison()?
                    };
                    if matches!(self.peek().kind, TokenKind::RBracket) {
                        self.advance();
                    } else if !self.lenient {
                        return Err(MathError::new(
                            "Expected `]` after radical index",
                            self.peek().span,
                        ));
                    }
                    Some(Box::new(idx_expr))
                } else {
                    None
                };
                let radicand = self.parse_script_atom()?;
                let span = cmd_span.merge(radicand.span);
                Ok(SpannedExpr::new(
                    MathExpr::Sqrt {
                        index,
                        radicand: Box::new(radicand),
                    },
                    span,
                ))
            }
            "bar" | "hat" | "tilde" | "vec" => {
                let kind = match cmd {
                    "bar" => AccentKind::Bar,
                    "hat" => AccentKind::Hat,
                    "tilde" => AccentKind::Tilde,
                    _ => AccentKind::Vec,
                };
                let inner = self.parse_script_atom()?;
                let span = cmd_span.merge(inner.span);
                Ok(SpannedExpr::new(
                    MathExpr::Accent {
                        kind,
                        inner: Box::new(inner),
                    },
                    span,
                ))
            }
            "text" | "operatorname" => {
                let inner = self.parse_script_atom()?;
                Ok(inner)
            }
            "sum" | "prod" => {
                let is_sum = cmd == "sum";
                let (var, lower, upper) = self.parse_big_op_bounds(cmd_span, true)?;
                let body = if self.lenient && self.is_structural_end() {
                    self.placeholder_here()
                } else {
                    self.parse_multiplicative()?
                };
                let span = cmd_span.merge(body.span);
                let node = if is_sum {
                    MathExpr::Sum {
                        var,
                        lower: Box::new(lower),
                        upper: Box::new(upper),
                        body: Box::new(body),
                    }
                } else {
                    MathExpr::Prod {
                        var,
                        lower: Box::new(lower),
                        upper: Box::new(upper),
                        body: Box::new(body),
                    }
                };
                Ok(SpannedExpr::new(node, span))
            }
            "int" => {
                let (_default_var, lower, upper) = self.parse_big_op_bounds(cmd_span, false)?;
                let body = if self.lenient && self.is_structural_end() {
                    self.placeholder_here()
                } else {
                    self.in_integral_depth += 1;
                    let body_res = self.parse_additive();
                    self.in_integral_depth -= 1;
                    body_res?
                };

                // Consume optional `dx` or `d x`
                let (var, end_span) = self.consume_optional_differential(body.span);
                let span = cmd_span.merge(end_span);
                Ok(SpannedExpr::new(
                    MathExpr::Integral {
                        var,
                        lower: Box::new(lower),
                        upper: Box::new(upper),
                        body: Box::new(body),
                    },
                    span,
                ))
            }
            other => Err(MathError::new(
                format!("Unsupported LaTeX command `\\{}`", other),
                cmd_span,
            )),
        }
    }

    fn parse_big_op_bounds(
        &mut self,
        op_span: Span,
        expect_var_eq: bool,
    ) -> Result<(String, SpannedExpr, SpannedExpr), MathError> {
        let mut lower_opt: Option<(String, SpannedExpr)> = None;
        let mut upper_opt: Option<SpannedExpr> = None;

        // Allow `_` and `^` in either order
        for _ in 0..2 {
            if matches!(self.peek().kind, TokenKind::Underscore) && lower_opt.is_none() {
                self.advance();
                if expect_var_eq {
                    let (v, lo) = self.parse_indexed_lower_bound()?;
                    lower_opt = Some((v, lo));
                } else {
                    let lo = self.parse_script_atom()?;
                    lower_opt = Some(("x".to_string(), lo));
                }
            } else if matches!(self.peek().kind, TokenKind::Caret) && upper_opt.is_none() {
                self.advance();
                let hi = self.parse_script_atom()?;
                upper_opt = Some(hi);
            }
        }

        let (var, lower) = match lower_opt {
            Some(pair) => pair,
            None if self.lenient => (
                if expect_var_eq { "i" } else { "x" }.to_string(),
                SpannedExpr::new(MathExpr::Placeholder, op_span),
            ),
            None => {
                return Err(MathError::new(
                    "Expected lower bound `_{...}` on big operator",
                    op_span,
                ));
            }
        };
        let upper = match upper_opt {
            Some(hi) => hi,
            None if self.lenient => SpannedExpr::new(MathExpr::Placeholder, op_span),
            None => {
                return Err(MathError::new(
                    "Expected upper bound `^{...}` on big operator",
                    op_span,
                ));
            }
        };
        Ok((var, lower, upper))
    }

    fn parse_indexed_lower_bound(&mut self) -> Result<(String, SpannedExpr), MathError> {
        let braced = matches!(self.peek().kind, TokenKind::LBrace);
        if braced {
            self.advance();
        }
        if self.lenient && matches!(self.peek().kind, TokenKind::RBrace | TokenKind::Eof) {
            if braced && matches!(self.peek().kind, TokenKind::RBrace) {
                self.advance();
            }
            return Ok(("i".to_string(), self.placeholder_here()));
        }
        let first = self.parse_additive()?;
        let result = if matches!(self.peek().kind, TokenKind::Eq) {
            self.advance();
            let var = first
                .as_canonical_ident()
                .ok_or_else(|| MathError::new("Expected index variable before `=`", first.span))?;
            let start_expr =
                if self.lenient && matches!(self.peek().kind, TokenKind::RBrace | TokenKind::Eof) {
                    self.placeholder_here()
                } else {
                    self.parse_additive()?
                };
            (var, start_expr)
        } else {
            ("i".to_string(), first)
        };
        if braced {
            if matches!(self.peek().kind, TokenKind::RBrace) {
                self.advance();
            } else if !self.lenient {
                return Err(MathError::new(
                    "Expected `}` after lower bound",
                    self.peek().span,
                ));
            }
        }
        Ok(result)
    }

    fn consume_optional_differential(&mut self, fallback_span: Span) -> (String, Span) {
        if let TokenKind::Ident(ref s) = self.peek().kind {
            if s == "d" {
                if let TokenKind::Ident(ref v) = self.peek_ahead(1).kind {
                    let v = v.clone();
                    self.advance();
                    let v_tok = self.advance();
                    return (v, v_tok.span);
                }
            } else if let Some(rest) = s.strip_prefix('d') {
                if !rest.is_empty() && !is_builtin_function(s) {
                    let var = rest.to_string();
                    let tok = self.advance();
                    return (var, tok.span);
                }
            }
        }
        ("x".to_string(), fallback_span)
    }
}

fn detect_derivative_frac(num: &SpannedExpr, den: &SpannedExpr) -> Option<(String, u8)> {
    // `\frac{d}{dx}` or `\frac{d}{d x}`
    if matches!(&num.expr, MathExpr::Ident(s) if s == "d") {
        if let MathExpr::Ident(ref s) = den.expr {
            if let Some(var) = s.strip_prefix('d') {
                if !var.is_empty() && !is_builtin_function(s) {
                    return Some((var.to_string(), 1));
                }
            }
        }
        if let MathExpr::Binary {
            op: BinOp::ImplicitMul | BinOp::Mul,
            lhs,
            rhs,
        } = &den.expr
        {
            if matches!(&lhs.expr, MathExpr::Ident(s) if s == "d") {
                if let Some(var) = rhs.as_canonical_ident() {
                    return Some((var, 1));
                }
            }
        }
    }
    // `\frac{d^2}{dx^2}`
    if let MathExpr::Binary {
        op: BinOp::Pow,
        lhs,
        rhs,
    } = &num.expr
    {
        if matches!(&lhs.expr, MathExpr::Ident(s) if s == "d") {
            if let MathExpr::Number(n) = rhs.expr {
                let order = n.round().clamp(1.0_f64, 4.0_f64) as u8;
                if let MathExpr::Binary {
                    op: BinOp::Pow,
                    lhs: d_lhs,
                    ..
                } = &den.expr
                {
                    if let MathExpr::Ident(ref s) = d_lhs.expr {
                        if let Some(var) = s.strip_prefix('d') {
                            if !var.is_empty() {
                                return Some((var.to_string(), order));
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Checks if an expression mentions any identifier in `targets`.
pub fn expr_mentions_any(expr: &SpannedExpr, targets: &[&str]) -> bool {
    match &expr.expr {
        MathExpr::Number(_) => false,
        MathExpr::Ident(name) => targets.contains(&name.as_str()),
        MathExpr::Subscript { base, sub } => {
            if let Some(canon) = expr.as_canonical_ident() {
                if targets.contains(&canon.as_str()) {
                    return true;
                }
            }
            expr_mentions_any(base, targets) || expr_mentions_any(sub, targets)
        }
        MathExpr::Accent { inner, .. }
        | MathExpr::Group(inner)
        | MathExpr::Abs(inner)
        | MathExpr::Unary { expr: inner, .. } => expr_mentions_any(inner, targets),
        MathExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_any(lhs, targets) || expr_mentions_any(rhs, targets)
        }
        MathExpr::Fraction { num, den } => {
            expr_mentions_any(num, targets) || expr_mentions_any(den, targets)
        }
        MathExpr::Sqrt { index, radicand } => {
            index
                .as_ref()
                .map(|i| expr_mentions_any(i, targets))
                .unwrap_or(false)
                || expr_mentions_any(radicand, targets)
        }
        MathExpr::Call { args, .. } | MathExpr::Tuple(args) => {
            args.iter().any(|a| expr_mentions_any(a, targets))
        }
        MathExpr::Derivative { var, body, .. } => {
            targets.contains(&var.as_str()) || expr_mentions_any(body, targets)
        }
        MathExpr::Integral {
            var,
            lower,
            upper,
            body,
        }
        | MathExpr::Sum {
            var,
            lower,
            upper,
            body,
        }
        | MathExpr::Prod {
            var,
            lower,
            upper,
            body,
        } => {
            expr_mentions_any(lower, targets)
                || expr_mentions_any(upper, targets)
                || (!targets.contains(&var.as_str()) && expr_mentions_any(body, targets))
        }
        MathExpr::ComparisonChain { first, rest } => {
            expr_mentions_any(first, targets)
                || rest.iter().any(|(_, e)| expr_mentions_any(e, targets))
        }
        MathExpr::Piecewise(branches) => branches.iter().any(|b| {
            b.condition
                .as_ref()
                .map(|c| expr_mentions_any(c, targets))
                .unwrap_or(false)
                || expr_mentions_any(&b.value, targets)
        }),
        MathExpr::Restricted {
            inner,
            restrictions,
        } => {
            expr_mentions_any(inner, targets)
                || restrictions.iter().any(|r| expr_mentions_any(r, targets))
        }
        MathExpr::Placeholder => false,
        MathExpr::MemberAccess { .. } => false,
        MathExpr::Index { target, index } => {
            expr_mentions_any(target, targets) || expr_mentions_any(index, targets)
        }
    }
}
