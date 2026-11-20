use super::ast::{
    AccentKind, BinOp, CmpOp, MathError, MathExpr, MathStatement, PiecewiseBranch, Span,
    SpannedExpr, UnaryOp,
};
use super::lexer::{Token, TokenKind, tokenize};
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
    let mut parser = Parser::new(tokens, None);
    let expr = parser.parse_comparison()?;
    parser.expect_eof()?;
    Ok(expr)
}

pub fn parse_expr_with_known_fns(
    input: &str,
    known_fns: &HashSet<String>,
) -> Result<SpannedExpr, MathError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens, Some(known_fns));
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
    let mut parser = Parser::new(tokens, known_fns);
    let stmt = parser.parse_top_level_statement()?;
    parser.expect_eof()?;
    Ok(stmt)
}

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    known_fns: Option<&'a HashSet<String>>,
    in_abs_depth: usize,
    in_integral_depth: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token>, known_fns: Option<&'a HashSet<String>>) -> Self {
        Self {
            tokens,
            pos: 0,
            known_fns,
            in_abs_depth: 0,
            in_integral_depth: 0,
        }
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
                let rhs = self.parse_comparison()?;

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
                            if name == "r" && params.len() == 1 && (params[0] == "theta" || params[0] == "t") {
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
                    if !expr_mentions_any(&rhs, &["x", "y", "t", "theta"]) {
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
                let rhs = self.parse_additive()?;
                // Check if chained comparison (e.g. `0 <= x <= 10`) -> return as Expr(ComparisonChain)
                if self.peek_cmp_op().is_some() {
                    let mut rest = vec![(op, rhs)];
                    while self.peek_cmp_op().is_some() {
                        let next_op = self.consume_cmp_op()?;
                        let next_rhs = self.parse_additive()?;
                        rest.push((next_op, next_rhs));
                    }
                    let span = lhs.span.merge(rest.last().map(|(_, e)| e.span).unwrap_or(lhs.span));
                    return Ok(MathStatement::Expr(SpannedExpr::new(
                        MathExpr::ComparisonChain {
                            first: Box::new(lhs),
                            rest,
                        },
                        span,
                    )));
                }
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
                let span = lhs.span.merge(rest.last().map(|(_, e)| e.span).unwrap_or(lhs.span));
                Ok(MathStatement::Expr(SpannedExpr::new(
                    MathExpr::ComparisonChain {
                        first: Box::new(lhs),
                        rest,
                    },
                    span,
                )))
            }
            _ => {
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
        if self.peek_cmp_op().is_none() {
            return Ok(first);
        }
        let mut rest = Vec::new();
        while self.peek_cmp_op().is_some() {
            let op = self.consume_cmp_op()?;
            let rhs = self.parse_additive()?;
            rest.push((op, rhs));
        }
        let end_span = rest.last().map(|(_, e)| e.span).unwrap_or(first.span);
        let span = first.span.merge(end_span);
        Ok(SpannedExpr::new(
            MathExpr::ComparisonChain {
                first: Box::new(first),
                rest,
            },
            span,
        ))
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
            TokenKind::Number(_)
            | TokenKind::Ident(_)
            | TokenKind::LParen
            | TokenKind::LBrace => true,
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
                                        let body = self.parse_unary()?;
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

                    self.advance();
                    let rhs = self.parse_unary()?;
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
            self.advance();
            let exp = self.parse_unary()?;
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
                    self.advance();
                    let sub = self.parse_script_atom()?;
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
        loop {
            args.push(self.parse_comparison()?);
            if matches!(self.peek().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        let close = self.expect(TokenKind::RParen, "Expected `)` to close argument list")?;
        Ok((args, close.span))
    }

    fn parse_script_atom(&mut self) -> Result<SpannedExpr, MathError> {
        if matches!(self.peek().kind, TokenKind::LBrace) {
            self.parse_braced_group()
        } else {
            self.parse_primary()
        }
    }

    fn parse_braced_group(&mut self) -> Result<SpannedExpr, MathError> {
        let open = self.expect(TokenKind::LBrace, "Expected `{`")?;
        let inner = self.parse_comparison()?;
        let close = self.expect(TokenKind::RBrace, "Expected `}`")?;
        Ok(SpannedExpr::new(inner.expr, open.span.merge(close.span)))
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

        let close = self.expect(TokenKind::RBrace, "Expected `}` to close piecewise expression")?;
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
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Number(n) => {
                self.advance();
                Ok(SpannedExpr::new(MathExpr::Number(n), tok.span))
            }
            TokenKind::Ident(ref name) => {
                let name = name.clone();
                self.advance();

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
                let first = self.parse_comparison()?;
                if matches!(self.peek().kind, TokenKind::Comma) {
                    let mut items = vec![first];
                    while matches!(self.peek().kind, TokenKind::Comma) {
                        self.advance();
                        items.push(self.parse_comparison()?);
                    }
                    let close = self.expect(TokenKind::RParen, "Expected `)` to close tuple")?;
                    Ok(SpannedExpr::new(
                        MathExpr::Tuple(items),
                        open.span.merge(close.span),
                    ))
                } else {
                    let close =
                        self.expect(TokenKind::RParen, "Expected `)` to close parenthesized group")?;
                    Ok(SpannedExpr::new(
                        MathExpr::Group(Box::new(first)),
                        open.span.merge(close.span),
                    ))
                }
            }
            TokenKind::Pipe => {
                let open = self.advance();
                self.in_abs_depth += 1;
                let inner_res = self.parse_comparison();
                self.in_abs_depth -= 1;
                let inner = inner_res?;
                let close =
                    self.expect(TokenKind::Pipe, "Expected closing `|` for absolute value")?;
                Ok(SpannedExpr::new(
                    MathExpr::Abs(Box::new(inner)),
                    open.span.merge(close.span),
                ))
            }
            TokenKind::LBrace => {
                if self.brace_has_top_level_colon() {
                    self.parse_piecewise()
                } else {
                    self.parse_braced_group()
                }
            }
            TokenKind::Eof => Err(MathError::new(
                "Unexpected end of expression",
                tok.span,
            )),
            _ => Err(MathError::new(
                format!("Unexpected token `{:?}` while parsing expression", tok.kind),
                tok.span,
            )),
        }
    }

    fn parse_latex_command(&mut self, cmd: &str, cmd_span: Span) -> Result<SpannedExpr, MathError> {
        match cmd {
            "frac" => {
                let num = self.parse_script_atom()?;
                let den = self.parse_script_atom()?;
                // Detect `\frac{d}{dx}` or `\frac{d^2}{dx^2}`
                if let Some((var, order)) = detect_derivative_frac(&num, &den) {
                    let body = self.parse_unary()?;
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
                    let idx_expr = self.parse_comparison()?;
                    self.expect(TokenKind::RBracket, "Expected `]` after radical index")?;
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
                let body = self.parse_multiplicative()?;
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
                self.in_integral_depth += 1;
                let body_res = self.parse_additive();
                self.in_integral_depth -= 1;
                let body = body_res?;

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

        let (var, lower) = lower_opt.ok_or_else(|| {
            MathError::new(
                "Expected lower bound `_{...}` on big operator",
                op_span,
            )
        })?;
        let upper = upper_opt.ok_or_else(|| {
            MathError::new(
                "Expected upper bound `^{...}` on big operator",
                op_span,
            )
        })?;
        Ok((var, lower, upper))
    }

    fn parse_indexed_lower_bound(&mut self) -> Result<(String, SpannedExpr), MathError> {
        let braced = matches!(self.peek().kind, TokenKind::LBrace);
        if braced {
            self.advance();
        }
        let first = self.parse_additive()?;
        let result = if matches!(self.peek().kind, TokenKind::Eq) {
            self.advance();
            let var = first.as_canonical_ident().ok_or_else(|| {
                MathError::new("Expected index variable before `=`", first.span)
            })?;
            let start_expr = self.parse_additive()?;
            (var, start_expr)
        } else {
            ("i".to_string(), first)
        };
        if braced {
            self.expect(TokenKind::RBrace, "Expected `}` after lower bound")?;
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
        MathExpr::MemberAccess { .. } => false,
        MathExpr::Index { target, index } => {
            expr_mentions_any(target, targets) || expr_mentions_any(index, targets)
        }
    }
}
