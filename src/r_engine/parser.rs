use super::ast::{RBinOp, RCallArg, RError, RExpr, RParam, RUnaryOp};
use super::lexer::{RToken, RTokenKind, tokenize_r};

pub fn parse_r_program(input: &str) -> Result<Vec<RExpr>, RError> {
    let tokens = tokenize_r(input)?;
    let mut parser = RParser::new(tokens);
    let mut stmts = Vec::new();
    parser.skip_separators();
    while !parser.is_eof() {
        stmts.push(parser.parse_expr()?);
        parser.skip_separators();
    }
    Ok(stmts)
}

struct RParser {
    tokens: Vec<RToken>,
    pos: usize,
}

impl RParser {
    fn new(tokens: Vec<RToken>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &RToken {
        self.tokens
            .get(self.pos)
            .unwrap_or_else(|| &self.tokens[self.tokens.len() - 1])
    }

    fn peek_ahead(&self, offset: usize) -> &RToken {
        let idx = (self.pos + offset).min(self.tokens.len().saturating_sub(1));
        &self.tokens[idx]
    }

    fn advance(&mut self) -> RToken {
        let tok = self.peek().clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn is_eof(&self) -> bool {
        matches!(self.peek().kind, RTokenKind::Eof)
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek().kind, RTokenKind::Newline) {
            self.advance();
        }
    }

    fn skip_separators(&mut self) {
        while matches!(self.peek().kind, RTokenKind::Newline | RTokenKind::Semicolon) {
            self.advance();
        }
    }

    fn expect(&mut self, expected: RTokenKind, msg: &str) -> Result<RToken, RError> {
        if self.peek().kind == expected {
            Ok(self.advance())
        } else {
            Err(RError::with_span(msg, self.peek().span))
        }
    }

    fn parse_expr(&mut self) -> Result<RExpr, RError> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> Result<RExpr, RError> {
        let lhs = self.parse_formula()?;
        match self.peek().kind {
            RTokenKind::AssignLeft | RTokenKind::Eq => {
                self.advance();
                self.skip_newlines();
                let rhs = self.parse_assignment()?;
                Ok(RExpr::Assign {
                    target: Box::new(lhs),
                    value: Box::new(rhs),
                })
            }
            RTokenKind::AssignRight => {
                self.advance();
                self.skip_newlines();
                let target = self.parse_assignment()?;
                Ok(RExpr::Assign {
                    target: Box::new(target),
                    value: Box::new(lhs),
                })
            }
            _ => Ok(lhs),
        }
    }

    fn parse_formula(&mut self) -> Result<RExpr, RError> {
        if matches!(self.peek().kind, RTokenKind::Tilde) {
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_or()?;
            return Ok(RExpr::Formula {
                lhs: None,
                rhs: Box::new(rhs),
            });
        }
        let lhs = self.parse_or()?;
        if matches!(self.peek().kind, RTokenKind::Tilde) {
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_or()?;
            return Ok(RExpr::Formula {
                lhs: Some(Box::new(lhs)),
                rhs: Box::new(rhs),
            });
        }
        Ok(lhs)
    }

    fn parse_or(&mut self) -> Result<RExpr, RError> {
        let mut lhs = self.parse_and()?;
        loop {
            let op = match self.peek().kind {
                RTokenKind::Or => RBinOp::Or,
                RTokenKind::OrOr => RBinOp::ShortOr,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_and()?;
            lhs = RExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<RExpr, RError> {
        let mut lhs = self.parse_not()?;
        loop {
            let op = match self.peek().kind {
                RTokenKind::And => RBinOp::And,
                RTokenKind::AndAnd => RBinOp::ShortAnd,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_not()?;
            lhs = RExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<RExpr, RError> {
        if matches!(self.peek().kind, RTokenKind::Bang) {
            self.advance();
            let inner = self.parse_not()?;
            return Ok(RExpr::Unary {
                op: RUnaryOp::Not,
                expr: Box::new(inner),
            });
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<RExpr, RError> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = match self.peek().kind {
                RTokenKind::EqEq => RBinOp::Eq,
                RTokenKind::Ne => RBinOp::Ne,
                RTokenKind::Lt => RBinOp::Lt,
                RTokenKind::Le => RBinOp::Le,
                RTokenKind::Gt => RBinOp::Gt,
                RTokenKind::Ge => RBinOp::Ge,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_additive()?;
            lhs = RExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<RExpr, RError> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.peek().kind {
                RTokenKind::Plus => RBinOp::Add,
                RTokenKind::Minus => RBinOp::Sub,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_multiplicative()?;
            lhs = RExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<RExpr, RError> {
        let mut lhs = self.parse_pipe()?;
        loop {
            let op = match self.peek().kind {
                RTokenKind::Star => RBinOp::Mul,
                RTokenKind::Slash => RBinOp::Div,
                RTokenKind::Mod => RBinOp::Mod,
                RTokenKind::IntDiv => RBinOp::IntDiv,
                RTokenKind::MatMul => RBinOp::MatMul,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_pipe()?;
            lhs = RExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_pipe(&mut self) -> Result<RExpr, RError> {
        let mut lhs = self.parse_colon()?;
        while matches!(self.peek().kind, RTokenKind::Pipe) {
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_colon()?;
            lhs = match rhs {
                RExpr::Call { callee, mut args } => {
                    args.insert(
                        0,
                        RCallArg {
                            name: None,
                            value: Some(lhs),
                        },
                    );
                    RExpr::Call { callee, args }
                }
                callee => RExpr::Call {
                    callee: Box::new(callee),
                    args: vec![RCallArg {
                        name: None,
                        value: Some(lhs),
                    }],
                },
            };
        }
        Ok(lhs)
    }

    fn parse_colon(&mut self) -> Result<RExpr, RError> {
        let mut lhs = self.parse_unary()?;
        while matches!(self.peek().kind, RTokenKind::Colon) {
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_unary()?;
            lhs = RExpr::Binary {
                op: RBinOp::Colon,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<RExpr, RError> {
        match self.peek().kind {
            RTokenKind::Minus => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(RExpr::Unary {
                    op: RUnaryOp::Neg,
                    expr: Box::new(operand),
                })
            }
            RTokenKind::Plus => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(RExpr::Unary {
                    op: RUnaryOp::Pos,
                    expr: Box::new(operand),
                })
            }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> Result<RExpr, RError> {
        let base = self.parse_postfix()?;
        if matches!(self.peek().kind, RTokenKind::Caret) {
            self.advance();
            self.skip_newlines();
            let exp = self.parse_unary()?;
            Ok(RExpr::Binary {
                op: RBinOp::Pow,
                lhs: Box::new(base),
                rhs: Box::new(exp),
            })
        } else {
            Ok(base)
        }
    }

    fn parse_postfix(&mut self) -> Result<RExpr, RError> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek().kind {
                RTokenKind::LParen => {
                    let args = self.parse_call_args()?;
                    expr = RExpr::Call {
                        callee: Box::new(expr),
                        args,
                    };
                }
                RTokenKind::LBracket => {
                    self.advance();
                    // Check for `[[ ... ]]`
                    if matches!(self.peek().kind, RTokenKind::LBracket) {
                        self.advance();
                        self.skip_newlines();
                        let idx = self.parse_expr()?;
                        self.skip_newlines();
                        self.expect(RTokenKind::RBracket, "Expected `]]`")?;
                        self.expect(RTokenKind::RBracket, "Expected closing `]` of `[[`")?;
                        expr = RExpr::DoubleIndex {
                            target: Box::new(expr),
                            index: Box::new(idx),
                        };
                    } else {
                        self.skip_newlines();
                        let mut indices = Vec::new();
                        if !matches!(self.peek().kind, RTokenKind::RBracket) {
                            loop {
                                self.skip_newlines();
                                if matches!(
                                    self.peek().kind,
                                    RTokenKind::Comma | RTokenKind::RBracket
                                ) {
                                    indices.push(None);
                                } else {
                                    indices.push(Some(self.parse_expr()?));
                                }
                                self.skip_newlines();
                                if matches!(self.peek().kind, RTokenKind::Comma) {
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(RTokenKind::RBracket, "Expected `]`")?;
                        expr = RExpr::Index {
                            target: Box::new(expr),
                            indices,
                        };
                    }
                }
                RTokenKind::Dollar => {
                    self.advance();
                    let tok = self.advance();
                    let field = match tok.kind {
                        RTokenKind::Ident(s) | RTokenKind::String(s) => s,
                        _ => {
                            return Err(RError::with_span(
                                "Expected field name after `$`",
                                tok.span,
                            ));
                        }
                    };
                    expr = RExpr::Member {
                        target: Box::new(expr),
                        field,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_call_args(&mut self) -> Result<Vec<RCallArg>, RError> {
        self.expect(RTokenKind::LParen, "Expected `(`")?;
        self.skip_newlines();
        let mut args = Vec::new();
        if matches!(self.peek().kind, RTokenKind::RParen) {
            self.advance();
            return Ok(args);
        }
        loop {
            self.skip_newlines();
            if matches!(self.peek().kind, RTokenKind::RParen) {
                break;
            }
            if matches!(self.peek().kind, RTokenKind::Comma) {
                args.push(RCallArg {
                    name: None,
                    value: None,
                });
                self.advance();
                continue;
            }
            // Named argument: `ident = expr` (ensure it's `Eq`, NOT `EqEq`)
            let is_named = matches!(
                (&self.peek().kind, &self.peek_ahead(1).kind),
                (RTokenKind::Ident(_) | RTokenKind::String(_), RTokenKind::Eq)
            );
            if is_named {
                let name_tok = self.advance();
                let name = match name_tok.kind {
                    RTokenKind::Ident(s) | RTokenKind::String(s) => s,
                    _ => unreachable!(),
                };
                self.advance(); // `=`
                self.skip_newlines();
                let val = self.parse_expr()?;
                args.push(RCallArg {
                    name: Some(name),
                    value: Some(val),
                });
            } else {
                let val = self.parse_expr()?;
                args.push(RCallArg {
                    name: None,
                    value: Some(val),
                });
            }
            self.skip_newlines();
            if matches!(self.peek().kind, RTokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.skip_newlines();
        self.expect(RTokenKind::RParen, "Expected `)` after arguments")?;
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<RExpr, RError> {
        let tok = self.peek().clone();
        match tok.kind {
            RTokenKind::Number(n) => {
                self.advance();
                Ok(RExpr::Number(n))
            }
            RTokenKind::String(s) => {
                self.advance();
                Ok(RExpr::String(s))
            }
            RTokenKind::True => {
                self.advance();
                Ok(RExpr::Bool(true))
            }
            RTokenKind::False => {
                self.advance();
                Ok(RExpr::Bool(false))
            }
            RTokenKind::Null => {
                self.advance();
                Ok(RExpr::Null)
            }
            RTokenKind::Na => {
                self.advance();
                Ok(RExpr::Na)
            }
            RTokenKind::Ident(name) => {
                self.advance();
                Ok(RExpr::Ident(name))
            }
            RTokenKind::Break => {
                self.advance();
                Ok(RExpr::Break)
            }
            RTokenKind::Next => {
                self.advance();
                Ok(RExpr::Next)
            }
            RTokenKind::Return => {
                self.advance();
                if matches!(self.peek().kind, RTokenKind::LParen) {
                    self.advance();
                    self.skip_newlines();
                    let val = if matches!(self.peek().kind, RTokenKind::RParen) {
                        None
                    } else {
                        Some(Box::new(self.parse_expr()?))
                    };
                    self.skip_newlines();
                    self.expect(RTokenKind::RParen, "Expected `)` after `return(...)`")?;
                    Ok(RExpr::Return(val))
                } else {
                    Ok(RExpr::Return(None))
                }
            }
            RTokenKind::LParen => {
                self.advance();
                self.skip_newlines();
                let inner = self.parse_expr()?;
                self.skip_newlines();
                self.expect(RTokenKind::RParen, "Expected `)`")?;
                Ok(inner)
            }
            RTokenKind::LBrace => {
                self.advance();
                let mut stmts = Vec::new();
                self.skip_separators();
                while !matches!(self.peek().kind, RTokenKind::RBrace | RTokenKind::Eof) {
                    stmts.push(self.parse_expr()?);
                    self.skip_separators();
                }
                self.expect(RTokenKind::RBrace, "Expected `}` to close block")?;
                Ok(RExpr::Block(stmts))
            }
            RTokenKind::If => {
                self.advance();
                self.skip_newlines();
                self.expect(RTokenKind::LParen, "Expected `(` after `if`")?;
                self.skip_newlines();
                let cond = self.parse_expr()?;
                self.skip_newlines();
                self.expect(RTokenKind::RParen, "Expected `)` after `if` condition")?;
                self.skip_newlines();
                let then_branch = self.parse_expr()?;

                // Look ahead across newlines for `else`
                let saved = self.pos;
                self.skip_newlines();
                let else_branch = if matches!(self.peek().kind, RTokenKind::Else) {
                    self.advance();
                    self.skip_newlines();
                    Some(Box::new(self.parse_expr()?))
                } else {
                    self.pos = saved;
                    None
                };

                Ok(RExpr::If {
                    cond: Box::new(cond),
                    then_branch: Box::new(then_branch),
                    else_branch,
                })
            }
            RTokenKind::For => {
                self.advance();
                self.skip_newlines();
                self.expect(RTokenKind::LParen, "Expected `(` after `for`")?;
                self.skip_newlines();
                let var_tok = self.advance();
                let var = match var_tok.kind {
                    RTokenKind::Ident(s) => s,
                    _ => {
                        return Err(RError::with_span(
                            "Expected loop variable name after `for (`",
                            var_tok.span,
                        ));
                    }
                };
                self.expect(RTokenKind::In, "Expected `in` in `for` loop")?;
                self.skip_newlines();
                let iter = self.parse_expr()?;
                self.skip_newlines();
                self.expect(RTokenKind::RParen, "Expected `)` after `for` header")?;
                self.skip_newlines();
                let body = self.parse_expr()?;
                Ok(RExpr::For {
                    var,
                    iter: Box::new(iter),
                    body: Box::new(body),
                })
            }
            RTokenKind::While => {
                self.advance();
                self.skip_newlines();
                self.expect(RTokenKind::LParen, "Expected `(` after `while`")?;
                self.skip_newlines();
                let cond = self.parse_expr()?;
                self.skip_newlines();
                self.expect(RTokenKind::RParen, "Expected `)` after `while` condition")?;
                self.skip_newlines();
                let body = self.parse_expr()?;
                Ok(RExpr::While {
                    cond: Box::new(cond),
                    body: Box::new(body),
                })
            }
            RTokenKind::Function => {
                self.advance();
                self.skip_newlines();
                self.expect(RTokenKind::LParen, "Expected `(` after `function`")?;
                self.skip_newlines();
                let mut params = Vec::new();
                while !matches!(self.peek().kind, RTokenKind::RParen | RTokenKind::Eof) {
                    let p_tok = self.advance();
                    let name = match p_tok.kind {
                        RTokenKind::Ident(s) => s,
                        _ => {
                            return Err(RError::with_span(
                                "Expected parameter name in `function(...)`",
                                p_tok.span,
                            ));
                        }
                    };
                    let default = if matches!(self.peek().kind, RTokenKind::Eq) {
                        self.advance();
                        self.skip_newlines();
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    params.push(RParam { name, default });
                    self.skip_newlines();
                    if matches!(self.peek().kind, RTokenKind::Comma) {
                        self.advance();
                        self.skip_newlines();
                    } else {
                        break;
                    }
                }
                self.expect(RTokenKind::RParen, "Expected `)` after parameter list")?;
                self.skip_newlines();
                let body = self.parse_expr()?;
                Ok(RExpr::FunctionDef {
                    params,
                    body: Box::new(body),
                })
            }
            _ => Err(RError::with_span(
                format!("Unexpected token `{:?}` in R expression", tok.kind),
                tok.span,
            )),
        }
    }
}
