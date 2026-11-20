use super::ast::{RError, RSpan};

#[derive(Debug, Clone, PartialEq)]
pub enum RTokenKind {
    Number(f64),
    String(String),
    Ident(String),
    // Keywords
    If,
    Else,
    For,
    In,
    While,
    Function,
    Return,
    Break,
    Next,
    True,
    False,
    Null,
    Na,
    // Operators & punctuation
    AssignLeft,  // <-
    AssignRight, // ->
    Eq,          // =
    Pipe,        // |>
    Tilde,       // ~
    Plus,
    Minus,
    Star,
    Slash,
    Caret,   // ^ or **
    Mod,     // %%
    IntDiv,  // %/%
    MatMul,  // %*%
    Colon,   // :
    Dollar,  // $
    EqEq,    // ==
    Ne,      // !=
    Lt,      // <
    Le,      // <=
    Gt,      // >
    Ge,      // >=
    And,     // &
    AndAnd,  // &&
    Or,      // |
    OrOr,    // ||
    Bang,    // !
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Newline,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RToken {
    pub kind: RTokenKind,
    pub span: RSpan,
}

impl RToken {
    fn new(kind: RTokenKind, start: usize, end: usize) -> Self {
        Self {
            kind,
            span: RSpan::new(start, end),
        }
    }
}

pub fn tokenize_r(input: &str) -> Result<Vec<RToken>, RError> {
    let mut tokens = Vec::new();
    let bytes = input.as_bytes();
    let len = input.len();
    let mut i = 0usize;

    while i < len {
        let b = bytes[i];

        // Spaces / tabs / CR
        if matches!(b, b' ' | b'\t' | b'\r') {
            i += 1;
            continue;
        }

        // Newlines
        if b == b'\n' {
            tokens.push(RToken::new(RTokenKind::Newline, i, i + 1));
            i += 1;
            continue;
        }

        // Line comments `# ...`
        if b == b'#' {
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // String literals `"..."` or `'...'`
        if b == b'"' || b == b'\'' {
            let quote = b;
            let start = i;
            i += 1;
            let mut out = String::new();
            let mut closed = false;
            while i < len {
                let c = bytes[i];
                if c == b'\\' && i + 1 < len {
                    i += 1;
                    match bytes[i] {
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'\\' => out.push('\\'),
                        b'"' => out.push('"'),
                        b'\'' => out.push('\''),
                        other => out.push(other as char),
                    }
                    i += 1;
                } else if c == quote {
                    i += 1;
                    closed = true;
                    break;
                } else {
                    let Some(ch) = input[i..].chars().next() else {
                        break;
                    };
                    out.push(ch);
                    i += ch.len_utf8();
                }
            }
            if !closed {
                return Err(RError::with_span(
                    "Unterminated string literal",
                    RSpan::new(start, i),
                ));
            }
            tokens.push(RToken::new(RTokenKind::String(out), start, i));
            continue;
        }

        // Numbers: digits or `.` followed by digit
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let start = i;
            let mut has_dot = false;
            while i < len {
                if bytes[i].is_ascii_digit() {
                    i += 1;
                } else if bytes[i] == b'.' && !has_dot {
                    has_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            if i < len && matches!(bytes[i], b'e' | b'E') {
                let mut e_i = i + 1;
                if e_i < len && matches!(bytes[e_i], b'+' | b'-') {
                    e_i += 1;
                }
                let d_start = e_i;
                while e_i < len && bytes[e_i].is_ascii_digit() {
                    e_i += 1;
                }
                if e_i > d_start {
                    i = e_i;
                }
            }
            // Optional integer suffix `L` (e.g. `1L`)
            if i < len && bytes[i] == b'L' {
                let num_str = &input[start..i];
                i += 1;
                let v = num_str.parse::<f64>().map_err(|_| {
                    RError::with_span(
                        format!("Invalid numeric literal `{}L`", num_str),
                        RSpan::new(start, i),
                    )
                })?;
                tokens.push(RToken::new(RTokenKind::Number(v), start, i));
                continue;
            }
            let num_str = &input[start..i];
            let v = num_str.parse::<f64>().map_err(|_| {
                RError::with_span(
                    format!("Invalid numeric literal `{}`", num_str),
                    RSpan::new(start, i),
                )
            })?;
            tokens.push(RToken::new(RTokenKind::Number(v), start, i));
            continue;
        }

        // Identifiers & Keywords (R allows `.` and `_` inside identifiers, e.g. `data.frame`, `na.rm`)
        if b.is_ascii_alphabetic() || b == b'_' || b == b'.' {
            let start = i;
            i += 1;
            while i < len
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
            {
                i += 1;
            }
            let ident = &input[start..i];
            let kind = match ident {
                "if" => RTokenKind::If,
                "else" => RTokenKind::Else,
                "for" => RTokenKind::For,
                "in" => RTokenKind::In,
                "while" => RTokenKind::While,
                "function" => RTokenKind::Function,
                "return" => RTokenKind::Return,
                "break" => RTokenKind::Break,
                "next" => RTokenKind::Next,
                "TRUE" | "T" => RTokenKind::True,
                "FALSE" | "F" => RTokenKind::False,
                "NULL" => RTokenKind::Null,
                "NA" | "NA_real_" | "NA_integer_" | "NA_character_" => RTokenKind::Na,
                "Inf" => RTokenKind::Number(f64::INFINITY),
                "NaN" => RTokenKind::Number(f64::NAN),
                _ => RTokenKind::Ident(ident.to_string()),
            };
            tokens.push(RToken::new(kind, start, i));
            continue;
        }

        // Backtick-quoted identifiers (e.g. `` `my var` ``)
        if b == b'`' {
            let start = i;
            i += 1;
            let name_start = i;
            while i < len && bytes[i] != b'`' {
                i += 1;
            }
            let name = input[name_start..i].to_string();
            if i < len {
                i += 1;
            }
            tokens.push(RToken::new(RTokenKind::Ident(name), start, i));
            continue;
        }

        // Special `%...%` infix operators (`%%`, `%/%`, `%*%`)
        if b == b'%' {
            let start = i;
            if i + 1 < len && bytes[i + 1] == b'%' {
                i += 2;
                tokens.push(RToken::new(RTokenKind::Mod, start, i));
                continue;
            }
            if i + 2 < len && bytes[i + 1] == b'/' && bytes[i + 2] == b'%' {
                i += 3;
                tokens.push(RToken::new(RTokenKind::IntDiv, start, i));
                continue;
            }
            if i + 2 < len && bytes[i + 1] == b'*' && bytes[i + 2] == b'%' {
                i += 3;
                tokens.push(RToken::new(RTokenKind::MatMul, start, i));
                continue;
            }
            return Err(RError::with_span(
                "Unsupported `%` operator (supported: `%%`, `%/%`, `%*%`)",
                RSpan::new(start, start + 1),
            ));
        }

        // Multi-char & single-char operators
        let start = i;
        let next = if i + 1 < len { bytes[i + 1] } else { 0 };
        match (b, next) {
            (b'<', b'-') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::AssignLeft, start, i));
            }
            (b'<', b'=') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::Le, start, i));
            }
            (b'>', b'=') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::Ge, start, i));
            }
            (b'-', b'>') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::AssignRight, start, i));
            }
            (b'|', b'>') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::Pipe, start, i));
            }
            (b'|', b'|') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::OrOr, start, i));
            }
            (b'&', b'&') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::AndAnd, start, i));
            }
            (b'=', b'=') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::EqEq, start, i));
            }
            (b'!', b'=') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::Ne, start, i));
            }
            (b'*', b'*') => {
                i += 2;
                tokens.push(RToken::new(RTokenKind::Caret, start, i));
            }
            _ => {
                i += 1;
                let kind = match b {
                    b'+' => RTokenKind::Plus,
                    b'-' => RTokenKind::Minus,
                    b'*' => RTokenKind::Star,
                    b'/' => RTokenKind::Slash,
                    b'^' => RTokenKind::Caret,
                    b':' => RTokenKind::Colon,
                    b'$' => RTokenKind::Dollar,
                    b'~' => RTokenKind::Tilde,
                    b'=' => RTokenKind::Eq,
                    b'<' => RTokenKind::Lt,
                    b'>' => RTokenKind::Gt,
                    b'&' => RTokenKind::And,
                    b'|' => RTokenKind::Or,
                    b'!' => RTokenKind::Bang,
                    b'(' => RTokenKind::LParen,
                    b')' => RTokenKind::RParen,
                    b'[' => RTokenKind::LBracket,
                    b']' => RTokenKind::RBracket,
                    b'{' => RTokenKind::LBrace,
                    b'}' => RTokenKind::RBrace,
                    b',' => RTokenKind::Comma,
                    b';' => RTokenKind::Semicolon,
                    other => {
                        let ch = input[start..].chars().next().unwrap_or(other as char);
                        return Err(RError::with_span(
                            format!("Unexpected character `{}` in R input", ch),
                            RSpan::new(start, start + ch.len_utf8()),
                        ));
                    }
                };
                tokens.push(RToken::new(kind, start, i));
            }
        }
    }

    tokens.push(RToken::new(RTokenKind::Eof, len, len));
    Ok(tokens)
}
