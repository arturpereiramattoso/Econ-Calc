use super::ast::{MathError, Span, unicode_char_to_greek_name};

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Number(f64),
    Ident(String),
    /// LaTeX command without leading backslash, e.g. `"frac"`, `"sqrt"`, `"sum"`, `"bar"`.
    Command(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Underscore,
    Prime,
    Dollar,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Pipe,
    Comma,
    Colon,
    Semicolon,
    Eq,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Arrow,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, start: usize, end: usize) -> Self {
        Self {
            kind,
            span: Span::new(start, end),
        }
    }
}

/// Tokenizes mathematical and LaTeX-style expressions with accurate byte spans.
pub fn tokenize(input: &str) -> Result<Vec<Token>, MathError> {
    let mut tokens = Vec::new();
    let bytes = input.as_bytes();
    let len = input.len();
    let mut i = 0usize;

    while i < len {
        let Some(ch) = input[i..].chars().next() else {
            break;
        };
        let ch_len = ch.len_utf8();

        if ch.is_whitespace() {
            i += ch_len;
            continue;
        }

        // Numbers: digits or leading `.` followed by digit
        if ch.is_ascii_digit()
            || (ch == '.' && i + 1 < len && bytes[i + 1].is_ascii_digit())
        {
            let start = i;
            let mut has_dot = false;
            while i < len {
                let b = bytes[i];
                if b.is_ascii_digit() {
                    i += 1;
                } else if b == b'.' && !has_dot {
                    has_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }

            // Optional scientific notation: only when `e`/`E` is followed by optional `+/-` + digit(s)
            if i < len && matches!(bytes[i], b'e' | b'E') {
                let mut exp_i = i + 1;
                if exp_i < len && matches!(bytes[exp_i], b'+' | b'-') {
                    exp_i += 1;
                }
                let digits_start = exp_i;
                while exp_i < len && bytes[exp_i].is_ascii_digit() {
                    exp_i += 1;
                }
                if exp_i > digits_start {
                    i = exp_i;
                }
            }

            let raw = &input[start..i];
            let val = raw.parse::<f64>().map_err(|_| {
                MathError::new(format!("Invalid number literal `{}`", raw), Span::new(start, i))
            })?;
            tokens.push(Token::new(TokenKind::Number(val), start, i));
            continue;
        }

        // LaTeX backslash commands: \frac, \alpha, \cdot, \left, etc.
        if ch == '\\' {
            let start = i;
            i += 1;
            if i >= len {
                return Err(MathError::new(
                    "Trailing backslash `\\` without command",
                    Span::new(start, i),
                ));
            }
            let Some(next_ch) = input[i..].chars().next() else {
                break;
            };
            // Single-char LaTeX spacing or escaped braces/pipes
            if !next_ch.is_ascii_alphabetic() {
                i += next_ch.len_utf8();
                match next_ch {
                    ',' | ';' | ':' | '!' | ' ' => {
                        // LaTeX spacing command: skip
                    }
                    '{' => tokens.push(Token::new(TokenKind::LBrace, start, i)),
                    '}' => tokens.push(Token::new(TokenKind::RBrace, start, i)),
                    '|' => tokens.push(Token::new(TokenKind::Pipe, start, i)),
                    _ => {
                        return Err(MathError::new(
                            format!("Unknown LaTeX escape `\\{}`", next_ch),
                            Span::new(start, i),
                        ));
                    }
                }
                continue;
            }

            let cmd_start = i;
            while i < len && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
            let cmd = &input[cmd_start..i];
            match cmd {
                "left" | "right" | "big" | "Big" | "bigg" | "Bigg" | "quad" | "qquad"
                | "displaystyle" | "limits" => {
                    // Pure layout/sizing directives ignored by parser
                }
                "cdot" | "times" | "ast" => {
                    tokens.push(Token::new(TokenKind::Star, start, i));
                }
                "div" => {
                    tokens.push(Token::new(TokenKind::Slash, start, i));
                }
                "le" | "leq" => {
                    tokens.push(Token::new(TokenKind::Le, start, i));
                }
                "ge" | "geq" => {
                    tokens.push(Token::new(TokenKind::Ge, start, i));
                }
                "ne" | "neq" => {
                    tokens.push(Token::new(TokenKind::Ne, start, i));
                }
                "to" | "rightarrow" => {
                    tokens.push(Token::new(TokenKind::Arrow, start, i));
                }
                "frac" | "dfrac" | "tfrac" => {
                    tokens.push(Token::new(TokenKind::Command("frac".to_string()), start, i));
                }
                "sqrt" | "sum" | "prod" | "int" | "bar" | "overline" | "hat" | "tilde"
                | "vec" | "text" | "operatorname" | "square" => {
                    let norm = if cmd == "overline" { "bar" } else { cmd };
                    tokens.push(Token::new(TokenKind::Command(norm.to_string()), start, i));
                }
                "abs" => {
                    tokens.push(Token::new(TokenKind::Ident("abs".to_string()), start, i));
                }
                "arcsin" => {
                    tokens.push(Token::new(TokenKind::Ident("asin".to_string()), start, i));
                }
                "arccos" => {
                    tokens.push(Token::new(TokenKind::Ident("acos".to_string()), start, i));
                }
                "arctan" => {
                    tokens.push(Token::new(TokenKind::Ident("atan".to_string()), start, i));
                }
                "varepsilon" => {
                    tokens.push(Token::new(TokenKind::Ident("epsilon".to_string()), start, i));
                }
                "varphi" => {
                    tokens.push(Token::new(TokenKind::Ident("phi".to_string()), start, i));
                }
                "vartheta" => {
                    tokens.push(Token::new(TokenKind::Ident("theta".to_string()), start, i));
                }
                other => {
                    // Greek letters (\alpha, \beta, \pi, \theta, ...), functions (\sin, \cos, \ln, \log, \exp, ...)
                    tokens.push(Token::new(TokenKind::Ident(other.to_string()), start, i));
                }
            }
            continue;
        }

        // Unicode Greek letters -> canonical ASCII ident names
        if let Some(greek_name) = unicode_char_to_greek_name(ch) {
            let start = i;
            i += ch_len;
            tokens.push(Token::new(
                TokenKind::Ident(greek_name.to_string()),
                start,
                i,
            ));
            continue;
        }

        // Unicode math operators & symbols
        match ch {
            '·' | '×' => {
                tokens.push(Token::new(TokenKind::Star, i, i + ch_len));
                i += ch_len;
                continue;
            }
            '÷' => {
                tokens.push(Token::new(TokenKind::Slash, i, i + ch_len));
                i += ch_len;
                continue;
            }
            '−' => {
                tokens.push(Token::new(TokenKind::Minus, i, i + ch_len));
                i += ch_len;
                continue;
            }
            '≤' => {
                tokens.push(Token::new(TokenKind::Le, i, i + ch_len));
                i += ch_len;
                continue;
            }
            '≥' => {
                tokens.push(Token::new(TokenKind::Ge, i, i + ch_len));
                i += ch_len;
                continue;
            }
            '≠' => {
                tokens.push(Token::new(TokenKind::Ne, i, i + ch_len));
                i += ch_len;
                continue;
            }
            '→' => {
                tokens.push(Token::new(TokenKind::Arrow, i, i + ch_len));
                i += ch_len;
                continue;
            }
            '√' => {
                tokens.push(Token::new(
                    TokenKind::Command("sqrt".to_string()),
                    i,
                    i + ch_len,
                ));
                i += ch_len;
                continue;
            }
            '∑' => {
                tokens.push(Token::new(
                    TokenKind::Command("sum".to_string()),
                    i,
                    i + ch_len,
                ));
                i += ch_len;
                continue;
            }
            '∏' => {
                tokens.push(Token::new(
                    TokenKind::Command("prod".to_string()),
                    i,
                    i + ch_len,
                ));
                i += ch_len;
                continue;
            }
            '∫' => {
                tokens.push(Token::new(
                    TokenKind::Command("int".to_string()),
                    i,
                    i + ch_len,
                ));
                i += ch_len;
                continue;
            }
            'Ȳ' => {
                tokens.push(Token::new(
                    TokenKind::Ident("Y_bar".to_string()),
                    i,
                    i + ch_len,
                ));
                i += ch_len;
                continue;
            }
            '′' => {
                tokens.push(Token::new(TokenKind::Prime, i, i + ch_len));
                i += ch_len;
                continue;
            }
            '□' => {
                tokens.push(Token::new(
                    TokenKind::Command("square".to_string()),
                    i,
                    i + ch_len,
                ));
                i += ch_len;
                continue;
            }
            _ => {}
        }

        // ASCII Identifiers (including underscores when NOT followed by `{`)
        if ch.is_ascii_alphabetic() {
            let start = i;
            i += 1;
            while i < len {
                let b = bytes[i];
                if b.is_ascii_alphanumeric() {
                    i += 1;
                } else if b == b'_' {
                    // Stop before `_{` so `x_{i+1}` parses as Subscript AST node
                    if i + 1 < len && bytes[i + 1] == b'{' {
                        break;
                    }
                    // Also stop before `_` if followed by nothing or non-alphanumeric
                    if i + 1 >= len || !bytes[i + 1].is_ascii_alphanumeric() {
                        break;
                    }
                    i += 1;
                } else {
                    break;
                }
            }
            let ident = &input[start..i];
            tokens.push(Token::new(TokenKind::Ident(ident.to_string()), start, i));
            continue;
        }

        // Multi-character and single-character ASCII punctuation/operators
        let start = i;
        match ch {
            '+' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Plus, start, i));
            }
            '-' => {
                if i + 1 < len && bytes[i + 1] == b'>' {
                    i += 2;
                    tokens.push(Token::new(TokenKind::Arrow, start, i));
                } else {
                    i += 1;
                    tokens.push(Token::new(TokenKind::Minus, start, i));
                }
            }
            '*' => {
                if i + 1 < len && bytes[i + 1] == b'*' {
                    i += 2;
                    tokens.push(Token::new(TokenKind::Caret, start, i));
                } else {
                    i += 1;
                    tokens.push(Token::new(TokenKind::Star, start, i));
                }
            }
            '/' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Slash, start, i));
            }
            '%' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Percent, start, i));
            }
            '^' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Caret, start, i));
            }
            '_' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Underscore, start, i));
            }
            '\'' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Prime, start, i));
            }
            '$' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Dollar, start, i));
            }
            '(' => {
                i += 1;
                tokens.push(Token::new(TokenKind::LParen, start, i));
            }
            ')' => {
                i += 1;
                tokens.push(Token::new(TokenKind::RParen, start, i));
            }
            '[' => {
                i += 1;
                tokens.push(Token::new(TokenKind::LBracket, start, i));
            }
            ']' => {
                i += 1;
                tokens.push(Token::new(TokenKind::RBracket, start, i));
            }
            '{' => {
                i += 1;
                tokens.push(Token::new(TokenKind::LBrace, start, i));
            }
            '}' => {
                i += 1;
                tokens.push(Token::new(TokenKind::RBrace, start, i));
            }
            '|' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Pipe, start, i));
            }
            ',' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Comma, start, i));
            }
            ':' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Colon, start, i));
            }
            ';' => {
                i += 1;
                tokens.push(Token::new(TokenKind::Semicolon, start, i));
            }
            '=' => {
                if i + 1 < len && bytes[i + 1] == b'=' {
                    i += 2;
                    tokens.push(Token::new(TokenKind::EqEq, start, i));
                } else {
                    i += 1;
                    tokens.push(Token::new(TokenKind::Eq, start, i));
                }
            }
            '!' => {
                if i + 1 < len && bytes[i + 1] == b'=' {
                    i += 2;
                    tokens.push(Token::new(TokenKind::Ne, start, i));
                } else {
                    return Err(MathError::new(
                        "Unexpected `!` (use `!=` for not-equal)",
                        Span::new(start, start + 1),
                    ));
                }
            }
            '<' => {
                if i + 1 < len && bytes[i + 1] == b'=' {
                    i += 2;
                    tokens.push(Token::new(TokenKind::Le, start, i));
                } else {
                    i += 1;
                    tokens.push(Token::new(TokenKind::Lt, start, i));
                }
            }
            '>' => {
                if i + 1 < len && bytes[i + 1] == b'=' {
                    i += 2;
                    tokens.push(Token::new(TokenKind::Ge, start, i));
                } else {
                    i += 1;
                    tokens.push(Token::new(TokenKind::Gt, start, i));
                }
            }
            other => {
                return Err(MathError::new(
                    format!("Unexpected character `{}`", other),
                    Span::new(start, start + ch_len),
                ));
            }
        }
    }

    tokens.push(Token::new(TokenKind::Eof, len, len));
    Ok(tokens)
}

/// Lenient tokenizer for live structural typing previews: tolerates a trailing `\` or `!`
/// while the user is mid-keystroke.
pub fn tokenize_lenient(input: &str) -> Result<Vec<Token>, MathError> {
    match tokenize(input) {
        Ok(toks) => Ok(toks),
        Err(err) => {
            // If failure is strictly a trailing `\` or `!` at end-of-input, replace with placeholder
            let trimmed = input.trim_end();
            if let Some(prefix) = trimmed.strip_suffix('\\') {
                let mut toks = tokenize(prefix)?;
                let eof_span = toks.last().map(|t| t.span).unwrap_or(Span::new(prefix.len(), input.len()));
                if matches!(toks.last().map(|t| &t.kind), Some(TokenKind::Eof)) {
                    toks.pop();
                }
                toks.push(Token::new(
                    TokenKind::Command("square".to_string()),
                    prefix.len(),
                    input.len(),
                ));
                toks.push(Token::new(TokenKind::Eof, eof_span.end, input.len()));
                return Ok(toks);
            }
            Err(err)
        }
    }
}
