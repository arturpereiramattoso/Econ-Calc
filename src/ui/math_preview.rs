use crate::math::ast::{
    AccentKind, BinOp, MathError, MathExpr, MathStatement, SpannedExpr, UnaryOp,
    greek_name_to_unicode,
};
use crate::math::parser::{
    parse_expr_with_known_fns, parse_statement_for_preview, parse_statement_with_known_fns,
};
use eframe::egui::{self, Color32, FontFamily, FontId, Painter, Pos2, Rounding, Sense, Stroke, Ui, Vec2};
use std::collections::HashSet;

// =============================================================================
// TEXT MEASUREMENT ABSTRACTION (ENABLES HEADLESS LAYOUT UNIT TESTS)
// =============================================================================

/// Measures glyph runs into `(width, ascent, descent)` in logical points.
pub trait TextMeasurer {
    fn measure(&self, text: &str, size: f32, italic: bool) -> (f32, f32, f32);
}

/// Native `egui::Painter` font galley measurer.
pub struct EguiTextMeasurer<'a> {
    pub painter: &'a Painter,
}

impl TextMeasurer for EguiTextMeasurer<'_> {
    fn measure(&self, text: &str, size: f32, _italic: bool) -> (f32, f32, f32) {
        let font = FontId::new(size.max(6.0_f32), FontFamily::Proportional);
        let galley = self
            .painter
            .layout_no_wrap(text.to_string(), font, Color32::WHITE);
        let rect = galley.rect;
        let h = rect.height().max(size);
        let ascent = h * 0.78_f32;
        let descent = h - ascent;
        (rect.width(), ascent, descent)
    }
}

/// Deterministic mock measurer for unit-testing `MathBox` layout metrics without a window.
#[cfg(test)]
pub struct MockTextMeasurer;

#[cfg(test)]
impl TextMeasurer for MockTextMeasurer {
    fn measure(&self, text: &str, size: f32, _italic: bool) -> (f32, f32, f32) {
        let width = (text.chars().count() as f32) * size * 0.56_f32;
        let ascent = size * 0.78_f32;
        let descent = size * 0.22_f32;
        (width, ascent, descent)
    }
}

// =============================================================================
// VISUAL MATH LAYOUT TREE (`MathBox`)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelimKind {
    Paren,
    Bracket,
    Brace,
    Pipe,
}

#[derive(Debug, Clone)]
pub struct MathBox {
    pub width: f32,
    pub ascent: f32,
    pub descent: f32,
    pub kind: MathBoxKind,
}

impl MathBox {
    pub fn height(&self) -> f32 {
        self.ascent + self.descent
    }

    pub fn text(m: &dyn TextMeasurer, text: impl Into<String>, size: f32, italic: bool) -> Self {
        let text = text.into();
        let (width, ascent, descent) = m.measure(&text, size, italic);
        Self {
            width,
            ascent,
            descent,
            kind: MathBoxKind::Text {
                text,
                size,
                italic,
                color_override: None,
            },
        }
    }

    pub fn colored_text(
        m: &dyn TextMeasurer,
        text: impl Into<String>,
        size: f32,
        italic: bool,
        color: Color32,
    ) -> Self {
        let text = text.into();
        let (width, ascent, descent) = m.measure(&text, size, italic);
        Self {
            width,
            ascent,
            descent,
            kind: MathBoxKind::Text {
                text,
                size,
                italic,
                color_override: Some(color),
            },
        }
    }

    pub fn space(width: f32) -> Self {
        Self {
            width,
            ascent: 0.0_f32,
            descent: 0.0_f32,
            kind: MathBoxKind::Space,
        }
    }

    pub fn row(children: Vec<MathBox>) -> Self {
        let mut width = 0.0_f32;
        let mut ascent = 0.0_f32;
        let mut descent = 0.0_f32;
        for c in &children {
            width += c.width;
            ascent = ascent.max(c.ascent);
            descent = descent.max(c.descent);
        }
        Self {
            width,
            ascent,
            descent,
            kind: MathBoxKind::Row(children),
        }
    }

    pub fn fraction(num: MathBox, den: MathBox, base_size: f32) -> Self {
        let bar_thickness = (base_size * 0.075_f32).clamp(1.0_f32, 2.2_f32);
        let gap = (base_size * 0.14_f32).max(2.0_f32);
        let pad_x = (base_size * 0.18_f32).max(2.5_f32);
        let axis_height = base_size * 0.28_f32;

        let width = num.width.max(den.width) + 2.0_f32 * pad_x;
        let ascent = axis_height + bar_thickness * 0.5_f32 + gap + num.height();
        let descent = (den.height() + gap + bar_thickness * 0.5_f32 - axis_height).max(0.0_f32);

        Self {
            width,
            ascent,
            descent,
            kind: MathBoxKind::Fraction {
                num: Box::new(num),
                den: Box::new(den),
                bar_thickness,
                gap,
                axis_height,
            },
        }
    }

    pub fn script(base: MathBox, sup: Option<MathBox>, sub: Option<MathBox>, base_size: f32) -> Self {
        let gap_x = (base_size * 0.06_f32).max(0.8_f32);
        let sup_shift = sup
            .as_ref()
            .map(|s| (base.ascent * 0.52_f32).max(base_size * 0.36_f32) + s.descent * 0.5_f32)
            .unwrap_or(0.0_f32);
        let sub_shift = sub
            .as_ref()
            .map(|s| base.descent * 0.35_f32 + s.ascent * 0.52_f32)
            .unwrap_or(0.0_f32);

        let sup_w = sup.as_ref().map(|s| s.width).unwrap_or(0.0_f32);
        let sub_w = sub.as_ref().map(|s| s.width).unwrap_or(0.0_f32);
        let script_w = sup_w.max(sub_w);
        let width = base.width + if script_w > 0.0_f32 { gap_x + script_w } else { 0.0_f32 };

        let ascent = sup
            .as_ref()
            .map(|s| base.ascent.max(sup_shift + s.ascent))
            .unwrap_or(base.ascent);
        let descent = sub
            .as_ref()
            .map(|s| base.descent.max(sub_shift + s.descent))
            .unwrap_or(base.descent);

        Self {
            width,
            ascent,
            descent,
            kind: MathBoxKind::Script {
                base: Box::new(base),
                sup: sup.map(Box::new),
                sub: sub.map(Box::new),
                sup_shift,
                sub_shift,
                gap_x,
            },
        }
    }

    pub fn radical(index: Option<MathBox>, radicand: MathBox, base_size: f32) -> Self {
        let bar_thickness = (base_size * 0.075_f32).clamp(1.0_f32, 2.0_f32);
        let bar_gap = (base_size * 0.14_f32).max(2.0_f32);
        let hook_w = (base_size * 0.58_f32).max(7.5_f32);
        let pad_right = (base_size * 0.14_f32).max(2.0_f32);
        let idx_extra_w = index
            .as_ref()
            .map(|i| (i.width - hook_w * 0.45_f32).max(0.0_f32))
            .unwrap_or(0.0_f32);

        let width = idx_extra_w + hook_w + radicand.width + pad_right;
        let idx_top = index
            .as_ref()
            .map(|i| radicand.ascent * 0.35_f32 + i.height() + 1.0_f32)
            .unwrap_or(0.0_f32);
        let ascent = (radicand.ascent + bar_gap + bar_thickness).max(idx_top);
        let descent = radicand.descent + 1.0_f32;

        Self {
            width,
            ascent,
            descent,
            kind: MathBoxKind::Radical {
                index: index.map(Box::new),
                radicand: Box::new(radicand),
                idx_extra_w,
                hook_w,
                bar_gap,
                bar_thickness,
            },
        }
    }

    pub fn delimited(left: DelimKind, right: DelimKind, inner: MathBox, base_size: f32) -> Self {
        let delim_w = match left {
            DelimKind::Pipe => (base_size * 0.28_f32).max(4.0_f32),
            DelimKind::Paren => (base_size * 0.34_f32 + inner.height() * 0.05_f32).clamp(5.0_f32, 11.0_f32),
            DelimKind::Bracket => (base_size * 0.34_f32).max(5.0_f32),
            DelimKind::Brace => (base_size * 0.44_f32).max(6.5_f32),
        };
        let pad_v = 1.5_f32;
        Self {
            width: delim_w * 2.0_f32 + inner.width + 2.0_f32,
            ascent: inner.ascent + pad_v,
            descent: inner.descent + pad_v,
            kind: MathBoxKind::Delimited {
                left,
                right,
                inner: Box::new(inner),
                delim_w,
            },
        }
    }

    pub fn accent(kind: AccentKind, inner: MathBox, base_size: f32) -> Self {
        let gap = (base_size * 0.10_f32).max(1.6_f32);
        let accent_h = match kind {
            AccentKind::Bar => 1.6_f32,
            AccentKind::Hat | AccentKind::Tilde | AccentKind::Vec => (base_size * 0.24_f32).max(3.2_f32),
        };
        Self {
            width: inner.width.max(base_size * 0.45_f32),
            ascent: inner.ascent + gap + accent_h,
            descent: inner.descent,
            kind: MathBoxKind::Accent {
                kind,
                inner: Box::new(inner),
                gap,
                accent_h,
            },
        }
    }

    pub fn big_op(
        m: &dyn TextMeasurer,
        symbol: &'static str,
        lower: Option<MathBox>,
        upper: Option<MathBox>,
        stacked: bool,
        base_size: f32,
    ) -> Self {
        let sym_size = if stacked {
            base_size * 1.48_f32
        } else {
            base_size * 1.65_f32
        };
        let sym_box = MathBox::text(m, symbol, sym_size, false);
        let gap = (base_size * 0.10_f32).max(1.5_f32);

        if stacked {
            let max_w = sym_box
                .width
                .max(lower.as_ref().map(|b| b.width).unwrap_or(0.0_f32))
                .max(upper.as_ref().map(|b| b.width).unwrap_or(0.0_f32));
            let sym_center_shift = base_size * 0.25_f32;
            let sym_half = sym_box.height() * 0.5_f32;
            let base_ascent = sym_center_shift + sym_half;
            let base_descent = (sym_half - sym_center_shift).max(0.0_f32);

            let ascent = base_ascent
                + upper
                    .as_ref()
                    .map(|u| gap + u.height())
                    .unwrap_or(0.0_f32);
            let descent = base_descent
                + lower
                    .as_ref()
                    .map(|l| gap + l.height())
                    .unwrap_or(0.0_f32);

            Self {
                width: max_w + 3.0_f32,
                ascent,
                descent,
                kind: MathBoxKind::BigOp {
                    symbol_box: Box::new(sym_box),
                    lower: lower.map(Box::new),
                    upper: upper.map(Box::new),
                    stacked: true,
                    gap,
                },
            }
        } else {
            // Integral style: bounds placed to upper-right and lower-right of the tall `∫`
            let bounds_w = lower
                .as_ref()
                .map(|l| l.width)
                .unwrap_or(0.0_f32)
                .max(upper.as_ref().map(|u| u.width).unwrap_or(0.0_f32));
            let sym_half = sym_box.height() * 0.5_f32;
            let axis = base_size * 0.26_f32;
            let ascent = (axis + sym_half).max(
                axis + sym_half * 0.65_f32
                    + upper.as_ref().map(|u| u.ascent).unwrap_or(0.0_f32),
            );
            let descent = (sym_half - axis).max(
                (sym_half * 0.65_f32 - axis).max(0.0_f32)
                    + lower.as_ref().map(|l| l.descent).unwrap_or(0.0_f32),
            );
            Self {
                width: sym_box.width + bounds_w + 3.0_f32,
                ascent,
                descent,
                kind: MathBoxKind::BigOp {
                    symbol_box: Box::new(sym_box),
                    lower: lower.map(Box::new),
                    upper: upper.map(Box::new),
                    stacked: false,
                    gap,
                },
            }
        }
    }

    pub fn piecewise(rows: Vec<(MathBox, Option<MathBox>)>, base_size: f32) -> Self {
        let row_gap = (base_size * 0.24_f32).max(3.0_f32);
        let col_gap = (base_size * 0.65_f32).max(8.0_f32);
        let brace_w = (base_size * 0.55_f32).max(8.0_f32);

        let mut max_val_w = 0.0_f32;
        let mut max_cond_w = 0.0_f32;
        let mut total_h = 0.0_f32;
        for (i, (val, cond)) in rows.iter().enumerate() {
            max_val_w = max_val_w.max(val.width);
            if let Some(c) = cond {
                max_cond_w = max_cond_w.max(c.width);
            }
            let row_h = val
                .height()
                .max(cond.as_ref().map(|c| c.height()).unwrap_or(0.0_f32));
            total_h += row_h;
            if i + 1 < rows.len() {
                total_h += row_gap;
            }
        }

        let axis = base_size * 0.28_f32;
        let ascent = total_h * 0.5_f32 + axis;
        let descent = (total_h * 0.5_f32 - axis).max(0.0_f32);
        let width = brace_w
            + 3.0_f32
            + max_val_w
            + if max_cond_w > 0.0_f32 {
                col_gap + max_cond_w
            } else {
                0.0_f32
            };

        Self {
            width,
            ascent,
            descent,
            kind: MathBoxKind::Piecewise {
                rows,
                max_val_w,
                row_gap,
                col_gap,
                brace_w,
            },
        }
    }

    pub fn error_span(m: &dyn TextMeasurer, text: impl Into<String>, size: f32) -> Self {
        let text = text.into();
        let (width, ascent, descent) = m.measure(&text, size, false);
        Self {
            width,
            ascent,
            descent: descent + 2.0_f32,
            kind: MathBoxKind::ErrorSpan { text, size },
        }
    }

    /// Active visual slot (`□`) rendered while typing incomplete structural math
    /// (empty denominator after `/`, empty exponent after `^`, empty subscript after `_`,
    /// empty radicand in `sqrt()`, or missing bounds/body in `\int`, `\sum`, `\prod`).
    pub fn placeholder_slot(size: f32) -> Self {
        let s = size.max(8.0_f32);
        Self {
            width: (s * 0.72_f32).max(8.5_f32),
            ascent: (s * 0.74_f32).max(7.5_f32),
            descent: (s * 0.16_f32).max(1.5_f32),
            kind: MathBoxKind::PlaceholderSlot { size: s },
        }
    }
}

#[derive(Debug, Clone)]
pub enum MathBoxKind {
    Text {
        text: String,
        size: f32,
        #[allow(dead_code)]
        italic: bool,
        color_override: Option<Color32>,
    },
    Space,
    Row(Vec<MathBox>),
    Fraction {
        num: Box<MathBox>,
        den: Box<MathBox>,
        bar_thickness: f32,
        gap: f32,
        axis_height: f32,
    },
    Script {
        base: Box<MathBox>,
        sup: Option<Box<MathBox>>,
        sub: Option<Box<MathBox>>,
        sup_shift: f32,
        sub_shift: f32,
        gap_x: f32,
    },
    Radical {
        index: Option<Box<MathBox>>,
        radicand: Box<MathBox>,
        idx_extra_w: f32,
        hook_w: f32,
        bar_gap: f32,
        bar_thickness: f32,
    },
    Delimited {
        left: DelimKind,
        right: DelimKind,
        inner: Box<MathBox>,
        delim_w: f32,
    },
    Accent {
        kind: AccentKind,
        inner: Box<MathBox>,
        gap: f32,
        accent_h: f32,
    },
    BigOp {
        symbol_box: Box<MathBox>,
        lower: Option<Box<MathBox>>,
        upper: Option<Box<MathBox>>,
        stacked: bool,
        gap: f32,
    },
    Piecewise {
        rows: Vec<(MathBox, Option<MathBox>)>,
        max_val_w: f32,
        row_gap: f32,
        col_gap: f32,
        brace_w: f32,
    },
    PlaceholderSlot {
        #[allow(dead_code)]
        size: f32,
    },
    ErrorSpan {
        text: String,
        size: f32,
    },
}

// =============================================================================
// AST -> MathBox LAYOUT BUILDER (WITH PARTIAL FALLBACK WHILE TYPING)
// =============================================================================

/// Result of parsing & laying out an expression string for live preview.
#[derive(Debug, Clone)]
pub struct PreviewBuild {
    pub layout: MathBox,
    pub parsed_statement: Option<MathStatement>,
    pub error: Option<MathError>,
    pub has_placeholders: bool,
    pub latex: String,
}

/// Builds a `MathBox` preview from a raw formula string (optionally preceded by a signature like `f(x)`).
/// Gracefully handles incomplete structural expressions while typing with active `□` slots.
pub fn build_preview_layout(
    m: &dyn TextMeasurer,
    sig_prefix: Option<&str>,
    raw_expr: &str,
    known_fns: &HashSet<String>,
    base_size: f32,
    sig_color: Option<Color32>,
) -> PreviewBuild {
    let trimmed = raw_expr.trim();
    if trimmed.is_empty() {
        let mut parts = Vec::new();
        if let Some(sig) = sig_prefix {
            if !sig.trim().is_empty() {
                parts.push(layout_signature_prefix(m, sig, base_size, sig_color));
            }
        }
        parts.push(MathBox::colored_text(
            m,
            "...",
            base_size,
            true,
            Color32::from_rgb(148, 163, 184),
        ));
        return PreviewBuild {
            layout: MathBox::row(parts),
            parsed_statement: None,
            error: None,
            has_placeholders: false,
            latex: String::new(),
        };
    }

    let layout_from_stmt = |stmt: MathStatement, has_placeholders: bool| -> PreviewBuild {
        let show_sig = sig_prefix.is_some()
            && !matches!(
                stmt,
                MathStatement::Equation { .. }
                    | MathStatement::Inequality { .. }
                    | MathStatement::Polar { .. }
                    | MathStatement::Parametric { .. }
                    | MathStatement::FunctionDef { .. }
                    | MathStatement::ParamAssign { .. }
            );
        let mut parts = Vec::new();
        if show_sig {
            if let Some(sig) = sig_prefix {
                if !sig.trim().is_empty() {
                    parts.push(layout_signature_prefix(m, sig, base_size, sig_color));
                }
            }
        }
        let stmt_box = layout_statement(m, &stmt, base_size);
        parts.push(stmt_box);

        let latex = if show_sig {
            if let Some(sig) = sig_prefix {
                if !sig.trim().is_empty() {
                    format!(
                        "{} = {}",
                        crate::math::to_latex_from_ast(sig),
                        stmt.to_latex()
                    )
                } else {
                    stmt.to_latex()
                }
            } else {
                stmt.to_latex()
            }
        } else {
            stmt.to_latex()
        };

        PreviewBuild {
            layout: MathBox::row(parts),
            parsed_statement: if has_placeholders { None } else { Some(stmt) },
            error: None,
            has_placeholders,
            latex,
        }
    };

    match parse_statement_with_known_fns(trimmed, Some(known_fns)) {
        Ok(stmt) => layout_from_stmt(stmt, false),
        Err(err) => {
            // Try structural lenient parse first so typing `/`, `1/`, `(x+1)/`, `x^`, `P_`,
            // `sqrt()`, `nthroot()`, `\int`, `\sum`, `\prod`, `d/dx` immediately renders 2D
            // layout with active `□` placeholder slots and NO red syntax error!
            if let Ok(lenient_stmt) = parse_statement_for_preview(trimmed, Some(known_fns)) {
                let has_ph = lenient_stmt.has_placeholder();
                return layout_from_stmt(lenient_stmt, has_ph);
            }

            // Graceful partial preview for genuine syntax errors: try parsing a balanced prefix
            let mut parts = Vec::new();
            if let Some(sig) = sig_prefix {
                if !sig.trim().is_empty() && !trimmed.contains('=') {
                    parts.push(layout_signature_prefix(m, sig, base_size, sig_color));
                }
            }

            let split_at = err.span.start.min(trimmed.len());
            let mut prefix_rendered = false;
            if split_at > 0 && trimmed.is_char_boundary(split_at) {
                let prefix = trimmed[..split_at].trim_end();
                if !prefix.is_empty() {
                    if let Ok(partial_ast) = parse_expr_with_known_fns(prefix, known_fns) {
                        parts.push(layout_expr(m, &partial_ast, base_size));
                        parts.push(MathBox::space(base_size * 0.2_f32));
                        let tail = &trimmed[split_at..];
                        if !tail.is_empty() {
                            parts.push(MathBox::error_span(m, tail, base_size));
                        }
                        prefix_rendered = true;
                    }
                }
            }
            if !prefix_rendered {
                // Split raw string around error span for inline highlight
                let start = err.span.start.min(trimmed.len());
                let end = err.span.end.clamp(start, trimmed.len());
                if trimmed.is_char_boundary(start) && trimmed.is_char_boundary(end) {
                    if start > 0 {
                        parts.push(MathBox::text(m, &trimmed[..start], base_size, false));
                    }
                    let err_slice = if end > start {
                        &trimmed[start..end]
                    } else if start < trimmed.len() {
                        &trimmed[start..]
                    } else {
                        "␣"
                    };
                    parts.push(MathBox::error_span(m, err_slice, base_size));
                    if end > start && end < trimmed.len() {
                        parts.push(MathBox::text(m, &trimmed[end..], base_size, false));
                    }
                } else {
                    parts.push(MathBox::error_span(m, trimmed, base_size));
                }
            }

            PreviewBuild {
                layout: MathBox::row(parts),
                parsed_statement: None,
                error: Some(err),
                has_placeholders: false,
                latex: trimmed.to_string(),
            }
        }
    }
}

fn layout_signature_prefix(
    m: &dyn TextMeasurer,
    sig: &str,
    base_size: f32,
    sig_color: Option<Color32>,
) -> MathBox {
    let mut items = Vec::new();
    if let Ok(ast) = parse_expr_with_known_fns(sig.trim(), &HashSet::new()) {
        items.push(layout_expr_colored(m, &ast, base_size, sig_color));
    } else {
        let t = sig.trim().to_string();
        items.push(if let Some(c) = sig_color {
            MathBox::colored_text(m, t, base_size, true, c)
        } else {
            MathBox::text(m, t, base_size, true)
        });
    }
    items.push(MathBox::space(base_size * 0.25_f32));
    items.push(MathBox::text(m, "=", base_size, false));
    items.push(MathBox::space(base_size * 0.25_f32));
    MathBox::row(items)
}

fn layout_expr_colored(
    m: &dyn TextMeasurer,
    node: &SpannedExpr,
    size: f32,
    color: Option<Color32>,
) -> MathBox {
    if let (MathExpr::Call { name, prime_order, args }, Some(c)) = (&node.expr, color) {
        let mut items = vec![layout_ident(m, name, size, Some(c))];
        if *prime_order > 0 {
            items.push(MathBox::colored_text(
                m,
                "′".repeat(*prime_order as usize),
                size,
                false,
                c,
            ));
        }
        let mut arg_boxes = Vec::new();
        for (i, a) in args.iter().enumerate() {
            if i > 0 {
                arg_boxes.push(MathBox::colored_text(m, ", ", size, false, c));
            }
            arg_boxes.push(layout_expr(m, a, size));
        }
        items.push(MathBox::delimited(
            DelimKind::Paren,
            DelimKind::Paren,
            MathBox::row(arg_boxes),
            size,
        ));
        return MathBox::row(items);
    }
    layout_expr(m, node, size)
}

pub fn layout_statement(m: &dyn TextMeasurer, stmt: &MathStatement, size: f32) -> MathBox {
    match stmt {
        MathStatement::FunctionDef { name, params, body } => {
            let mut items = vec![layout_ident(m, name, size, None)];
            let mut p_boxes = Vec::new();
            for (i, p) in params.iter().enumerate() {
                if i > 0 {
                    p_boxes.push(MathBox::text(m, ", ", size, false));
                }
                p_boxes.push(layout_ident(m, p, size, None));
            }
            items.push(MathBox::delimited(
                DelimKind::Paren,
                DelimKind::Paren,
                MathBox::row(p_boxes),
                size,
            ));
            items.push(MathBox::space(size * 0.25_f32));
            items.push(MathBox::text(m, "=", size, false));
            items.push(MathBox::space(size * 0.25_f32));
            items.push(layout_expr(m, body, size));
            MathBox::row(items)
        }
        MathStatement::ParamAssign { name, value } => MathBox::row(vec![
            layout_ident(m, name, size, None),
            MathBox::space(size * 0.25_f32),
            MathBox::text(m, "=", size, false),
            MathBox::space(size * 0.25_f32),
            layout_expr(m, value, size),
        ]),
        MathStatement::Equation { lhs, rhs } => MathBox::row(vec![
            layout_expr(m, lhs, size),
            MathBox::space(size * 0.25_f32),
            MathBox::text(m, "=", size, false),
            MathBox::space(size * 0.25_f32),
            layout_expr(m, rhs, size),
        ]),
        MathStatement::Inequality { lhs, op, rhs } => MathBox::row(vec![
            layout_expr(m, lhs, size),
            MathBox::space(size * 0.25_f32),
            MathBox::text(m, op.as_unicode(), size, false),
            MathBox::space(size * 0.25_f32),
            layout_expr(m, rhs, size),
        ]),
        MathStatement::Parametric {
            name,
            param,
            x_expr,
            y_expr,
        } => {
            let mut items = Vec::new();
            if let Some(n) = name {
                items.push(layout_ident(m, n, size, None));
                items.push(MathBox::delimited(
                    DelimKind::Paren,
                    DelimKind::Paren,
                    layout_ident(m, param, size, None),
                    size,
                ));
                items.push(MathBox::space(size * 0.25_f32));
                items.push(MathBox::text(m, "=", size, false));
                items.push(MathBox::space(size * 0.25_f32));
            }
            let tuple_inner = MathBox::row(vec![
                layout_expr(m, x_expr, size),
                MathBox::text(m, ", ", size, false),
                layout_expr(m, y_expr, size),
            ]);
            items.push(MathBox::delimited(
                DelimKind::Paren,
                DelimKind::Paren,
                tuple_inner,
                size,
            ));
            MathBox::row(items)
        }
        MathStatement::Polar { param, r_expr } => {
            let mut items = vec![layout_ident(m, "r", size, None)];
            if param != "theta" {
                items.push(MathBox::delimited(
                    DelimKind::Paren,
                    DelimKind::Paren,
                    layout_ident(m, param, size, None),
                    size,
                ));
            }
            items.push(MathBox::space(size * 0.25_f32));
            items.push(MathBox::text(m, "=", size, false));
            items.push(MathBox::space(size * 0.25_f32));
            items.push(layout_expr(m, r_expr, size));
            MathBox::row(items)
        }
        MathStatement::Expr(expr) => layout_expr(m, expr, size),
    }
}

pub fn layout_expr(m: &dyn TextMeasurer, node: &SpannedExpr, size: f32) -> MathBox {
    let script_size = (size * 0.72_f32).max(8.5_f32);
    let frac_size = (size * 0.84_f32).max(9.0_f32);

    match &node.expr {
        MathExpr::Number(n) => {
            let text = format_number_display(*n);
            MathBox::text(m, text, size, false)
        }
        MathExpr::Ident(name) => layout_ident(m, name, size, None),
        MathExpr::Subscript { base, sub } => {
            let b = layout_expr(m, base, size);
            let s = layout_expr(m, sub, script_size);
            MathBox::script(b, None, Some(s), size)
        }
        MathExpr::Accent { kind, inner } => {
            let inner_box = layout_expr(m, inner, size);
            MathBox::accent(*kind, inner_box, size)
        }
        MathExpr::Group(inner) => {
            let inner_box = layout_expr(m, inner, size);
            MathBox::delimited(DelimKind::Paren, DelimKind::Paren, inner_box, size)
        }
        MathExpr::Abs(inner) => {
            let inner_box = layout_expr(m, inner, size);
            MathBox::delimited(DelimKind::Pipe, DelimKind::Pipe, inner_box, size)
        }
        MathExpr::Unary { op, expr } => {
            let sym = match op {
                UnaryOp::Neg => "−",
                UnaryOp::Pos => "+",
            };
            MathBox::row(vec![
                MathBox::text(m, sym, size, false),
                layout_expr(m, expr, size),
            ])
        }
        MathExpr::Fraction { num, den } => {
            let n = unwrap_group_for_slot(num);
            let d = unwrap_group_for_slot(den);
            let num_box = layout_expr(m, n, frac_size);
            let den_box = layout_expr(m, d, frac_size);
            MathBox::fraction(num_box, den_box, size)
        }
        MathExpr::Sqrt { index, radicand } => {
            let idx_box = index.as_ref().map(|i| {
                layout_expr(m, unwrap_group_for_slot(i), (size * 0.62_f32).max(8.0_f32))
            });
            let rad_box = layout_expr(m, unwrap_group_for_slot(radicand), size);
            MathBox::radical(idx_box, rad_box, size)
        }
        MathExpr::Binary { op, lhs, rhs } => match op {
            BinOp::Pow => {
                // Combine `Subscript(base, sub) ^ exp` into a single dual Script box!
                if let MathExpr::Subscript { base, sub } = &lhs.expr {
                    let b = layout_expr(m, base, size);
                    let sub_b = layout_expr(m, sub, script_size);
                    let sup_b = layout_expr(m, unwrap_group_for_slot(rhs), script_size);
                    return MathBox::script(b, Some(sup_b), Some(sub_b), size);
                }
                let base_b = layout_expr(m, lhs, size);
                let exp_b = layout_expr(m, unwrap_group_for_slot(rhs), script_size);
                MathBox::script(base_b, Some(exp_b), None, size)
            }
            BinOp::Add => MathBox::row(vec![
                layout_expr(m, lhs, size),
                MathBox::space(size * 0.22_f32),
                MathBox::text(m, "+", size, false),
                MathBox::space(size * 0.22_f32),
                layout_expr(m, rhs, size),
            ]),
            BinOp::Sub => MathBox::row(vec![
                layout_expr(m, lhs, size),
                MathBox::space(size * 0.22_f32),
                MathBox::text(m, "−", size, false),
                MathBox::space(size * 0.22_f32),
                layout_expr(m, rhs, size),
            ]),
            BinOp::Mul => MathBox::row(vec![
                layout_expr(m, lhs, size),
                MathBox::space(size * 0.16_f32),
                MathBox::text(m, "·", size, false),
                MathBox::space(size * 0.16_f32),
                layout_expr(m, rhs, size),
            ]),
            BinOp::ImplicitMul => MathBox::row(vec![
                layout_expr(m, lhs, size),
                MathBox::space(size * 0.08_f32),
                layout_expr(m, rhs, size),
            ]),
            BinOp::Mod => MathBox::row(vec![
                layout_expr(m, lhs, size),
                MathBox::space(size * 0.2_f32),
                MathBox::text(m, "mod", size * 0.9_f32, false),
                MathBox::space(size * 0.2_f32),
                layout_expr(m, rhs, size),
            ]),
        },
        MathExpr::Call {
            name,
            prime_order,
            args,
        } => {
            if *prime_order == 0 && name == "abs" && args.len() == 1 {
                return MathBox::delimited(
                    DelimKind::Pipe,
                    DelimKind::Pipe,
                    layout_expr(m, &args[0], size),
                    size,
                );
            }
            if *prime_order == 0 && name == "sqrt" && args.len() == 1 {
                return MathBox::radical(None, layout_expr(m, &args[0], size), size);
            }

            let mut items = vec![layout_ident(m, name, size, None)];
            if *prime_order > 0 {
                items.push(MathBox::text(
                    m,
                    "′".repeat(*prime_order as usize),
                    size,
                    false,
                ));
            }
            let mut arg_boxes = Vec::new();
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    arg_boxes.push(MathBox::text(m, ", ", size, false));
                }
                arg_boxes.push(layout_expr(m, a, size));
            }
            items.push(MathBox::delimited(
                DelimKind::Paren,
                DelimKind::Paren,
                MathBox::row(arg_boxes),
                size,
            ));
            MathBox::row(items)
        }
        MathExpr::Derivative { var, order, body } => {
            let num = if *order <= 1 {
                MathBox::text(m, "d", frac_size, false)
            } else {
                MathBox::script(
                    MathBox::text(m, "d", frac_size, false),
                    Some(MathBox::text(m, order.to_string(), script_size, false)),
                    None,
                    frac_size,
                )
            };
            let den = if *order <= 1 {
                MathBox::row(vec![
                    MathBox::text(m, "d", frac_size, false),
                    layout_ident(m, var, frac_size, None),
                ])
            } else {
                MathBox::row(vec![
                    MathBox::text(m, "d", frac_size, false),
                    MathBox::script(
                        layout_ident(m, var, frac_size, None),
                        Some(MathBox::text(m, order.to_string(), script_size, false)),
                        None,
                        frac_size,
                    ),
                ])
            };
            let op_frac = MathBox::fraction(num, den, size);
            let body_box = layout_expr(m, body, size);
            MathBox::row(vec![op_frac, MathBox::space(size * 0.16_f32), body_box])
        }
        MathExpr::Integral {
            var,
            lower,
            upper,
            body,
        } => {
            let lo = layout_expr(m, lower, script_size);
            let hi = layout_expr(m, upper, script_size);
            let op = MathBox::big_op(m, "∫", Some(lo), Some(hi), false, size);
            let body_b = layout_expr(m, body, size);
            MathBox::row(vec![
                op,
                MathBox::space(size * 0.15_f32),
                body_b,
                MathBox::space(size * 0.22_f32),
                MathBox::text(m, "d", size, false),
                layout_ident(m, var, size, None),
            ])
        }
        MathExpr::Sum {
            var,
            lower,
            upper,
            body,
        } => {
            let lo = MathBox::row(vec![
                layout_ident(m, var, script_size, None),
                MathBox::text(m, "=", script_size, false),
                layout_expr(m, lower, script_size),
            ]);
            let hi = layout_expr(m, upper, script_size);
            let op = MathBox::big_op(m, "∑", Some(lo), Some(hi), true, size);
            MathBox::row(vec![
                op,
                MathBox::space(size * 0.18_f32),
                layout_expr(m, body, size),
            ])
        }
        MathExpr::Prod {
            var,
            lower,
            upper,
            body,
        } => {
            let lo = MathBox::row(vec![
                layout_ident(m, var, script_size, None),
                MathBox::text(m, "=", script_size, false),
                layout_expr(m, lower, script_size),
            ]);
            let hi = layout_expr(m, upper, script_size);
            let op = MathBox::big_op(m, "∏", Some(lo), Some(hi), true, size);
            MathBox::row(vec![
                op,
                MathBox::space(size * 0.18_f32),
                layout_expr(m, body, size),
            ])
        }
        MathExpr::ComparisonChain { first, rest } => {
            let mut items = vec![layout_expr(m, first, size)];
            for (op, rhs) in rest {
                items.push(MathBox::space(size * 0.22_f32));
                items.push(MathBox::text(m, op.as_unicode(), size, false));
                items.push(MathBox::space(size * 0.22_f32));
                items.push(layout_expr(m, rhs, size));
            }
            MathBox::row(items)
        }
        MathExpr::Piecewise(branches) => {
            let sub_size = (size * 0.88_f32).max(10.0_f32);
            let mut rows = Vec::with_capacity(branches.len());
            for b in branches {
                let val_b = layout_expr(m, &b.value, sub_size);
                let cond_b = if let Some(cond) = &b.condition {
                    Some(MathBox::row(vec![
                        MathBox::text(m, "if ", sub_size * 0.9_f32, true),
                        layout_expr(m, cond, sub_size),
                    ]))
                } else {
                    Some(MathBox::text(m, "otherwise", sub_size * 0.9_f32, true))
                };
                rows.push((val_b, cond_b));
            }
            MathBox::piecewise(rows, size)
        }
        MathExpr::Restricted {
            inner,
            restrictions,
        } => {
            let cond_size = (size * 0.86_f32).max(9.5_f32);
            let mut cond_items = Vec::new();
            for (i, cond) in restrictions.iter().enumerate() {
                if i > 0 {
                    cond_items.push(MathBox::text(m, ", ", cond_size, false));
                }
                cond_items.push(layout_expr(m, cond, cond_size));
            }
            MathBox::row(vec![
                layout_expr(m, inner, size),
                MathBox::space(size * 0.35_f32),
                MathBox::delimited(
                    DelimKind::Brace,
                    DelimKind::Brace,
                    MathBox::row(cond_items),
                    cond_size,
                ),
            ])
        }
        MathExpr::Placeholder => MathBox::placeholder_slot(size),
        MathExpr::Tuple(items) => {
            let mut inner = Vec::new();
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    inner.push(MathBox::text(m, ", ", size, false));
                }
                inner.push(layout_expr(m, item, size));
            }
            MathBox::delimited(DelimKind::Paren, DelimKind::Paren, MathBox::row(inner), size)
        }
        MathExpr::MemberAccess { object, field } => MathBox::row(vec![
            MathBox::text(m, object, size, false),
            MathBox::text(m, "$", size, false),
            MathBox::text(m, field, size, false),
        ]),
        MathExpr::Index { target, index } => MathBox::row(vec![
            layout_expr(m, target, size),
            MathBox::delimited(
                DelimKind::Bracket,
                DelimKind::Bracket,
                layout_expr(m, index, size),
                size,
            ),
        ]),
    }
}

fn layout_ident(
    m: &dyn TextMeasurer,
    name: &str,
    size: f32,
    color_override: Option<Color32>,
) -> MathBox {
    let make_text = |s: &str, sz: f32, italic: bool| -> MathBox {
        if let Some(c) = color_override {
            MathBox::colored_text(m, s, sz, italic, c)
        } else {
            MathBox::text(m, s, sz, italic)
        }
    };

    if let Some(unicode_greek) = greek_name_to_unicode(name) {
        return make_text(unicode_greek, size, false);
    }

    let sub_size = (size * 0.72_f32).max(8.5_f32);
    if name == "price_x" {
        return MathBox::script(
            make_text("P", size, true),
            None,
            Some(make_text("x", sub_size, true)),
            size,
        );
    }
    if name == "price_y" {
        return MathBox::script(
            make_text("P", size, true),
            None,
            Some(make_text("y", sub_size, true)),
            size,
        );
    }
    if let Some(base) = name.strip_suffix("_bar") {
        return MathBox::accent(
            AccentKind::Bar,
            layout_ident(m, base, size, color_override),
            size,
        );
    }
    if let Some(base) = name.strip_suffix("_hat") {
        return MathBox::accent(
            AccentKind::Hat,
            layout_ident(m, base, size, color_override),
            size,
        );
    }
    if let Some(base) = name.strip_suffix("_tilde") {
        return MathBox::accent(
            AccentKind::Tilde,
            layout_ident(m, base, size, color_override),
            size,
        );
    }
    if let Some((base, sub)) = name.split_once('_') {
        if !base.is_empty() && !sub.is_empty() {
            let base_box = layout_ident(m, base, size, color_override);
            let sub_box = layout_ident(m, sub, sub_size, color_override);
            return MathBox::script(base_box, None, Some(sub_box), size);
        }
    }

    make_text(name, size, name.len() == 1)
}

fn unwrap_group_for_slot(expr: &SpannedExpr) -> &SpannedExpr {
    match &expr.expr {
        MathExpr::Group(inner) => unwrap_group_for_slot(inner),
        _ => expr,
    }
}

fn format_number_display(v: f64) -> String {
    if !v.is_finite() {
        return "NaN".to_string();
    }
    if v.fract().abs() < 1e-12_f64 && v.abs() < 1e12_f64 {
        format!("{}", v as i64)
    } else {
        let s = format!("{:.5}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

// =============================================================================
// NATIVE EGUI PAINTER RENDERER
// =============================================================================

/// Paints a laid-out `MathBox` at `(origin_x, baseline_y)` using `egui::Painter`.
pub fn paint_math_box(
    painter: &Painter,
    mbox: &MathBox,
    origin_x: f32,
    baseline_y: f32,
    default_color: Color32,
    error_color: Color32,
) {
    match &mbox.kind {
        MathBoxKind::Space => {}
        MathBoxKind::Text {
            text,
            size,
            color_override,
            ..
        } => {
            let color = color_override.unwrap_or(default_color);
            let font = FontId::new(size.max(6.0_f32), FontFamily::Proportional);
            let top_y = baseline_y - mbox.ascent;
            painter.text(Pos2::new(origin_x, top_y), egui::Align2::LEFT_TOP, text, font, color);
        }
        MathBoxKind::PlaceholderSlot { .. } => {
            let slot_rect = egui::Rect::from_min_max(
                Pos2::new(origin_x + 0.8_f32, baseline_y - mbox.ascent + 1.0_f32),
                Pos2::new(origin_x + mbox.width - 0.8_f32, baseline_y + mbox.descent - 0.5_f32),
            );
            let accent = Color32::from_rgb(96, 165, 250);
            painter.rect_filled(slot_rect, Rounding::same(3.0_f32), accent.gamma_multiply(0.14_f32));
            // Subtle dashed border segments around the active visual placeholder slot
            let dash_stroke = Stroke::new(1.15_f32, accent.gamma_multiply(0.75_f32));
            let dash_len = 3.0_f32;
            let gap_len = 2.0_f32;
            let draw_dashed = |p0: Pos2, p1: Pos2| {
                let d = p1 - p0;
                let total = d.length();
                if total < 1.0_f32 {
                    return;
                }
                let dir = d / total;
                let mut dist = 0.0_f32;
                while dist < total {
                    let seg_end = (dist + dash_len).min(total);
                    painter.line_segment([p0 + dir * dist, p0 + dir * seg_end], dash_stroke);
                    dist += dash_len + gap_len;
                }
            };
            draw_dashed(slot_rect.left_top(), slot_rect.right_top());
            draw_dashed(slot_rect.right_top(), slot_rect.right_bottom());
            draw_dashed(slot_rect.right_bottom(), slot_rect.left_bottom());
            draw_dashed(slot_rect.left_bottom(), slot_rect.left_top());
        }
        MathBoxKind::ErrorSpan { text, size } => {
            let font = FontId::new(size.max(6.0_f32), FontFamily::Monospace);
            let top_y = baseline_y - mbox.ascent;
            painter.text(
                Pos2::new(origin_x, top_y),
                egui::Align2::LEFT_TOP,
                text,
                font,
                error_color,
            );
            // Draw crisp squiggly/error underline
            let y_line = baseline_y + mbox.descent - 1.5_f32;
            let mut pts = Vec::new();
            let mut x = origin_x;
            let end_x = origin_x + mbox.width.max(6.0_f32);
            let mut up = true;
            while x < end_x {
                pts.push(Pos2::new(x, y_line + if up { -1.0_f32 } else { 1.0_f32 }));
                x += 3.0_f32;
                up = !up;
            }
            pts.push(Pos2::new(end_x, y_line));
            if pts.len() >= 2 {
                painter.add(egui::Shape::line(pts, Stroke::new(1.2_f32, error_color)));
            }
        }
        MathBoxKind::Row(children) => {
            let mut cursor_x = origin_x;
            for child in children {
                paint_math_box(painter, child, cursor_x, baseline_y, default_color, error_color);
                cursor_x += child.width;
            }
        }
        MathBoxKind::Fraction {
            num,
            den,
            bar_thickness,
            gap,
            axis_height,
        } => {
            let bar_y = baseline_y - *axis_height;
            painter.line_segment(
                [Pos2::new(origin_x + 1.0_f32, bar_y), Pos2::new(origin_x + mbox.width - 1.0_f32, bar_y)],
                Stroke::new(*bar_thickness, default_color),
            );
            let num_x = origin_x + 0.5_f32 * (mbox.width - num.width);
            let num_baseline = bar_y - *bar_thickness * 0.5_f32 - *gap - num.descent;
            paint_math_box(painter, num, num_x, num_baseline, default_color, error_color);

            let den_x = origin_x + 0.5_f32 * (mbox.width - den.width);
            let den_baseline = bar_y + *bar_thickness * 0.5_f32 + *gap + den.ascent;
            paint_math_box(painter, den, den_x, den_baseline, default_color, error_color);
        }
        MathBoxKind::Script {
            base,
            sup,
            sub,
            sup_shift,
            sub_shift,
            gap_x,
        } => {
            paint_math_box(painter, base, origin_x, baseline_y, default_color, error_color);
            let script_x = origin_x + base.width + *gap_x;
            if let Some(s) = sup {
                paint_math_box(
                    painter,
                    s,
                    script_x,
                    baseline_y - *sup_shift,
                    default_color,
                    error_color,
                );
            }
            if let Some(s) = sub {
                paint_math_box(
                    painter,
                    s,
                    script_x,
                    baseline_y + *sub_shift,
                    default_color,
                    error_color,
                );
            }
        }
        MathBoxKind::Radical {
            index,
            radicand,
            idx_extra_w,
            hook_w,
            bar_gap,
            bar_thickness,
        } => {
            let top_y = baseline_y - radicand.ascent - *bar_gap;
            let bot_y = baseline_y + radicand.descent;
            let mid_y = top_y + (bot_y - top_y) * 0.56_f32;
            let hook_x0 = origin_x + *idx_extra_w;
            let rad_x = hook_x0 + *hook_w;

            // Optional nth-root index tucked above the hook shelf
            if let Some(idx_box) = index {
                let idx_x = (hook_x0 + *hook_w * 0.42_f32 - idx_box.width).max(origin_x);
                let idx_base = mid_y - 2.0_f32 - idx_box.descent;
                paint_math_box(painter, idx_box, idx_x, idx_base, default_color, error_color);
            }

            // Drawn radical glyph + horizontal overbar
            let pts = vec![
                Pos2::new(hook_x0 + 0.5_f32, mid_y + 1.5_f32),
                Pos2::new(hook_x0 + *hook_w * 0.28_f32, mid_y),
                Pos2::new(hook_x0 + *hook_w * 0.54_f32, bot_y),
                Pos2::new(rad_x - 0.5_f32, top_y),
                Pos2::new(rad_x + radicand.width + 1.5_f32, top_y),
            ];
            painter.add(egui::Shape::line(
                pts,
                Stroke::new((*bar_thickness).max(1.25_f32), default_color),
            ));

            paint_math_box(painter, radicand, rad_x, baseline_y, default_color, error_color);
        }
        MathBoxKind::Delimited {
            left,
            right,
            inner,
            delim_w,
        } => {
            let top_y = baseline_y - mbox.ascent;
            let bot_y = baseline_y + mbox.descent;
            paint_delimiter(painter, *left, true, origin_x, top_y, bot_y, *delim_w, default_color);
            paint_math_box(
                painter,
                inner,
                origin_x + *delim_w + 1.0_f32,
                baseline_y,
                default_color,
                error_color,
            );
            paint_delimiter(
                painter,
                *right,
                false,
                origin_x + *delim_w + inner.width + 2.0_f32,
                top_y,
                bot_y,
                *delim_w,
                default_color,
            );
        }
        MathBoxKind::Accent {
            kind,
            inner,
            gap,
            accent_h,
        } => {
            let inner_x = origin_x + 0.5_f32 * (mbox.width - inner.width);
            paint_math_box(painter, inner, inner_x, baseline_y, default_color, error_color);
            let y_bot = baseline_y - inner.ascent - *gap;
            let y_top = y_bot - *accent_h;
            let x0 = origin_x + 0.8_f32;
            let x1 = origin_x + mbox.width - 0.8_f32;
            let xm = 0.5_f32 * (x0 + x1);
            let stroke = Stroke::new(1.3_f32, default_color);
            match kind {
                AccentKind::Bar => {
                    painter.line_segment([Pos2::new(x0, y_bot), Pos2::new(x1, y_bot)], stroke);
                }
                AccentKind::Hat => {
                    painter.add(egui::Shape::line(
                        vec![Pos2::new(x0, y_bot), Pos2::new(xm, y_top), Pos2::new(x1, y_bot)],
                        stroke,
                    ));
                }
                AccentKind::Tilde => {
                    let ym = 0.5_f32 * (y_top + y_bot);
                    let pts = vec![
                        Pos2::new(x0, ym + 0.8_f32),
                        Pos2::new(x0 + 0.33_f32 * (x1 - x0), y_top),
                        Pos2::new(x0 + 0.67_f32 * (x1 - x0), y_bot),
                        Pos2::new(x1, ym - 0.8_f32),
                    ];
                    painter.add(egui::Shape::line(pts, stroke));
                }
                AccentKind::Vec => {
                    let ym = 0.5_f32 * (y_top + y_bot);
                    painter.line_segment([Pos2::new(x0, ym), Pos2::new(x1, ym)], stroke);
                    painter.add(egui::Shape::line(
                        vec![
                            Pos2::new(x1 - 2.5_f32, ym - 1.8_f32),
                            Pos2::new(x1, ym),
                            Pos2::new(x1 - 2.5_f32, ym + 1.8_f32),
                        ],
                        stroke,
                    ));
                }
            }
        }
        MathBoxKind::BigOp {
            symbol_box,
            lower,
            upper,
            stacked,
            gap,
        } => {
            if *stacked {
                let sym_x = origin_x + 0.5_f32 * (mbox.width - symbol_box.width);
                let sym_center_y = baseline_y - symbol_box.height() * 0.12_f32;
                let sym_baseline = sym_center_y + (symbol_box.ascent - symbol_box.descent) * 0.5_f32;
                paint_math_box(painter, symbol_box, sym_x, sym_baseline, default_color, error_color);

                if let Some(u) = upper {
                    let ux = origin_x + 0.5_f32 * (mbox.width - u.width);
                    let u_base = sym_baseline - symbol_box.ascent - *gap - u.descent;
                    paint_math_box(painter, u, ux, u_base, default_color, error_color);
                }
                if let Some(l) = lower {
                    let lx = origin_x + 0.5_f32 * (mbox.width - l.width);
                    let l_base = sym_baseline + symbol_box.descent + *gap + l.ascent;
                    paint_math_box(painter, l, lx, l_base, default_color, error_color);
                }
            } else {
                let sym_baseline = baseline_y + symbol_box.height() * 0.08_f32;
                paint_math_box(
                    painter,
                    symbol_box,
                    origin_x,
                    sym_baseline,
                    default_color,
                    error_color,
                );
                let bx = origin_x + symbol_box.width + 1.0_f32;
                if let Some(u) = upper {
                    let u_base = sym_baseline - symbol_box.ascent * 0.65_f32;
                    paint_math_box(painter, u, bx, u_base, default_color, error_color);
                }
                if let Some(l) = lower {
                    let l_base = sym_baseline + symbol_box.descent * 0.75_f32 + l.ascent * 0.4_f32;
                    paint_math_box(painter, l, bx, l_base, default_color, error_color);
                }
            }
        }
        MathBoxKind::Piecewise {
            rows,
            max_val_w,
            row_gap,
            col_gap,
            brace_w,
        } => {
            let top_y = baseline_y - mbox.ascent;
            let bot_y = baseline_y + mbox.descent;
            paint_delimiter(
                painter,
                DelimKind::Brace,
                true,
                origin_x,
                top_y,
                bot_y,
                *brace_w,
                default_color,
            );
            let val_x = origin_x + *brace_w + 3.0_f32;
            let cond_x = val_x + *max_val_w + *col_gap;
            let mut cursor_y = top_y;
            for (val, cond) in rows {
                let row_ascent = val
                    .ascent
                    .max(cond.as_ref().map(|c| c.ascent).unwrap_or(0.0_f32));
                let row_descent = val
                    .descent
                    .max(cond.as_ref().map(|c| c.descent).unwrap_or(0.0_f32));
                let row_base = cursor_y + row_ascent;
                paint_math_box(painter, val, val_x, row_base, default_color, error_color);
                if let Some(c) = cond {
                    paint_math_box(painter, c, cond_x, row_base, default_color, error_color);
                }
                cursor_y += row_ascent + row_descent + *row_gap;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_delimiter(
    painter: &Painter,
    kind: DelimKind,
    is_left: bool,
    x: f32,
    top_y: f32,
    bot_y: f32,
    w: f32,
    color: Color32,
) {
    let stroke = Stroke::new(1.3_f32, color);
    let mid_y = 0.5_f32 * (top_y + bot_y);
    match kind {
        DelimKind::Pipe => {
            let cx = x + 0.5_f32 * w;
            painter.line_segment([Pos2::new(cx, top_y), Pos2::new(cx, bot_y)], stroke);
        }
        DelimKind::Bracket => {
            let outer = if is_left { x + 1.5_f32 } else { x + w - 1.5_f32 };
            let inner = if is_left { x + w - 1.0_f32 } else { x + 1.0_f32 };
            painter.add(egui::Shape::line(
                vec![
                    Pos2::new(inner, top_y),
                    Pos2::new(outer, top_y),
                    Pos2::new(outer, bot_y),
                    Pos2::new(inner, bot_y),
                ],
                stroke,
            ));
        }
        DelimKind::Paren => {
            let steps = 10usize;
            let mut pts = Vec::with_capacity(steps + 1);
            let outer = if is_left { x + 1.2_f32 } else { x + w - 1.2_f32 };
            let inner = if is_left { x + w - 1.0_f32 } else { x + 1.0_f32 };
            for i in 0..=steps {
                let t = (i as f32) / (steps as f32);
                let u = 2.0_f32 * t - 1.0_f32; // -1..1
                let px = outer + (inner - outer) * (u * u);
                let py = top_y + (bot_y - top_y) * t;
                pts.push(Pos2::new(px, py));
            }
            painter.add(egui::Shape::line(pts, stroke));
        }
        DelimKind::Brace => {
            let tip = if is_left { x + 1.0_f32 } else { x + w - 1.0_f32 };
            let mid_x = x + 0.5_f32 * w;
            let edge = if is_left { x + w - 1.0_f32 } else { x + 1.0_f32 };
            let h = (bot_y - top_y).max(4.0_f32);
            let q = (h * 0.14_f32).min(6.0_f32);
            let pts = vec![
                Pos2::new(edge, top_y),
                Pos2::new(mid_x, top_y + q),
                Pos2::new(mid_x, mid_y - q),
                Pos2::new(tip, mid_y),
                Pos2::new(mid_x, mid_y + q),
                Pos2::new(mid_x, bot_y - q),
                Pos2::new(edge, bot_y),
            ];
            painter.add(egui::Shape::line(pts, stroke));
        }
    }
}

/// Renders a typographic mathematical identifier (`P_x`, `P_y`, `Y_bar`, `\alpha_1`, `c_1`, etc.)
/// natively via `MathBox` subscripts, overbars, and Greek glyphs — completely avoiding missing
/// Unicode subscript glyphs (`P□`).
pub fn render_math_ident_inline(
    ui: &mut Ui,
    ident: &str,
    base_size: f32,
    color: Color32,
) -> egui::Response {
    let measurer = EguiTextMeasurer {
        painter: ui.painter(),
    };
    let mbox = layout_ident(&measurer, ident, base_size, Some(color));
    let desired = Vec2::new(mbox.width.max(8.0_f32) + 2.0_f32, mbox.height().max(base_size + 2.0_f32));
    let (rect, resp) = ui.allocate_exact_size(desired, Sense::hover());
    let top_offset = (rect.height() - mbox.height()) * 0.5_f32;
    let baseline_y = rect.top() + top_offset + mbox.ascent;
    paint_math_box(
        ui.painter(),
        &mbox,
        rect.left() + 1.0_f32,
        baseline_y,
        color,
        color,
    );
    resp
}

/// Interactive "Math-as-Written" strip that supports clicking directly on the 2D formula
/// to focus/edit it, showing an active focus highlight ring and optional blinking caret.
#[allow(clippy::too_many_arguments)]
pub fn render_math_preview_strip_interactive(
    ui: &mut Ui,
    sig_prefix: Option<&str>,
    raw_expr: &str,
    known_fns: &HashSet<String>,
    base_size: f32,
    text_color: Color32,
    sig_color: Option<Color32>,
    bg_color: Color32,
    border_color: Color32,
    error_color: Color32,
    focused: bool,
    accent_color: Color32,
) -> (PreviewBuild, egui::Response) {
    let measurer = EguiTextMeasurer {
        painter: ui.painter(),
    };
    let build = build_preview_layout(&measurer, sig_prefix, raw_expr, known_fns, base_size, sig_color);

    let pad_x = 8.0_f32;
    let pad_y = 5.0_f32;
    let desired_w = ui.available_width().max(build.layout.width + 2.0_f32 * pad_x + 6.0_f32);
    let desired_h = (build.layout.height() + 2.0_f32 * pad_y).max(base_size + 10.0_f32);

    let (rect, resp) = ui.allocate_exact_size(Vec2::new(desired_w, desired_h), Sense::click());
    let stroke_color = if build.error.is_some() {
        error_color
    } else if focused {
        accent_color
    } else if resp.hovered() {
        accent_color.gamma_multiply(0.55_f32)
    } else {
        border_color
    };

    let painter = ui.painter_at(rect.intersect(ui.clip_rect()));
    painter.rect_filled(rect, Rounding::same(6.0_f32), bg_color);
    painter.rect_stroke(
        rect,
        Rounding::same(6.0_f32),
        Stroke::new(if focused { 1.4_f32 } else { 1.0_f32 }, stroke_color),
    );

    let content_h = build.layout.height();
    let top_offset = (rect.height() - content_h) * 0.5_f32;
    let baseline_y = rect.top() + top_offset + build.layout.ascent;
    let origin_x = rect.left() + pad_x;

    paint_math_box(
        &painter,
        &build.layout,
        origin_x,
        baseline_y,
        text_color,
        error_color,
    );

    // Subtle live caret indicator at the end of the 2D preview while actively focused
    if focused && !build.has_placeholders {
        let blink_on = ((ui.input(|i| i.time) * 2.0_f64) as i64) % 2 == 0;
        if blink_on {
            let cx = origin_x + build.layout.width + 2.5_f32;
            let cy0 = (baseline_y - base_size * 0.78_f32).max(rect.top() + 3.0_f32);
            let cy1 = (baseline_y + base_size * 0.22_f32).min(rect.bottom() - 3.0_f32);
            painter.line_segment([Pos2::new(cx, cy0), Pos2::new(cx, cy1)], Stroke::new(1.5_f32, accent_color));
        }
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(450));
    }

    (build, resp)
}

/// Convenience widget that allocates and paints a crisp native typeset preview strip in `ui`.
#[allow(clippy::too_many_arguments, dead_code)]
pub fn render_math_preview_strip(
    ui: &mut Ui,
    sig_prefix: Option<&str>,
    raw_expr: &str,
    known_fns: &HashSet<String>,
    base_size: f32,
    text_color: Color32,
    sig_color: Option<Color32>,
    bg_color: Color32,
    border_color: Color32,
    error_color: Color32,
) -> PreviewBuild {
    let measurer = EguiTextMeasurer {
        painter: ui.painter(),
    };
    let build = build_preview_layout(&measurer, sig_prefix, raw_expr, known_fns, base_size, sig_color);

    let pad_x = 8.0_f32;
    let pad_y = 4.0_f32;
    let desired_w = ui.available_width().max(build.layout.width + 2.0_f32 * pad_x);
    let desired_h = (build.layout.height() + 2.0_f32 * pad_y).max(base_size + 8.0_f32);

    let (rect, _resp) = ui.allocate_exact_size(Vec2::new(desired_w, desired_h), Sense::hover());
    let stroke_color = if build.error.is_some() {
        error_color
    } else {
        border_color
    };

    let painter = ui.painter_at(rect.intersect(ui.clip_rect()));
    painter.rect_filled(rect, Rounding::same(6.0_f32), bg_color);
    painter.rect_stroke(rect, Rounding::same(6.0_f32), Stroke::new(1.0_f32, stroke_color));

    let content_h = build.layout.height();
    let top_offset = (rect.height() - content_h) * 0.5_f32;
    let baseline_y = rect.top() + top_offset + build.layout.ascent;
    let origin_x = rect.left() + pad_x;

    paint_math_box(
        &painter,
        &build.layout,
        origin_x,
        baseline_y,
        text_color,
        error_color,
    );

    build
}

// =============================================================================
// SMART INPUT ERGONOMICS (NON-DESTRUCTIVE MATH SHORTCUTS)
// =============================================================================

/// Applies Desmos-style smart shortcuts to a math input string as the user types:
/// - `>=` -> `≥`, `<=` -> `≤`, `!=` -> `≠`, `->` -> `→`
/// - Greek names followed by a boundary delimiter (`alpha ` -> `α `, `theta)` -> `θ)`, `pi*` -> `π*`).
///
/// Returns `true` if the string was modified.
pub fn apply_smart_math_shortcuts(input: &mut String) -> bool {
    let original = input.clone();

    // Operator shortcuts
    let mut s = input
        .replace(">=", "≥")
        .replace("<=", "≤")
        .replace("!=", "≠")
        .replace("->", "→");

    // Word-boundary Greek shortcuts only when followed by a non-identifier character
    // (so typing `alpha_1` or mid-word doesn't disrupt, while `alpha ` / `alpha*` / `theta)` converts cleanly)
    const SHORTCUTS: &[(&str, &str)] = &[
        ("alpha", "α"),
        ("beta", "β"),
        ("gamma", "γ"),
        ("delta", "δ"),
        ("epsilon", "ε"),
        ("zeta", "ζ"),
        ("eta", "η"),
        ("theta", "θ"),
        ("iota", "ι"),
        ("kappa", "κ"),
        ("lambda", "λ"),
        ("mu", "μ"),
        ("nu", "ν"),
        ("xi", "ξ"),
        ("pi", "π"),
        ("rho", "ρ"),
        ("sigma", "σ"),
        ("tau", "τ"),
        ("upsilon", "υ"),
        ("phi", "φ"),
        ("chi", "χ"),
        ("psi", "ψ"),
        ("omega", "ω"),
    ];

    for &(word, sym) in SHORTCUTS {
        s = replace_delimited_greek(&s, word, sym);
    }

    // Structural operator shortcuts triggered specifically on trailing space (non-destructive)
    const SPACE_OP_SHORTCUTS: &[(&str, &str)] = &[
        ("sqrt ", "√"),
        ("int ", "∫"),
        ("sum ", "∑"),
        ("prod ", "∏"),
        ("infty ", "∞"),
    ];
    for &(word, sym) in SPACE_OP_SHORTCUTS {
        s = replace_space_operator(&s, word, sym);
    }

    if s != original {
        *input = s;
        true
    } else {
        false
    }
}

fn replace_space_operator(src: &str, word_with_space: &str, sym: &str) -> String {
    if !src.contains(word_with_space) {
        return src.to_string();
    }
    let is_word_ch = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '\\';
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    let mut prev: Option<char> = None;

    while let Some(pos) = rest.find(word_with_space) {
        let prefix = &rest[..pos];
        if let Some(ch) = prefix.chars().last() {
            prev = Some(ch);
        }
        let valid_before = !prev.map(is_word_ch).unwrap_or(false);
        out.push_str(prefix);
        if valid_before {
            out.push_str(sym);
        } else {
            out.push_str(word_with_space);
        }
        prev = Some(' ');
        rest = &rest[pos + word_with_space.len()..];
    }
    out.push_str(rest);
    out
}

fn replace_delimited_greek(src: &str, word: &str, sym: &str) -> String {
    if !src.contains(word) {
        return src.to_string();
    }
    let is_word_ch = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '\\';
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    let mut prev: Option<char> = None;

    while let Some(pos) = rest.find(word) {
        let prefix = &rest[..pos];
        if let Some(ch) = prefix.chars().last() {
            prev = Some(ch);
        }
        let after = &rest[pos + word.len()..];
        let next = after.chars().next();

        let valid_before = !prev.map(is_word_ch).unwrap_or(false);
        // Only trigger once the user types a delimiter after the word (space, operator, paren, etc.)
        let valid_after = next
            .map(|c| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(false);

        out.push_str(prefix);
        if valid_before && valid_after {
            out.push_str(sym);
        } else {
            out.push_str(word);
        }
        prev = word.chars().last();
        rest = after;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fraction_layout_metrics_exceed_plain_text() {
        let m = MockTextMeasurer;
        let known = HashSet::new();
        let plain = build_preview_layout(&m, None, "a + b", &known, 16.0, None);
        let frac = build_preview_layout(&m, None, r"\frac{a + b}{c + d}", &known, 16.0, None);

        assert!(frac.error.is_none());
        assert!(frac.layout.height() > plain.layout.height() * 1.5);
        assert!(frac.layout.ascent > plain.layout.ascent);
        assert!(frac.layout.descent > plain.layout.descent);
        assert_eq!(frac.latex, r"\frac{a + b}{c + d}");
    }

    #[test]
    fn test_scripts_radical_and_auto_scaling_delimiters() {
        let m = MockTextMeasurer;
        let known = HashSet::new();

        let base = build_preview_layout(&m, None, "x", &known, 16.0, None);
        let sup = build_preview_layout(&m, None, "x^{2}", &known, 16.0, None);
        let sub = build_preview_layout(&m, None, "P_x", &known, 16.0, None);
        let rad = build_preview_layout(&m, None, r"\sqrt[3]{x + 1}", &known, 16.0, None);
        let parens_tall = build_preview_layout(&m, None, r"(\frac{a}{b})", &known, 16.0, None);
        let parens_short = build_preview_layout(&m, None, "(x)", &known, 16.0, None);

        assert!(sup.layout.ascent > base.layout.ascent);
        assert!(sub.layout.descent > base.layout.descent);
        assert!(rad.layout.width > base.layout.width);
        assert!(rad.layout.ascent > base.layout.ascent);
        assert!(parens_tall.layout.height() > parens_short.layout.height() * 1.4);
    }

    #[test]
    fn test_big_operators_accent_and_piecewise_metrics() {
        let m = MockTextMeasurer;
        let known = HashSet::new();

        let sum_box = build_preview_layout(&m, None, r"\sum_{i=1}^{n} i^2", &known, 16.0, None);
        let int_box = build_preview_layout(&m, None, r"\int_0^1 x^2 dx", &known, 16.0, None);
        let bar_box = build_preview_layout(&m, None, r"\bar{Y}", &known, 16.0, None);
        let plain_y = build_preview_layout(&m, None, "Y", &known, 16.0, None);
        let pw_box = build_preview_layout(&m, None, "{ x < 0: -x, x >= 0: x^2 }", &known, 16.0, None);

        assert!(sum_box.error.is_none());
        assert!(sum_box.layout.height() > 30.0);
        assert!(int_box.error.is_none());
        assert!(int_box.layout.height() > 22.0);
        assert!(bar_box.layout.ascent > plain_y.layout.ascent);
        assert!(pw_box.error.is_none());
        assert!(pw_box.layout.height() > plain_y.layout.height() * 1.8);
    }

    #[test]
    fn test_partial_incomplete_expression_fallback_and_smart_shortcuts() {
        let m = MockTextMeasurer;
        let known = HashSet::new();
        let partial = build_preview_layout(&m, Some("f(x)"), "2x + ", &known, 16.0, None);
        assert!(partial.error.is_some());
        assert!(partial.layout.width > 0.0);

        let mut text = "alpha + beta >= 1 -> x <= 5".to_string();
        assert!(apply_smart_math_shortcuts(&mut text));
        assert_eq!(text, "α + β ≥ 1 → x ≤ 5");

        // Does not corrupt identifiers containing underscores or backslash commands
        let mut untouched = r"\alpha + alpha_1".to_string();
        assert!(!apply_smart_math_shortcuts(&mut untouched));
    }

    #[test]
    fn test_live_fraction_and_structural_placeholder_slots() {
        let m = MockTextMeasurer;
        let known = HashSet::new();
        let plain = build_preview_layout(&m, None, "x + 1", &known, 16.0, None);

        for expr in ["/", "1/", "(x+1)/", "g(x,y) = ln(5x)/"] {
            let preview = build_preview_layout(&m, None, expr, &known, 16.0, None);
            assert!(
                preview.error.is_none(),
                "Expected no red error while typing structural fraction `{}`: {:?}",
                expr,
                preview.error
            );
            assert!(preview.has_placeholders);
            assert!(
                preview.layout.height() > plain.layout.height() * 1.4,
                "Expected stacked 2D fraction height for `{}`",
                expr
            );
        }

        for expr in ["x^", "P_", "sqrt()", r"\int_0^1", r"\sum_{i=1}"] {
            let preview = build_preview_layout(&m, None, expr, &known, 16.0, None);
            assert!(preview.error.is_none(), "Unexpected error on `{}`", expr);
            assert!(preview.has_placeholders);
            assert!(preview.layout.width > 0.0);
        }
    }
}
