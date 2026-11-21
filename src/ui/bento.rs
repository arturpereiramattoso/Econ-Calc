use eframe::egui::{
    self, Color32, FontFamily, FontId, Margin, RichText, Rounding, Stroke, TextStyle, Ui, Vec2,
};
use egui_plot::{HLine, Line, LineStyle, PlotPoints, PlotUi, VLine};

// =============================================================================
// OIKONOMOS (Οἰκονόμος) — MINERAL & MANUSCRIPT PALETTE ("VIGÍLIA" & "PERGAMINHO")
// =============================================================================

#[derive(Debug, Clone, Copy)]
pub struct ThemePalette {
    /// Outer window backdrop behind cards (`#13110F` Vigília dark / `#E4DCCD` Pergaminho light).
    pub outer_bg: Color32,
    /// Primary card surface (`#1C1916` walnut dark / `#F5EFE4` parchment light).
    pub card_bg: Color32,
    /// Card header / tab bar strip (`#24201C` dark / `#ECE5D7` light).
    pub card_header_bg: Color32,
    /// Deep canvas / console / text input well (`#0F0E0C` atramentum dark / `#FAF6EE` vellum light).
    pub extreme_bg: Color32,
    /// Plot canvas background (`#12100E` dark / `#F8F3E9` light).
    pub plot_bg: Color32,
    /// Inner capsule cards for functions & sliders (`#26221D` dark / `#EFE8DB` light).
    pub sub_card_bg: Color32,
    /// Interactive pills & buttons (`#2B2620` dark / `#E6DECE` light).
    pub pill_bg: Color32,
    /// Hovered pills & buttons (`#383129` dark / `#D6CCB9` light).
    pub pill_hover: Color32,
    /// Active selection fill (`#4C6C8E` Lapis dark / `#34567E` Lapis light).
    pub pill_active: Color32,
    /// Subtle 1px card & divider border (`#3A332B` aged bronze dark / `#C6BCAA` drypoint light).
    pub border: Color32,
    /// Primary legible text (`#E8E0CE` raw linen dark / `#221D18` iron-gall ink light).
    pub text_primary: Color32,
    /// Secondary labels & inactive tabs (`#B0A694` limestone dark / `#5C5246` walnut sepia light).
    pub text_secondary: Color32,
    /// Muted hints & metadata (`#7C7263` umber dark / `#847868` earth gray light).
    pub text_muted: Color32,
    /// Primary mineral accent: Byzantine Lapis Lazuli (`#7098C0` dark / `#2E527E` light).
    pub accent_blue: Color32,
    /// Cinnabar / Manuscript Rubric (`#C45848` Kinnabari dark / `#A8382B` light).
    pub accent_red: Color32,
    /// Malachite / Verdigris (`#5C9E7A` Prasinos dark / `#307250` light).
    pub accent_green: Color32,
    /// Ochre Gold / Beeswax Candlelight (`#CEA24E` Chrysochra dark / `#A67422` light).
    pub accent_amber: Color32,
    /// Byzantine Porphyry / Tyrian Muted Purple (`#9E7698` Porphyra dark / `#76446A` light).
    pub accent_purple: Color32,
}

impl ThemePalette {
    pub fn for_mode(dark: bool) -> Self {
        if dark {
            // "Vigília" — Monastic Charcoal, Walnut, Linen & Mineral Pigments
            Self {
                outer_bg: Color32::from_rgb(19, 17, 15),         // #13110F (Atramentum / Charcoal)
                card_bg: Color32::from_rgb(28, 25, 22),          // #1C1916 (Dark Walnut)
                card_header_bg: Color32::from_rgb(36, 32, 28),   // #24201C (Cedar Header)
                extreme_bg: Color32::from_rgb(15, 14, 12),       // #0F0E0C (Deep Inkwell)
                plot_bg: Color32::from_rgb(18, 16, 14),          // #12100E (Abax Slate)
                sub_card_bg: Color32::from_rgb(38, 34, 29),      // #26221D (Kellion Cell)
                pill_bg: Color32::from_rgb(43, 38, 32),          // #2B2620
                pill_hover: Color32::from_rgb(56, 49, 41),       // #383129
                pill_active: Color32::from_rgb(76, 108, 142),    // #4C6C8E (Tempered Lapis)
                border: Color32::from_rgb(58, 51, 43),           // #3A332B (Aged Bronze / Umber)
                text_primary: Color32::from_rgb(232, 224, 206),  // #E8E0CE (Unbleached Linen)
                text_secondary: Color32::from_rgb(176, 166, 148), // #B0A694 (Warm Limestone)
                text_muted: Color32::from_rgb(124, 114, 99),     // #7C7263 (Muted Sepia)
                accent_blue: Color32::from_rgb(112, 152, 192),   // #7098C0 (Lapis Lazuli)
                accent_red: Color32::from_rgb(196, 88, 72),      // #C45848 (Cinnabar Rubric)
                accent_green: Color32::from_rgb(92, 158, 122),   // #5C9E7A (Malachite)
                accent_amber: Color32::from_rgb(206, 162, 78),   // #CEA24E (Ochre Gold / Sankir)
                accent_purple: Color32::from_rgb(158, 118, 152), // #9E7698 (Porphyry)
            }
        } else {
            // "Pergaminho" — Scriptorium Vellum, Iron-Gall Ink & Mineral Rubrics
            Self {
                outer_bg: Color32::from_rgb(228, 220, 205),      // #E4DCCD (Binding Linen)
                card_bg: Color32::from_rgb(245, 239, 228),       // #F5EFE4 (Cotton Parchment)
                card_header_bg: Color32::from_rgb(236, 229, 215), // #ECE5D7 (Folio Margin)
                extreme_bg: Color32::from_rgb(250, 246, 238),    // #FAF6EE (Clean Vellum)
                plot_bg: Color32::from_rgb(248, 243, 233),       // #F8F3E9 (Geometric Folio)
                sub_card_bg: Color32::from_rgb(239, 232, 219),   // #EFE8DB
                pill_bg: Color32::from_rgb(230, 222, 206),       // #E6DECE
                pill_hover: Color32::from_rgb(214, 204, 185),    // #D6CCB9
                pill_active: Color32::from_rgb(52, 86, 126),     // #34567E (Byzantine Ultramarine)
                border: Color32::from_rgb(198, 188, 170),        // #C6BCAA (Rule Line)
                text_primary: Color32::from_rgb(34, 29, 24),     // #221D18 (Iron-Gall Ink)
                text_secondary: Color32::from_rgb(92, 82, 70),   // #5C5246 (Walnut Sepia)
                text_muted: Color32::from_rgb(132, 120, 104),    // #847868 (Warm Earth Gray)
                accent_blue: Color32::from_rgb(46, 82, 126),     // #2E527E (Lapis Lazuli)
                accent_red: Color32::from_rgb(168, 56, 43),      // #A8382B (Cinnabar Rubric)
                accent_green: Color32::from_rgb(48, 114, 80),    // #307250 (Deep Malachite)
                accent_amber: Color32::from_rgb(166, 116, 34),   // #A67422 (Raw Ochre)
                accent_purple: Color32::from_rgb(118, 68, 106),  // #76446A (Imperial Porphyry)
            }
        }
    }
}

/// Applies the Dark Slate IDE / Bento Cards theme (or Light Mode equivalent) to `egui::Context`.
pub fn apply_bento_theme(ctx: &egui::Context, dark: bool) {
    let pal = ThemePalette::for_mode(dark);
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(6.0_f32, 5.0_f32);
    style.spacing.window_margin = Margin::same(8.0_f32);
    style.spacing.button_padding = Vec2::new(8.0_f32, 4.0_f32);
    style.spacing.slider_width = 115.0_f32;

    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(15.0_f32, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Body,
        FontId::new(12.8_f32, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Monospace,
        FontId::new(12.2_f32, FontFamily::Monospace),
    );
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(12.2_f32, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Small,
        FontId::new(11.0_f32, FontFamily::Proportional),
    );

    let mut visuals = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };

    visuals.panel_fill = pal.outer_bg;
    visuals.window_fill = pal.card_bg;
    visuals.window_rounding = Rounding::same(8.0_f32);
    visuals.window_stroke = Stroke::new(1.0_f32, pal.border);
    visuals.menu_rounding = Rounding::same(6.0_f32);
    visuals.extreme_bg_color = pal.extreme_bg;
    visuals.faint_bg_color = pal.sub_card_bg;
    visuals.code_bg_color = pal.extreme_bg;
    visuals.override_text_color = Some(pal.text_primary);
    visuals.slider_trailing_fill = true;

    // Noninteractive containers / separators
    visuals.widgets.noninteractive.bg_fill = pal.card_bg;
    visuals.widgets.noninteractive.weak_bg_fill = pal.card_bg;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, pal.border);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, pal.text_secondary);
    visuals.widgets.noninteractive.rounding = Rounding::same(6.0_f32);

    // Inactive buttons, pills & sliders
    visuals.widgets.inactive.bg_fill = pal.pill_bg;
    visuals.widgets.inactive.weak_bg_fill = pal.pill_bg;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, pal.border);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, pal.text_primary);
    visuals.widgets.inactive.rounding = Rounding::same(5.0_f32);

    // Hovered widgets
    visuals.widgets.hovered.bg_fill = pal.pill_hover;
    visuals.widgets.hovered.weak_bg_fill = pal.pill_hover;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, pal.accent_blue.gamma_multiply(0.75_f32));
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, pal.text_primary);
    visuals.widgets.hovered.rounding = Rounding::same(5.0_f32);

    // Active / pressed widgets
    visuals.widgets.active.bg_fill = pal.pill_active;
    visuals.widgets.active.weak_bg_fill = pal.pill_active;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0_f32, pal.accent_blue);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0_f32, Color32::WHITE);
    visuals.widgets.active.rounding = Rounding::same(5.0_f32);

    // Open dropdowns
    visuals.widgets.open.bg_fill = pal.pill_hover;
    visuals.widgets.open.weak_bg_fill = pal.pill_hover;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0_f32, pal.accent_blue);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0_f32, pal.text_primary);
    visuals.widgets.open.rounding = Rounding::same(5.0_f32);

    // Selection & slider trailing fill accent
    visuals.selection.bg_fill = pal.accent_blue.gamma_multiply(0.85_f32);
    visuals.selection.stroke = Stroke::new(1.0_f32, pal.accent_blue);

    style.visuals = visuals;
    ctx.set_style(style);
}

// =============================================================================
// REUSABLE BENTO CARD & HEADER HELPERS
// =============================================================================

/// Window control actions triggered from a Bento card's top-right header buttons.
#[derive(Debug, Clone, Copy, Default)]
pub struct CardHeaderAction {
    pub minimize_clicked: bool,
    pub popout_clicked: bool,
}

/// Renders a rounded Bento Card container (`Rounding::same(8.0)`, 1px subtle border, `card_bg`).
pub fn render_bento_card_frame<R>(
    ui: &mut Ui,
    pal: ThemePalette,
    size: Option<Vec2>,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> R {
    let frame = egui::Frame::none()
        .fill(pal.card_bg)
        .stroke(Stroke::new(1.0_f32, pal.border))
        .rounding(Rounding::same(8.0_f32))
        .inner_margin(Margin::same(0.0_f32));

    if let Some(sz) = size {
        ui.allocate_ui_with_layout(sz, egui::Layout::top_down(egui::Align::LEFT), |ui| {
            ui.set_min_size(sz);
            frame
                .show(ui, |ui| {
                    ui.set_min_size(sz);
                    ui.set_max_size(sz);
                    add_contents(ui)
                })
                .inner
        })
        .inner
    } else {
        let avail = ui.available_size();
        frame
            .show(ui, |ui| {
                ui.set_min_size(avail);
                add_contents(ui)
            })
            .inner
    }
}

/// Renders a Bento card's top header strip with selectable tabs on the left and window controls (`—`, `▢`) on the right.
#[allow(clippy::too_many_arguments)]
pub fn render_card_tab_header<T: Copy + PartialEq>(
    ui: &mut Ui,
    pal: ThemePalette,
    active_tab: &mut T,
    tabs: &[(T, &str)],
    minimized: bool,
    show_minimize: bool,
    show_popout: bool,
    min_tooltip: &str,
    popout_tooltip: &str,
) -> CardHeaderAction {
    let mut action = CardHeaderAction::default();
    let header_rounding = if minimized {
        Rounding::same(8.0_f32)
    } else {
        Rounding {
            nw: 8.0_f32,
            ne: 8.0_f32,
            sw: 0.0_f32,
            se: 0.0_f32,
        }
    };

    let resp = egui::Frame::none()
        .fill(pal.card_header_bg)
        .rounding(header_rounding)
        .inner_margin(Margin::symmetric(8.0_f32, 5.0_f32))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0_f32;
                for &(tab_val, label) in tabs {
                    let selected = *active_tab == tab_val;
                    let text = RichText::new(label)
                        .font(FontId::proportional(12.2_f32))
                        .color(if selected {
                            pal.text_primary
                        } else {
                            pal.text_secondary
                        });
                    let text = if selected { text.strong() } else { text };
                    let btn = egui::Button::new(text)
                        .fill(if selected {
                            pal.pill_bg
                        } else {
                            Color32::TRANSPARENT
                        })
                        .stroke(if selected {
                            Stroke::new(1.0_f32, pal.border)
                        } else {
                            Stroke::NONE
                        })
                        .rounding(Rounding::same(5.0_f32));
                    let tab_resp = ui.add(btn);
                    if selected {
                        // Crisp active-tab accent underline
                        let r = tab_resp.rect;
                        ui.painter().line_segment(
                            [
                                egui::Pos2::new(r.left() + 4.0_f32, r.bottom() - 0.5_f32),
                                egui::Pos2::new(r.right() - 4.0_f32, r.bottom() - 0.5_f32),
                            ],
                            Stroke::new(2.0_f32, pal.accent_blue),
                        );
                    }
                    if tab_resp.clicked() {
                        *active_tab = tab_val;
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 3.0_f32;
                    if show_popout
                        && ui
                            .add(header_icon_button("▢", pal.text_secondary))
                            .on_hover_text(popout_tooltip)
                            .clicked()
                    {
                        action.popout_clicked = true;
                    }
                    if show_minimize
                        && ui
                            .add(header_icon_button(
                                if minimized { "+" } else { "—" },
                                pal.text_secondary,
                            ))
                            .on_hover_text(min_tooltip)
                            .clicked()
                    {
                        action.minimize_clicked = true;
                    }
                });
            });
        })
        .response;

    if !minimized {
        let r = resp.rect;
        ui.painter().line_segment(
            [
                egui::Pos2::new(r.left(), r.bottom()),
                egui::Pos2::new(r.right(), r.bottom()),
            ],
            Stroke::new(1.0_f32, pal.border),
        );
        ui.add_space(-ui.spacing().item_spacing.y);
    }

    action
}

/// Renders a Bento card's top header strip with a single title + custom right controls.
pub fn render_card_title_header(
    ui: &mut Ui,
    pal: ThemePalette,
    title: &str,
    minimized: bool,
    extra_controls: impl FnOnce(&mut Ui),
) {
    let header_rounding = if minimized {
        Rounding::same(8.0_f32)
    } else {
        Rounding {
            nw: 8.0_f32,
            ne: 8.0_f32,
            sw: 0.0_f32,
            se: 0.0_f32,
        }
    };

    let resp = egui::Frame::none()
        .fill(pal.card_header_bg)
        .rounding(header_rounding)
        .inner_margin(Margin::symmetric(10.0_f32, 6.0_f32))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(title)
                        .font(FontId::proportional(13.0_f32))
                        .strong()
                        .color(pal.text_primary),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 4.0_f32;
                    extra_controls(ui);
                });
            });
        })
        .response;

    if !minimized {
        let r = resp.rect;
        ui.painter().line_segment(
            [
                egui::Pos2::new(r.left(), r.bottom()),
                egui::Pos2::new(r.right(), r.bottom()),
            ],
            Stroke::new(1.0_f32, pal.border),
        );
        ui.add_space(-ui.spacing().item_spacing.y);
    }
}

/// Renders a slim sub-toolbar strip right beneath a card header (with subtle bottom divider).
pub fn render_card_sub_toolbar(ui: &mut Ui, pal: ThemePalette, add_contents: impl FnOnce(&mut Ui)) {
    let resp = egui::Frame::none()
        .fill(pal.card_bg)
        .inner_margin(Margin::symmetric(8.0_f32, 4.0_f32))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0_f32;
                add_contents(ui);
            });
        })
        .response;

    let r = resp.rect;
    ui.painter().line_segment(
        [
            egui::Pos2::new(r.left(), r.bottom()),
            egui::Pos2::new(r.right(), r.bottom()),
        ],
        Stroke::new(1.0_f32, pal.border.gamma_multiply(0.75_f32)),
    );
    ui.add_space(-ui.spacing().item_spacing.y);
}

/// Compact borderless icon button for Bento card headers (`—`, `▢`, `<`, `⋮`).
pub fn header_icon_button(label: &str, color: Color32) -> egui::Button<'_> {
    egui::Button::new(
        RichText::new(label)
            .font(FontId::proportional(12.0_f32))
            .color(color),
    )
    .fill(Color32::TRANSPARENT)
    .stroke(Stroke::NONE)
    .rounding(Rounding::same(4.0_f32))
    .min_size(Vec2::new(20.0_f32, 18.0_f32))
}

/// Compact pill button for toolbars and quick actions.
pub fn pill_button(label: &str, active: bool, pal: ThemePalette) -> egui::Button<'_> {
    let text_color = if active {
        Color32::WHITE
    } else {
        pal.text_primary
    };
    egui::Button::new(
        RichText::new(label)
            .font(FontId::proportional(12.0_f32))
            .color(text_color),
    )
    .fill(if active { pal.pill_active } else { pal.pill_bg })
    .stroke(Stroke::new(
        1.0_f32,
        if active { pal.accent_blue } else { pal.border },
    ))
    .rounding(Rounding::same(6.0_f32))
}

/// Capsule frame (`Rounding::same(8.0)`, `#232B3D`) used for function and slider rows.
pub fn capsule_frame(pal: ThemePalette, border_color: Color32) -> egui::Frame {
    egui::Frame::none()
        .fill(pal.sub_card_bg)
        .stroke(Stroke::new(1.0_f32, border_color))
        .rounding(Rounding::same(8.0_f32))
        .inner_margin(Margin::symmetric(8.0_f32, 6.0_f32))
}

/// Curated mineral & manuscript curve swatch palette (Byzantine & Russian iconographic pigments).
pub const CURVE_COLOR_SWATCHES: &[Color32] = &[
    Color32::from_rgb(184, 74, 57),   // Cinnabar Rubric / Kinnabari (#B84A39)
    Color32::from_rgb(72, 118, 168),  // Lapis Lazuli (#4876A8)
    Color32::from_rgb(76, 142, 104),  // Malachite / Prasinos (#4C8E68)
    Color32::from_rgb(196, 152, 68),  // Ochre Gold / Chrysochra (#C49844)
    Color32::from_rgb(142, 92, 130),  // Imperial Porphyry (#8E5C82)
    Color32::from_rgb(168, 104, 62),  // Burnt Sienna (#A8683E)
    Color32::from_rgb(68, 132, 136),  // Azurite Verdigris (#448488)
    Color32::from_rgb(154, 74, 88),   // Madder Carmine / Rubia (#9A4A58)
    Color32::from_rgb(118, 136, 78),  // Monastic Olive / Elaia (#76884E)
    Color32::from_rgb(138, 112, 90),  // Raw Umber (#8A705A)
    Color32::from_rgb(214, 204, 184), // Limestone Chalk / Creta (#D6CCB8)
    Color32::from_rgb(128, 122, 112), // Silverpoint Lead / Plumbum (#807A70)
];

/// Draws a crisp, classical engraved 2D plot curve without synthetic neon glow,
/// modulated by `opacity` (`0.1..=1.0`).
#[allow(clippy::too_many_arguments)]
pub fn draw_glowing_plot_line_with_opacity(
    plot_ui: &mut PlotUi,
    points: Vec<[f64; 2]>,
    series_label: &str,
    color: Color32,
    stroke_width: f32,
    line_style: LineStyle,
    opacity: f32,
) {
    if points.is_empty() {
        return;
    }
    let alpha = opacity.clamp(0.1_f32, 1.0_f32);
    plot_ui.line(
        Line::new(PlotPoints::from(points))
            .name(series_label)
            .width(stroke_width)
            .style(line_style)
            .color(color.gamma_multiply(alpha)),
    );
}

/// Draws a crisp 2D plot curve at full opacity (`1.0`).
#[allow(dead_code)]
pub fn draw_glowing_plot_line(
    plot_ui: &mut PlotUi,
    points: Vec<[f64; 2]>,
    series_label: &str,
    color: Color32,
    stroke_width: f32,
    line_style: LineStyle,
) {
    draw_glowing_plot_line_with_opacity(
        plot_ui,
        points,
        series_label,
        color,
        stroke_width,
        line_style,
        1.0_f32,
    );
}

/// Draws the principal Cartesian axes `x = 0` and `y = 0` with sober manuscript rule lines,
/// highlighting active X/Y axes in warm ochre during `Shift` axis-zoom hover & drag.
pub fn draw_principal_axes_highlighted(
    plot_ui: &mut PlotUi,
    pal: ThemePalette,
    highlight_x: bool,
    highlight_y: bool,
) {
    let base_color = pal.text_muted.gamma_multiply(0.55_f32);
    let highlight_color = pal.accent_amber;

    if highlight_y {
        plot_ui.vline(VLine::new(0.0_f64).width(2.0_f32).color(highlight_color));
    } else {
        plot_ui.vline(VLine::new(0.0_f64).width(1.2_f32).color(base_color));
    }

    if highlight_x {
        plot_ui.hline(HLine::new(0.0_f64).width(2.0_f32).color(highlight_color));
    } else {
        plot_ui.hline(HLine::new(0.0_f64).width(1.2_f32).color(base_color));
    }
}

/// Draws the principal Cartesian axes `x = 0` and `y = 0` for crisp quadrant readability.
#[allow(dead_code)]
pub fn draw_principal_axes(plot_ui: &mut PlotUi, pal: ThemePalette) {
    draw_principal_axes_highlighted(plot_ui, pal, false, false);
}
