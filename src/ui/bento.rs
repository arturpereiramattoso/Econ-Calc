use eframe::egui::{
    self, Color32, FontFamily, FontId, Margin, RichText, Rounding, Stroke, TextStyle, Ui, Vec2,
};
use egui_plot::{HLine, Line, LineStyle, PlotPoints, PlotUi, VLine};

// =============================================================================
// DARK SLATE IDE / BENTO THEME PALETTE
// =============================================================================

#[derive(Debug, Clone, Copy)]
pub struct ThemePalette {
    /// Outer window backdrop behind Bento cards (`#11151E` dark).
    pub outer_bg: Color32,
    /// Primary Bento card surface (`#1A202C` dark).
    pub card_bg: Color32,
    /// Bento card header / tab bar strip (`#202736` dark).
    pub card_header_bg: Color32,
    /// Deep canvas / console / text input well (`#0E121A` dark).
    pub extreme_bg: Color32,
    /// Plot canvas background (`#0D1119` dark).
    pub plot_bg: Color32,
    /// Inner capsule cards for functions & sliders (`#232B3D` dark).
    pub sub_card_bg: Color32,
    /// Interactive pills & buttons (`#252D3F` dark).
    pub pill_bg: Color32,
    /// Hovered pills & buttons (`#313C54` dark).
    pub pill_hover: Color32,
    /// Active selection / focus blue (`#3B82F6` dark).
    pub pill_active: Color32,
    /// Subtle 1px card & divider border (`#2B3448` dark).
    pub border: Color32,
    /// Primary legible text (`#E2E8F0` dark).
    pub text_primary: Color32,
    /// Secondary labels & inactive tabs (`#94A3B8` dark).
    pub text_secondary: Color32,
    /// Muted hints & metadata (`#64748B` dark).
    pub text_muted: Color32,
    /// Accent blue for sliders, links, focus (`#60A5FA` dark).
    pub accent_blue: Color32,
    /// Coral / neon red for primary curve & errors (`#F87171` dark).
    pub accent_red: Color32,
    /// Emerald green for execution & equilibrium (`#34D399` dark).
    pub accent_green: Color32,
    /// Amber for warnings & calculus probes (`#FBBF24` dark).
    pub accent_amber: Color32,
    /// Purple for secondary diagnostics (`#A78BFA` dark).
    pub accent_purple: Color32,
}

impl ThemePalette {
    pub fn for_mode(dark: bool) -> Self {
        if dark {
            Self {
                outer_bg: Color32::from_rgb(17, 21, 30),         // #11151E
                card_bg: Color32::from_rgb(26, 32, 44),          // #1A202C
                card_header_bg: Color32::from_rgb(32, 39, 54),   // #202736
                extreme_bg: Color32::from_rgb(14, 18, 26),       // #0E121A
                plot_bg: Color32::from_rgb(13, 17, 25),          // #0D1119
                sub_card_bg: Color32::from_rgb(35, 43, 61),      // #232B3D
                pill_bg: Color32::from_rgb(37, 45, 63),          // #252D3F
                pill_hover: Color32::from_rgb(49, 60, 84),       // #313C54
                pill_active: Color32::from_rgb(59, 130, 246),    // #3B82F6
                border: Color32::from_rgb(43, 52, 72),           // #2B3448
                text_primary: Color32::from_rgb(226, 232, 240),  // #E2E8F0
                text_secondary: Color32::from_rgb(148, 163, 184), // #94A3B8
                text_muted: Color32::from_rgb(100, 116, 139),    // #64748B
                accent_blue: Color32::from_rgb(96, 165, 250),    // #60A5FA
                accent_red: Color32::from_rgb(248, 113, 113),    // #F87171
                accent_green: Color32::from_rgb(52, 211, 153),   // #34D399
                accent_amber: Color32::from_rgb(251, 191, 36),   // #FBBF24
                accent_purple: Color32::from_rgb(167, 139, 250), // #A78BFA
            }
        } else {
            Self {
                outer_bg: Color32::from_rgb(226, 232, 240),      // #E2E8F0
                card_bg: Color32::from_rgb(255, 255, 255),       // #FFFFFF
                card_header_bg: Color32::from_rgb(241, 245, 249), // #F1F5F9
                extreme_bg: Color32::from_rgb(248, 250, 252),    // #F8FAFC
                plot_bg: Color32::from_rgb(255, 255, 255),       // #FFFFFF
                sub_card_bg: Color32::from_rgb(241, 245, 249),   // #F1F5F9
                pill_bg: Color32::from_rgb(226, 232, 240),       // #E2E8F0
                pill_hover: Color32::from_rgb(203, 213, 225),    // #CBD5E1
                pill_active: Color32::from_rgb(37, 99, 235),     // #2563EB
                border: Color32::from_rgb(203, 213, 225),        // #CBD5E1
                text_primary: Color32::from_rgb(15, 23, 42),     // #0F172A
                text_secondary: Color32::from_rgb(71, 85, 105),  // #475569
                text_muted: Color32::from_rgb(100, 116, 139),    // #64748B
                accent_blue: Color32::from_rgb(37, 99, 235),     // #2563EB
                accent_red: Color32::from_rgb(220, 38, 38),      // #DC2626
                accent_green: Color32::from_rgb(5, 150, 105),    // #059669
                accent_amber: Color32::from_rgb(217, 119, 6),    // #D97706
                accent_purple: Color32::from_rgb(124, 58, 237),  // #7C3AED
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

/// Curated vibrant curve swatch palette tailored for Dark Slate & Light Bento canvases.
pub const CURVE_COLOR_SWATCHES: &[Color32] = &[
    Color32::from_rgb(248, 113, 113), // Coral Red (#F87171)
    Color32::from_rgb(56, 189, 248),  // Sky Blue (#38BDF8)
    Color32::from_rgb(52, 211, 153),  // Emerald (#34D399)
    Color32::from_rgb(167, 139, 250), // Violet (#A78BFA)
    Color32::from_rgb(251, 191, 36),  // Amber (#FBBF24)
    Color32::from_rgb(236, 72, 153),  // Rose Pink (#EC4899)
    Color32::from_rgb(45, 212, 191),  // Teal (#2DD4BF)
    Color32::from_rgb(251, 146, 60),  // Tangerine (#FB923C)
    Color32::from_rgb(96, 165, 250),  // Royal Blue (#60A5FA)
    Color32::from_rgb(163, 230, 53),  // Lime (#A3E635)
    Color32::from_rgb(226, 232, 240), // Ice Slate (#E2E8F0)
    Color32::from_rgb(148, 163, 184), // Cool Gray (#94A3B8)
];

/// Draws a neon-glowing 2D plot curve: soft translucent aura (`3x` stroke width)
/// underneath the crisp primary curve stroke, modulated by `opacity` (`0.1..=1.0`).
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
    // Outer soft glow pass (unnamed so legend shows only a single entry per curve)
    plot_ui.line(
        Line::new(PlotPoints::from(points.clone()))
            .width(stroke_width * 3.0_f32)
            .style(line_style)
            .color(color.gamma_multiply(0.20_f32 * alpha)),
    );
    // Crisp core stroke
    plot_ui.line(
        Line::new(PlotPoints::from(points))
            .name(series_label)
            .width(stroke_width)
            .style(line_style)
            .color(color.gamma_multiply(alpha)),
    );
}

/// Draws a neon-glowing 2D plot curve at full opacity (`1.0`).
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

/// Highlights the principal Cartesian axes `x = 0` and `y = 0`, optionally illuminating
/// X and/or Y with a cyan/blue glow during `Shift` axis-zoom hover & drag.
pub fn draw_principal_axes_highlighted(
    plot_ui: &mut PlotUi,
    pal: ThemePalette,
    highlight_x: bool,
    highlight_y: bool,
) {
    let base_color = pal.text_muted.gamma_multiply(0.55_f32);
    let glow_color = Color32::from_rgb(56, 189, 248);

    if highlight_y {
        plot_ui.vline(
            VLine::new(0.0_f64)
                .width(6.5_f32)
                .color(glow_color.gamma_multiply(0.24_f32)),
        );
        plot_ui.vline(VLine::new(0.0_f64).width(2.2_f32).color(glow_color));
    } else {
        plot_ui.vline(VLine::new(0.0_f64).width(1.25_f32).color(base_color));
    }

    if highlight_x {
        plot_ui.hline(
            HLine::new(0.0_f64)
                .width(6.5_f32)
                .color(glow_color.gamma_multiply(0.24_f32)),
        );
        plot_ui.hline(HLine::new(0.0_f64).width(2.2_f32).color(glow_color));
    } else {
        plot_ui.hline(HLine::new(0.0_f64).width(1.25_f32).color(base_color));
    }
}

/// Highlights the principal Cartesian axes `x = 0` and `y = 0` for crisp quadrant readability.
#[allow(dead_code)]
pub fn draw_principal_axes(plot_ui: &mut PlotUi, pal: ThemePalette) {
    draw_principal_axes_highlighted(plot_ui, pal, false, false);
}
