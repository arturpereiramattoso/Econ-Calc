use super::compiler::{CompiledExpr, EvalContext, EvalScratch};

/// Classification for automatically discovered Points of Interest on a 2D graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoiKind {
    Root,
    YIntercept,
    LocalMax,
    LocalMin,
    Intersection,
    #[allow(dead_code)]
    Tangency,
}

impl PoiKind {
    pub fn label(self) -> &'static str {
        match self {
            PoiKind::Root => "Root",
            PoiKind::YIntercept => "Y-Intercept",
            PoiKind::LocalMax => "Local Max",
            PoiKind::LocalMin => "Local Min",
            PoiKind::Intersection => "Intersection",
            PoiKind::Tangency => "Tangency",
        }
    }
}

/// An automatically detected mathematical Point of Interest on a curve or pair of curves.
#[derive(Debug, Clone, PartialEq)]
pub struct PointOfInterest {
    pub x: f64,
    pub y: f64,
    pub kind: PoiKind,
    pub label: String,
}

/// Finds all real roots of `f(x) = 0` inside `[x_min, x_max]` using adaptive sign-change
/// scanning + Brent's root-finding method, plus near-zero local minimum polishing for
/// even-multiplicity tangent roots (e.g. `(x - 2)^2 = 0`).
pub fn find_real_roots_1d<F>(mut f: F, x_min: f64, x_max: f64, scan_steps: usize) -> Vec<f64>
where
    F: FnMut(f64) -> Option<f64>,
{
    if !x_min.is_finite() || !x_max.is_finite() || x_max <= x_min {
        return Vec::new();
    }
    let steps = scan_steps.clamp(64, 1200);
    let span = x_max - x_min;
    let dx = span / (steps as f64);

    let mut samples: Vec<(f64, Option<f64>)> = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let x = x_min + dx * (i as f64);
        let y = f(x).filter(|v| v.is_finite());
        samples.push((x, y));
    }

    let mut roots: Vec<f64> = Vec::new();
    let dedup_tol = (span * 1e-4_f64).max(1e-5_f64);

    let push_unique = |roots: &mut Vec<f64>, r: f64| {
        if r.is_finite()
            && r >= x_min - 1e-7_f64
            && r <= x_max + 1e-7_f64
            && !roots.iter().any(|&ex| (ex - r).abs() < dedup_tol)
        {
            roots.push(if r.abs() < 1e-11_f64 { 0.0_f64 } else { r });
        }
    };

    // 1. Exact/near-zero sample hits & bracketed sign changes refined with Brent's method
    for i in 0..steps {
        let (x0, y0_opt) = samples[i];
        let (x1, y1_opt) = samples[i + 1];
        let (Some(y0), Some(y1)) = (y0_opt, y1_opt) else {
            continue;
        };

        if y0.abs() < 1e-10_f64 {
            push_unique(&mut roots, x0);
            continue;
        }
        if y1.abs() < 1e-10_f64 {
            push_unique(&mut roots, x1);
            continue;
        }

        if (y0 > 0.0_f64 && y1 < 0.0_f64) || (y0 < 0.0_f64 && y1 > 0.0_f64) {
            // Guard against vertical asymptote sign flip (e.g. `1 / x`)
            if let Some(root) = brent_root(&mut f, x0, x1, y0, y1, 1e-10_f64, 48) {
                if let Some(check) = f(root) {
                    if check.is_finite() && check.abs() < 1e-4_f64 {
                        push_unique(&mut roots, root);
                    }
                }
            }
        }
    }

    // 2. Detect tangent / even-multiplicity roots where |f(x)| reaches a local minimum near 0
    for i in 1..steps {
        let (xm, ym_opt) = samples[i];
        let (xl, yl_opt) = samples[i - 1];
        let (xr, yr_opt) = samples[i + 1];
        let (Some(yl), Some(ym), Some(yr)) = (yl_opt, ym_opt, yr_opt) else {
            continue;
        };
        if ym.abs() <= yl.abs()
            && ym.abs() <= yr.abs()
            && ym.abs() < 0.25_f64 * (yl.abs() + yr.abs() + 1e-3_f64)
        {
            if let Some((best_x, best_abs)) = golden_min_abs(&mut f, xl, xr, 36) {
                if best_abs < 1e-6_f64 && (best_x - xm).abs() <= dx * 1.5_f64 {
                    push_unique(&mut roots, best_x);
                }
            }
        }
    }

    roots.sort_by(|a, b| a.total_cmp(b));
    roots
}

/// Brent's root-finding algorithm on a sign-changing bracket `[a, b]`.
fn brent_root<F>(
    f: &mut F,
    mut a: f64,
    mut b: f64,
    mut fa: f64,
    mut fb: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64>
where
    F: FnMut(f64) -> Option<f64>,
{
    if fa * fb > 0.0_f64 {
        return None;
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    for _ in 0..max_iter {
        if (fb > 0.0_f64 && fc > 0.0_f64) || (fb < 0.0_f64 && fc < 0.0_f64) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 2.0_f64 * f64::EPSILON * b.abs() + 0.5_f64 * tol;
        let xm = 0.5_f64 * (c - b);
        if xm.abs() <= tol1 || fb.abs() < 1e-13_f64 {
            return Some(b);
        }

        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p, q) = if (a - c).abs() < 1e-15_f64 {
                (2.0_f64 * xm * s, 1.0_f64 - s)
            } else {
                let q_r = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0_f64 * xm * q_r * (q_r - r) - (b - a) * (r - 1.0_f64)),
                    (q_r - 1.0_f64) * (r - 1.0_f64) * (s - 1.0_f64),
                )
            };
            let (p, q) = if p > 0.0_f64 { (p, -q) } else { (-p, q) };
            let min1 = 3.0_f64 * xm * q - (tol1 * q).abs();
            let min2 = (e * q).abs();
            if 2.0_f64 * p < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }

        a = b;
        fa = fb;
        if d.abs() > tol1 {
            b += d;
        } else {
            b += tol1.copysign(xm);
        }
        fb = f(b).filter(|v| v.is_finite())?;
    }
    Some(b)
}

/// Golden-section search minimizing `|f(x)|` over `[lo, hi]` for even-multiplicity roots.
fn golden_min_abs<F>(f: &mut F, mut lo: f64, mut hi: f64, iters: usize) -> Option<(f64, f64)>
where
    F: FnMut(f64) -> Option<f64>,
{
    let phi = 0.5_f64 * (5.0_f64.sqrt() - 1.0_f64);
    let mut c = hi - phi * (hi - lo);
    let mut d = lo + phi * (hi - lo);
    let mut fc = f(c)?.abs();
    let mut fd = f(d)?.abs();

    for _ in 0..iters {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - phi * (hi - lo);
            fc = f(c)?.abs();
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + phi * (hi - lo);
            fd = f(d)?.abs();
        }
    }
    let best_x = 0.5_f64 * (lo + hi);
    let best_abs = f(best_x)?.abs();
    Some((best_x, best_abs))
}

/// Evaluates a compiled 1D expression at `x` with reusable scratch buffers.
pub fn eval_compiled_at_x(
    compiled: &CompiledExpr,
    ctx: &EvalContext<'_>,
    x: f64,
    scratch: &mut EvalScratch,
) -> Option<f64> {
    compiled
        .eval(
            ctx,
            &[
                ("x", x),
                ("L", x),
                ("Q", x),
                ("Y", x),
                ("y", 0.0_f64),
                ("K", 0.0_f64),
            ],
            scratch,
        )
        .ok()
        .filter(|v| v.is_finite())
}

/// Solves all real roots of `diff_compiled(x) = 0` inside `[x_min, x_max]`.
pub fn solve_compiled_roots_1d(
    diff_compiled: &CompiledExpr,
    ctx: &EvalContext<'_>,
    x_min: f64,
    x_max: f64,
    scan_steps: usize,
) -> Vec<f64> {
    let mut scratch = EvalScratch::new();
    find_real_roots_1d(
        |x| eval_compiled_at_x(diff_compiled, ctx, x, &mut scratch),
        x_min,
        x_max,
        scan_steps,
    )
}

/// Converts a finite floating-point number to a crisp rational fraction `(p, q)` using
/// continued fractions when within `tol` and `q <= max_den`.
/// Returns `None` for integers (`q == 1`), non-finite values, or irrational numbers.
pub fn float_to_fraction(val: f64, max_den: i64, tol: f64) -> Option<(i64, i64)> {
    if !val.is_finite() || val.abs() > 1e9_f64 {
        return None;
    }
    if (val - val.round()).abs() < tol {
        return None; // Already an exact integer
    }
    let sign = if val < 0.0_f64 { -1_i64 } else { 1_i64 };
    let x0 = val.abs();

    let mut p0: i64 = 0;
    let mut q0: i64 = 1;
    let mut p1: i64 = 1;
    let mut q1: i64 = 0;
    let mut x = x0;

    for _ in 0..32 {
        let a = x.floor() as i64;
        let p2 = a.checked_mul(p1).and_then(|v| v.checked_add(p0))?;
        let q2 = a.checked_mul(q1).and_then(|v| v.checked_add(q0))?;
        if q2 > max_den {
            break;
        }
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        let approx = (p1 as f64) / (q1 as f64);
        if (approx - x0).abs() <= tol * (1.0_f64 + x0) {
            if q1 > 1 {
                return Some((sign * p1, q1));
            }
            return None;
        }
        let frac = x - (a as f64);
        if frac.abs() < 1e-14_f64 {
            break;
        }
        x = 1.0_f64 / frac;
    }

    if q1 > 1 {
        let approx = (p1 as f64) / (q1 as f64);
        if (approx - x0).abs() <= tol * (1.0_f64 + x0) {
            return Some((sign * p1, q1));
        }
    }
    None
}

/// Automatically detects Points of Interest (Roots, Y-Intercept, Local Max/Min, Intersections)
/// for a target compiled 1D curve against a list of other visible compiled 1D curves.
#[allow(clippy::too_many_arguments)]
pub fn detect_curve_pois(
    target_name: &str,
    target: &CompiledExpr,
    others: &[(&str, &CompiledExpr)],
    ctx: &EvalContext<'_>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Vec<PointOfInterest> {
    let mut pois = Vec::new();
    if x_max <= x_min || !x_min.is_finite() || !x_max.is_finite() {
        return pois;
    }
    let y_margin = (y_max - y_min).abs().max(2.0_f64) * 0.15_f64;
    let in_y = |y: f64| y.is_finite() && y >= y_min - y_margin && y <= y_max + y_margin;

    // 1. Roots: f(x) = 0
    for rx in solve_compiled_roots_1d(target, ctx, x_min, x_max, 220) {
        if in_y(0.0_f64) {
            pois.push(PointOfInterest {
                x: rx,
                y: 0.0_f64,
                kind: PoiKind::Root,
                label: format!("Root ({:.3}, 0)", rx),
            });
        }
        if pois.len() >= 12 {
            break;
        }
    }

    // 2. Y-Intercept: (0, f(0))
    if x_min <= 0.0_f64 && x_max >= 0.0_f64 {
        let mut scratch = EvalScratch::new();
        if let Some(y0) = eval_compiled_at_x(target, ctx, 0.0_f64, &mut scratch) {
            if in_y(y0) && y0.abs() > 1e-5_f64 {
                pois.push(PointOfInterest {
                    x: 0.0_f64,
                    y: y0,
                    kind: PoiKind::YIntercept,
                    label: format!("Y-Intercept (0, {:.3})", y0),
                });
            }
        }
    }

    // 3. Local Extrema: f'(x) = 0
    let h = ((x_max - x_min) * 1e-5_f64).clamp(1e-6_f64, 1e-3_f64);
    let mut d_scratch = EvalScratch::new();
    let crit_xs = find_real_roots_1d(
        |x| {
            let fp = eval_compiled_at_x(target, ctx, x + h, &mut d_scratch)?;
            let fm = eval_compiled_at_x(target, ctx, x - h, &mut d_scratch)?;
            Some((fp - fm) / (2.0_f64 * h))
        },
        x_min + 2.0_f64 * h,
        x_max - 2.0_f64 * h,
        200,
    );

    let h2 = (h * 4.0_f64).clamp(1e-5_f64, 1e-2_f64);
    for cx in crit_xs.into_iter().take(8) {
        let (Some(y0), Some(yp), Some(ym)) = (
            eval_compiled_at_x(target, ctx, cx, &mut d_scratch),
            eval_compiled_at_x(target, ctx, cx + h2, &mut d_scratch),
            eval_compiled_at_x(target, ctx, cx - h2, &mut d_scratch),
        ) else {
            continue;
        };
        if !in_y(y0) {
            continue;
        }
        let d2 = (yp - 2.0_f64 * y0 + ym) / (h2 * h2);
        if d2.abs() < 1e-5_f64 {
            continue; // inflection / flat line, skip
        }
        let kind = if d2 < 0.0_f64 {
            PoiKind::LocalMax
        } else {
            PoiKind::LocalMin
        };
        pois.push(PointOfInterest {
            x: cx,
            y: y0,
            kind,
            label: format!("{} ({:.3}, {:.3})", kind.label(), cx, y0),
        });
    }

    // 4. Intersections against every other visible 1D curve: f(x) - g(x) = 0
    let mut s_a = EvalScratch::new();
    let mut s_b = EvalScratch::new();
    for &(other_name, other_expr) in others {
        let xs = find_real_roots_1d(
            |x| {
                let ya = eval_compiled_at_x(target, ctx, x, &mut s_a)?;
                let yb = eval_compiled_at_x(other_expr, ctx, x, &mut s_b)?;
                Some(ya - yb)
            },
            x_min,
            x_max,
            200,
        );
        for ix in xs.into_iter().take(6) {
            if let Some(iy) = eval_compiled_at_x(target, ctx, ix, &mut s_a) {
                if in_y(iy) {
                    pois.push(PointOfInterest {
                        x: ix,
                        y: iy,
                        kind: PoiKind::Intersection,
                        label: format!("{} ∩ {} ({:.3}, {:.3})", target_name, other_name, ix, iy),
                    });
                }
            }
        }
    }

    pois
}
