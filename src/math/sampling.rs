use super::ast::CmpOp;
use super::compiler::{CompiledExpr, EvalContext, EvalScratch};

/// Samples a compiled 1D expression `y = f(x)` across `[sample_min, sample_max]` with
/// asymptote and piecewise jump-discontinuity splitting.
#[allow(clippy::too_many_arguments)]
pub fn sample_1d_branches(
    compiled: &CompiledExpr,
    ctx: &EvalContext<'_>,
    sample_min: f64,
    sample_max: f64,
    vis_y_min: f64,
    vis_y_max: f64,
    samples: usize,
    asymptote_jump_factor: f64,
) -> Vec<Vec<[f64; 2]>> {
    let samples = samples.clamp(32, 4000);
    let span_x = (sample_max - sample_min).max(1e-6_f64);
    let span_y = (vis_y_max - vis_y_min).abs().max(2.0_f64);
    let y_margin = span_y * 2.5_f64;
    let y_low = vis_y_min - y_margin;
    let y_high = vis_y_max + y_margin;
    let jump_limit = span_y * asymptote_jump_factor.max(1.5_f64);
    // Secondary threshold for sharp piecewise jumps (e.g. step function jumping by 2 within 1 sample)
    let step_jump_limit = span_y * 0.25_f64;

    let mut scratch = EvalScratch::new();
    compiled.bind_slots(
        ctx,
        &[
            ("x", sample_min),
            ("y", 10.0_f64),
            ("L", sample_min),
            ("K", 10.0_f64),
            ("Q", sample_min),
            ("Y", sample_min),
        ],
        &mut scratch,
    );

    let x_slots: Vec<usize> = ["x", "L", "Q", "Y"]
        .iter()
        .filter_map(|name| compiled.slot_of(name))
        .collect();

    let eval_at_x = |x: f64, scr: &mut EvalScratch| -> Option<f64> {
        for &slot in &x_slots {
            compiled.set_slot(slot, x, scr);
        }
        compiled.eval_with_bound_slots(ctx, scr).ok()
    };

    let mut branches: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut current_branch: Vec<[f64; 2]> = Vec::new();
    let mut prev_pt: Option<[f64; 2]> = None;

    for i in 0..=samples {
        let x = sample_min + span_x * (i as f64) / (samples as f64);
        let Some(y) = eval_at_x(x, &mut scratch) else {
            if !current_branch.is_empty() {
                branches.push(std::mem::take(&mut current_branch));
            }
            prev_pt = None;
            continue;
        };

        if !y.is_finite() || y < y_low || y > y_high {
            if let Some([px, py]) = prev_pt {
                if y.is_finite() && (py >= y_low && py <= y_high) && (y - py).abs() < jump_limit {
                    let target_y = y.clamp(y_low, y_high);
                    let t = ((target_y - py) / (y - py)).clamp(0.0_f64, 1.0_f64);
                    let cx = px + t * (x - px);
                    current_branch.push([cx, target_y]);
                }
            }
            if !current_branch.is_empty() {
                branches.push(std::mem::take(&mut current_branch));
            }
            prev_pt = None;
            continue;
        }

        if let Some([px, py]) = prev_pt {
            let dy = (y - py).abs();
            if dy > jump_limit {
                let mid_x = 0.5_f64 * (px + x);
                let mid_y = eval_at_x(mid_x, &mut scratch).unwrap_or(f64::NAN);
                let is_monotonic = mid_y.is_finite()
                    && ((mid_y >= py && mid_y <= y) || (mid_y <= py && mid_y >= y));
                if !is_monotonic && !current_branch.is_empty() {
                    branches.push(std::mem::take(&mut current_branch));
                }
            } else if dy > step_jump_limit {
                // Detect piecewise jump discontinuity via 6-step interval halving:
                // on a smooth curve, |f(hi) - f(lo)| shrinks ~64x; at a jump it stays ~dy.
                let mut lo_x = px;
                let mut lo_y = py;
                let mut hi_x = x;
                let mut hi_y = y;
                let mut discontinuous = false;
                for _ in 0..6 {
                    let mid_x = 0.5_f64 * (lo_x + hi_x);
                    let Some(mid_y) = eval_at_x(mid_x, &mut scratch) else {
                        discontinuous = true;
                        break;
                    };
                    if !mid_y.is_finite() {
                        discontinuous = true;
                        break;
                    }
                    if (mid_y - lo_y).abs() >= (hi_y - mid_y).abs() {
                        hi_x = mid_x;
                        hi_y = mid_y;
                    } else {
                        lo_x = mid_x;
                        lo_y = mid_y;
                    }
                }
                if (hi_y - lo_y).abs() > 0.35_f64 * dy {
                    discontinuous = true;
                }
                if discontinuous && !current_branch.is_empty() {
                    branches.push(std::mem::take(&mut current_branch));
                }
            }
        }

        current_branch.push([x, y]);
        prev_pt = Some([x, y]);
    }

    if !current_branch.is_empty() {
        branches.push(current_branch);
    }

    branches
}

/// Samples a parametric curve `(x(t), y(t))` over `[t_min, t_max]`.
#[allow(clippy::too_many_arguments)]
pub fn sample_parametric_branches(
    x_compiled: &CompiledExpr,
    y_compiled: &CompiledExpr,
    param_name: &str,
    ctx: &EvalContext<'_>,
    t_min: f64,
    t_max: f64,
    samples: usize,
) -> Vec<Vec<[f64; 2]>> {
    let samples = samples.clamp(32, 4000);
    let span_t = (t_max - t_min).max(1e-6_f64);

    let mut x_scratch = EvalScratch::new();
    let mut y_scratch = EvalScratch::new();
    x_compiled.bind_slots(ctx, &[(param_name, t_min)], &mut x_scratch);
    y_compiled.bind_slots(ctx, &[(param_name, t_min)], &mut y_scratch);
    let x_slot = x_compiled.slot_of(param_name);
    let y_slot = y_compiled.slot_of(param_name);

    let mut branches = Vec::new();
    let mut current = Vec::new();

    for i in 0..=samples {
        let t = t_min + span_t * (i as f64) / (samples as f64);
        if let Some(s) = x_slot {
            x_compiled.set_slot(s, t, &mut x_scratch);
        }
        if let Some(s) = y_slot {
            y_compiled.set_slot(s, t, &mut y_scratch);
        }

        let x_val = x_compiled.eval_with_bound_slots(ctx, &mut x_scratch).ok();
        let y_val = y_compiled.eval_with_bound_slots(ctx, &mut y_scratch).ok();

        match (x_val, y_val) {
            (Some(x), Some(y)) if x.is_finite() && y.is_finite() && x.abs() < 1e6 && y.abs() < 1e6 => {
                current.push([x, y]);
            }
            _ => {
                if current.len() >= 2 {
                    branches.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
    }

    if current.len() >= 2 {
        branches.push(current);
    }

    branches
}

/// Samples a polar curve `r = f(theta)` over `[theta_min, theta_max]` in Cartesian coordinates.
pub fn sample_polar_branches(
    r_compiled: &CompiledExpr,
    param_name: &str,
    ctx: &EvalContext<'_>,
    theta_min: f64,
    theta_max: f64,
    samples: usize,
) -> Vec<Vec<[f64; 2]>> {
    let samples = samples.clamp(32, 4000);
    let span = (theta_max - theta_min).max(1e-6_f64);

    let mut scratch = EvalScratch::new();
    r_compiled.bind_slots(
        ctx,
        &[(param_name, theta_min), ("theta", theta_min), ("t", theta_min)],
        &mut scratch,
    );
    let slots: Vec<usize> = [param_name, "theta", "t"]
        .iter()
        .filter_map(|n| r_compiled.slot_of(n))
        .collect();

    let mut branches = Vec::new();
    let mut current = Vec::new();

    for i in 0..=samples {
        let th = theta_min + span * (i as f64) / (samples as f64);
        for &s in &slots {
            r_compiled.set_slot(s, th, &mut scratch);
        }
        match r_compiled.eval_with_bound_slots(ctx, &mut scratch) {
            Ok(r) if r.is_finite() && r.abs() < 1e6_f64 => {
                current.push([r * th.cos(), r * th.sin()]);
            }
            _ => {
                if current.len() >= 2 {
                    branches.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
    }

    if current.len() >= 2 {
        branches.push(current);
    }

    branches
}

/// Extracts 0-level contour polylines for an implicit equation `F(x, y) - G(x, y) = 0`
/// using 2D Marching Squares with saddle-point disambiguation and segment chaining.
#[allow(clippy::too_many_arguments)]
pub fn sample_implicit_contours(
    diff_compiled: &CompiledExpr,
    ctx: &EvalContext<'_>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    grid_x: usize,
    grid_y: usize,
) -> Vec<Vec<[f64; 2]>> {
    let nx = grid_x.clamp(16, 180);
    let ny = grid_y.clamp(16, 180);
    let dx = (x_max - x_min).max(1e-6_f64) / (nx as f64);
    let dy = (y_max - y_min).max(1e-6_f64) / (ny as f64);

    let mut scratch = EvalScratch::new();
    diff_compiled.bind_slots(
        ctx,
        &[("x", x_min), ("y", y_min), ("L", x_min), ("K", y_min)],
        &mut scratch,
    );
    let x_slots: Vec<usize> = ["x", "L"]
        .iter()
        .filter_map(|n| diff_compiled.slot_of(n))
        .collect();
    let y_slots: Vec<usize> = ["y", "K"]
        .iter()
        .filter_map(|n| diff_compiled.slot_of(n))
        .collect();

    let eval_xy = |x: f64, y: f64, scr: &mut EvalScratch| -> f64 {
        for &s in &x_slots {
            diff_compiled.set_slot(s, x, scr);
        }
        for &s in &y_slots {
            diff_compiled.set_slot(s, y, scr);
        }
        diff_compiled
            .eval_with_bound_slots(ctx, scr)
            .unwrap_or(f64::NAN)
    };

    let cols = nx + 1;
    let rows = ny + 1;
    let mut grid = vec![f64::NAN; cols * rows];

    for j in 0..rows {
        let y = y_min + (j as f64) * dy;
        for i in 0..cols {
            let x = x_min + (i as f64) * dx;
            grid[j * cols + i] = eval_xy(x, y, &mut scratch);
        }
    }

    let lerp_pt = |x0: f64, y0: f64, v0: f64, x1: f64, y1: f64, v1: f64| -> [f64; 2] {
        let denom = v1 - v0;
        let t = if denom.abs() < 1e-15_f64 {
            0.5_f64
        } else {
            (-v0 / denom).clamp(0.0_f64, 1.0_f64)
        };
        [x0 + t * (x1 - x0), y0 + t * (y1 - y0)]
    };

    let mut segments: Vec<([f64; 2], [f64; 2])> = Vec::new();

    for j in 0..ny {
        let y0 = y_min + (j as f64) * dy;
        let y1 = y0 + dy;
        for i in 0..nx {
            let x0 = x_min + (i as f64) * dx;
            let x1 = x0 + dx;

            // Corners: v0 = BL, v1 = BR, v2 = TR, v3 = TL
            let v0 = grid[j * cols + i];
            let v1 = grid[j * cols + i + 1];
            let v2 = grid[(j + 1) * cols + i + 1];
            let v3 = grid[(j + 1) * cols + i];

            if !v0.is_finite() || !v1.is_finite() || !v2.is_finite() || !v3.is_finite() {
                continue;
            }

            let mut mask = 0u8;
            if v0 >= 0.0_f64 {
                mask |= 1;
            }
            if v1 >= 0.0_f64 {
                mask |= 2;
            }
            if v2 >= 0.0_f64 {
                mask |= 4;
            }
            if v3 >= 0.0_f64 {
                mask |= 8;
            }

            if mask == 0 || mask == 15 {
                continue;
            }

            // Edges: 0 = Bottom (v0-v1), 1 = Right (v1-v2), 2 = Top (v3-v2), 3 = Left (v0-v3)
            let edge_pt = |edge: u8| -> [f64; 2] {
                match edge {
                    0 => lerp_pt(x0, y0, v0, x1, y0, v1),
                    1 => lerp_pt(x1, y0, v1, x1, y1, v2),
                    2 => lerp_pt(x0, y1, v3, x1, y1, v2),
                    _ => lerp_pt(x0, y0, v0, x0, y1, v3),
                }
            };

            match mask {
                1 | 14 => segments.push((edge_pt(3), edge_pt(0))),
                2 | 13 => segments.push((edge_pt(0), edge_pt(1))),
                3 | 12 => segments.push((edge_pt(3), edge_pt(1))),
                4 | 11 => segments.push((edge_pt(1), edge_pt(2))),
                6 | 9 => segments.push((edge_pt(0), edge_pt(2))),
                7 | 8 => segments.push((edge_pt(3), edge_pt(2))),
                5 => {
                    let center = eval_xy(0.5_f64 * (x0 + x1), 0.5_f64 * (y0 + y1), &mut scratch);
                    if center >= 0.0_f64 {
                        segments.push((edge_pt(3), edge_pt(2)));
                        segments.push((edge_pt(0), edge_pt(1)));
                    } else {
                        segments.push((edge_pt(3), edge_pt(0)));
                        segments.push((edge_pt(1), edge_pt(2)));
                    }
                }
                10 => {
                    let center = eval_xy(0.5_f64 * (x0 + x1), 0.5_f64 * (y0 + y1), &mut scratch);
                    if center >= 0.0_f64 {
                        segments.push((edge_pt(3), edge_pt(0)));
                        segments.push((edge_pt(1), edge_pt(2)));
                    } else {
                        segments.push((edge_pt(3), edge_pt(2)));
                        segments.push((edge_pt(0), edge_pt(1)));
                    }
                }
                _ => {}
            }
        }
    }

    chain_contour_segments(segments, (dx.min(dy)) * 0.05_f64)
}

fn chain_contour_segments(
    segments: Vec<([f64; 2], [f64; 2])>,
    tol: f64,
) -> Vec<Vec<[f64; 2]>> {
    if segments.is_empty() {
        return Vec::new();
    }
    let tol2 = (tol.max(1e-7_f64)).powi(2);
    let close = |a: [f64; 2], b: [f64; 2]| -> bool {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        dx * dx + dy * dy <= tol2
    };

    let mut used = vec![false; segments.len()];
    let mut polylines: Vec<Vec<[f64; 2]>> = Vec::new();

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;
        let (p0, p1) = segments[start_idx];
        let mut chain = vec![p0, p1];

        // Extend forward
        while let Some(&tail) = chain.last() {
            let mut found = None;
            for (i, &(a, b)) in segments.iter().enumerate() {
                if used[i] {
                    continue;
                }
                if close(tail, a) {
                    found = Some((i, b));
                    break;
                } else if close(tail, b) {
                    found = Some((i, a));
                    break;
                }
            }
            if let Some((i, next_pt)) = found {
                used[i] = true;
                chain.push(next_pt);
            } else {
                break;
            }
        }

        // Extend backward
        while let Some(&head) = chain.first() {
            let mut found = None;
            for (i, &(a, b)) in segments.iter().enumerate() {
                if used[i] {
                    continue;
                }
                if close(head, a) {
                    found = Some((i, b));
                    break;
                } else if close(head, b) {
                    found = Some((i, a));
                    break;
                }
            }
            if let Some((i, prev_pt)) = found {
                used[i] = true;
                chain.insert(0, prev_pt);
            } else {
                break;
            }
        }

        polylines.push(chain);
    }

    polylines
}

/// Shaded inequality region + boundary curves for `y op f(x)` or general `F(x, y) op G(x, y)`.
#[derive(Debug, Clone)]
pub struct InequalityRegion {
    pub boundary_curves: Vec<Vec<[f64; 2]>>,
    pub shaded_polygons: Vec<Vec<[f64; 2]>>,
    pub strict: bool,
}

/// Samples a 1D inequality of the form `y < f(x)`, `y <= f(x)`, `y > f(x)`, `y >= f(x)`.
#[allow(clippy::too_many_arguments)]
pub fn sample_explicit_y_inequality(
    rhs_compiled: &CompiledExpr,
    op: CmpOp,
    ctx: &EvalContext<'_>,
    sample_min: f64,
    sample_max: f64,
    vis_y_min: f64,
    vis_y_max: f64,
    samples: usize,
    asymptote_jump_factor: f64,
) -> InequalityRegion {
    let branches = sample_1d_branches(
        rhs_compiled,
        ctx,
        sample_min,
        sample_max,
        vis_y_min,
        vis_y_max,
        samples,
        asymptote_jump_factor,
    );

    let shade_above = matches!(op, CmpOp::Gt | CmpOp::Ge);
    let strict = matches!(op, CmpOp::Lt | CmpOp::Gt);
    let bound_y = if shade_above { vis_y_max } else { vis_y_min };

    let mut polygons = Vec::with_capacity(branches.len());
    for branch in &branches {
        if branch.len() < 2 {
            continue;
        }
        let mut poly = Vec::with_capacity(branch.len() + 2);
        poly.push([branch[0][0], bound_y]);
        for &[x, y] in branch {
            poly.push([x, y.clamp(vis_y_min, vis_y_max)]);
        }
        poly.push([branch[branch.len() - 1][0], bound_y]);
        polygons.push(poly);
    }

    InequalityRegion {
        boundary_curves: branches,
        shaded_polygons: polygons,
        strict,
    }
}

/// Samples a general 2D inequality `F(x, y) op G(x, y)` (passed as compiled `F(x, y) - G(x, y)`).
#[allow(clippy::too_many_arguments)]
pub fn sample_implicit_2d_inequality(
    diff_compiled: &CompiledExpr,
    op: CmpOp,
    ctx: &EvalContext<'_>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    grid_x: usize,
    grid_y: usize,
) -> InequalityRegion {
    let boundary_curves =
        sample_implicit_contours(diff_compiled, ctx, x_min, x_max, y_min, y_max, grid_x, grid_y);
    let strict = matches!(op, CmpOp::Lt | CmpOp::Gt);

    // Build merged horizontal scanline rectangles where `diff op 0` holds
    let nx = (grid_x / 2).clamp(16, 64);
    let ny = (grid_y / 2).clamp(16, 64);
    let dx = (x_max - x_min).max(1e-6_f64) / (nx as f64);
    let dy = (y_max - y_min).max(1e-6_f64) / (ny as f64);

    let mut scratch = EvalScratch::new();
    diff_compiled.bind_slots(
        ctx,
        &[("x", x_min), ("y", y_min), ("L", x_min), ("K", y_min)],
        &mut scratch,
    );
    let x_slots: Vec<usize> = ["x", "L"]
        .iter()
        .filter_map(|n| diff_compiled.slot_of(n))
        .collect();
    let y_slots: Vec<usize> = ["y", "K"]
        .iter()
        .filter_map(|n| diff_compiled.slot_of(n))
        .collect();

    let mut polygons = Vec::new();
    for j in 0..ny {
        let y0 = y_min + (j as f64) * dy;
        let y1 = y0 + dy;
        let yc = 0.5_f64 * (y0 + y1);
        for &s in &y_slots {
            diff_compiled.set_slot(s, yc, &mut scratch);
        }

        let mut run_start: Option<f64> = None;
        for i in 0..nx {
            let x0 = x_min + (i as f64) * dx;
            let xc = x0 + 0.5_f64 * dx;
            for &s in &x_slots {
                diff_compiled.set_slot(s, xc, &mut scratch);
            }
            let val = diff_compiled
                .eval_with_bound_slots(ctx, &mut scratch)
                .unwrap_or(f64::NAN);
            let inside = op.eval(val, 0.0_f64);

            if inside {
                if run_start.is_none() {
                    run_start = Some(x0);
                }
            } else if let Some(rx0) = run_start.take() {
                polygons.push(vec![[rx0, y0], [x0, y0], [x0, y1], [rx0, y1]]);
            }
        }
        if let Some(rx0) = run_start.take() {
            polygons.push(vec![[rx0, y0], [x_max, y0], [x_max, y1], [rx0, y1]]);
        }
    }

    InequalityRegion {
        boundary_curves,
        shaded_polygons: polygons,
        strict,
    }
}
