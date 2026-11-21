use super::ast::{AccentKind, BinOp, CmpOp, MathError, MathExpr, Span, SpannedExpr, UnaryOp};
use super::graph::ParsedFunction;
use crate::collections::IndexMap;
use std::collections::HashMap;

/// External environment bridge (implemented by the R engine in Phase 3) so Desmos math
/// expressions can read R scalars, vectors, DataFrame columns (`df$col`), and call R functions.
pub trait MathExternalEnv {
    fn get_scalar(&self, _name: &str) -> Option<f64> {
        None
    }

    fn get_vector_element(&self, _name: &str, _index_1based: usize) -> Option<f64> {
        None
    }

    fn get_member_scalar(&self, _object: &str, _field: &str) -> Option<f64> {
        None
    }

    fn get_member_element(
        &self,
        _object: &str,
        _field: &str,
        _index_1based: usize,
    ) -> Option<f64> {
        None
    }

    fn call_function(&self, _name: &str, _args: &[f64]) -> Option<Result<f64, String>> {
        None
    }
}

#[allow(dead_code)]
pub struct NoopExternalEnv;
impl MathExternalEnv for NoopExternalEnv {}

#[derive(Debug, Clone)]
enum Op {
    PushConst(f64),
    LoadVar(usize),
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Abs,
    Sqrt,
    NthRoot,
    CmpChain {
        ops: Vec<CmpOp>,
    },
    Call {
        name: String,
        prime_order: u8,
        argc: usize,
        span: Span,
    },
    Derivative {
        var: String,
        order: u8,
        body_idx: usize,
        span: Span,
    },
    Integral {
        var: String,
        lower_idx: usize,
        upper_idx: usize,
        body_idx: usize,
        span: Span,
    },
    Sum {
        var: String,
        lower_idx: usize,
        upper_idx: usize,
        body_idx: usize,
    },
    Prod {
        var: String,
        lower_idx: usize,
        upper_idx: usize,
        body_idx: usize,
    },
    Piecewise {
        /// `(Option<cond_sub_idx>, val_sub_idx)`
        branches: Vec<(Option<usize>, usize)>,
    },
    Restricted {
        inner_idx: usize,
        cond_indices: Vec<usize>,
    },
    Tangent {
        fn_name: String,
        x0_idx: usize,
        span: Span,
    },
    Elasticity {
        fn_name: String,
        x_idx: usize,
        span: Span,
    },
    Marginal {
        fn_name: String,
        x_idx: usize,
        span: Span,
    },
    Average {
        fn_name: String,
        x_idx: usize,
        span: Span,
    },
    AreaBetween {
        f_name: String,
        g_name: String,
        a_idx: Option<usize>,
        b_idx: Option<usize>,
        span: Span,
    },
    MemberAccess {
        object: String,
        field: String,
        span: Span,
    },
    IndexVar {
        target_name: String,
        span: Span,
    },
    IndexMember {
        object: String,
        field: String,
        span: Span,
    },
}

/// Flat bytecode representation of a math expression with interned variable slots.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    ops: Vec<Op>,
    var_names: Vec<String>,
    sub_exprs: Vec<CompiledExpr>,
}

/// Reusable execution buffers to minimize heap allocations during 60 FPS curve sampling.
#[derive(Debug, Default, Clone)]
pub struct EvalScratch {
    stack: Vec<f64>,
    slots: Vec<f64>,
    slot_bound: Vec<bool>,
}

impl EvalScratch {
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(64),
            slots: Vec::with_capacity(32),
            slot_bound: Vec::with_capacity(32),
        }
    }
}

/// Compiled user-defined N-ary workspace function.
#[derive(Debug, Clone)]
pub struct CompiledUserFn {
    #[allow(dead_code)]
    pub name: String,
    pub params: Vec<String>,
    pub body: CompiledExpr,
}

impl CompiledUserFn {
    pub fn from_parsed(pf: &ParsedFunction) -> Self {
        Self {
            name: pf.name.clone(),
            params: pf.params.clone(),
            body: CompiledExpr::compile(&pf.body),
        }
    }
}

/// Runtime evaluation environment for compiled math expressions.
pub struct EvalContext<'a> {
    pub params: &'a IndexMap<String, f64>,
    pub functions: &'a HashMap<String, CompiledUserFn>,
    pub external: Option<&'a dyn MathExternalEnv>,
    pub diff_h: f64,
    pub integral_steps: usize,
    pub max_recursion_depth: usize,
}

impl<'a> EvalContext<'a> {
    pub fn new(
        params: &'a IndexMap<String, f64>,
        functions: &'a HashMap<String, CompiledUserFn>,
    ) -> Self {
        Self {
            params,
            functions,
            external: None,
            diff_h: 1e-5_f64,
            integral_steps: 128,
            max_recursion_depth: 16,
        }
    }

    pub fn with_external(mut self, external: &'a dyn MathExternalEnv) -> Self {
        self.external = Some(external);
        self
    }

    pub fn with_diff_h(mut self, h: f64) -> Self {
        self.diff_h = h.clamp(1e-9_f64, 1e-2_f64);
        self
    }

    pub fn with_integral_steps(mut self, steps: usize) -> Self {
        self.integral_steps = steps.clamp(8, 2000);
        self
    }
}

impl CompiledExpr {
    pub fn compile(expr: &SpannedExpr) -> Self {
        let mut builder = CompilerBuilder {
            ops: Vec::new(),
            var_names: Vec::new(),
            var_map: HashMap::new(),
            sub_exprs: Vec::new(),
        };
        builder.emit(expr);
        Self {
            ops: builder.ops,
            var_names: builder.var_names,
            sub_exprs: builder.sub_exprs,
        }
    }

    #[allow(dead_code)]
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.var_names.iter().position(|n| n == name)
    }

    /// Populates `scratch` slots from `ctx` and optional coordinate overrides (`x`, `y`, `t`, `theta`).
    pub fn bind_slots(
        &self,
        ctx: &EvalContext<'_>,
        overrides: &[(&str, f64)],
        scratch: &mut EvalScratch,
    ) {
        let n = self.var_names.len();
        scratch.slots.resize(n, f64::NAN);
        scratch.slot_bound.resize(n, false);

        for (i, name) in self.var_names.iter().enumerate() {
            if let Some((_, val)) = overrides.iter().rev().find(|(k, _)| *k == name.as_str()) {
                scratch.slots[i] = *val;
                scratch.slot_bound[i] = true;
                continue;
            }
            if let Some(&val) = lookup_param_with_alias(ctx.params, name) {
                scratch.slots[i] = val;
                scratch.slot_bound[i] = true;
                continue;
            }
            if let Some(ext) = ctx.external {
                if let Some(val) = ext.get_scalar(name) {
                    scratch.slots[i] = val;
                    scratch.slot_bound[i] = true;
                    continue;
                }
            }
            if let Some(c) = builtin_constant(name) {
                scratch.slots[i] = c;
                scratch.slot_bound[i] = true;
                continue;
            }
            scratch.slots[i] = f64::NAN;
            scratch.slot_bound[i] = false;
        }
    }

    /// Updates a single pre-resolved variable slot in O(1) without heap allocations.
    #[inline]
    pub fn set_slot(&self, slot: usize, val: f64, scratch: &mut EvalScratch) {
        if slot < scratch.slots.len() {
            scratch.slots[slot] = val;
            scratch.slot_bound[slot] = true;
        }
    }

    /// Evaluates using slots already populated in `scratch`.
    pub fn eval_with_bound_slots(
        &self,
        ctx: &EvalContext<'_>,
        scratch: &mut EvalScratch,
    ) -> Result<f64, MathError> {
        self.eval_internal(ctx, scratch, &[], 0)
    }

    /// Convenience one-shot evaluation with coordinate/local overrides.
    pub fn eval(
        &self,
        ctx: &EvalContext<'_>,
        overrides: &[(&str, f64)],
        scratch: &mut EvalScratch,
    ) -> Result<f64, MathError> {
        self.bind_slots(ctx, overrides, scratch);
        self.eval_internal(ctx, scratch, overrides, 0)
    }

    fn eval_internal(
        &self,
        ctx: &EvalContext<'_>,
        scratch: &mut EvalScratch,
        active_overrides: &[(&str, f64)],
        depth: usize,
    ) -> Result<f64, MathError> {
        if depth > ctx.max_recursion_depth {
            return Err(MathError::new(
                "Maximum function recursion depth exceeded",
                Span::default(),
            ));
        }
        scratch.stack.clear();

        for op in &self.ops {
            match op {
                Op::PushConst(v) => scratch.stack.push(*v),
                Op::LoadVar(slot) => {
                    if *slot < scratch.slot_bound.len() && scratch.slot_bound[*slot] {
                        scratch.stack.push(scratch.slots[*slot]);
                    } else {
                        let name = self
                            .var_names
                            .get(*slot)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        return Err(MathError::new(
                            format!("Undefined variable `{}`", name),
                            Span::default(),
                        ));
                    }
                }
                Op::Neg => {
                    let v = pop_or_err(&mut scratch.stack)?;
                    scratch.stack.push(-v);
                }
                Op::Add => {
                    let b = pop_or_err(&mut scratch.stack)?;
                    let a = pop_or_err(&mut scratch.stack)?;
                    scratch.stack.push(a + b);
                }
                Op::Sub => {
                    let b = pop_or_err(&mut scratch.stack)?;
                    let a = pop_or_err(&mut scratch.stack)?;
                    scratch.stack.push(a - b);
                }
                Op::Mul => {
                    let b = pop_or_err(&mut scratch.stack)?;
                    let a = pop_or_err(&mut scratch.stack)?;
                    scratch.stack.push(a * b);
                }
                Op::Div => {
                    let b = pop_or_err(&mut scratch.stack)?;
                    let a = pop_or_err(&mut scratch.stack)?;
                    scratch.stack.push(a / b);
                }
                Op::Pow => {
                    let b = pop_or_err(&mut scratch.stack)?;
                    let a = pop_or_err(&mut scratch.stack)?;
                    scratch.stack.push(a.powf(b));
                }
                Op::Mod => {
                    let b = pop_or_err(&mut scratch.stack)?;
                    let a = pop_or_err(&mut scratch.stack)?;
                    scratch.stack.push(a.rem_euclid(b));
                }
                Op::Abs => {
                    let v = pop_or_err(&mut scratch.stack)?;
                    scratch.stack.push(v.abs());
                }
                Op::Sqrt => {
                    let v = pop_or_err(&mut scratch.stack)?;
                    scratch.stack.push(v.sqrt());
                }
                Op::NthRoot => {
                    let rad = pop_or_err(&mut scratch.stack)?;
                    let idx = pop_or_err(&mut scratch.stack)?;
                    let val = if idx == 0.0_f64 {
                        f64::NAN
                    } else if rad < 0.0_f64 && (idx.round() - idx).abs() < 1e-9_f64 {
                        let n = idx.round() as i64;
                        if n % 2 != 0 {
                            -((-rad).powf(1.0_f64 / idx))
                        } else {
                            f64::NAN
                        }
                    } else {
                        rad.powf(1.0_f64 / idx)
                    };
                    scratch.stack.push(val);
                }
                Op::CmpChain { ops } => {
                    let count = ops.len() + 1;
                    if scratch.stack.len() < count {
                        return Err(MathError::new("Stack underflow in comparison", Span::default()));
                    }
                    let start = scratch.stack.len() - count;
                    let mut ok = true;
                    for (idx, cmp) in ops.iter().enumerate() {
                        let lhs = scratch.stack[start + idx];
                        let rhs = scratch.stack[start + idx + 1];
                        if !cmp.eval(lhs, rhs) {
                            ok = false;
                            break;
                        }
                    }
                    scratch.stack.truncate(start);
                    scratch.stack.push(if ok { 1.0_f64 } else { 0.0_f64 });
                }
                Op::Call {
                    name,
                    prime_order,
                    argc,
                    span,
                } => {
                    if scratch.stack.len() < *argc {
                        return Err(MathError::new("Stack underflow in function call", *span));
                    }
                    let start = scratch.stack.len() - *argc;
                    let args: Vec<f64> = scratch.stack.drain(start..).collect();
                    let val = eval_function_call(
                        name,
                        *prime_order,
                        &args,
                        *span,
                        ctx,
                        scratch,
                        &self.var_names,
                        active_overrides,
                        depth,
                    )?;
                    scratch.stack.push(val);
                }
                Op::Derivative {
                    var,
                    order,
                    body_idx,
                    span,
                } => {
                    let sub = &self.sub_exprs[*body_idx];
                    let x0 = resolve_var_value(
                        var,
                        ctx,
                        &self.var_names,
                        scratch,
                        active_overrides,
                    )
                    .unwrap_or(0.0_f64);
                    let mut child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let val = eval_numerical_derivative_compiled(
                        sub,
                        var,
                        x0,
                        *order,
                        ctx,
                        &mut child_overrides,
                        depth + 1,
                    )
                    .map_err(|mut e| {
                        if e.span == Span::default() {
                            e.span = *span;
                        }
                        e
                    })?;
                    scratch.stack.push(val);
                }
                Op::Integral {
                    var,
                    lower_idx,
                    upper_idx,
                    body_idx,
                    span,
                } => {
                    let mut child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let a = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*lower_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    let b = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*upper_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    if !a.is_finite() || !b.is_finite() {
                        scratch.stack.push(f64::NAN);
                        continue;
                    }
                    let val = eval_definite_integral_compiled(
                        &self.sub_exprs[*body_idx],
                        var,
                        a,
                        b,
                        ctx,
                        &mut child_overrides,
                        depth + 1,
                    )
                    .map_err(|mut e| {
                        if e.span == Span::default() {
                            e.span = *span;
                        }
                        e
                    })?;
                    scratch.stack.push(val);
                }
                Op::Sum {
                    var,
                    lower_idx,
                    upper_idx,
                    body_idx,
                    ..
                } => {
                    let mut child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let lo_f = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*lower_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    let hi_f = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*upper_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    let val = eval_aggregation_compiled(
                        &self.sub_exprs[*body_idx],
                        var,
                        lo_f,
                        hi_f,
                        true,
                        ctx,
                        &mut child_overrides,
                        depth + 1,
                    )?;
                    scratch.stack.push(val);
                }
                Op::Prod {
                    var,
                    lower_idx,
                    upper_idx,
                    body_idx,
                    ..
                } => {
                    let mut child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let lo_f = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*lower_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    let hi_f = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*upper_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    let val = eval_aggregation_compiled(
                        &self.sub_exprs[*body_idx],
                        var,
                        lo_f,
                        hi_f,
                        false,
                        ctx,
                        &mut child_overrides,
                        depth + 1,
                    )?;
                    scratch.stack.push(val);
                }
                Op::Piecewise { branches } => {
                    let child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let mut out = f64::NAN;
                    for (cond_opt, val_idx) in branches {
                        let matched = if let Some(cond_idx) = cond_opt {
                            let c = eval_sub_with_owned_overrides(
                                &self.sub_exprs[*cond_idx],
                                ctx,
                                &child_overrides,
                                &mut sub_scratch,
                                depth + 1,
                            )?;
                            c.is_finite() && c != 0.0_f64
                        } else {
                            true
                        };
                        if matched {
                            out = eval_sub_with_owned_overrides(
                                &self.sub_exprs[*val_idx],
                                ctx,
                                &child_overrides,
                                &mut sub_scratch,
                                depth + 1,
                            )?;
                            break;
                        }
                    }
                    scratch.stack.push(out);
                }
                Op::Restricted {
                    inner_idx,
                    cond_indices,
                } => {
                    let mut child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let val = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*inner_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    if !val.is_finite() {
                        scratch.stack.push(f64::NAN);
                        continue;
                    }
                    // Bind evaluated `y` value so inline range restrictions like `{x >= 0, y >= 0}` work!
                    if let Some(entry) = child_overrides.iter_mut().find(|(k, _)| k == "y") {
                        entry.1 = val;
                    } else {
                        child_overrides.push(("y".to_string(), val));
                    }
                    let mut all_ok = true;
                    for &c_idx in cond_indices {
                        let c = eval_sub_with_owned_overrides(
                            &self.sub_exprs[c_idx],
                            ctx,
                            &child_overrides,
                            &mut sub_scratch,
                            depth + 1,
                        )
                        .unwrap_or(0.0_f64);
                        if !c.is_finite() || c == 0.0_f64 {
                            all_ok = false;
                            break;
                        }
                    }
                    scratch.stack.push(if all_ok { val } else { f64::NAN });
                }
                Op::Tangent {
                    fn_name,
                    x0_idx,
                    span,
                } => {
                    let child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let x0 = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*x0_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    let x_cur = resolve_var_value(
                        "x",
                        ctx,
                        &self.var_names,
                        scratch,
                        active_overrides,
                    )
                    .unwrap_or(x0);
                    let (y0, slope) =
                        eval_fn_val_and_deriv(fn_name, x0, *span, ctx, depth + 1)?;
                    scratch.stack.push(y0 + slope * (x_cur - x0));
                }
                Op::Elasticity {
                    fn_name,
                    x_idx,
                    span,
                } => {
                    let child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let xv = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*x_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    let (y0, dy) = eval_fn_val_and_deriv(fn_name, xv, *span, ctx, depth + 1)?;
                    let elas = if y0.abs() < 1e-15_f64 {
                        f64::NAN
                    } else {
                        (dy * xv) / y0
                    };
                    scratch.stack.push(elas);
                }
                Op::Marginal {
                    fn_name,
                    x_idx,
                    span,
                } => {
                    let child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let xv = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*x_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    let (_, dy) = eval_fn_val_and_deriv(fn_name, xv, *span, ctx, depth + 1)?;
                    scratch.stack.push(dy);
                }
                Op::Average {
                    fn_name,
                    x_idx,
                    span,
                } => {
                    let child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let xv = eval_sub_with_owned_overrides(
                        &self.sub_exprs[*x_idx],
                        ctx,
                        &child_overrides,
                        &mut sub_scratch,
                        depth + 1,
                    )?;
                    let (y0, _) = eval_fn_val_and_deriv(fn_name, xv, *span, ctx, depth + 1)?;
                    let avg = if xv.abs() < 1e-15_f64 {
                        f64::NAN
                    } else {
                        y0 / xv
                    };
                    scratch.stack.push(avg);
                }
                Op::AreaBetween {
                    f_name,
                    g_name,
                    a_idx,
                    b_idx,
                    span,
                } => {
                    let child_overrides = build_inherited_overrides(
                        &self.var_names,
                        scratch,
                        active_overrides,
                    );
                    let mut sub_scratch = EvalScratch::new();
                    let (a, b) = if let (Some(ai), Some(bi)) = (a_idx, b_idx) {
                        let av = eval_sub_with_owned_overrides(
                            &self.sub_exprs[*ai],
                            ctx,
                            &child_overrides,
                            &mut sub_scratch,
                            depth + 1,
                        )?;
                        let bv = eval_sub_with_owned_overrides(
                            &self.sub_exprs[*bi],
                            ctx,
                            &child_overrides,
                            &mut sub_scratch,
                            depth + 1,
                        )?;
                        (av, bv)
                    } else {
                        let eq_x = find_surplus_equilibrium_x(f_name, g_name, ctx, *span, depth + 1)?;
                        (0.0_f64, eq_x)
                    };
                    let val = eval_area_between_fns(f_name, g_name, a, b, *span, ctx, depth + 1)?;
                    scratch.stack.push(val);
                }
                Op::MemberAccess {
                    object,
                    field,
                    span,
                } => {
                    let ext = ctx.external.ok_or_else(|| {
                        MathError::new(
                            format!("Cannot resolve `{}${}` without R workspace", object, field),
                            *span,
                        )
                    })?;
                    let val = ext.get_member_scalar(object, field).ok_or_else(|| {
                        MathError::new(
                            format!("Member `{}${}` not found or not numeric", object, field),
                            *span,
                        )
                    })?;
                    scratch.stack.push(val);
                }
                Op::IndexVar { target_name, span } => {
                    let idx_f = pop_or_err(&mut scratch.stack)?;
                    let idx = idx_f.round();
                    if !idx.is_finite() || idx < 1.0_f64 {
                        scratch.stack.push(f64::NAN);
                        continue;
                    }
                    let ext = ctx.external.ok_or_else(|| {
                        MathError::new(
                            format!("Cannot index `{}[{}]` without R workspace", target_name, idx),
                            *span,
                        )
                    })?;
                    let val = ext
                        .get_vector_element(target_name, idx as usize)
                        .unwrap_or(f64::NAN);
                    scratch.stack.push(val);
                }
                Op::IndexMember {
                    object,
                    field,
                    span,
                } => {
                    let idx_f = pop_or_err(&mut scratch.stack)?;
                    let idx = idx_f.round();
                    if !idx.is_finite() || idx < 1.0_f64 {
                        scratch.stack.push(f64::NAN);
                        continue;
                    }
                    let ext = ctx.external.ok_or_else(|| {
                        MathError::new(
                            format!("Cannot index `{}${}[{}]` without R workspace", object, field, idx),
                            *span,
                        )
                    })?;
                    let val = ext
                        .get_member_element(object, field, idx as usize)
                        .unwrap_or(f64::NAN);
                    scratch.stack.push(val);
                }
            }
        }

        scratch.stack.pop().ok_or_else(|| {
            MathError::new("Expression produced no value", Span::default())
        })
    }
}

fn pop_or_err(stack: &mut Vec<f64>) -> Result<f64, MathError> {
    stack
        .pop()
        .ok_or_else(|| MathError::new("Internal evaluation stack underflow", Span::default()))
}

fn builtin_constant(name: &str) -> Option<f64> {
    match name {
        "e" => Some(std::f64::consts::E),
        "pi" | "PI" => Some(std::f64::consts::PI),
        "phi" => Some(0.5_f64 * (1.0_f64 + 5.0_f64.sqrt())),
        "infty" => Some(f64::INFINITY),
        _ => None,
    }
}

fn lookup_param_with_alias<'a>(params: &'a IndexMap<String, f64>, name: &str) -> Option<&'a f64> {
    if let Some(v) = params.get(name) {
        return Some(v);
    }
    match name {
        "P_x" => params.get("price_x"),
        "price_x" => params.get("P_x"),
        "P_y" => params.get("price_y"),
        "price_y" => params.get("P_y"),
        _ => None,
    }
}

fn resolve_var_value(
    name: &str,
    ctx: &EvalContext<'_>,
    var_names: &[String],
    scratch: &EvalScratch,
    active_overrides: &[(&str, f64)],
) -> Option<f64> {
    if let Some((_, v)) = active_overrides.iter().rev().find(|(k, _)| *k == name) {
        return Some(*v);
    }
    if let Some(slot) = var_names.iter().position(|n| n == name) {
        if slot < scratch.slot_bound.len() && scratch.slot_bound[slot] {
            return Some(scratch.slots[slot]);
        }
    }
    if let Some(&v) = lookup_param_with_alias(ctx.params, name) {
        return Some(v);
    }
    if let Some(ext) = ctx.external {
        if let Some(v) = ext.get_scalar(name) {
            return Some(v);
        }
    }
    builtin_constant(name)
}

fn build_inherited_overrides(
    var_names: &[String],
    scratch: &EvalScratch,
    active_overrides: &[(&str, f64)],
) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = Vec::with_capacity(var_names.len() + active_overrides.len());
    for (i, name) in var_names.iter().enumerate() {
        if i < scratch.slot_bound.len() && scratch.slot_bound[i] {
            out.push((name.clone(), scratch.slots[i]));
        }
    }
    for &(k, v) in active_overrides {
        if let Some(existing) = out.iter_mut().find(|(ek, _)| ek == k) {
            existing.1 = v;
        } else {
            out.push((k.to_string(), v));
        }
    }
    out
}

fn eval_sub_with_owned_overrides(
    sub: &CompiledExpr,
    ctx: &EvalContext<'_>,
    owned: &[(String, f64)],
    sub_scratch: &mut EvalScratch,
    depth: usize,
) -> Result<f64, MathError> {
    let refs: Vec<(&str, f64)> = owned.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    sub.bind_slots(ctx, &refs, sub_scratch);
    sub.eval_internal(ctx, sub_scratch, &refs, depth)
}

fn eval_numerical_derivative_compiled(
    sub: &CompiledExpr,
    var: &str,
    x0: f64,
    order: u8,
    ctx: &EvalContext<'_>,
    overrides: &mut Vec<(String, f64)>,
    depth: usize,
) -> Result<f64, MathError> {
    let base_h = ctx.diff_h.clamp(1e-9_f64, 1e-2_f64);
    let mut sub_scratch = EvalScratch::new();

    let mut eval_at = |x: f64| -> Result<f64, MathError> {
        if let Some(entry) = overrides.iter_mut().find(|(k, _)| k == var) {
            entry.1 = x;
        } else {
            overrides.push((var.to_string(), x));
        }
        eval_sub_with_owned_overrides(sub, ctx, overrides, &mut sub_scratch, depth)
    };

    if order <= 1 {
        let h = base_h.max(x0.abs() * base_h * 0.1_f64);
        let fp = eval_at(x0 + h)?;
        let fm = eval_at(x0 - h)?;
        Ok((fp - fm) / (2.0_f64 * h))
    } else if order == 2 {
        let h = (base_h * 10.0_f64).clamp(1e-6_f64, 1e-2_f64);
        let fp = eval_at(x0 + h)?;
        let f0 = eval_at(x0)?;
        let fm = eval_at(x0 - h)?;
        Ok((fp - 2.0_f64 * f0 + fm) / (h * h))
    } else {
        // Higher-order via recursive central difference
        let h = (base_h * 20.0_f64).clamp(1e-5_f64, 1e-2_f64);
        let dp = eval_numerical_derivative_compiled(
            sub,
            var,
            x0 + h,
            order - 1,
            ctx,
            overrides,
            depth,
        )?;
        let dm = eval_numerical_derivative_compiled(
            sub,
            var,
            x0 - h,
            order - 1,
            ctx,
            overrides,
            depth,
        )?;
        Ok((dp - dm) / (2.0_f64 * h))
    }
}

fn eval_definite_integral_compiled(
    sub: &CompiledExpr,
    var: &str,
    a: f64,
    b: f64,
    ctx: &EvalContext<'_>,
    overrides: &mut Vec<(String, f64)>,
    depth: usize,
) -> Result<f64, MathError> {
    if (b - a).abs() < 1e-15_f64 {
        return Ok(0.0_f64);
    }
    let steps = if ctx.integral_steps.is_multiple_of(2) {
        ctx.integral_steps.max(4)
    } else {
        ctx.integral_steps + 1
    };
    let h = (b - a) / (steps as f64);
    let mut sub_scratch = EvalScratch::new();

    // Pre-bind once, then mutate slot directly if present for speed
    let initial_refs: Vec<(&str, f64)> = overrides.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    sub.bind_slots(ctx, &initial_refs, &mut sub_scratch);
    let var_slot = sub.slot_of(var);
    let mut eval_at = |x: f64| -> Result<f64, MathError> {
        if let Some(slot) = var_slot {
            sub.set_slot(slot, x, &mut sub_scratch);
        }
        if let Some(entry) = overrides.iter_mut().find(|(k, _)| k == var) {
            entry.1 = x;
        } else {
            overrides.push((var.to_string(), x));
        }
        let refs: Vec<(&str, f64)> = overrides.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        sub.eval_internal(ctx, &mut sub_scratch, &refs, depth)
    };

    let mut sum = eval_at(a)? + eval_at(b)?;
    for i in 1..steps {
        let x = a + (i as f64) * h;
        let fx = eval_at(x)?;
        if !fx.is_finite() {
            return Ok(f64::NAN);
        }
        if i % 2 == 1 {
            sum += 4.0_f64 * fx;
        } else {
            sum += 2.0_f64 * fx;
        }
    }
    Ok(sum * h / 3.0_f64)
}

#[allow(clippy::too_many_arguments)]
fn eval_aggregation_compiled(
    sub: &CompiledExpr,
    var: &str,
    lo_f: f64,
    hi_f: f64,
    is_sum: bool,
    ctx: &EvalContext<'_>,
    overrides: &mut Vec<(String, f64)>,
    depth: usize,
) -> Result<f64, MathError> {
    if !lo_f.is_finite() || !hi_f.is_finite() {
        return Ok(f64::NAN);
    }
    let lo = lo_f.round() as i64;
    let hi = hi_f.round() as i64;
    let mut acc = if is_sum { 0.0_f64 } else { 1.0_f64 };
    if hi < lo {
        return Ok(acc);
    }
    let count = (hi - lo + 1).min(100_000);
    let mut sub_scratch = EvalScratch::new();
    let initial_refs: Vec<(&str, f64)> = overrides.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    sub.bind_slots(ctx, &initial_refs, &mut sub_scratch);
    let var_slot = sub.slot_of(var);

    for offset in 0..count {
        let k_val = (lo + offset) as f64;
        if let Some(slot) = var_slot {
            sub.set_slot(slot, k_val, &mut sub_scratch);
        }
        if let Some(entry) = overrides.iter_mut().find(|(k, _)| k == var) {
            entry.1 = k_val;
        } else {
            overrides.push((var.to_string(), k_val));
        }
        let refs: Vec<(&str, f64)> = overrides.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let term = sub.eval_internal(ctx, &mut sub_scratch, &refs, depth)?;
        if is_sum {
            acc += term;
        } else {
            acc *= term;
        }
    }
    Ok(acc)
}

#[allow(clippy::too_many_arguments)]
fn eval_function_call(
    name: &str,
    prime_order: u8,
    args: &[f64],
    span: Span,
    ctx: &EvalContext<'_>,
    scratch: &EvalScratch,
    var_names: &[String],
    active_overrides: &[(&str, f64)],
    depth: usize,
) -> Result<f64, MathError> {
    // 1. User-defined workspace function (supports N-ary and `f'(x)` / `f''(x)`)
    if let Some(user_fn) = ctx.functions.get(name) {
        if prime_order > 0 {
            if args.len() != 1 || user_fn.params.is_empty() {
                return Err(MathError::new(
                    format!("Derivative `{}'` requires 1 argument", name),
                    span,
                ));
            }
            let param_name = &user_fn.params[0];
            let mut overrides: Vec<(String, f64)> = vec![(param_name.clone(), args[0])];
            return eval_numerical_derivative_compiled(
                &user_fn.body,
                param_name,
                args[0],
                prime_order,
                ctx,
                &mut overrides,
                depth + 1,
            );
        }

        if args.len() != user_fn.params.len() {
            return Err(MathError::new(
                format!(
                    "Function `{}` expects {} argument(s), got {}",
                    name,
                    user_fn.params.len(),
                    args.len()
                ),
                span,
            ));
        }

        let mut fn_overrides: Vec<(&str, f64)> = Vec::with_capacity(args.len() + 2);
        for (p, &v) in user_fn.params.iter().zip(args.iter()) {
            fn_overrides.push((p.as_str(), v));
        }
        let mut sub_scratch = EvalScratch::new();
        user_fn.body.bind_slots(ctx, &fn_overrides, &mut sub_scratch);
        return user_fn
            .body
            .eval_internal(ctx, &mut sub_scratch, &fn_overrides, depth + 1);
    }

    // 2. Built-in functions (including `sin'(x)` numerical derivative if primed!)
    if prime_order > 0 {
        if args.len() == 1 {
            let h = ctx.diff_h.clamp(1e-8_f64, 1e-3_f64);
            let x0 = args[0];
            if let (Some(fp), Some(fm)) = (
                eval_builtin(name, &[x0 + h]),
                eval_builtin(name, &[x0 - h]),
            ) {
                if prime_order == 1 {
                    return Ok((fp - fm) / (2.0_f64 * h));
                } else if let Some(f0) = eval_builtin(name, &[x0]) {
                    let h2 = (h * 10.0_f64).clamp(1e-6_f64, 1e-2_f64);
                    let fp2 = eval_builtin(name, &[x0 + h2]).unwrap_or(f64::NAN);
                    let fm2 = eval_builtin(name, &[x0 - h2]).unwrap_or(f64::NAN);
                    return Ok((fp2 - 2.0_f64 * f0 + fm2) / (h2 * h2));
                }
            }
        }
    } else if let Some(val) = eval_builtin(name, args) {
        return Ok(val);
    }

    // 2b. Arbitrary-base subscripted log, e.g. `log_2(8)` = 3
    if prime_order == 0 && args.len() == 1 {
        if let Some(base_str) = name.strip_prefix("log_") {
            if let Ok(base) = base_str.parse::<f64>() {
                return Ok(args[0].ln() / base.ln());
            }
        }
    }

    // 3. External R function bridge (e.g., `predict(...)`, `mean(...)`, user R closures)
    if prime_order == 0 {
        if let Some(ext) = ctx.external {
            if let Some(res) = ext.call_function(name, args) {
                return res.map_err(|msg| MathError::new(msg, span));
            }
        }
    }

    // 4. Desmos-style implicit multiplication fallback: `a(x + 1)` where `a` is a bound parameter
    if prime_order == 0 && args.len() == 1 {
        if let Some(var_val) =
            resolve_var_value(name, ctx, var_names, scratch, active_overrides)
        {
            return Ok(var_val * args[0]);
        }
    }

    Err(MathError::new(
        format!("Unknown function `{}`", name),
        span,
    ))
}

pub fn eval_fn_val_and_deriv(
    fn_name: &str,
    x0: f64,
    span: Span,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<(f64, f64), MathError> {
    let h = ctx.diff_h.clamp(1e-8_f64, 1e-3_f64);
    let step = h.max(x0.abs() * h * 0.1_f64);
    if let Some(user_fn) = ctx.functions.get(fn_name) {
        if user_fn.params.is_empty() {
            return Err(MathError::new(
                format!("Function `{}` takes no parameters", fn_name),
                span,
            ));
        }
        let param = user_fn.params[0].clone();
        let mut s = EvalScratch::new();
        let y0 = user_fn.body.eval(ctx, &[(param.as_str(), x0)], &mut s)?;
        let mut ov = vec![(param.clone(), x0)];
        let dy = eval_numerical_derivative_compiled(
            &user_fn.body,
            &param,
            x0,
            1,
            ctx,
            &mut ov,
            depth,
        )?;
        return Ok((y0, dy));
    }
    if let (Some(y0), Some(fp), Some(fm)) = (
        eval_builtin(fn_name, &[x0]),
        eval_builtin(fn_name, &[x0 + step]),
        eval_builtin(fn_name, &[x0 - step]),
    ) {
        return Ok((y0, (fp - fm) / (2.0_f64 * step)));
    }
    Err(MathError::new(
        format!("Unknown function `{}` in calculus/economics operator", fn_name),
        span,
    ))
}

fn eval_single_fn_at(
    fn_name: &str,
    x: f64,
    span: Span,
    ctx: &EvalContext<'_>,
    scratch: &mut EvalScratch,
) -> Result<f64, MathError> {
    if let Some(user_fn) = ctx.functions.get(fn_name) {
        if let Some(param) = user_fn.params.first() {
            return user_fn.body.eval(ctx, &[(param.as_str(), x)], scratch);
        }
    }
    if let Some(v) = eval_builtin(fn_name, &[x]) {
        return Ok(v);
    }
    Err(MathError::new(format!("Unknown function `{}`", fn_name), span))
}

/// Finds the first positive intersection `f(x*) = g(x*)` for `surplus(f, g)`.
pub fn find_surplus_equilibrium_x(
    f_name: &str,
    g_name: &str,
    ctx: &EvalContext<'_>,
    span: Span,
    _depth: usize,
) -> Result<f64, MathError> {
    let mut s1 = EvalScratch::new();
    let mut s2 = EvalScratch::new();
    // Verify both functions exist
    let _ = eval_single_fn_at(f_name, 1.0_f64, span, ctx, &mut s1)?;
    let _ = eval_single_fn_at(g_name, 1.0_f64, span, ctx, &mut s2)?;

    let roots = super::solver::find_real_roots_1d(
        |x| {
            let yf = eval_single_fn_at(f_name, x, span, ctx, &mut s1).ok()?;
            let yg = eval_single_fn_at(g_name, x, span, ctx, &mut s2).ok()?;
            Some(yf - yg)
        },
        1e-4_f64,
        200.0_f64,
        400,
    );
    roots
        .into_iter()
        .find(|&r| r > 1e-4_f64)
        .ok_or_else(|| {
            MathError::new(
                format!(
                    "No positive equilibrium intersection found between `{}` and `{}`",
                    f_name, g_name
                ),
                span,
            )
        })
}

/// Computes `∫_a^b (f(x) - g(x)) dx` via composite Simpson's rule.
pub fn eval_area_between_fns(
    f_name: &str,
    g_name: &str,
    a: f64,
    b: f64,
    span: Span,
    ctx: &EvalContext<'_>,
    _depth: usize,
) -> Result<f64, MathError> {
    if !a.is_finite() || !b.is_finite() {
        return Ok(f64::NAN);
    }
    if (b - a).abs() < 1e-12_f64 {
        return Ok(0.0_f64);
    }
    let steps = if ctx.integral_steps.is_multiple_of(2) {
        ctx.integral_steps.max(4)
    } else {
        ctx.integral_steps + 1
    };
    let h = (b - a) / (steps as f64);
    let mut s1 = EvalScratch::new();
    let mut s2 = EvalScratch::new();
    let mut eval_diff = |x: f64| -> Result<f64, MathError> {
        let yf = eval_single_fn_at(f_name, x, span, ctx, &mut s1)?;
        let yg = eval_single_fn_at(g_name, x, span, ctx, &mut s2)?;
        Ok(yf - yg)
    };
    let mut sum = eval_diff(a)? + eval_diff(b)?;
    for i in 1..steps {
        let x = a + (i as f64) * h;
        let d = eval_diff(x)?;
        if !d.is_finite() {
            return Ok(f64::NAN);
        }
        sum += if i % 2 == 1 { 4.0_f64 * d } else { 2.0_f64 * d };
    }
    Ok(sum * h / 3.0_f64)
}

/// Extracts a function identifier name from either `f` or `f(x)` when passed into
/// meta-operators like `tangent(f, x0)`, `elasticity(f, x)`, `area(f, g, a, b)`, `surplus(f, g)`.
pub fn extract_fn_arg_name(expr: &SpannedExpr) -> Option<String> {
    if let Some(id) = expr.as_canonical_ident() {
        return Some(id);
    }
    if let MathExpr::Call {
        name,
        prime_order: 0,
        ..
    } = &expr.expr
    {
        return Some(name.clone());
    }
    None
}

fn eval_builtin(name: &str, args: &[f64]) -> Option<f64> {
    match (name, args) {
        ("abs", &[x]) => Some(x.abs()),
        ("sqrt", &[x]) => Some(x.sqrt()),
        ("cbrt", &[x]) => Some(x.cbrt()),
        ("sin", &[x]) => Some(x.sin()),
        ("cos", &[x]) => Some(x.cos()),
        ("tan", &[x]) => Some(x.tan()),
        ("sec", &[x]) => Some(1.0_f64 / x.cos()),
        ("csc", &[x]) => Some(1.0_f64 / x.sin()),
        ("cot", &[x]) => Some(x.cos() / x.sin()),
        ("asin" | "arcsin", &[x]) => Some(x.asin()),
        ("acos" | "arccos", &[x]) => Some(x.acos()),
        ("atan" | "arctan", &[x]) => Some(x.atan()),
        ("atan2", &[y, x]) => Some(y.atan2(x)),
        ("sinh", &[x]) => Some(x.sinh()),
        ("cosh", &[x]) => Some(x.cosh()),
        ("tanh", &[x]) => Some(x.tanh()),
        ("asinh", &[x]) => Some(x.asinh()),
        ("acosh", &[x]) => Some(x.acosh()),
        ("atanh", &[x]) => Some(x.atanh()),
        ("exp", &[x]) => Some(x.exp()),
        ("ln", &[x]) => Some(x.ln()),
        ("log", &[x]) => Some(x.log10()),
        ("log", &[x, base]) => Some(x.ln() / base.ln()),
        ("log10", &[x]) => Some(x.log10()),
        ("log2", &[x]) => Some(x.log2()),
        ("floor", &[x]) => Some(x.floor()),
        ("ceil", &[x]) => Some(x.ceil()),
        ("round", &[x]) => Some(x.round()),
        ("sign" | "signum", &[x]) => Some(if x > 0.0_f64 {
            1.0_f64
        } else if x < 0.0_f64 {
            -1.0_f64
        } else {
            0.0_f64
        }),
        ("min", xs) if !xs.is_empty() => Some(xs.iter().copied().fold(f64::INFINITY, f64::min)),
        ("max", xs) if !xs.is_empty() => Some(xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        _ => None,
    }
}

struct CompilerBuilder {
    ops: Vec<Op>,
    var_names: Vec<String>,
    var_map: HashMap<String, usize>,
    sub_exprs: Vec<CompiledExpr>,
}

impl CompilerBuilder {
    fn intern_var(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.var_map.get(name) {
            return idx;
        }
        let idx = self.var_names.len();
        self.var_names.push(name.to_string());
        self.var_map.insert(name.to_string(), idx);
        idx
    }

    fn add_sub(&mut self, expr: &SpannedExpr) -> usize {
        let compiled = CompiledExpr::compile(expr);
        // Ensure sub-expression variables are also tracked in parent slots if they might be overridden
        for vn in &compiled.var_names {
            self.intern_var(vn);
        }
        let idx = self.sub_exprs.len();
        self.sub_exprs.push(compiled);
        idx
    }

    fn emit(&mut self, node: &SpannedExpr) {
        match &node.expr {
            MathExpr::Number(v) => self.ops.push(Op::PushConst(*v)),
            MathExpr::Ident(name) => {
                let slot = self.intern_var(name);
                self.ops.push(Op::LoadVar(slot));
            }
            MathExpr::Subscript { .. } | MathExpr::Accent { .. } => {
                if let Some(canon) = node.as_canonical_ident() {
                    let slot = self.intern_var(&canon);
                    self.ops.push(Op::LoadVar(slot));
                } else if let MathExpr::Accent {
                    kind: AccentKind::Bar | AccentKind::Hat | AccentKind::Tilde | AccentKind::Vec,
                    inner,
                } = &node.expr
                {
                    self.emit(inner);
                } else if let MathExpr::Subscript { base, .. } = &node.expr {
                    self.emit(base);
                }
            }
            MathExpr::Group(inner) => self.emit(inner),
            MathExpr::Abs(inner) => {
                self.emit(inner);
                self.ops.push(Op::Abs);
            }
            MathExpr::Unary { op, expr } => {
                self.emit(expr);
                if matches!(op, UnaryOp::Neg) {
                    self.ops.push(Op::Neg);
                }
            }
            MathExpr::Binary { op, lhs, rhs } => {
                self.emit(lhs);
                self.emit(rhs);
                let byte_op = match op {
                    BinOp::Add => Op::Add,
                    BinOp::Sub => Op::Sub,
                    BinOp::Mul | BinOp::ImplicitMul => Op::Mul,
                    BinOp::Pow => Op::Pow,
                    BinOp::Mod => Op::Mod,
                };
                self.ops.push(byte_op);
            }
            MathExpr::Fraction { num, den } => {
                self.emit(num);
                self.emit(den);
                self.ops.push(Op::Div);
            }
            MathExpr::Sqrt { index, radicand } => {
                if let Some(idx_expr) = index {
                    self.emit(idx_expr);
                    self.emit(radicand);
                    self.ops.push(Op::NthRoot);
                } else {
                    self.emit(radicand);
                    self.ops.push(Op::Sqrt);
                }
            }
            MathExpr::Call {
                name,
                prime_order,
                args,
            } => {
                // Functional forms: `sum(body, i, a, b)`, `prod(body, i, a, b)`, `int(body, x, a, b)`, `diff(body, x)`
                if *prime_order == 0 {
                    if matches!(name.as_str(), "sum" | "prod" | "int") && args.len() == 4 {
                        if let Some(var) = args[1].as_canonical_ident() {
                            let body_idx = self.add_sub(&args[0]);
                            let lower_idx = self.add_sub(&args[2]);
                            let upper_idx = self.add_sub(&args[3]);
                            let op = match name.as_str() {
                                "sum" => Op::Sum {
                                    var,
                                    lower_idx,
                                    upper_idx,
                                    body_idx,
                                },
                                "prod" => Op::Prod {
                                    var,
                                    lower_idx,
                                    upper_idx,
                                    body_idx,
                                },
                                _ => Op::Integral {
                                    var,
                                    lower_idx,
                                    upper_idx,
                                    body_idx,
                                    span: node.span,
                                },
                            };
                            self.ops.push(op);
                            return;
                        }
                    }
                    if name == "diff" && args.len() == 2 {
                        if let Some(var) = args[1].as_canonical_ident() {
                            let body_idx = self.add_sub(&args[0]);
                            self.ops.push(Op::Derivative {
                                var,
                                order: 1,
                                body_idx,
                                span: node.span,
                            });
                            return;
                        }
                    }
                    // Native computational economics & calculus meta-operators:
                    // `tangent(f, x0)`, `elasticity(f, x)` / `epsilon(f, x)`,
                    // `marginal(f, x)`, `average(f, x)`, `area(f, g, a, b)`, `surplus(f, g)`
                    if name == "tangent" && args.len() == 2 {
                        if let Some(fn_name) = extract_fn_arg_name(&args[0]) {
                            self.intern_var("x");
                            let x0_idx = self.add_sub(&args[1]);
                            self.ops.push(Op::Tangent {
                                fn_name,
                                x0_idx,
                                span: node.span,
                            });
                            return;
                        }
                    }
                    if matches!(name.as_str(), "elasticity" | "epsilon") && args.len() == 2 {
                        if let Some(fn_name) = extract_fn_arg_name(&args[0]) {
                            let x_idx = self.add_sub(&args[1]);
                            self.ops.push(Op::Elasticity {
                                fn_name,
                                x_idx,
                                span: node.span,
                            });
                            return;
                        }
                    }
                    if name == "marginal" && args.len() == 2 {
                        if let Some(fn_name) = extract_fn_arg_name(&args[0]) {
                            let x_idx = self.add_sub(&args[1]);
                            self.ops.push(Op::Marginal {
                                fn_name,
                                x_idx,
                                span: node.span,
                            });
                            return;
                        }
                    }
                    if name == "average" && args.len() == 2 {
                        if let Some(fn_name) = extract_fn_arg_name(&args[0]) {
                            let x_idx = self.add_sub(&args[1]);
                            self.ops.push(Op::Average {
                                fn_name,
                                x_idx,
                                span: node.span,
                            });
                            return;
                        }
                    }
                    if matches!(name.as_str(), "area" | "surplus") && (args.len() == 2 || args.len() == 4) {
                        if let (Some(f_name), Some(g_name)) =
                            (extract_fn_arg_name(&args[0]), extract_fn_arg_name(&args[1]))
                        {
                            let (a_idx, b_idx) = if args.len() == 4 {
                                (Some(self.add_sub(&args[2])), Some(self.add_sub(&args[3])))
                            } else {
                                (None, None)
                            };
                            self.ops.push(Op::AreaBetween {
                                f_name,
                                g_name,
                                a_idx,
                                b_idx,
                                span: node.span,
                            });
                            return;
                        }
                    }
                }

                for a in args {
                    self.emit(a);
                }
                self.ops.push(Op::Call {
                    name: name.clone(),
                    prime_order: *prime_order,
                    argc: args.len(),
                    span: node.span,
                });
            }
            MathExpr::Derivative { var, order, body } => {
                self.intern_var(var);
                let body_idx = self.add_sub(body);
                self.ops.push(Op::Derivative {
                    var: var.clone(),
                    order: *order,
                    body_idx,
                    span: node.span,
                });
            }
            MathExpr::Integral {
                var,
                lower,
                upper,
                body,
            } => {
                let lower_idx = self.add_sub(lower);
                let upper_idx = self.add_sub(upper);
                let body_idx = self.add_sub(body);
                self.ops.push(Op::Integral {
                    var: var.clone(),
                    lower_idx,
                    upper_idx,
                    body_idx,
                    span: node.span,
                });
            }
            MathExpr::Sum {
                var,
                lower,
                upper,
                body,
            } => {
                let lower_idx = self.add_sub(lower);
                let upper_idx = self.add_sub(upper);
                let body_idx = self.add_sub(body);
                self.ops.push(Op::Sum {
                    var: var.clone(),
                    lower_idx,
                    upper_idx,
                    body_idx,
                });
            }
            MathExpr::Prod {
                var,
                lower,
                upper,
                body,
            } => {
                let lower_idx = self.add_sub(lower);
                let upper_idx = self.add_sub(upper);
                let body_idx = self.add_sub(body);
                self.ops.push(Op::Prod {
                    var: var.clone(),
                    lower_idx,
                    upper_idx,
                    body_idx,
                });
            }
            MathExpr::ComparisonChain { first, rest } => {
                self.emit(first);
                let mut ops = Vec::with_capacity(rest.len());
                for (cmp, rhs) in rest {
                    self.emit(rhs);
                    ops.push(*cmp);
                }
                self.ops.push(Op::CmpChain { ops });
            }
            MathExpr::Piecewise(branches) => {
                let mut compiled_branches = Vec::with_capacity(branches.len());
                for b in branches {
                    let cond_idx = b.condition.as_ref().map(|c| self.add_sub(c));
                    let val_idx = self.add_sub(&b.value);
                    compiled_branches.push((cond_idx, val_idx));
                }
                self.ops.push(Op::Piecewise {
                    branches: compiled_branches,
                });
            }
            MathExpr::Restricted {
                inner,
                restrictions,
            } => {
                let inner_idx = self.add_sub(inner);
                let cond_indices: Vec<usize> =
                    restrictions.iter().map(|c| self.add_sub(c)).collect();
                self.ops.push(Op::Restricted {
                    inner_idx,
                    cond_indices,
                });
            }
            MathExpr::Placeholder => {
                self.ops.push(Op::PushConst(f64::NAN));
            }
            MathExpr::Tuple(items) => {
                if let Some(first) = items.first() {
                    self.emit(first);
                } else {
                    self.ops.push(Op::PushConst(f64::NAN));
                }
            }
            MathExpr::MemberAccess { object, field } => {
                self.ops.push(Op::MemberAccess {
                    object: object.clone(),
                    field: field.clone(),
                    span: node.span,
                });
            }
            MathExpr::Index { target, index } => {
                self.emit(index);
                match &target.expr {
                    MathExpr::MemberAccess { object, field } => {
                        self.ops.push(Op::IndexMember {
                            object: object.clone(),
                            field: field.clone(),
                            span: node.span,
                        });
                    }
                    _ => {
                        let name = target
                            .as_canonical_ident()
                            .unwrap_or_else(|| "?".to_string());
                        self.ops.push(Op::IndexVar {
                            target_name: name,
                            span: node.span,
                        });
                    }
                }
            }
        }
    }
}
