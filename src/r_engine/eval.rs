use super::ast::{RBinOp, RCallArg, RError, RExpr, RParam, RUnaryOp};
use super::object::RObject;
use super::stdlib::{EvalArg, call_builtin};
use crate::collections::IndexMap;
use nalgebra::DMatrix;

/// Bridge allowing the embedded R engine to invoke active Desmos workspace functions & parameters.
pub trait DesmosBridge {
    fn has_desmos_fn(&self, name: &str) -> bool;
    fn call_desmos_fn(&self, name: &str, args: &[f64]) -> Option<Result<f64, String>>;
    fn get_desmos_param(&self, name: &str) -> Option<f64>;
}

const MAX_LOOP_ITERS: usize = 100_000;
const MAX_CALL_DEPTH: usize = 64;

enum Flow {
    Val(RObject),
    Return(RObject),
    Break,
    Next,
}

pub struct REvaluator<'a> {
    pub global_env: &'a mut IndexMap<String, RObject>,
    frames: Vec<IndexMap<String, RObject>>,
    bridge: Option<&'a dyn DesmosBridge>,
    pub stdout: Vec<String>,
    call_depth: usize,
}

impl<'a> REvaluator<'a> {
    pub fn new(
        global_env: &'a mut IndexMap<String, RObject>,
        bridge: Option<&'a dyn DesmosBridge>,
    ) -> Self {
        Self {
            global_env,
            frames: Vec::new(),
            bridge,
            stdout: Vec::new(),
            call_depth: 0,
        }
    }

    pub fn eval_program(&mut self, stmts: &[RExpr]) -> Result<RObject, RError> {
        let mut last = RObject::Null;
        for stmt in stmts {
            match self.eval_node(stmt)? {
                Flow::Val(v) | Flow::Return(v) => last = v,
                Flow::Break | Flow::Next => {
                    return Err(RError::new("`break`/`next` used outside of a loop"));
                }
            }
        }
        Ok(last)
    }

    fn lookup(&self, name: &str) -> Option<RObject> {
        for frame in self.frames.iter().rev() {
            if let Some(v) = frame.get(name) {
                return Some(v.clone());
            }
        }
        if let Some(v) = self.global_env.get(name) {
            return Some(v.clone());
        }
        // Built-in R constants
        if name == "pi" {
            return Some(RObject::scalar(std::f64::consts::PI));
        }
        if name == "e" {
            return Some(RObject::scalar(std::f64::consts::E));
        }
        // Bridge to active Desmos workspace parameters
        if let Some(b) = self.bridge {
            if let Some(val) = b.get_desmos_param(name) {
                return Some(RObject::scalar(val));
            }
        }
        None
    }

    fn assign_var(&mut self, name: String, value: RObject) {
        if let Some(top) = self.frames.last_mut() {
            top.insert(name, value);
        } else {
            self.global_env.insert(name, value);
        }
    }

    fn snapshot_env(&self) -> IndexMap<String, RObject> {
        let mut merged = self.global_env.clone();
        for frame in &self.frames {
            for (k, v) in frame {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }

    fn eval_value(&mut self, expr: &RExpr) -> Result<RObject, RError> {
        match self.eval_node(expr)? {
            Flow::Val(v) | Flow::Return(v) => Ok(v),
            Flow::Break => Err(RError::new("Unexpected `break` in expression")),
            Flow::Next => Err(RError::new("Unexpected `next` in expression")),
        }
    }

    fn eval_node(&mut self, expr: &RExpr) -> Result<Flow, RError> {
        match expr {
            RExpr::Null => Ok(Flow::Val(RObject::Null)),
            RExpr::Na => Ok(Flow::Val(RObject::Numeric(vec![None]))),
            RExpr::Bool(b) => Ok(Flow::Val(RObject::bool_scalar(*b))),
            RExpr::Number(n) => Ok(Flow::Val(RObject::scalar(*n))),
            RExpr::String(s) => Ok(Flow::Val(RObject::char_scalar(s.clone()))),
            RExpr::Ident(name) => self
                .lookup(name)
                .map(Flow::Val)
                .ok_or_else(|| RError::new(format!("object '{}' not found", name))),
            RExpr::Formula { lhs, rhs } => Ok(Flow::Val(RObject::Formula {
                lhs: lhs.as_ref().map(|e| e.to_source_string()),
                rhs: rhs.to_source_string(),
            })),
            RExpr::Unary { op, expr } => {
                let val = self.eval_value(expr)?;
                Ok(Flow::Val(eval_unary(*op, &val)?))
            }
            RExpr::Binary { op, lhs, rhs } => {
                // Short-circuit `&&` and `||`
                if *op == RBinOp::ShortAnd {
                    let lv = self.eval_value(lhs)?;
                    let lb = lv.is_truthy().map_err(RError::new)?;
                    if !lb {
                        return Ok(Flow::Val(RObject::bool_scalar(false)));
                    }
                    let rv = self.eval_value(rhs)?;
                    return Ok(Flow::Val(RObject::bool_scalar(
                        rv.is_truthy().map_err(RError::new)?,
                    )));
                }
                if *op == RBinOp::ShortOr {
                    let lv = self.eval_value(lhs)?;
                    let lb = lv.is_truthy().map_err(RError::new)?;
                    if lb {
                        return Ok(Flow::Val(RObject::bool_scalar(true)));
                    }
                    let rv = self.eval_value(rhs)?;
                    return Ok(Flow::Val(RObject::bool_scalar(
                        rv.is_truthy().map_err(RError::new)?,
                    )));
                }
                let lv = self.eval_value(lhs)?;
                let rv = self.eval_value(rhs)?;
                Ok(Flow::Val(eval_binary(*op, &lv, &rv)?))
            }
            RExpr::Assign { target, value } => {
                let val = self.eval_value(value)?;
                self.eval_assignment(target, val.clone())?;
                Ok(Flow::Val(val))
            }
            RExpr::Block(stmts) => {
                let mut last = RObject::Null;
                for s in stmts {
                    match self.eval_node(s)? {
                        Flow::Val(v) => last = v,
                        signal => return Ok(signal),
                    }
                }
                Ok(Flow::Val(last))
            }
            RExpr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cv = self.eval_value(cond)?;
                if cv.is_truthy().map_err(RError::new)? {
                    self.eval_node(then_branch)
                } else if let Some(eb) = else_branch {
                    self.eval_node(eb)
                } else {
                    Ok(Flow::Val(RObject::Null))
                }
            }
            RExpr::For { var, iter, body } => {
                let seq_obj = self.eval_value(iter)?;
                let items = iter_elements(&seq_obj);
                for (count, item) in items.into_iter().enumerate() {
                    if count >= MAX_LOOP_ITERS {
                        return Err(RError::new("`for` loop exceeded iteration safety limit"));
                    }
                    self.assign_var(var.clone(), item);
                    match self.eval_node(body)? {
                        Flow::Val(_) | Flow::Next => {}
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                    }
                }
                Ok(Flow::Val(RObject::Null))
            }
            RExpr::While { cond, body } => {
                let mut count = 0usize;
                loop {
                    count += 1;
                    if count > MAX_LOOP_ITERS {
                        return Err(RError::new("`while` loop exceeded iteration safety limit"));
                    }
                    let cv = self.eval_value(cond)?;
                    if !cv.is_truthy().map_err(RError::new)? {
                        break;
                    }
                    match self.eval_node(body)? {
                        Flow::Val(_) | Flow::Next => {}
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                    }
                }
                Ok(Flow::Val(RObject::Null))
            }
            RExpr::FunctionDef { params, body } => {
                let env = self.snapshot_env();
                Ok(Flow::Val(RObject::Closure {
                    params: params.clone(),
                    body: body.clone(),
                    env,
                }))
            }
            RExpr::Return(opt) => {
                let val = match opt {
                    Some(e) => self.eval_value(e)?,
                    None => RObject::Null,
                };
                Ok(Flow::Return(val))
            }
            RExpr::Break => Ok(Flow::Break),
            RExpr::Next => Ok(Flow::Next),
            RExpr::Member { target, field } => {
                let obj = self.eval_value(target)?;
                Ok(Flow::Val(eval_member(&obj, field)?))
            }
            RExpr::DoubleIndex { target, index } => {
                let obj = self.eval_value(target)?;
                let idx = self.eval_value(index)?;
                Ok(Flow::Val(eval_double_index(&obj, &idx)?))
            }
            RExpr::Index { target, indices } => {
                let obj = self.eval_value(target)?;
                let mut eval_idxs = Vec::with_capacity(indices.len());
                for opt in indices {
                    match opt {
                        Some(e) => eval_idxs.push(Some(self.eval_value(e)?)),
                        None => eval_idxs.push(None),
                    }
                }
                Ok(Flow::Val(eval_index(&obj, &eval_idxs)?))
            }
            RExpr::Call { callee, args } => {
                let val = self.eval_call(callee, args)?;
                Ok(Flow::Val(val))
            }
        }
    }

    fn eval_assignment(&mut self, target: &RExpr, value: RObject) -> Result<(), RError> {
        match target {
            RExpr::Ident(name) => {
                self.assign_var(name.clone(), value);
                Ok(())
            }
            RExpr::Member { target: inner, field } => {
                let RExpr::Ident(var_name) = inner.as_ref() else {
                    return Err(RError::new("Member assignment requires a variable name"));
                };
                let mut obj = self
                    .lookup(var_name)
                    .ok_or_else(|| RError::new(format!("object '{}' not found", var_name)))?;
                match &mut obj {
                    RObject::List(map) => {
                        map.insert(field.clone(), value);
                    }
                    RObject::DataFrame { columns, row_count } => {
                        let col_len = value.r_len();
                        let col_val = if col_len == 1 && *row_count > 1 {
                            match value {
                                RObject::Numeric(v) => RObject::Numeric(vec![v[0]; *row_count]),
                                RObject::Logical(v) => RObject::Logical(vec![v[0]; *row_count]),
                                RObject::Character(v) => {
                                    RObject::Character(vec![v[0].clone(); *row_count])
                                }
                                other => other,
                            }
                        } else {
                            if *row_count == 0 {
                                *row_count = col_len;
                            }
                            value
                        };
                        columns.insert(field.clone(), col_val);
                    }
                    _ => {
                        return Err(RError::new("`$<-` is only supported on lists and data.frames"));
                    }
                }
                self.assign_var(var_name.clone(), obj);
                Ok(())
            }
            RExpr::Index {
                target: inner,
                indices,
            } => {
                let RExpr::Ident(var_name) = inner.as_ref() else {
                    return Err(RError::new("Indexed assignment requires a variable name"));
                };
                let mut obj = self
                    .lookup(var_name)
                    .ok_or_else(|| RError::new(format!("object '{}' not found", var_name)))?;
                let mut eval_idxs = Vec::with_capacity(indices.len());
                for opt in indices {
                    match opt {
                        Some(e) => eval_idxs.push(Some(self.eval_value(e)?)),
                        None => eval_idxs.push(None),
                    }
                }
                assign_index(&mut obj, &eval_idxs, &value)?;
                self.assign_var(var_name.clone(), obj);
                Ok(())
            }
            _ => Err(RError::new("Invalid assignment target")),
        }
    }

    fn eval_call(&mut self, callee: &RExpr, raw_args: &[RCallArg]) -> Result<RObject, RError> {
        // Special form: `subset(df, condition)` evaluates `condition` with `df` columns in scope
        if let RExpr::Ident(fn_name) = callee {
            if fn_name == "subset" && raw_args.len() >= 2 {
                return self.eval_subset_special(raw_args);
            }
            if fn_name == "ls" {
                let names: Vec<Option<String>> = self
                    .global_env
                    .keys()
                    .map(|k| Some(k.clone()))
                    .collect();
                return Ok(RObject::Character(names));
            }
            if fn_name == "rm" {
                for a in raw_args {
                    if let Some(RExpr::Ident(n)) = &a.value {
                        self.global_env.remove(n);
                    }
                }
                return Ok(RObject::Null);
            }
        }

        // Evaluate arguments
        let mut args = Vec::with_capacity(raw_args.len());
        for a in raw_args {
            let val = match &a.value {
                Some(expr) => self.eval_value(expr)?,
                None => RObject::Null,
            };
            args.push(EvalArg {
                name: a.name.clone(),
                value: val,
            });
        }

        // 1. If callee is an identifier, check user closure in env, then stdlib, then Desmos bridge
        if let RExpr::Ident(name) = callee {
            if let Some(RObject::Closure { params, body, env }) = self.lookup(name) {
                return self.invoke_closure(&params, &body, &env, &args);
            }

            let env_snap = self.snapshot_env();
            if let Some(res) = call_builtin(name, &args, &env_snap, &mut self.stdout)? {
                return Ok(res);
            }

            if let Some(bridge) = self.bridge {
                if bridge.has_desmos_fn(name) {
                    return eval_desmos_bridge_call(name, &args, bridge);
                }
            }

            return Err(RError::new(format!("could not find function \"{}\"", name)));
        }

        // 2. General expression evaluating to a closure (e.g. `(function(x) x + 1)(5)`)
        let callee_val = self.eval_value(callee)?;
        if let RObject::Closure { params, body, env } = callee_val {
            return self.invoke_closure(&params, &body, &env, &args);
        }
        Err(RError::new("attempt to apply non-function"))
    }

    fn eval_subset_special(&mut self, raw_args: &[RCallArg]) -> Result<RObject, RError> {
        let df_expr = raw_args[0]
            .value
            .as_ref()
            .ok_or_else(|| RError::new("`subset()` requires a data.frame"))?;
        let cond_expr = raw_args[1]
            .value
            .as_ref()
            .ok_or_else(|| RError::new("`subset()` requires a condition"))?;
        let df_obj = self.eval_value(df_expr)?;
        let RObject::DataFrame { columns, row_count } = &df_obj else {
            return Err(RError::new("`subset()` first argument must be a data.frame"));
        };

        // Push data.frame columns into a temporary lexical frame
        let mut frame = IndexMap::new();
        for (k, v) in columns {
            frame.insert(k.clone(), v.clone());
        }
        self.frames.push(frame);
        let mask_res = self.eval_value(cond_expr);
        self.frames.pop();
        let mask_obj = mask_res?;
        let mask = mask_obj.as_logical_options();

        let mut keep_rows = Vec::new();
        for r in 0..*row_count {
            if !mask.is_empty() && mask[r % mask.len()] == Some(true) {
                keep_rows.push(r);
            }
        }
        let mut out_cols = IndexMap::new();
        for (k, col) in columns {
            out_cols.insert(k.clone(), slice_rows_by_indices(col, &keep_rows));
        }
        Ok(RObject::DataFrame {
            columns: out_cols,
            row_count: keep_rows.len(),
        })
    }

    fn invoke_closure(
        &mut self,
        params: &[RParam],
        body: &RExpr,
        captured_env: &IndexMap<String, RObject>,
        args: &[EvalArg],
    ) -> Result<RObject, RError> {
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(RError::new("Maximum R function recursion depth exceeded"));
        }
        self.call_depth += 1;

        let mut local_frame = captured_env.clone();
        let mut bound = vec![false; params.len()];

        // 1. Bind named arguments
        for a in args {
            if let Some(ref name) = a.name {
                if let Some(idx) = params.iter().position(|p| &p.name == name) {
                    local_frame.insert(name.clone(), a.value.clone());
                    bound[idx] = true;
                }
            }
        }

        // 2. Bind positional arguments into remaining unbound parameters
        let mut next_param = 0usize;
        for a in args {
            if a.name.is_none() {
                while next_param < params.len() && bound[next_param] {
                    next_param += 1;
                }
                if next_param < params.len() {
                    local_frame.insert(params[next_param].name.clone(), a.value.clone());
                    bound[next_param] = true;
                    next_param += 1;
                }
            }
        }

        self.frames.push(local_frame);

        // 3. Evaluate defaults for any unbound parameters (inside the new frame so defaults can reference earlier params!)
        for (i, p) in params.iter().enumerate() {
            if !bound[i] {
                if let Some(ref def_expr) = p.default {
                    match self.eval_value(def_expr) {
                        Ok(v) => self.assign_var(p.name.clone(), v),
                        Err(e) => {
                            self.frames.pop();
                            self.call_depth -= 1;
                            return Err(e);
                        }
                    }
                } else {
                    self.frames.pop();
                    self.call_depth -= 1;
                    return Err(RError::new(format!(
                        "argument \"{}\" is missing, with no default",
                        p.name
                    )));
                }
            }
        }

        let res = self.eval_node(body);
        self.frames.pop();
        self.call_depth -= 1;

        match res? {
            Flow::Val(v) | Flow::Return(v) => Ok(v),
            Flow::Break | Flow::Next => Err(RError::new("`break`/`next` outside loop in function")),
        }
    }
}

fn eval_desmos_bridge_call(
    name: &str,
    args: &[EvalArg],
    bridge: &dyn DesmosBridge,
) -> Result<RObject, RError> {
    let vec_args: Vec<Vec<Option<f64>>> = args
        .iter()
        .map(|a| a.value.as_numeric_options())
        .collect();
    if vec_args.iter().any(|v| v.is_empty()) {
        return Ok(RObject::Numeric(Vec::new()));
    }
    let out_len = vec_args.iter().map(|v| v.len()).max().unwrap_or(1);
    let mut out = Vec::with_capacity(out_len);
    let mut call_buf = Vec::with_capacity(vec_args.len());

    for i in 0..out_len {
        call_buf.clear();
        let mut any_na = false;
        for v in &vec_args {
            match v[i % v.len()] {
                Some(x) if x.is_finite() => call_buf.push(x),
                _ => {
                    any_na = true;
                    break;
                }
            }
        }
        if any_na {
            out.push(None);
            continue;
        }
        match bridge.call_desmos_fn(name, &call_buf) {
            Some(Ok(y)) if y.is_finite() => out.push(Some(y)),
            Some(Ok(_)) => out.push(None),
            Some(Err(e)) => return Err(RError::new(e)),
            None => return Err(RError::new(format!("could not find function \"{}\"", name))),
        }
    }
    Ok(RObject::Numeric(out))
}

fn iter_elements(obj: &RObject) -> Vec<RObject> {
    match obj {
        RObject::Numeric(v) => v.iter().map(|&x| RObject::Numeric(vec![x])).collect(),
        RObject::Logical(v) => v.iter().map(|&x| RObject::Logical(vec![x])).collect(),
        RObject::Character(v) => v.iter().map(|x| RObject::Character(vec![x.clone()])).collect(),
        RObject::List(m) => m.values().cloned().collect(),
        RObject::DataFrame { columns, .. } => columns.values().cloned().collect(),
        _ => Vec::new(),
    }
}

fn eval_unary(op: RUnaryOp, val: &RObject) -> Result<RObject, RError> {
    match op {
        RUnaryOp::Pos => Ok(RObject::Numeric(val.as_numeric_options())),
        RUnaryOp::Neg => Ok(RObject::Numeric(
            val.as_numeric_options()
                .into_iter()
                .map(|o| o.map(|v| -v))
                .collect(),
        )),
        RUnaryOp::Not => Ok(RObject::Logical(
            val.as_logical_options()
                .into_iter()
                .map(|o| o.map(|b| !b))
                .collect(),
        )),
    }
}

fn eval_binary(op: RBinOp, lhs: &RObject, rhs: &RObject) -> Result<RObject, RError> {
    match op {
        RBinOp::Colon => {
            let a = lhs
                .as_scalar_f64()
                .ok_or_else(|| RError::new("`:` requires numeric scalar bounds"))?;
            let b = rhs
                .as_scalar_f64()
                .ok_or_else(|| RError::new("`:` requires numeric scalar bounds"))?;
            let step = if b >= a { 1.0_f64 } else { -1.0_f64 };
            let mut out = Vec::new();
            let mut curr = a;
            while ((step > 0.0_f64 && curr <= b + 1e-11_f64)
                || (step < 0.0_f64 && curr >= b - 1e-11_f64))
                && out.len() < 100_000
            {
                out.push(Some(curr));
                curr += step;
            }
            Ok(RObject::Numeric(out))
        }
        RBinOp::MatMul => eval_matmul(lhs, rhs),
        RBinOp::Add
        | RBinOp::Sub
        | RBinOp::Mul
        | RBinOp::Div
        | RBinOp::Pow
        | RBinOp::Mod
        | RBinOp::IntDiv => {
            let lv = lhs.as_numeric_options();
            let rv = rhs.as_numeric_options();
            if lv.is_empty() || rv.is_empty() {
                return Ok(RObject::Numeric(Vec::new()));
            }
            let len = lv.len().max(rv.len());
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let res = match (lv[i % lv.len()], rv[i % rv.len()]) {
                    (Some(a), Some(b)) => {
                        let v = match op {
                            RBinOp::Add => a + b,
                            RBinOp::Sub => a - b,
                            RBinOp::Mul => a * b,
                            RBinOp::Div => a / b,
                            RBinOp::Pow => a.powf(b),
                            RBinOp::Mod => a.rem_euclid(b),
                            RBinOp::IntDiv => (a / b).floor(),
                            _ => unreachable!(),
                        };
                        (!v.is_nan()).then_some(v)
                    }
                    _ => None,
                };
                out.push(res);
            }
            // Preserve matrix shape when operating on a matrix and scalar/same-size matrix
            if let RObject::Matrix {
                nrow,
                ncol,
                rownames,
                colnames,
                ..
            } = lhs
            {
                if out.len() == nrow * ncol {
                    return Ok(RObject::Matrix {
                        data: out,
                        nrow: *nrow,
                        ncol: *ncol,
                        rownames: rownames.clone(),
                        colnames: colnames.clone(),
                    });
                }
            }
            Ok(RObject::Numeric(out))
        }
        RBinOp::Eq | RBinOp::Ne | RBinOp::Lt | RBinOp::Le | RBinOp::Gt | RBinOp::Ge => {
            if matches!(lhs, RObject::Character(_)) || matches!(rhs, RObject::Character(_)) {
                let lv = lhs.as_character_options();
                let rv = rhs.as_character_options();
                if lv.is_empty() || rv.is_empty() {
                    return Ok(RObject::Logical(Vec::new()));
                }
                let len = lv.len().max(rv.len());
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let res = match (&lv[i % lv.len()], &rv[i % rv.len()]) {
                        (Some(a), Some(b)) => Some(match op {
                            RBinOp::Eq => a == b,
                            RBinOp::Ne => a != b,
                            RBinOp::Lt => a < b,
                            RBinOp::Le => a <= b,
                            RBinOp::Gt => a > b,
                            RBinOp::Ge => a >= b,
                            _ => unreachable!(),
                        }),
                        _ => None,
                    };
                    out.push(res);
                }
                return Ok(RObject::Logical(out));
            }
            let lv = lhs.as_numeric_options();
            let rv = rhs.as_numeric_options();
            if lv.is_empty() || rv.is_empty() {
                return Ok(RObject::Logical(Vec::new()));
            }
            let len = lv.len().max(rv.len());
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let res = match (lv[i % lv.len()], rv[i % rv.len()]) {
                    (Some(a), Some(b)) => Some(match op {
                        RBinOp::Eq => (a - b).abs() <= 1e-12_f64,
                        RBinOp::Ne => (a - b).abs() > 1e-12_f64,
                        RBinOp::Lt => a < b,
                        RBinOp::Le => a <= b,
                        RBinOp::Gt => a > b,
                        RBinOp::Ge => a >= b,
                        _ => unreachable!(),
                    }),
                    _ => None,
                };
                out.push(res);
            }
            Ok(RObject::Logical(out))
        }
        RBinOp::And | RBinOp::Or => {
            let lv = lhs.as_logical_options();
            let rv = rhs.as_logical_options();
            if lv.is_empty() || rv.is_empty() {
                return Ok(RObject::Logical(Vec::new()));
            }
            let len = lv.len().max(rv.len());
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let a = lv[i % lv.len()];
                let b = rv[i % rv.len()];
                // Kleene 3-valued logic matching R's `NA` semantics
                let res = match op {
                    RBinOp::And => match (a, b) {
                        (Some(false), _) | (_, Some(false)) => Some(false),
                        (Some(true), Some(true)) => Some(true),
                        _ => None,
                    },
                    RBinOp::Or => match (a, b) {
                        (Some(true), _) | (_, Some(true)) => Some(true),
                        (Some(false), Some(false)) => Some(false),
                        _ => None,
                    },
                    _ => unreachable!(),
                };
                out.push(res);
            }
            Ok(RObject::Logical(out))
        }
        RBinOp::ShortAnd | RBinOp::ShortOr => unreachable!(),
    }
}

fn eval_matmul(lhs: &RObject, rhs: &RObject) -> Result<RObject, RError> {
    let (lr, lc, ldata) = match lhs {
        RObject::Matrix {
            data, nrow, ncol, ..
        } => (*nrow, *ncol, data.clone()),
        other => {
            let v = other.as_numeric_options();
            (1, v.len(), v)
        }
    };
    let (rr, rc, rdata) = match rhs {
        RObject::Matrix {
            data, nrow, ncol, ..
        } => (*nrow, *ncol, data.clone()),
        other => {
            let v = other.as_numeric_options();
            (v.len(), 1, v)
        }
    };
    if lc != rr {
        return Err(RError::new(format!(
            "`%*%`: non-conformable arguments ([{}x{}] %*% [{}x{}])",
            lr, lc, rr, rc
        )));
    }
    let lf: Vec<f64> = ldata.iter().map(|o| o.unwrap_or(f64::NAN)).collect();
    let rf: Vec<f64> = rdata.iter().map(|o| o.unwrap_or(f64::NAN)).collect();
    let lm = DMatrix::from_column_slice(lr, lc, &lf);
    let rm = DMatrix::from_column_slice(rr, rc, &rf);
    let prod = lm * rm;
    let out: Vec<Option<f64>> = prod
        .as_slice()
        .iter()
        .map(|&v| (!v.is_nan()).then_some(v))
        .collect();
    Ok(RObject::Matrix {
        data: out,
        nrow: lr,
        ncol: rc,
        rownames: None,
        colnames: None,
    })
}

fn eval_member(obj: &RObject, field: &str) -> Result<RObject, RError> {
    match obj {
        RObject::List(map) => Ok(map.get(field).cloned().unwrap_or(RObject::Null)),
        RObject::DataFrame { columns, .. } => {
            Ok(columns.get(field).cloned().unwrap_or(RObject::Null))
        }
        _ => Err(RError::new("`$ operator is invalid for atomic vectors`")),
    }
}

fn eval_double_index(obj: &RObject, idx: &RObject) -> Result<RObject, RError> {
    match (obj, idx) {
        (RObject::List(map), RObject::Character(names))
        | (RObject::DataFrame { columns: map, .. }, RObject::Character(names)) => {
            let key = names
                .first()
                .and_then(|o| o.as_deref())
                .ok_or_else(|| RError::new("invalid subscript in `[[`"))?;
            Ok(map.get(key).cloned().unwrap_or(RObject::Null))
        }
        (RObject::List(map), _) | (RObject::DataFrame { columns: map, .. }, _) => {
            let pos = idx
                .as_scalar_f64()
                .ok_or_else(|| RError::new("invalid numeric subscript in `[[`"))?
                .round() as isize;
            if pos >= 1 && (pos as usize) <= map.len() {
                Ok(map
                    .get_index((pos - 1) as usize)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(RObject::Null))
            } else {
                Err(RError::new("subscript out of bounds in `[[`"))
            }
        }
        (RObject::Numeric(v), _) => {
            let pos = idx
                .as_scalar_f64()
                .ok_or_else(|| RError::new("invalid subscript in `[[`"))?
                .round() as isize;
            if pos >= 1 && (pos as usize) <= v.len() {
                Ok(RObject::Numeric(vec![v[(pos - 1) as usize]]))
            } else {
                Err(RError::new("subscript out of bounds in `[[`"))
            }
        }
        _ => Err(RError::new("`[[` indexing not supported on this object")),
    }
}

/// Resolves R-style 1-based positive, negative, or logical-mask indices against `len`.
/// Returns `Vec<Option<usize>>` (0-based indices, where `None` represents out-of-bounds / `NA`).
fn resolve_indices(idx_obj: &RObject, len: usize) -> Result<Vec<Option<usize>>, RError> {
    match idx_obj {
        RObject::Logical(mask) => {
            if mask.is_empty() {
                return Ok(Vec::new());
            }
            let mut out = Vec::new();
            for i in 0..len {
                match mask[i % mask.len()] {
                    Some(true) => out.push(Some(i)),
                    Some(false) => {}
                    None => out.push(None),
                }
            }
            Ok(out)
        }
        _ => {
            let raw = idx_obj.as_numeric_options();
            let has_pos = raw.iter().any(|o| matches!(o, Some(v) if *v > 0.0_f64));
            let has_neg = raw.iter().any(|o| matches!(o, Some(v) if *v < 0.0_f64));
            if has_pos && has_neg {
                return Err(RError::new("only 0's may be mixed with negative subscripts"));
            }
            if has_neg {
                let excluded: Vec<usize> = raw
                    .iter()
                    .filter_map(|o| match o {
                        Some(v) if *v < 0.0_f64 => Some(((-v).round() as usize).saturating_sub(1)),
                        _ => None,
                    })
                    .collect();
                let mut out = Vec::new();
                for i in 0..len {
                    if !excluded.contains(&i) {
                        out.push(Some(i));
                    }
                }
                return Ok(out);
            }
            let mut out = Vec::with_capacity(raw.len());
            for opt in raw {
                match opt {
                    Some(v) => {
                        let k = v.trunc() as isize;
                        if k == 0 {
                            continue;
                        }
                        if k >= 1 && (k as usize) <= len {
                            out.push(Some((k - 1) as usize));
                        } else {
                            out.push(None);
                        }
                    }
                    None => out.push(None),
                }
            }
            Ok(out)
        }
    }
}

fn slice_vector<T: Clone>(src: &[Option<T>], indices: &[Option<usize>]) -> Vec<Option<T>> {
    indices
        .iter()
        .map(|opt| opt.and_then(|i| src.get(i).cloned().flatten()))
        .collect()
}

fn slice_rows_by_indices(col: &RObject, rows: &[usize]) -> RObject {
    match col {
        RObject::Numeric(v) => RObject::Numeric(
            rows.iter()
                .map(|&r| v.get(r).copied().flatten())
                .collect(),
        ),
        RObject::Logical(v) => RObject::Logical(
            rows.iter()
                .map(|&r| v.get(r).copied().flatten())
                .collect(),
        ),
        RObject::Character(v) => RObject::Character(
            rows.iter()
                .map(|&r| v.get(r).cloned().flatten())
                .collect(),
        ),
        other => other.clone(),
    }
}

fn eval_index(obj: &RObject, indices: &[Option<RObject>]) -> Result<RObject, RError> {
    if indices.is_empty() {
        return Ok(obj.clone());
    }

    match obj {
        RObject::Numeric(v) => {
            let Some(ref idx) = indices[0] else {
                return Ok(obj.clone());
            };
            let resolved = resolve_indices(idx, v.len())?;
            Ok(RObject::Numeric(slice_vector(v, &resolved)))
        }
        RObject::Logical(v) => {
            let Some(ref idx) = indices[0] else {
                return Ok(obj.clone());
            };
            let resolved = resolve_indices(idx, v.len())?;
            Ok(RObject::Logical(slice_vector(v, &resolved)))
        }
        RObject::Character(v) => {
            let Some(ref idx) = indices[0] else {
                return Ok(obj.clone());
            };
            let resolved = resolve_indices(idx, v.len())?;
            Ok(RObject::Character(slice_vector(v, &resolved)))
        }
        RObject::Matrix {
            data, nrow, ncol, ..
        } => {
            if indices.len() == 1 {
                let Some(ref idx) = indices[0] else {
                    return Ok(obj.clone());
                };
                let resolved = resolve_indices(idx, data.len())?;
                return Ok(RObject::Numeric(slice_vector(data, &resolved)));
            }
            let r_idxs: Vec<Option<usize>> = match &indices[0] {
                Some(r_obj) => resolve_indices(r_obj, *nrow)?,
                None => (0..*nrow).map(Some).collect(),
            };
            let c_idxs: Vec<Option<usize>> = match &indices[1] {
                Some(c_obj) => resolve_indices(c_obj, *ncol)?,
                None => (0..*ncol).map(Some).collect(),
            };
            let out_r = r_idxs.len();
            let out_c = c_idxs.len();
            let mut out = Vec::with_capacity(out_r * out_c);
            for &c_opt in &c_idxs {
                for &r_opt in &r_idxs {
                    let val = match (r_opt, c_opt) {
                        (Some(r), Some(c)) => data.get(c * nrow + r).copied().flatten(),
                        _ => None,
                    };
                    out.push(val);
                }
            }
            if out_r == 1 || out_c == 1 {
                Ok(RObject::Numeric(out))
            } else {
                Ok(RObject::Matrix {
                    data: out,
                    nrow: out_r,
                    ncol: out_c,
                    rownames: None,
                    colnames: None,
                })
            }
        }
        RObject::DataFrame { columns, row_count } => {
            if indices.len() == 1 {
                // `df["col"]` or `df[1]` selects columns
                let Some(ref c_obj) = indices[0] else {
                    return Ok(obj.clone());
                };
                let col_keys = resolve_column_selection(columns, c_obj)?;
                let mut new_cols = IndexMap::new();
                for k in col_keys {
                    if let Some(col) = columns.get(&k) {
                        new_cols.insert(k, col.clone());
                    }
                }
                return Ok(RObject::DataFrame {
                    columns: new_cols,
                    row_count: *row_count,
                });
            }

            let row_idxs: Vec<Option<usize>> = match &indices[0] {
                Some(r_obj) => resolve_indices(r_obj, *row_count)?,
                None => (0..*row_count).map(Some).collect(),
            };
            let col_keys: Vec<String> = match &indices[1] {
                Some(c_obj) => resolve_column_selection(columns, c_obj)?,
                None => columns.keys().cloned().collect(),
            };

            // R behavior: `df[, "x"]` (single column) drops to vector; multi-col stays DataFrame
            if col_keys.len() == 1 && indices[1].is_some() {
                if let Some(col) = columns.get(&col_keys[0]) {
                    return Ok(slice_column_opt(col, &row_idxs));
                }
            }

            let mut new_cols = IndexMap::new();
            for k in col_keys {
                if let Some(col) = columns.get(&k) {
                    new_cols.insert(k, slice_column_opt(col, &row_idxs));
                }
            }
            Ok(RObject::DataFrame {
                columns: new_cols,
                row_count: row_idxs.len(),
            })
        }
        RObject::List(map) => {
            let Some(ref idx) = indices[0] else {
                return Ok(obj.clone());
            };
            let keys = resolve_column_selection(map, idx)?;
            let mut out = IndexMap::new();
            for k in keys {
                if let Some(v) = map.get(&k) {
                    out.insert(k, v.clone());
                }
            }
            Ok(RObject::List(out))
        }
        _ => Err(RError::new("object of this type is not subsettable")),
    }
}

fn slice_column_opt(col: &RObject, row_idxs: &[Option<usize>]) -> RObject {
    match col {
        RObject::Numeric(v) => RObject::Numeric(slice_vector(v, row_idxs)),
        RObject::Logical(v) => RObject::Logical(slice_vector(v, row_idxs)),
        RObject::Character(v) => RObject::Character(slice_vector(v, row_idxs)),
        other => other.clone(),
    }
}

fn resolve_column_selection(
    map: &IndexMap<String, RObject>,
    sel: &RObject,
) -> Result<Vec<String>, RError> {
    if let RObject::Character(names) = sel {
        let mut out = Vec::new();
        for n in names.iter().flatten() {
            if map.contains_key(n) {
                out.push(n.clone());
            } else {
                return Err(RError::new(format!("undefined column selected: `{}`", n)));
            }
        }
        return Ok(out);
    }
    let resolved = resolve_indices(sel, map.len())?;
    let mut out = Vec::new();
    for idx in resolved.into_iter().flatten() {
        if let Some((k, _)) = map.get_index(idx) {
            out.push(k.clone());
        }
    }
    Ok(out)
}

fn assign_index(
    obj: &mut RObject,
    indices: &[Option<RObject>],
    value: &RObject,
) -> Result<(), RError> {
    match obj {
        RObject::Numeric(v) => {
            let Some(Some(idx)) = indices.first() else {
                return Ok(());
            };
            let resolved = resolve_indices(idx, v.len())?;
            let rhs = value.as_numeric_options();
            if rhs.is_empty() {
                return Ok(());
            }
            for (k, slot_opt) in resolved.iter().enumerate() {
                if let Some(slot) = slot_opt {
                    if *slot < v.len() {
                        v[*slot] = rhs[k % rhs.len()];
                    }
                }
            }
            Ok(())
        }
        RObject::Matrix {
            data, nrow, ncol, ..
        } if indices.len() == 2 => {
            let r_idxs: Vec<Option<usize>> = match &indices[0] {
                Some(r_obj) => resolve_indices(r_obj, *nrow)?,
                None => (0..*nrow).map(Some).collect(),
            };
            let c_idxs: Vec<Option<usize>> = match &indices[1] {
                Some(c_obj) => resolve_indices(c_obj, *ncol)?,
                None => (0..*ncol).map(Some).collect(),
            };
            let rhs = value.as_numeric_options();
            if rhs.is_empty() {
                return Ok(());
            }
            let mut k = 0usize;
            for &c_opt in &c_idxs {
                for &r_opt in &r_idxs {
                    if let (Some(r), Some(c)) = (r_opt, c_opt) {
                        let pos = c * (*nrow) + r;
                        if pos < data.len() {
                            data[pos] = rhs[k % rhs.len()];
                        }
                    }
                    k += 1;
                }
            }
            Ok(())
        }
        _ => Err(RError::new("Indexed assignment not supported for this type")),
    }
}
