pub mod ast;
pub mod eval;
pub mod lexer;
pub mod lm;
pub mod object;
pub mod parser;
pub mod stdlib;

use crate::collections::IndexMap;
use crate::math::MathExternalEnv;
pub use ast::RError;
pub use eval::{DesmosBridge, REvaluator};
pub use object::RObject;
pub use parser::parse_r_program;

#[derive(Debug, Clone)]
pub struct RExecResult {
    pub value: RObject,
    pub stdout: Vec<String>,
}

/// Clean execution trait decoupling UI/Workspace from the R backend so an optional Tier-2
/// out-of-process R kernel can be swapped in without changing callers.
#[allow(dead_code)]
pub trait RExecutor {
    fn execute(
        &mut self,
        code: &str,
        bridge: Option<&dyn DesmosBridge>,
    ) -> Result<RExecResult, RError>;

    fn env(&self) -> &IndexMap<String, RObject>;
    fn env_mut(&mut self) -> &mut IndexMap<String, RObject>;

    fn get(&self, name: &str) -> Option<&RObject> {
        self.env().get(name)
    }

    fn set(&mut self, name: String, value: RObject) {
        self.env_mut().insert(name, value);
    }

    fn remove(&mut self, name: &str) -> Option<RObject> {
        self.env_mut().remove(name)
    }

    fn clear(&mut self) {
        self.env_mut().clear();
    }
}

/// Tier-1 100% offline embedded R-subset interpreter.
#[derive(Debug, Clone, Default)]
pub struct EmbeddedREngine {
    #[allow(dead_code)]
    env: IndexMap<String, RObject>,
}

impl EmbeddedREngine {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            env: IndexMap::new(),
        }
    }

    /// Evaluates R code against an externally owned environment map (used by `Workspace`).
    pub fn eval_in_env(
        env: &mut IndexMap<String, RObject>,
        code: &str,
        bridge: Option<&dyn DesmosBridge>,
    ) -> Result<RExecResult, RError> {
        let stmts = parse_r_program(code)?;
        let mut evaluator = REvaluator::new(env, bridge);
        let value = evaluator.eval_program(&stmts)?;
        Ok(RExecResult {
            value,
            stdout: evaluator.stdout,
        })
    }

    /// Evaluates R code in read-only mode (on a cloned env) for pure expression previews.
    pub fn eval_readonly(
        env: &IndexMap<String, RObject>,
        code: &str,
        bridge: Option<&dyn DesmosBridge>,
    ) -> Result<RObject, RError> {
        let mut cloned = env.clone();
        let res = Self::eval_in_env(&mut cloned, code, bridge)?;
        Ok(res.value)
    }
}

impl RExecutor for EmbeddedREngine {
    fn execute(
        &mut self,
        code: &str,
        bridge: Option<&dyn DesmosBridge>,
    ) -> Result<RExecResult, RError> {
        Self::eval_in_env(&mut self.env, code, bridge)
    }

    fn env(&self) -> &IndexMap<String, RObject> {
        &self.env
    }

    fn env_mut(&mut self) -> &mut IndexMap<String, RObject> {
        &mut self.env
    }
}

/// Exposes an R environment (`IndexMap<String, RObject>`) to the Desmos math engine (`src/math/`),
/// enabling Desmos expressions to reference R scalars, vectors, DataFrame columns (`df$y`),
/// fitted `lm` models, and R closures.
pub struct REngineMathBridge<'a> {
    pub r_objects: &'a IndexMap<String, RObject>,
}

impl MathExternalEnv for REngineMathBridge<'_> {
    fn get_scalar(&self, name: &str) -> Option<f64> {
        self.r_objects.get(name)?.as_scalar_f64()
    }

    fn get_vector_element(&self, name: &str, index_1based: usize) -> Option<f64> {
        if index_1based == 0 {
            return None;
        }
        let obj = self.r_objects.get(name)?;
        let vals = obj.as_numeric_options();
        vals.get(index_1based - 1).copied().flatten()
    }

    fn get_member_scalar(&self, object: &str, field: &str) -> Option<f64> {
        let obj = self.r_objects.get(object)?;
        let member = match obj {
            RObject::DataFrame { columns, .. } => columns.get(field)?,
            RObject::List(map) => map.get(field)?,
            _ => return None,
        };
        if let Some(s) = member.as_scalar_f64() {
            return Some(s);
        }
        member.as_numeric_options().into_iter().next().flatten()
    }

    fn get_member_element(&self, object: &str, field: &str, index_1based: usize) -> Option<f64> {
        if index_1based == 0 {
            return None;
        }
        let obj = self.r_objects.get(object)?;
        let member = match obj {
            RObject::DataFrame { columns, .. } => columns.get(field)?,
            RObject::List(map) => map.get(field)?,
            _ => return None,
        };
        let vals = member.as_numeric_options();
        vals.get(index_1based - 1).copied().flatten()
    }

    fn call_function(&self, name: &str, args: &[f64]) -> Option<Result<f64, String>> {
        let obj = self.r_objects.get(name)?;

        // 1. User-defined R closure called from a Desmos expression: e.g. `my_r_fn(x)`
        if matches!(obj, RObject::Closure { .. }) {
            let arg_list: Vec<String> = args.iter().map(|v| format!("({:.15})", v)).collect();
            let call_src = format!("{}({})", name, arg_list.join(", "));
            return Some(
                EmbeddedREngine::eval_readonly(self.r_objects, &call_src, None)
                    .map_err(|e| e.message)
                    .and_then(|res| {
                        res.as_numeric_options()
                            .into_iter()
                            .next()
                            .flatten()
                            .ok_or_else(|| format!("R function `{}` returned non-numeric/NA", name))
                    }),
            );
        }

        // 2. Fitted `lm` model called as a function in Desmos: e.g. `fit(x)` -> `predict(fit, x)`
        if obj.is_lm_model() && args.len() == 1 {
            let newdata = RObject::scalar(args[0]);
            return Some(
                lm::predict_lm(obj, Some(&newdata), self.r_objects)
                    .map_err(|e| e.message)
                    .and_then(|res| {
                        res.as_numeric_options()
                            .into_iter()
                            .next()
                            .flatten()
                            .ok_or_else(|| format!("Model `{}` prediction returned NA", name))
                    }),
            );
        }

        None
    }
}

/// Extracts a 2D scatter/line series `Vec<[f64; 2]>` when a Desmos expression references an R
/// vector, `df$col`, `predict(fit)`, or `(df$x, df$y)` pair directly.
pub fn try_extract_r_plot_series(
    expr_str: &str,
    r_objects: &IndexMap<String, RObject>,
) -> Option<Vec<[f64; 2]>> {
    let trimmed = expr_str.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Pattern 1: `(x_r_expr, y_r_expr)` where both evaluate to R vectors of len >= 2
    if let Some(inner) = trimmed.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        if let Some((left, right)) = split_top_level_pair(inner) {
            if let (Ok(lx), Ok(ry)) = (
                EmbeddedREngine::eval_readonly(r_objects, left, None),
                EmbeddedREngine::eval_readonly(r_objects, right, None),
            ) {
                let xs = lx.as_numeric_options();
                let ys = ry.as_numeric_options();
                if xs.len() >= 2 && ys.len() >= 2 {
                    let n = xs.len().min(ys.len());
                    let mut pts = Vec::with_capacity(n);
                    for i in 0..n {
                        if let (Some(x), Some(y)) = (xs[i], ys[i]) {
                            if x.is_finite() && y.is_finite() {
                                pts.push([x, y]);
                            }
                        }
                    }
                    if !pts.is_empty() {
                        return Some(pts);
                    }
                }
            }
        }
    }

    // Pattern 2: single R vector expression (`df$y`, `predict(fit)`, `fitted(fit)`, `v`)
    if trimmed.contains('$')
        || trimmed.starts_with("predict(")
        || trimmed.starts_with("fitted(")
        || trimmed.starts_with("residuals(")
        || r_objects
            .get(trimmed)
            .map(|o| o.r_len() >= 2)
            .unwrap_or(false)
    {
        if let Ok(obj) = EmbeddedREngine::eval_readonly(r_objects, trimmed, None) {
            let ys = obj.as_numeric_options();
            if ys.len() >= 2 {
                let mut pts = Vec::with_capacity(ys.len());
                for (i, opt_y) in ys.into_iter().enumerate() {
                    if let Some(y) = opt_y {
                        if y.is_finite() {
                            pts.push([(i + 1) as f64, y]);
                        }
                    }
                }
                if !pts.is_empty() {
                    return Some(pts);
                }
            }
        }
    }

    None
}

fn split_top_level_pair(s: &str) -> Option<(&str, &str)> {
    let mut depth = 0usize;
    let mut split_idx = None;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                if split_idx.is_some() {
                    return None;
                }
                split_idx = Some(i);
            }
            _ => {}
        }
    }
    let idx = split_idx?;
    Some((s[..idx].trim(), s[idx + 1..].trim()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{parse_expr, CompiledExpr, CompiledWorkspace, EvalContext, EvalScratch};

    #[test]
    fn test_recycling_and_na_propagation() {
        let mut engine = EmbeddedREngine::new();

        // Standard R length recycling
        let res = engine.execute("c(1, 2, 3, 4) + c(10, 20)", None).unwrap();
        assert_eq!(
            res.value,
            RObject::Numeric(vec![Some(11.0), Some(22.0), Some(13.0), Some(24.0)])
        );

        // NA arithmetic & Kleene 3-valued logic
        let na_arith = engine.execute("c(1, NA, 3) + 5", None).unwrap();
        assert_eq!(
            na_arith.value,
            RObject::Numeric(vec![Some(6.0), None, Some(8.0)])
        );

        let kleene = engine
            .execute("c(TRUE | NA, FALSE & NA, TRUE & NA, FALSE | NA)", None)
            .unwrap();
        assert_eq!(
            kleene.value,
            RObject::Logical(vec![Some(true), Some(false), None, None])
        );

        // na.rm handling
        assert_eq!(
            engine.execute("mean(c(2, NA, 6))", None).unwrap().value,
            RObject::Numeric(vec![None])
        );
        assert_eq!(
            engine
                .execute("mean(c(2, NA, 6), na.rm = TRUE)", None)
                .unwrap()
                .value,
            RObject::scalar(4.0)
        );
    }

    #[test]
    fn test_indexing_subsetting_matrix_dataframe_and_pipe() {
        let mut engine = EmbeddedREngine::new();
        engine.execute("v <- c(10, -5, 20, -2, 30)", None).unwrap();

        assert_eq!(
            engine.execute("v[-c(2, 4)]", None).unwrap().value,
            RObject::numeric_vec(vec![10.0, 20.0, 30.0])
        );
        assert_eq!(
            engine.execute("v[v > 0]", None).unwrap().value,
            RObject::numeric_vec(vec![10.0, 20.0, 30.0])
        );

        // Matrix slicing (column-major)
        engine
            .execute("m <- matrix(1:6, nrow = 2, ncol = 3)", None)
            .unwrap();
        assert_eq!(
            engine.execute("m[2, 3]", None).unwrap().value,
            RObject::scalar(6.0)
        );
        assert_eq!(
            engine.execute("m[, 2]", None).unwrap().value,
            RObject::numeric_vec(vec![3.0, 4.0])
        );

        // DataFrame access & subset() & native pipe |>
        engine
            .execute("df <- data.frame(x = 1:5, y = c(2, 4, 6, 8, 10))", None)
            .unwrap();
        assert_eq!(
            engine.execute("df$y", None).unwrap().value,
            RObject::numeric_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0])
        );
        assert_eq!(
            engine.execute("df[[\"x\"]][3]", None).unwrap().value,
            RObject::scalar(3.0)
        );
        let piped = engine
            .execute("df |> subset(x > 2) |> nrow()", None)
            .unwrap();
        assert_eq!(piped.value, RObject::scalar(3.0));
    }

    #[test]
    fn test_control_flow_and_closures_with_defaults() {
        let mut engine = EmbeddedREngine::new();
        let script = r#"
            total <- 0
            for (i in 1:5) {
                if (i == 3) {
                    next
                }
                total <- total + i
            }
            scale_add <- function(x, mult = 2, offset = 1) {
                return(x * mult + offset)
            }
            c(total, scale_add(5), scale_add(5, offset = 10))
        "#;
        let out = engine.execute(script, None).unwrap();
        // total = 1+2+4+5 = 12; scale_add(5) = 11; scale_add(5, offset=10) = 20
        assert_eq!(out.value, RObject::numeric_vec(vec![12.0, 11.0, 20.0]));
    }

    #[test]
    fn test_lm_regression_accuracy_and_generics() {
        let mut engine = EmbeddedREngine::new();
        // True model: y = 2.0 + 3.0 * x1 - 1.5 * x2
        let script = r#"
            x1 <- c(1, 2, 3, 4, 5, 6)
            x2 <- c(2, 1, 4, 3, 6, 5)
            y  <- 2.0 + 3.0 * x1 - 1.5 * x2
            df <- data.frame(x1 = x1, x2 = x2, y = y)
            fit <- lm(y ~ x1 + x2, data = df)
        "#;
        engine.execute(script, None).unwrap();

        let coefs = engine
            .execute("coef(fit)", None)
            .unwrap()
            .value
            .as_f64_vec();
        assert_eq!(coefs.len(), 3);
        assert!((coefs[0] - 2.0).abs() < 1e-9, "b0 = {}", coefs[0]);
        assert!((coefs[1] - 3.0).abs() < 1e-9, "b1 = {}", coefs[1]);
        assert!((coefs[2] - (-1.5)).abs() < 1e-9, "b2 = {}", coefs[2]);

        let resids = engine
            .execute("residuals(fit)", None)
            .unwrap()
            .value
            .as_f64_vec();
        assert!(resids.iter().all(|e| e.abs() < 1e-9));

        let preds = engine
            .execute("predict(fit, data.frame(x1 = c(10), x2 = c(4)))", None)
            .unwrap()
            .value
            .as_f64_vec();
        // 2 + 3*10 - 1.5*4 = 26
        assert!((preds[0] - 26.0).abs() < 1e-9);

        let sum_txt = engine.execute("summary(fit)", None).unwrap().value;
        assert!(sum_txt
            .preview_string()
            .contains("Multiple R-squared: 1.00000"));
    }

    #[test]
    fn test_bidirectional_desmos_r_bridge() {
        // Set up Desmos workspace with parameter `a = 3.0` and function `f(x) = a * x^2`
        let mut desmos_params = IndexMap::new();
        desmos_params.insert("a".to_string(), 3.0_f64);
        let mut desmos_fns = IndexMap::new();
        desmos_fns.insert("f(x)".to_string(), "a * x^2".to_string());
        let compiled_ws = CompiledWorkspace::build(&desmos_fns);

        struct TestBridge<'a> {
            params: &'a IndexMap<String, f64>,
            ws: &'a CompiledWorkspace,
        }
        impl DesmosBridge for TestBridge<'_> {
            fn has_desmos_fn(&self, name: &str) -> bool {
                self.ws.functions.contains_key(name)
            }
            fn call_desmos_fn(&self, name: &str, args: &[f64]) -> Option<Result<f64, String>> {
                let f = self.ws.functions.get(name)?;
                let ctx = EvalContext::new(self.params, &self.ws.functions);
                let mut overrides = Vec::new();
                for (p, &v) in f.params.iter().zip(args.iter()) {
                    overrides.push((p.as_str(), v));
                }
                let mut scr = EvalScratch::new();
                Some(
                    f.body
                        .eval(&ctx, &overrides, &mut scr)
                        .map_err(|e| e.message),
                )
            }
            fn get_desmos_param(&self, name: &str) -> Option<f64> {
                self.params.get(name).copied()
            }
        }

        let bridge = TestBridge {
            params: &desmos_params,
            ws: &compiled_ws,
        };

        // 1. R -> Desmos: evaluate Desmos `f(x)` over R `seq(1, 3, by = 1)` and read slider `a`
        let mut r_engine = EmbeddedREngine::new();
        let r_res = r_engine
            .execute("ys <- f(seq(1, 3, by = 1)); ys + a", Some(&bridge))
            .unwrap();
        // f(1)=3, f(2)=12, f(3)=27; +3 => 6, 15, 30
        assert_eq!(r_res.value, RObject::numeric_vec(vec![6.0, 15.0, 30.0]));

        // 2. Desmos -> R: evaluate R closure, DataFrame column element, and lm prediction inside Desmos AST
        r_engine
            .execute(
                r#"
                df <- data.frame(x = c(1, 2, 3, 4), y = c(5, 7, 9, 11))
                fit <- lm(y ~ x, data = df)
                my_fn <- function(t) t * 10 + 1
                "#,
                Some(&bridge),
            )
            .unwrap();

        let math_bridge = REngineMathBridge {
            r_objects: r_engine.env(),
        };
        let math_ctx =
            EvalContext::new(&desmos_params, &compiled_ws.functions).with_external(&math_bridge);
        let mut scratch = EvalScratch::new();

        let ast1 = parse_expr("df$y[3] + my_fn(2) + fit(5)").unwrap();
        let val1 = CompiledExpr::compile(&ast1)
            .eval(&math_ctx, &[], &mut scratch)
            .unwrap();
        // df$y[3] = 9, my_fn(2) = 21, fit(5) = 3 + 2*5 = 13 => 43
        assert!((val1 - 43.0).abs() < 1e-9, "got {}", val1);
    }

    #[test]
    fn test_r_engine_security_bounds() {
        let mut engine = EmbeddedREngine::new();
        assert!(engine
            .execute("matrix(1, nrow = 1e12, ncol = 1e12)", None)
            .is_err());
        assert!(engine
            .execute("matrix(1, nrow = 1000, ncol = 1000, byrow = TRUE)", None)
            .is_err());

        let deep_r = format!("{}1{}", "(".repeat(400), ")".repeat(400));
        assert!(engine.execute(&deep_r, None).is_err());
    }
}
