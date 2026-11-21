use super::ast::{MathExpr, SpannedExpr};
use super::parser::is_builtin_function;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleError {
    pub cycle: Vec<String>,
}

impl std::fmt::Display for CycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cyclic function dependency: {}", self.cycle.join(" -> "))
    }
}

impl std::error::Error for CycleError {}

/// Parsed user-defined function signature & body for dependency resolution and evaluation.
#[derive(Debug, Clone)]
pub struct ParsedFunction {
    #[allow(dead_code)]
    pub sig: String,
    pub name: String,
    pub params: Vec<String>,
    pub body: SpannedExpr,
}

/// Parses a function signature string like `"f(x)"` or `"u(x, y)"` into `(name, params)`.
pub fn parse_function_signature(sig: &str) -> Option<(String, Vec<String>)> {
    let trimmed = sig.trim();
    let (raw_name, rest) = trimmed.split_once('(')?;
    let inner = rest.strip_suffix(')')?;
    let name = raw_name.trim().trim_start_matches('\\').to_string();
    if name.is_empty() {
        return None;
    }
    let params: Vec<String> = inner
        .split(',')
        .map(|p| p.trim().trim_start_matches('\\').to_string())
        .filter(|p| !p.is_empty())
        .collect();
    Some((name, params))
}

/// Collects all function names called within `expr` (excluding builtins).
pub fn collect_called_functions(expr: &SpannedExpr, out: &mut HashSet<String>) {
    match &expr.expr {
        MathExpr::Call { name, args, .. } => {
            if !is_builtin_function(name) {
                out.insert(name.clone());
            }
            for a in args {
                collect_called_functions(a, out);
            }
        }
        MathExpr::Number(_)
        | MathExpr::Ident(_)
        | MathExpr::Placeholder
        | MathExpr::MemberAccess { .. } => {}
        MathExpr::Subscript { base, sub } => {
            collect_called_functions(base, out);
            collect_called_functions(sub, out);
        }
        MathExpr::Accent { inner, .. }
        | MathExpr::Group(inner)
        | MathExpr::Abs(inner)
        | MathExpr::Unary { expr: inner, .. }
        | MathExpr::Derivative { body: inner, .. } => {
            collect_called_functions(inner, out);
        }
        MathExpr::Binary { lhs, rhs, .. } => {
            collect_called_functions(lhs, out);
            collect_called_functions(rhs, out);
        }
        MathExpr::Fraction { num, den } => {
            collect_called_functions(num, out);
            collect_called_functions(den, out);
        }
        MathExpr::Sqrt { index, radicand } => {
            if let Some(i) = index {
                collect_called_functions(i, out);
            }
            collect_called_functions(radicand, out);
        }
        MathExpr::Integral {
            lower, upper, body, ..
        }
        | MathExpr::Sum {
            lower, upper, body, ..
        }
        | MathExpr::Prod {
            lower, upper, body, ..
        } => {
            collect_called_functions(lower, out);
            collect_called_functions(upper, out);
            collect_called_functions(body, out);
        }
        MathExpr::ComparisonChain { first, rest } => {
            collect_called_functions(first, out);
            for (_, r) in rest {
                collect_called_functions(r, out);
            }
        }
        MathExpr::Piecewise(branches) => {
            for b in branches {
                if let Some(c) = &b.condition {
                    collect_called_functions(c, out);
                }
                collect_called_functions(&b.value, out);
            }
        }
        MathExpr::Restricted {
            inner,
            restrictions,
        } => {
            collect_called_functions(inner, out);
            for r in restrictions {
                collect_called_functions(r, out);
            }
        }
        MathExpr::Tuple(items) => {
            for item in items {
                collect_called_functions(item, out);
            }
        }
        MathExpr::Index { target, index } => {
            collect_called_functions(target, out);
            collect_called_functions(index, out);
        }
    }
}

/// Topological sort and cycle detection over user-defined workspace functions.
pub fn topological_sort_functions(
    functions: &[ParsedFunction],
) -> Result<Vec<String>, CycleError> {
    let defined: HashSet<&str> = functions.iter().map(|f| f.name.as_str()).collect();
    let mut deps: HashMap<&str, Vec<&str>> = HashMap::new();

    for f in functions {
        let mut called = HashSet::new();
        collect_called_functions(&f.body, &mut called);
        let mut local_deps: Vec<&str> = called
            .iter()
            .filter_map(|c| defined.get(c.as_str()).copied())
            .collect();
        local_deps.sort_unstable();
        deps.insert(f.name.as_str(), local_deps);
    }

    // 0 = unvisited, 1 = visiting (on stack), 2 = done
    let mut state: HashMap<&str, u8> = HashMap::new();
    let mut stack: Vec<&str> = Vec::new();
    let mut order: Vec<String> = Vec::with_capacity(functions.len());

    fn dfs<'a>(
        node: &'a str,
        deps: &HashMap<&'a str, Vec<&'a str>>,
        state: &mut HashMap<&'a str, u8>,
        stack: &mut Vec<&'a str>,
        order: &mut Vec<String>,
    ) -> Result<(), CycleError> {
        state.insert(node, 1);
        stack.push(node);

        if let Some(neighbors) = deps.get(node) {
            for &dep in neighbors {
                match state.get(dep).copied().unwrap_or(0) {
                    1 => {
                        let start_idx = stack.iter().position(|&n| n == dep).unwrap_or(0);
                        let mut cycle: Vec<String> =
                            stack[start_idx..].iter().map(|s| (*s).to_string()).collect();
                        cycle.push(dep.to_string());
                        return Err(CycleError { cycle });
                    }
                    0 => dfs(dep, deps, state, stack, order)?,
                    _ => {}
                }
            }
        }

        stack.pop();
        state.insert(node, 2);
        order.push(node.to_string());
        Ok(())
    }

    for f in functions {
        if state.get(f.name.as_str()).copied().unwrap_or(0) == 0 {
            dfs(f.name.as_str(), &deps, &mut state, &mut stack, &mut order)?;
        }
    }

    Ok(order)
}

fn is_reserved_or_coord(name: &str) -> bool {
    matches!(
        name,
        "x" | "y"
            | "z"
            | "t"
            | "theta"
            | "r"
            | "L"
            | "K"
            | "Q"
            | "Y"
            | "e"
            | "pi"
            | "PI"
            | "phi"
            | "infty"
    )
}

/// Discovers free (unbound) parameter symbols referenced in `expr` that do not yet exist in
/// `bound_locals`, `known_fns`, `existing_params`, or `existing_r_objects`.
pub fn discover_unbound_symbols(
    expr: &SpannedExpr,
    bound_locals: &HashSet<String>,
    known_fns: &HashSet<String>,
    existing_params: &HashSet<String>,
    existing_r_objects: &HashSet<String>,
) -> Vec<String> {
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    let mut locals = bound_locals.clone();

    fn walk(
        node: &SpannedExpr,
        locals: &mut HashSet<String>,
        known_fns: &HashSet<String>,
        existing_params: &HashSet<String>,
        existing_r_objects: &HashSet<String>,
        found: &mut Vec<String>,
        seen: &mut HashSet<String>,
    ) {
        let mut maybe_record = |ident: &str| {
            if !is_reserved_or_coord(ident)
                && !is_builtin_function(ident)
                && !locals.contains(ident)
                && !known_fns.contains(ident)
                && !existing_params.contains(ident)
                && !existing_r_objects.contains(ident)
                && seen.insert(ident.to_string())
            {
                found.push(ident.to_string());
            }
        };

        match &node.expr {
            MathExpr::Number(_) | MathExpr::Placeholder => {}
            MathExpr::Ident(name) => maybe_record(name),
            MathExpr::Accent { .. } | MathExpr::Subscript { .. } => {
                if let Some(canon) = node.as_canonical_ident() {
                    maybe_record(&canon);
                } else {
                    match &node.expr {
                        MathExpr::Accent { inner, .. } => {
                            walk(inner, locals, known_fns, existing_params, existing_r_objects, found, seen);
                        }
                        MathExpr::Subscript { base, sub } => {
                            walk(base, locals, known_fns, existing_params, existing_r_objects, found, seen);
                            walk(sub, locals, known_fns, existing_params, existing_r_objects, found, seen);
                        }
                        _ => {}
                    }
                }
            }
            MathExpr::Group(inner)
            | MathExpr::Abs(inner)
            | MathExpr::Unary { expr: inner, .. } => {
                walk(inner, locals, known_fns, existing_params, existing_r_objects, found, seen);
            }
            MathExpr::Binary { lhs, rhs, .. } => {
                walk(lhs, locals, known_fns, existing_params, existing_r_objects, found, seen);
                walk(rhs, locals, known_fns, existing_params, existing_r_objects, found, seen);
            }
            MathExpr::Fraction { num, den } => {
                walk(num, locals, known_fns, existing_params, existing_r_objects, found, seen);
                walk(den, locals, known_fns, existing_params, existing_r_objects, found, seen);
            }
            MathExpr::Sqrt { index, radicand } => {
                if let Some(i) = index {
                    walk(i, locals, known_fns, existing_params, existing_r_objects, found, seen);
                }
                walk(radicand, locals, known_fns, existing_params, existing_r_objects, found, seen);
            }
            MathExpr::Call { name, args, .. } => {
                // Skip function-reference arguments in economics/calculus meta-operators
                let fn_arg_skip = match name.as_str() {
                    "tangent" | "elasticity" | "marginal" | "average" => 1,
                    "epsilon" if args.len() == 2 => 1,
                    "area" | "surplus" => 2,
                    _ => 0,
                };
                // If a 1-arg call uses an unknown symbol, only treat it as a free parameter
                // if it isn't a known/builtin function (e.g. `a(x+1)`).
                if args.len() == 1
                    && !is_builtin_function(name)
                    && !known_fns.contains(name)
                    && !existing_r_objects.contains(name)
                {
                    maybe_record(name);
                }
                for (i, a) in args.iter().enumerate() {
                    if i < fn_arg_skip && a.as_canonical_ident().is_some() {
                        continue;
                    }
                    walk(a, locals, known_fns, existing_params, existing_r_objects, found, seen);
                }
            }
            MathExpr::Derivative { var, body, .. } => {
                let inserted = locals.insert(var.clone());
                walk(body, locals, known_fns, existing_params, existing_r_objects, found, seen);
                if inserted {
                    locals.remove(var);
                }
            }
            MathExpr::Integral {
                var,
                lower,
                upper,
                body,
            }
            | MathExpr::Sum {
                var,
                lower,
                upper,
                body,
            }
            | MathExpr::Prod {
                var,
                lower,
                upper,
                body,
            } => {
                walk(lower, locals, known_fns, existing_params, existing_r_objects, found, seen);
                walk(upper, locals, known_fns, existing_params, existing_r_objects, found, seen);
                let inserted = locals.insert(var.clone());
                walk(body, locals, known_fns, existing_params, existing_r_objects, found, seen);
                if inserted {
                    locals.remove(var);
                }
            }
            MathExpr::ComparisonChain { first, rest } => {
                walk(first, locals, known_fns, existing_params, existing_r_objects, found, seen);
                for (_, r) in rest {
                    walk(r, locals, known_fns, existing_params, existing_r_objects, found, seen);
                }
            }
            MathExpr::Piecewise(branches) => {
                for b in branches {
                    if let Some(c) = &b.condition {
                        walk(c, locals, known_fns, existing_params, existing_r_objects, found, seen);
                    }
                    walk(&b.value, locals, known_fns, existing_params, existing_r_objects, found, seen);
                }
            }
            MathExpr::Restricted {
                inner,
                restrictions,
            } => {
                walk(inner, locals, known_fns, existing_params, existing_r_objects, found, seen);
                for r in restrictions {
                    walk(r, locals, known_fns, existing_params, existing_r_objects, found, seen);
                }
            }
            MathExpr::Tuple(items) => {
                for item in items {
                    walk(item, locals, known_fns, existing_params, existing_r_objects, found, seen);
                }
            }
            MathExpr::MemberAccess { .. } => {}
            MathExpr::Index { target, index } => {
                walk(target, locals, known_fns, existing_params, existing_r_objects, found, seen);
                walk(index, locals, known_fns, existing_params, existing_r_objects, found, seen);
            }
        }
    }

    walk(
        expr,
        &mut locals,
        known_fns,
        existing_params,
        existing_r_objects,
        &mut found,
        &mut seen,
    );
    found
}
