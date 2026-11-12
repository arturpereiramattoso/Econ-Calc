use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ReportFormat {
    QuartoQmd,
    RMarkdownRmd,
}

impl ReportFormat {
    pub fn extension(self) -> &'static str {
        match self {
            ReportFormat::QuartoQmd => "qmd",
            ReportFormat::RMarkdownRmd => "Rmd",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReportFormat::QuartoQmd => "Quarto (.qmd)",
            ReportFormat::RMarkdownRmd => "R Markdown (.Rmd)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ActiveModelKind {
    CustomExpression,
    CobbDouglas,
    ConstrainedUtility,
    GameTheory2x2,
}

impl ActiveModelKind {
    pub fn title(self) -> &'static str {
        match self {
            ActiveModelKind::CustomExpression => "Custom Mathematical Expression & Numerical Calculus",
            ActiveModelKind::CobbDouglas => "Cobb-Douglas Production & Isoquant Analysis",
            ActiveModelKind::ConstrainedUtility => "Lagrangian Constrained Utility Maximization",
            ActiveModelKind::GameTheory2x2 => "2x2 Normal-Form Game Theory Equilibrium Solver",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GridRow {
    pub index: usize,
    pub x: f64,
    pub primary_y: f64,
    pub secondary_y: f64,
    pub marginal_or_slope: f64,
}

#[derive(Debug, Clone)]
pub struct ExportPayload {
    pub model_kind: ActiveModelKind,
    pub report_format: ReportFormat,
    pub custom_expr: String,
    pub secondary_expr: String,
    pub tfp_a: f64,
    pub alpha: f64,
    pub beta: f64,
    pub price_x: f64,
    pub price_y: f64,
    pub income: f64,
    pub target_output: f64,
    pub x_star: f64,
    pub y_star: f64,
    pub lambda_star: f64,
    pub indirect_utility: f64,
    pub game_matrix: [[(f64, f64); 2]; 2],
    pub game_summary: String,
    pub latex_derivation: String,
    pub grid_rows: Vec<GridRow>,
}

#[derive(Debug, Clone)]
pub struct ExportOutcome {
    pub export_dir: PathBuf,
    pub launched_program: String,
    pub status_message: String,
}

/// Converts UNIX timestamp seconds into `(year, month, day, hour, minute, second)` in UTC
/// using Howard Hinnant's civil-from-days algorithm in pure Rust.
fn utc_timestamp_parts() -> (i32, u32, u32, u32, u32, u32) {
    let total_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(1_794_480_000);

    let secs_of_day = (total_secs % 86_400) as u32;
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;

    let z = (total_secs / 86_400) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = (y + if m <= 2 { 1 } else { 0 }) as i32;

    (year, m, d, hour, minute, second)
}

pub fn export_and_launch_positron(payload: &ExportPayload) -> Result<ExportOutcome, String> {
    let (year, month, day, hour, minute, second) = utc_timestamp_parts();
    let folder_name = format!(
        "econ_workspace_{:04}{:02}{:02}_{:02}{:02}{:02}",
        year, month, day, hour, minute, second
    );

    let base_dir = Path::new("exports");
    let export_dir = base_dir.join(&folder_name);
    fs::create_dir_all(&export_dir)
        .map_err(|e| format!("Failed to create export directory {:?}: {}", export_dir, e))?;

    // 1. Write project.Rproj
    let rproj_path = export_dir.join("project.Rproj");
    write_rproj_file(&rproj_path)?;

    // 2. Write simulation_data.csv via the `csv` crate
    let csv_path = export_dir.join("simulation_data.csv");
    write_simulation_csv(&csv_path, &payload.grid_rows)?;

    // 3. Write model_analysis.R
    let r_script_path = export_dir.join("model_analysis.R");
    write_r_analysis_script(&r_script_path, payload)?;

    // 4. Write report.qmd or report.Rmd
    let report_filename = format!("report.{}", payload.report_format.extension());
    let report_path = export_dir.join(report_filename);
    write_literate_report(&report_path, payload)?;

    // 5. Launch Positron IDE or fallback to Windows Explorer
    let (launched_program, status_message) = launch_workspace_viewer(&export_dir);

    Ok(ExportOutcome {
        export_dir,
        launched_program,
        status_message,
    })
}

fn write_rproj_file(path: &Path) -> Result<(), String> {
    let content = "\
Version: 1.0

RestoreWorkspace: Default
SaveWorkspace: Default
AlwaysSaveHistory: Default

EnableCodeIndexing: Yes
UseSpacesForTab: Yes
NumSpacesForTab: 2
Encoding: UTF-8

RnwWeave: knitr
LaTeX: pdfLaTeX
";
    let mut file = File::create(path)
        .map_err(|e| format!("Could not create {:?}: {}", path, e))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("Could not write {:?}: {}", path, e))
}

fn write_simulation_csv(path: &Path, rows: &[GridRow]) -> Result<(), String> {
    let mut wtr = csv::Writer::from_path(path)
        .map_err(|e| format!("Failed to initialize CSV writer at {:?}: {}", path, e))?;

    for row in rows {
        wtr.serialize(row)
            .map_err(|e| format!("Failed to serialize CSV row: {}", e))?;
    }

    wtr.flush()
        .map_err(|e| format!("Failed to flush CSV writer: {}", e))
}

fn write_r_analysis_script(path: &Path, payload: &ExportPayload) -> Result<(), String> {
    let script = format!(
        r##"# ==============================================================================
# econ-calc Reproducible Analysis Script for Posit Positron IDE
# Active Model: {model_title}
# Target Architecture: Windows 11 ARM64 (aarch64-pc-windows-msvc)
# ==============================================================================

suppressPackageStartupMessages({{
  library(tidyverse)
}})

# Load exported numerical simulation grid
sim_data <- read_csv("simulation_data.csv", show_col_types = FALSE)

# Active structural parameters exported from econ-calc
params <- tibble(
  A = {tfp_a:.6},
  alpha = {alpha:.6},
  beta = {beta:.6},
  price_x = {price_x:.6},
  price_y = {price_y:.6},
  income = {income:.6},
  target_output = {target_output:.6},
  x_star = {x_star:.6},
  y_star = {y_star:.6},
  lambda_star = {lambda_star:.6},
  indirect_utility = {indirect_utility:.6}
)

print(params)
glimpse(sim_data)

# Generate primary publication-quality ggplot2 figure
p_main <- ggplot(sim_data, aes(x = x)) +
  geom_line(aes(y = primary_y, color = "Primary Curve"), linewidth = 1.15) +
  geom_line(aes(y = secondary_y, color = "Secondary / Constraint"), linewidth = 0.95, linetype = "dashed") +
  geom_point(
    data = params,
    aes(x = x_star, y = y_star, color = "Solved Equilibrium (x*, y*)"),
    size = 3.6
  ) +
  scale_color_manual(
    name = "Series",
    values = c(
      "Primary Curve" = "#2563eb",
      "Secondary / Constraint" = "#dc2626",
      "Solved Equilibrium (x*, y*)" = "#16a34a"
    )
  ) +
  labs(
    title = "{model_title}",
    subtitle = sprintf(
      "Equilibrium: x* = %.4f, y* = %.4f | alpha = %.3f, beta = %.3f",
      params$x_star, params$y_star, params$alpha, params$beta
    ),
    x = "x (Input L / Good X)",
    y = "y (Input K / Good Y / f(x))",
    caption = "Generated natively by econ-calc (Windows 11 ARM64)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

print(p_main)
ggsave("equilibrium_plot.png", plot = p_main, width = 8.5, height = 5.5, dpi = 300)
"##,
        model_title = payload.model_kind.title(),
        tfp_a = payload.tfp_a,
        alpha = payload.alpha,
        beta = payload.beta,
        price_x = payload.price_x,
        price_y = payload.price_y,
        income = payload.income,
        target_output = payload.target_output,
        x_star = payload.x_star,
        y_star = payload.y_star,
        lambda_star = payload.lambda_star,
        indirect_utility = payload.indirect_utility,
    );

    let mut file = File::create(path)
        .map_err(|e| format!("Could not create R script {:?}: {}", path, e))?;
    file.write_all(script.as_bytes())
        .map_err(|e| format!("Could not write R script {:?}: {}", path, e))
}

fn write_literate_report(path: &Path, payload: &ExportPayload) -> Result<(), String> {
    let yaml_header = match payload.report_format {
        ReportFormat::QuartoQmd => format!(
            "---\ntitle: \"{}\"\nsubtitle: \"Reproducible Computational Economics Report (econ-calc ARM64)\"\nformat:\n  html:\n    toc: true\n    code-fold: false\n    theme: cosmo\nexecute:\n  echo: true\n  warning: false\n---\n",
            payload.model_kind.title()
        ),
        ReportFormat::RMarkdownRmd => format!(
            "---\ntitle: \"{}\"\nsubtitle: \"Reproducible Computational Economics Report (econ-calc ARM64)\"\noutput:\n  html_document:\n    toc: true\n    theme: flatly\n---\n",
            payload.model_kind.title()
        ),
    };

    let report_body = format!(
        r##"{yaml_header}
## 1. Structural Specification & Analytical Derivation

The active workspace in `econ-calc` evaluated the **{model_title}** template. Below is the exact symbolic and numerical specification exported from the IDE:

{latex_block}

## 2. Structural Parameter Environment

| Parameter | Symbol | Value |
|:---|:---|---:|
| Total Factor Productivity | $A$ | `{tfp_a:.4}` |
| Output / Utility Elasticity 1 | $\alpha$ | `{alpha:.4}` |
| Output / Utility Elasticity 2 | $\beta$ | `{beta:.4}` |
| Price of Good $x$ (or Wage $w$) | $P_x$ | `{price_x:.4}` |
| Price of Good $y$ (or Rental $r$) | $P_y$ | `{price_y:.4}` |
| Budget / Endowment Income | $I$ | `{income:.4}` |
| Target Isoquant Output | $\bar{{Y}}$ | `{target_output:.4}` |
| Optimal $x^*$ | $x^*$ | `{x_star:.6}` |
| Optimal $y^*$ | $y^*$ | `{y_star:.6}` |
| Shadow Price / Multiplier | $\lambda^*$ | `{lambda_star:.6}` |
| Indirect Utility / Objective | $V^*$ | `{indirect_utility:.6}` |

### 2x2 Normal-Form Game Summary (if active)

```text
{game_summary}
```

## 3. Reproducible R Analysis & Visualization

```r
library(tidyverse)

sim_data <- read_csv("simulation_data.csv", show_col_types = FALSE)
head(sim_data, 10)

source("model_analysis.R")
```
"##,
        yaml_header = yaml_header,
        model_title = payload.model_kind.title(),
        latex_block = payload.latex_derivation,
        tfp_a = payload.tfp_a,
        alpha = payload.alpha,
        beta = payload.beta,
        price_x = payload.price_x,
        price_y = payload.price_y,
        income = payload.income,
        target_output = payload.target_output,
        x_star = payload.x_star,
        y_star = payload.y_star,
        lambda_star = payload.lambda_star,
        indirect_utility = payload.indirect_utility,
        game_summary = payload.game_summary,
    );

    let mut file = File::create(path)
        .map_err(|e| format!("Could not create report {:?}: {}", path, e))?;
    file.write_all(report_body.as_bytes())
        .map_err(|e| format!("Could not write report {:?}: {}", path, e))
}
fn launch_workspace_viewer(export_dir: &Path) -> (String, String) {
    // Primary attempt: launch Posit's Positron IDE directly on the workspace directory
    if let Ok(child) = Command::new("positron").arg(export_dir).spawn() {
        return (
            "positron".to_string(),
            format!(
                "Launched Positron IDE (PID {}) with workspace {:?}",
                child.id(),
                export_dir
            ),
        );
    }
    // Windows 11 ARM64 cmd/shim fallback (positron.cmd)
    if cfg!(target_os = "windows") {
        if let Ok(child) = Command::new("cmd").args(["/C", "positron"]).arg(export_dir).spawn() {
            return (
                "positron.cmd".to_string(),
                format!(
                    "Launched Positron CLI shim (PID {}) on {:?}",
                    child.id(),
                    export_dir
                ),
            );
        }
    }
    // Fallback to Windows Explorer on Windows
    if let Ok(child) = Command::new("explorer").arg(export_dir).spawn() {
        return (
            "explorer.exe".to_string(),
            format!(
                "positron not found on PATH; opened folder in Windows Explorer (PID {}) at {:?}",
                child.id(),
                export_dir
            ),
        );
    }
    (
        "filesystem-only".to_string(),
        format!(
            "Exported reproducible bundle to {:?} (positron and explorer unavailable in current environment)",
            export_dir
        ),
    )
}
