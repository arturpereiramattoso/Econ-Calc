use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(target_arch = "aarch64")]
pub const ARCH_SHORT_LABEL: &str = "ARM64 Snapdragon";
#[cfg(target_arch = "aarch64")]
pub const ARCH_FULL_LABEL: &str =
    "Windows 11 ARM64 (Qualcomm Snapdragon / aarch64-pc-windows-msvc)";

#[cfg(target_arch = "x86_64")]
pub const ARCH_SHORT_LABEL: &str = "x64 Intel/AMD";
#[cfg(target_arch = "x86_64")]
pub const ARCH_FULL_LABEL: &str = "Windows 10/11 x64 (Intel & AMD / x86_64-pc-windows-msvc)";

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const ARCH_SHORT_LABEL: &str = std::env::consts::ARCH;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const ARCH_FULL_LABEL: &str = std::env::consts::ARCH;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    QuartoQmd,
    RMarkdownRmd,
    MarkdownMd,
}

impl ReportFormat {
    pub fn extension(self) -> &'static str {
        match self {
            ReportFormat::QuartoQmd => "qmd",
            ReportFormat::RMarkdownRmd => "Rmd",
            ReportFormat::MarkdownMd => "md",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReportFormat::QuartoQmd => "Quarto (.qmd)",
            ReportFormat::RMarkdownRmd => "R Markdown (.Rmd)",
            ReportFormat::MarkdownMd => "Markdown (.md)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActiveModelKind {
    CustomExpression,
    CobbDouglas,
    ConstrainedUtility,
    GameTheory2x2,
}

impl ActiveModelKind {
    pub fn title(self) -> &'static str {
        match self {
            ActiveModelKind::CustomExpression => {
                "Custom Mathematical Expression & Numerical Calculus"
            }
            ActiveModelKind::CobbDouglas => "Cobb-Douglas Production & Isoquant Analysis",
            ActiveModelKind::ConstrainedUtility => "Lagrangian Constrained Utility Maximization",
            ActiveModelKind::GameTheory2x2 => "2x2 Normal-Form Game Theory Equilibrium Solver",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone)]
pub struct ProjectFileEntry {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub size_bytes: u64,
}

/// Lists files in a project directory safely without blocking on deep trees.
pub fn list_project_directory(dir: &Path) -> Vec<ProjectFileEntry> {
    let mut entries = Vec::new();
    let Ok(read_dir) = fs::read_dir(dir) else {
        return entries;
    };

    for item in read_dir.flatten() {
        let path = item.path();
        let name = item.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        let metadata = item.metadata().ok();
        let is_dir = metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false);
        let size_bytes = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
        entries.push(ProjectFileEntry {
            name,
            path,
            is_dir,
            size_bytes,
        });
    }

    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then_with(|| a.name.cmp(&b.name)));
    entries
}

/// Saves arbitrary text content (Markdown, Quarto, RMarkdown, R script, or CSV) to a path.
pub fn save_text_file(path: &Path, content: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {:?}: {}", parent, e))?;
        }
    }
    let mut file =
        File::create(path).map_err(|e| format!("Failed to create {:?}: {}", path, e))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("Failed to write {:?}: {}", path, e))
}

/// Reads a text file (Markdown, Quarto, RMarkdown, R script, or CSV) from disk.
pub fn load_text_file(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("Failed to read {:?}: {}", path, e))
}

/// Converts UNIX timestamp seconds into `(year, month, day, hour, minute, second)` in UTC.
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

    let rproj_path = export_dir.join("project.Rproj");
    write_rproj_file(&rproj_path)?;

    let csv_path = export_dir.join("simulation_data.csv");
    write_simulation_csv(&csv_path, &payload.grid_rows)?;

    let r_script_path = export_dir.join("model_analysis.R");
    write_r_analysis_script(&r_script_path, payload)?;

    let report_filename = format!("report.{}", payload.report_format.extension());
    let report_path = export_dir.join(report_filename);
    write_literate_report(&report_path, payload)?;

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
    save_text_file(path, content)
}

pub fn write_simulation_csv(path: &Path, rows: &[GridRow]) -> Result<(), String> {
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
# EconCalc Studio — Reproducible Analysis Script for Positron / RStudio
# Active Model: {model_title}
# ==============================================================================

suppressPackageStartupMessages({{
  library(tidyverse)
}})

sim_data <- read_csv("simulation_data.csv", show_col_types = FALSE)

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
    x = "x",
    y = "y"
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

    save_text_file(path, &script)
}

fn write_literate_report(path: &Path, payload: &ExportPayload) -> Result<(), String> {
    let yaml_header = match payload.report_format {
        ReportFormat::QuartoQmd => format!(
            "---\ntitle: \"{}\"\nsubtitle: \"Reproducible Computational Report\"\nformat:\n  html:\n    toc: true\n    code-fold: false\n    theme: cosmo\nexecute:\n  echo: true\n  warning: false\n---\n",
            payload.model_kind.title()
        ),
        ReportFormat::RMarkdownRmd => format!(
            "---\ntitle: \"{}\"\nsubtitle: \"Reproducible Computational Report\"\noutput:\n  html_document:\n    toc: true\n    theme: flatly\n---\n",
            payload.model_kind.title()
        ),
        ReportFormat::MarkdownMd => format!("# {}\n\n", payload.model_kind.title()),
    };

    let report_body = format!(
        r##"{yaml_header}
## 1. Specification & Analytical Notes

{latex_block}

## 2. Parameter Summary

| Parameter | Symbol | Value |
|:---|:---|---:|
| Total Factor Productivity | $A$ | `{tfp_a:.4}` |
| Elasticity 1 | $\alpha$ | `{alpha:.4}` |
| Elasticity 2 | $\beta$ | `{beta:.4}` |
| Price $x$ | $P_x$ | `{price_x:.4}` |
| Price $y$ | $P_y$ | `{price_y:.4}` |
| Income / Endowment | $I$ | `{income:.4}` |
| Target Output | $\bar{{Y}}$ | `{target_output:.4}` |
| Optimal $x^*$ | $x^*$ | `{x_star:.6}` |
| Optimal $y^*$ | $y^*$ | `{y_star:.6}` |
| Multiplier $\lambda^*$ | $\lambda^*$ | `{lambda_star:.6}` |
| Indirect Utility $V^*$ | $V^*$ | `{indirect_utility:.6}` |

### Notes & Data Summary

```text
{game_summary}
```

## 3. Reproducible R Script

```r
library(tidyverse)
sim_data <- read_csv("simulation_data.csv", show_col_types = FALSE)
head(sim_data, 10)
source("model_analysis.R")
```
"##,
        yaml_header = yaml_header,
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

    save_text_file(path, &report_body)
}

fn launch_workspace_viewer(export_dir: &Path) -> (String, String) {
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
    if cfg!(target_os = "windows") {
        if let Ok(child) = Command::new("cmd")
            .args(["/C", "positron"])
            .arg(export_dir)
            .spawn()
        {
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
    if let Ok(child) = Command::new("explorer").arg(export_dir).spawn() {
        return (
            "explorer.exe".to_string(),
            format!(
                "positron not found on PATH; opened folder in Explorer (PID {}) at {:?}",
                child.id(),
                export_dir
            ),
        );
    }
    (
        "filesystem-only".to_string(),
        format!("Exported bundle to {:?}", export_dir),
    )
}
