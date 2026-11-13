<!-- README.md -->
# [Project Name]

> **Status: Early Development / Pre-Alpha** — Architecture and core foundations are actively being designed and built. Nothing is production-ready yet, and **contributors are warmly welcome**.

A **100% offline, native, cross-platform desktop workspace** written in Rust that unifies three workflows into one lightweight application:

1. **Interactive Math & Graphing (Desmos-inspired):** Real-time 2D/3D function plotting, reactive parameter sliders, and live expression evaluation.
2. **Data-Science Workbench (RStudio-inspired):** Interactive REPL console, environment/variable inspector, virtualized table/dataframe viewer, and integrated plot pane.
3. **Modern Polyglot IDE (Positron / VS Code-inspired):** Modular multi-pane workspace, keyboard-first navigation, command palette, syntax-highlighted editor, and optional LSP/DAP code intelligence.

---

## Why This Project Exists

Modern data-science and math tools force an uncomfortable trade-off:

* **Web & Electron IDEs** (VS Code, Positron, JupyterLab) are powerful, but consume gigabytes of RAM, lag on older CPUs/iGPUs, and do little until you install and configure external runtimes.
* **Lightweight math tools** (Desmos, GeoGebra, engineering scratchpads) offer instant visual intuition, but are either browser-bound or disconnected from real datasets and programming workflows.
* **Classic native IDEs** (RStudio, Spyder) are tightly bound to a single primary language ecosystem.

**[Project Name]** exists to answer a simple question: *Can a single native Rust codebase run instantaneously on a 2–4 GB RAM laptop with zero mandatory dependencies, while scaling cleanly into a full workstation IDE on powerful machines?*

---

## Core Strategy: One Codebase, Two Tiers

Instead of maintaining two separate applications—or shipping a bloated binary by default—all subsystems are separated behind **strict Rust traits, bounded message channels, and Cargo feature flags**:

### Tier 1 — Universal "Lite" Core *(Default Build)*
Built for older, budget, and constrained hardware (2–4 GB RAM, older Intel/AMD `x86_64`, `ARM64`/Snapdragon laptops, HDD/eMMC storage, older integrated GPUs):
* **Zero mandatory external dependencies:** Double-click and start graphing, inspecting CSVs, and evaluating expressions offline immediately.
* **Tiny footprint & zero idle waste:** Strictly event-driven rendering—0% CPU/GPU usage when idle.
* **Streaming & virtualization everywhere:** Large datasets and files are paged from disk and rendered only for visible rows/lines—never loaded monolithically into RAM.
* **Defensive graphics initialization:** Handles older drivers gracefully with fallback/software rendering paths.
* **Fast, low-RAM compilation:** Heavy dependencies stay out of the default build so contributors can even compile the app on a 4 GB RAM machine.

### Tier 2 — "Heavyweight Workstation" Edition *(Opt-In Features & Sidecars)*
Unlocked via **Cargo feature flags** and **lazily spawned out-of-process workers** on capable hardware:
* High-throughput analytics engines (e.g., Polars, DuckDB, Arrow-backed data providers).
* Persistent out-of-process kernels (Python, R, Julia) over async IPC.
* Full LSP (autocomplete, diagnostics, go-to-definition) and DAP (step debugging).
* Hardware-accelerated 3D plotting and dense point-cloud/surface shaders.

---

## Non-Negotiable Engineering Principles

To keep the codebase fast, portable, and maintainable as it grows, we hold ourselves to these rules:

1. **UI Never Couples to Heavy Backends:** All data engines, plotters, kernels, and language servers sit behind clean traits or IPC protocols.
2. **UI Thread Never Blocks:** Math sampling, AST evaluation, file indexing, and script execution run on background workers communicating via bounded channels.
3. **Process Isolation for External Runtimes:** Crash-prone or heavy external tools (Python/R/Julia, LSPs, DAPs) run out-of-process and spawn only when requested.
4. **Cross-Platform Hardware Safety:** Full support for Windows and Linux across `x86_64` and `aarch64`, with scalar fallbacks (never assuming AVX2 or modern Vulkan/DX12 without runtime checks).
5. **Honest Scope Boundaries:**
   * We are **not** trying to clone 40,000 VS Code extensions.
   * We will **never** require cloud accounts, telemetry, or an internet connection.
   * We will **not** ship a "half-working everything" prototype; each milestone must be genuinely useful on its own.

---

## Planned Milestones

We are building using a **"Wedge & Expand"** roadmap so that every milestone delivers a usable, self-contained tool rather than waiting years for v1.0.

### Milestone 0 — Core Shell & Low-Spec Foundation
- [ ] Workspace crate layout with strict trait boundaries (`core`, `math`, `data`, `ui`, `ipc`).
- [ ] Event-driven GUI shell with zero idle CPU/GPU spin and tested software/fallback rendering.
- [ ] Multi-pane docking layout, command palette skeleton, and keyboard-first navigation.
- [ ] CI matrix checking `x86_64` and `aarch64` builds, `clippy`, and unit tests.

### Milestone 1 — The Wedge: *Offline Desmos + Instant Data Inspector*
*Goal: A standalone, zero-dependency portable binary anyone can use immediately.*
- [ ] Reactive math expression parser, AST evaluator, and dependency graph (parameter sliders trigger instant minimal recomputation).
- [ ] Interactive 2D graphing canvas (adaptive sampling, asymptote/discontinuity handling, zoom-invariant strokes, smooth pan/zoom).
- [ ] Virtualized, streaming CSV/delimited table viewer capable of browsing multi-GB files with minimal RAM.
- [ ] Lightweight built-in scratchpad/console for quick numeric expressions and plotting table columns directly onto the math canvas.

### Milestone 2 — The Data-Science Loop: *RStudio-Style Workflow*
*Goal: Daily reproducible data analysis with external runtimes.*
- [ ] Out-of-process kernel protocol (starting with Python and/or R via stdio/IPC, spawned lazily).
- [ ] Docked interactive REPL console synchronized with code execution.
- [ ] Environment & Variable Inspector (inspect scalar values, arrays, and paged dataframes without freezing the UI).
- [ ] Unified Plot Pane with plot history, zoom/pan, and offline export (SVG/PNG).

### Milestone 3 — Polyglot IDE & Tier-2 Workstation Scale
*Goal: Full Positron-class workflows and heavy local compute.*
- [ ] Syntax-highlighted code editor with virtualized text layout and Tree-sitter integration.
- [ ] Generic LSP client (diagnostics, hover docs, completions) and DAP client (breakpoints, stepping, variable inspection).
- [ ] Opt-in Tier-2 Cargo features: Polars/DuckDB query engine integration and hardware-accelerated 3D math/data shaders.
- [ ] Extensible plugin/kernel discovery architecture.

---

## High-Level Architecture

```text
+-----------------------------------------------------------------------+
|                       UI & Docking Shell (Event-Driven)               |
|  [Graph Canvas]   [Code Editor]   [REPL Console]   [Table / Env View] |
+----------------------------------+------------------------------------+
                                   |
                Traits & Bounded Async Channels (No Blocking)
                                   |
        +--------------------------+--------------------------+
        |                                                     |
        v                                                     v
+-------------------------------+             +-------------------------------+
|     Tier 1: Built-In Core     |             |   Tier 2: Feature-Gated / IPC |
| - Reactive Math AST & Grapher |             | - Polars / DuckDB Engine      |
| - Streaming CSV/Table Pager   |             | - 3D GPU Shaders              |
| - Lightweight Scratchpad      |             | - Out-of-Process Kernels      |
+-------------------------------+             |   (Python/R/Julia, LSP, DAP)  |
                                              +-------------------------------+
