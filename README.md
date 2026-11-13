# Econ-Calc

> **Status: Early Development / Pre-Alpha** — Started as an offline calculator for Economics undergraduates in Rust, now evolving into a unified offline math, data-science, and IDE workspace. **Contributors and testers are warmly welcome.**

**Econ-Calc** is a **100% offline, native, cross-platform desktop workspace** written in Rust that brings together three workflows in one lightweight application:

1. **Interactive Math & Graphing (Desmos-inspired):** Real-time 2D/3D function plotting, reactive parameter sliders (ideal for comparative statics, supply/demand shifts, and calculus), and live expression evaluation.
2. **Data-Science Workbench (RStudio-inspired):** Interactive REPL console, environment/variable inspector, virtualized table/dataframe viewer, and integrated plot pane.
3. **Modern Polyglot IDE (Positron / VS Code-inspired):** Modular multi-pane workspace, keyboard-first navigation, command palette, syntax-highlighted editor, and optional LSP/DAP code intelligence.

---

## Why Econ-Calc Exists

Students, researchers, and engineers—especially in economics, statistics, and applied sciences—often have to juggle tools that force painful trade-offs:

* **Web & Electron IDEs** (VS Code, Positron, JupyterLab) are powerful, but consume gigabytes of RAM, struggle on older CPUs/iGPUs, and do little until external runtimes and packages are configured.
* **Lightweight math tools** (Desmos, GeoGebra, engineering scratchpads) provide instant visual intuition, but are browser-bound or disconnected from real datasets and programming workflows.
* **Classic native IDEs** (RStudio, Spyder) are tightly coupled to a single primary language ecosystem.

**Econ-Calc** exists to bridge that gap: *a single native Rust codebase that boots instantaneously on a 2–4 GB RAM laptop with zero mandatory dependencies, while scaling cleanly into a full workstation IDE on capable hardware.*

---

## Core Strategy: One Codebase, Two Tiers

Instead of maintaining separate applications—or shipping a bloated binary by default—subsystems are designed around **strict Rust traits, bounded message channels, and Cargo feature flags**:

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

To keep Econ-Calc fast, portable, and maintainable as it grows, we hold ourselves to these rules:

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
- [x] Initial Rust repository, Cargo manifest (`Cargo.toml`), source tree (`src/`), and GitHub Actions workflows (`.github/workflows/`).
- [ ] Refactor into a modular workspace crate layout with strict trait boundaries (`core`, `math`, `data`, `ui`, `ipc`).
- [ ] Event-driven GUI shell with zero idle CPU/GPU spin and tested software/fallback rendering.
- [ ] Multi-pane docking layout, command palette skeleton, and keyboard-first navigation.
- [ ] CI matrix checking `x86_64` and `aarch64` builds, `clippy`, and unit tests.

### Milestone 1 — The Wedge: *Offline Desmos + Instant Data Inspector*
*Goal: A standalone, zero-dependency portable binary anyone (including econ undergrads and analysts) can use immediately.*
- [ ] Reactive math expression parser, AST evaluator, and dependency graph (parameter sliders trigger instant minimal recomputation).
- [ ] Interactive 2D graphing canvas (adaptive sampling, asymptote/discontinuity handling, zoom-invariant strokes, smooth pan/zoom).
- [ ] Virtualized, streaming CSV/delimited table viewer capable of browsing multi-GB files with minimal RAM.
- [ ] Lightweight built-in scratchpad/console for quick numeric/econometric expressions and plotting table columns directly onto the math canvas.

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
```

---

## Building from Source

### Prerequisites
* [Rust & Cargo](https://rustup.rs/) (stable toolchain)

### Build & Run
```bash
# Clone the repository
git clone https://github.com/arturpereiramattoso/Econ-Calc.git
cd Econ-Calc

# Build and run the default Tier 1 build
cargo run

# Run the test suite
cargo test

# Run linter checks
cargo clippy --all-targets
```

> *(Future)* Optional Tier-2 workstation capabilities will live behind explicit Cargo feature flags so default builds remain lightweight and fast to compile.

---

## Looking for Help & Contributors!

Building a project at the intersection of graphing calculators, data science, and IDEs is ambitious—**no single person has every skill needed to do it well**. Whether you want to write core Rust, optimize math algorithms, design clean UI ergonomics, or test builds on an old laptop, your help can directly shape Econ-Calc.

### Areas Where Help Is Needed Most

* **Math & Graphing Engine:** Expression parsing, interval arithmetic, adaptive curve sampling, implicit equation plotting (`f(x, y) = 0`), economics/calculus helpers, and reactive parameter dependency graphs.
* **UI, Graphics & Low-Spec Portability:** Event-driven GUI architecture, crisp zoom-invariant 2D/3D rendering, custom docking panes, and testing fallback rendering on older Intel/AMD iGPUs or ARM64 devices.
* **Data & Virtualization:** Streaming disk-backed CSV/Parquet readers, chunked table paging, and virtualized grid scrolling with minimal heap allocations.
* **Language Tooling & IPC:** Designing clean trait boundaries and async stdio/socket protocols for Jupyter/Ark-style kernels, LSP, and DAP.
* **UX, Design & Documentation:** Keyboard-first shortcuts, accessible contrast/themes, clear tutorials, and reproducible benchmarks (RAM usage, cold-start time, idle CPU).

### How to Get Involved

1. **Check [Issues](https://github.com/arturpereiramattoso/Econ-Calc/issues) & [Pull Requests](https://github.com/arturpereiramattoso/Econ-Calc/pulls):** Look for tasks or open a new issue if you'd like to propose something.
2. **Open an Issue Before Big PRs:** Because keeping Tier 1 lean and decoupled is critical, please open an issue to discuss new dependencies or architectural changes before writing large PRs.
3. **Test on Modest Hardware:** Even if you don't write Rust yet, testing builds on 2–4 GB RAM machines, Linux distros, Windows laptops, or ARM64 devices and reporting memory/GPU behavior is hugely valuable.

### Contribution Standards
Before submitting a pull request, please ensure:
* `cargo test` passes.
* `cargo clippy --all-targets` reports no new warnings.
* Default (`Tier 1`) builds do not pull in heavy external or GPU-mandatory dependencies without a feature gate.
* Avoid `.unwrap()` / `.expect()` in production paths unless an invariant is strictly guaranteed and documented.

---

## License

Licensed under the [Apache License 2.0](LICENSE).
