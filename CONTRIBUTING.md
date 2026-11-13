# Contributing to Econ-Calc

Thank you for your interest in helping build **Econ-Calc**! Because our core promise is a **100% offline, native Rust app that runs well even on 2–4 GB RAM hardware** while scaling cleanly to workstation workflows, we follow strict engineering guardrails.

---

## Fair Warning: I, Artur, am alone in this project and with near zero coding skills, therefore the entire code until now is made with Gemini 4 Argon.

---

## Core Rules for All Contributions

1. **Protect Tier 1 (Universal Lite Core):**
   * Do **not** add heavy external, cloud, or GPU-mandatory dependencies to the default build.
   * Heavy analytics engines, persistent language kernels, LSP/DAP tools, and 3D GPU features must live behind **Cargo feature flags** or **out-of-process IPC**.
2. **Never Block the UI Thread:**
   * Math sampling, AST evaluation, file indexing, CSV parsing, and script execution must run on background threads/workers via bounded channels.
3. **Zero Idle Waste & Bounded Memory:**
   * UI rendering must be event-driven (no continuous 60 FPS spin when idle).
   * Never load large files or datasets monolithically into RAM—use streaming and virtualized paging.
4. **Cross-Platform & Defensive Hardware Safety:**
   * Code must compile and run on Windows and Linux across both `x86_64` and `aarch64`.
   * Never assume AVX2/modern SIMD without runtime detection and scalar fallbacks.
   * Handle missing/older GPU drivers gracefully.
5. **No Blind Panics:**
   * Avoid `.unwrap()` and `.expect()` in production paths unless an invariant is strictly guaranteed and documented in a comment.

---

## Before You Start a Large Change

* **Open an Issue first** before introducing new crates, changing public traits, or modifying workspace architecture.
* For bug fixes, docs, tests, or small improvements, feel free to open a Pull Request directly.

---

## Local Verification Checklist

Before submitting a Pull Request, please run:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

If your change touches UI rendering or data loading, please mention in your PR description:
* Your OS, CPU architecture (`x86_64` or `aarch64`), and GPU/driver environment.
* Any noticeable change in idle CPU usage, startup time, or RAM footprint.
