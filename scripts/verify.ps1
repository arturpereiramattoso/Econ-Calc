<#
.SYNOPSIS
    Local verification & architecture hygiene gate for Oikonomos (Econ-Calc).
    Enforces CONTRIBUTING.md checks (fmt, clippy, test) plus static unwrap & monolith tracking.
#>
param(
    [switch]$SkipCargo
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "=== Oikonomos (Econ-Calc) Verification Gate ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"

# 1. Static Architecture & Guardrail Checks (always run, no Rust toolchain required)
Write-Host "`n[1/4] Static Guardrail Audit (src/app.rs size & production .unwrap()/.expect() count)" -ForegroundColor Yellow

$rsFiles = Get-ChildItem -Path (Join-Path $RepoRoot "src") -Recurse -Filter "*.rs"
$totalBytes = ($rsFiles | Measure-Object -Property Length -Sum).Sum
$appRs = Get-Item (Join-Path $RepoRoot "src\app.rs")
$appLines = (Get-Content -LiteralPath $appRs.FullName).Count
$appPct = [math]::Round(($appRs.Length * 100.0) / $totalBytes, 1)

Write-Host ("  src/app.rs: {0:N0} bytes ({1:N0} lines) — {2}% of src/ ({3:N0} bytes total)" -f $appRs.Length, $appLines, $appPct, $totalBytes)

$prodUnwraps = 0
$prodExpects = 0
$perFile = @()
foreach ($f in $rsFiles) {
    $lines = Get-Content -LiteralPath $f.FullName
    $inTest = $false
    $u = 0; $e = 0
    foreach ($line in $lines) {
        if ($line -match "#\[cfg\(test\)\]") { $inTest = $true }
        if (-not $inTest) {
            if ($line -match "\.unwrap\(") { $u++ }
            if ($line -match "\.expect\(") { $e++ }
        }
    }
    $prodUnwraps += $u
    $prodExpects += $e
    if (($u + $e) -gt 0) {
        $rel = $f.FullName.Substring($RepoRoot.Path.Length + 1)
        $perFile += [PSCustomObject]@{ File = $rel; Unwrap = $u; Expect = $e }
    }
}

Write-Host ("  Production (non-test) panic calls: .unwrap()={0}, .expect()={1}" -f $prodUnwraps, $prodExpects)
if ($perFile.Count -gt 0) {
    $perFile | Sort-Object -Property @{Expression={$_.Unwrap + $_.Expect}; Descending=$true} | Select-Object -First 8 | Format-Table -AutoSize
}

# 2. Cargo Checks (fmt, clippy, test)
if ($SkipCargo) {
    Write-Host "Skipping Cargo checks (-SkipCargo specified)." -ForegroundColor DarkYellow
    exit 0
}

$cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargoCmd) {
    Write-Host "`n[INFO] 'cargo' is not installed on PATH on this machine." -ForegroundColor DarkYellow
    Write-Host "       Static audit completed above; fmt/clippy/test will be enforced in GitHub Actions (.github/workflows/rust.yml)."
    exit 0
}

Push-Location -LiteralPath $RepoRoot
try {
    Write-Host "`n[2/4] Running cargo fmt --check..." -ForegroundColor Yellow
    cargo fmt --check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n[3/4] Running cargo clippy --all-targets -- -D warnings..." -ForegroundColor Yellow
    cargo clippy --all-targets -- -D warnings
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n[4/4] Running cargo test..." -ForegroundColor Yellow
    cargo test
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`nAll Oikonomos verification checks passed." -ForegroundColor Green
} finally {
    Pop-Location
}
