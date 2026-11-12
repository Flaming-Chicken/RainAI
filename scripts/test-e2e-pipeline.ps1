# RainAI Complete End-to-End Pipeline & CLI Verification Script
#
# Exercises:
# 1. Rust unit & integration test suites
# 2. End-to-end data contribution and reconciliation pipeline test
# 3. Cloudflare Edge Staging Worker API contract test
# 4. CLI subcommands (template, validate, quota, triage, reconcile)

$ErrorActionPreference = "Stop"
Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "   RainAI End-to-End Pipeline & Staging Verification   " -ForegroundColor Cyan
Write-Host "========================================================`n" -ForegroundColor Cyan

# Step 1: Run Rust E2E Pipeline Integration Test
Write-Host "[1/4] Running Rust E2E Pipeline Integration Test..." -ForegroundColor Yellow
cargo test -p utilities --test e2e_contribution_pipeline_tests -- --nocapture
if ($LASTEXITCODE -ne 0) {
    Write-Error "Rust E2E pipeline test failed."
    exit 1
}
Write-Host "  -> Rust E2E Pipeline Test Passed!`n" -ForegroundColor Green

# Step 2: Run Full Ingestion & Contribution Test Suites
Write-Host "[2/4] Running Contribution & Ingestion Integration Tests..." -ForegroundColor Yellow
cargo test -p utilities --test contribution_tests --test ingest_tests
if ($LASTEXITCODE -ne 0) {
    Write-Error "Contribution/Ingestion tests failed."
    exit 1
}
Write-Host "  -> Contribution & Ingestion Tests Passed!`n" -ForegroundColor Green

# Step 3: Run Cloudflare Edge Staging Worker E2E Contract Test
Write-Host "[3/4] Running Cloudflare Edge Staging Worker E2E Test..." -ForegroundColor Yellow
node crates/web/tests/worker_e2e_test.mjs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Worker E2E test failed."
    exit 1
}
Write-Host "  -> Cloudflare Edge Staging Worker Test Passed!`n" -ForegroundColor Green

# Step 4: Exercise CLI Subcommands
Write-Host "[4/4] Exercising rainai_contribute CLI Subcommands..." -ForegroundColor Yellow

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("rainai_cli_test_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
$templatePath = Join-Path $tempDir "manifest_template.json"

try {
    # 4a. Template generation
    Write-Host "  -> Testing 'template' subcommand..."
    cargo run -q -p utilities --bin rainai_contribute -- template --out $templatePath
    if (-not (Test-Path $templatePath)) {
        throw "Failed to generate template at $templatePath"
    }

    # 4b. Manifest validation
    Write-Host "  -> Testing 'validate' subcommand on template..."
    cargo run -q -p utilities --bin rainai_contribute -- validate $templatePath

    # 4c. Quota inspection
    Write-Host "  -> Testing 'quota' subcommand..."
    cargo run -q -p utilities --bin rainai_contribute -- quota

    # 4d. Triage inspection on empty/test quarantine
    Write-Host "  -> Testing 'triage' subcommand..."
    $testQuarantine = Join-Path $tempDir "quarantine"
    New-Item -ItemType Directory -Path $testQuarantine -Force | Out-Null
    cargo run -q -p utilities --bin rainai_contribute -- triage $testQuarantine

    # 4e. Reconcile on test target
    Write-Host "  -> Testing 'reconcile' subcommand..."
    $testDataset = Join-Path $tempDir "dataset"
    New-Item -ItemType Directory -Path $testDataset -Force | Out-Null
    cargo run -q -p utilities --bin rainai_contribute -- reconcile $testDataset

    # 4f. Binary Attribution Dictionary Export
    Write-Host "  -> Testing 'export-attributions' subcommand..."
    $testBin = Join-Path $tempDir "attributions.bin"
    cargo run -q -p utilities --bin rainai_contribute -- export-attributions --manifest Data/rain/manifest_provenance.json --out $testBin
    if (-not (Test-Path $testBin)) {
        throw "Failed to export binary attributions to $testBin"
    }

    Write-Host "`n========================================================" -ForegroundColor Green
    Write-Host "   ALL END-TO-END PIPELINE & STAGING TESTS PASSED!     " -ForegroundColor Green
    Write-Host "========================================================`n" -ForegroundColor Green
}
finally {
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
