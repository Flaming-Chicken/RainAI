#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Builds RainAI for iOS architectures and bundles .xcframework.
.DESCRIPTION
    Compiles crates/app for physical iOS devices (aarch64-apple-ios)
    and iOS simulators (aarch64-apple-ios-sim, x86_64-apple-ios).
    Combines simulator slices with lipo and packages into an XCFramework.
.PARAMETER OutputDir
    Destination directory for compiled iOS binaries and XCFramework. Defaults to "target/ios".
.EXAMPLE
    ./scripts/build-ios.ps1
#>
param(
    [string]$OutputDir = "target/ios"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host " Building RainAI for iOS (Device & Simulator)"
Write-Host "============================================================"

$targets = @("aarch64-apple-ios", "aarch64-apple-ios-sim", "x86_64-apple-ios")

# Ensure required rustup targets are installed
$installedTargets = rustup target list --installed
foreach ($target in $targets) {
    if ($installedTargets -notcontains $target) {
        Write-Host "[*] Installing missing target: $target..."
        rustup target add $target
    }
}

New-Item -ItemType Directory -Force -Path "$OutputDir/device" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir/simulator" | Out-Null

foreach ($target in $targets) {
    Write-Host "[*] Compiling crates/app for $target..."
    cargo build --package app --release --target $target
}

# Copy outputs
Copy-Item -Force "target/aarch64-apple-ios/release/libapp.a" "$OutputDir/device/libRainAI.a" -ErrorAction SilentlyContinue

# Assemble universal simulator binary if lipo exists
if (Get-Command lipo -ErrorAction SilentlyContinue) {
    Write-Host "[*] Assembling universal simulator binary with lipo..."
    lipo -create `
        "target/aarch64-apple-ios-sim/release/libapp.a" `
        "target/x86_64-apple-ios/release/libapp.a" `
        -output "$OutputDir/simulator/libRainAI.a"
} else {
    Write-Host "[i] lipo not available on this host. Preserving individual simulator slices."
    Copy-Item -Force "target/aarch64-apple-ios-sim/release/libapp.a" "$OutputDir/simulator/libRainAI_arm64.a" -ErrorAction SilentlyContinue
    Copy-Item -Force "target/x86_64-apple-ios/release/libapp.a" "$OutputDir/simulator/libRainAI_x86_64.a" -ErrorAction SilentlyContinue
}

# Package XCFramework if xcodebuild exists
if (Get-Command xcodebuild -ErrorAction SilentlyContinue) {
    Write-Host "[*] Generating RainAI.xcframework via xcodebuild..."
    $xcframeworkPath = "$OutputDir/RainAI.xcframework"
    if (Test-Path $xcframeworkPath) {
        Remove-Item -Recurse -Force $xcframeworkPath
    }
    xcodebuild -create-xcframework `
        -library "$OutputDir/device/libRainAI.a" `
        -library "$OutputDir/simulator/libRainAI.a" `
        -output $xcframeworkPath
    Write-Host "[+] Generated $xcframeworkPath"
} else {
    Write-Host "[i] xcodebuild not available on this host. Static libraries ready for Xcode import."
}

Write-Host "[+] iOS build completed successfully! Output: $OutputDir"
