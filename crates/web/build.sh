#!/usr/bin/env bash
# ==============================================================================
# RainAI Web Crate Build Delegator
# For Cloudflare Pages builds configured with root directory "crates/web"
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[*] Delegating build from crates/web to repository root deploy pipeline..."
cd "$ROOT_DIR"
exec bash deploy.sh
