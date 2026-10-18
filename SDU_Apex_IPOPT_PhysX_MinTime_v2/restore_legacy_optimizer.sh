#!/usr/bin/env bash
# Restores the actual IPOPT source (without replacing it with a random search).
set -euo pipefail
REPO="${1:-..}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/run_ipopt.py" --repo-root "$REPO" --restore-only
