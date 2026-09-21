#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINED="${1:-$ROOT/retrained_recommendation/best}"

for mode in learned heuristic always-on; do
  "$ROOT/scripts/run_main_eval.sh" "$TRAINED" "$mode"
done
