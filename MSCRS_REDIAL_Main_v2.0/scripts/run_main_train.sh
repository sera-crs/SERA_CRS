#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/code/improved"
export MSCRS_ROOT="$ROOT"

test -d "$ROOT/model/DialoGPT-small"
test -d "$ROOT/model/roberta-base"
test -f "$ROOT/initial_baseline/model.pt"
"$ROOT/scripts/build_assets.sh"
"$ROOT/scripts/run_recommendation_train.sh"
"$ROOT/scripts/run_generation_train.sh"
