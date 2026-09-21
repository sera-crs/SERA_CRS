#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/code/improved"
TRAINED="${1:-$ROOT/retrained_generation/best}"
OUT="$ROOT/reproduced/generation"
export MSCRS_ROOT="$ROOT"

test -d "$TRAINED/generator"
test -f "$TRAINED/model.pt"
test -f "$TRAINED/enhanced_model.pt"
mkdir -p "$OUT"

cd "$CODE"
python generate_sera.py \
  --dataset redial --paper_multimodal_fusion --multimodal_lambda 0.5 \
  --tokenizer "$ROOT/model/DialoGPT-small" \
  --text_tokenizer "$ROOT/model/roberta-base" \
  --model "$TRAINED/generator" --text_encoder "$ROOT/model/roberta-base" \
  --prompt_encoder "$TRAINED" --enhanced_checkpoint "$TRAINED/enhanced_model.pt" \
  --item_tag_file "$CODE/assets/item_tags_r8.pt" \
  --scene_memory_file "$CODE/assets/scene_memory_r8.pt" \
  --alpha 0.5 --beta 1.5 --scene_top_k 3 --batch_size 8 \
  --output "$OUT/predictions.jsonl"
python evaluate_generation.py "$OUT/predictions.jsonl" \
  --output "$OUT/metrics.json"
