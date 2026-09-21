#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/code/improved"
OUT="$ROOT/reproduced"
TRAINED="${1:-$ROOT/retrained_recommendation/best}"
GATE_MODE="${2:-learned}"
GATE_ARGS=()
case "$GATE_MODE" in
  learned) ;;
  heuristic)
    GATE_ARGS+=(--hard_gate_inference)
    OUT="$OUT/gate_heuristic"
    ;;
  always-on)
    GATE_ARGS+=(--always_on_gate)
    OUT="$OUT/gate_always_on"
    ;;
  *)
    echo "gate mode must be learned, heuristic, or always-on" >&2
    exit 2
    ;;
esac
export MSCRS_ROOT="$ROOT"

test -d "$ROOT/model/DialoGPT-small"
test -d "$ROOT/model/roberta-base"
test -f "$TRAINED/model.pt"
test -f "$TRAINED/enhanced_model.pt"
test -f "$CODE/assets/asset_summary.json"
mkdir -p "$OUT"

cd "$CODE"
python train_rec_sera.py \
  --dataset redial --seed 22 \
  --tokenizer "$ROOT/model/DialoGPT-small" \
  --text_tokenizer "$ROOT/model/roberta-base" \
  --model "$ROOT/model/DialoGPT-small" \
  --text_encoder "$ROOT/model/roberta-base" \
  --prompt_encoder "$TRAINED" \
  --enhanced --enhanced_checkpoint "$TRAINED/enhanced_model.pt" \
  --paper_multimodal_fusion --multimodal_lambda 0.5 \
  --item_tag_file "$CODE/assets/item_tags_r8.pt" \
  --scene_memory_file "$CODE/assets/scene_memory_r8.pt" \
  --scene_top_k 3 --scene_tag_threshold 0.5 \
  --scene_semantic_weight 1.0 --scene_positive_weight 1.0 --scene_negative_weight 1.0 \
  --scene_temperature 0.1 --scene_item_degree_power 1.0 --scene_edge_degree_power 1.0 \
  --hard_alpha 0.5 --scene_beta 1.5 --negative_preference_penalty 1.0 \
  --coverage_top_k 10 --preference_threshold 0.0 --coverage_smoothness 0.1 \
  --gate_threshold 0.5 \
  "${GATE_ARGS[@]}" \
  --generation_loss_weight 0.0 \
  --per_device_eval_batch_size 128 --num_workers 0 \
  --eval_calibration --calibration_split test \
  --calibration_alpha_scales 1.0 --calibration_beta_scales 1.0 \
  --calibration_output "$OUT/main_test_metrics.json" \
  --output_dir "$OUT/runtime" 2>&1 | tee "$OUT/main_test.log"
