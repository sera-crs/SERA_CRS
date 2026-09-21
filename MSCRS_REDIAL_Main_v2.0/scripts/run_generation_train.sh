#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/code/improved"
REC="${1:-$ROOT/retrained_recommendation/best}"
OUT="$ROOT/retrained_generation"
export MSCRS_ROOT="$ROOT"

test -f "$REC/model.pt"
test -f "$REC/enhanced_model.pt"
mkdir -p "$OUT"

cd "$CODE"
python train_rec_sera.py \
  --dataset redial --seed 22 \
  --tokenizer "$ROOT/model/DialoGPT-small" \
  --text_tokenizer "$ROOT/model/roberta-base" \
  --model "$ROOT/model/DialoGPT-small" \
  --text_encoder "$ROOT/model/roberta-base" \
  --prompt_encoder "$REC" \
  --enhanced --enhanced_checkpoint "$REC/enhanced_model.pt" \
  --paper_multimodal_fusion --multimodal_lambda 0.5 \
  --item_tag_file "$CODE/assets/item_tags_r8.pt" \
  --scene_memory_file "$CODE/assets/scene_memory_r8.pt" \
  --scene_top_k 3 --scene_tag_threshold 0.5 \
  --scene_semantic_weight 1.0 --scene_positive_weight 1.0 --scene_negative_weight 1.0 \
  --scene_temperature 0.1 --scene_item_degree_power 1.0 --scene_edge_degree_power 1.0 \
  --hard_alpha 0.5 --scene_beta 1.5 --negative_preference_penalty 1.0 \
  --coverage_top_k 10 --preference_threshold 0.0 --coverage_smoothness 0.1 \
  --gate_threshold 0.5 --gate_margin 0.0 \
  --preference_loss_weight 1.0 --negative_preference_loss_weight 1.0 \
  --evidence_loss_weight 1.0 --gate_loss_weight 1.0 \
  --generation_loss_weight 1.0 --cl_weight 0.0001 --selection_metric generation_loss \
  --learning_rate 0.0001 --weight_decay 0.01 \
  --num_train_epochs 10 --num_warmup_steps 200 \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 8 \
  --num_workers 0 --output_dir "$OUT" 2>&1 | tee "$OUT/training.log"
