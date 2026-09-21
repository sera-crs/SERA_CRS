#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/code/improved"
OUT="$ROOT/retrained_recommendation"
export MSCRS_ROOT="$ROOT"

test -d "$ROOT/model/DialoGPT-small"
test -d "$ROOT/model/roberta-base"
test -f "$ROOT/initial_baseline/model.pt"
test -f "$CODE/assets/item_tags_r8.pt"
test -f "$CODE/assets/scene_memory_r8.pt"
test -f "$CODE/assets/asset_summary.json"
mkdir -p "$OUT"

cd "$CODE"
python train_rec_sera.py \
  --dataset redial --seed 22 \
  --tokenizer "$ROOT/model/DialoGPT-small" \
  --text_tokenizer "$ROOT/model/roberta-base" \
  --model "$ROOT/model/DialoGPT-small" \
  --text_encoder "$ROOT/model/roberta-base" \
  --prompt_encoder "$ROOT/initial_baseline" \
  --enhanced --paper_multimodal_fusion --multimodal_lambda 0.5 \
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
  --generation_loss_weight 0.0 --cl_weight 0.0001 --selection_metric ndcg@50 \
  --learning_rate 0.0001 --weight_decay 0.01 \
  --num_train_epochs 5 --num_warmup_steps 200 \
  --per_device_train_batch_size 40 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 128 \
  --num_workers 0 --output_dir "$OUT" 2>&1 | tee "$OUT/training.log"
