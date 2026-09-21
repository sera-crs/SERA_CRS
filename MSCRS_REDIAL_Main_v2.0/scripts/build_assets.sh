#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/code/improved"
SOURCE="$CODE/redial_item_categories.json"
SCHEMA="$CODE/redial_attribute_schema.json"
export MSCRS_ROOT="$ROOT"

python "$CODE/build_preference_labels.py"
if [[ ! -f "$SOURCE" ]]; then
  python "$CODE/fetch_redial_metadata.py" --output "$SOURCE"
fi
python "$CODE/build_assets.py" \
  --tag_backend metadata \
  --category_file "$SOURCE" \
  --attribute_schema "$SCHEMA" \
  --num_tags 8 \
  --output_dir "$CODE/assets"
