import argparse
import hashlib
import json
import os
import re
import urllib.parse
from collections import defaultdict
from typing import Dict, Iterable, List

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = os.environ.get(
        "MSCRS_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    parser.add_argument("--data_dir", default=os.path.join(project_root, "rec_data", "redial"))
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "assets"))
    parser.add_argument("--text_encoder", default=os.path.join(project_root, "model", "roberta-base"))
    parser.add_argument("--num_tags", type=int, default=8)
    parser.add_argument("--tag_backend", choices=["metadata"], default="metadata")
    parser.add_argument(
        "--category_file",
        default=os.path.join(os.path.dirname(__file__), "redial_item_categories.json"),
    )
    parser.add_argument(
        "--attribute_schema",
        default=os.path.join(os.path.dirname(__file__), "redial_attribute_schema.json"),
    )
    parser.add_argument("--preference_file", default=None)
    parser.add_argument("--scene_batch_size", type=int, default=128)
    parser.add_argument("--scene_max_length", type=int, default=200)
    parser.add_argument("--episode_max_gap", type=int, default=4)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _load_rows(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _conversation_number(row: dict) -> int:
    if "conv_id" in row:
        return int(row["conv_id"])
    return int(str(row["identity"]).split("/", 1)[0])


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_tags(item_ids, category_file, schema_file, expected_tags):
    raw = _load_json(category_file)
    schema = _load_json(schema_file)
    tag_names = []
    tag_group_ids = []
    group_names = []
    group_is_exclusive = []
    patterns = []
    for group_id, group in enumerate(schema.get("groups", [])):
        group_names.append(str(group["name"]))
        group_is_exclusive.append(bool(group.get("exclusive", False)))
        for attribute in group.get("attributes", []):
            tag_names.append(str(attribute["name"]))
            tag_group_ids.append(group_id)
            patterns.append(
                tuple(re.compile(value, re.IGNORECASE) for value in attribute["patterns"])
            )
    if len(tag_names) != expected_tags:
        raise ValueError(
            f"attribute schema defines {len(tag_names)} tags, expected {expected_tags}"
        )
    tags = torch.zeros(len(item_ids), len(tag_names))
    source_coverage = 0
    for row, item in enumerate(item_ids):
        categories = raw.get(str(item), [])
        if categories:
            source_coverage += 1
        text = " ".join(
            urllib.parse.unquote(str(value)).rsplit("Category:", 1)[-1]
            for value in categories
        )
        for column, compiled in enumerate(patterns):
            if any(pattern.search(text) for pattern in compiled):
                tags[row, column] = 1
    metadata = {
        "tag_names": tag_names,
        "tag_group_ids": torch.tensor(tag_group_ids, dtype=torch.long),
        "group_names": group_names,
        "group_is_exclusive": torch.tensor(group_is_exclusive, dtype=torch.bool),
    }
    metrics = {
        "backend": "predefined_metadata",
        "schema_version": schema.get("schema_version"),
        "num_tags": len(tag_names),
        "tag_names": tag_names,
        "group_names": group_names,
        "tag_frequency": tags.sum(dim=0).to(torch.long).tolist(),
        "item_coverage": float(tags.any(dim=1).float().mean()),
        "metadata_coverage": source_coverage / max(1, len(item_ids)),
        "category_source_sha256": _sha256(category_file),
        "attribute_schema_sha256": _sha256(schema_file),
    }
    return tags, metrics, metadata


def _build_scenes(
    train_file: str,
    item_to_local: Dict[int, int],
    events: Dict[str, dict],
    episode_max_gap: int,
) -> List[dict]:
    rows_by_conversation = defaultdict(list)
    for row in _load_rows(train_file):
        conversation_id = _conversation_number(row)
        if "context_tokens" in row:
            visible = row.get("context_entities", [])
            identity = str(row.get("identity", f"{conversation_id}/0"))
        else:
            visible = row.get("entity", [])
            identity = f"{conversation_id}/{len(row.get('context', []))}"
        try:
            context_length = int(identity.split("/", 1)[1])
        except (IndexError, ValueError):
            continue
        rows_by_conversation[conversation_id].append(
            (context_length, identity, sorted(set(map(int, visible))))
        )

    scenes = []
    for event_key, conversation_events in events.items():
        if not event_key.startswith("train:"):
            continue
        if not isinstance(conversation_events, dict):
            continue
        conversation_id = int(event_key.split(":", 1)[1])
        positive_events = sorted(
            (
                int(event.get("available_from_context_length", 10 ** 9)),
                int(movie),
                str(event.get("feedback", "")).strip(),
            )
            for movie, event in conversation_events.items()
            if event.get("label") == "positive"
            and int(movie) in item_to_local
        )
        groups = []
        for event in positive_events:
            if not groups or event[0] - groups[-1][-1][0] > episode_max_gap:
                groups.append([event])
            else:
                groups[-1].append(event)
        rows = sorted(rows_by_conversation.get(conversation_id, []))
        for group in groups:
            movie_ids = sorted({event[1] for event in group})
            if len(movie_ids) < 2:
                continue
            episode_start = group[0][0]
            episode_end = group[-1][0]
            eligible = [row for row in rows if row[0] >= episode_end]
            if not eligible:
                continue
            _, identity, visible = eligible[0]
            movie_ids = sorted(set(movie_ids).intersection(visible))
            if len(movie_ids) < 2:
                continue
            feedback = []
            for _, _, text in group:
                if text and text not in feedback:
                    feedback.append(text)
            text = " ".join(feedback)
            if not text:
                continue
            scenes.append(
                {
                    "conversation_id": conversation_id,
                    "source_identity": identity,
                    "episode_start": episode_start,
                    "episode_end": episode_end,
                    "text": text,
                    "movie_ids": movie_ids,
                    "entity_ids": visible,
                }
            )
    return scenes


@torch.no_grad()
def _encode_scene_texts(
    scenes: List[dict],
    encoder_path: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    model = AutoModel.from_pretrained(encoder_path).to(device).eval()
    outputs = []
    for start in range(0, len(scenes), batch_size):
        texts = [scene["text"] for scene in scenes[start : start + batch_size]]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        outputs.append(F.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(outputs)


def _scene_assets(
    scenes: List[dict],
    scene_embeddings: torch.Tensor,
    item_ids: List[int],
    item_tags: torch.Tensor,
) -> dict:
    item_to_local = {item: index for index, item in enumerate(item_ids)}
    incidence_rows = []
    incidence_cols = []
    profiles = torch.zeros(len(scenes), item_tags.shape[1])
    conversation_ids = torch.tensor([scene["conversation_id"] for scene in scenes])
    for index, scene in enumerate(scenes):
        local_movies = [item_to_local[item] for item in scene["movie_ids"]]
        incidence_rows.extend([index] * len(local_movies))
        incidence_cols.extend(local_movies)

        profiles[index] = item_tags[local_movies].mean(dim=0)
    indices = torch.tensor([incidence_rows, incidence_cols], dtype=torch.long)
    incidence = torch.sparse_coo_tensor(
        indices, torch.ones(len(incidence_rows)), size=(len(scenes), len(item_ids))
    ).coalesce()
    return {
        "scene_embeddings": scene_embeddings,
        "scene_tag_profiles": profiles,
        "scene_movie_incidence": incidence,
        "scene_conversation_ids": conversation_ids,
        "scene_metadata": scenes,
    }


def main():
    args = parse_args()
    if args.episode_max_gap < 0:
        raise ValueError("--episode_max_gap must be non-negative")
    if args.num_tags <= 0:
        raise ValueError("--num_tags must be positive")
    if not os.path.isfile(args.category_file):
        raise FileNotFoundError(f"missing item metadata source: {args.category_file}")
    if not os.path.isfile(args.attribute_schema):
        raise FileNotFoundError(f"missing attribute schema: {args.attribute_schema}")
    os.makedirs(args.output_dir, exist_ok=True)
    item_ids = [int(item) for item in _load_json(os.path.join(args.data_dir, "item_ids.json"))]
    item_to_local = {item: index for index, item in enumerate(item_ids)}
    item_tags, metrics, metadata = _metadata_tags(
        item_ids, args.category_file, args.attribute_schema, args.num_tags
    )
    item_tag_path = os.path.join(args.output_dir, f"item_tags_r{args.num_tags}.pt")
    torch.save(
        {
            "item_ids": torch.tensor(item_ids),
            "item_tags": item_tags,
            "tag_group_ids": metadata["tag_group_ids"],
            "group_is_exclusive": metadata["group_is_exclusive"],
            "tag_names": metadata["tag_names"],
            "group_names": metadata["group_names"],
            "metrics": metrics,
        },
        item_tag_path,
    )

    preference_file = args.preference_file or os.path.join(
        args.data_dir, "conversation_preferences.json"
    )
    if not os.path.isfile(preference_file):
        raise FileNotFoundError(
            f"positive historical scenes require preference labels: {preference_file}"
        )
    preference_payload = _load_json(preference_file)
    events = preference_payload.get("events", {})
    if not isinstance(events, dict) or not events:
        raise ValueError("preference labels must contain turn-level events")
    train_file = os.path.join(args.data_dir, "train_data.jsonl")
    if not os.path.isfile(train_file):
        train_file = os.path.join(args.data_dir, "train_data_train.jsonl")
    scenes = _build_scenes(train_file, item_to_local, events, args.episode_max_gap)
    if not scenes:
        raise RuntimeError(
            "no positive multi-movie training scenes were built; inspect preference coverage"
        )
    embeddings = _encode_scene_texts(
        scenes,
        args.text_encoder,
        args.scene_batch_size,
        args.scene_max_length,
        args.device,
    )
    assets = _scene_assets(scenes, embeddings, item_ids, item_tags)
    assets["item_ids"] = torch.tensor(item_ids)
    assets["tag_group_ids"] = metadata["tag_group_ids"]
    assets["group_is_exclusive"] = metadata["group_is_exclusive"]
    assets["tag_names"] = metadata["tag_names"]
    assets["group_names"] = metadata["group_names"]
    assets["tag_metrics"] = metrics
    scene_path = os.path.join(args.output_dir, f"scene_memory_r{args.num_tags}.pt")
    torch.save(assets, scene_path)

    summary = {
        "schema_version": "sera-v2.0",
        "tag_backend": "predefined_metadata",
        "tag_count": args.num_tags,
        "tag_names": metadata["tag_names"],
        "group_names": metadata["group_names"],
        "num_items": len(item_ids),
        "num_scenes": len(scenes),
        "memory_split": "train",
        "scene_granularity": "local_contiguous_feedback_episode",
        "minimum_positive_items": 2,
        "preference_visibility": "visible seeker text without questionnaire labels",
        "attribute_source": os.path.basename(args.category_file),
        "attribute_source_sha256": metrics["category_source_sha256"],
        "attribute_schema": os.path.basename(args.attribute_schema),
        "attribute_schema_sha256": metrics["attribute_schema_sha256"],
        "item_tag_file": os.path.basename(item_tag_path),
        "item_tag_sha256": _sha256(item_tag_path),
        "scene_memory_file": os.path.basename(scene_path),
        "scene_memory_sha256": _sha256(scene_path),
        "preference_file_sha256": _sha256(preference_file),
        "text_encoder": os.path.basename(os.path.normpath(args.text_encoder)),
        "build_command": "scripts/build_assets.sh",
        "seed": args.seed,
        "same_dialogue_masking": "training_only",
        "selection_rule": "choose R on validation metrics only; never use test metrics",
    }
    with open(os.path.join(args.output_dir, "asset_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
