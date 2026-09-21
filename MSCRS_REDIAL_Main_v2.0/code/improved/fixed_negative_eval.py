

import argparse
import hashlib
import json
import os
import random
from typing import Dict, Iterable, List

import torch


FORMAT_VERSION = 1


def _canonical_hash(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_expanded_targets(data_file: str) -> List[Dict]:

    examples = []
    with open(data_file, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            dialog = json.loads(line)
            if "context_tokens" in dialog:
                targets = [int(dialog["items"])]
                identity = str(dialog.get("identity", f"line-{line_number}"))
            else:
                if not dialog.get("rec") or not any(dialog.get("context", [])):
                    continue
                targets = [int(item) for item in dialog["rec"]]
                identity = f'{dialog["conv_id"]}/{len(dialog.get("context", []))}'
            for target in targets:
                examples.append({
                    "sample_id": len(examples),
                    "identity": identity,
                    "label": target,
                })
    return examples


def load_test_item_mentions(data_file: str, official_item_ids: Iterable[int]) -> List[int]:

    official_item_ids = [int(item) for item in official_item_ids]
    official_set = set(official_item_ids)
    mentioned = set()
    with open(data_file, encoding="utf-8") as stream:
        for line in stream:
            dialog = json.loads(line)
            if "context_tokens" in dialog:
                mentioned.add(int(dialog["items"]))
            else:
                mentioned.update(int(item) for item in dialog.get("rec", []))


            mentioned.update(int(item) for item in dialog.get("all_movies", []))
    unknown = mentioned - official_set
    if unknown:
        raise ValueError(f"test item mentions contain {len(unknown)} non-candidate ids")

    return [item for item in official_item_ids if item in mentioned]


def build_fixed_candidates(
    item_ids: Iterable[int],
    examples: List[Dict],
    dataset: str,
    split: str = "test",
    seed: int = 42,
    num_negatives: int = 99,
    candidate_pool: str = "official_item_ids",
    pool_item_ids: Iterable[int] = None,
) -> Dict:
    item_ids = [int(item) for item in item_ids]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item_ids contains duplicates")
    if pool_item_ids is None:
        pool_item_ids = item_ids
    pool_item_ids = [int(item) for item in pool_item_ids]
    if len(pool_item_ids) != len(set(pool_item_ids)):
        raise ValueError("pool_item_ids contains duplicates")
    if not set(pool_item_ids).issubset(set(item_ids)):
        raise ValueError("pool_item_ids must be a subset of item_ids")
    if len(pool_item_ids) <= num_negatives:
        raise ValueError(
            f"candidate pool has {len(pool_item_ids)} items; need at least {num_negatives + 1}"
        )

    item_set = set(item_ids)
    pool_set = set(pool_item_ids)
    pool_positions = {item: index for index, item in enumerate(pool_item_ids)}
    records = []
    for expected_id, example in enumerate(examples):
        sample_id = int(example["sample_id"])
        label = int(example["label"])
        if sample_id != expected_id:
            raise ValueError("sample_ids must be contiguous and ordered")
        if label not in item_set:
            raise ValueError(f"test label {label} is absent from item_ids")
        if label not in pool_set:
            raise ValueError(f"test label {label} is absent from the candidate pool")


        seed_material = (
            f'{seed}\0{sample_id}\0{example.get("identity", "")}\0{label}'
        ).encode("utf-8")
        local_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = random.Random(local_seed)
        label_position = pool_positions[label]
        sampled_positions = rng.sample(range(len(pool_item_ids) - 1), num_negatives)
        negatives = [
            pool_item_ids[position if position < label_position else position + 1]
            for position in sampled_positions
        ]
        records.append({
            "sample_id": sample_id,
            "identity": str(example.get("identity", "")),
            "label": label,
            "negatives": negatives,
        })

    target_signature = [
        [record["sample_id"], record["identity"], record["label"]]
        for record in records
    ]
    payload = {
        "format_version": FORMAT_VERSION,
        "protocol": "one_positive_plus_fixed_random_negatives",
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "num_negatives": int(num_negatives),
        "num_candidates": int(num_negatives + 1),
        "candidate_pool": candidate_pool,
        "num_pool_items": len(pool_item_ids),
        "num_examples": len(records),
        "item_ids_sha256": _canonical_hash(item_ids),
        "pool_item_ids_sha256": _canonical_hash(pool_item_ids),
        "targets_sha256": _canonical_hash(target_signature),
        "examples": records,
    }
    payload["artifact_sha256"] = _canonical_hash(payload)
    return payload


def save_fixed_candidates(payload: Dict, output_file: str, overwrite: bool = False):
    if os.path.exists(output_file) and not overwrite:
        with open(output_file, encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("artifact_sha256") != payload.get("artifact_sha256"):
            raise FileExistsError(
                f"{output_file} exists with different contents; pass --overwrite explicitly"
            )
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


class FixedCandidateEvaluator:


    def __init__(self, candidate_file: str, item_ids: Iterable[int], device=None):
        with open(candidate_file, encoding="utf-8") as stream:
            self.payload = json.load(stream)
        if self.payload.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported fixed-candidate format")
        declared_hash = self.payload.get("artifact_sha256")
        unsigned_payload = {
            key: value for key, value in self.payload.items()
            if key != "artifact_sha256"
        }
        if declared_hash != _canonical_hash(unsigned_payload):
            raise ValueError("fixed-candidate artifact hash is invalid")

        item_ids = [int(item) for item in item_ids]
        if self.payload.get("item_ids_sha256") != _canonical_hash(item_ids):
            raise ValueError("fixed candidates were built for a different item_ids ordering")
        records = self.payload.get("examples", [])
        if len(records) != int(self.payload.get("num_examples", -1)):
            raise ValueError("fixed-candidate artifact has an invalid example count")

        item_to_local = {item: index for index, item in enumerate(item_ids)}
        candidate_rows = []
        labels = []
        for expected_id, record in enumerate(records):
            if int(record["sample_id"]) != expected_id:
                raise ValueError("fixed-candidate sample_ids are not contiguous")
            label = int(record["label"])
            negatives = [int(item) for item in record["negatives"]]
            if len(negatives) != int(self.payload["num_negatives"]):
                raise ValueError(f"sample {expected_id} has the wrong negative count")
            if label in negatives or len(negatives) != len(set(negatives)):
                raise ValueError(f"sample {expected_id} has invalid negatives")
            try:
                candidate_rows.append(
                    [item_to_local[label]] + [item_to_local[item] for item in negatives]
                )
            except KeyError as error:
                raise ValueError(f"candidate item {error.args[0]} is not in item_ids") from error
            labels.append(label)

        self.candidate_local_ids = torch.tensor(candidate_rows, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.device = device or torch.device("cpu")
        self.reset_metric()

    @property
    def metadata(self) -> Dict:
        return {
            key: self.payload[key]
            for key in [
                "protocol", "dataset", "split", "seed", "num_negatives",
                "num_candidates", "candidate_pool", "num_pool_items",
                "num_examples", "item_ids_sha256", "targets_sha256",
                "artifact_sha256",
            ]
        }

    def reset_metric(self):
        self.metric = {
            "recall@1": 0.0,
            "recall@10": 0.0,
            "recall@50": 0.0,
            "mrr@10": 0.0,
            "mrr@50": 0.0,
            "ndcg@10": 0.0,
            "ndcg@50": 0.0,
            "rank_sum": 0.0,
            "count": 0.0,
        }

    def evaluate(self, logits: torch.Tensor, labels: torch.Tensor, sample_ids: torch.Tensor):
        sample_ids_cpu = sample_ids.detach().long().cpu()
        if sample_ids_cpu.numel() == 0:
            return
        if sample_ids_cpu.min() < 0 or sample_ids_cpu.max() >= len(self.labels):
            raise ValueError("batch contains an out-of-range sample_id")
        expected = self.labels[sample_ids_cpu].to(labels.device)
        if not torch.equal(expected.long(), labels.detach().long()):
            raise ValueError("test labels do not match the frozen candidate artifact")

        candidate_ids = self.candidate_local_ids[sample_ids_cpu].to(logits.device)
        sampled_scores = logits.gather(1, candidate_ids)


        ranks = 1 + (sampled_scores[:, 1:] > sampled_scores[:, :1]).sum(dim=1)
        ranks = ranks.detach().float().cpu()

        for k in (1, 10, 50):
            self.metric[f"recall@{k}"] += float((ranks <= k).sum())
        for k in (10, 50):
            hit = ranks <= k
            self.metric[f"mrr@{k}"] += float(torch.where(hit, 1.0 / ranks, 0.0).sum())
            self.metric[f"ndcg@{k}"] += float(
                torch.where(hit, 1.0 / torch.log2(ranks + 1.0), 0.0).sum()
            )
        self.metric["rank_sum"] += float(ranks.sum())
        self.metric["count"] += float(ranks.numel())

    def report(self) -> Dict[str, torch.Tensor]:
        return {
            key: torch.tensor([value], device=self.device)
            for key, value in self.metric.items()
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="redial")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default="/root/autodl-tmp/MSCRS/rec_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument(
        "--candidate-pool",
        choices=["official_item_ids", "test_item_mentions"],
        default="official_item_ids",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    preferred = os.path.join(dataset_dir, f"{args.split}_data.jsonl")
    alternate = os.path.join(dataset_dir, f"{args.split}_data_train.jsonl")
    data_file = preferred if os.path.isfile(preferred) else alternate
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"cannot find {preferred} or {alternate}")
    with open(os.path.join(dataset_dir, "item_ids.json"), encoding="utf-8") as stream:
        item_ids = json.load(stream)
    examples = load_expanded_targets(data_file)
    pool_item_ids = (
        item_ids if args.candidate_pool == "official_item_ids"
        else load_test_item_mentions(data_file, item_ids)
    )
    payload = build_fixed_candidates(
        item_ids=item_ids,
        examples=examples,
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        num_negatives=args.num_negatives,
        candidate_pool=args.candidate_pool,
        pool_item_ids=pool_item_ids,
    )
    save_fixed_candidates(payload, args.output, overwrite=args.overwrite)
    print(json.dumps({key: value for key, value in payload.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
