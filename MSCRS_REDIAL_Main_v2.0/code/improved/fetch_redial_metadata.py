import argparse
import hashlib
import json
import os
import time

import requests


def _project_root():
    return os.environ.get(
        "MSCRS_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(_project_root(), "rec_data", "redial"))
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "redial_item_categories.json"),
    )
    parser.add_argument("--endpoint", default="https://dbpedia.org/sparql")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def _request(endpoint, query, retries):
    for attempt in range(retries):
        try:
            response = requests.post(
                endpoint,
                data={"query": query, "format": "application/sparql-results+json"},
                headers={"User-Agent": "SERA-CRS-metadata-builder/2.0"},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.retries <= 0:
        raise ValueError("batch_size and retries must be positive")
    with open(os.path.join(args.data_dir, "item_ids.json"), encoding="utf-8") as stream:
        item_ids = [int(value) for value in json.load(stream)]
    with open(os.path.join(args.data_dir, "entity2id.json"), encoding="utf-8") as stream:
        entity2id = json.load(stream)
    id2entity = {
        int(identity): entity.strip("<>")
        for entity, identity in entity2id.items()
        if str(entity).startswith("<http")
    }
    categories = {str(item): [] for item in item_ids}
    uri_to_item = {id2entity[item]: item for item in item_ids if item in id2entity}
    uris = sorted(uri_to_item)
    for start in range(0, len(uris), args.batch_size):
        batch = uris[start : start + args.batch_size]
        values = " ".join(f"<{uri}>" for uri in batch)
        query = (
            "SELECT ?item ?category WHERE { VALUES ?item { "
            + values
            + " } OPTIONAL { ?item <http://purl.org/dc/terms/subject> ?category } }"
        )
        result = _request(args.endpoint, query, args.retries)
        for binding in result["results"]["bindings"]:
            item = uri_to_item[binding["item"]["value"]]
            category = binding.get("category", {}).get("value")
            if category is not None:
                categories[str(item)].append(category)
    categories = {
        key: sorted(set(values)) for key, values in sorted(categories.items(), key=lambda pair: int(pair[0]))
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(categories, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    summary = {
        "endpoint": args.endpoint,
        "items": len(item_ids),
        "covered_items": sum(bool(values) for values in categories.values()),
        "sha256": digest,
    }
    with open(args.output + ".summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
