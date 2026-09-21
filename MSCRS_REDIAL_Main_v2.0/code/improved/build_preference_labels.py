import argparse
import json
import os
import re
from collections import Counter, defaultdict


MOVIE_PATTERN = re.compile(r"@(\d+)")
POSITIVE_PATTERN = re.compile(
    r"\b(?:i|we)\s+(?:(?:really|also|actually|absolutely|definitely)\s+)?"
    r"(?:love(?:d)?|like(?:d)?|enjoy(?:ed)?)\b|"
    r"\bmy\s+favo(?:u)?rite\b|"
    r"\b(?:it|that|this|they|those|both|one|movie|movies)\s+"
    r"(?:(?:is|are|was|were|seems?|sounds?)\s+)?(?:really\s+)?"
    r"(?:great|good|amazing|awesome|funny|nice|classic|best)\b",
    re.IGNORECASE,
)
NEGATIVE_PATTERN = re.compile(
    r"\b(?:i|we)\s+(?:(?:really|also|actually|absolutely)\s+)?"
    r"(?:hate(?:d)?|dislike(?:d)?|didn['’]?t\s+like|don['’]?t\s+like|"
    r"did\s+not\s+like|do\s+not\s+like)\b|"
    r"\b(?:it|that|this|they|those|both|one|movie|movies)\s+"
    r"(?:(?:is|are|was|were)\s+)?(?:awful|terrible|boring|bad|the\s+worst)\b",
    re.IGNORECASE,
)
NEUTRAL_PATTERN = re.compile(
    r"\b(?:i|we)\s+(?:haven['’]?t\s+(?:seen|watched)|have\s+not\s+"
    r"(?:seen|watched)|never\s+(?:seen|watched)|didn['’]?t\s+(?:see|watch))\b",
    re.IGNORECASE,
)


def _project_root():
    return os.environ.get(
        "MSCRS_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )


def _rows(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=os.path.join(_project_root(), "raw_data", "redial"))
    parser.add_argument("--data_dir", default=os.path.join(_project_root(), "rec_data", "redial"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--splits", default="train,valid,test")
    return parser.parse_args()


def _cue(text):
    if NEGATIVE_PATTERN.search(text):
        return "negative"
    if POSITIVE_PATTERN.search(text):
        return "positive"
    if NEUTRAL_PATTERN.search(text):
        return "neutral"
    return None


def main():
    args = parse_args()
    if args.output is None:
        args.output = os.path.join(args.data_dir, "conversation_preferences.json")
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    if not splits or any(value not in {"train", "valid", "test"} for value in splits):
        raise ValueError("--splits must contain train, valid and/or test")

    raw_dialogues = {}
    for split in splits:
        for dialogue in _rows(os.path.join(args.raw_dir, f"{split}.jsonl")):
            conversation_id = int(dialogue["conversationId"])
            raw_dialogues[(split, conversation_id)] = dialogue

    targets_by_turn = defaultdict(list)
    context_movies_by_turn = {}
    with open(os.path.join(args.data_dir, "item_ids.json"), encoding="utf-8") as stream:
        item_ids = set(json.load(stream))
    for split in splits:
        for row in _rows(os.path.join(args.data_dir, f"{split}_data.jsonl")):
            conversation, turn = map(int, str(row["identity"]).split("/", 1))
            targets_by_turn[(split, conversation, turn)].append(int(row["items"]))
            context_movies_by_turn[(split, conversation, turn)] = [
                int(entity)
                for entity in row.get("context_entities", [])
                if int(entity) in item_ids
            ]

    movie_mapping = defaultdict(dict)
    training_votes = defaultdict(lambda: defaultdict(int))
    exact_turns = partial_turns = 0
    for (split, conversation, turn), targets in targets_by_turn.items():
        dialogue = raw_dialogues.get((split, conversation))
        if dialogue is None or turn >= len(dialogue.get("messages", [])):
            continue
        raw_movies = list(
            dict.fromkeys(MOVIE_PATTERN.findall(str(dialogue["messages"][turn].get("text", ""))))
        )
        if len(raw_movies) == len(targets):
            exact_turns += 1
        elif raw_movies and targets:
            partial_turns += 1
        for raw_movie, target in zip(raw_movies, targets):
            movie_mapping[(split, conversation)][raw_movie] = target
            if split == "train":
                training_votes[raw_movie][target] += 1

    for (split, conversation, turn), context_movies in context_movies_by_turn.items():
        dialogue = raw_dialogues.get((split, conversation))
        if dialogue is None:
            continue
        raw_context = []
        for message in dialogue.get("messages", [])[:turn]:
            raw_context.extend(MOVIE_PATTERN.findall(str(message.get("text", ""))))
        raw_context = list(dict.fromkeys(raw_context))
        context_movies = list(dict.fromkeys(context_movies))
        if len(raw_context) == len(context_movies):
            for raw_movie, global_movie in zip(raw_context, context_movies):
                movie_mapping[(split, conversation)].setdefault(raw_movie, global_movie)

    training_mapping = {}
    for raw_movie, votes in training_votes.items():
        ordered = sorted(votes.items(), key=lambda pair: pair[1], reverse=True)
        best_movie, best_count = ordered[0]
        if best_count / sum(votes.values()) >= 0.9:
            training_mapping[raw_movie] = best_movie

    events = {}
    statistics = {split: Counter() for split in splits}
    for (split, conversation), dialogue in raw_dialogues.items():
        seeker_id = dialogue.get("initiatorWorkerId")
        pending_movies = []
        per_movie_events = {}
        for message_index, message in enumerate(dialogue.get("messages", [])):
            for raw_movie in MOVIE_PATTERN.findall(str(message.get("text", ""))):
                if raw_movie not in pending_movies:
                    pending_movies.append(raw_movie)
            if message.get("senderWorkerId") != seeker_id:
                continue
            text = str(message.get("text", ""))
            label = _cue(text)
            if label is not None:
                for raw_movie in pending_movies:
                    global_movie = movie_mapping[(split, conversation)].get(raw_movie)
                    if global_movie is None:
                        global_movie = training_mapping.get(raw_movie)
                    if global_movie is None:
                        statistics[split]["unmapped"] += 1
                        continue
                    if str(global_movie) in per_movie_events:
                        continue
                    per_movie_events[str(global_movie)] = {
                        "label": label,
                        "available_from_context_length": message_index + 1,
                        "feedback": text,
                        "raw_movie_id": raw_movie,
                    }
                    statistics[split][label] += 1
            pending_movies = []
        events[f"{split}:{conversation}"] = per_movie_events
        statistics[split]["conversations"] += 1

    preferences = {}
    for key, per_movie_events in events.items():
        buckets = {"positive": [], "negative": [], "neutral": []}
        for movie_id, event in per_movie_events.items():
            buckets[event["label"]].append(int(movie_id))
        preferences[key] = {
            label: sorted(movie_ids) for label, movie_ids in buckets.items()
        }

    payload = {
        "preferences": preferences,
        "events": events,
        "method": {
            "source": "explicit polarity cues in visible seeker text",
            "questionnaire_used": False,
            "mapping_fallback": "training-split raw-to-processed movie mapping only",
            "availability": "message index + 1",
            "training_supervision_split": "train",
        },
        "statistics": {
            "exact_aligned_turns": exact_turns,
            "partial_aligned_turns": partial_turns,
            "splits": {key: dict(value) for key, value in statistics.items()},
        },
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(json.dumps(payload["statistics"], indent=2))


if __name__ == "__main__":
    main()
