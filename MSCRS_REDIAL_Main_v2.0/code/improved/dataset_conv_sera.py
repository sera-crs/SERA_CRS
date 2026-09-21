

import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils import padded_tensor


def _project_root():
    return Path(os.environ.get("MSCRS_ROOT", Path(__file__).resolve().parents[2]))


class SERAConvDataset(Dataset):
    def __init__(
        self,
        dataset,
        split,
        tokenizer,
        prompt_tokenizer,
        context_max_length=200,
        response_max_length=128,
        prompt_max_length=200,
        entity_max_length=32,
        debug=False,
    ):
        self.dataset = dataset
        self.split = split
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.context_max_length = int(context_max_length)
        self.response_max_length = int(response_max_length)
        self.prompt_max_length = int(prompt_max_length) - 1
        self.entity_max_length = int(entity_max_length)
        data_dir = _project_root() / "rec_data" / dataset
        preference_path = data_dir / "conversation_preferences.json"
        payload = json.loads(preference_path.read_text(encoding="utf-8"))
        self.preference_events = payload.get("events", {})
        preferred = data_dir / f"{split}_data.jsonl"
        fallback = data_dir / f"{split}_data_train.jsonl"
        data_file = preferred if preferred.is_file() else fallback
        if not data_file.is_file():
            raise FileNotFoundError(f"missing generation source: {data_file}")
        self.data = []
        with data_file.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream):
                if debug and line_number >= 512:
                    break
                self._append(json.loads(line))

    @staticmethod
    def _conversation_uid(split, conversation_id):
        offsets = {"train": 0, "valid": 1_000_000_000, "test": 2_000_000_000}
        return offsets.get(split, 3_000_000_000) + int(conversation_id)

    def _preference(self, conversation_id, context_turn_count):
        events = self.preference_events.get(f"{self.split}:{conversation_id}")
        visible = {"positive": [], "negative": [], "neutral": []}
        if not isinstance(events, dict):
            return visible
        for movie_id, event in events.items():
            label = event.get("label")
            if (
                label in visible
                and int(event.get("available_from_context_length", 10 ** 9))
                <= context_turn_count
            ):
                visible[label].append(int(movie_id))
        return visible

    @staticmethod
    def _dialogue_text(turns):
        pieces = []
        for index, utterance in enumerate(turns):
            if utterance:
                role = "User" if index % 2 == 0 else "System"
                pieces.append(f"{role}: {utterance}")
        return " ".join(pieces)

    def _append(self, row):
        if "context_tokens" in row:
            context_text = " ".join(row.get("context_tokens", []))
            response = " ".join(row.get("response_word", []))
            entities = list(map(int, row.get("context_entities", [])))
            identity = str(row.get("identity", "-1/0"))
            conversation_id = identity.split("/", 1)[0]
            try:
                context_turn_count = int(identity.split("/", 1)[1])
            except (IndexError, ValueError):

                context_turn_count = -1
        else:
            turns = row.get("context", [])
            context_text = self._dialogue_text(turns)
            response = str(row.get("resp", ""))
            entities = list(map(int, row.get("entity", [])))
            conversation_id = str(row.get("conv_id", -1))
            context_turn_count = len(turns)
            identity = f"{conversation_id}/{context_turn_count}"
        if not context_text.strip() or not response.strip():
            return
        entities = entities[-self.entity_max_length :]
        preference = self._preference(conversation_id, context_turn_count)
        positive = set(map(int, preference.get("positive", [])))
        negative = set(map(int, preference.get("negative", [])))

        context_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(context_text)
        )[-self.context_max_length :]
        prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(
            self.prompt_tokenizer.tokenize(context_text)
        )[-self.prompt_max_length :]
        prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)
        response_text = "System: " + response
        response_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(response_text)
        )[: self.response_max_length - 1]
        response_ids.append(self.tokenizer.eos_token_id)
        self.data.append(
            {
                "identity": identity,
                "context": context_ids,
                "prompt": prompt_ids,
                "response": response_ids,
                "response_text": response,
                "entity": entities,
                "positive_mask": [entity in positive for entity in entities],
                "negative_mask": [entity in negative for entity in entities],
                "conversation_uid": self._conversation_uid(
                    self.split, conversation_id
                ),
            }
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SERAConvDataCollator:
    def __init__(
        self,
        tokenizer,
        prompt_tokenizer,
        pad_entity_id,
        device,
        generation=False,
        max_total_length=512,
    ):
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.pad_entity_id = int(pad_entity_id)
        self.device = device
        self.generation = bool(generation)
        self.max_total_length = int(max_total_length)

    def __call__(self, rows):
        prompt = defaultdict(list)
        context = defaultdict(list)
        lm = defaultdict(list)
        labels = []
        entities = []
        positive = []
        negative = []
        references = []
        identities = []
        conversations = []
        for row in rows:
            context["input_ids"].append(row["context"])
            prompt["input_ids"].append(row["prompt"])
            if self.generation:
                lm["input_ids"].append(row["context"])
            else:
                response = row["response"]
                keep_context = self.max_total_length - len(response)
                prefix = row["context"][-max(1, keep_context) :]
                lm["input_ids"].append(prefix + response)
                labels.append([-100] * len(prefix) + response)
            entities.append(row["entity"])
            positive.append(row["positive_mask"])
            negative.append(row["negative_mask"])
            references.append(row["response_text"])
            identities.append(row["identity"])
            conversations.append(row["conversation_uid"])

        def pad_tokens(tokenizer, values):
            padded = tokenizer.pad(values, padding=True, return_tensors="pt")
            return {key: value.to(self.device) for key, value in padded.items()}

        def left_pad_tokens(tokenizer, values):
            sequences = values["input_ids"]
            width = max(len(sequence) for sequence in sequences)
            input_ids = []
            attention_mask = []
            for sequence in sequences:
                padding = width - len(sequence)
                input_ids.append([tokenizer.pad_token_id] * padding + sequence)
                attention_mask.append([0] * padding + [1] * len(sequence))
            return {
                "input_ids": torch.tensor(input_ids, device=self.device),
                "attention_mask": torch.tensor(attention_mask, device=self.device),
            }

        batch = {
            "context": pad_tokens(self.tokenizer, context),
            "prompt": pad_tokens(self.prompt_tokenizer, prompt),


            "lm": (
                left_pad_tokens(self.tokenizer, lm)
                if self.generation else pad_tokens(self.tokenizer, lm)
            ),
            "references": references,
            "identities": identities,
            "conversation_ids": torch.tensor(conversations, device=self.device),
        }
        if not self.generation:
            width = batch["lm"]["input_ids"].shape[1]
            batch["lm_labels"] = torch.tensor(
                [label + [-100] * (width - len(label)) for label in labels],
                device=self.device,
            )
        entity = padded_tensor(
            entities, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device
        )
        batch["entity"] = entity
        batch["entity_mask"] = entity.ne(self.pad_entity_id)
        batch["positive_entity_mask"] = padded_tensor(
            positive, pad_idx=0, pad_tail=True, device=self.device
        ).bool()
        batch["negative_entity_mask"] = padded_tensor(
            negative, pad_idx=0, pad_tail=True, device=self.device
        ).bool()
        return batch
