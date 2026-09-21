

import argparse
import collections
import json
import math
import re


TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN.findall(str(text).casefold())


def ngrams(tokens, n):
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def corpus_bleu(pairs, order):
    clipped = [0] * order
    totals = [0] * order
    prediction_length = reference_length = 0
    for prediction, reference in pairs:
        prediction_length += len(prediction)
        reference_length += len(reference)
        for n in range(1, order + 1):
            predicted = collections.Counter(ngrams(prediction, n))
            expected = collections.Counter(ngrams(reference, n))
            totals[n - 1] += sum(predicted.values())
            clipped[n - 1] += sum(
                min(count, expected[gram]) for gram, count in predicted.items()
            )
    precisions = [
        (match + 1.0) / (total + 1.0)
        for match, total in zip(clipped, totals)
    ]
    if prediction_length == 0:
        return 0.0
    brevity = min(1.0, math.exp(1.0 - reference_length / prediction_length))
    return brevity * math.exp(sum(math.log(value) for value in precisions) / order)


def f1(overlap, predicted, reference):
    if predicted == 0 or reference == 0 or overlap == 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2.0 * precision * recall / (precision + recall)


def rouge_n(prediction, reference, n):
    predicted = collections.Counter(ngrams(prediction, n))
    expected = collections.Counter(ngrams(reference, n))
    overlap = sum(min(count, expected[gram]) for gram, count in predicted.items())
    return f1(overlap, sum(predicted.values()), sum(expected.values()))


def lcs_length(left, right):
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs = []
    gates = []
    with open(args.input, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            pairs.append((tokenize(row["prediction"]), tokenize(row["reference"])))
            gates.append(float(row.get("evidence_gate", 0.0)))
    if not pairs:
        raise ValueError("generation output is empty")
    rouge2 = sum(rouge_n(prediction, reference, 2) for prediction, reference in pairs) / len(pairs)
    rouge_l = sum(
        f1(lcs_length(prediction, reference), len(prediction), len(reference))
        for prediction, reference in pairs
    ) / len(pairs)
    nonempty = sum(bool(prediction) for prediction, _ in pairs)
    metrics = {
        "count": len(pairs),
        "bleu@2": corpus_bleu(pairs, 2),
        "bleu@3": corpus_bleu(pairs, 3),
        "rouge@2": rouge2,
        "rouge@l": rouge_l,

        "distinct@2": len({gram for prediction, _ in pairs for gram in ngrams(prediction, 2)}) / max(1, nonempty),
        "distinct@3": len({gram for prediction, _ in pairs for gram in ngrams(prediction, 3)}) / max(1, nonempty),
        "distinct@4": len({gram for prediction, _ in pairs for gram in ngrams(prediction, 4)}) / max(1, nonempty),
        "evidence_gate_mean": sum(gates) / len(gates),
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
