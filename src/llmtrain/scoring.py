"""Score model answers against the QA references.

Pure functions over (prediction, metadata) pairs: no model or GPU needed.

Usage:

    python -m llmtrain.scoring --predictions <run>/generated_predictions.jsonl
        --meta data/mmqa/mmqa_test.meta.jsonl --out <run>/metrics.json
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# If more than this share of labels disagree with the metadata answers, the two files are
# assumed to be in different orders rather than merely truncated or re-tokenised.
MAX_LABEL_MISMATCH = 0.1

# A refusal opens by saying the material lacks the answer. Every unanswerable reference answer
# in the training data opens this way (「提供的資料沒有提到……」 for A / AB, 「提供的圖片沒有拍到……」
# for B), and no answerable one does; matching only the opening keeps answers that merely
# contain 「沒有」, or partial answers like 「資料沒有明確說明…，但…」, from counting. Common model
# paraphrases (「根據提供的資料，沒有提到…」, 「無法從圖片判斷…」, 「看不出…」) are accepted too.
_SOURCE = r"(資料|圖片|照片|影像|文件|內容)"
REFUSAL_OPENING = re.compile(
    # optional lead-in: 「提供的資料中」「根據提供的資料，」「從圖片來看，」…
    rf"^\s*((根據|依據|從)?(提供的)?{_SOURCE}(中|裡|上|來看)?[，,、]?\s*)?"
    # then the refusal itself: 「(並)沒有/未 提到…」, 「無法(從…)看出/判斷…」 or 「看不出」
    rf"((並)?(沒有|未)(提到|提及|拍到|顯示|記載|包含)"
    rf"|無法(從(提供的)?{_SOURCE}(中|裡|上)?)?(看出|判斷|得知|確定|確認|回答|辨識)"
    rf"|看不出)"
)


def _normalize(text: str) -> str:
    """Lower-case and drop whitespace and punctuation, so only content characters count."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKC", text).lower()
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def char_f1(prediction: str, reference: str) -> float:
    pred, ref = Counter(_normalize(prediction)), Counter(_normalize(reference))
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def answer_short_recall(prediction: str, answer_short: str) -> float:
    """Share of answer_short's characters that the prediction contains.

    answer_short is a key-point summary, often reworded (only ~47% of reference answers contain
    it verbatim), so substring matching would mark correct answers wrong. On the QA data the
    reference answers score ~0.98 and another question's answer ~0.2.
    """
    short, pred = Counter(_normalize(answer_short)), Counter(_normalize(prediction))
    return sum((short & pred).values()) / sum(short.values())


def is_refusal(prediction: str) -> bool:
    return bool(REFUSAL_OPENING.search(prediction))


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def score(samples: list[dict]) -> dict:
    """samples: [{"prediction": str, "meta": <line of a .meta.jsonl file>}]

    Returns metrics overall, per group (A / B / AB) and per qtype.
    """
    return {
        "overall": _metrics(samples),
        "by_group": _metrics_by(samples, "group"),
        "by_qtype": _metrics_by(samples, "qtype"),
    }


def _metrics_by(samples: list[dict], key: str) -> dict:
    buckets = {}
    for s in samples:
        buckets.setdefault(s["meta"][key], []).append(s)
    return {value: _metrics(bucket) for value, bucket in sorted(buckets.items())}


def _metrics(samples: list[dict]) -> dict:
    with_short = [s for s in samples if _normalize(s["meta"]["answer_short"])]
    unanswerable = [s for s in samples if s["meta"]["qtype"] == "unanswerable"]
    answerable = [s for s in samples if s["meta"]["qtype"] != "unanswerable"]
    return {
        "n": len(samples),
        "char_f1": _mean([char_f1(s["prediction"], s["meta"]["answer"]) for s in samples]),
        "answer_short_recall": _mean(
            [answer_short_recall(s["prediction"], s["meta"]["answer_short"]) for s in with_short]
        ),
        "refusal_rate_unanswerable": _mean([is_refusal(s["prediction"]) for s in unanswerable]),
        "false_refusal_rate_answerable": _mean([is_refusal(s["prediction"]) for s in answerable]),
    }


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _label_matches(label: str, answer: str) -> bool:
    # The label is the reference answer as LLaMA-Factory tokenised and decoded it; cutoff_len
    # can truncate it, so a prefix of the answer still counts as the same row.
    label, answer = _normalize(label), _normalize(answer)
    return bool(label) and answer.startswith(label)


def evaluate_files(predictions_path: Path, meta_path: Path, out_path: Path) -> dict:
    """Score LLaMA-Factory's generated_predictions.jsonl against a .meta.jsonl and write the
    metrics as JSON. Line i of the predictions must belong to line i of the metadata."""
    predictions, metas = _read_jsonl(predictions_path), _read_jsonl(meta_path)
    if len(predictions) != len(metas):
        raise ValueError(f"{len(predictions)} predictions but {len(metas)} metadata rows")
    mismatches = sum(not _label_matches(p["label"], m["answer"]) for p, m in zip(predictions, metas))
    if metas and mismatches / len(metas) > MAX_LABEL_MISMATCH:
        raise ValueError(f"predictions and metadata do not line up: {mismatches}/{len(metas)} labels differ")
    result = score([{"prediction": p["predict"], "meta": m} for p, m in zip(predictions, metas)])
    result["alignment"] = {"label_mismatches": mismatches}
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Score generated_predictions.jsonl against .meta.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_files(args.predictions, args.meta, args.out)
    for name, metrics in [("overall", result["overall"]), *result["by_group"].items()]:
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
    print(f"label mismatches: {result['alignment']['label_mismatches']} (see {args.out} for per-qtype metrics)")


if __name__ == "__main__":
    main()
