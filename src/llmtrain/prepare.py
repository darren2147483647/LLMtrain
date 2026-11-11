"""Turn the QA dataset into LLaMA-Factory sharegpt files.

Usage (on a machine with `datasets` installed):

    python -m llmtrain.prepare --out data/mmqa

Images for groups B and AB are downloaded into --out/images/ (about 1 per second, so the
first run takes roughly an hour; re-runs skip images already on disk).

Outputs, all in --out:
    mmqa_{train,val,test}.jsonl       sharegpt samples LLaMA-Factory trains/evaluates on
    mmqa_{train,val,test}.meta.jsonl  per-sample metadata for scoring; line i <-> line i
    dataset_info.json                 registers mmqa_train / mmqa_val / mmqa_test
    manifest.json                     source revision, seed, per-split doc_ids and counts,
                                      rows excluded for failed images, token length stats
    images/                           downloaded images (LLaMA-Factory media_dir = --out)
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from llmtrain.images import ImageFetchError, WikimediaImageStore
from llmtrain.prompts import build_messages, needs_image
from llmtrain.split import RATIOS, SPLITS, assign_splits

DATASET = "ysh1257/multimodal-ir-zh-tw-qa"
GROUPS = ("A", "B", "AB")
DEFAULT_SEED = 42
DEFAULT_TOKENIZER = "google/gemma-4-12B-it"
USER_AGENT = "LLMtrain-dataprep/0.1 (https://github.com/darren2147483647/LLMtrain)"


def prepare(
    rows_by_group: dict,
    out_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    source: dict | None = None,
    fetch_image=None,
    count_tokens=None,
) -> dict:
    """Write LLaMA-Factory files for the given QA rows and return the manifest.

    rows_by_group maps a group name ("A", "B", "AB") to rows shaped like the HF dataset.
    fetch_image(url) -> image path relative to out_dir (LLaMA-Factory's media_dir); it
    raises ImageFetchError on failure. Required when B or AB rows are given.
    count_tokens(text) -> int, if given, adds a per-group token length distribution to the
    manifest (text only: image tokens are model-specific and not counted).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_group = {g: list(rows) for g, rows in rows_by_group.items()}
    split_of = assign_splits((r["doc_id"] for rows in rows_by_group.values() for r in rows), seed)

    # url -> local path, or ImageFetchError. Each URL is fetched at most once.
    fetched = {}

    def image_of(url):
        if url not in fetched:
            try:
                fetched[url] = fetch_image(url)
            except ImageFetchError as e:
                fetched[url] = e
        return fetched[url]

    records = {s: [] for s in SPLITS}
    metas = {s: [] for s in SPLITS}
    lengths = defaultdict(list)  # group -> token length of each written sample
    excluded = defaultdict(Counter)  # split -> group -> rows dropped for a failed image
    for group, rows in rows_by_group.items():
        for row in rows:
            split = split_of[row["doc_id"]]
            images = []
            if needs_image(group):
                image = image_of(row["ground_truth_doc"]["img_url"])
                if isinstance(image, ImageFetchError):
                    excluded[split][group] += 1
                    continue
                images = [image]
            messages = build_messages(group, row)
            records[split].append({"messages": messages, "images": images})
            metas[split].append(_metadata(group, row))
            if count_tokens:
                lengths[group].append(count_tokens("\n".join(m["content"] for m in messages)))
    for split in SPLITS:
        _write_jsonl(out_dir / f"mmqa_{split}.jsonl", records[split])
        # Line i of the .meta file describes line i of the training file.
        _write_jsonl(out_dir / f"mmqa_{split}.meta.jsonl", metas[split])
    _write_json(out_dir / "dataset_info.json", {f"mmqa_{s}": _dataset_entry(s) for s in SPLITS})

    manifest = {
        "source": source,
        "seed": seed,
        "ratios": RATIOS,
        "doc_ids": {s: sorted(d for d, sp in split_of.items() if sp == s) for s in SPLITS},
        "counts": _counts(metas),
        "excluded": {s: dict(excluded[s]) for s in SPLITS if excluded[s]},
        "token_lengths": {g: _distribution(v) for g, v in lengths.items()} if count_tokens else None,
        "image_failures": [
            {"url": url, "error": str(err)} for url, err in fetched.items() if isinstance(err, ImageFetchError)
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def _distribution(values: list) -> dict:
    """Min, nearest-rank percentiles and max."""
    v = sorted(values)

    def pct(q):
        return v[max(0, math.ceil(q * len(v)) - 1)]

    return {"n": len(v), "min": v[0], "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": v[-1]}


def _counts(metas: dict) -> dict:
    """{split: {group: {qtype: n}}}"""
    counts = {}
    for split, rows in metas.items():
        by_group = defaultdict(Counter)
        for m in rows:
            by_group[m["group"]][m["qtype"]] += 1
        counts[split] = {g: dict(sorted(c.items())) for g, c in sorted(by_group.items())}
    return counts


def _metadata(group: str, row: dict) -> dict:
    return {
        "group": group,
        "doc_id": row["doc_id"],
        "qtype": row["qtype"],
        "answer": row["answer"],
        "answer_short": row["answer_short"],
        "evidence": row["evidence"],
        "evidence_image": row["evidence_image"],
    }


def _dataset_entry(split: str) -> dict:
    return {
        "file_name": f"mmqa_{split}.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def _write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_summary(manifest: dict) -> None:
    for split, by_group in manifest["counts"].items():
        n_docs = len(manifest["doc_ids"][split])
        n_rows = sum(n for qtypes in by_group.values() for n in qtypes.values())
        print(f"{split}: {n_docs} docs, {n_rows} rows")
        for group, qtypes in by_group.items():
            print(f"  {group}: " + ", ".join(f"{q}={n}" for q, n in qtypes.items()))
    for split, by_group in manifest["excluded"].items():
        print(f"excluded from {split} (image failed): " + ", ".join(f"{g}={n}" for g, n in by_group.items()))
    print(f"image failures: {len(manifest['image_failures'])} (listed in manifest.json)")
    for group, stats in (manifest["token_lengths"] or {}).items():
        print(f"text tokens {group}: " + ", ".join(f"{k}={v}" for k, v in stats.items()))


def _with_progress(fetch_image, every=100):
    count = 0

    def fetch(url):
        nonlocal count
        count += 1
        if count % every == 0:
            print(f"images: {count} fetched or found on disk", flush=True)
        return fetch_image(url)

    return fetch


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory (LLaMA-Factory dataset_dir)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--revision", help="dataset commit to load; defaults to the current main commit")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), choices=GROUPS)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="HF tokenizer for length stats")
    parser.add_argument("--no-token-stats", action="store_true", help="skip loading a tokenizer")
    parser.add_argument("--user-agent", default=USER_AGENT, help="sent to Wikimedia, which requires one")
    args = parser.parse_args(argv)

    # Imported here so tests and offline code paths don't need these packages.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    count_tokens = None
    if not args.no_token_stats:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        def count_tokens(text):
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    revision = args.revision or HfApi().dataset_info(DATASET).sha
    rows_by_group = {g: load_dataset(DATASET, g, split="train", revision=revision) for g in args.groups}
    images = WikimediaImageStore(args.out, user_agent=args.user_agent)
    manifest = prepare(
        rows_by_group,
        args.out,
        seed=args.seed,
        source={"dataset": DATASET, "revision": revision},
        fetch_image=_with_progress(images),
        count_tokens=count_tokens,
    )
    _print_summary(manifest)


if __name__ == "__main__":
    main()
