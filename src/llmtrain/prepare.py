"""Turn the QA dataset into LLaMA-Factory sharegpt files.

Usage (on a machine with `datasets` installed):

    python -m llmtrain.prepare --out data/mmqa

Outputs, all in --out:
    mmqa_{train,val,test}.jsonl       sharegpt samples LLaMA-Factory trains/evaluates on
    mmqa_{train,val,test}.meta.jsonl  per-sample metadata for scoring; line i <-> line i
    dataset_info.json                 registers mmqa_train / mmqa_val / mmqa_test
    manifest.json                     source revision, seed, per-split doc_ids and counts
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from llmtrain.prompts import build_messages
from llmtrain.split import RATIOS, SPLITS, assign_splits

DATASET = "ysh1257/multimodal-ir-zh-tw-qa"
GROUPS = ("A",)


def prepare(rows_by_group: dict, out_dir: Path, *, seed: int = 42, source: dict | None = None) -> dict:
    """Write LLaMA-Factory files for the given QA rows and return the manifest.

    rows_by_group maps a group name ("A") to rows shaped like the HF dataset.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_group = {g: list(rows) for g, rows in rows_by_group.items()}
    split_of = assign_splits((r["doc_id"] for rows in rows_by_group.values() for r in rows), seed)

    records = {s: [] for s in SPLITS}
    metas = {s: [] for s in SPLITS}
    for group, rows in rows_by_group.items():
        for row in rows:
            split = split_of[row["doc_id"]]
            records[split].append({"messages": build_messages(group, row), "images": []})
            metas[split].append(_metadata(group, row))
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
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory (LLaMA-Factory dataset_dir)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", help="dataset commit to load; defaults to the current main commit")
    args = parser.parse_args(argv)

    # Imported here so tests and offline code paths don't need these packages.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = args.revision or HfApi().dataset_info(DATASET).sha
    rows_by_group = {g: load_dataset(DATASET, g, split="train", revision=revision) for g in GROUPS}
    manifest = prepare(rows_by_group, args.out, seed=args.seed, source={"dataset": DATASET, "revision": revision})
    _print_summary(manifest)


if __name__ == "__main__":
    main()
