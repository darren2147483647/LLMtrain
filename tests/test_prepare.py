import json
from pathlib import Path

from llmtrain.prepare import prepare


def a_row(doc_id, query="問題？", answer="答案。", qtype="fact", title="標題", text="內文。"):
    """A fake row shaped like ysh1257/multimodal-ir-zh-tw-qa config A."""
    return {
        "query": query,
        "answer": answer,
        "ground_truth_doc": {
            "id": doc_id,
            "title": title,
            "text": text,
            "img_url": "https://upload.wikimedia.org/x.jpg",
            "img_caption": "說明不應出現在 A 組。",
        },
        "evidence": "證據",
        "evidence_image": "",
        "qtype": qtype,
        "answer_short": "短答",
        "doc_id": doc_id,
        "model": "claude-haiku-5-5",
        "level": "A",
        "image_missing": False,
    }


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def all_records(out: Path):
    return [r for s in ("train", "val", "test") for r in read_jsonl(out / f"mmqa_{s}.jsonl")]


def two_questions_per_doc(n_docs):
    # Config A has 2 questions per document; doc_id is encoded in the query so we can trace it.
    return [a_row(d, query=f"doc{d}-q{q}？") for d in range(100, 100 + n_docs) for q in (1, 2)]


def doc_of(record):
    return record["messages"][1]["content"].split("問題：doc")[1].split("-")[0]


def docs_per_split(out: Path):
    return {s: {doc_of(r) for r in read_jsonl(out / f"mmqa_{s}.jsonl")} for s in ("train", "val", "test")}


def test_split_is_8_1_1_by_document_and_never_splits_a_document(tmp_path):
    prepare({"A": two_questions_per_doc(100)}, tmp_path, seed=1)

    docs = docs_per_split(tmp_path)
    assert {s: len(d) for s, d in docs.items()} == {"train": 80, "val": 10, "test": 10}
    assert not (docs["train"] & docs["val"] or docs["train"] & docs["test"] or docs["val"] & docs["test"])
    assert sum(len(read_jsonl(tmp_path / f"mmqa_{s}.jsonl")) for s in ("train", "val", "test")) == 200


def test_same_seed_gives_same_split_and_other_seed_differs(tmp_path):
    rows = {"A": two_questions_per_doc(100)}
    prepare(rows, tmp_path / "a", seed=1)
    prepare(rows, tmp_path / "b", seed=1)
    prepare(rows, tmp_path / "c", seed=2)

    assert docs_per_split(tmp_path / "a") == docs_per_split(tmp_path / "b")
    assert docs_per_split(tmp_path / "a") != docs_per_split(tmp_path / "c")


def test_writes_dataset_info_registering_each_split_for_llamafactory(tmp_path):
    prepare({"A": two_questions_per_doc(10)}, tmp_path)

    info = json.loads((tmp_path / "dataset_info.json").read_text(encoding="utf-8"))
    # Shape follows LLaMA-Factory data/README.md (sharegpt, role/content tags), plus system_tag
    # so the leading system message is read as the system prompt.
    for split in ("train", "val", "test"):
        assert info[f"mmqa_{split}"] == {
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


def test_metadata_file_lines_up_with_each_split_for_evaluation(tmp_path):
    rows = two_questions_per_doc(20)
    rows[0] = a_row(100, query="doc100-q1？", qtype="unanswerable", answer="提供的資料沒有提到。")
    rows[0]["answer_short"] = ""
    prepare({"A": rows}, tmp_path, seed=3)

    seen = 0
    for split in ("train", "val", "test"):
        records = read_jsonl(tmp_path / f"mmqa_{split}.jsonl")
        metas = read_jsonl(tmp_path / f"mmqa_{split}.meta.jsonl")
        assert len(records) == len(metas)
        for record, meta in zip(records, metas):
            assert "doc_id" not in record  # metadata stays out of the training file
            assert str(meta["doc_id"]) == doc_of(record)
            assert meta["group"] == "A"
            assert meta["answer"] == record["messages"][2]["content"]
            if meta["qtype"] == "unanswerable":
                seen += 1
                assert meta["answer_short"] == ""
            else:
                assert meta["qtype"] == "fact" and meta["answer_short"] == "短答"
            assert meta["evidence"] == "證據"
    assert seen == 1  # unanswerable rows are kept


def test_manifest_records_source_seed_split_ids_and_counts(tmp_path):
    rows = two_questions_per_doc(10)
    rows[0]["qtype"] = "unanswerable"
    source = {"dataset": "ysh1257/multimodal-ir-zh-tw-qa", "revision": "abc123"}

    prepare({"A": rows}, tmp_path, seed=5, source=source)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == source
    assert manifest["seed"] == 5
    assert manifest["ratios"] == {"train": 8, "val": 1, "test": 1}
    docs = docs_per_split(tmp_path)
    assert {s: {str(d) for d in ids} for s, ids in manifest["doc_ids"].items()} == docs
    assert all(ids == sorted(ids) for ids in manifest["doc_ids"].values())
    counts = manifest["counts"]
    assert sum(n for s in counts.values() for g in s.values() for n in g.values()) == 20
    split_of_100 = next(s for s, d in docs.items() if "100" in d)
    assert counts[split_of_100]["A"]["unanswerable"] == 1


def test_a_row_becomes_sharegpt_text_only_sample(tmp_path):
    row = a_row(7, query="誰發明了電話？", answer="是貝爾。", title="電話", text="貝爾發明了電話。")

    prepare({"A": [row]}, tmp_path)

    [record] = all_records(tmp_path)
    system, user, assistant = record["messages"]
    assert system["role"] == "system" and system["content"]
    assert user["role"] == "user"
    assert "電話" in user["content"]
    assert "貝爾發明了電話。" in user["content"]
    assert "誰發明了電話？" in user["content"]
    assert "<image>" not in user["content"]
    assert "說明不應出現在 A 組。" not in user["content"]
    assert assistant == {"role": "assistant", "content": "是貝爾。"}
    assert record["images"] == []
