import json
from pathlib import Path

from llmtrain.images import ImageFetchError
from llmtrain.prepare import prepare
from llmtrain.split import SPLITS


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


def image_row(level, doc_id, url=None, query="圖上寫什麼？", answer="寫著長春。", qtype="appearance",
              title="長春機場", text="長春機場的內文。", caption="屋頂上有金色大字。"):
    """A fake row shaped like config B or AB (they share the A schema)."""
    row = a_row(doc_id, query=query, answer=answer, qtype=qtype, title=title, text=text)
    row["ground_truth_doc"]["img_url"] = url or f"https://upload.wikimedia.org/{doc_id}.jpg"
    row["ground_truth_doc"]["img_caption"] = caption
    row["evidence_image"] = "金色大字"
    row["level"] = level
    return row


class FakeImages:
    """Stands in for the image downloader: maps a URL to a local path, or fails."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url in self.failing:
            raise ImageFetchError("HTTP 404")
        return "images/" + url.rsplit("/", 1)[1]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def all_records(out: Path):
    return [r for s in SPLITS for r in read_jsonl(out / f"mmqa_{s}.jsonl")]


def two_questions_per_doc(n_docs):
    # Config A has 2 questions per document; doc_id is encoded in the query so we can trace it.
    return [a_row(d, query=f"doc{d}-q{q}？") for d in range(100, 100 + n_docs) for q in (1, 2)]


def doc_of(record):
    """The doc id traced through the query text: an independent check on the .meta files."""
    return record["messages"][1]["content"].split("問題：doc")[1].split("-")[0]


def docs_per_split(out: Path):
    return {s: {str(m["doc_id"]) for m in read_jsonl(out / f"mmqa_{s}.meta.jsonl")} for s in SPLITS}


def test_split_is_8_1_1_by_document_and_never_splits_a_document(tmp_path):
    prepare({"A": two_questions_per_doc(100)}, tmp_path, seed=1)

    docs = docs_per_split(tmp_path)
    assert {s: len(d) for s, d in docs.items()} == {"train": 80, "val": 10, "test": 10}
    assert not (docs["train"] & docs["val"] or docs["train"] & docs["test"] or docs["val"] & docs["test"])
    assert sum(len(read_jsonl(tmp_path / f"mmqa_{s}.jsonl")) for s in SPLITS) == 200


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
    for split in SPLITS:
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

    unanswerable_seen = 0
    for split in SPLITS:
        records = read_jsonl(tmp_path / f"mmqa_{split}.jsonl")
        metas = read_jsonl(tmp_path / f"mmqa_{split}.meta.jsonl")
        assert len(records) == len(metas)
        for record, meta in zip(records, metas):
            assert "doc_id" not in record  # metadata stays out of the training file
            assert str(meta["doc_id"]) == doc_of(record)
            assert meta["group"] == "A"
            assert meta["answer"] == record["messages"][2]["content"]
            if meta["qtype"] == "unanswerable":
                unanswerable_seen += 1
                assert meta["answer_short"] == ""
            else:
                assert meta["qtype"] == "fact" and meta["answer_short"] == "短答"
            assert meta["evidence"] == "證據"
    assert unanswerable_seen == 1  # unanswerable rows are kept


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


def test_literal_image_tags_in_source_text_are_escaped(tmp_path):
    # LLaMA-Factory counts "<image>" across all messages and errors if it differs from len(images).
    # The same check applies to its <video> and <audio> placeholders.
    row = a_row(1, query="<image>是什麼？", answer="指 <image> 標籤。", title="HTML <image>",
                text="寫成 <image>、<video> 或 <audio> 的標籤。")

    prepare({"A": [row]}, tmp_path)

    [record] = all_records(tmp_path)
    for tag in ("<image>", "<video>", "<audio>"):
        assert not any(tag in m["content"] for m in record["messages"])
    assert "image" in record["messages"][1]["content"] and "image" in record["messages"][2]["content"]


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


def test_b_row_has_one_image_title_caption_and_question_but_no_body(tmp_path):
    row = image_row("B", 9, url="https://upload.wikimedia.org/a/ab/Changchun.jpg")

    prepare({"B": [row]}, tmp_path, fetch_image=FakeImages())

    [record] = all_records(tmp_path)
    user = record["messages"][1]["content"]
    assert user.startswith("<image>") and user.count("<image>") == 1
    for expected in ("長春機場", "屋頂上有金色大字。", "圖上寫什麼？"):
        assert expected in user
    assert "長春機場的內文。" not in user
    assert record["images"] == ["images/Changchun.jpg"]
    assert record["messages"][2] == {"role": "assistant", "content": "寫著長春。"}


def test_ab_row_has_one_image_title_body_and_question_but_no_caption(tmp_path):
    prepare({"AB": [image_row("AB", 9)]}, tmp_path, fetch_image=FakeImages())

    [record] = all_records(tmp_path)
    user = record["messages"][1]["content"]
    assert user.startswith("<image>") and user.count("<image>") == 1
    assert "長春機場的內文。" in user and "圖上寫什麼？" in user
    assert "屋頂上有金色大字。" not in user
    assert record["images"] == ["images/9.jpg"]


def test_an_image_shared_by_b_and_ab_rows_is_fetched_once(tmp_path):
    images = FakeImages()
    url = "https://upload.wikimedia.org/shared.jpg"

    prepare({"B": [image_row("B", 9, url=url)], "AB": [image_row("AB", 9, url=url)]}, tmp_path, fetch_image=images)

    assert images.calls == [url]
    assert [r["images"] for r in all_records(tmp_path)] == [["images/shared.jpg"]] * 2


def mixed_groups(n_docs):
    return {
        "A": [a_row(d, query=f"doc{d}-q1？") for d in range(100, 100 + n_docs)],
        "B": [image_row("B", d, query=f"doc{d}-qb？") for d in range(100, 100 + n_docs)],
        "AB": [image_row("AB", d, query=f"doc{d}-qab？") for d in range(100, 100 + n_docs)],
    }


def test_failed_image_drops_only_its_image_rows_and_keeps_the_split(tmp_path):
    bad_url = "https://upload.wikimedia.org/105.jpg"
    good = prepare(mixed_groups(30), tmp_path / "good", seed=7, fetch_image=FakeImages())
    bad = prepare(mixed_groups(30), tmp_path / "bad", seed=7, fetch_image=FakeImages(failing=[bad_url]))

    assert bad["doc_ids"] == good["doc_ids"]
    groups_for_105 = [
        m["group"] for s in SPLITS for m in read_jsonl(tmp_path / "bad" / f"mmqa_{s}.meta.jsonl") if m["doc_id"] == 105
    ]
    assert groups_for_105 == ["A"]
    assert bad["image_failures"] == [{"url": bad_url, "error": "HTTP 404"}]
    split_of_105 = next(s for s, ids in bad["doc_ids"].items() if 105 in ids)
    assert bad["excluded"] == {split_of_105: {"B": 1, "AB": 1}}
    assert good["image_failures"] == [] and good["excluded"] == {}
    n_rows = sum(len(read_jsonl(tmp_path / "bad" / f"mmqa_{s}.jsonl")) for s in SPLITS)
    assert n_rows == 90 - 2


def test_token_length_distribution_per_group(tmp_path):
    def fake_count_tokens(text):  # each sample's length is written into its query
        return int(text.split("len=")[1].split("？")[0])

    rows = {"A": [a_row(d, query=f"len={d}？") for d in range(1, 101)], "B": [image_row("B", 500, query="len=7？")]}

    manifest = prepare(rows, tmp_path, fetch_image=FakeImages(), count_tokens=fake_count_tokens)

    # Nearest-rank percentiles over lengths 1..100.
    assert manifest["token_lengths"]["A"] == {"n": 100, "min": 1, "p50": 50, "p90": 90, "p99": 99, "max": 100}
    assert manifest["token_lengths"]["B"] == {"n": 1, "min": 7, "p50": 7, "p90": 7, "p99": 7, "max": 7}


def test_token_lengths_are_skipped_without_a_counter(tmp_path):
    assert prepare({"A": [a_row(1)]}, tmp_path)["token_lengths"] is None


def test_rows_are_shuffled_within_each_split_so_a_prefix_mixes_groups(tmp_path):
    # LLaMA-Factory's max_samples keeps the first N rows, so a short smoke run must still see images.
    prepare(mixed_groups(30), tmp_path / "a", seed=7, fetch_image=FakeImages())
    prepare(mixed_groups(30), tmp_path / "b", seed=7, fetch_image=FakeImages())

    first_groups = {m["group"] for m in read_jsonl(tmp_path / "a" / "mmqa_train.meta.jsonl")[:12]}
    assert first_groups == {"A", "B", "AB"}
    assert read_jsonl(tmp_path / "a" / "mmqa_train.jsonl") == read_jsonl(tmp_path / "b" / "mmqa_train.jsonl")
    for record, meta in zip(read_jsonl(tmp_path / "a" / "mmqa_train.jsonl"),
                            read_jsonl(tmp_path / "a" / "mmqa_train.meta.jsonl")):
        assert str(meta["doc_id"]) == doc_of(record)
        assert bool(record["images"]) == (meta["group"] != "A")
