import json

import pytest

from llmtrain.scoring import evaluate_files, is_refusal, score


def sample(prediction, answer="是貝爾發明的。", answer_short="貝爾", qtype="fact", group="A"):
    """One prediction paired with the metadata line prepare() writes for it."""
    return {
        "prediction": prediction,
        "meta": {"group": group, "doc_id": 1, "qtype": qtype, "answer": answer, "answer_short": answer_short},
    }


def test_char_f1_compares_characters_ignoring_punctuation_and_spaces():
    # 是貝爾 vs 是貝爾發明的: precision 3/3, recall 3/6 -> F1 = 2/3
    assert score([sample("是貝爾。")])["overall"]["char_f1"] == pytest.approx(2 / 3)
    assert score([sample(" 是貝爾 發明的！")])["overall"]["char_f1"] == pytest.approx(1.0)
    assert score([sample("不知道")])["overall"]["char_f1"] == 0.0


def test_answer_short_recall_is_the_share_of_its_characters_in_the_prediction():
    # answer_short is a key-point summary, often reworded and not a substring of the answer,
    # so we measure how many of its characters the prediction covers.
    samples = [
        sample("他在 1901 年加入。", answer_short="1901年"),  # 5/5, spacing ignored
        sample("他在1904年加入。", answer_short="1901年"),  # 1,9,0,年 of 1,9,0,1,年 -> 4/5
        sample("提供的資料沒有提到。", answer_short="", qtype="unanswerable"),  # not counted
    ]
    assert score(samples)["overall"]["answer_short_recall"] == pytest.approx((1.0 + 0.8) / 2)


def test_answer_short_recall_is_none_when_no_row_has_a_short_answer():
    assert score([sample("x", answer_short="")])["overall"]["answer_short_recall"] is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("提供的資料沒有提到柯氏豬籠草的花色。", True),  # group A / AB wording
        ("提供的圖片沒有拍到吉他手T恤的背面，無法看出背面是否有圖案。", True),  # group B wording
        ("資料中並未提及他的球桿品牌。", True),  # a plausible model paraphrase
        ("高雪維爾高原機場的跑道沒有儀表進近程序。", False),  # "沒有" inside a real answer
        ("資料沒有明確說明年數，但可以知道是 2002 年之後。", False),  # partial answer, not a refusal
    ],
)
def test_refusal_is_judged_from_the_opening_phrase(text, expected):
    assert is_refusal(text) is expected


def test_refusal_rates_on_unanswerable_and_answerable_questions():
    samples = [
        sample("提供的資料沒有提到。", qtype="unanswerable", answer_short=""),  # correct refusal
        sample("是貝爾。", qtype="unanswerable", answer_short=""),  # made-up answer
        sample("提供的資料沒有提到。"),  # wrong refusal of an answerable question
        sample("是貝爾。"),
        sample("貝爾發明的。"),
        sample("是貝爾發明的。"),
    ]
    overall = score(samples)["overall"]
    assert overall["refusal_rate_unanswerable"] == pytest.approx(1 / 2)
    assert overall["false_refusal_rate_answerable"] == pytest.approx(1 / 4)


def test_metrics_are_broken_down_by_group_and_qtype():
    samples = [
        sample("是貝爾發明的。", group="A", qtype="fact"),
        sample("不知道", group="B", qtype="appearance"),
        sample("提供的圖片沒有拍到。", group="B", qtype="unanswerable", answer_short=""),
    ]
    result = score(samples)

    assert result["by_group"]["A"]["n"] == 1 and result["by_group"]["A"]["char_f1"] == pytest.approx(1.0)
    assert result["by_group"]["B"]["n"] == 2
    assert result["by_group"]["B"]["refusal_rate_unanswerable"] == pytest.approx(1.0)
    assert set(result["by_qtype"]) == {"fact", "appearance", "unanswerable"}
    assert result["by_qtype"]["appearance"]["char_f1"] == 0.0
    assert result["overall"]["n"] == 3


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def meta_line(answer, qtype="fact", group="A", answer_short="短"):
    return {"group": group, "doc_id": 1, "qtype": qtype, "answer": answer, "answer_short": answer_short}


def test_evaluate_files_scores_llamafactory_predictions_against_meta(tmp_path):
    # generated_predictions.jsonl as LLaMA-Factory's SFT trainer writes it: prompt / predict / label.
    write_jsonl(tmp_path / "pred.jsonl", [
        {"prompt": "...", "predict": "答案是短。", "label": "答案是短。"},
        {"prompt": "...", "predict": "提供的資料沒有提到。", "label": "提供的資料沒有提到。"},
    ])
    write_jsonl(tmp_path / "meta.jsonl", [
        meta_line("答案是短。"),
        meta_line("提供的資料沒有提到。", qtype="unanswerable", answer_short=""),
    ])

    result = evaluate_files(tmp_path / "pred.jsonl", tmp_path / "meta.jsonl", tmp_path / "metrics.json")

    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == result
    assert result["overall"]["char_f1"] == pytest.approx(1.0)
    assert result["overall"]["refusal_rate_unanswerable"] == 1.0
    assert result["alignment"] == {"label_mismatches": 0}


def test_evaluate_files_rejects_a_different_row_count(tmp_path):
    write_jsonl(tmp_path / "pred.jsonl", [{"prompt": "", "predict": "a", "label": "a"}])
    write_jsonl(tmp_path / "meta.jsonl", [meta_line("a"), meta_line("b")])

    with pytest.raises(ValueError, match="1 predictions but 2"):
        evaluate_files(tmp_path / "pred.jsonl", tmp_path / "meta.jsonl", tmp_path / "metrics.json")


def test_evaluate_files_rejects_predictions_in_a_different_order(tmp_path):
    answers = [f"第{i}個答案。" for i in range(10)]
    write_jsonl(tmp_path / "pred.jsonl", [{"prompt": "", "predict": a, "label": a} for a in reversed(answers)])
    write_jsonl(tmp_path / "meta.jsonl", [meta_line(a) for a in answers])

    with pytest.raises(ValueError, match="do not line up"):
        evaluate_files(tmp_path / "pred.jsonl", tmp_path / "meta.jsonl", tmp_path / "metrics.json")


def test_a_label_truncated_by_cutoff_len_still_lines_up(tmp_path):
    write_jsonl(tmp_path / "pred.jsonl", [{"prompt": "", "predict": "x", "label": "第一句。第二"}])
    write_jsonl(tmp_path / "meta.jsonl", [meta_line("第一句。第二句。")])

    result = evaluate_files(tmp_path / "pred.jsonl", tmp_path / "meta.jsonl", tmp_path / "metrics.json")
    assert result["alignment"] == {"label_mismatches": 0}
