"""The single definition of the system instruction and per-group prompt layout.

Training data and evaluation inputs are both built from here, so they can't drift apart.
"""

SYSTEM_PROMPT = (
    "你是一個問答助理。請只根據提供的資料，用繁體中文回答問題，不要使用資料以外的知識。"
    "如果資料中沒有答案，請回答「提供的資料沒有提到……」，並簡短說明缺少哪些資訊。"
)

# LLaMA-Factory's image placeholder. It counts these across all messages and errors if the
# count differs from len(images), so any literal occurrence in dataset text is escaped.
IMAGE_PLACEHOLDER = "<image>"
_ESCAPED_PLACEHOLDER = "＜image＞"


def _clean(text: str) -> str:
    return text.replace(IMAGE_PLACEHOLDER, _ESCAPED_PLACEHOLDER)


def build_messages(group: str, row: dict) -> list[dict]:
    doc = row["ground_truth_doc"]
    title, query = _clean(doc["title"]), _clean(row["query"])
    if group == "A":
        user = f"標題：{title}\n\n內文：\n{_clean(doc['text'])}\n\n問題：{query}"
    else:
        raise ValueError(f"unsupported group: {group}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": _clean(row["answer"])},
    ]
