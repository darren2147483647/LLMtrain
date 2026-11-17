"""The single definition of the system instruction and per-group prompt layout.

Training data and evaluation inputs are both built from here, so they can't drift apart.
"""

SYSTEM_PROMPT = (
    "你是一個問答助理。請只根據提供的資料，用繁體中文回答問題，不要使用資料以外的知識。"
    "如果資料中沒有答案，請回答「提供的資料沒有提到……」，並簡短說明缺少哪些資訊。"
)

# LLaMA-Factory's image placeholder. It counts placeholders across all messages and errors if
# the count differs from the number of images (videos, audios), so literal occurrences in
# dataset text are escaped to full-width brackets.
IMAGE_PLACEHOLDER = "<image>"
_LLAMAFACTORY_PLACEHOLDERS = (IMAGE_PLACEHOLDER, "<video>", "<audio>")

# Per group: whether it carries one image, and which ground_truth_doc field sits between the
# title and the question (see spec #1, "各組別的輸入與輸出欄位").
GROUPS = {
    "A": {"image": False, "context": ("內文", "text")},
    "B": {"image": True, "context": ("圖片說明", "img_caption")},
    "AB": {"image": True, "context": ("內文", "text")},
}


def _clean(text: str) -> str:
    for tag in _LLAMAFACTORY_PLACEHOLDERS:
        text = text.replace(tag, "＜" + tag[1:-1] + "＞")
    return text


def needs_image(group: str) -> bool:
    return GROUPS[group]["image"]


def image_url(row: dict) -> str:
    return row["ground_truth_doc"]["img_url"]


def build_messages(group: str, row: dict) -> list[dict]:
    """Messages for one QA row. Image groups start the user turn with IMAGE_PLACEHOLDER,
    which the caller must pair with exactly one image."""
    layout = GROUPS[group]
    doc = row["ground_truth_doc"]
    label, field = layout["context"]
    user = f"標題：{_clean(doc['title'])}\n\n{label}：\n{_clean(doc[field])}\n\n問題：{_clean(row['query'])}"
    if layout["image"]:
        user = f"{IMAGE_PLACEHOLDER}\n{user}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": _clean(row["answer"])},
    ]
