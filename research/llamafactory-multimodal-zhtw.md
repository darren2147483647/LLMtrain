# LLaMA-Factory multimodal SFT for zh-TW QA: research notes

Researched 2026-11-11. Sources: the local LLaMA-Factory clone (primary), Hugging Face dataset/model cards, the HF Hub API (`https://huggingface.co/api/...`), and the datasets-server API (`https://datasets-server.huggingface.co/...`).

**LLaMA-Factory version examined:** `0.9.6.dev0` (`LLaMA-Factory/src/llamafactory/extras/env.py:22`), commit `8bf48110e1b86f66e352d10b1a45a6d6e03142e1` ("[ci] Trigger docs workflow for README PRs (#10959)", 2026-11-10). It is a shallow clone: `git log` shows one commit. All repo paths below are relative to `LLaMA-Factory/`.

---

## TL;DR and recommendations

- **Recommended models.** All three are non-Chinese-origin and supported in this version.
  1. **Gemma 4 12B**: `google/gemma-4-12B-it`, template `gemma4n`. Apache-2.0, ungated, and multilingual (pre-trained on 140+ languages). This is the primary pick for a single 24–48 GB GPU.
  2. **Gemma 4 E4B**: `google/gemma-4-E4B-it`, template `gemma4n`. About 8B parameters including embeddings. Use it for fast iteration or debugging, or if the GPU has about 16–24 GB.
  3. **Ministral 3 8B or 14B**: `mistralai/Ministral-3-{8B,14B}-{Base,Instruct}-2512`, template `ministral3`. Apache-2.0, and the card lists Chinese as a supported language. This is the non-Google alternative. The Instruct checkpoints ship in **FP8**, so check how LoRA behaves on them, or use the BF16 Base checkpoints.
  - Larger option for a GPU with 48 GB or more: **Mistral Small 3.1/3.2 24B** (template `mistral_small`).
  - Fallback if Gemma 4 has problems: **Gemma 3 12B** (`google/gemma-3-12b-it`, template `gemma3`). It is gated under the Gemma terms.
  - **Llama 3.2 Vision is not recommended.** Its card says "for image+text applications, English is the only language supported".
- **Data format.** Use the **sharegpt** format: a `messages` list plus an `images` list of local file paths. The number of `<image>` placeholders in the messages must equal `len(images)`. Text-only rows (empty `images`) can be mixed into the same dataset. Relative image paths are resolved against `media_dir`, which defaults to `dataset_dir`. This only applies to local-file datasets, not datasets loaded with `hf_hub_url`.
- **Datasets.**
  - The QA dataset has configs `A` (text QA, 4,057 rows), `B` (image QA, 1,926 rows) and `AB` (text+image QA, 1,962 rows). These three are labelled "SFT" on the card. The other two configs, `L2` and `L5` (2,000 rows each), are for retrieval evaluation.
  - **The owner's guess is confirmed**, with one correction: the config is named `AB`, not "A+B". `A` = text-only, `B` = image (+title +caption), `AB` = document + image.
  - Neither dataset embeds images. Both store only Wikimedia `img_url` strings. You must download the images before training, because LLaMA-Factory calls `Image.open()` on string paths and cannot fetch URLs.
  - QA `doc_id` joins to the corpus `id` (spot-checked).
- **Training target.** The SFT configs are built to train an open-book (RAG) answerer: given context (document text and/or image) and a question, generate `answer`. LLaMA-Factory SFT trains that generator. It does **not** train the retriever, which is a separate model and out of scope for LLaMA-Factory SFT.
- **Main gotchas:**
  - **Leakage.** Split train/eval by `doc_id`, not by row. The same document appears in A, B and AB.
  - **Source documents are the retrieval test set.** The QA documents are the test-split documents of the source retrieval dataset.
  - **Gemma 4 thinking.** Gemma 4 templates are reasoning templates, so set `enable_thinking: false`.
  - **No neat packing.** Neat packing is not supported for gemma4.

---

## 1. Supported vision-language models and which are non-Chinese-origin

### How support is defined in this version

- `README.md:285-341`: the "Supported Models" table, which gives model families, sizes and `template` names.
- `src/llamafactory/extras/constants.py`: `register_model_group(..., template=..., multimodal=True)`. These entries define the WebUI model list and the HF IDs.
- `src/llamafactory/data/template.py`: each template's `mm_plugin=get_mm_plugin(...)`. This is where image support actually lives.
- `src/llamafactory/model/model_utils/visual.py:216+`: `_register_composite_model(...)`. This defines which submodules `freeze_vision_tower` and `freeze_multi_modal_projector` act on.

There are 49 `multimodal=True` groups in `constants.py`. Phi-3.5-vision, Phi-4-multimodal, Idefics and SmolVLM are **not** registered (grep of `constants.py`, `template.py` and `mm_plugin.py` found nothing), so they are not supported in this version. The `phi`/`phi4` templates are text-only.

### Non-Chinese-origin VLM candidates

| Family | Template (`template.py` line) | HF IDs registered (`constants.py` line) | Size (params from HF API `safetensors.total`) | License (HF `cardData.license`) | zh / zh-TW notes (model card) |
|---|---|---|---|---|---|
| **Gemma 4** (Google DeepMind) | `gemma4` (L1287) for 26B-A4B and 31B. `gemma4n` (L1313) for E2B, E4B and 12B. Both are `ReasoningTemplate` with `mm_plugin` `gemma4` | `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` (L886). `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-12B-it` (L900) | E4B: 7.996B (4.5B effective). 12B: 11.96B. 26B-A4B: 25.2B total, 3.8B active. 31B: 31.27B | **apache-2.0**, ungated | "multilingual support in over 140 languages", "Out-of-the-box support for 35+ languages, pre-trained on 140+ languages" ([gemma-4-12B-it card](https://huggingface.co/google/gemma-4-12B-it)). No zh-TW-specific claim. The card footnote "Excluding Chinese language" applies only to the 12B audio benchmarks (CoVoST/FLEURS). Indirect evidence: the mr-right-zhtw corpus text was translated to zh-TW **with Gemma 4** (dataset card, "Translation") |
| **Gemma 3** (Google) | `gemma3` (L1255), plugin `gemma3`, image token `<image_soft_token>` | `google/gemma-3-{4b,12b,27b}-{pt,it}` (L818). The 1B/270M models are text-only | 4.30B / 12.19B / 27.43B | `gemma` (Gemma Terms), **gated (manual)** | "multilingual support in over 140 languages"; "Images, normalized to 896 x 896 resolution and encoded to 256 tokens each"; 128K context for 4B/12B/27B ([Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3)). No zh-TW-specific claim |
| **Gemma 3n** (Google) | `gemma3n` (L1271) | `google/gemma-3n-E2B(-it)`, `google/gemma-3n-E4B(-it)` (L862) | not fetched | Gemma Terms (unverified, card not fetched) | not checked |
| **PaliGemma / PaliGemma 2** (Google) | `paligemma` (L2202) | `google/paligemma-3b-*`, `google/paligemma2-{3b,10b,28b}-{pt,mix}-{224,448,896}` (L2276, L2304) | paligemma2-10b-mix-448: 9.66B | `gemma`, gated | README:212 says these are pre-trained models and "you need to fine-tune them with `paligemma` template for chat completion". This is a weak fit for zh-TW chat QA (my assessment) |
| **Ministral 3** (Mistral AI, France) | `ministral3` (L2148), plugin `pixtral`, image token `[IMG]` | `mistralai/Ministral-3-{3B,8B,14B}-{Base,Instruct}-2512` (L2067) | 8B: 8.92B (8.4B LM + 0.4B vision). 14B: 13.95B | **apache-2.0**, ungated | Card language list includes `zh`; "Supports dozens of languages, including ... Chinese" ([Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)). The Instruct checkpoints are **FP8** ("fitting in 12GB of VRAM in FP8"); the Base checkpoints are BF16 |
| **Mistral Small 3.1 / 3.2** (Mistral AI) | `mistral_small` (L2136), plugin `pixtral` | `mistralai/Mistral-Small-3.1-24B-{Base,Instruct}-2503`, `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (L2114) | 24.01B | **apache-2.0**, ungated | Language list includes `zh`; "fitting within a single RTX 4090 or a 32GB RAM MacBook once quantized" ([3.1 card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)) |
| **Pixtral 12B** (Mistral AI) | `pixtral` (L2272) | `mistral-community/pixtral-12b` (L2438) | ~12B | apache-2.0 | `mistralai/Pixtral-12B-2409` card language list includes `zh`. It is superseded by Ministral 3 / Mistral Small 3.x (my assessment) |
| **Llama 3.2 Vision** (Meta) | `mllama` (L1759), cross-attention plugin | `meta-llama/Llama-3.2-{11B,90B}-Vision(-Instruct)` (L1695) | 11B: 10.67B | `llama3.2` (custom), gated | "**Note for image+text applications, English is the only language supported.**" Chinese is not among the 8 supported text languages ([MODEL_CARD_VISION.md](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md)) |
| **Llama 4** (Meta) | `llama4` (L1737) | `meta-llama/Llama-4-Scout-17B-16E(-Instruct)`, `Llama-4-Maverick-17B-128E(-Instruct)` (L1719) | 109B / 402B total (README:312) | Llama 4 license | Too large for a single workstation |
| **LLaVA-1.5 / LLaVA-NeXT** (llava-hf) | `llava` (L1828), `llava_next_mistral` (L1882), `llava_next_llama3` (L1852) | `llava-hf/llava-1.5-{7b,13b}-hf`, `llava-hf/llava-v1.6-vicuna-*`, `llava-v1.6-mistral-7b-hf`, `llama3-llava-next-8b-hf` (L1743-1798) | llama3-llava-next-8b: 8.36B | llama3 (for the llama3 variant) | Card language: `en` only. Older generation (my assessment). **Avoid** `llava_next_yi` (Yi base) and `llava_next_qwen` (Qwen base), which are Chinese-origin bases |
| **Granite Vision 3.2** (IBM) | `granite3_vision` (L1492) | `ibm-granite/granite-vision-3.2-2b` (L1212) | 2.98B | apache-2.0 | "Currently the model supports English instructions" (card) |
| **LFM2.5-VL** (Liquid AI) | `lfm2_vl` (L1670) | `LiquidAI/LFM2.5-VL-1.6B` (L1548) | 1.60B | `other` / `lfm1.0` | Card: "Improved multilingual vision understanding in Arabic, Chinese, ..." Tiny; possibly useful as a smoke-test model |

**Excluded as Chinese-origin.** All of the following are registered as multimodal but excluded under the project constraint: Qwen2-VL/2.5-VL/3-VL/3.5/Omni/Audio, InternVL, Intern-S1, GLM-4.1V/4.5V/4.6V/GLM-OCR, MiniCPM-V/o, Yi-VL, Kimi-VL (Moonshot), Keye-VL (Kuaishou), MiMo-VL (Xiaomi), ERNIE-4.5-VL (Baidu), dots.ocr (rednote), Youtu-VL (Tencent), MOSS-VL (OpenMOSS), Video-LLaVA (LanguageBind), and `llava_next_qwen`/`llava_next_yi`.

### VRAM implications

The only source in the repo is the README "Hardware Requirement" table (`README.md:504-515`, marked "*estimated*"):
- Freeze/LoRA in 16-bit: about `2x` GB (7B → 16GB, 14B → 32GB, 30B → 64GB).
- QLoRA 8-bit: about `x` GB.
- QLoRA 4-bit: about `x/2` GB (7B → 6GB, 14B → 12GB, 30B → 24GB).

Here `x` is the parameter count in billions. Applying the table (derived, not measured; it excludes activations from image tokens and long documents, which matter here):

| Model | LoRA bf16 (~2x) | QLoRA 4-bit (~x/2) |
|---|---|---|
| Gemma 4 E4B (8B total incl. per-layer embeddings) | ~16 GB | ~4–6 GB |
| Gemma 4 12B / Gemma 3 12B | ~24 GB | ~6 GB |
| Ministral 3 14B | ~28 GB | ~7 GB |
| Mistral Small 3.x 24B | ~48 GB | ~12 GB |
| Gemma 4 26B-A4B (MoE: all 25.2B weights must be resident even though 3.8B are active) | ~50 GB | ~13 GB |

QLoRA is enabled with `quantization_bit: 4` and `quantization_method: bnb` (`examples/train_qlora/qwen3_lora_sft_otfq.yaml:3-4`; field definitions at `src/llamafactory/hparams/model_args.py:279-286`). It needs `bitsandbytes` (`requirements/bitsandbytes.txt`).

### Model-specific notes from source

- **Gemma 4 thinking mode.** `gemma4` and `gemma4n` are `ReasoningTemplate`s with `thought_words=("<|channel>thought\n", "<channel|>")` and a system slot containing `<|think|>` (`template.py:1287-1334`). `enable_thinking` defaults to `True` (`hparams/data_args.py:124-127`). With that default, an **empty CoT is prepended to every response and included in the loss** (`template.py:540-551`; `data/README.md:92`). Our data has no CoT, so set `enable_thinking: false` in both training and inference.
- **Gemma 4: no neat packing.** `"Neat packing is not supported for gemma4, gpt_oss models for now."` (`data/collator.py:513-514`).
- **Gemma 4 freeze keys.** `projector_keys=["model.embed_vision","model.embed_audio"]`, `vision_model_keys=["vision_tower","audio_tower"]` (`model/model_utils/visual.py:237-242`). The 12B model is "encoder-free" ("projecting raw image patches ... directly into the LLM's embedding space", per the card), so `freeze_vision_tower` probably has little to freeze on 12B (inference from the card plus the keys; unverified).
- **Mistral 3 / Ministral 3.** `mistral3` freezes `model.multi_modal_projector` (`visual.py:350-353`). Whether LoRA trains cleanly on the FP8 Instruct weights is **unverified**.
- **Transformers version.** The pinned range is `transformers>=4.55.0,<=5.8.0,!=4.57.0,!=5.6.0` (`pyproject.toml`, `src/llamafactory/extras/misc.py:114`). Which minimum transformers version provides Gemma 4 / Ministral 3 classes is **unverified**. The Gemma 4 card only says "the latest version of Transformers".

---

## 2. Multimodal dataset format LLaMA-Factory expects

### Formats

- Supported `formatting` values: `alpaca` (the default), `sharegpt` and `openai` (`src/llamafactory/data/parser.py:33`; `data/converter.py:370-374`).
- Allowed file types: json, jsonl, csv, parquet, arrow (`data/README.md:5`).
- The docs say multimodal image datasets should follow sharegpt: "An additional column `images` is required. Please refer to the sharegpt format" (`data/README.md:156-158`). In code, the alpaca converter also passes `_images` through (`converter.py:127`), so alpaca with an `images` column works too. sharegpt is the documented path.

### Canonical sharegpt multimodal row

From `data/README.md:310-349`:

```json
[{"conversations": [{"from": "human", "value": "<image>user instruction"},
                    {"from": "gpt", "value": "model response"}],
  "images": ["image path (required)"]}]
```
```json
"dataset_name": {"file_name": "data.json", "formatting": "sharegpt",
  "columns": {"messages": "conversations", "images": "images"}}
```

The repo's own demo uses `role`/`content` keys instead of `from`/`value`, which requires `tags` (`data/dataset_info.json:27-40`, `mllm_demo`):

```json
"mllm_demo": {"file_name": "mllm_demo.json", "formatting": "sharegpt",
  "columns": {"messages": "messages", "images": "images"},
  "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
```

The rows in `data/mllm_demo.json` look like `{"messages":[{"role":"user","content":"<image>Who are they?"},...], "images":["mllm_demo_data/1.jpg", ...]}`. Images are stored under `data/mllm_demo_data/`.

### `dataset_info.json` fields

All from `data/README.md:7-45`:
- **Source:** `hf_hub_url`, `ms_hub_url`, `script_url`, `cloud_file_name`, or `file_name`.
- **Other top-level fields:** `formatting`, `ranking`, `subset`, `split` (default `train`), `folder`, `num_samples`.
- **`columns`:** `prompt`, `query`, `response`, `history`, `messages` (default `conversations`), `system`, `tools`, `images`, `videos`, `audios`, `chosen`, `rejected`, `kto_tag`.
- **`tags` (sharegpt only):** `role_tag` (default `from`), `content_tag` (`value`), `user_tag` (`human`), `assistant_tag` (`gpt`), `observation_tag`, `function_tag`, `system_tag`.

`dataset_info.json` must be in `dataset_dir` (default `./data`). `dataset_dir` can also be `REMOTE:<hf repo>` or `ONLINE` (`parser.py:100-121`).

### The `<image>` placeholder

- `IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")` (`extras/constants.py:52`). It can be overridden with an environment variable.
- Strict count check: `"The number of images does not match the number of <image> tokens"` is raised if `len(images)` differs from the total count of `<image>` across all messages (`data/mm_plugin.py:190-217`). Watch for literal `<image>` text inside documents.
- Each plugin replaces `<image>` with the model's own token. For example, Gemma 3 uses `boi_token`/`full_image_sequence` (`mm_plugin.py:876-914`) and Gemma 4 uses `<|image|>` (`template.py:1303-1307`).

### How image paths are resolved

- `media_dir`: "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`." (`hparams/data_args.py:42-44`, defaulted at `:165-166`).
- `DatasetConverter._find_medias` (`data/converter.py:43-77`) joins `os.path.join(media_dir, path)` and uses that path if the file exists. Otherwise it logs `"Media ... does not exist in media_dir. Use original path."` and keeps the original string, so absolute paths also work. **This only happens when `load_from in ["script", "file"]`** (`converter.py:54`). With `hf_hub_url`, path strings are used as-is.
- Accepted image values are `str | bytes | {"path","bytes"} | BinaryIO | PIL.Image` (`mm_plugin.py:61-65`). Strings are opened with `Image.open(image)` (`mm_plugin.py:258-276`), so **URLs are not supported**. HF `Image()` features (`{"bytes","path"}` dicts) are accepted.
- Images are resized when `width*height > image_max_pixels`, using `image_max_pixels` and `image_min_pixels` (`mm_plugin.py:219-230`; defaults `768*768` and `32*32`, `hparams/model_args.py:305-312`).

### Multiple images per sample and mixing text-only rows

- **Multiple images per sample:** yes. `images` is a list. The first `mllm_demo.json` row has two `<image>` placeholders and two image paths across two turns.
- **Text-only rows in the same dataset:** yes. An empty list or `None` becomes `None` (`converter.py:45-50`). If an entire batch has no images, the collator injects a 64×64 white fake image "to avoid process hanging in zero3/fsdp case" (`data/collator.py:355-374`).
- The official multimodal example mixes datasets: `dataset: mllm_demo,identity,alpaca_en_demo` (`examples/train_lora/qwen3vl_lora_sft.yaml:15`). `identity` and `alpaca_en_demo` are text-only.

---

## 3. Structure of the two HF datasets

### 3a. `ericssonbear/mr-right-zhtw` (corpus/database)

Sources: [dataset card](https://huggingface.co/datasets/ericssonbear/mr-right-zhtw), `/info` and `/splits` from datasets-server, HF API (sha `4bb7b3ec...`, last modified 2026-08-10, not gated, CC BY-SA 4.0).

| config | split | rows (`/info`) | content |
|---|---|---:|---|
| `corpus` | train | 769,245 | documents: 8 parquet shards, download 582 MB, dataset size 1.78 GB |
| `queries-test` | test | 2,047 | test queries |
| `queries-val` | validation | 100 | |
| `queries-finetune` | train | 900 | human-annotated train split; includes `*_zhtw_hv` columns |
| `queries-test-hv` | test | 1,100 | human-verified zh for the test queries |
| `queries-mt-en` | train | 2,147 | MT English |
| `pseudo-pairs-zh` | train | 32,359 | pseudo-labelled pairs |
| `pseudo-pairs-hard` | train | 27,055 | hard-negative pairs with `judge_score` |

**`corpus` columns** (`/first-rows` features): `id` (int64), `title`, `title_zhtw`, `doc_text`, `doc_text_zhtw`, `img_caption`, `img_url`, `caption_source`, `sha1`, `image_path`.
- `id` is "upstream Mr. Right document id — non-contiguous (37,112 of 806,357 dropped)".
- `image_path` is `<sha1[:2]>/<sha1>.<ext>`. Example row: `id: 0`, `image_path: "9e/9ef9d7355134314d792fe8498295cd09a6124bb6.jpg"`.

**Images.** "**Images are not redistributed here.** Every row carries `img_url` (Wikimedia) and `sha1`". The full tree is "~1.2 TB / 769,217 files". Wikimedia rate-limits to "roughly 1 image/s per egress IP" and requires a descriptive User-Agent (card, "Getting the images").

**`img_caption`.** It is "not a one-line caption": "median 663 characters, with markdown section headers (`**主體：**`, `**場景與背景：**`)". 82% contain `**` markup.

**Provenance** (card, "How it was built"):
- `doc_text_zhtw`: machine-translated by **Gemma 4**, then normalised for Taiwan usage.
- Captions: `own` captions (187,057) were generated with **Qwen3-VL-8B-Instruct**; `external` captions (582,188) came from other lab members' VLM pipelines.
- Queries: English queries are human-written (MTurk); `*_zhtw` is Gemma 4 MT.

**Queries schema.** `query_img` / `query_text` / `query_multi`, their `_zhtw` counterparts, `id` (the gold document), `title`, `title_zhtw`. "`id` joins to `corpus.id`; relevance is one gold document per query." The card warns not to feed the query image with `query_multi` (the "image trap").

### 3b. `ysh1257/multimodal-ir-zh-tw-qa` (QA)

Sources: [dataset card](https://huggingface.co/datasets/ysh1257/multimodal-ir-zh-tw-qa), `/info`, `/first-rows`, `/statistics`, HF API (sha `2444b4db...`, last modified 2026-11-05, not gated, CC BY-SA 4.0, about 15.6 MB total).

Per the card, the QA pairs are "generated from the documents in [Sigoso12/multimodal-ir-zh-tw]". The generator was `claude-haiku-5-5` for every row (`/statistics` `model` frequencies). The linked code repo `github.com/1-xingcys/irlab-multi-rag-data` returned 404 from the GitHub API, so the generation prompts are **unverified**.

| config | card purpose | context given to the generator (card) | docs | rows (`/info`) |
|---|---|---|---:|---:|
| `L2` | retrieval evaluation | zh document + zh title | 2000 | 2000 |
| `L5` | retrieval evaluation | zh doc + zh title + English title/text + image + existing queries as style reference | 2000 | 2000 (`image_missing`: 37 true, per `/statistics`) |
| `A` | SFT | "text QA for SFT: zh document + zh title + English title/text; 2 questions per document" | 2043 | 4057 |
| `B` | SFT | "image QA for SFT: zh title + image + image caption; the answer is visible in the image" | 1926 | 1926 |
| `AB` | SFT | "text + image QA for SFT: zh document + zh title + English title/text + image; the answer uses both the document and the image" | 1962 | 1962 |

**Columns** (`/info`, identical across the SFT configs):
- `query`, `answer`, `evidence`, `evidence_image`, `qtype`, `answer_short`, `doc_id` (int64), `model`, `level`, `image_missing` (bool).
- `ground_truth_doc` is a struct `{id, title, text, img_url, img_caption}`. In `L2`/`L5` it is only `{id, title, text}`.
- Each config has a single `train` split, so there is no predefined eval split.

**`qtype` distributions** (`/statistics`):
- `A`: fact 1327, unanswerable 614, compose 605, explain 557, describe 408, list 341, reverse 205.
- `B`: appearance 819, detail 434, overview 382, unanswerable 291.
- `AB`: integrate 1050, visual_lookup 592, unanswerable 320.
- Unanswerable rows teach the reply 「提供的資料沒有提到……」 (card).

**Lengths in characters** (`/statistics` mean / median):

| | `answer` | `query` |
|---|---|---|
| A | 52.8 / 49 | 22.5 / 19 |
| B | 53.7 / 49 | 20.0 / 19 |
| AB | 77.4 / 77 | 28.3 / 19 |

**Images.** No image bytes are stored. `ground_truth_doc.img_url` is a Wikimedia URL (for example `https://upload.wikimedia.org/wikipedia/commons/0/05/Nikolay_Nikolaevich_Zubov_02.jpg`). The Hub API tags list only `modality:text`.

### What A / B / AB mean: confirmed

The owner's hypothesis is **confirmed by the dataset card and the data**, with these refinements:
- **`A` = text-only QA.** The context is the zh document, zh title and English text. `evidence_image` is empty for all 4,057 rows (`/statistics`: no non-empty values).
- **`B` = image QA.** The context is zh title + image + **image caption**, not the document body. `evidence` is quoted "from the image caption" (card). `evidence_image` is always non-empty (min length 14). Example (B, `doc_id` 368508): query 「長春大房身機場航廈屋頂上的金色大字是什麼？」 → answer 「是「長春」兩個字。…」.
- **`AB` = document text + image QA.** "the answer uses both the document and the image". Example: 「我看過一種豬籠草，瓶身是綠底帶咖啡紅褐色直條紋…，它原產於哪個島嶼？」 (`qtype` `visual_lookup`).
- The config is named **`AB`**, not "A+B". `L2`/`L5` are a separate retrieval-evaluation family and are not SFT data.

### How QA rows reference corpus documents

- `doc_id` is "`id` in the source dataset" (card), and `ground_truth_doc.id == doc_id` in every sampled row.
- The source, `Sigoso12/multimodal-ir-zh-tw`, has a 769,245-record document file, the same count as the `mr-right-zhtw` corpus ([Sigoso12 card](https://huggingface.co/datasets/Sigoso12/multimodal-ir-zh-tw)).
- **Spot-check via datasets-server `/filter`:**
  - QA `doc_id` 368508 matches `mr-right-zhtw` `corpus` `id` 368508 with an identical `img_url` (`image_path` `c6/c621d5b0b98069c51b411b405994cea032ff5853.jpg`).
  - QA `doc_id` 568189 matches `queries-test` `id` 568189 with an identical `img_url`.
  - Two further lookups failed with transient datasets-server errors, so only 2 IDs were verified, not all.
- Conclusion: QA `doc_id` matches `mr-right-zhtw` `corpus.id` and `queries-test.id`. This is verified for the two sampled IDs; full coverage is **unverified**.
- The QA card says the source documents "are the test split of Sigoso12/multimodal-ir-zh-tw: if you train on these QA, evaluate on other documents". The `doc_id` range is 196–806138 (`/statistics`).
- Because `ground_truth_doc` already embeds title, text, `img_url` and `img_caption`, **the corpus join is only needed for**:
  - `sha1` (image integrity check) and `image_path`;
  - distractor/negative documents if we simulate retrieval;
  - English fields (`doc_text`).

---

## 4. Conversion to LLaMA-Factory format

### Proposed mapping (one sharegpt JSON or JSONL per config, or one merged file)

| LLaMA-Factory field | `A` (text) | `B` (image) | `AB` (text+image) |
|---|---|---|---|
| `messages[0]` (role `user`) | context block (`ground_truth_doc.title` + `ground_truth_doc.text`) + `query` | `"<image>"` + optional `ground_truth_doc.title` + `query` | `"<image>"` + `title` + `text` + `query` |
| `messages[1]` (role `assistant`) | `answer` | `answer` | `answer` |
| `images` | `[]` (text-only) | `[<local path of downloaded ground_truth_doc.img_url>]` | same as B |
| `system` (optional column) | fixed zh-TW RAG instruction, e.g. answer only from the provided material and reply 「提供的資料沒有提到……」 otherwise | same | same |
| kept as metadata, not trained | `doc_id`, `qtype`, `answer_short`, `evidence`, `evidence_image`, `level` | same | same |

Example `dataset_info.json` entry. The file lives in our own `dataset_dir`, and paths are relative to `media_dir`:

```json
"mmir_zhtw_sft": {
  "file_name": "mmir_zhtw_sft.jsonl",
  "formatting": "sharegpt",
  "columns": {"messages": "messages", "images": "images", "system": "system"},
  "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
}
```

### What the converter must do

1. Load `A`, `B` and `AB` (and not `L2`/`L5`) with `datasets` on the workstation.
2. Collect the unique `ground_truth_doc.img_url` values for B and AB. At most about 2,000 unique docs, because all three configs draw on the same ~2,000 test documents; exact unique count unverified.
3. Download each image with a real User-Agent, honouring `Retry-After`. At about 1 image/s that is under an hour (derived from the corpus card's rate-limit note).
4. Optionally join to the corpus on `doc_id` to get `sha1` and verify the bytes. Store images as `<sha1[:2]>/<sha1>.<ext>` to match the corpus `image_path` layout.
5. Drop or flag rows whose image fails to download. An image-less B/AB row would have a `<image>` placeholder with no image and would fail the count check.
6. Escape or strip any literal `<image>` text in documents and captions (`mm_plugin.py:190-217`).
7. Do a **document-level split** for eval: hold out a set of `doc_id`s across A, B and AB together. Do not use `val_size`, which splits by row: A has 2 questions per document, and the same document appears in A, B and AB.
8. Write local JSON/JSONL plus a `dataset_info.json` entry, and point `dataset_dir`/`media_dir` at them. Do not use `hf_hub_url`: `media_dir` resolution only applies to local files (`converter.py:54`), and the HF rows carry URLs, not files.

### Gotchas and the training-target question

- **Retrieval vs generation.** The project and datasets are "multimodal IR". The QA card, however, labels `A`/`B`/`AB` as "Training data for an open-book (RAG) answerer, a VLM for `B` and `AB`". LLaMA-Factory SFT trains a **generator**, so the natural training target is `answer`: given the gold context plus the question, produce the answer. Retrieval (finding the document) needs a separate embedding model.
  - The lab's released retriever, `ericssonbear/qwen3-vl-emb-8b-mrright-zhtw-lora`, is Qwen-based, which conflicts with the no-Chinese-origin constraint.
  - `L2`/`L5` exist to evaluate retrieval, not the generator.
- **Gold-only context vs realistic RAG context.** Training only on the gold document teaches answering but not ignoring irrelevant passages. One option is to add hard-negative documents from the corpus (for example via `pseudo-pairs-hard`) as distractors. This is a design decision; see the open questions.
- **Should B include `img_caption`?** The B generator saw the caption, and `evidence` is quoted from it. If the caption is put in the prompt, the model can learn to read the caption instead of the image. The card's intent ("the answer is visible in the image") suggests image + title only.
- **Thinking templates.** For Gemma 4, set `enable_thinking: false` (see §1). Otherwise empty thought tokens become part of the learned response.
- **`cutoff_len`.** `img_caption` has a median of 663 characters (corpus card), document texts run to several hundred characters, and each image costs model-specific tokens (Gemma 3: 256 tokens per image, per its model card). The default `cutoff_len` is 2048 (`hparams/data_args.py:46-49`). Measure token lengths after conversion and raise it if needed.
- **Evaluation leakage.** The QA documents are the retrieval *test* documents. The QA card says to evaluate on other documents. Generator eval should use held-out `doc_id`s, and any retrieval evaluation on `queries-test` would overlap with these training documents.
- **Data provenance vs the "no Chinese-origin models" rule.**
  - Corpus captions were generated by Qwen3-VL (`own`) and by unspecified lab VLMs (`external`).
  - B's `evidence` and answers draw on those captions.
  - Corpus and query translations come from Gemma 4; the QA was generated by `claude-haiku-5-5`.
  - The constraint as stated applies to the model being fine-tuned. Whether it also covers training-data provenance is an owner decision.
- **Licensing.** Both datasets are CC BY-SA 4.0. The images are not covered: "Individual images remain under their own Wikimedia licenses" (corpus card).

---

## 5. Training configuration

### Reference example

The only multimodal LoRA SFT example in this version is `examples/train_lora/qwen3vl_lora_sft.yaml`. It is Qwen-based, so use it only as a structural template. Other multimodal examples: `examples/train_lora/qwen3vl_lora_dpo.yaml`, `examples/train_full/qwen3vl_full_sft.yaml`, and `examples/train_lora/mossvl_lora_sft.yaml`. `examples/README.md:52-55` documents `llamafactory-cli train examples/train_lora/qwen3vl_lora_sft.yaml`. There is no Gemma/Mistral multimodal example YAML in `examples/`.

Key lines (`qwen3vl_lora_sft.yaml:1-46`):
- `image_max_pixels: 262144`, `video_max_pixels: 16384`, `trust_remote_code: true`
- `stage: sft`, `finetuning_type: lora`, `lora_rank: 8`, `lora_target: all`
- `dataset: mllm_demo,identity,alpaca_en_demo`, `template: qwen3_vl_nothink`, `cutoff_len: 2048`, `preprocessing_num_workers: 16`, `dataloader_num_workers: 4`
- `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 8`, `learning_rate: 1.0e-4`, `num_train_epochs: 3.0`, `lr_scheduler_type: cosine`, `warmup_ratio: 0.1`, `bf16: true`
- eval is commented out (`val_size`, `eval_strategy`, ...)

### Multimodal-specific hyperparameters

| field | default | source | meaning |
|---|---|---|---|
| `freeze_vision_tower` | `True` | `hparams/finetuning_args.py:569-572` | exclude vision-tower modules from training/LoRA targets |
| `freeze_multi_modal_projector` | `True` | `finetuning_args.py:573-576` | exclude projector modules |
| `freeze_language_model` | `False` | `finetuning_args.py:577-580` | exclude the LLM |
| `image_max_pixels` / `image_min_pixels` | `768*768` / `32*32` | `hparams/model_args.py:305-312` | resize bounds applied by `_preprocess_image` |
| `image_do_pan_and_scan` | `False` | `model_args.py:313-316` | Gemma 3 pan-and-scan crops |
| `crop_to_patches` | `False` | `model_args.py:317-320` | InternVL only |
| `media_dir` | `dataset_dir` | `hparams/data_args.py:42-44,165-166` | image root |
| `enable_thinking` | `True` | `hparams/data_args.py:124-127` | set to `false` for Gemma 4 with no-CoT data |
| `quantization_bit` / `quantization_method` | `None` / `bnb` | `model_args.py:279-286` | QLoRA |

The freeze flags become LoRA target exclusions in `patch_target_modules` and `get_forbidden_modules` (`model/model_utils/visual.py:173-213`), using per-model keys from `_register_composite_model`. The defaults are `multi_modal_projector`, `vision_tower` and `language_model`/`lm_head` (`visual.py:68-91`).

Sketch for the primary candidate. This is derived from the example plus the source defaults and has not been run:

```yaml
model_name_or_path: google/gemma-4-12B-it
image_max_pixels: 589824        # 768*768 default; tune for VRAM
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
freeze_vision_tower: true
freeze_multi_modal_projector: true
dataset: mmir_zhtw_sft
dataset_dir: data_mmir          # holds dataset_info.json
media_dir: data_mmir/images
template: gemma4n
enable_thinking: false
cutoff_len: 4096
# quantization_bit: 4           # QLoRA if VRAM-limited
bf16: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
```

### Dependencies for this version

There is **no top-level `requirements.txt`**. Core dependencies are in `pyproject.toml` `[project].dependencies` (lines 38-87):

- **Core:**
  - `torch>=2.4.0`, `torchvision>=0.19.0`, `torchaudio>=2.4.0`
  - `transformers>=4.55.0,<=5.8.0,!=4.57.0,!=5.6.0`
  - `datasets>=2.16.0,<=4.0.0`, `accelerate>=1.3.0,<=1.11.0`
  - `peft>=0.18.0,<=0.18.1`, `trl>=0.18.0,<=0.24.0`, `torchdata>=0.10.0,<=0.11.0`
- **GUI:** `gradio>=4.38.0,<=5.50.0`, `matplotlib>=3.7.0`, `tyro<0.9.0`
- **Other:** `einops`, `numpy`, `pandas`, `scipy`, `sentencepiece`, `tiktoken`, `modelscope`, `hf-transfer`, `safetensors`, `av>=10.0.0,<=16.0.0`, `fire`, `omegaconf`, `packaging`, `protobuf`, `pyyaml`, `pydantic`, `uvicorn`, `fastapi`, `sse-starlette`
- `requires-python = ">=3.11.0"`

Optional files in `requirements/`:
- `bitsandbytes.txt`: `bitsandbytes>=0.39.0` (needed for QLoRA)
- `deepspeed.txt`: `deepspeed>=0.10.0,<=0.18.4`
- `metrics.txt`: `nltk`, `jieba`, `rouge-chinese` (needed for `predict_with_generate` BLEU/ROUGE)
- `liger-kernel.txt`: `liger-kernel>=0.6.3`

Install per the README (`README.md:529-533`): `pip install -e .` then `pip install -r requirements/metrics.txt` (plus `-r requirements/deepspeed.txt` if needed).

The README "Requirement" table (`README.md:484-502`) lists older minimums (for example python 3.11, transformers 4.49.0 minimum) that are lower than the pins in `pyproject.toml`. Treat `pyproject.toml` as authoritative.

---

## Open questions for the owner

1. **Model choice.** Gemma 4 12B (recommended), Gemma 4 E4B (cheaper), or Ministral 3 8B/14B (non-Google, FP8 Instruct weights)? This depends on the remote GPU's VRAM. **What GPU and VRAM does the workstation have?**
2. **Training target.** Confirm the goal is an open-book answerer (context + question → `answer`) rather than a retriever. If retrieval is also required, it needs a separate non-Chinese embedding model, outside LLaMA-Factory SFT.
3. **Context design.**
   - Gold document only, or gold document plus retrieved distractors from the corpus?
   - For B, include `img_caption` in the prompt or not?
   - Include the English `title`/`doc_text`?
4. **Which configs.** Train on `A` + `B` + `AB` mixed (about 7,945 rows), or a subset? Should `answer_short` be used only for EM/F1 evaluation?
5. **Evaluation split.** How many `doc_id`s should be held out? Is there another document set for final evaluation? (The QA card advises evaluating on other documents.)
6. **Data provenance policy.** Does "no Chinese-origin models" also exclude training data generated by Qwen3-VL (the corpus `own` captions, which B answers depend on)?
7. **Image access.** Can the workstation reach `upload.wikimedia.org`, and what User-Agent/contact should be used for the download?
8. **Thinking mode.** Is plain answering (`enable_thinking: false`) acceptable, or should reasoning traces be generated?

### Unverified items
- Minimum transformers version that supports Gemma 4 / Ministral 3.
- LoRA on Ministral 3 FP8 Instruct weights.
- Whether `freeze_vision_tower` has any effect on encoder-free Gemma 4 12B.
- Exact count of unique documents and images across A/B/AB.
- That every QA `doc_id` exists in the `mr-right-zhtw` corpus (only 2 IDs were spot-checked).
- The QA generation code and prompts (the GitHub repo was not reachable).
- Gemma 3n and PaliGemma card details (gated cards not fetched).
