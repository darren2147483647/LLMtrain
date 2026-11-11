# 工作紀錄

## 2026-11-11

### 目標

用 LLaMA-Factory 微調一個繁體中文多模態問答模型：給模型一份資料（文字、圖片或兩者）和一個問題，模型只根據這份資料回答；資料中沒有答案時要明確拒答。模型不需要檢索能力。本地只負責建構程式和設定，實際訓練在遠端工作站執行。

### 已確定的決定

| 項目 | 決定 |
|---|---|
| 模型 | Gemma 4 12B（`google/gemma-4-12B-it`）。之後可能換成更大的模型，設定會設計成容易更換 |
| 方法 | QLoRA（4-bit 量化，只訓練 LoRA adapter） |
| 不使用的模型 | 中國模型（Qwen、InternVL、GLM 等）不能作為被訓練的模型；訓練資料中有部分內容由 Qwen 生成是允許的 |
| 訓練資料 | `ysh1257/multimodal-ir-zh-tw-qa` 的 A（純文字）、B（圖片）、AB（文字＋圖片）三組，共約 7,945 筆 |
| 無法回答題 | 保留在訓練資料中，讓模型學會拒答 |
| B 組提示詞 | 放標題和圖片說明 |
| 干擾文件 | 暫時不放 |
| 切分 | 訓練／評估／測試＝8/1/1，按文件（`doc_id`）切分，同一份文件不會同時出現在訓練和測試 |
| 依賴管理 | `pyproject.toml`（LLaMA-Factory 本身沒有 `requirements.txt`） |
| 圖片 | 先假設工作站可以存取維基圖片 |

### 研究發現

- A、B、AB 分別是純文字、圖片、文字＋圖片，與原本的猜測一致；第三組的名稱是 `AB`。
- 兩個資料集都沒有圖片檔，只有維基的圖片網址，訓練前要先下載。維基限制大約每秒 1 張。
- B 組全部 1,926 筆都有圖片網址，資料集的 `image_missing` 欄位全部為 `False`，沒有發現漏放圖片的情況。
- 問答資料由 `claude-haiku-5-5` 生成；語料庫的部分圖片說明由 Qwen3-VL 生成。
- Llama 3.2 Vision 只支援英文的圖文輸入，不適用。
- Gemma 4 預設會開啟「思考模式」，訓練時必須關閉，否則每個答案前都會被加上空的思考標記。

詳細的來源和比較記錄在 [research/llamafactory-multimodal-zhtw.md](research/llamafactory-multimodal-zhtw.md)。

### 規格和工作項目

完整的訓練計畫寫在 GitHub issue [#1](https://github.com/darren2147483647/LLMtrain/issues/1)，包含模型、方法、訓練架構、各組別的輸入與輸出欄位、checkpoint 保存方式和預期結果。

計畫拆成 6 個工作項目：

| Issue | 內容 | 需要先完成 | 狀態 |
|---|---|---|---|
| [#2](https://github.com/darren2147483647/LLMtrain/issues/2) | 專案骨架＋A 組資料準備 | — | 完成，待合併 |
| [#3](https://github.com/darren2147483647/LLMtrain/issues/3) | 圖片下載＋B / AB 組資料準備 | #2 | 未開始 |
| [#4](https://github.com/darren2147483647/LLMtrain/issues/4) | 評估評分 | #2 | 未開始 |
| [#5](https://github.com/darren2147483647/LLMtrain/issues/5) | 冒煙測試訓練＋checkpoint 保存 | #3 | 未開始，需要在工作站執行 |
| [#6](https://github.com/darren2147483647/LLMtrain/issues/6) | 預測＋評估串接（基準流程） | #4、#5 | 未開始 |
| [#7](https://github.com/darren2147483647/LLMtrain/issues/7) | Gemma 4 12B 主訓練、匯出、操作說明 | #6 | 未開始 |

### 完成的程式（#2）

分支 `issue-2-a-data-prep`：

- `pyproject.toml`：專案依賴。LLaMA-Factory 從官方 GitHub 安裝，鎖定 commit `8bf4811`，所以工作站不需要本地的 LLaMA-Factory 副本。
- `src/llmtrain/prompts.py`：system 指令和提示詞格式，訓練和評估共用。
- `src/llmtrain/split.py`：按文件切分 8/1/1，固定隨機種子。
- `src/llmtrain/prepare.py`：資料準備。目前處理 A 組，輸出 LLaMA-Factory 可以直接讀取的訓練、評估、測試資料，以及評估用的 metadata 和記錄資料版本、切分結果的清單。
- `tests/`：7 個自動化測試，使用假資料，不需要網路，在本地 conda 環境 `llm-factory`（只有 Python 3.11 和 pytest）中全部通過。

另外用 300 筆真實的 A 組資料跑過一次資料準備，輸出正常。

### 其他調整

- 本地的 `LLaMA-Factory/` 資料夾不再被 git 追蹤，只作為查閱原始碼的參考。
- 文件放置原則：給 agent 讀的文件放在 `docs/`（不推送）；給人讀的文件（例如這份紀錄、之後的工作站操作說明）放在會推送到 GitHub 的位置。

### 待確認

- 用 `datasets` 從 Hugging Face 讀取資料的部分還沒有實際執行過（本地沒有安裝 `datasets`），要在工作站第一次執行時確認。
- 遠端工作站的 GPU 規格。

### 下一步

#3（圖片下載＋B / AB 組資料準備）和 #4（評估評分）可以同時進行。
