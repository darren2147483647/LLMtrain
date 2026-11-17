# 工作紀錄

## 2026-11-11

### 目標

用 LLaMA-Factory 微調一個繁體中文多模態問答模型：給模型一份資料（文字、圖片或兩者）和一個問題，模型只根據這份資料回答；資料中沒有答案時要明確拒答。模型不需要檢索能力。本地只負責建構程式和設定，實際訓練在遠端工作站執行。

### 已確定的決定

| 項目 | 決定 |
|---|---|
| 模型 | Gemma 4 E4B（`google/gemma-4-E4B-it`）。原本選 Gemma 4 12B，因 transformers 不支援而更換（見下方「下午」段落）。之後可能換成更大的模型，只需改設定檔 |
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
| [#3](https://github.com/darren2147483647/LLMtrain/issues/3) | 圖片下載＋B / AB 組資料準備 | #2 | 完成，待合併 |
| [#4](https://github.com/darren2147483647/LLMtrain/issues/4) | 評估評分 | #2 | 完成，待合併 |
| [#5](https://github.com/darren2147483647/LLMtrain/issues/5) | 冒煙測試訓練＋checkpoint 保存 | #3 | 程式和設定完成，待工作站執行 |
| [#6](https://github.com/darren2147483647/LLMtrain/issues/6) | 預測＋評估串接（基準流程） | #4、#5 | 程式和設定完成，待工作站執行 |
| [#7](https://github.com/darren2147483647/LLMtrain/issues/7) | 主訓練（改為 Gemma 4 E4B）、匯出、操作說明 | #6 | 程式、設定和操作說明完成，待工作站執行 |

（狀態更新於當天稍晚，見下方「下午」段落。）

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

（已完成，見下方「下午」段落。）

### 下午：#3–#7

**模型改為 Gemma 4 E4B。** 實作時查證發現，`google/gemma-4-12B-it` 的模型類型是 `gemma4_unified`，transformers 要到 main 分支才有，而 LLaMA-Factory（鎖定版本和最新版都一樣）只允許 transformers 到 5.8.0，也還沒有支援這個類型。Gemma 4 的 E4B、26B-A4B、31B 是 `gemma4` 類型，transformers 5.5.0 起就支援。主模型因此改用 E4B，之後要升級時，可以換成同類型的 26B-A4B 或 31B，只需要改設定檔。

**完成的內容：**

- **圖片下載（#3）：**
  - 從維基下載 B 和 AB 組的圖片，每秒約 1 張。
  - 遇到限流時會依伺服器要求的時間等待後重試。
  - 中斷後重跑時，已經下載的圖片會跳過。
  - 下載失敗的資料會被排除並記錄，不影響切分結果。
  - 實際下載過 6 張圖片測試，全部成功。下載的是原圖，B 和 AB 組估計需要 15 GB 以上的空間，目前先假設工作站空間足夠。
- **評估評分（#4）：**
  - **字元級 F1：** 預測與參考答案的字元重疊程度。
  - **要點召回率（`answer_short_recall`）：** 參考要點的字元有多少出現在預測中。原本規格寫的是「`answer_short` 命中率」，但 `answer_short` 是改寫過的要點摘要，連參考答案本身都只有 47% 直接包含它，所以改用字元召回率。參考答案的召回率約 0.98，別題的答案約 0.2，分得出好壞。
  - **拒答率和誤拒答率：** 拒答是依答案開頭的句型判定。用全部 7,945 筆真實答案驗證，1,225 筆無法回答題全部判為拒答；可回答題只有 1 筆被判為拒答，而那一筆的內容其實就是拒答。
  - 所有指標都分組別（A / B / AB）和題型報告。
- **訓練啟動腳本（#5、#6）：** `python -m llmtrain.run train / predict / export`。
  - **train：** 每次訓練會建立一個以模型和時間命名的資料夾，存放 checkpoint，以及當次的設定、資料清單和執行過的指令。
  - **續訓：** 只要再跑一次同一個資料夾，就會從最新的 checkpoint 接著訓練。
  - **predict：** 預測完成後會自動評分。
- **設定檔（#5、#7）：**
  - 冒煙測試用 LFM2.5-VL-1.6B 跑 12 步。
  - 主訓練是 Gemma 4 E4B QLoRA，每 200 步存檔並評估，最多保留 3 個 checkpoint，結束時保留評估 loss 最低的一個。
  - 另有預測和匯出的設定。
- **操作說明：** [RUNBOOK.md](RUNBOOK.md)，涵蓋工作站上從安裝到匯出的每一步。
- **測試：** 共 42 個，在本地全部通過。啟動腳本用假的 `llamafactory-cli` 跑過完整流程。

**還沒驗證、需要在工作站上確認的事：**

1. 從 Hugging Face 讀取資料，以及完整下載所有圖片。
2. 冒煙測試能跑完，checkpoint 的保留數量正確，中斷後能續訓。
3. 預測結果和參考資料能正確對上，評分能產出結果。
4. Gemma 4 E4B 能載入，VRAM 足夠訓練，`cutoff_len` 夠用。

### 下一步

在工作站上照 [RUNBOOK.md](RUNBOOK.md) 依序執行步驟 1 到 3（安裝、準備資料、冒煙測試）。之後再跑基準評估（步驟 4）和主訓練（步驟 5）。
