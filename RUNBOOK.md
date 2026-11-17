# 工作站操作說明

在遠端工作站上，從安裝到匯出模型的完整步驟。所有指令都在 repo 根目錄執行。訓練和預測的輸出會放在 `runs/`，資料放在 `data/mmqa/`，這兩個資料夾都不會被 git 追蹤。

## 0. 需求

- Linux 和 NVIDIA GPU。Gemma 4 E4B 用 QLoRA 4-bit 訓練時，粗估需要 6 GB 以上的 VRAM；圖片和長文件會增加用量，實際需求以步驟 3 冒煙測試量到的為準。
- Python 3.11 以上。
- 磁碟空間：圖片約 15 GB 以上（下載的是維基原圖）、模型快取約 20 GB、checkpoint 每個數百 MB。
- 可以連到 `huggingface.co` 和 `upload.wikimedia.org`。

## 1. 安裝

```bash
git clone https://github.com/darren2147483647/LLMtrain.git
cd LLMtrain
conda create -n llmtrain python=3.11 -y
conda activate llmtrain
# 先依工作站的 CUDA 版本安裝 torch，見 https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[test]"
python -m pytest -q   # 應該全部通過
```

`pip install -e .` 會依 `pyproject.toml` 安裝 LLaMA-Factory，版本固定在 commit `8bf4811`，另外還會裝 transformers 5.5–5.8、bitsandbytes 等套件。

## 2. 準備資料（第一次約 1 小時）

```bash
python -m llmtrain.prepare --out data/mmqa
```

- 這一步會從 Hugging Face 讀取 A、B、AB 三組問答，下載 B 和 AB 組的維基圖片（每秒約 1 張），並按文件以 8/1/1 切成訓練、評估、測試集。
- 中斷後重跑同一個指令即可，已經下載的圖片不會重新下載。
- 完成後會印出：
  - 各切分的文件數和筆數，以及各題型的筆數；
  - 因為圖片下載失敗而被排除的筆數；
  - 各組的文字 token 長度分佈（p50 / p90 / p99 / max）。
- 下載失敗的圖片清單、資料集版本和切分結果，都記錄在 `data/mmqa/manifest.json`。

**決定 `cutoff_len`：** token 長度統計只算文字，不含圖片 token。如果 p99 加上圖片 token 數接近或超過 4096，就調高 `configs/train_gemma4_e4b_qlora.yaml` 和 `configs/predict_gemma4_e4b.yaml` 裡的 `cutoff_len`。

## 3. 冒煙測試（幾分鐘）

這一步用極小的模型 LFM2.5-VL-1.6B 跑 12 步，確認整條流程能跑通，再開始花 GPU 時間訓練主模型。

```bash
python -m llmtrain.run train configs/smoke_lfm25vl_qlora.yaml --data data/mmqa
```

確認以下幾點：

- 訓練跑完沒有錯誤，log 中有評估 loss（`eval_loss`）。
- `runs/LFM2.5-VL-1.6B-<日期時間>/` 裡有：
  - 最多 3 個 `checkpoint-*` 資料夾；
  - 最終的 adapter（`adapter_model.safetensors`）；
  - 當次的設定 `config-*.yaml`、資料清單 `data_manifest.json`，以及執行過的指令 `commands.txt`。
- **續訓：** 訓練中途按 Ctrl+C 中斷，再執行

  ```bash
  python -m llmtrain.run train configs/smoke_lfm25vl_qlora.yaml --data data/mmqa --resume runs/LFM2.5-VL-1.6B-<日期時間>
  ```

  log 應該顯示 `Resuming training from ...checkpoint-*`。

接著測試預測和評分：

```bash
python -m llmtrain.run predict configs/predict_smoke_lfm25vl.yaml --data data/mmqa --adapter runs/LFM2.5-VL-1.6B-<日期時間>
```

完成後會印出指標，`runs/predict-.../` 裡會有 `generated_predictions.jsonl` 和 `metrics.json`。冒煙模型的分數不重要，重點是流程能跑完。

## 4. 基準：未微調的 Gemma 4 E4B

```bash
python -m llmtrain.run predict configs/predict_gemma4_e4b.yaml --data data/mmqa
```

輸出在 `runs/predict-gemma-4-E4B-it-base-<日期時間>/metrics.json`，這就是之後要比較的基準。

## 5. 主訓練

```bash
python -m llmtrain.run train configs/train_gemma4_e4b_qlora.yaml --data data/mmqa
```

- 約 2,400 步（3 個 epoch）。每 200 步存一次 checkpoint 並計算評估 loss，結束時保留評估 loss 最低的 adapter。
- 中斷後用 `--resume runs/gemma-4-E4B-it-<日期時間>` 繼續訓練。
- `plot_loss: true` 會在 run 資料夾中產生 loss 曲線圖。

**VRAM 不夠時**，依序嘗試以下調整。可以直接改設定檔，也可以在指令後面加上 `key=value` 臨時覆寫：

1. 調低 `image_max_pixels`，例如 `image_max_pixels=262144`；
2. 調低 `cutoff_len`；
3. 調低 `lora_rank`。

調整後的參數會記錄在 run 資料夾的 `commands.txt`。

## 6. 評估微調後的模型

```bash
python -m llmtrain.run predict configs/predict_gemma4_e4b.yaml --data data/mmqa --adapter runs/gemma-4-E4B-it-<日期時間>
```

把產生的 `metrics.json` 和步驟 4 的基準比較。

| 指標 | 意義 | 期望 |
|---|---|---|
| `char_f1` | 預測與參考答案的字元級 F1 | 高於基準 |
| `answer_short_recall` | 參考要點（`answer_short`）的字元有多少出現在預測中 | 高於基準 |
| `refusal_rate_unanswerable` | 無法回答題中，模型拒答的比例 | 越高越好 |
| `false_refusal_rate_answerable` | 可回答題中，模型誤拒答的比例 | 越低越好 |

`metrics.json` 裡還有依組別（`by_group`：A / B / AB）和題型（`by_qtype`）拆開的指標。`alignment.label_mismatches` 應該接近 0；如果數字很大，代表預測和參考資料沒有對上。

## 7. 匯出完整模型

```bash
python -m llmtrain.run export configs/export_gemma4_e4b.yaml --adapter runs/gemma-4-E4B-it-<日期時間>
```

這一步會把 adapter 合併進 bf16 基礎模型，輸出到 `runs/gemma-4-E4B-it-<日期時間>/merged/`。

## 換成更大的模型

把三個 `*_gemma4_e4b*.yaml` 複製一份，修改 `model_name_or_path` 和 `template`：

- `google/gemma-4-26B-A4B-it` 或 `google/gemma-4-31B-it`，搭配 `template: gemma4`；
- QLoRA 4-bit 粗估分別需要 13 GB 和 16 GB 以上的 VRAM。

**Gemma 4 12B 目前不能用。** 它的模型類型是 `gemma4_unified`，transformers 要到 main 分支才有，而 LLaMA-Factory 只允許到 transformers 5.8.0。
