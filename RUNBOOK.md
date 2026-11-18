# 工作站操作說明

在遠端工作站上，從安裝到匯出模型的完整步驟。**所有指令都在 repo 根目錄（`~/LLMtrain`）執行**：資料、設定檔和輸出都用相對路徑。訓練和預測的輸出會放在 `runs/`，資料放在 `data/mmqa/`，這兩個資料夾都不會被 git 追蹤。

### 在國網 Nano5 上

- **登入節點不能執行計算工作。** 下面的指令要在計算節點上執行：可以在 VS Code tunnel 的終端機裡執行（tunnel 本身就是一個 Slurm 工作），或用 `sbatch` 送出。
- **超過幾分鐘的工作用 `sbatch` 送出**（資料準備、預測、訓練），不要依賴 tunnel：tunnel 最長 2 小時（`dev` partition），斷線或結束時，在裡面執行的程式都會一起停止。
- 送出工作前先 `cd ~/LLMtrain && mkdir -p logs`：工作會在送出時所在的目錄執行，而 Slurm 不會自動建立 log 資料夾。
- 送出時沿用目前的環境變數。先啟用 conda 環境（`ml load miniconda3/24.11.1 && conda activate llmtrain`）再 `sbatch`，工作裡的 `python` 才會是 `llmtrain` 環境的版本。
- 一行送出：`sbatch -A <計畫代碼> -p normal -t 03:00:00 --gpus-per-node=1 -c 4 -J <名稱> -o logs/%x_%j.log --wrap "<指令>"`。
- 查看工作：`squeue -u $USER`（只查一次，不要用 `watch`）；取消：`scancel <工作編號>`。

## 0. 需求

- Linux 和 NVIDIA GPU。Gemma 4 E4B 用 QLoRA 4-bit 訓練時，粗估需要 6 GB 以上的 VRAM；圖片和長文件會增加用量，實際需求以步驟 3 冒煙測試量到的為準。
- Python 3.11 以上。
- 磁碟空間：圖片約 15 GB 以上（下載的是維基原圖）、模型快取約 20 GB、checkpoint 每個數百 MB。
- 可以連到 `huggingface.co` 和 `upload.wikimedia.org`。

## 1. 安裝

```bash
git clone https://github.com/darren2147483647/LLMtrain.git
cd LLMtrain
ml load miniconda3/24.11.1          # 國網 Nano5 用模組提供 conda；其他機器略過這行
conda create -n llmtrain python=3.11 -y
conda activate llmtrain
# 先安裝和 GPU 驅動相容的 torch。nvidia-smi 右上角的 "CUDA Version" 是驅動支援的最高版本，
# torch 的 CUDA 版本不能比它新，否則會出現 "NVIDIA driver on your system is too old"。
# 國網 Nano5 的驅動是 CUDA 12.4，用這行（cu124 最新只到 torch 2.6.0）：
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 驅動是 12.6 以上的機器，可改用 cu126 / cu128，見 https://pytorch.org/get-started/locally/
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 最後必須是 True
pip install -e ".[test]"
python -m pytest -q   # 應該全部通過
```

`pip install -e .` 會依 `pyproject.toml` 安裝 LLaMA-Factory，版本固定在 commit `8bf4811`，另外還會裝 transformers 5.5–5.8、bitsandbytes，以及 LLaMA-Factory 預測時要求的 `jieba`、`nltk`、`rouge-chinese`。

裝套件時用 `python -m pip install ...`，確保裝進目前這個 Python 所屬的環境；直接打 `pip` 可能會用到別的環境的 pip。

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

**冒煙模型只用來測試訓練，不能用來測試預測。** 這個 LLaMA-Factory 版本在 LFM2-VL 產生答案時，會把 `image_rows` / `image_cols` / `image_sizes` 傳給 `generate()`，transformers 會以 `The following model_kwargs are not used by the model` 報錯。預測和評分流程在步驟 4 用 Gemma 4 E4B 驗證。

## 4. 基準：未微調的 Gemma 4 E4B（約 30–90 分鐘）

```bash
python -m llmtrain.run predict configs/predict_gemma4_e4b.yaml --data data/mmqa
```

在 Nano5 上用 `sbatch` 送出，partition 用 `normal`、時間設 3 小時。第一次執行會先下載 Gemma 4 E4B（約 16 GB）。

完成時會印出 `scored 792 predictions ...` 和各項指標。輸出在 `runs/predict-gemma-4-E4B-it-base-<日期時間>/metrics.json`，這就是之後要比較的基準。這一步同時驗證了預測檔和參考資料的對齊（`metrics.json` 的 `alignment.label_mismatches` 應該接近 0）。

失敗時，log 最後幾行只是 `run.py` 回報「LLaMA-Factory 失敗了」，真正的原因在前面：

```bash
grep -n -E "Error|error:" logs/<log 檔> | grep -v CalledProcessError | head -n 5
```

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
