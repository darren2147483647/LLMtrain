# 交接：繁中多模態問答 QLoRA 訓練

最後更新：2026-11-18。完整經過見 [WORKLOG.md](WORKLOG.md)，操作步驟見 [RUNBOOK.md](RUNBOOK.md)。

## 現在的狀態

程式、設定和測試都在 `main`。在國網 Nano5 上已經完成：安裝、資料準備、冒煙測試的訓練和續訓。**還沒有跑過任何預測，也還沒有訓練主模型。**

| 步驟（RUNBOOK） | 狀態 | 結果 |
|---|---|---|
| 1. 安裝 | ✅ | torch 2.6.0+cu124；42 個測試通過 |
| 2. 準備資料 | ✅ | 7,945 筆，圖片失敗 0，輸出在 `~/LLMtrain/data/mmqa` |
| 3. 冒煙測試訓練、續訓 | ✅ | `runs/LFM2.5-VL-1.6B-20261118-105455`，最佳 `checkpoint-12`（eval_loss 0.803） |
| 3. 冒煙測試預測 | ❌ 不做 | LFM2-VL 的預測在 LLaMA-Factory 中不能用（見下方「已知問題」） |
| **4. 基準評估（Gemma 4 E4B）** | **⬜ 下一步** | |
| 5. 主訓練 | ⬜ | |
| 6. 評估微調後的模型 | ⬜ | |
| 7. 匯出 | ⬜ | |

## 下一個工作：基準評估

在 Nano5 上，先啟用 conda 環境，再從 repo 根目錄送出：

```bash
ml load miniconda3/24.11.1 && conda activate llmtrain
cd ~/LLMtrain && git pull && mkdir -p logs
sbatch -J baseline_e4b -p normal -t 03:00:00 swork.sh python -m llmtrain.run predict configs/predict_gemma4_e4b.yaml --data data/mmqa
```

**成功的條件：**
- `tail -n 5 logs/baseline_e4b_<工作編號>.log` 印出 `scored 792 predictions ...`；
- `runs/predict-gemma-4-E4B-it-base-<時間>/metrics.json` 裡的 `alignment.label_mismatches` 接近 0；
- 把 `metrics.json` 的 `overall` 和 `by_group` 記進 WORKLOG，這就是之後要比較的基準。

**同時要確認的事：**
- Gemma 4 E4B 能載入，第一次會下載約 16 GB；
- 圖片 token 加上文字後沒有超過 `cutoff_len: 4096`（log 裡不應該有被截斷的警告）；
- VRAM 用量（之後主訓練要參考）。

**失敗時：** log 最後幾行只是 `run.py` 的 `CalledProcessError`，真正的原因在前面：

```bash
grep -n -E "Error|error:" logs/baseline_e4b_<工作編號>.log | grep -v CalledProcessError | head -n 5
```

**成功之後：** 跑主訓練，約 2,400 步，時間不確定，先給足時間：

```bash
sbatch -J train_e4b -p normal -t 24:00:00 swork.sh python -m llmtrain.run train configs/train_gemma4_e4b_qlora.yaml --data data/mmqa
```

## 不在 repo 裡的檔案

**只在本機和工作站：** `vscode_tunnel.sh`、`prepare_data.sh`、`swork.sh`。它們含有計畫代碼和 email，所以列在 `.gitignore`，不推送到 GitHub。這個 repo 是公開的，不要把它們 commit 進去。

**只在工作站：**
- `data/mmqa/`：資料和約 2,000 張圖片；
- `runs/`：訓練和預測的輸出；
- `logs/`：Slurm 工作的 log。

這些都在 `.gitignore` 裡。重要的數字記在 WORKLOG 裡就好，檔案本身不需要推送。

## 已知問題和注意事項

- **torch：** Nano5 的驅動是 CUDA 12.4，只能用 `torch==2.6.0` 的 cu124 版本。
- **安裝套件：** 用 `~/.conda/envs/llmtrain/bin/python -m pip install ...`，只打 `pip` 曾經裝到別的環境。
- **LFM2-VL 不能產生答案：** 錯誤是 `model_kwargs are not used by the model: ['image_rows', 'image_cols', 'image_sizes']`，原因是 LLaMA-Factory 支援不完整。冒煙模型只用來測試訓練。
- **Gemma 4 12B 不能用：** 它的模型類型是 `gemma4_unified`，LLaMA-Factory 允許的 transformers 版本（最高 5.8.0）不支援。要升級模型時，改用同為 `gemma4` 類型的 26B-A4B 或 31B，只需要改設定檔。
- **tunnel 只用來編輯和短時間的測試：** 最長 2 小時，常常斷線。關掉介面不會停止計費，用完要 `scancel`。
- **Slurm 的寄信通知目前收不到：** 還沒確認原因。查看工作狀態用 `sh scheck.sh`。
- **`warmup_ratio is deprecated` 警告：** 不影響訓練，可以忽略。

## Issue 狀態

- **#2、#3、#4、#5：** 已完成並關閉。
- **#6：預測和評估的串接。** 等基準評估跑完，就是最後的驗證。
- **#7：主訓練、匯出、操作說明。** 等主訓練跑完才能驗證。
- **#1：** 規格本身，保持開啟，直到 #6 和 #7 都完成。

## 中途做出的決定

詳細原因見 #1 的留言。

- 主模型從 Gemma 4 12B 改為 **Gemma 4 E4B**。
- `answer_short` 的評分改用**字元召回率**，不用「是否命中」。
- 資料準備會打亂每個切分裡的資料順序，因為 LLaMA-Factory 的 `max_samples` 只取前 N 筆。
- 圖片下載原圖；工作站的空間足夠。
