本專案的目標是微調一個繁體中文多模態問答模型，資料庫和QA(Question-Answering)資料列在下方，我預期使用llm factory進行
本地的任務是建構訓練架構與程式，實際訓練可以到遠端工作站執行，不必在本地運行
套件安裝需求請記錄在requirements.txt，不要直接安裝在本地，如果必須在本地下載，在我的指引下使用conda環境 -> 根據流程改用pyproject.toml
不要使用中國模型(如qwen)

資料庫:
https://huggingface.co/datasets/ericssonbear/mr-right-zhtw

QA:
https://huggingface.co/datasets/ysh1257/multimodal-ir-zh-tw-qa

note:
如果QA中出現A、B、A+B，代表的意思可能是僅文字、僅圖片、兩者皆有
在確認遠端機器能否安裝claude前，agent規範與閱讀文件無提交必要
原本預期使用Gemma4 12B，因transformer不支援，暫且改用Gemma4 E4B