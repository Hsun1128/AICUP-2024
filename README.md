# AICUP-2024


## Data Structure

```
AICUP-2024/     # 專案資料夾
├─ Baseline/                  # 官方提供的範例資料夾
│   ├─ bm25_retrieve.py      # 範例程式
│   └─ README.md             # 簡介
├─ CompetitionDataset/        # 資料集
│   ├─ dataset/
│   │   └─ preliminary/
│   │        ├─ ground_truths_example.json      # 包含真實答案的範例文件
│   │        ├─ pred_retrieve.json              # 預測檢索結果的範例文件
│   │        └─ questions_example.json          # 包含問題的範例文件
│   └─ reference/                        # 參考資料文件夾
│        ├─ faq/                         # 常見問題集資料
│        │   └─ pid_map_content.json
│        ├─ finance/                     # 財務相關資料
│        │   ├─ 0.pdf
│        │   ├─ 1.pdf
│        │   └─ ...
│        └─ insurance/                   # 保險相關資料
│             ├─ 0.pdf
│             ├─ 1.pdf
│             └─ ...
├─ LangChain_ORC/                         # OCR讀取PDF
│   ├─ faq_format.py
│   ├─ pdf_extractor.py
│   └─ preprocess.sh
├─ custom_dicts/                                  # 自定義詞典
│   ├─ origin_dict/                              # 原始詞典
│   │   ├─ common_use_dict.txt                  # 常用詞
│   │   └─ insurance_dict.txt                   # 保險用詞
│   ├─ with_freqency/                            # 含詞頻詞典
│   │   ├─ common_use_dict_with_frequency.txt   # 常用詞含詞頻
│   │   └─ insurance_dict_with_frequency.txt    # 保險用詞含詞頻
│   ├─ add_dict_frequency.py                     # 將原始辭典添加詞頻
│   ├─ dict.txt.big                              # 繁中辭典
│   └─ stopwords.txt                             # 停止詞
├─ word2vec/
│   ├─ corpusSegDone.txt                         # 分詞後的資料集
│   ├─ load_pretrain.py                          # 載入預訓練 Word2Vec model
│   ├─ model.bin                                 # 自己訓練的 model
│   ├─ segment_corpus.log                        # 資料集分詞詞頻預覽
│   ├─ segment_corpus.py                         # 將資料集分詞
│   ├─ train_word2vec.py                         # 訓練 Word2Vec model
│   ├─ transfer_vec2bin.py                       # 將 model從 .vec 轉換成 .bin
│   └─ wiki.zh.bin                               # wiki.zh 預訓練 model
├─ utils/                 # 工具包
│   └─ env.py            # 檢查環境變數
├─ .env                   # 環境變數
├─ .gitignore
├─ README.md
├─ bm25_retrieve_v2.py    # 主程式
├─ docker-compose.yaml    # docker compose
├─ requirements.txt       # 需安裝的 module
├─ retrieve.log           # 檢索狀況
├─ retrieve_v2.sh         # 運行主程式
├─ score.log              # 評分結果
└─ score.py               # 評估運行結果
```
