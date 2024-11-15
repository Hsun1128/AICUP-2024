# AICUP-2024

<p align="center">
  <a href="#"><img src="./2024_AICUP_Banner.jpg" alt="AICUP Team Logo"></a>
</p>

> [!TIP]
> This program is designed for efficient retrieval and evaluation tasks, especially useful for document-based question answering. It includes BM25 and FAISS-based retrieval methods, along with optional preprocessing and scoring tools, for AICUP-2024 competition datasets. This project supports various settings, such as custom dictionaries and weighted scoring, enabling fine-tuning for optimized retrieval accuracy.

## 🚀 Program Description

For a comprehensive overview of the program's logic and algorithms, refer to the [utils/README.md](utils/README.md) file.

## 📦 Installation and Start

> [!WARNING]
> Ensure that your CUDA version and GPU are compatible with the dependencies in the `requirements.txt` file. For GPU-accelerated tasks, make sure you have the appropriate CUDA version installed and that your GPU drivers are up to date.

1. **Install required dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```

2. **Set the question range in [`bm25_retrieve_v2.py`](bm25_retrieve_v2.py):**

    Modify the `RANGE` variable to specify which questions to process. For example:
    ```python
    RANGE = range(0, 150)  # Process questions range from 0 to 149 (total 150 questions)
    ```
    
    This setting controls which subset of questions will be processed during retrieval.

3. **Open [`retrieve_v2.sh`](retrieve_v2.sh) and check the file paths for the following variables:**

    - `--question_path`: Path to the file containing the questions to retrieve.
    - `--source_path`: Path to the folder containing the source documents or dataset.
    - `--output_path`: Path where the retrieval results will be saved.
    - `--load_path`: Path to a custom dictionary or resource file, such as a frequency dictionary.

    Example of [`retrieve_v2.sh`](retrieve_v2.sh) file:
    ```bash
    #!/bin/bash
    # ... check word2vec/wiki.zh.bin
    python3 bm25_retrieve_v2.py \
      --question_path ./CompetitionDataset/dataset/preliminary/questions_example.json \
      --source_path ./CompetitionDataset/reference \
      --output_path ./CompetitionDataset/dataset/preliminary/pred_retrieve.json \
      --load_path ./custom_dicts/with_frequency
    ```

4. **Once you've verified the file paths, open your terminal, navigate to the directory where the `retrieve_v2.sh` script is located, and run the following command to execute it:**
   
   ```bash
   ./retrieve_v2.sh
   ```
   
   This will start the retrieval process, and the results will be saved to the file specified in `--output_path`.

5. **After the script finishes running, you can check the output at the location specified in the `--output_path` to view the retrieval results.**
> [!TIP]
> You can find detailed logs in the [logs/](logs/) directory:
> - `retrieve_YYYY-MM-DD_HH-MM-SS.log`: Contains retrieval process logs and results
> - `chunk_YYYY-MM-DD_HH-MM-SS.log`: Contains word segmentation results and chunking details

> [!NOTE]
> If you have a **ground_truths.json** file, you can also run [`python3 score.py`](score.py) to evaluate the retrieval results. The evaluation results will be saved in `logs/score_YYYY-MM-DD_HH-MM-SS.log`.


6. **If you want to experiment with different parameters, you can modify the settings in [`config.yaml`](config.yaml):**

   The `config.yaml` file contains various configurable parameters that control the retrieval behavior:

   ```yaml
   # Core parameters
   load_all_data: false       # Whether to load all data at once (not developed yet)
   bm25_k1: 0.5               # BM25 k1 parameter
   bm25_b: 0.7                # BM25 b parameter
   chunk_size: 500            # Size of text chunks
   overlap: 100               # Overlap between chunks

   # Scoring method weights
   base_weights:
     bm25: 0.20              # Weight for BM25 score
     faiss: 0.3              # Weight for FAISS similarity
     importance: 0.0         # Weight for term importance
     semantic: 0.1           # Weight for semantic matching
     coverage: 0.1           # Weight for query coverage
     position: 0.1           # Weight for position scoring
     density: 0.15           # Weight for term density
     context: 0.05           # Weight for context similarity
   ```

   Adjust these parameters based on your specific needs and run the retrieval script again to see the effects.

> [!IMPORTANT]
> Some parameters in `config.yaml` are still under development and may need further tuning. Please check back later for the optimal settings.

> [!TIP]
> 🌟 **You may want to add common words or stop words. For details, refer to [custom_dicts/README.md](custom_dicts/README.md).**  
> 🧠 **If you want to train your own Word2Vec model, check out [word2vec/README.md](word2vec/README.md) for more information.**

## 🔄 [Optional] Data Preprocessing 

   If you want to preprocess the data using OCR and text extraction, you can use the tools in the [LangChain_ORC/](LangChain_ORC/) directory:

   ```bash
   cd LangChain_ORC
   ./preprocess.sh
   ```

   This will:
   - Extract text from PDF files in the reference directory
   - Format FAQ documents for better retrieval
   - Generate preprocessed files in a structured format

   See [LangChain_ORC/README.md](LangChain_ORC/README.md) for detailed preprocessing instructions and options.

> [!NOTE]
> Data preprocessing is optional. The retrieval system will work with raw files, but preprocessing may improve results for certain document types.

##  🐳 Quick Start with Docker
### Run [`retrieve_v2.sh`](retrieve_v2.sh)
1. Build and run the container.
    ```bash
    docker compose up -d --build
    ```
2. Run the [retrieve_v2.sh](retrieve_v2.sh)
    ```bash
    docker exec baseline conda run -n baseline bash /app/retrieve_v2.sh
    ```

### RUN [`retrieve_v3.sh`](retrieve_v3.sh)
#### **Part 1 run the preprocess**

1. Get into the [LangChain_ORC/](LangChain_ORC/)
    ```bash
    cd LangChain_ORC
    ```
2. Run the docker and the OCR process
    ```bash
    docker compose up -d --build
    ```
#### **Part 2 Run the [`retrieve_v3.sh`](retrieve_v3.sh)**
1. Build and run the container.
    ```bash
    docker compose up -d --build
    ```
2. Run the [`retrieve_v3.sh`](retrieve_v3.sh)
    ```bash
    docker exec baseline conda run -n baseline bash /app/retrieve_v3.sh
    ```


## 📂Data Structure

```bash
AICUP-2024/     # 專案資料夾
├─ Baseline/                 # 官方提供的範例資料夾
│   ├─ README.md               # 簡介
│   ├─ README.pdf              # pdf 簡介
│   ├─ bm25_retrieve.py        # 範例程式
│   ├─ bm25_with_faiss.py      # 測試 faiss
│   ├─ langchain_test.py       # 測試 langchain
│   ├─ new_bm25_retrieve.py    # 測試新 bm25 程式
│   └─ retrieve-test.py        # 測試 retrieve
├─ CompetitionDataset/                        # 資料集
│   ├─ dataset/
│   │   └─ preliminary/
│   │        ├─ ground_truths_example.json      # 包含真實答案的範例文件
│   │        ├─ pred_retrieve.json              # 預測檢索結果的範例文件
│   │        └─ questions_example.json          # 包含問題的範例文件
│   └─ reference/                      # 參考資料文件夾
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
├─ LangChain_ORC/             # OCR讀取PDF
│   ├─ docker-compose.yaml      # docker
│   ├─ dockerfile               # 建立 docker
│   ├─ faq_format.py            # faq format
│   ├─ pdf_extractor.py         # pdf format
│   └─ preprocess.sh            # preprocess 主程式
├─ custom_dicts/                               # 自定義詞典
│   ├─ origin_dict/                              # 原始詞典
│   │   ├─ common_use_dict.txt                     # 常用詞
│   │   └─ insurance_dict.txt                      # 保險用詞
│   ├─ with_freqency/                            # 含詞頻詞典
│   │   ├─ common_use_dict_with_frequency.txt      # 常用詞含詞頻
│   │   └─ insurance_dict_with_frequency.txt       # 保險用詞含詞頻
│   ├─ add_dict_frequency.py                     # 將原始辭典添加詞頻
│   ├─ dict.txt.big                              # 繁中辭典
│   └─ stopwords.txt                             # 停止詞
├─ word2vec/                 # Word2Vec 詞向量模型
│   ├─ corpusSegDone.txt       # 分詞後的資料集
│   ├─ load_pretrain.py        # 載入預訓練 Word2Vec model
│   ├─ model.bin               # 自己訓練的 model
│   ├─ segment_corpus.log      # 資料集分詞詞頻預覽
│   ├─ segment_corpus.py       # 將資料集分詞
│   ├─ train_word2vec.py       # 訓練 Word2Vec model
│   ├─ transfer_vec2bin.py     # 將 model從 .vec 轉換成 .bin
│   └─ wiki.zh.bin             # wiki.zh 預訓練 model
├─ utils/                                      # 工具包
│   ├─ rag_processor/                            # 存放 RAG 的所有功能 (功能尚未分離完全)
│   │   ├─ readers/                                # 檔案讀取工具
│   │   │   ├─ __init__.py                           # 匯出文件讀取模組
│   │   │   ├─ document_loader.py                    # 文件讀取器
│   │   │   ├─ json_reader.py                        # 讀取 json
│   │   │   └─ pdf_reader.py                         # 讀取 pdf
│   │   ├─ retrieval_system/                       # 檢索工具
│   │   │   ├─ __pycache__
│   │   │   ├─ __init.py                             # 匯出檢索模組
│   │   │   ├─ bm25_retrival.py                      # BM25 (未分離)
│   │   │   ├─ context_similarity.py                 # 計算上下文相似度
│   │   │   ├─ faiss_retrieval.py                    # faiss 向量檢索
│   │   │   ├─ position_score.py                     # 計算位置得分
│   │   │   ├─ query_coverage.py                     # 計算查詢詞覆蓋率
│   │   │   ├─ reranker.py                           # 向量搜索重排序 (未完成)
│   │   │   ├─ retrieval_system.py                   # 整合檢索器
│   │   │   ├─ semantic_similarity.py                # 計算語意相似度
│   │   │   ├─ term_density.py                       # 計算詞密度
│   │   │   └─ term_importance.py                    # 計算詞項重要性
│   │   ├─ scoring/                                # 評分工具
│   │   │   ├─ __pycache__
│   │   │   ├─ __init__                              # 匯出評分模組
│   │   │   ├─ base_scorer.py                        # 基礎評分 (未完成)
│   │   │   ├─ bm25_scorer.py                        # BM25 主要評分
│   │   │   └─ weighted_scorer.py                    # 多維度加權評分
│   │   ├─ __init__.py                             # 匯出 RAG 系統中各處理模組的核心元件，提供統一的接口供外部使用
│   │   ├─ config.py                               # 讀取、儲存 config 文件
│   │   ├─ document_processor.py                   # 文件處理器
│   │   ├─ document_score_calculator.py            # 文件評分器
│   │   ├─ query_processor.py                      # 查詢處理器
│   │   └─ resource_loader.py                      # 資源載入器
│   ├─ RAGProcessor.py                           # RAG檢索器
│   ├─ README.md                                 # 整個檢索方式的說明
│   ├─ __init__.py
│   └─ env.py                                    # 檢查環境變數
├─ logs/                                    # 運行日誌
│   ├─ retrieve_xxxx-xx-xx_xx-xx-xx.log       # 檢索狀況
│   ├─ chunk_xxxx-xx-xx_xx-xx-xx.log          # 文件分塊狀況
│   └─ score_xxxx-xx-xx_xx-xx-xx.log          # 評分結果
├─ .env                   # 環境變數
├─ .gitignore
├─ README.md              # 程式使用說明文件
├─ bm25_retrieve_v2.py    # 主程式 (直接讀取PDF)
├─ bm25_retrieve_v3.py    # 主程式 (載入預處理json，json_reader未更新完成，尚未與v2合併)
├─ config.yaml            # 主程式運行參數 (部分參數尚未設置完成)
├─ docker-compose.yaml    # docker compose
├─ dockerfile             # 建立 docker
├─ requirements.txt       # 需安裝的 module
├─ retrieve.log           # 檢索狀況
├─ retrieve_v2.sh         # 運行主程式 (直接讀取PDF)
├─ retrieve_v3.sh         # 運行主程式 (載入預處理json，json_reader未更新完成，尚未與v2合併)
├─ score.log              # 評分結果
└─ score.py               # 評估運行結果
```
