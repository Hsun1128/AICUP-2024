# Preprocess

這是一個使用 LangChain 和 Tesseract OCR 的文件預處理系統，主要用於處理 PDF 文件和 FAQ 資料的轉換。

> [!NOTE]
> 系統需求
> - Docker 環境
> - NVIDIA GPU (支援 CUDA)
> - Docker Compose

## 專案結構
```
LangChain_ORC/
├── dockerfile # Docker 環境配置文件
├── docker-compose.yaml # Docker Compose 配置文件
├── pdf_extractor.py # PDF 文件處理主程式
├── faq_format.py # FAQ 資料格式轉換程式
└── preprocess.sh # 資料預處理腳本
```


## 環境設置

本專案使用 Docker 容器運行，基於 NVIDIA CUDA 環境，主要包含：

- CUDA 12.1.0
- cuDNN 8
- Ubuntu 22.04
- Python 3.10
- Tesseract OCR（支援英文和繁體中文）

### 建立容器

```bash
docker-compose up -d
```

## 如何使用

```bash
./preprocess.sh
```

這個腳本會依序執行：
1. FAQ 資料的格式轉換
2. 金融文件的 PDF 處理
3. 保險文件的 PDF 處理


## FAQ 格式轉換工具(`faq_format.py`)

這個工具用於將 FAQ（常見問題集）數據從原始格式轉換為特定的 JSON 格式。

### 功能說明

此工具主要完成以下轉換工作：
- 讀取原始的 FAQ 數據（`pid_map_content.json`）
- 將問答對轉換為指定格式
- 為每個文件生成獨立的 JSON 輸出文件

### 輸入格式

原始數據格式（pid_map_content.json）示例：
```json
{
    "file_id_1": [
        {
            "question": "問題1",
            "answers": ["答案1", "答案2"]
        },
        {
            "question": "問題2",
            "answers": ["答案1"]
        }
    ]
}
```
### 輸出格式

每個輸出文件（`{file_id}.json`）的格式：
```json
{
    "id": "file_id",
    "contents": "問題1\t答案1, 答案2\n問題2\t答案1\n"
}
```


### 使用方法

#### 命令行參數


```bash
python faq_format.py --folder <數據文件夾> --input_root <輸入根目錄> --output_root <輸出根目錄>
```

參數說明：
- `--folder`：數據文件夾名稱
- `--input_root`：輸入文件根目錄路徑
- `--output_root`：輸出文件根目錄路徑


### 目錄結構

預期的目錄結構如下：
```
CompetitionDataset/
├── reference/
│ └── faq/
│ └── pid_map_content.json
└── ref_contents/
└── faq/
├── file_id_1.json
├── file_id_2.json
└── ...
```

> [!WARNING]
> 1. 確保輸入目錄中存在 `pid_map_content.json` 文件
> 2. 程序會自動創建輸出目錄（如果不存在）
> 3. 輸出文件中的問答對使用製表符（\t）分隔
> 4. 多個答案使用逗號（,）連接

## PDF 文件處理工具(`pdf_extractor.py`)

這是一個使用 Python 開發的 PDF 文件處理工具，支援 OCR 功能，可以將 PDF 文件中的文字內容（包含圖片中的文字）提取並轉換為 JSON 格式。

### 功能特點

- 支援中英文 OCR 辨識
- 多進程並行處理，提高效率
- 自動檢測系統依賴
- 詳細的日誌記錄
- 處理統計資訊收集
- 進度條顯示處理進度

### 使用方法

#### 基本使用

```bash
python pdf_extractor.py /path/to/input/folder /path/to/output/folder
```

### 進階選項

- 指定工作進程數：
```bash
python pdf_extractor.py /path/to/input/folder /path/to/output/folder --workers 4
```

### 輸出結構

```
output_folder/
├── [轉換後的JSON檔案]
└── logs/
    ├── pdf_extractor_20240106_123456.log  # 詳細日誌
    └── statistics_20240106_123456.json     # 統計資料
```

### JSON 輸出格式

每個 PDF 檔案會產生對應的 JSON 檔案，格式如下：
```json
{
    "id": "檔案名稱",
    "contents": "提取的文字內容"
}
```

### 統計資料格式

```json
{
    "start_time": "開始時間",
    "end_time": "結束時間",
    "total_files": "總檔案數",
    "successful_files": "成功處理檔案數",
    "failed_files": "失敗檔案數",
    "total_processing_time": "總處理時間",
    "average_processing_time": "平均處理時間",
    "success_rate": "成功率"
}
```

### 主要功能說明

1. **OCR 支援**
   - 使用 Tesseract OCR 引擎
   - 支援中文（繁體）和英文辨識
   - 可辨識表格結構

2. **多進程處理**
   - 自動偵測 CPU 核心數
   - 可手動指定工作進程數
   - 提供進度條顯示處理進度

3. **日誌系統**
   - 彩色控制台輸出
   - 詳細的檔案日誌
   - 自動分割大型日誌檔案

4. **錯誤處理**
   - 完整的異常捕捉
   - 詳細的錯誤日誌
   - 處理失敗統計

> [!WARNING]
> 1. 確保系統已安裝 Tesseract OCR 及中文語言包
> 2. 輸入資料夾必須包含 PDF 檔案
> 3. 程式會自動建立輸出資料夾及日誌資料夾
> 4. 大型 PDF 檔案可能需要較長處理時間
> 5. 建議預留足夠的系統記憶體


> [!TIP]
> 1. 如果遇到 Tesseract 相關錯誤：
>    ```bash
>    sudo apt-get update
>    sudo apt-get install tesseract-ocr tesseract-ocr-chi-tra
>    ```
> 2. 如果遇到記憶體不足：
>    - 減少同時處理的工作進程數
>    - 使用 `--workers` 參數指定較小的數值
> 3. 如果遇到權限問題：
>    - 確保對輸入和輸出資料夾有讀寫權限
>    - 檢查日誌資料夾的權限設定
