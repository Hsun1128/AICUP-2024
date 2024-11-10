## 1. 環境

### 1.1 基礎環境
- 作業系統：Ubuntu 18.04 LTS
- 程式語言：Python 3.9
- GPU：NVIDIA A100 (32GB)
- CUDA：12.1

### 1.2 主要套件
- 文件處理：
  - pdfplumber==0.11.4：PDF文件解析
  - langchain==0.3.2：文件分塊與向量化
  - langchain-core==0.3.9：文件分塊與向量化
  - langchain-community==0.3.1：文件分塊與向量化
  - tqdm==4.66.5：進度顯示
- 文本處理：
  - jieba==0.42.1：中文分詞
  - rank_bm25==0.2.2：BM25演算法實現
- 向量檢索：
  - faiss-gpu==1.7.2：向量索引與檢索
  - faiss-gpu-cu12==1.9.0.0：CUDA 12.1支援
  - torch==2.4.0+cu121：深度學習框架
  - scikit-learn==1.5.2：相似度計算
- 詞向量模型：
  - gensim==4.3.3：word2vec模型訓練

### 1.3 預訓練模型
1. 文本向量化模型：
   - 模型名稱：BAAI/bge-m3
   - 來源：HuggingFace (https://huggingface.co/BAAI/bge-m3)
   - 用途：文本向量表示

2. 詞向量模型：
   - 基於繁體中文維基百科語料訓練的word2vec模型
   - 來源：https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.vec
   - 格式：下載後轉換為.bin格式以提升載入速度與記憶體使用效率
   - 向量維度：400維
   

### 1.4 外部資源
- 繁體中文詞典：
  - dict.txt.big (結巴分詞繁體版詞典)
  - 來源：github.com/fxsjy/jieba
- 自定義詞典：
  - insurance_dict.txt：保險專業詞彙
  - common_use_dict.txt：財務會計常用詞彙

## 2. 演算方法與模型架構

本研究提出一個混合式檢索架構，結合了BM25詞頻統計與FAISS向量檢索的優勢，並創新性地引入多維度評分機制。系統架構主要包含以下核心組件：

### 2.1 檢索核心架構
本系統實現了雙重檢索機制：

1. **改良版BM25檢索引擎**
   - 採用優化的BM25Okapi算法：
     ```
     score(D,Q) = ∑(IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl)))
     ```
   - 關鍵參數優化：
     - k1=0.5：控制詞頻(TF)的飽和曲線
     - b=0.7：調節文檔長度正規化程度
     - epsilon=0.5：平滑因子，避免零分問題

2. **神經網路向量檢索**
   - 採用BAAI/bge-m3預訓練模型進行文本向量化
   - 實現L2正規化的向量表示
   - 基於餘弦相似度的高效檢索機制

### 2.2 多維度評分系統
本系統設計了七個獨立評分模組，每個模組針對不同的相關性維度：

1. **詞項重要性評分**
   ```python
   score = ∑(TF * IDF * position_weight)
   position_weight = 1 / (position + 1)
   ```
   - 考慮詞頻、逆文檔頻率與位置權重
   - 動態調整專業術語權重

2. **語義相似度評分**
   ```python
   doc_vector = ∑(IDF(word) * word2vec(word)) / ∑IDF(word)
   score = cosine_similarity(query_vector, doc_vector)
   ```
   - 使用加權詞向量表示文檔語義
   - IDF加權降低常見詞影響

3. **查詢覆蓋度評分**
   ```python
   weighted_coverage = ∑(log(N/df[term]) for term in intersection)
   score = weighted_coverage / len(expanded_query)
   ```
   - 評估查詢詞的覆蓋程度
   - 考慮擴展查詢詞的匹配情況

4. **位置感知評分**
   ```python
   score = 1 / (1 + std_pos + avg_pos/len(doc_tokens))
   ```
   - 考慮匹配詞的位置分布
   - 結合平均位置和離散程度

5. **詞密度評分**
   ```python
   density = ∑(log(N/df[w]) for w in window if w in intersection) / window_size
   score = max(density for each sliding window)
   ```
   - 使用滑動窗口計算局部密度
   - 動態調整窗口大小(預設20)

6. **上下文相似度評分**
   ```python
   context_score = ∑(cosine_similarity(term_vector, context_term_vector))
   score = context_score / (len(intersection) + 1)
   ```
   - 分析匹配詞的上下文語境
   - 固定上下文窗口大小(±3詞)

### 2.3 自適應權重機制
本系統實現了基於查詢特徵的動態權重調整：

1. **基礎權重配置**
   ```python
   base_weights = {
       'bm25': 0.20, 'faiss': 0.30, 'importance': 0.00,
       'semantic': 0.10, 'coverage': 0.10, 'position': 0.10,
       'density': 0.15, 'context': 0.05
   }
   ```

2. **動態調整策略**
   - 短查詢(≤2詞)：增強語義理解
     - semantic * 1.2
     - context * 1.2
     - bm25 * 0.8
   - 長查詢(>2詞)：強化詞頻匹配
     - bm25 * 1.2
     - coverage * 1.2
     - semantic * 0.8
   - 高多樣性查詢(>1.5)：增強語義理解
     - semantic * 1.2
     - context * 1.2

## 3. 創新性

本研究的創新性主要體現在以下三個方面：

### 3.1 自適應權重機制
本研究提出了一個基於查詢特徵的動態權重調整機制：

1. **查詢長度感知**
   - 短查詢優化：增強語義理解能力
   - 長查詢優化：強化詞頻匹配精度

2. **查詢多樣性適應**
   - 自動識別查詢詞多樣性
   - 動態調整檢索策略權重

3. **專業術語識別**
   - 自動識別領域特定詞彙
   - 優化專業術語的檢索權重

### 3.2 多維度評分整合
創新性地設計了一個模組化的評分框架：

1. **模組化設計**
   - 七個獨立評分模組，關注不同相關性維度
   - 支援靈活的模組組合與擴展

2. **動態權重分配**
   - 基於查詢特徵的自適應權重調整
   - 實現檢索策略的動態優化

### 3.3 混合檢索策略
實現了詞頻統計與神經網路的深度融合：

1. **檢索方法互補**
   - 結合BM25的精確匹配與FAISS的語義理解
   - 實現高精與高召回率的平衡

2. **後處理優化**
   - 多維度重排序機制
   - 動態閾值調整

## 4. 資料處理

本研究在資料處理方面採取了全面的優化策略：

### 4.1 文件預處理
1. **智能分塊機制**
   - 採用RecursiveCharacterTextSplitter進行文件分塊
   - 實現多層級分隔符優先序處理：['\n\n', '\n', '!', '?', '。', ';']
   - 分塊參數設定：
     - 分塊大小(chunk_size)：500
     - 重疊長度(overlap)：100

2. **文本正規化處理**
   - 實現多層次文本清理：
     ```python
     # 中文字間空白移除
     re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '')
     # 連續標點符號清理
     re.sub(r'([,、\$]){3,}', '')
     # 重複字符去除
     re.sub(r'([\u4e00-\u9fff])\1+', r'\1')
     # 頁碼格式移除
     re.sub(r'-\s*\d+\s*-|第\s*\d+\s*頁，\s*共\s*\d+\s*頁', '')
     ```

3. **並行處理優化**
   - PDF文件：採用ProcessPoolExecutor進行CPU密集型運算
   - JSON文件：使用ThreadPoolExecutor處理I/O密集型任務

### 4.2 查詢處理
1. **分詞優化**
   - 整合jieba分詞器：
     - 基礎詞典：dict.txt.big (繁體中文詞典)
     - 自定義詞典：
       - insurance_dict.txt (保險專業詞典)
       - common_use_dict.txt (財務會計常用詞典)
   - 停用詞過濾：
     - stopwords.txt：包含標點符號、語助詞、連接詞、代詞等

2. **查詢擴展**
   - word2vec模型：
     - 基於wiki語料訓練
     - 向量維度：400

### 4.3 向量化處理
1. **模型選擇**
   - 採用BAAI/bge-m3模型
   - 生成1024維向量表示

2. **索引優化**
   - FAISS索引建立
   - 增量更新機制
   - 檢索效率優化

### 4.4 資料增強
1. **文本增強**
   - 同義詞替換
   - 上下文擴展
   - 專業術語識別

2. **資料清理**
   - 特殊字符處理
   - 格式標準化
   - 冗餘內容去除

## 5. 訓練方式

本研究未進行模型訓練，主要原因如下：

### 5.1 預訓練模型的優勢
1. **模型成熟度**
   - BAAI/bge-m3已在大規模中文語料上訓練
   - 對金融、保險等專業領域有良好理解
   - 向量表示質量穩定可靠

2. **資源效率**
   - 避免耗費大量計算資源進行訓練
   - 減少模型調整時間
   - 降低過擬合風險

3. **任務適配性**
   - 檢索任務主要依賴模型的向量表示能力
   - 預訓練模型在語義理解上表現足夠好
   - 通過多維度評分機制彌補領域適應性

### 5.2 系統重點
本研究將重點放在以下方面：
1. 多維度評分機制的設計與實現
2. 檢索策略的優化與整合
3. 自適應權重機制的開發
4. 系統效能的優化
