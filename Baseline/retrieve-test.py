import jieba  # 用於中文文本分詞
import os
import pdfplumber  # 用於從PDF文件中提取文字的工具
import pytesseract
from PIL import Image
from collections import Counter

# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
	pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

	# 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
	pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
	pdf_text = ''
	for _, page in enumerate(pages):  # 迴圈遍歷每一頁
		text = page.extract_text()  # 提取頁面的文本內容
		if text:
			pdf_text += text.replace(" ", "").replace("\n", "")  # 去除內容中的空格和換行符
		else:
			# 嘗試提取圖片中的文字
			print('嘗試擷取圖片中的內容...')
			try:
				for img in page.images:
					# 獲取圖片的位置信息
					x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
					
					# 確保邊界框在頁面邊界內
					x0 = max(x0, 0)
					top = max(top, 0)
					x1 = min(x1, page.width)
					bottom = min(bottom, page.height)
					
					# 提取圖片
					img_obj = page.within_bbox((x0, top, x1, bottom)).to_image()
					img_data = img_obj.original
					# 使用OCR提取圖片中的文字
					text_from_image = pytesseract.image_to_string(img_data, lang='chi_sim')
					pdf_text += text_from_image.replace(" ", "").replace("\n", "")
			except:
				print('擷取失敗')
				pass

	pdf.close()  # 關閉PDF文件

	return pdf_text  # 返回萃取出的文本

# 加載停用詞文件
def load_stopwords(filepath):
	with open(filepath, 'r', encoding='utf-8') as file:
		stopwords = set(line.strip() for line in file)
	print('loadding stopwords success')
	print(stopwords)
	return stopwords

# 使用 jieba 進行分詞並過濾停用詞
def jieba_cut_with_stopwords(words: str, stopwords: list=None) -> list:
	return [word for word in words if word not in stopwords and word.strip()]

# 假設停用詞文件路徑為 stopwords.txt
stopwords = load_stopwords('custom_dicts/stopwords.txt')

# 讀取
text = read_pdf('競賽資料集/reference/insurance/3.pdf')
print(text)

print('='*100)

########################################################
# 使用 jieba 分詞
########################################################

# 載入AICUP/custom_dicts/with_frequency中的所有.txt
load_path = "./custom_dicts/with_frequency"
for filename in os.listdir(load_path):
	if filename.endswith('.txt'):
		jieba.load_userdict(os.path.join(load_path, filename))

# 搜尋引擎模式
words = jieba.cut_for_search(text)
words = list(words)
clean_words = jieba_cut_with_stopwords(words, stopwords)
word_counts = Counter(clean_words)
#print("分詞結果：", "/".join(words))
print(word_counts)
print(type(words))  # type is generator

########################################################
# 使用 embedding model
print('='*100)
input(...)
########################################################

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
tokens = tokenizer.tokenize(text)
clean_tokens = jieba_cut_with_stopwords(tokens, stopwords)
token_counts = Counter(clean_tokens)
#print("model分詞：", tokens)
print(token_counts)
print(type(tokens))  # type is list



########################################################
# 文本處理器測試
########################################################
class TextProcessor:
    def __init__(self, stopwords_filepath: str=STOPWORDS_FILEPATH, bm25_k1: float=BM25_K1, bm25_b: float=BM25_B, use_expansion: bool=USE_EXPANSION, expanded_topn: int=EXPANDED_TOPN, chunk_size: int=CHUNK_SIZE, overlap: int=OVERLAP, expansion_model_path: str=EXPANSION_MODEL_PATH, use_faiss: bool=USE_FAISS):
        """
        初始化文本處理器
        
        Args:
            stopwords_filepath (str): 停用詞文件路徑
            bm25_k1 (float): BM25 k1參數
            bm25_b (float): BM25 b參數
            use_expansion (bool): 是否使用查詢擴展
            expanded_topn (int): 每個詞擴展的相似詞數量
            chunk_size (int): 文本分塊大小
            overlap (int): 分塊重疊大小
            expansion_model_path (str): word2vec模型路徑
            use_faiss (bool): 是否使用FAISS
            
        Example:
            >>> processor = TextProcessor("stopwords.txt")
            >>> print(processor.chunk_size)
            500
        """
        self.stopwords = self.load_stopwords(stopwords_filepath)  # 載入停用詞
        self.bm25_k1 = bm25_k1  # BM25參數k1
        self.bm25_b = bm25_b    # BM25參數b
        self.expanded_topn = expanded_topn  # 查詢擴展時每個詞的相似詞數量
        self.use_expansion = use_expansion  # 是否使用查詢擴展
        self.word2vec_model = self.load_word2vec_model(os.path.join(os.path.dirname(__file__), expansion_model_path))  # 載入word2vec模型
        self.chunk_size = chunk_size  # 文本分塊大小
        self.overlap = overlap    # 文本分塊重疊大小
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, length_function=len, is_separator_regex=False, keep_separator=False, separators=['\n\n', '\n', '!', '?', '。', ';'])
        self.use_faiss = use_faiss  # 是否使用FAISS
        if self.use_faiss:
            self.embeddings = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-m3', model_kwargs = {'device': 'cuda'}, encode_kwargs = {'normalize_embeddings': True})

    # 載入word2vec模型
    def load_word2vec_model(self, model_path: str):
        model = KeyedVectors.load(model_path)  # 載入二進制格式的word2vec模型
        logger.info('Word2Vec model loaded successfully')
        return model

    # 載入停用詞
    def load_stopwords(self, filepath: str) -> set[str]:
        with open(filepath, 'r', encoding='utf-8') as file:
            stopwords = set(line.strip() for line in file)
        logger.info('Loading stopwords success')
        logger.info(f'stopwords: {stopwords}')
        return stopwords

    # 使用jieba進行分詞並移除停用詞
    def jieba_cut_with_stopwords(self, text: str) -> List[str]:
        words = jieba.cut_for_search(text)
        return [word for word in words if word not in self.stopwords and word.strip()]  # clean stopwords

    # 文本切分
    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        text_len = len(text)
        for i in range(0, text_len, self.chunk_size - self.overlap):
            end = min(i + self.chunk_size, text_len)
            chunk = text[i:end]
            #print(chunk, '\n')
            chunks.append(chunk)
            if end == text_len:
                break
        return chunks

    # 查詢擴展
    def expanded_query(self, tokenized_query: Set[str]) -> Set[str]:
        """
        對輸入的分詞後查詢進行擴展，增加相似詞以提高召回率
        
        Args:
            tokenized_query (Set[str]): 分詞後的查詢詞集合，例如 {"台灣", "總統府"}
            
        Returns:
            Set[str]: 擴展後的查詢詞集合，例如 {"台灣", "總統府", "中華民國", "政府"}
            
        Example:
            >>> processor = TextProcessor()
            >>> query = {"台灣", "總統府"}
            >>> expanded = processor.expanded_query(query)
            >>> print(expanded)
            {"台灣", "總統府", "中華民國", "政府"}
        """
        if not self.use_expansion:
            return tokenized_query  # 如果不使用擴展，返回原始查詢
        
        expanded_query = []
        for word in tokenized_query:
            if word in self.word2vec_model.key_to_index:
                topn_words = self.word2vec_model.most_similar(word, topn=self.expanded_topn)
                topn_words = set([w[0] for w in topn_words if w[0] not in self.stopwords and w[0].strip() and len(w[0]) != 1])  # clean stopwords
                expanded_query.extend([word] + list(topn_words))  # 包含原始詞和擴展詞
        return set(expanded_query)

    def _faiss_retrieve(self, query: str, chunked_corpus: List[str], key_idx_map: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """
        使用FAISS進行向量檢索
        
        Args:
            query (str): 查詢文本，例如 "台灣總統府在哪裡?"
            chunked_corpus (List[str]): 切分後的文本列表，例如 ["台灣總統府位於台北市中正區...", "總統府是一棟巴洛克式建築..."]
            key_idx_map (List[Tuple[int, int]]): 文本片段對應的(file_key, chunk_index)列表，例如 [(1, 0), (1, 1)]
            
        Returns:
            Dict[Tuple[int, int], float]: 包含檢索結果的字典，格式為 {(file_key, chunk_index): score}
                 例如 {(1, 0): 0.85, (1, 1): 0.75}
                 - file_key (int): 文件ID，例如 1 
                 - chunk_index (int): 文本片段索引，例如 0
                 - score (float): 相似度分數，範圍0-1，例如 0.85
                 
        Example:
            >>> processor = TextProcessor()
            >>> query = "台灣總統府在哪裡?"
            >>> corpus = ["台灣總統府位於台北市中正區...", "總統府是一棟巴洛克式建築..."]
            >>> key_idx_map = [(1, 0), (1, 1)]
            >>> results = processor._faiss_retrieve(query, corpus, key_idx_map)
            >>> print(results)
            {(1, 0): 0.85, (1, 1): 0.75}
        """
        # 初始化結果字典
        faiss_results: Dict[Tuple[int, int], float] = {}
        
        # 使用FAISS建立向量索引
        vector_store = FAISS.from_texts(chunked_corpus, self.embeddings, normalize_L2=True)
        
        # 進行相似度搜索，返回前5個最相似的文檔及其分數
        faiss_ans = vector_store.similarity_search_with_score(query, k=5)

        # 記錄分隔線
        logger.info('-'*100)
        
        # 處理每個檢索結果
        for doc, score in faiss_ans:
            # 找到文檔在chunked_corpus中的索引
            faiss_actual_index: int = chunked_corpus.index(doc.page_content)
            # 獲取對應的文檔鍵值
            faiss_chunk_key: Tuple[int, int] = key_idx_map[faiss_actual_index]
            # 儲存分數
            faiss_results[faiss_chunk_key] = float(score)
            # 記錄檢索結果
            logger.info(f'FAISS Score: [{score:.4f}], PDF: {faiss_chunk_key[0]}, '
                      f'Chunk: {faiss_chunk_key[1]}, metadata: [{doc.metadata}]')
                      
        return faiss_results


    # 根據查詢語句和指定的來源，檢索答案
    def _calculate_document_scores(self, tokenized_corpus: List[List[str]], 
                                 doc_tokens: List[str],
                                 expanded_query: Set[str], 
                                 doc_freq: Counter[str],
                                 query_vector: np.ndarray) -> Tuple[float, float, float, float, float, float, Set[str], Counter[str]]:
        """
        計算文檔的各項評分指標
        
        Args:
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果列表,例如 [["台灣", "總統府",...], [...],...]
            doc_tokens (List[str]): 當前文檔的分詞結果,例如 ["台灣", "總統府",...]
            expanded_query (Set[str]): 擴展後的查詢詞集合,例如 {"台灣", "總統府", "位置",...}
            doc_freq (Counter[str]): 詞頻統計,例如 Counter({"台灣": 10, "總統府": 5,...})
            query_vector (np.ndarray): 查詢向量,例如 array([0.2, 0.3, 0.1,...])
            
        Returns:
            Tuple[float, float, float, float, float, float, Set[str], Counter[str]]: 包含以下8個評分指標:
                - term_importance (float): 詞項重要性分數 (0-1),例如 0.8
                - semantic_similarity (float): 語義相似度分數 (0-1),例如 0.7
                - query_coverage (float): 查詢覆蓋率分數 (0-1),例如 0.6
                - position_score (float): 詞位置分數 (0-1),例如 0.5
                - term_density (float): 詞密度分數 (0-1),例如 0.4
                - context_similarity (float): 上下文相似度分數 (0-1),例如 0.6
                - intersection (Set[str]): 查詢詞和文檔詞的交集,例如 {"台灣", "總統府"}
                - frequency (Counter[str]): 交集詞在文檔中的頻率統計,例如 Counter({"台灣": 2, "總統府": 1})
                
        Example:
            >>> processor = TextProcessor(...)
            >>> scores = processor._calculate_document_scores(
            ...     tokenized_corpus=[["台灣", "總統府"], ["台北", "101"]],
            ...     doc_tokens=["台灣", "總統府", "位置"],
            ...     expanded_query={"台灣", "總統府", "位置"},
            ...     doc_freq=Counter({"台灣": 2, "總統府": 1}),
            ...     query_vector=np.array([0.2, 0.3, 0.1])
            ... )
            >>> print(scores)
            (0.8, 0.7, 0.6, 0.5, 0.4, 0.6, {"台灣", "總統府"}, Counter({"台灣": 2, "總統府": 1}))
        """
        # 計算查詢詞和文檔詞的交集
        intersection = set(expanded_query) & set(doc_tokens)
        
        # 計算交集中每個詞在文檔中的頻率
        frequency = Counter(token for token in doc_tokens if token in intersection)
        
        # 計算各項評分指標
        term_importance = self._calculate_term_importance(
            intersection, doc_tokens, frequency, doc_freq, tokenized_corpus)
            
        semantic_similarity = self._calculate_semantic_similarity(
            doc_tokens, query_vector, doc_freq, tokenized_corpus)
            
        query_coverage = self._calculate_query_coverage(
            intersection, expanded_query, tokenized_corpus, doc_freq)
            
        position_score = self._calculate_position_score(
            doc_tokens, intersection)
            
        term_density = self._calculate_term_density(
            doc_tokens, intersection, tokenized_corpus, doc_freq)
            
        context_similarity = self._calculate_context_similarity(
            doc_tokens, intersection)
            
        return (term_importance, semantic_similarity, query_coverage, 
                position_score, term_density, context_similarity,
                intersection, frequency)

    def _process_retrieved_document(self, doc_index: int, doc_content: str, 
                                  chunked_corpus: List[str], key_idx_map: List[Tuple[int, int]], 
                                  tokenized_corpus: List[List[str]], scores: List[float],
                                  expanded_query: Set[str], doc_freq: Counter[str], 
                                  query_vector: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Dict[str, float]]:
        """
        處理單個檢索到的文檔，計算各項評分指標並記錄日誌
        
        Args:
            doc_index (int): 文檔在檢索結果中的排名索引,例如 0 表示第一個檢索結果
            doc_content (str): 文檔內容,例如 "台灣總統府位於台北市中正區重慶南路一段122號..."
            chunked_corpus (List[str]): 切分後的語料庫,例如 ["文檔1內容", "文檔2內容",...]
            key_idx_map (List[Tuple[int, int]]): 文檔索引到(PDF ID, chunk ID)的映射,例如 [(1, 0), (1, 1), (2, 0)]
            tokenized_corpus (List[List[str]]): 分詞後的語料庫,例如 [["台灣", "總統府",...], [...],...]
            scores (List[float]): BM25分數列表,例如 [0.8, 0.6, 0.4,...]
            expanded_query (Set[str]): 擴展後的查詢詞集合,例如 {"台灣", "總統府", "位置",...}
            doc_freq (Counter[str]): 詞頻統計,例如 Counter({"台灣": 10, "總統府": 5,...})
            query_vector (np.ndarray): 查詢向量,例如 array([0.2, 0.3, 0.1,...])
            
        Returns:
            Tuple[Optional[Tuple[int, int]], Dict[str, float]]:
                - 文檔鍵值(PDF ID, chunk ID)或None(處理失敗時),例如 (1, 2) 或 None
                - 包含各項評分的字典,例如 {
                    "bm25_score": 0.8,
                    "term_importance": 0.7,
                    "semantic_similarity": 0.6,
                    "query_coverage": 0.8,
                    "position_score": 0.5,
                    "term_density": 0.4,
                    "context_similarity": 0.6
                  }
                  
        Example:
            >>> processor = TextProcessor(...)
            >>> doc_index = 0
            >>> doc_content = "台灣總統府位於台北市中正區..."
            >>> chunk_key, scores = processor._process_retrieved_document(
            ...     doc_index, doc_content, chunked_corpus, key_idx_map,
            ...     tokenized_corpus, scores, expanded_query, doc_freq, query_vector)
            >>> print(chunk_key)
            (1, 2)
            >>> print(scores)
            {'bm25_score': 0.8, 'term_importance': 0.7, ...}
        """
        try:
            # 找到當前文檔在chunked_corpus中的索引位置
            actual_index: int = chunked_corpus.index(doc_content)
            # 獲取對應的文檔鍵值
            chunk_key: Tuple[int, int] = key_idx_map[actual_index]
            # 獲取文檔的分詞結果
            doc_tokens: List[str] = tokenized_corpus[actual_index]
            
            # 計算文檔的各項評分指標
            scores_dict: Dict[str, float] = {}
            (scores_dict['term_importance'], scores_dict['semantic_similarity'],
             scores_dict['query_coverage'], scores_dict['position_score'],
             scores_dict['term_density'], scores_dict['context_similarity'],
             intersection, frequency) = self._calculate_document_scores(
                tokenized_corpus, doc_tokens, expanded_query, doc_freq, query_vector)
            
            # 記錄BM25分數
            scores_dict['bm25_score'] = scores[actual_index]
            
            # 記錄檢索結果的詳細信息
            logger.info(f'BM25 Rank {doc_index + 1}: PDF {chunk_key[0]}, Chunk {chunk_key[1]}, '
                      f'Score: {scores_dict["bm25_score"]:.4f}, '
                      f'Term Importance: {scores_dict["term_importance"]:.4f}, '
                      f'Semantic Similarity: {scores_dict["semantic_similarity"]:.4f}, '
                      f'Query Coverage: {scores_dict["query_coverage"]:.4f}, '
                      f'Position Score: {scores_dict["position_score"]:.4f}, '
                      f'Term Density: {scores_dict["term_density"]:.4f}, '
                      f'Context Similarity: {scores_dict["context_similarity"]:.4f}, '
                      f'Frequency: {frequency.most_common(10)}')
                      
            return chunk_key, scores_dict
            
        except ValueError:
            logger.error(f'Chunk not found in chunked_corpus: {doc_content}')
            return None, {}

    def _calculate_query_features(self, tokenized_query: Set[str], expanded_query: Set[str]) -> Tuple[int, float]:
        """
        計算查詢的複雜度特徵
        
        Args:
            tokenized_query (Set[str]): 分詞後的原始查詢詞集合，例如 {"台灣", "總統府", "哪裡"}
            expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"台灣", "總統府", "哪裡", "中華民國", "政府", "位置"}
            
        Returns:
            Tuple[int, float]:
                - query_length (int): 原始查詢的詞數量，例如 3 (對應上述例子)
                - query_diversity (float): 查詢擴展的多樣性,計算為擴展詞數量與原始詞數量的比值，例如 2.0 (6/3)
                
        Example:
            >>> processor = TextProcessor(...)
            >>> tokenized = {"台灣", "總統府", "哪裡"}
            >>> expanded = {"台灣", "總統府", "哪裡", "中華民國", "政府", "位置"}
            >>> length, diversity = processor._calculate_query_features(tokenized, expanded)
            >>> print(length, diversity)
            3 2.0
        """
        query_length = len(tokenized_query)  # 查詢長度
        query_diversity = len(expanded_query) / (len(tokenized_query) + 1)  # 查詢擴展的多樣性
        return query_length, query_diversity

    def _process_query(self, qs: str) -> Tuple[Set[str], Set[str]]:
        """
        處理查詢字串，包括分詞和查詢擴展
        
        Args:
            qs (str): 原始查詢字串，例如"台灣的總統府在哪裡?"
            
        Returns:
            Tuple[Set[str], Set[str]]:
                - tokenized_query (Set[str]): 分詞後的原始查詢詞集合，例如 {"台灣", "總統府", "哪裡"}
                - expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"台灣", "總統府", "哪裡", "中華民國", "政府", "位置"}
                
        Notes:
            - 使用jieba進行分詞
            - 使用word2vec進行查詢擴展
            - 記錄查詢相關日誌
            
        Example:
            >>> processor = TextProcessor(...)
            >>> tokenized, expanded = processor._process_query("台灣的總統府在哪裡?")
            >>> print(tokenized)
            {'台灣', '總統府', '哪裡'}
            >>> print(expanded)  
            {'台灣', '總統府', '哪裡', '中華民國', '政府', '位置'}
        """
        # 對查詢語句進行分詞
        tokenized_query = set(self.jieba_cut_with_stopwords(qs))
        
        # 使用word2vec進行查詢擴展
        expanded_query = self.expanded_query(tokenized_query)

        # 記錄查詢相關信息
        logger.info(f'query分詞結果: {"/".join(tokenized_query)}')
        unique_expanded_query = set(expanded_query) - set(tokenized_query)
        logger.info(f'query擴展詞: {"/".join(unique_expanded_query)}')
        
        return tokenized_query, expanded_query

    def _calculate_retrieval_scores(self, ans: List[str], chunked_corpus: List[str], 
                                  key_idx_map: List[Tuple[int, int]], tokenized_corpus: List[List[str]],
                                  scores: List[float], expanded_query: Set[str], 
                                  doc_freq: Counter[str], query_vector: np.ndarray) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, float]]]:
        """
        計算檢索結果的各項評分指標
        
        Args:
            ans (List[str]): 檢索到的文檔內容列表，例如 ["台灣總統府位於台北市中正區...", "總統府是一棟巴洛克式建築..."]
            chunked_corpus (List[str]): 分塊後的語料庫，例如 ["台灣總統府位於台北市中正區...", "總統府是一棟巴洛克式建築..."]
            key_idx_map (List[Tuple[int, int]]): 每個文本片段對應的(file_key, chunk_index)列表，例如 [(1, 0), (1, 1)]
            tokenized_corpus (List[List[str]]): 分詞後的語料庫，例如 [["台灣", "總統府", ...], ["總統府", "巴洛克", ...]]
            scores (List[float]): BM25分數列表，例如 [0.8, 0.6]
            expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"台灣", "總統府", "位置"}
            doc_freq (Counter[str]): 詞頻統計，例如 Counter({"台灣": 10, "總統府": 5})
            query_vector (np.ndarray): 查詢向量，例如 array([0.2, 0.3, 0.1])
            
        Returns:
            Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, float]]]:
                - retrieved_keys: 檢索到的文檔鍵值列表，例如 [(1, 0), (1, 1)]
                - score_dicts: 包含各項評分指標的字典，例如
                    {
                        (1, 0): {
                            "bm25_score": 0.8,
                            "term_importance": 0.7,
                            "semantic_similarity": 0.6,
                            "query_coverage": 0.5,
                            "position_score": 0.4,
                            "term_density": 0.3,
                            "context_similarity": 0.5
                        },
                        (1, 1): {...}
                    }
                
        Notes:
            計算的評分指標包括:
            - BM25分數 (0-1): 基於詞頻和文檔長度的相關性分數
            - 詞項重要性 (0-1): 基於TF-IDF的詞重要性分數
            - 語義相似度 (0-1): 基於詞向量的語義相似度
            - 查詢覆蓋率 (0-1): 查詢詞在文檔中的覆蓋程度
            - 詞位置分數 (0-1): 查詢詞在文檔中的位置分布
            - 詞密度 (0-1): 查詢詞在文檔中的密集程度
            - 上下文相似度 (0-1): 查詢詞周圍上下文的相似度
            
        Example:
            >>> processor = TextProcessor()
            >>> ans = ["台灣總統府位於台北市中正區", "總統府是一棟巴洛克式建築"]
            >>> chunked_corpus = ["台灣總統府位於台北市中正區", "總統府是一棟巴洛克式建築"]
            >>> key_idx_map = [(1, 0), (1, 1)]
            >>> tokenized_corpus = [["台灣", "總統府"], ["總統府", "巴洛克"]]
            >>> scores = [0.8, 0.6]
            >>> expanded_query = {"台灣", "總統府", "位置"}
            >>> doc_freq = Counter({"台灣": 10, "總統府": 5})
            >>> query_vector = np.array([0.2, 0.3, 0.1])
            >>> keys, scores = processor._calculate_retrieval_scores(
            ...     ans, chunked_corpus, key_idx_map, tokenized_corpus,
            ...     scores, expanded_query, doc_freq, query_vector
            ... )
            >>> print(keys)
            [(1, 0), (1, 1)]
            >>> print(scores[(1, 0)]["bm25_score"])
            0.8
        """
        retrieved_keys: List[Tuple[int, int]] = []
        score_dicts: Dict[Tuple[int, int], Dict[str, float]] = {}
        
        for index, doc in enumerate(ans):
            chunk_key, scores_dict = self._process_retrieved_document(
                index, doc, chunked_corpus, key_idx_map, tokenized_corpus,
                scores, expanded_query, doc_freq, query_vector
            )
            
            if chunk_key:
                retrieved_keys.append(chunk_key)
                score_dicts[chunk_key] = scores_dict
                
        return retrieved_keys, score_dicts

    def _get_top_documents(self, scores: List[float], chunked_corpus: List[str], n: int = 5) -> List[str]:
        """
        根據BM25分數獲取前N個最相關的文檔
        
        Args:
            scores (List[float]): 每個文檔的BM25分數列表，例如 [0.8, 0.6, 0.4]
            chunked_corpus (List[str]): 分塊後的文檔內容列表，例如 ["文檔1內容", "文檔2內容"]
            n (int, optional): 要返回的最相關文檔數量. 默認為5
            
        Returns:
            List[str]: 包含前N個最相關文檔內容的列表，例如 ["最相關文檔內容", "次相關文檔內容"]
            
        Example:
            >>> processor = TextProcessor()
            >>> scores = [0.8, 0.6, 0.4]
            >>> corpus = ["文檔1", "文檔2", "文檔3"]
            >>> top_docs = processor._get_top_documents(scores, corpus, n=2)
            >>> print(top_docs)
            ["文檔1", "文檔2"]
            
        Notes:
            1. 使用sorted和lambda函數對分數進行降序排序
            2. 獲取前N個最高分數的文檔索引
            3. 根據索引從語料庫中提取對應的文檔內容
        """
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [chunked_corpus[i] for i in top_n_indices]

    def _get_faiss_or_bm25_result(self, retrieved_keys: List[Tuple[int, int]], score_results: Dict[str, Dict[Tuple[int, int], float]],
                                  faiss_results: Dict[Tuple[int, int], float], query_length: int,
                                  query_diversity: float, weighted_scores: Optional[Dict[Tuple[int, int], float]]) -> Optional[Tuple[int, int]]:
        """
        根據FAISS和BM25結果決定最終返回的文檔ID
        
        Args:
            retrieved_keys (List[Tuple[int, int]]): BM25檢索到的文檔鍵值列表，每個元素為(文檔ID, chunk_idx)的元組
            score_results (Dict[str, Dict[Tuple[int, int], float]]): 包含各項BM25評分指標的字典，格式為{'bm25_results': {(doc_id, chunk_idx): score}, ...}
            faiss_results (Dict[Tuple[int, int], float]): FAISS檢索結果字典，格式為{(文檔ID,chunk_idx): 相似度分數}
            query_length (int): 查詢長度，即查詢詞的數量
            query_diversity (float): 查詢多樣性分數，範圍0-1
            weighted_scores (Optional[Dict[Tuple[int, int], float]]): 計算後的加權分數字典，格式為{(文檔ID, chunk_idx): 加權分數}
            
        Returns:
            Optional[Tuple[int, int]]: 返回得分最高的文檔鍵值(doc_id, chunk_idx)，如果沒有找到任何結果則返回None
            
        Notes:
            1. 如果同時有FAISS和BM25結果，返回加權分數最高的文檔
            2. 如果只有BM25結果，返回BM25最佳結果
            3. 如果沒有任何結果，返回None
        """
        # 如果有FAISS結果，使用加權分數
        if score_results['bm25_results'] and faiss_results:
            # 記錄最終排序結果
            logger.info('-'*100)
            logger.info('Final Rankings (Adaptive Weights):')
            logger.info(f'Query Features - Length: {query_length}, Diversity: {query_diversity:.2f}')
            
            # 返回得分最高的文檔鍵值
            if weighted_scores:
                best_key = max(weighted_scores.items(), key=lambda x: x[1])[0]
                return best_key[0]

        # 如果只有BM25結果，返回BM25最佳結果
        elif retrieved_keys:
            return retrieved_keys[0][0]
            
        # 如果沒有找到任何結果，返回None
        return None

    def BM25_retrieve(self, qs: str, source: List[int], corpus_dict: dict) -> Optional[int]:
        """
        使用BM25算法進行文檔檢索，並結合多個評分指標進行排序
        
        Args:
            qs (str): 查詢字符串，用戶輸入的問題
            source (List[int]): 源文檔ID列表，包含所有可能的文檔ID
            corpus_dict (dict): 語料庫字典，格式為{文檔ID: 文檔內容}，存儲所有文檔的內容
            
        Returns:
            Optional[int]: 返回最相關的文檔ID，如果沒有找到則返回None
            
        Example:
            >>> processor = TextProcessor(stopwords_filepath)
            >>> query = "如何申請專利?"
            >>> source = [1, 2, 3, 4, 5] 
            >>> corpus = {1: "專利申請流程...", 2: "商標註冊說明..."}
            >>> result = processor.BM25_retrieve(query, source, corpus)
            >>> print(result)
            1
        """
        # 1. 文檔預處理
        chunked_corpus, key_idx_map = self._prepare_corpus(source, corpus_dict)
        
        # 2. BM25基礎檢索
        bm25_results = self._process_and_score_bm25(chunked_corpus, qs)
        scores = bm25_results['scores']
        ans = bm25_results['top_docs']
        expanded_query = bm25_results['expanded_query']
        tokenized_query = bm25_results['tokenized_query']
        tokenized_corpus = bm25_results['tokenized_corpus']
        
        # 3. 特徵計算
        feature_results = self._analyze_query_features(tokenized_query, expanded_query, tokenized_corpus)
        doc_freq = feature_results['doc_freq']
        query_vector = feature_results['query_vector']
        query_length = feature_results['query_length']
        query_diversity = feature_results['query_diversity']
        
        # 4. 多維度評分
        retrieved_keys, score_dicts = self._calculate_retrieval_scores(
            ans, chunked_corpus, key_idx_map, tokenized_corpus,
            scores, expanded_query, doc_freq, query_vector
        )
        
        # 將評分結果分配到對應類別
        score_results = self._distribute_scores(score_dicts)
        
        # 5. 結果整合與返回
        return self._integrate_results(
            qs, chunked_corpus, key_idx_map, retrieved_keys, 
            score_results, query_length, query_diversity
        )

    def _process_and_score_bm25(self, chunked_corpus: List[str], query: str) -> Dict[str, Any]:
        """
        處理文檔並計算BM25分數
        
        Args:
            chunked_corpus (List[str]): 切分後的文檔列表，例如 ["文檔1內容", "文檔2內容", ...]
            query (str): 查詢字符串，例如 "如何申請專利?"
            
        Returns:
            Dict[str, Any]: 包含以下鍵值的字典:
                - scores (np.ndarray): BM25分數列表，例如 [0.8, 0.6, 0.4, ...]
                - top_docs (List[str]): 得分最高的文檔列表，例如 ["文檔1內容", "文檔2內容", ...]
                - expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"專利", "申請", "流程"}
                - tokenized_query (Set[str]): 原始查詢分詞結果，例如 {"專利", "申請"}
                - tokenized_corpus (List[List[str]]): 分詞後的文檔列表，例如 [["文檔", "內容"], ["文檔", "內容"]]
                
        Example:
            >>> processor = TextProcessor(...)
            >>> corpus = ["專利申請流程說明", "商標註冊相關規定"]
            >>> query = "如何申請專利?"
            >>> result = processor._process_and_score_bm25(corpus, query)
            >>> print(result)
            {
                'scores': array([0.8, 0.2]),
                'top_docs': ['專利申請流程說明'],
                'expanded_query': {'專利', '申請', '流程'},
                'tokenized_query': {'專利', '申請'},
                'tokenized_corpus': [['專利', '申請', '流程'], ['商標', '註冊']]
            }
        """
        tokenized_corpus = [self.jieba_cut_with_stopwords(doc) for doc in chunked_corpus]
        bm25 = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b, epsilon=0.5)
        tokenized_query, expanded_query = self._process_query(query)
        scores = bm25.get_scores(expanded_query)
        top_docs = self._get_top_documents(scores, chunked_corpus)
        
        return {
            'scores': scores,
            'top_docs': top_docs,
            'expanded_query': expanded_query,
            'tokenized_query': tokenized_query,
            'tokenized_corpus': tokenized_corpus
        }

    def _analyze_query_features(self, tokenized_query: List[str], expanded_query: Set[str], 
                              tokenized_corpus: List[List[str]]) -> Dict[str, Any]:
        """
        分析查詢特徵，包括文檔頻率、查詢向量和查詢特徵
        
        Args:
            tokenized_query (List[str]): 分詞後的原始查詢，例如 ["台灣", "總統府"]
            expanded_query (Set[str]): 擴展後的查詢，例如 {"台灣", "總統府", "政府"} 
            tokenized_corpus (List[List[str]]): 分詞後的文檔列表，例如 [["台灣", "總統府"], ["政府", "機關"]]
            
        Returns:
            Dict[str, Any]: 包含以下鍵值的字典:
                - doc_freq (Counter[str]): 文檔頻率字典，例如 Counter({"台灣": 2, "總統府": 1})
                - query_vector (np.ndarray): 查詢向量，例如 array([0.2, 0.3, 0.1])
                - query_length (int): 查詢長度，例如 2
                - query_diversity (float): 查詢多樣性分數，例如 0.8
                
        Example:
            >>> processor = TextProcessor()
            >>> result = processor._analyze_query_features(
            ...     ["台灣", "總統府"],
            ...     {"台灣", "總統府", "政府"},
            ...     [["台灣", "總統府"], ["政府", "機關"]]
            ... )
            >>> print(result)
            {
                'doc_freq': Counter({"台灣": 2, "總統府": 1}),
                'query_vector': array([0.2, 0.3, 0.1]),
                'query_length': 2,
                'query_diversity': 0.8
            }
        """
        doc_freq = self._calculate_doc_frequencies(tokenized_corpus)
        query_vector = self._calculate_query_vector(tokenized_query, tokenized_corpus, doc_freq)
        query_length, query_diversity = self._calculate_query_features(tokenized_query, expanded_query)
        
        return {
            'doc_freq': doc_freq,
            'query_vector': query_vector,
            'query_length': query_length,
            'query_diversity': query_diversity
        }

    def _integrate_results(self, query: str, chunked_corpus: List[str], key_idx_map: List[Tuple[int, int]],
                          retrieved_keys: List[Tuple[int, int]], score_results: Dict[str, Dict[Tuple[int, int], float]], 
                          query_length: int, query_diversity: float) -> Optional[Tuple[int, int]]:
        """
        整合檢索結果，結合FAISS和BM25的結果
        
        Args:
            query (str): 原始查詢字符串，例如 "台灣總統府在哪裡?"
            chunked_corpus (List[str]): 切分後的文檔列表，例如 ["台灣總統府...", "位於台北市..."]
            key_idx_map (List[Tuple[int, int]]): 文檔ID與索引的映射關係，例如 [(1, 0), (1, 1), (2, 0)]
            retrieved_keys (List[Tuple[int, int]]): BM25檢索到的文檔鍵值列表，例如 [(1, 0), (2, 0)]
            score_results (Dict[str, Dict[Tuple[int, int], float]]): 包含各項評分結果的字典，格式如:
                {
                    'bm25_results': {(1, 0): 0.8, (2, 0): 0.6},
                    'term_importance': {(1, 0): 0.7, (2, 0): 0.5},
                    'semantic_similarity': {(1, 0): 0.6, (2, 0): 0.4},
                    'query_coverage': {(1, 0): 0.8, (2, 0): 0.3},
                    'position_scores': {(1, 0): 0.9, (2, 0): 0.7},
                    'term_density': {(1, 0): 0.5, (2, 0): 0.4},
                    'context_similarity': {(1, 0): 0.7, (2, 0): 0.6}
                }
            query_length (int): 查詢長度，例如 5
            query_diversity (float): 查詢多樣性分數，例如 0.8
            
        Returns:
            Optional[Tuple[int, int]]: 最終選擇的文檔ID和chunk索引，例如 (1, 0)，如果沒有找到則返回None
            
        Example:
            >>> processor = TextProcessor()
            >>> result = processor._integrate_results(
            ...     "台灣總統府在哪裡?",
            ...     ["台灣總統府...", "位於台北市..."],
            ...     [(1, 0), (1, 1)],
            ...     [(1, 0)],
            ...     {'bm25_results': {(1, 0): 0.8}, ...},
            ...     5,
            ...     0.8
            ... )
            >>> print(result)
            (1, 0)
        """
        faiss_results: Dict[Tuple[int, int], float] = {}
        if self.use_faiss:
            faiss_results = self._faiss_retrieve(query, chunked_corpus, key_idx_map)

        weighted_scores: Optional[Dict[Tuple[int, int], float]] = None
        if score_results['bm25_results'] and faiss_results:
            weighted_scores = self._calculate_weighted_scores(
                score_results['bm25_results'], faiss_results, 
                score_results['term_importance'], score_results['semantic_similarity'],
                score_results['query_coverage'], score_results['position_scores'], 
                score_results['term_density'], score_results['context_similarity'],
                query_length, query_diversity
            )

        return self._get_faiss_or_bm25_result(retrieved_keys, score_results, faiss_results,
                                             query_length, query_diversity, weighted_scores)

    def _distribute_scores(self, score_dicts: Dict[Tuple[int, int], Dict[str, float]]) -> Dict[str, Dict[Tuple[int, int], float]]:
        """
        將計算的評分結果分配到對應的字典中
        
        Args:
            score_dicts (Dict[Tuple[int, int], Dict[str, float]]): 包含每個文檔所有評分指標的字典
                格式: {(doc_id, chunk_idx): {
                    'bm25_score': float,
                    'term_importance': float,
                    'semantic_similarity': float,
                    'query_coverage': float,
                    'position_score': float,
                    'term_density': float,
                    'context_similarity': float
                }}
                
        Returns:
            Dict[str, Dict[Tuple[int, int], float]]: 包含所有評分類型的字典
                格式: {
                    'bm25_results': {(doc_id, chunk_idx): score},
                    'term_importance': {(doc_id, chunk_idx): score},
                    'semantic_similarity': {(doc_id, chunk_idx): score},
                    'query_coverage': {(doc_id, chunk_idx): score},
                    'position_scores': {(doc_id, chunk_idx): score},
                    'term_density': {(doc_id, chunk_idx): score},
                    'context_similarity': {(doc_id, chunk_idx): score}
                }
                
        Example:
            >>> processor = TextProcessor()
            >>> scores = {
            ...     (1, 0): {
            ...         'bm25_score': 0.8,
            ...         'term_importance': 0.7,
            ...         'semantic_similarity': 0.6,
            ...         'query_coverage': 0.5,
            ...         'position_score': 0.4,
            ...         'term_density': 0.3,
            ...         'context_similarity': 0.5
            ...     }
            ... }
            >>> results = processor._distribute_scores(scores)
            >>> print(results['bm25_results'][(1, 0)])
            0.8
        """
        results: Dict[str, Dict[Tuple[int, int], float]] = {
            'bm25_results': {},
            'term_importance': {},
            'semantic_similarity': {},
            'query_coverage': {},
            'position_scores': {},
            'term_density': {},
            'context_similarity': {}
        }
        
        for chunk_key, scores in score_dicts.items():
            results['bm25_results'][chunk_key] = scores['bm25_score']
            results['term_importance'][chunk_key] = scores['term_importance']
            results['semantic_similarity'][chunk_key] = scores['semantic_similarity']
            results['query_coverage'][chunk_key] = scores['query_coverage']
            results['position_scores'][chunk_key] = scores['position_score']
            results['term_density'][chunk_key] = scores['term_density']
            results['context_similarity'][chunk_key] = scores['context_similarity']
            
        return results

    def _calculate_doc_frequencies(self, tokenized_corpus: List[List[str]]) -> Counter[str]:
        """
        計算文檔頻率(每個詞出現在多少文檔中)

        Args:
            tokenized_corpus (List[List[str]]): 分詞後的文檔集合，每個文檔是一個詞列表
                例如: [["台灣", "總統府"], ["台北", "101"]]

        Returns:
            Counter[str]: 包含每個詞的文檔頻率的Counter對象
                例如: Counter({"台灣": 1, "總統府": 1, "台北": 1, "101": 1})

        Example:
            >>> processor = TextProcessor()
            >>> corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> doc_freq = processor._calculate_doc_frequencies(corpus)
            >>> print(doc_freq)
            Counter({"台灣": 1, "總統府": 1, "台北": 1, "101": 1})

        Notes:
            - 對每個文檔中的詞進行去重，確保每個詞在一個文檔中只被計算一次
            - 使用Counter累加每個詞在不同文檔中的出現次數
            - 返回的Counter對象可用於計算IDF值和其他相關指標
        """
        doc_freq = Counter()
        for doc in tokenized_corpus:
            # 對每個文檔中的詞進行去重
            doc_freq.update(set(doc))
        return doc_freq

    def _calculate_weighted_scores(self, bm25_results: Dict[Tuple[int, int], float], 
                                 faiss_results: Dict[Tuple[int, int], float],
                                 term_importance: Dict[Tuple[int, int], float], 
                                 semantic_similarity: Dict[Tuple[int, int], float],
                                 query_coverage: Dict[Tuple[int, int], float], 
                                 position_scores: Dict[Tuple[int, int], float],
                                 term_density: Dict[Tuple[int, int], float], 
                                 context_similarity: Dict[Tuple[int, int], float],
                                 query_length: int, 
                                 query_diversity: float) -> Dict[Tuple[int, int], float]:
        """
        計算多指標加權分數
        
        Args:
            bm25_results (Dict[Tuple[int, int], float]): BM25檢索結果分數，格式為{(文檔ID,chunk_idx): 分數}
            faiss_results (Dict[Tuple[int, int], float]): FAISS檢索結果分數，格式為{(文檔ID,chunk_idx): 分數}
            term_importance (Dict[Tuple[int, int], float]): 詞項重要性分數，格式為{(文檔ID,chunk_idx): 分數}
            semantic_similarity (Dict[Tuple[int, int], float]): 語義相似度分數，格式為{(文檔ID,chunk_idx): 分數}
            query_coverage (Dict[Tuple[int, int], float]): 查詢覆蓋率分數，格式為{(文檔ID,chunk_idx): 分數}
            position_scores (Dict[Tuple[int, int], float]): 詞位置分數，格式為{(文檔ID,chunk_idx): 分數}
            term_density (Dict[Tuple[int, int], float]): 詞密度分數，格式為{(文檔ID,chunk_idx): 分數}
            context_similarity (Dict[Tuple[int, int], float]): 上下文相似度分數，格式為{(文檔ID,chunk_idx): 分數}
            query_length (int): 查詢長度，例如 3
            query_diversity (float): 查詢多樣性，例如 1.5
            
        Returns:
            Dict[Tuple[int, int], float]: 包含每個文檔最終加權分數的字典，格式為{(文檔ID,chunk_idx): 加權分數}
            
        Example:
            >>> processor = TextProcessor()
            >>> bm25_results = {(1,0): 0.8, (1,1): 0.6}
            >>> faiss_results = {(1,0): 0.7, (1,1): 0.5}
            >>> term_importance = {(1,0): 0.6, (1,1): 0.4}
            >>> semantic_similarity = {(1,0): 0.7, (1,1): 0.5}
            >>> query_coverage = {(1,0): 0.8, (1,1): 0.6}
            >>> position_scores = {(1,0): 0.7, (1,1): 0.5}
            >>> term_density = {(1,0): 0.6, (1,1): 0.4}
            >>> context_similarity = {(1,0): 0.7, (1,1): 0.5}
            >>> weighted_scores = processor._calculate_weighted_scores(
            ...     bm25_results, faiss_results, term_importance,
            ...     semantic_similarity, query_coverage, position_scores,
            ...     term_density, context_similarity, 3, 1.5
            ... )
            >>> print(weighted_scores)
            {(1,0): 0.7125, (1,1): 0.5125}
            
        Notes:
            - 對每個指標進行正規化處理
            - 根據查詢特徵動態調整權重
            - 計算加權總分
        """
        # 正規化分數的輔助函數
        def normalize_scores(scores):
            max_score = max(scores.values())
            min_score = min(scores.values())
            return {k: (v - min_score)/(max_score - min_score) if max_score != min_score else 1 
                   for k, v in scores.items()}

        # 正規化所有指標的分數
        normalized_bm25 = normalize_scores(bm25_results)
        normalized_faiss = {k: 1 - normalize_scores(faiss_results)[k] for k in faiss_results}
        normalized_importance = normalize_scores(term_importance)
        normalized_semantic = normalize_scores(semantic_similarity) if semantic_similarity else {}
        normalized_coverage = normalize_scores(query_coverage)
        normalized_position = normalize_scores(position_scores) if position_scores else {}
        normalized_density = normalize_scores(term_density) if term_density else {}
        normalized_context = normalize_scores(context_similarity) if context_similarity else {}

        # 設定基礎權重
        base_weights = {
            'bm25': 0.2,
            'faiss': 0.3,
            'importance': 0.10,
            'semantic': 0.10,
            'coverage': 0.1,
            'position': 0.1,
            'density': 0.05,
            'context': 0.05
        }
        
        # 根據查詢特徵調整權重
        adjusted_weights = base_weights.copy()
        if query_length <= 2:  # 短查詢
            adjusted_weights['semantic'] *= 1.2  # 增加語義相似度的權重
            adjusted_weights['context'] *= 1.2   # 增加上下文相似度的權重
            adjusted_weights['bm25'] *= 0.8      # 降低BM25的權重
        else:  # 長查詢
            adjusted_weights['bm25'] *= 1.2      # 增加BM25的權重
            adjusted_weights['coverage'] *= 1.2   # 增加查詢覆蓋率的權重
            adjusted_weights['semantic'] *= 0.8   # 降低語義相似度的權重
        
        if query_diversity > 1.5:  # 查詢擴展效果明顯
            adjusted_weights['semantic'] *= 1.2   # 增加語義相似度的權重
            adjusted_weights['context'] *= 1.2    # 增加上下文相似度的權重
        
        # 正規化調整後的權重
        weight_sum = sum(adjusted_weights.values())
        adjusted_weights = {k: v/weight_sum for k, v in adjusted_weights.items()}
        
        # 計算加權分數
        weighted_scores = {}
        for key in set(bm25_results.keys()) | set(faiss_results.keys()):
            score = (
                adjusted_weights['bm25'] * normalized_bm25.get(key, 0) +
                adjusted_weights['faiss'] * normalized_faiss.get(key, 0) +
                adjusted_weights['importance'] * normalized_importance.get(key, 0) +
                adjusted_weights['semantic'] * normalized_semantic.get(key, 0) +
                adjusted_weights['coverage'] * normalized_coverage.get(key, 0) +
                adjusted_weights['position'] * normalized_position.get(key, 0) +
                adjusted_weights['density'] * normalized_density.get(key, 0) +
                adjusted_weights['context'] * normalized_context.get(key, 0)
            )
            weighted_scores[key] = score

        # 記錄權重和分數詳情
        logger.info(f'Adjusted Weights: {adjusted_weights}')
        for key in sorted(weighted_scores.keys(), key=lambda k: weighted_scores[k], reverse=True):
            logger.info(
                f'PDF {key[0]}, Chunk {key[1]}: '
                f'Final Score={weighted_scores[key]:.4f}, '
                f'BM25={normalized_bm25.get(key, 0):.4f}, '
                f'FAISS={normalized_faiss.get(key, 0):.4f}, '
                f'Importance={normalized_importance.get(key, 0):.4f}, '
                f'Semantic={normalized_semantic.get(key, 0):.4f}, '
                f'Coverage={normalized_coverage.get(key, 0):.4f}, '
                f'Position={normalized_position.get(key, 0):.4f}, '
                f'Density={normalized_density.get(key, 0):.4f}, '
                f'Context={normalized_context.get(key, 0):.4f}'
            )

        return weighted_scores

    def _calculate_term_importance(self, intersection: Set[str], doc_tokens: List[str], 
                                 frequency: Counter[str], doc_freq: Counter[str], 
                                 tokenized_corpus: List[List[str]]) -> float:
        """
        計算詞項重要性得分
        
        Args:
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
            frequency (Counter[str]): 交集中每個詞在文檔中的頻率，例如 Counter({"台灣": 2, "總統府": 1})
            doc_freq (Counter[str]): 每個詞在所有文檔中的出現次數，例如 Counter({"台灣": 10, "總統府": 5})
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果，例如 [["台灣", "總統府"], ["台北", "101"]]
            
        Returns:
            float: 詞項重要性得分，範圍為0-1，例如 0.75
            
        Example:
            >>> processor = TextProcessor()
            >>> intersection = {"台灣", "總統府"}
            >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
            >>> frequency = Counter({"台灣": 2, "總統府": 1})
            >>> doc_freq = Counter({"台灣": 10, "總統府": 5})
            >>> tokenized_corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> score = processor._calculate_term_importance(
            ...     intersection, doc_tokens, frequency, doc_freq, tokenized_corpus)
            >>> print(score)
            0.75
            
        Notes:
            - 考慮詞頻(TF)、逆文檔頻率(IDF)和位置權重
            - 位置權重：詞的位置越靠前，權重越大
            - 最終得分是TF-IDF和位置權重的乘積
        """
        importance_score = 0.0
        for term in intersection:
            tf = float(frequency[term])  # 詞頻
            idf = float(np.log(len(tokenized_corpus) / doc_freq[term]))  # 逆文檔頻率
            # 獲取詞在文檔中的所有位置
            positions = [i for i, t in enumerate(doc_tokens) if t == term]
            # 計算位置權重(位置越靠前權重越大)
            position_weight = float(sum(1/(pos + 1) for pos in positions))
            importance_score += tf * idf * position_weight
        return float(importance_score)

    def _calculate_semantic_similarity(self, doc_tokens: List[str], query_vector: np.ndarray,
                                    doc_freq: Counter[str], tokenized_corpus: List[List[str]]) -> float:
        """
        計算語義相似度得分
        
        Args:
            doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
            query_vector (np.ndarray): 查詢向量，例如 array([0.2, 0.3, 0.1, ...])
            doc_freq (Counter[str]): 每個詞在所有文檔中的出現次數，例如 Counter({"台灣": 10, "總統府": 5})
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果，例如 [["台灣", "總統府"], ["台北", "101"]]
            
        Returns:
            float: 語義相似度得分，範圍為0-1，例如 0.75
            
        Example:
            >>> processor = TextProcessor()
            >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
            >>> query_vector = np.array([0.2, 0.3, 0.1])
            >>> doc_freq = Counter({"台灣": 10, "總統府": 5})
            >>> tokenized_corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> score = processor._calculate_semantic_similarity(
            ...     doc_tokens, query_vector, doc_freq, tokenized_corpus)
            >>> print(score)
            0.75
            
        Notes:
            - 使用加權文檔向量計算與查詢向量的餘弦相似度
            - 權重使用IDF值
        """
        doc_vector: np.ndarray = np.zeros(self.word2vec_model.vector_size)
        total_weight: float = 0.0
        for word in doc_tokens:
            if word in self.word2vec_model:
                weight: float = np.log(len(tokenized_corpus) / (doc_freq[word] + 1))
                doc_vector += weight * self.word2vec_model[word]
                total_weight += weight
        if total_weight > 0:
            doc_vector /= total_weight
            return float(cosine_similarity(query_vector.reshape(1, -1), doc_vector.reshape(1, -1))[0][0])
        return 0.0

    def _calculate_query_coverage(self, intersection: Set[str], expanded_query: Set[str],
                                tokenized_corpus: List[List[str]], doc_freq: Counter[str]) -> float:
        """
        計算查詢覆蓋率得分
        
        Args:
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"台灣", "總統府", "中華民國", "政府"}
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果，例如 [["台灣", "總統府"], ["台北", "101"]]
            doc_freq (Counter[str]): 每個詞在所有文檔中的出現次數，例如 Counter({"台灣": 10, "總統府": 5})
            
        Returns:
            float: 查詢覆蓋率得分，範圍為0-1，例如 0.75
            
        Example:
            >>> processor = TextProcessor()
            >>> intersection = {"台灣", "總統府"}
            >>> expanded_query = {"台灣", "總統府", "中華民國", "政府"}
            >>> tokenized_corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> doc_freq = Counter({"台灣": 10, "總統府": 5})
            >>> score = processor._calculate_query_coverage(
            ...     intersection, expanded_query, tokenized_corpus, doc_freq)
            >>> print(score)
            0.75
            
        Notes:
            - 考慮查詢詞在文檔中的覆蓋程度
            - 使用IDF加權計算覆蓋率
        """
        if len(expanded_query) == 0:
            return 0.0
            
        weighted_coverage: float = sum(np.log(len(tokenized_corpus) / doc_freq[term]) 
                                     for term in intersection)
        return float(weighted_coverage / len(expanded_query))

    def _calculate_position_score(self, doc_tokens: List[str], intersection: Set[str]) -> float:
        """
        計算詞位置得分
        
        Args:
            doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            
        Returns:
            float: 詞位置得分，範圍為0-1，例如 0.8。分數越高表示查詢詞位置越靠前且分布越集中
            
        Example:
            >>> processor = TextProcessor()
            >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
            >>> intersection = {"台灣", "總統府"}
            >>> score = processor._calculate_position_score(doc_tokens, intersection)
            >>> print(score)
            0.8
            
        Notes:
            - 考慮查詢詞在文檔中的位置分布
            - 位置越靠前且分布越集中，得分越高
            - 使用平均位置和標準差計算分數
        """
        positions: List[int] = []
        for i, token in enumerate(doc_tokens):
            if token in intersection:
                positions.append(i)
        if positions:
            avg_pos: float = np.mean(positions)  # 平均位置
            std_pos: float = np.std(positions) if len(positions) > 1 else 0  # 位置標準差
            return float(1 / (1 + std_pos + avg_pos/len(doc_tokens)))
        return 0.0

    def _calculate_term_density(self, doc_tokens: List[str], intersection: Set[str],
                              tokenized_corpus: List[List[str]], doc_freq: Counter[str]) -> float:
        """
        計算詞密度得分
        
        Args:
            doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果，例如 [["台灣", "總統府"], ["台北", "101"]]
            doc_freq (Counter[str]): 每個詞在所有文檔中的出現次數，例如 Counter({"台灣": 2, "總統府": 1})
            
        Returns:
            float: 詞密度得分，範圍為0-1，例如 0.8。分數越高表示查詢詞在局部區域的密度越大
            
        Example:
            >>> processor = TextProcessor()
            >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
            >>> intersection = {"台灣", "總統府"}
            >>> tokenized_corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> doc_freq = Counter({"台灣": 2, "總統府": 1})
            >>> score = processor._calculate_term_density(
            ...     doc_tokens, intersection, tokenized_corpus, doc_freq)
            >>> print(score)
            0.8
            
        Notes:
            - 使用滑動窗口計算查詢詞的局部密度
            - 考慮IDF權重
        """
        window_size = min(20, len(doc_tokens))  # 動態窗口大小
        max_density = 0.0
        for i in range(len(doc_tokens) - window_size + 1):
            window = doc_tokens[i:i + window_size]
            weighted_matches = sum(np.log(len(tokenized_corpus) / doc_freq[w]) 
                                for w in window if w in intersection)
            density = weighted_matches / window_size
            max_density = max(max_density, density)
        return float(max_density)

    def _calculate_context_similarity(self, doc_tokens: List[str], intersection: Set[str]) -> float:
        """
        計算上下文相似度得分
        
        Args:
            doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            
        Returns:
            float: 上下文相似度得分，範圍為0-1，例如 0.8。分數越高表示查詢詞與上下文的語義相似度越高
            
        Example:
            >>> processor = TextProcessor()
            >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
            >>> intersection = {"台灣", "總統府"}
            >>> score = processor._calculate_context_similarity(doc_tokens, intersection)
            >>> print(score)
            0.8
            
        Notes:
            - 計算查詢詞與其上下文詞的語義相似度
            - 使用固定大小的上下文窗口
            - 使用word2vec計算詞向量的餘弦相似度
        """
        context_size: int = 3  # 上下文窗口大小
        context_score: float = 0.0
        
        for i, token in enumerate(doc_tokens):
            if token in intersection:
                # 獲取詞的上下文窗口
                start: int = max(0, i - context_size)
                end: int = min(len(doc_tokens), i + context_size + 1)
                context: List[str] = doc_tokens[start:end]
                
                # 計算詞與上下文的相似度
                for c_token in context:
                    if c_token in self.word2vec_model and token in self.word2vec_model:
                        similarity: float = cosine_similarity(
                            self.word2vec_model[token].reshape(1, -1),
                            self.word2vec_model[c_token].reshape(1, -1)
                        )[0][0]
                        context_score += similarity
                        
        return float(context_score / (len(intersection) + 1))

    def _calculate_query_vector(self, tokenized_query: Set[str], tokenized_corpus: List[List[str]], doc_freq: Counter[str]) -> np.ndarray:
        """
        計算查詢向量，使用word2vec加權平均

        Args:
            tokenized_query (Set[str]): 分詞後的查詢詞集合，例如 {"台灣", "總統府"}
            tokenized_corpus (List[List[str]]): 分詞後的文檔集合，例如 [["台灣", "總統府"], ["台北", "101"]]
            doc_freq (Counter[str]): 詞頻統計字典，例如 Counter({"台灣": 2, "總統府": 1})

        Returns:
            np.ndarray: 查詢向量，使用word2vec加權平均計算得到，例如 array([0.2, 0.3, 0.1,...])
            
        Example:
            >>> processor = TextProcessor()
            >>> query = {"台灣", "總統府"}
            >>> corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> doc_freq = Counter({"台灣": 2, "總統府": 1})
            >>> vector = processor._calculate_query_vector(query, corpus, doc_freq)
            >>> print(vector.shape)
            (300,)
            
        Notes:
            - 使用IDF作為權重計算查詢向量
            - 對於不在word2vec模型中的詞會被忽略
            - 如果沒有任何詞的權重，返回零向量
        """
        query_vector: np.ndarray = np.zeros(self.word2vec_model.vector_size)
        total_weight: float = 0.0
        
        for word in tokenized_query:
            if word in self.word2vec_model:
                # 使用IDF作為權重
                weight: float = np.log(len(tokenized_corpus) / (doc_freq[word] + 1))
                query_vector += weight * self.word2vec_model[word]
                total_weight += weight
                
        if total_weight > 0:
            query_vector /= total_weight
            
        return query_vector

    def _prepare_corpus(self, source: List[int], corpus_dict: Dict[int, Union[str, List[Any]]]) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        準備語料庫數據，將原始文檔切分並建立索引映射
        
        Args:
            source (List[int]): 來源文件ID列表，例如 [1, 2, 3]
            corpus_dict (Dict[int, Union[str, List[Any]]]): 語料庫字典，key為文件ID，value為文件內容或Document列表
                例如 {1: "文件1內容", 2: [Document對象1, Document對象2]}
            
        Returns:
            Tuple[List[str], List[Tuple[int, int]]]: 包含以下兩個元素:
                - chunked_corpus (List[str]): 切分後的文本片段列表，例如 ["片段1", "片段2"]
                - key_idx_map (List[Tuple[int, int]]): 每個文本片段對應的(file_key, chunk_index)列表
                    例如 [(1, 0), (1, 1), (2, 0)]
                    
        Example:
            >>> processor = TextProcessor()
            >>> source = [1, 2]
            >>> corpus_dict = {
            ...     1: "文件1內容",
            ...     2: [Document(page_content="片段1"), Document(page_content="片段2")]
            ... }
            >>> chunked, key_map = processor._prepare_corpus(source, corpus_dict)
            >>> print(chunked)
            ["文件1內容", "片段1", "片段2"]
            >>> print(key_map)
            [(1, 0), (2, 0), (2, 1)]
        """
        chunked_corpus: List[str] = []  # 存儲所有切分後的文本片段
        key_idx_map: List[Tuple[int, int]] = []  # 存儲每個文本片段對應的(file_key, chunk_index)
        
        # 遍歷每個來源文件ID
        for file_key in source:
            # 獲取對應文件的內容
            corpus = corpus_dict[int(file_key)]
            # 對每個文件內容進行切分
            for idx, chunk in enumerate(corpus):
                key_idx_map.append((file_key, idx))
                try:    
                    # 如果是Document對象,取其page_content屬性
                    chunked_corpus.append(chunk.page_content)
                except AttributeError:
                    # 如果不是Document對象,直接添加文本內容
                    chunked_corpus.append(corpus)
                    break
                    
        return chunked_corpus, key_idx_map
