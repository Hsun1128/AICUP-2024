import os
import re
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from typing import List, Set
from collections import Counter
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
from gensim.models import KeyedVectors  # 確保引入gensim

from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, filename='retrieve.log', filemode='w', format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# 設置全局參數
RANGE = range(0, 150)

USE_EXPANSION = 0
EXPANSION_MODEL_PATH = '../word2vec/wiki.zh.bin'
EXPANDED_TOPN = 2

USE_FAISS = 1

BM25_K1 = 0.5
BM25_B = 0.7

CHUNK_SIZE = 500
OVERLAP = 100


# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path, files_to_load):
    #masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    masked_file_ls = files_to_load  # 指定資料夾中的檔案列表
    corpus_dict = {}
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(read_pdf, os.path.join(source_path, f'{file}.pdf')): file for file in masked_file_ls}
        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            try:
                corpus_dict[file] = future.result()
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")

    return corpus_dict


splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, length_function=len, is_separator_regex=False, keep_separator=False, separators=['\n\n', '\n', '!', '?', '。', ';'])
# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, splitter=splitter, page_infos: list=None):
    #pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件
    pdf_text = PDFPlumberLoader(pdf_loc, extract_images = True).load()
    for doc in pdf_text:
        # 清理內容
        clean_content = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', doc.page_content).replace('\n', '').strip()
        doc.page_content = clean_content
    pdf_text = splitter.split_documents(pdf_text)
    """
    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text.replace(" ", "").replace("\n", "")  # 去除內容中的空格和換行符
    pdf.close()  # 關閉PDF文件
    """
    return pdf_text  # 返回萃取出的文本

class TextProcessor:
    def __init__(self, stopwords_filepath: str, bm25_k1: float=BM25_K1, bm25_b: float=BM25_B, use_expansion: bool=USE_EXPANSION, expanded_topn: int=EXPANDED_TOPN, chunk_size: int=CHUNK_SIZE, overlap: int=OVERLAP, expansion_model_path: str=EXPANSION_MODEL_PATH, use_faiss: bool=USE_FAISS):
        self.stopwords = self.load_stopwords(stopwords_filepath)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.expanded_topn = expanded_topn
        self.use_expansion = use_expansion
        self.word2vec_model = self.load_word2vec_model(os.path.join(os.path.dirname(__file__), expansion_model_path))  # 載入word2vec模型
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, length_function=len, is_separator_regex=False, keep_separator=False, separators=['\n\n', '\n', '!', '?', '。', ';'])
        self.use_faiss = use_faiss
        if self.use_faiss:
            self.embeddings = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-m3', model_kwargs = {'device': 'cuda'}, encode_kwargs = {'normalize_embeddings': True})

    def load_word2vec_model(self, model_path: str):
        model = KeyedVectors.load(model_path)  # 載入二進制格式的word2vec模型
        logger.info('Word2Vec model loaded successfully')
        return model

    def load_stopwords(self, filepath: str) -> set[str]:
        with open(filepath, 'r', encoding='utf-8') as file:
            stopwords = set(line.strip() for line in file)
        logger.info('Loading stopwords success')
        logger.info(f'stopwords: {stopwords}')
        return stopwords

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

    def expanded_query(self, tokenized_query: Set[str]) -> Set[str]:
        if not self.use_expansion:
            return tokenized_query  # 如果不使用擴展，返回原始查詢
        
        expanded_query = []
        for word in tokenized_query:
            if word in self.word2vec_model.key_to_index:
                topn_words = self.word2vec_model.most_similar(word, topn=self.expanded_topn)
                topn_words = set([w[0] for w in topn_words if w[0] not in self.stopwords and w[0].strip() and len(w[0]) != 1])  # clean stopwords
                expanded_query.extend([word] + list(topn_words))  # 包含原始詞和擴展詞
        return set(expanded_query)

    # 根據查詢語句和指定的來源，檢索答案
    def BM25_retrieve(self, qs: str, source: List[int], corpus_dict: dict):
        chunked_corpus = []
        key_idx_map = []  # 用於存儲每個chunk對應的(file_key, idx)
        # 將文檔切分並存儲到列表中，同時記錄映射關係
        for file_key in source:
            corpus = corpus_dict[int(file_key)]
            for idx, chunk in enumerate(corpus):
                key_idx_map.append((file_key, idx))
                try:    
                    chunked_corpus.append(chunk.page_content)
                except:
                    chunked_corpus.append(corpus)
                    break
                
        # 分詞並建立 BM25 模型
        tokenized_corpus = [self.jieba_cut_with_stopwords(doc) for doc in chunked_corpus]
        bm25 = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b, epsilon=0.5)

        # 查詢分詞
        tokenized_query = set(self.jieba_cut_with_stopwords(qs))
        
        # 使用word2vec擴展查詢詞
        expanded_query = self.expanded_query(tokenized_query)

        # 獲取所有文檔的分數
        scores = bm25.get_scores(expanded_query)
        # 獲取分數最高的文檔索引
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        # 根據索引獲取對應的文檔
        ans = [chunked_corpus[i] for i in top_n_indices]

        # 查看搜索結果
        logger.info(f'query分詞結果: {"/".join(tokenized_query)}')
        unique_expanded_query = set(expanded_query) - set(tokenized_query)
        logger.info(f'query擴展詞: {"/".join(unique_expanded_query)}')
        
        retrieved_keys = []
        bm25_results = {}
        term_importance = {}
        semantic_similarity = {}  # 語義相似度指標
        query_coverage = {}      # 查詢覆蓋率指標
        position_scores = {}     # 詞位置得分
        term_density = {}        # 詞密度得分
        context_similarity = {}  # 上下文相似度
        
        # 計算詞頻-逆文檔頻率(TF-IDF)
        doc_freq = Counter()
        for doc in tokenized_corpus:
            doc_freq.update(set(doc))
        
        # 計算查詢向量 (使用word2vec加權平均)
        query_vector = np.zeros(self.word2vec_model.vector_size)
        total_weight = 0
        for word in tokenized_query:
            if word in self.word2vec_model:
                # 使用IDF作為權重
                weight = np.log(len(tokenized_corpus) / (doc_freq[word] + 1))
                query_vector += weight * self.word2vec_model[word]
                total_weight += weight
        if total_weight > 0:
            query_vector /= total_weight
        
        # 計算查詢的複雜度特徵
        query_length = len(tokenized_query)
        query_diversity = len(expanded_query) / (len(tokenized_query) + 1)  # 查詢擴展的多樣性
        
        for index, a in enumerate(ans):
            try:
                actual_index = chunked_corpus.index(a)
                chunk_key = key_idx_map[actual_index]
                doc_tokens = tokenized_corpus[actual_index]
                intersection = set(expanded_query) & set(doc_tokens)
                frequency = Counter(token for token in doc_tokens if token in intersection)
                
                # 計算詞的重要性得分 (加入位置權重)
                importance_score = 0
                for term in intersection:
                    tf = frequency[term]
                    idf = np.log(len(tokenized_corpus) / doc_freq[term])
                    positions = [i for i, t in enumerate(doc_tokens) if t == term]
                    position_weight = sum(1/(pos + 1) for pos in positions)  # 越前面的位置權重越大
                    importance_score += tf * idf * position_weight
                term_importance[chunk_key] = importance_score
                
                # 計算語義相似度 (使用加權文檔向量)
                doc_vector = np.zeros(self.word2vec_model.vector_size)
                total_weight = 0
                for word in doc_tokens:
                    if word in self.word2vec_model:
                        weight = np.log(len(tokenized_corpus) / (doc_freq[word] + 1))
                        doc_vector += weight * self.word2vec_model[word]
                        total_weight += weight
                if total_weight > 0:
                    doc_vector /= total_weight
                    semantic_similarity[chunk_key] = cosine_similarity(
                        query_vector.reshape(1, -1), 
                        doc_vector.reshape(1, -1)
                    )[0][0]
                
                # 計算查詢覆蓋率 (考慮詞的重要性)
                weighted_coverage = sum(np.log(len(tokenized_corpus) / doc_freq[term]) 
                                     for term in intersection)
                query_coverage[chunk_key] = weighted_coverage / len(expanded_query)
                
                # 計算詞位置得分
                positions = []
                for i, token in enumerate(doc_tokens):
                    if token in intersection:
                        positions.append(i)
                if positions:
                    avg_pos = np.mean(positions)
                    std_pos = np.std(positions) if len(positions) > 1 else 0
                    position_scores[chunk_key] = 1 / (1 + std_pos + avg_pos/len(doc_tokens))
                
                # 計算詞密度得分 (使用滑動窗口)
                window_size = min(20, len(doc_tokens))  # 動態窗口大小
                max_density = 0
                for i in range(len(doc_tokens) - window_size + 1):
                    window = doc_tokens[i:i + window_size]
                    weighted_matches = sum(np.log(len(tokenized_corpus) / doc_freq[w]) 
                                        for w in window if w in intersection)
                    density = weighted_matches / window_size
                    max_density = max(max_density, density)
                term_density[chunk_key] = max_density
                
                # 計算上下文相似度
                context_size = 3  # 上下文窗口大小
                context_score = 0
                for i, token in enumerate(doc_tokens):
                    if token in intersection:
                        start = max(0, i - context_size)
                        end = min(len(doc_tokens), i + context_size + 1)
                        context = doc_tokens[start:end]
                        for c_token in context:
                            if c_token in self.word2vec_model and token in self.word2vec_model:
                                context_score += cosine_similarity(
                                    self.word2vec_model[token].reshape(1, -1),
                                    self.word2vec_model[c_token].reshape(1, -1)
                                )[0][0]
                context_similarity[chunk_key] = context_score / (len(intersection) + 1)
                
                score = scores[actual_index]
                bm25_results[chunk_key] = score
                
                logger.info(f'BM25 Rank {index + 1}: PDF {chunk_key[0]}, Chunk {chunk_key[1]}, '
                          f'Score: {score:.4f}, Term Importance: {importance_score:.4f}, '
                          f'Semantic Similarity: {semantic_similarity.get(chunk_key, 0):.4f}, '
                          f'Query Coverage: {query_coverage[chunk_key]:.4f}, '
                          f'Position Score: {position_scores.get(chunk_key, 0):.4f}, '
                          f'Term Density: {term_density.get(chunk_key, 0):.4f}, '
                          f'Context Similarity: {context_similarity.get(chunk_key, 0):.4f}, '
                          f'Frequency: {frequency.most_common(10)}')
                retrieved_keys.append(chunk_key)
            except ValueError:
                logger.error(f'Chunk not found in chunked_corpus: {a}')

        faiss_results = {}
        if self.use_faiss:
            vector_store = FAISS.from_texts(chunked_corpus, self.embeddings, normalize_L2=True)
            faiss_ans = vector_store.similarity_search_with_score(qs, k=5)

            logger.info('-'*100)
            for doc, score in faiss_ans:
                faiss_actual_index = chunked_corpus.index(doc.page_content)
                faiss_chunk_key = key_idx_map[faiss_actual_index]
                faiss_results[faiss_chunk_key] = score
                logger.info(f'FAISS Score: [{score:.4f}], PDF: {faiss_chunk_key[0]}, '
                          f'Chunk: {faiss_chunk_key[1]}, metadata: [{doc.metadata}]')

        # 使用多指標排序和自適應權重系統
        if bm25_results and faiss_results:
            def normalize_scores(scores):
                max_score = max(scores.values())
                min_score = min(scores.values())
                return {k: (v - min_score)/(max_score - min_score) if max_score != min_score else 1 
                       for k, v in scores.items()}

            # 正規化所有指標
            normalized_bm25 = normalize_scores(bm25_results)
            normalized_faiss = {k: 1 - normalize_scores(faiss_results)[k] for k in faiss_results}
            normalized_importance = normalize_scores(term_importance)
            normalized_semantic = normalize_scores(semantic_similarity) if semantic_similarity else {}
            normalized_coverage = normalize_scores(query_coverage)
            normalized_position = normalize_scores(position_scores) if position_scores else {}
            normalized_density = normalize_scores(term_density) if term_density else {}
            normalized_context = normalize_scores(context_similarity) if context_similarity else {}

            # 根據查詢特徵動態調整權重
            base_weights = {
                'bm25': 0.2,
                'faiss': 0.2,
                'importance': 0.15,
                'semantic': 0.15,
                'coverage': 0.1,
                'position': 0.1,
                'density': 0.05,
                'context': 0.05
            }
            
            # 根據查詢特徵調整權重
            adjusted_weights = base_weights.copy()
            if query_length <= 2:  # 短查詢
                adjusted_weights['semantic'] *= 1.2
                adjusted_weights['context'] *= 1.2
                adjusted_weights['bm25'] *= 0.8
            else:  # 長查詢
                adjusted_weights['bm25'] *= 1.2
                adjusted_weights['coverage'] *= 1.2
                adjusted_weights['semantic'] *= 0.8
            
            if query_diversity > 1.5:  # 查詢擴展效果明顯
                adjusted_weights['semantic'] *= 1.2
                adjusted_weights['context'] *= 1.2
            
            # 正規化調整後的權重
            weight_sum = sum(adjusted_weights.values())
            adjusted_weights = {k: v/weight_sum for k, v in adjusted_weights.items()}
            
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

            logger.info('-'*100)
            logger.info('Final Rankings (Adaptive Weights):')
            logger.info(f'Query Features - Length: {query_length}, Diversity: {query_diversity:.2f}')
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

            if weighted_scores:
                best_key = max(weighted_scores.items(), key=lambda x: x[1])[0]
                return best_key[0]

        elif retrieved_keys:
            return retrieved_keys[0][0]
            
        return None


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--load_path', type=str, default="../custom_dicts/with_frequency", help='自定義字典的路徑（可選）')  # 自定義字典的路徑
    parser.add_argument('--zhTW_dict_path', type=str, default="../custom_dicts/dict.txt.big", help='繁中字典的路徑（可選）')  # 繁中字典的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    # 載入繁中字典 
    jieba.set_dictionary(args.zhTW_dict_path)
    # 載入自定義字典的路徑
    if os.path.exists(args.load_path):
        # 遍歷所有 .txt 文件
        for filename in os.listdir(args.load_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(args.load_path, filename)
                jieba.load_userdict(file_path)
                logger.info(f"載入自定義字典: {file_path}")
            else:
                logger.info(f"沒有自定義字典，只載入原始字典")

    processor = TextProcessor('../custom_dicts/stopwords.txt')

    logger.info(f'BM25_K1: {BM25_K1}')
    logger.info(f'BM25_B: {BM25_B}')

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    """
    # 載入所有資料集
    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance)

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance)
    """

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    for q_dict in tqdm((q for q in qs_ref['questions'] if q['qid']-1 in RANGE), total=len(RANGE), desc="Processing questions"):
        logger.info(f'{"="*65} QID: {q_dict["qid"]} {"="*65}')
        if q_dict['category'] == 'finance':
            source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
            corpus_dict_finance = load_data(source_path_finance, q_dict['source'])
            # 進行檢索
            retrieved = processor.BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定考資料路徑
            corpus_dict_insurance = load_data(source_path_insurance, q_dict['source'])
            retrieved = processor.BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = processor.BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
    logger.info(f'已成功儲存檔案於 {args.output_path}\n')
