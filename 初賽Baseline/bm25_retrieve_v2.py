import os
import re
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from typing import List, Set
from collections import Counter
import logging

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
    pdf_text = PDFPlumberLoader(pdf_loc).load()
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
        for index, a in enumerate(ans):
            try:
                actual_index = chunked_corpus.index(a)
                chunk_key = key_idx_map[actual_index]
                intersection = set(expanded_query) & set(tokenized_corpus[actual_index])
                frequency = Counter(token for token in tokenized_corpus[actual_index] if token in intersection)
                score = scores[actual_index]
                logger.info(f'Rank {index + 1}: PDF {chunk_key[0]}, Chunk {chunk_key[1]}, Score: {score:.4f}, words frequency: {frequency.most_common(10)}')
                retrieved_keys.append(chunk_key)
            except ValueError:
                logger.error(f'Chunk not found in chunked_corpus: {a}')

        #print(retrieved_keys)
        for file_key, chunk in retrieved_keys:
            pass
            #print(f'file_key: {file_key}, chunk: {chunk}, {corpus_dict[int(file_key)][chunk]}')

        if self.use_faiss:
            vector_store = FAISS.from_texts(chunked_corpus, self.embeddings, normalize_L2=True)
            faiss_ans = vector_store.similarity_search_with_score(qs, k=3)

            print('-'*100)
            faiss_retrieved_keys = []
            for doc, score in faiss_ans:
                faiss_actual_index = chunked_corpus.index(doc.page_content)
                faiss_chunk_key = key_idx_map[faiss_actual_index]
                logger.info(f'*[{score:.4f}], PDF: {faiss_chunk_key[0]}, Chunk: {faiss_chunk_key[1]}, metadata: [{doc.metadata}]')
                faiss_retrieved_keys.append(faiss_chunk_key)

        # 返回所有相關的檔案名或根據需求返回最佳匹配
        if retrieved_keys:
            return retrieved_keys[0][0]  # 返回最相關的檔案名
        else:
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
