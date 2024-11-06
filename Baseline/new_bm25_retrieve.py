import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具
import pytesseract # 用於提取PDF中的圖片內容
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
import concurrent.futures
import logging
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 新增 tokenizer 初始化
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')


# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {}
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(read_pdf, os.path.join(source_path, file)): file for file in masked_file_ls}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(masked_file_ls)):
            file = futures[future]
            try:
                corpus_dict[int(file.replace('.pdf', ''))] = future.result()
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    image_extraction_attempted = False

    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text.replace(" ", "").replace("\n", "")  # 去除內容中的空格和換行符
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本


# 根據查詢語句和指定的來源，檢索答案
def BM25_retrieve(qs, source, corpus_dict):
    # 設定分塊參數
    chunk_size = 500  # 每個chunk的字元數
    chunk_overlap = 200  # overlap的字元數
    
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    # 將長文本分塊
    chunked_corpus = []
    doc_to_chunks = {}  # 追蹤每個chunk屬於哪個原始文檔
    
    for doc_idx, doc in enumerate(filtered_corpus):
        # 如果文本長度小於chunk_size，直接加入
        if len(doc) <= chunk_size:
            chunked_corpus.append(doc)
            doc_to_chunks[len(chunked_corpus)-1] = doc_idx
            continue
            
        # 將長文本分塊
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunk = doc[start:end]
            chunked_corpus.append(chunk)
            doc_to_chunks[len(chunked_corpus)-1] = doc_idx
            start = end - chunk_overlap

    # 對分塊後的corpus進行檢索
    tokenized_corpus = [tokenizer.tokenize(chunk) for chunk in chunked_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenizer.tokenize(qs)
    
    # 獲取最相關的chunk
    top_chunks = bm25.get_top_n(tokenized_query, chunked_corpus, n=5)
    
    # 找出這些chunks對應的原始文檔
    chunk_scores = {}
    for chunk in top_chunks:
        chunk_idx = chunked_corpus.index(chunk)
        doc_idx = doc_to_chunks[chunk_idx]
        chunk_scores[doc_idx] = chunk_scores.get(doc_idx, 0) + 1
    
    # 選擇得分最高的文檔
    best_doc_idx = max(chunk_scores.items(), key=lambda x: x[1])[0]
    best_doc = filtered_corpus[best_doc_idx]
    
    # 找回對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == best_doc]
    return res[0]


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--load_path', type=str, default="../custom_dicts/with_frequency", help='自定義字典的路徑（可選）')  # 自定義字典的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    # 移除 jieba 相關的字典載入程式碼
    if os.path.exists(args.load_path):
        print('使用 BAAI/bge-m3 tokenizer，忽略自定義字典')

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance)

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance)

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            # 進行檢索
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
    print(f'已成功儲存檔案於 {args.output_path}')
