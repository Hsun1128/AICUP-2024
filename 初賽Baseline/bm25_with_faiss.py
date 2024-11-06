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
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 在檔案開頭初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

# 初始化模型和設備
model = AutoModel.from_pretrained('BAAI/bge-m3')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 全局配置參數
MAX_LENGTH = 1024  # tokenizer 最大長度
CHUNK_SIZE = 500  # 文本分塊大小
OVERLAP_SIZE = 200  # 重疊部分大小
BATCH_SIZE = 32  # 批處理大小
TOP_K_BM25 = 5  # BM25 篩選文檔數
TOP_K_FAISS = 1  # Faiss 返回結果數
BM25_K1 = 1.6  # BM25 k1 參數
BM25_B = 1  # BM25 b 參數

def get_embeddings(texts, batch_size=BATCH_SIZE):
    try:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                encoded = tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=MAX_LENGTH, 
                    return_tensors='pt'
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.no_grad():
                    outputs = model(**encoded)
                    batch_embeddings = outputs.last_hidden_state[:, 0]
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                    embeddings.append(batch_embeddings.cpu().numpy())
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue
        return np.concatenate(embeddings, axis=0)
    except Exception as e:
        logger.error(f"Error in get_embeddings: {e}")
        raise

def create_chunks_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE):
    """創建重疊的文本塊，並生成每個塊的概述"""
    chunks = []
    positions = []
    start = 0
    
    while start < len(text):
        # 確定當前塊的結束位置
        end = min(start + chunk_size, len(text))
        
        # 如果不是第一個塊，往前擴展重疊部分
        if start > 0:
            start = max(start - overlap, 0)
        
        chunk = text[start:end]
        chunks.append(chunk)
        positions.append((start, end))
        
        # 下一個塊的起始位置
        start = end
    
    # 為每個塊創建概述（包含位置信息和上下文）
    chunk_overviews = []
    for i, (chunk, (start, end)) in enumerate(zip(chunks, positions)):
        context = f"位置 {start}-{end}"
        if i > 0:
            # 添加前文參考
            context += f" | 前文: {chunks[i-1][-50:]}"
        if i < len(chunks) - 1:
            # 添加後文參考
            context += f" | 後文: {chunks[i+1][:50]}"
        
        chunk_overviews.append(f"{context} | 內容: {chunk}")
    
    return chunks, chunk_overviews, positions

# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {}
    
    with ProcessPoolExecutor() as executor:
        # 建立任務列表
        futures = {executor.submit(read_pdf, os.path.join(source_path, file)): file for file in masked_file_ls}
        # 使用 tqdm 顯示進度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(masked_file_ls), desc="Loading PDFs"):
            file = futures[future]
            try:
                corpus_dict[int(file.replace('.pdf', ''))] = future.result()
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
    
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
        """
        else:
            
            if not image_extraction_attempted:
                logger.info('嘗試擷取圖片中的內容...')
                image_extraction_attempted = True

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
                logger.warning('擷取失敗', exc_info=True)
        """
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本


def prepare_chunks(filtered_corpus):
    """準備文檔分塊和概述"""
    all_chunks = []
    all_overviews = []
    doc_to_chunks = {}
    current_chunk_idx = 0
    
    for doc_idx, doc in enumerate(filtered_corpus):
        chunks, overviews, _ = create_chunks_with_overlap(doc)
        all_chunks.extend(chunks)
        all_overviews.extend(overviews)
        
        chunk_count = len(chunks)
        doc_to_chunks[doc_idx] = list(range(current_chunk_idx, current_chunk_idx + chunk_count))
        current_chunk_idx += chunk_count
        
    return all_chunks, all_overviews, doc_to_chunks

def perform_bm25_search(all_chunks, tokenizer, query):
    """執行BM25搜索"""
    tokenized_corpus = [tokenizer.tokenize(chunk) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)
    tokenized_query = tokenizer.tokenize(query)
    return bm25.get_scores(tokenized_query)

def get_best_docs(chunk_scores, doc_to_chunks, top_k=TOP_K_BM25):
    """計算並返回最佳文檔索引"""
    doc_final_scores = {}
    for chunk_idx, score in enumerate(chunk_scores):
        for doc_idx, chunk_indices in doc_to_chunks.items():
            if chunk_idx in chunk_indices:
                if doc_idx not in doc_final_scores:
                    doc_final_scores[doc_idx] = []
                doc_final_scores[doc_idx].append(score)
                break
    
    return sorted(
        doc_final_scores.keys(),
        key=lambda x: sum(sorted(doc_final_scores[x], reverse=True)[:top_k])/top_k,
        reverse=True
    )[:top_k]

def create_faiss_index(overview_embeddings):
    """創建和配置Faiss索引"""
    res = faiss.StandardGpuResources()
    dimension = overview_embeddings.shape[1]
    nlist = max(1, int(len(overview_embeddings) ** 0.5 * 0.23))
    
    if len(overview_embeddings) < 80:
        index = faiss.IndexFlatIP(dimension)
    else:
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(overview_embeddings)
        index.nprobe = min(nlist, 3)
    
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(overview_embeddings)
    return index

def BM25_retrieve(qs, source, corpus_dict):
    """主函數：結合BM25和向量搜索進行文檔檢索"""
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    # 準備文檔分塊
    all_chunks, all_overviews, doc_to_chunks = prepare_chunks(filtered_corpus)
    
    # BM25搜索
    chunk_scores = perform_bm25_search(all_chunks, tokenizer, qs)
    
    # 獲取最佳文檔
    best_doc_indices = get_best_docs(chunk_scores, doc_to_chunks)
    
    # 準備候選文檔
    candidate_overviews = []
    for doc_idx in best_doc_indices:
        doc_chunks = doc_to_chunks[doc_idx]
        candidate_overviews.extend([all_overviews[i] for i in doc_chunks])
    
    # 向量搜索
    query_embedding = get_embeddings([qs])
    overview_embeddings = get_embeddings(candidate_overviews)
    
    # 創建和使用Faiss索引
    index = create_faiss_index(overview_embeddings)
    similarities, indices = index.search(query_embedding, TOP_K_FAISS)
    
    # 找出最佳匹配文檔
    best_chunk_idx = indices[0][0]
    chunk_count = 0
    for doc_idx in best_doc_indices:
        doc_chunk_count = len(doc_to_chunks[doc_idx])
        if best_chunk_idx < chunk_count + doc_chunk_count:
            best_match_idx = doc_idx
            break
        chunk_count += doc_chunk_count
    
    # 返回結果
    matched_doc = filtered_corpus[best_match_idx]
    return next(key for key, value in corpus_dict.items() if value == matched_doc)


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--load_path', type=str, default="../custom_dicts/with_frequency", help='自定義字典的路徑（可選）')  # 自定義字典的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

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
