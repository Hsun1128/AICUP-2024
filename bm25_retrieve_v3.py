# 導入所需的套件
import os
import json
import argparse
import logging
from dataclasses import asdict

# 導入自定義環境設定
from utils.env import load_env
from utils.RAGProcessor import Retrieval, RAGProcessorConfig
from utils.rag_processor import DocumentLoader

# 導入數據處理相關套件
from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具


# 設定日誌記錄
logging.basicConfig(level=logging.INFO, filename='retrieve.log', filemode='w', format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# 載入環境變數
load_env()

RANGE = range(0, 150)  # 處理問題的範圍

if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--load_path', type=str, default="./custom_dicts/with_frequency", help='自定義字典的路徑（可選）')  # 自定義字典的路徑
    parser.add_argument('--zhTW_dict_path', type=str, default="./custom_dicts/dict.txt.big", help='繁中字典的路徑（可選）')  # 繁中字典的路徑

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

    # 初始化文本處理器
    config = RAGProcessorConfig.from_yaml('./config.yaml')
    retrieval = Retrieval(config)

    logger.info(f'Config:\n{json.dumps(asdict(config), indent=2)}')

    # 讀取問題文件
    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    # 預加載資料集
    if config.load_all_data:
        # 載入所有資料集
        source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
        corpus_dict_insurance = DocumentLoader.auto_load_data(source_path_insurance, config=config)

        source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
        corpus_dict_finance = DocumentLoader.auto_load_data(source_path_finance, config=config)

        source_path_faq = os.path.join(args.source_path, 'faq')  # 設定考資料路徑
        corpus_dict_faq = DocumentLoader.auto_load_data(source_path_faq, config=config)
    

    # 處理每個問題
    for q_dict in tqdm((q for q in qs_ref['questions'] if q['qid']-1 in RANGE), total=len(RANGE), desc="Processing questions"):
        logger.info(f'{"="*65} QID: {q_dict["qid"]} {"="*65}')
        if q_dict['category'] == 'finance':
            if not config.load_all_data:
                source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
                corpus_dict_finance = DocumentLoader.auto_load_data(source_path_finance, q_dict['source'], config)
            # 進行檢索
            retrieved = retrieval.BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            if not config.load_all_data:
                source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定考資料路徑
                corpus_dict_insurance = DocumentLoader.auto_load_data(source_path_insurance, q_dict['source'], config)
            retrieved = retrieval.BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            if not config.load_all_data:
                source_path_faq = os.path.join(args.source_path, 'faq')  # 設定考資料路徑
                corpus_dict_faq = DocumentLoader.auto_load_data(source_path_faq, q_dict['source'], config)
            retrieved = retrieval.BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
    logger.info(f'已成功儲存檔案於 {args.output_path}\n')
