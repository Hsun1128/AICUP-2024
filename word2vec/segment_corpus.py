import os
import json
import jieba
import pdfplumber  # 用於從PDF文件中提取文字的工具
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import logging
from tqdm import tqdm  # Import tqdm for progress bar
from collections import Counter  # Import Counter to calculate word frequency
import re

logging.basicConfig(level=logging.INFO, filename=os.path.join(os.path.dirname(__file__), 'segment_corpus.log'), filemode='w', format='%(asctime)s:%(message)s')
logger = logging.getLogger(__name__)

# 載入停用詞
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set(line.strip() for line in file)
    return stopwords

# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件
    pdf_lines = []  # 初始化一个列表以存储逐行文本
    
    for page in pdf.pages:  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            text = text.replace(' ', '').replace('\n', '')  # 去除內容中的空格和換行符
            lines = re.sub(r'[，。;；,.]', '\n', text).split('\n')  # 拆分每個段落成一行
            pdf_lines.extend(line.strip() for line in lines if line.strip())  # 添加非空行到列表

    pdf.close()  # 關閉PDF文件

    return pdf_lines  # 返回逐行的文本列表

# 載入資料集
def load_data(source_path):
    corpus_dict = {}
    pdf_files = [filename for filename in os.listdir(source_path) if filename.endswith('.pdf')]
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(read_pdf, os.path.join(source_path, filename)): filename for filename in pdf_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing PDFs'):
            filename = futures[future]
            try:
                corpus_dict[filename] = future.result()
            except Exception as e:
                print(f'Error processing file {filename}: {e}')
    return corpus_dict

# 斷詞並儲存結果及詞頻
def segment_and_save(corpus_dict, stopwords, output_filepath):
    word_counter = Counter()  # Initialize a Counter to count word frequencies
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for filename, lines in tqdm(corpus_dict.items(), desc='Segmenting and saving', total=len(corpus_dict)):
            for text in lines:  # 遍歷每一行文本
                words = jieba.cut(text)  # 進行分词
                filtered_words = [word for word in words if word not in stopwords and word.strip()]
                word_counter.update(filtered_words)  # Update the counter with filtered words
                f.write(' '.join(filtered_words) + '\n')  # 每行寫入分詞结果

            #f.write('\n')  # 每篇文章之間空兩行

    # Output the word frequency using logger
    for word, count in word_counter.most_common():
        logger.info(f'{word}: {count}')  # Log word frequency

# 新增函數以處理JSON資料
def process_json_data(json_data):
    json_dict = {}  # 初始化一個字典以存儲問題和答案
    for key, value in json_data.items():
        questions_answers = []  # 初始化一個列表以存儲問題和答案對
        for item in value:
            question = item['question']
            answers = item['answers']
            # 將問題和答案組成字典並添加到列表中
            questions_answers.append(question)  # 將問題添加到列表中
            for answer in answers:
                questions_answers.append(answer)  # 將每個答案添加到列表中
        json_dict[key] = questions_answers  # 使用key作為字典的鍵
    return json_dict  # 返回包含問題和答案的字典

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 獲取該檔案的位置
    source_path_insurance = os.path.join(current_dir, '../競賽資料集/reference/insurance')  # 設定保險資料路徑
    source_path_finance = os.path.join(current_dir, '../競賽資料集/reference/finance')  # 設定金融資料路徑
    json_data_path = os.path.join(current_dir, '../競賽資料集/reference/faq/pid_map_content.json')  # 設定JSON資料路徑
    stopwords_filepath = os.path.join(current_dir, '../custom_dicts/stopwords.txt')  # 停用詞文件路徑
    output_filepath = os.path.join(current_dir, 'corpusSegDone.txt')  # 輸出文件路徑
    load_path = os.path.join(current_dir, '../custom_dicts/with_frequency')  # 自定義字典的路徑
    zhTW_dict_path = os.path.join(current_dir, '../custom_dicts/dict.txt.big')  # 自定義字典的路徑

    # 載入繁中字典
    jieba.set_dictionary(zhTW_dict_path)

    # 載入自定義字典
    if os.path.exists(load_path):
        for filename in os.listdir(load_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(load_path, filename)
                jieba.load_userdict(file_path)
                logger.info(f'載入自定義字典: {file_path}')
            else:
                logger.info(f'沒有自定義字典，只載入原始字典')

    stopwords = load_stopwords(stopwords_filepath)
    corpus_dict_insurance = load_data(source_path_insurance)  # 載入保險資料
    #corpus_dict_finance = load_data(source_path_finance)  # 載入金融資料

    # 載入JSON資料
    with open(json_data_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 將JSON資料轉換為列表
    corpus_list_json = process_json_data(json_data)  # 將每個問題和答案轉換為列表

    # 將保險、金融和JSON資料合併
    combined_corpus_dict = {**corpus_dict_insurance, **corpus_list_json,} # **corpus_dict_finance}  # 
    segment_and_save(combined_corpus_dict, stopwords, output_filepath) 
