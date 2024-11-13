import json
import os
from tqdm import tqdm
import argparse

def main(folder, input_root, output_root):
    """
    將FAQ數據從原始格式轉換為指定的JSON格式
    
    參數:
        folder (str): 數據文件夾名稱
        input_root (str): 輸入文件根目錄路徑
        output_root (str): 輸出文件根目錄路徑
    """
    # 檢查並創建輸出目錄
    if os.path.exists(os.path.join(output_root, folder)) is False:
        os.makedirs(os.path.join(output_root, folder))

    # 讀取原始FAQ數據
    with open(os.path.join(input_root, folder, 'pid_map_content.json'), 'r') as f:
        data = json.load(f)
        
    files = list(data.keys())

    # 遍歷處理每個文件的FAQ數據
    for file in tqdm(files):
        text = ""
        questions = data[file]
        # 將每個問題及其答案轉換為指定格式
        for question in questions:
            # 問題和答案用製表符分隔，多個答案用逗號連接
            text += question['question'] + '\t'
            text += ", ".join(question['answers']) + '\n'

        # 創建輸出格式
        output = {
            "id": file,          # 文件ID
            "contents": text     # 格式化後的FAQ內容
        }

        # 將處理後的數據保存為JSON文件
        with open(os.path.join(output_root, folder, f'{output["id"]}.json'), 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

# 命令行使用示例：
# python faq_format.py --folder faq --input_root ../CompetitionDataset/reference --output_root ../CompetitionDataset/ref_contents
if __name__ == '__main__':
    # 設置命令行參數解析
    parser = argparse.ArgumentParser(description='FAQ數據格式轉換工具')
    parser.add_argument('--folder', type=str, help='數據文件夾名稱')
    parser.add_argument('--input_root', type=str, help='輸入文件根目錄路徑')
    parser.add_argument('--output_root', type=str, help='輸出文件根目錄路徑')
    args = parser.parse_args()

    # 執行主程序
    main(args.folder, args.input_root, args.output_root)