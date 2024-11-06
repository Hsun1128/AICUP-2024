import os

# 設定詞頻
default_frequency = 500

# 讀取load_path中的所有.txt文件，為每個詞添加詞頻
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
load_path = os.path.join(current_dir,'./custom_dicts/origin_dict')
save_path = os.path.join(current_dir,'./custom_dicts/with_frequency')

# 確保保存的目錄存在
os.makedirs(save_path, exist_ok=True)

# 遍歷load_path中的所有.txt文件
for filename in os.listdir(load_path):
    if filename.endswith('.txt'):
        load_filename = os.path.join(load_path, filename)
        with open(load_filename, 'r', encoding='utf-8') as file:
            words = file.readlines()  # 讀取每行詞彙

        # 將詞頻添加到每行並保存
        save_filename = os.path.join(save_path, f"{os.path.splitext(filename)[0]}_with_frequency.txt")
        with open(save_filename, 'w', encoding='utf-8') as file:
            for word in words:
                word = word.strip()  # 去除每行詞彙後的空白字元
                if word:  # 確保行不為空
                    file.write(f"{word} {default_frequency}\n")  # 加上詞頻並寫入新文件

        print(f"詞頻已添加並儲存為 {save_filename}")
