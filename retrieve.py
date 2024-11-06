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
