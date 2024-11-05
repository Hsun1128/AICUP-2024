import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader

source_path = os.path.join('../競賽資料集/reference', 'insurance')  # 設定參考資料路徑

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

text = "Hello, world!"
print(os.path.join(source_path, f'1.pdf'))

document = PDFPlumberLoader(os.path.join(source_path, f'1.pdf')).load_and_split(text_splitter)
#print(f'all: {document}')

for chunk in document:
    print(f'chunk: {chunk}')
