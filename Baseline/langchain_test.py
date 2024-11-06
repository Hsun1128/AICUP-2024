import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader

source_path = os.path.join('../CompetitionDataset/reference', 'finance')  # 設定參考資料路徑

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

print(os.path.join(source_path, f'31.pdf'))

document = PDFPlumberLoader(os.path.join(source_path, f'1.pdf')).load_and_split(text_splitter)
import pdfplumber

with pdfplumber.open(os.path.join(source_path, f'1.pdf')) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                print(row)  # 每列的內容

#print(f'all: {document}')

for chunk in document:
    print(f'chunk: {chunk}')
