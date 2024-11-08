import re
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.document_loaders import PDFPlumberLoader
from utils.rag_processor.config import RAGProcessorConfig

def read_pdf(pdf_loc: str, config: RAGProcessorConfig, page_infos: Optional[List[int]]=None) -> List[Document]:
    """
    讀取PDF文件並進行文本處理
    
    Args:
        pdf_loc (str): PDF文件路徑
        splitter (RecursiveCharacterTextSplitter): 文本分割器
        page_infos (Optional[List[int]]): 頁面範圍，如[0,5]表示讀取前5頁
        
    Returns:
        List[Document]: 處理後的Document列表
    """
    pdf_text = PDFPlumberLoader(pdf_loc).load()
    for doc in pdf_text:
        # 清理內容
        clean_content = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', doc.page_content).replace('\n', '').strip()  # 移除中文字之間的空白
        clean_content = re.sub(r'([,、\$]){3,}', '', clean_content)  # 移除連續3個以上的標點符號
        clean_content = re.sub(r'([\u4e00-\u9fff])\1+', r'\1', clean_content)  # 移除連續重複的中文字
        clean_content = re.sub(r'-\s*\d+\s*-', '', clean_content)  # 移除頁碼格式
        clean_content = re.sub(r'第\s*\d+\s*頁，\s*共\s*\d+\s*頁', '', clean_content)  # 移除頁碼說明
        doc.page_content = clean_content
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.overlap, length_function=len, is_separator_regex=False, keep_separator=False, separators=['\n\n', '\n', '!', '?', '。', ';'])
    pdf_text = splitter.split_documents(pdf_text)
    return pdf_text  # 返回萃取出的文本 

__all__ = ["read_pdf"]