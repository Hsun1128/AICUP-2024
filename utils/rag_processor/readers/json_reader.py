import json
import logging
import os
import re
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.rag_processor.config import RAGProcessorConfig

logger = logging.getLogger(__name__)

def load_single_json(file_id: int, source_path: str, config: RAGProcessorConfig) -> tuple[int, List[Document]]:
    json_path = os.path.join(source_path, f"{file_id}.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 清理內容
            clean_content = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', data["contents"]).replace('\n', '').strip()  # 移除中文字之間的空白
            clean_content = re.sub(r'([,、\$]){3,}', '', clean_content)  # 移除連續3個以上的標點符號
            clean_content = re.sub(r'([\u4e00-\u9fff])\1+', r'\1', clean_content)  # 移除連續重複的中文字
            clean_content = re.sub(r'-\s*\d+\s*-', '', clean_content)  # 移除頁碼格式
            clean_content = re.sub(r'第\s*\d+\s*頁，\s*共\s*\d+\s*頁', '', clean_content)  # 移除頁碼說明
            document = Document(
                page_content=clean_content,
                metadata={"id": data["id"]}
            )
            splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.overlap, length_function=len, is_separator_regex=False, keep_separator=False, separators=['\n\n', '\n', '!', '?', '。', ';'])
            document = splitter.split_documents([document])
            return file_id, document
    except Exception as e:
        logger.error(f"Error loading JSON file {json_path}: {e}")
        return file_id, []
    
__all__ = ["load_single_json"]