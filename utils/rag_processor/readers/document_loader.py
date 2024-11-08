from langchain_core.documents import Document
from utils.rag_processor.config import RAGProcessorConfig
from utils.rag_processor.readers.pdf_reader import read_pdf
from typing import List, Dict
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import logging
import json

logger = logging.getLogger(__name__)
class DocumentLoader:
    # 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
    @staticmethod
    def load_data(source_path: str, files_to_load: List[int], config: RAGProcessorConfig, source_type: str) -> Dict[int, List[Document]]:
        """
        Load reference data from PDF files in a directory.

        Args:
            source_path (str): The directory path containing PDF files.
            files_to_load (List[int]): List of file IDs to load.
            config (RAGProcessorConfig): Configuration object.

        Returns:
            Dict[int, List[Document]]: Mapping of file ID to list of Documents.
        """
        if config.source_type == "pdf":
            return DocumentLoader.load_data_pdf(source_path, files_to_load, config)
        elif config.source_type == "json":
            return DocumentLoader.load_json_data(source_path, files_to_load)
        else:
            raise ValueError(f"Unsupported source type: {config.source_type}")

        
    @staticmethod
    def load_data_pdf(source_path: str, files_to_load: List[int], config: RAGProcessorConfig) -> Dict[int, List[Document]]:
        """
        載入參考資料
        
        Args:
            source_path (str): PDF文件所在目錄路徑
            files_to_load (List[int]): 要載入的文件ID列表
            
        Returns:
            Dict[int, List[Document]]: 文件ID到Document列表的映射
            
        Example:
            >>> source_path = "./data/pdf"
            >>> files = [1, 2, 3]
            >>> corpus = load_data(source_path, files)
            >>> print(len(corpus[1]))  # 第一個文件的Document數量
            5
        """
        #masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
        masked_file_ls = files_to_load  # 指定資料夾中的檔案列表
        corpus_dict = {}
        
        # 使用多進程處理PDF文件
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(read_pdf, os.path.join(source_path, f'{file}.pdf'), config): file for file in masked_file_ls}
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                try:
                    corpus_dict[file] = future.result()
                except Exception as e:
                    logger.error(f"Error processing file {file}: {e}")

        return corpus_dict
    
    @staticmethod
    def load_json_data(source_path: str, files_to_load: List[int]) -> Dict[int, List[Document]]:
        """
        Load JSON data directly from JSON files in a directory.

        Args:
            source_path (str): The directory path containing JSON files.
            files_to_load (List[int]): List of file IDs to load.

        Returns:
            Dict[int, List[Document]]: Mapping of file ID to list of Documents.
        """
        corpus_dict = {}

        for file_id in files_to_load:
            json_path = os.path.join(source_path, f"{file_id}.json")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    document = Document(
                        page_content=data["contents"],  # Assumes 'contents' holds the main text
                        metadata={"id": data["id"]}
                    )
                    corpus_dict[file_id] = [document]  # Wrap in a list to match load_data output
            except Exception as e:
                logger.error(f"Error loading JSON file {json_path}: {e}")

        return corpus_dict
__all__ = ["DocumentLoader"]