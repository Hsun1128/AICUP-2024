from langchain_core.documents import Document
from utils.rag_processor.config import RAGProcessorConfig
from utils.rag_processor.readers.pdf_reader import read_pdf
from utils.rag_processor.readers.json_reader import load_single_json
from typing import List, Dict
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from tqdm import tqdm  # 新增進度條庫

logger = logging.getLogger(__name__)
class DocumentLoader:
    # 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
    @staticmethod
    def auto_load_data(source_path: str, files_to_load: List[int]=None, config: RAGProcessorConfig=RAGProcessorConfig()) -> Dict[int, List[Document]]:
        """
        Load reference data from files in a directory, automatically detecting file type.

        Args:
            source_path (str): The directory path containing the files.
            config (RAGProcessorConfig): Configuration object.
            files_to_load (List[int]): List of file IDs to load.

        Returns:
            Dict[int, List[Document]]: Mapping of file ID to list of Documents.
        """
        # Get first file extension to determine type
        if not files_to_load:
            files_to_load = os.listdir(source_path)
            test_path = os.path.splitext(os.path.join(source_path, os.listdir(source_path)[0]))[0]
        else:
            first_file = f"{files_to_load[0]}"
            test_path = os.path.join(source_path, first_file)
        
        # Try common extensions
        if os.path.exists(test_path + ".pdf"):
            return DocumentLoader.load_pdf_data(source_path, files_to_load, config)
        elif os.path.exists(test_path + ".json"):
            return DocumentLoader.load_json_data(source_path, files_to_load, config)
        else:
            raise ValueError(f"Could not detect supported file type in {source_path}")

        
    @staticmethod
    def load_pdf_data(source_path: str, files_to_load: List[int], config: RAGProcessorConfig) -> Dict[int, List[Document]]:
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
    def load_json_data(source_path: str, files_to_load: List[int], config: RAGProcessorConfig) -> Dict[int, List[Document]]:
        """
        Load JSON data directly from JSON files in a directory using multi-threading.

        Args:
            source_path (str): The directory path containing JSON files.
            files_to_load (List[int]): List of file IDs to load.

        Returns:
            Dict[int, List[Document]]: Mapping of file ID to list of Documents.
        """
        masked_file_ls = files_to_load  # 指定資料夾中的檔案列表
        corpus_dict = {}

        # Use ThreadPoolExecutor for I/O bound JSON loading
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(load_single_json, file_id, source_path, config): file_id for file_id in masked_file_ls}
            for future in concurrent.futures.as_completed(futures):
                try:
                    file_id, documents = future.result()
                    if documents:  # Only add if documents were successfully loaded
                        corpus_dict[file_id] = documents
                except Exception as e:
                    logger.error(f"Error processing future: {e}")

        return corpus_dict
    
__all__ = ["DocumentLoader"]