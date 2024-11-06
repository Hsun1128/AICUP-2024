"""
有些東西需要確定被安裝:
pip install "unstructured[pdf]" langchain-unstructured tqdm
sudo apt-get install tesseract-ocr tesseract-ocr-chi-tra

執行指令:
python pdf_extractor.py /path/to/input/folder /path/to/output/folder

輸出結構：
output_folder/
├── [轉換後的JSON檔案]
└── logs/
    ├── pdf_extractor_20240106_123456.log  # 詳細日誌
    └── statistics_20240106_123456.json     # 統計資料
"""
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
import subprocess
from unstructured.partition.pdf import partition_pdf
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import time
from datetime import datetime
import sys

class CustomFormatter(logging.Formatter):
    """自定義日誌格式化器，添加顏色"""
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class PDFExtractor:
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = None):
        """
        初始化PDF提取器
        
        Args:
            input_dir (str): 輸入PDF檔案的資料夾路徑
            output_dir (str): 輸出JSON檔案的資料夾路徑
            max_workers (int, optional): 最大工作進程數
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers or max(1, psutil.cpu_count(logical=False) - 1)
        
        # 建立必要的資料夾
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日誌系統
        self._setup_logging()
        
        # 檢查依賴
        self._check_dependencies()
        
        # 初始化統計資料
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_processing_time': 0
        }

    def _setup_logging(self):
        """設置日誌系統"""
        # 獲取logger
        self.logger = logging.getLogger('PDFExtractor')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除現有的處理器
        self.logger.handlers = []
        
        # 建立日誌檔案名稱（包含時間戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'pdf_extractor_{timestamp}.log'
        
        # 檔案處理器（詳細日誌）
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台處理器（帶顏色）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = CustomFormatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 為每次執行建立一個統計檔案
        self.stats_file = self.log_dir / f'statistics_{timestamp}.json'

    def _check_dependencies(self):
        """檢查必要的依賴"""
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                 capture_output=True, 
                                 text=True)
            self.logger.info(f"Tesseract version: {result.stdout.split()[1]}")
            
            result = subprocess.run(['tesseract', '--list-langs'], 
                                 capture_output=True, 
                                 text=True)
            if 'chi_tra' not in result.stdout:
                self.logger.warning("Traditional Chinese language pack (chi_tra) not found!")
                self.logger.warning("Please install it for better Chinese text recognition.")
        except FileNotFoundError:
            self.logger.critical("Tesseract is not installed!")
            raise

    @staticmethod
    def process_single_pdf(args: Tuple[Path, Path]) -> Tuple[str, bool, str, float]:
        """
        處理單個PDF檔案
        
        Returns:
            Tuple[str, bool, str, float]: (檔案名, 是否成功, 訊息, 處理時間)
        """
        pdf_path, output_path = args
        start_time = time.time()
        
        try:
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",
                languages=["chi_tra", "eng"],
                ocr_config=r'--psm 1 --oem 1',
                infer_table_structure=True,
                max_partition=20,
                include_page_breaks=True
            )
            
            contents = []
            for element in elements:
                if hasattr(element, 'text'):
                    contents.append(element.text)
            
            content = "\n".join(contents)
            
            result = {
                "id": pdf_path.stem,
                "contents": content
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            process_time = time.time() - start_time
            return pdf_path.name, True, "Success", process_time
            
        except Exception as e:
            process_time = time.time() - start_time
            return pdf_path.name, False, str(e), process_time

    def update_statistics(self, success: bool, process_time: float):
        """更新處理統計資料"""
        self.stats['total_files'] += 1
        self.stats['total_processing_time'] += process_time
        if success:
            self.stats['successful_files'] += 1
        else:
            self.stats['failed_files'] += 1

    def save_statistics(self):
        """儲存統計資料到JSON檔案"""
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['total_files']
            if self.stats['total_files'] > 0 else 0
        )
        self.stats['success_rate'] = (
            self.stats['successful_files'] / self.stats['total_files'] * 100
            if self.stats['total_files'] > 0 else 0
        )
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # 記錄最終統計結果
        self.logger.info("處理完成！統計資料：")
        self.logger.info(f"總檔案數: {self.stats['total_files']}")
        self.logger.info(f"成功處理: {self.stats['successful_files']}")
        self.logger.info(f"處理失敗: {self.stats['failed_files']}")
        self.logger.info(f"平均處理時間: {self.stats['average_processing_time']:.2f}秒")
        self.logger.info(f"成功率: {self.stats['success_rate']:.1f}%")
        self.logger.info(f"詳細統計已儲存至: {self.stats_file}")

    def process_all_pdfs(self):
        """使用多進程處理所有PDF檔案"""
        self.stats['start_time'] = datetime.now().isoformat()
        
        # 取得所有PDF檔案
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"在 {self.input_dir} 中沒有找到PDF檔案")
            return
        
        self.logger.info(f"找到 {len(pdf_files)} 個PDF檔案")
        self.logger.info(f"使用 {self.max_workers} 個工作進程進行平行處理")
        
        # 準備處理參數
        process_args = [
            (pdf_path, self.output_dir / f"{pdf_path.stem}.json")
            for pdf_path in pdf_files
        ]
        
        # 使用進度條追蹤進度
        with tqdm(total=len(pdf_files), desc="處理PDF檔案", unit="檔案") as pbar:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_pdf = {
                    executor.submit(self.process_single_pdf, args): args[0]
                    for args in process_args
                }
                
                for future in as_completed(future_to_pdf):
                    pdf_name, success, message, process_time = future.result()
                    
                    # 更新統計資料
                    self.update_statistics(success, process_time)
                    
                    # 記錄處理結果
                    if success:
                        self.logger.info(f"成功: {pdf_name} - 處理時間: {process_time:.2f}秒")
                    else:
                        self.logger.error(f"失敗: {pdf_name} - {message} - 處理時間: {process_time:.2f}秒")
                    
                    pbar.update(1)
        
        # 儲存最終統計資料
        self.save_statistics()

def main():
    """主程式入口點"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text from PDF files including OCR')
    parser.add_argument('input_dir', help='Input directory containing PDF files')
    parser.add_argument('output_dir', help='Output directory for JSON files')
    parser.add_argument('--workers', type=int, help='Number of worker processes', default=None)
    
    args = parser.parse_args()
    
    try:
        extractor = PDFExtractor(args.input_dir, args.output_dir, args.workers)
        extractor.process_all_pdfs()
    except Exception as e:
        logging.critical(f"程式執行失敗: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()