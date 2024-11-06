import os
from dotenv import load_dotenv
from pathlib import Path

def load_env(start_path=None):
    """從當前目錄往上層查找並加載 .env 文件"""
    # 如果沒有提供起始目錄，則使用當前工作目錄
    start_path = start_path or Path(os.getcwd())

    # 查找 .env 文件直到達到根目錄
    current_path = start_path
    while current_path != current_path.root:
        env_file = current_path / '.env'
        if env_file.exists():
            print(f'Found .env at: {env_file}')
            load_dotenv(env_file)
            return True
        current_path = current_path.parent

    # 如果找不到 .env 文件，返回 False
    print('.env file not found.')
    return False
