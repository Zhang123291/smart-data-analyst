# data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import json
import pickle
import warnings

warnings.filterwarnings('ignore')

from config import Config


class DataLoader:
    """通用数据加载器"""

    def __init__(self):
        self.config = Config()

    def detect_file_type(self, file_path: Union[str, Path]) -> str:
        """检测文件类型"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix in self.config.SUPPORTED_FORMATS:
            return self.config.SUPPORTED_FORMATS[suffix]
        else:
            raise ValueError(f"不支持的文件格式: {suffix}。支持格式: {list(self.config.SUPPORTED_FORMATS.keys())}")

    def load_data(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """加载数据文件"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_type = self.detect_file_type(file_path)

        try:
            if file_type == 'csv':
                # 自动检测分隔符
                df = self._load_csv(file_path, **kwargs)
            elif file_type == 'excel':
                df = pd.read_excel(file_path, **kwargs)
            elif file_type == 'json':
                df = pd.read_json(file_path, **kwargs)
            elif file_type == 'parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_type == 'feather':
                df = pd.read_feather(file_path, **kwargs)
            elif file_type == 'hdf5':
                df = pd.read_hdf(file_path, **kwargs)
            elif file_type == 'pickle':
                df = pd.read_pickle(file_path, **kwargs)
            elif file_type == 'text':
                df = self._load_text(file_path, **kwargs)
            elif file_type == 'tsv':
                df = pd.read_csv(file_path, sep='\t', **kwargs)
            else:
                raise ValueError(f"未知的文件类型: {file_type}")

            print(f"✓ 成功加载文件: {file_path.name}")
            print(f"  形状: {df.shape}")
            print(f"  内存使用: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

            return df

        except Exception as e:
            raise Exception(f"加载文件失败: {str(e)}")

    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """智能加载CSV文件"""
        # 尝试不同的编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1']

        for encoding in encodings:
            try:
                # 读取前几行来检测分隔符
                with open(file_path, 'r', encoding=encoding) as f:
                    sample = f.read(1024)

                # 检测分隔符
                separators = [',', ';', '\t', '|']
                sep_counts = {sep: sample.count(sep) for sep in separators}
                detected_sep = max(sep_counts, key=sep_counts.get) if max(sep_counts.values()) > 0 else ','

                # 加载数据
                df = pd.read_csv(file_path, sep=detected_sep, encoding=encoding, **kwargs)
                print(f"  检测到分隔符: '{detected_sep}', 编码: {encoding}")
                return df

            except UnicodeDecodeError:
                continue
            except Exception:
                continue

        # 如果所有编码都失败，使用默认参数
        return pd.read_csv(file_path, **kwargs)

    def _load_text(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """加载文本文件"""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 尝试解析为数据
        data = []
        for line in lines:
            # 简单的解析逻辑，可以根据需要扩展
            parts = line.strip().split()
            if parts:
                data.append(parts)

        return pd.DataFrame(data)

    def load_multiple_files(self, file_pattern: str) -> Dict[str, pd.DataFrame]:
        """加载多个文件"""
        data_dir = self.config.DATA_DIR
        files = list(data_dir.glob(file_pattern))

        if not files:
            raise FileNotFoundError(f"没有找到匹配的文件: {file_pattern}")

        datasets = {}
        for file_path in files:
            try:
                df = self.load_data(file_path)
                datasets[file_path.stem] = df
            except Exception as e:
                print(f"✗ 加载文件失败 {file_path.name}: {str(e)}")

        return datasets

    def save_data(self, df: pd.DataFrame, file_path: Union[str, Path],
                  file_type: str = 'auto', **kwargs):
        """保存数据"""
        file_path = Path(file_path)

        if file_type == 'auto':
            file_type = file_path.suffix.lower().lstrip('.')

        save_dir = file_path.parent
        save_dir.mkdir(exist_ok=True)

        try:
            if file_type in ['csv', 'txt']:
                df.to_csv(file_path, index=False, **kwargs)
            elif file_type in ['xlsx', 'xls']:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_type == 'parquet':
                df.to_parquet(file_path, **kwargs)
            elif file_type == 'feather':
                df.to_feather(file_path, **kwargs)
            elif file_type == 'json':
                df.to_json(file_path, **kwargs)
            elif file_type == 'pickle':
                df.to_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")

            print(f"✓ 数据已保存到: {file_path}")

        except Exception as e:
            raise Exception(f"保存文件失败: {str(e)}")