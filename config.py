# config.py
import os
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """配置类"""

    # 项目路径
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    TEMPLATES_DIR = PROJECT_ROOT / "templates"

    # 创建必要的目录
    for directory in [DATA_DIR, OUTPUT_DIR, TEMPLATES_DIR]:
        directory.mkdir(exist_ok=True)

    # 支持的输入文件格式
    SUPPORTED_FORMATS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.json': 'json',
        '.parquet': 'parquet',
        '.feather': 'feather',
        '.h5': 'hdf5',
        '.pkl': 'pickle',
        '.txt': 'text',
        '.tsv': 'tsv'
    }

    # 可视化配置（无字体配置）
    VISUALIZATION_CONFIG = {
        'style': 'seaborn-v0_8-darkgrid',
        'color_palette': 'husl',
        'figure_size': (12, 8),
        'dpi': 100,
        'save_format': 'png'
    }

    # 分析配置
    ANALYSIS_CONFIG = {
        'missing_threshold': 0.3,
        'correlation_threshold': 0.7,
        'outlier_method': 'iqr',
        'max_categories': 20,
        'sample_size': 10000
    }

    @classmethod
    def setup(cls):
        """初始化配置"""
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")

    @classmethod
    def get_supported_formats(cls):
        """获取支持的文件格式"""
        return list(cls.SUPPORTED_FORMATS.keys())