# usage_example.py
"""
数据分析AI使用示例
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from main import DataAnalysisAI


def example_analysis():
    """使用示例"""

    # 1. 创建AI实例
    ai = DataAnalysisAI()

    # 2. 加载数据（替换为你的文件路径）
    # 例如：boston_housing.csv, iris.csv, 等等
    file_path = "boston_housing.csv"  # 修改为你的文件路径

    # 3. 运行完整分析
    ai.run_full_analysis(file_path)

    # 或者分别运行各个部分：
    # ai.load_data(file_path)
    # ai.explore_data()          # 仅探索数据
    # ai.visualize_data()        # 仅可视化
    # ai.generate_report()       # 仅生成报告


def batch_analysis():
    """批量分析多个文件"""
    ai = DataAnalysisAI()

    # 支持的文件格式
    supported_formats = ['.csv', '.xlsx', '.json']

    # 扫描data目录下的所有数据文件
    data_dir = Path("data")
    if data_dir.exists():
        for file_format in supported_formats:
            for file_path in data_dir.glob(f"*{file_format}"):
                print(f"\n{'=' * 60}")
                print(f"分析文件: {file_path.name}")
                print(f"{'=' * 60}")

                try:
                    ai.run_full_analysis(str(file_path))
                except Exception as e:
                    print(f"分析失败: {str(e)}")


if __name__ == "__main__":
    print("数据分析AI系统 - 使用示例")
    print("-" * 40)

    # 运行单个文件分析
    example_analysis()

    # 或者运行批量分析
    # batch_analysis()