# main.py
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import argparse
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入自定义模块
from config import Config
from data_loader import DataLoader
from data_explorer import DataExplorer
from data_visualizer import DataVisualizer
from report_generator import ReportGenerator

# 设置matplotlib，移除中文字体配置
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib参数（不指定中文字体）
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-darkgrid')


class DataAnalysisAI:
    """数据分析AI主类"""

    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.df = None
        self.explorer = None
        self.visualizer = None
        self.report_generator = None

    def load_data(self, file_path: str, **kwargs):
        """加载数据文件"""
        print(f"\n{'=' * 50}")
        print(f"Loading Data: {file_path}")
        print(f"{'=' * 50}")

        try:
            self.df = self.data_loader.load_data(file_path, **kwargs)
            self.explorer = DataExplorer(self.df)
            self.visualizer = DataVisualizer(self.df)
            self.report_generator = ReportGenerator(self.df)
            return True
        except Exception as e:
            print(f"✗ Failed to load data: {str(e)}")
            return False

    def explore_data(self):
        """探索数据"""
        if self.df is None:
            print("Please load data first")
            return

        print(f"\n{'=' * 50}")
        print("Data Exploration Analysis")
        print(f"{'=' * 50}")

        # 基本信息
        basic_info = self.explorer.get_basic_info()
        print(f"Dataset Shape: {basic_info['shape']}")
        print(f"Number of Columns: {len(basic_info['columns'])}")
        print(f"Number of Rows: {basic_info['shape'][0]}")
        print(f"Memory Usage: {basic_info['memory_usage_mb']:.2f} MB")
        print(f"Duplicate Rows: {basic_info['duplicate_rows']} ({basic_info['duplicate_rate'] * 100:.1f}%)")

        # 缺失值分析
        missing_info = self.explorer.analyze_missing_values()
        print(f"\nMissing Values Analysis:")
        print(
            f"Total Missing Values: {missing_info['total_missing']} ({missing_info['total_missing_percentage']:.1f}%)")
        if missing_info['columns_with_missing']:
            print(f"Columns with Missing Values:")
            for col in missing_info['columns_with_missing'][:10]:  # 只显示前10个
                missing_pct = missing_info['missing_by_column_percentage'][col]
                print(f"  - {col}: {missing_info['missing_by_column'][col]} ({missing_pct:.1f}%)")

        # 数据类型分析
        type_info = self.explorer.analyze_data_types()
        print(f"\nData Type Distribution:")
        print(f"Numeric Columns: {type_info['numeric_count']}")
        print(f"Categorical Columns: {type_info['categorical_count']}")
        print(f"Datetime Columns: {type_info['datetime_count']}")
        print(f"Boolean Columns: {type_info['boolean_count']}")

        # 数值特征分析
        if type_info['numeric_count'] > 0:
            numeric_info = self.explorer.analyze_numeric_features()
            print(f"\nNumeric Features Statistics (first 3):")
            for col in numeric_info['numeric_columns_list'][:3]:
                stats = numeric_info['column_stats'][col]
                print(f"  {col}:")
                print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                print(f"    Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

        # 分类特征分析
        if type_info['categorical_count'] > 0:
            categorical_info = self.explorer.analyze_categorical_features()
            print(f"\nCategorical Features Statistics (first 3):")
            for col in categorical_info['categorical_columns_list'][:3]:
                stats = categorical_info['column_stats'][col]
                print(f"  {col}: {stats['unique_values']} unique values")
                if stats['most_common']:
                    print(f"    Most common: {stats['most_common']} ({stats['most_common_percentage']:.1f}%)")

        # 生成总结报告
        summary = self.explorer.generate_summary()
        print(f"\n{'=' * 50}")
        print("Data Issues Summary:")
        print(f"{'=' * 50}")
        if summary['issues']:
            for i, issue in enumerate(summary['issues'][:10], 1):  # 只显示前10个
                print(f"{i}. {issue}")
            if len(summary['issues']) > 10:
                print(f"... and {len(summary['issues']) - 10} more issues")
        else:
            print("✓ No significant data issues found")

        print(f"\nRecommendations:")
        for i, rec in enumerate(summary['recommendations'][:10], 1):  # 只显示前10个
            print(f"{i}. {rec}")
        if len(summary['recommendations']) > 10:
            print(f"... and {len(summary['recommendations']) - 10} more recommendations")

    def visualize_data(self, save_dir: Optional[str] = None):
        """可视化数据"""
        if self.df is None:
            print("Please load data first")
            return

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
        else:
            save_path = self.config.OUTPUT_DIR / "visualizations"
            save_path.mkdir(exist_ok=True)

        print(f"\n{'=' * 50}")
        print("Data Visualization")
        print(f"{'=' * 50}")
        print(f"Charts will be saved to: {save_path}")

        try:
            # 1. 缺失值可视化
            print("\n1. Creating missing values visualization...")
            missing_path = save_path / "missing_values.png"
            self.visualizer.plot_missing_values(str(missing_path))

            # 2. 数据类型可视化
            print("\n2. Creating data types visualization...")
            dtype_path = save_path / "data_types.png"
            self.visualizer.plot_data_types(str(dtype_path))

            # 3. 数值特征分布
            print("\n3. Creating numeric distributions...")
            numeric_path = save_path / "numeric_distributions.png"
            self.visualizer.plot_numeric_distributions(save_path=str(numeric_path))

            # 4. 分类特征分布
            print("\n4. Creating categorical distributions...")
            categorical_path = save_path / "categorical_distributions.png"
            self.visualizer.plot_categorical_distributions(save_path=str(categorical_path))

            # 5. 相关性热力图
            print("\n5. Creating correlation heatmap...")
            corr_path = save_path / "correlation_heatmap.png"
            self.visualizer.plot_correlation_heatmap(save_path=str(corr_path))

            # 6. 箱线图
            print("\n6. Creating box plots...")
            box_path = save_path / "boxplots.png"
            self.visualizer.plot_boxplots(save_path=str(box_path))

            print(f"\n✓ All charts saved to: {save_path}")

        except Exception as e:
            print(f"✗ Error during visualization: {str(e)}")

    def generate_report(self, output_file: Optional[str] = None):
        """生成分析报告"""
        if self.df is None:
            print("Please load data first")
            return

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.config.OUTPUT_DIR / f"analysis_report_{timestamp}.html"

        print(f"\n{'=' * 50}")
        print("Generating Analysis Report")
        print(f"{'=' * 50}")

        try:
            report_path = self.report_generator.generate_html_report(str(output_file))
            print(f"✓ Analysis report generated: {report_path}")

            # 同时生成文本总结
            text_report = self.config.OUTPUT_DIR / f"summary_{timestamp}.txt"
            self.report_generator.generate_text_summary(str(text_report))
            print(f"✓ Text summary generated: {text_report}")

        except Exception as e:
            print(f"✗ Failed to generate report: {str(e)}")

    def run_full_analysis(self, file_path: str):
        """运行完整的数据分析流程"""
        print(f"\n{'=' * 60}")
        print("Data Analysis AI - Complete Analysis Pipeline")
        print(f"{'=' * 60}")

        # 1. 加载数据
        if not self.load_data(file_path):
            return

        # 2. 数据探索
        self.explore_data()

        # 3. 数据可视化
        self.visualize_data()

        # 4. 生成报告
        self.generate_report()

        print(f"\n{'=' * 60}")
        print("✓ Analysis Completed!")
        print(f"{'=' * 60}")
        print(f"Results saved to: {self.config.OUTPUT_DIR}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Data Analysis AI System')
    parser.add_argument('file', type=str, nargs='?', help='Path to data file')
    parser.add_argument('--explore', action='store_true', help='Only perform data exploration')
    parser.add_argument('--visualize', action='store_true', help='Only perform data visualization')
    parser.add_argument('--report', action='store_true', help='Only generate report')
    parser.add_argument('--output', type=str, help='Output directory')

    args = parser.parse_args()

    # 初始化系统
    Config.setup()
    ai = DataAnalysisAI()

    # 设置输出目录
    if args.output:
        Config.OUTPUT_DIR = Path(args.output)
        Config.OUTPUT_DIR.mkdir(exist_ok=True)

    # 如果没有指定文件，交互式输入
    if not args.file:
        print("\nData Analysis AI System")
        print("-" * 30)
        print("Supported file formats:")
        print(", ".join(Config.get_supported_formats()))

        file_path = input("\nPlease enter data file path: ").strip()

        if not file_path:
            print("No file path provided")
            return
    else:
        file_path = args.file

    # 检查文件是否存在
    if not Path(file_path).exists():
        print(f"File does not exist: {file_path}")
        return

    # 根据参数运行相应的分析
    if args.explore:
        if ai.load_data(file_path):
            ai.explore_data()
    elif args.visualize:
        if ai.load_data(file_path):
            ai.visualize_data()
    elif args.report:
        if ai.load_data(file_path):
            ai.generate_report()
    else:
        # 运行完整分析
        ai.run_full_analysis(file_path)


if __name__ == "__main__":
    main()