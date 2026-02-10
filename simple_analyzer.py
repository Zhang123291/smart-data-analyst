# simple_analysis.py - 简化的数据分析脚本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.style.use('seaborn-v0_8-darkgrid')


class SimpleDataAnalyzer:
    """简化的数据分析器"""

    def __init__(self):
        self.df = None
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self, file_path):
        """加载数据文件"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 根据文件扩展名选择加载方法
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.csv':
                self.df = pd.read_csv(file_path)
            elif suffix in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path)
            elif suffix == '.json':
                self.df = pd.read_json(file_path)
            elif suffix == '.parquet':
                self.df = pd.read_parquet(file_path)
            else:
                # 尝试自动检测
                self.df = pd.read_csv(file_path)

            print(f"✓ Loaded: {file_path.name}")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            return True

        except Exception as e:
            print(f"✗ Failed to load file: {str(e)}")
            return False

    def basic_analysis(self):
        """基础分析"""
        print("\n" + "=" * 60)
        print("BASIC DATA ANALYSIS")
        print("=" * 60)

        # 基本信息
        print(f"\n1. Dataset Info:")
        print(f"   Rows: {len(self.df)}")
        print(f"   Columns: {len(self.df.columns)}")
        print(f"   Memory: {self.df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # 数据类型
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"\n2. Data Types:")
        print(f"   Numeric columns: {len(numeric_cols)}")
        print(f"   Categorical columns: {len(categorical_cols)}")

        # 缺失值分析
        missing_total = self.df.isnull().sum().sum()
        missing_percent = (missing_total / (self.df.shape[0] * self.df.shape[1])) * 100

        print(f"\n3. Missing Values:")
        print(f"   Total missing: {missing_total} ({missing_percent:.2f}%)")

        # 显示有缺失值的列
        missing_by_col = self.df.isnull().sum()
        missing_cols = missing_by_col[missing_by_col > 0]
        if len(missing_cols) > 0:
            print(f"   Columns with missing values:")
            for col, count in missing_cols.items():
                percent = (count / len(self.df)) * 100
                print(f"     - {col}: {count} ({percent:.2f}%)")

        # 数值列统计
        if numeric_cols:
            print(f"\n4. Numeric Columns Statistics (first 5):")
            for col in numeric_cols[:5]:
                print(f"   {col}:")
                print(f"     Mean: {self.df[col].mean():.2f}, "
                      f"Std: {self.df[col].std():.2f}, "
                      f"Min: {self.df[col].min():.2f}, "
                      f"Max: {self.df[col].max():.2f}")

    def create_visualizations(self):
        """创建可视化图表"""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = self.output_dir / f"visualizations_{timestamp}"
        viz_dir.mkdir(exist_ok=True)

        # 1. 缺失值热力图
        print("\n1. Creating missing values heatmap...")
        self._plot_missing_values(viz_dir / "missing_values.png")

        # 2. 数值特征分布
        print("\n2. Creating numeric distributions...")
        self._plot_numeric_distributions(viz_dir / "numeric_distributions.png")

        # 3. 相关性热力图
        print("\n3. Creating correlation heatmap...")
        self._plot_correlation_heatmap(viz_dir / "correlation_heatmap.png")

        # 4. 箱线图
        print("\n4. Creating box plots...")
        self._plot_boxplots(viz_dir / "boxplots.png")

        print(f"\n✓ All visualizations saved to: {viz_dir}")

    def _plot_missing_values(self, save_path):
        """绘制缺失值图"""
        if self.df.isnull().sum().sum() == 0:
            print("   No missing values to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 缺失值矩阵
        missing_data = self.df.isnull()
        axes[0].imshow(missing_data, aspect='auto', cmap='binary', interpolation='nearest')
        axes[0].set_xlabel('Column Index')
        axes[0].set_ylabel('Row Index')
        axes[0].set_title('Missing Values Matrix')

        # 每列缺失值数量
        missing_by_col = self.df.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0]

        if len(missing_by_col) > 0:
            axes[1].barh(missing_by_col.index, missing_by_col.values, color='salmon')
            axes[1].set_xlabel('Missing Count')
            axes[1].set_title('Missing Values per Column')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

    def _plot_numeric_distributions(self, save_path):
        """绘制数值特征分布"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            print("   No numeric columns to plot")
            return

        # 限制显示数量
        plot_cols = numeric_cols[:9]  # 最多显示9个

        n_cols = 3
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, col in enumerate(plot_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols

            ax = axes[row_idx, col_idx]
            ax.hist(self.df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(self.df[col].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
            ax.axvline(self.df[col].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{col} Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(len(plot_cols), n_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            axes[row_idx, col_idx].axis('off')

        plt.suptitle('Numeric Features Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

    def _plot_correlation_heatmap(self, save_path):
        """绘制相关性热力图"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            print("   Not enough numeric columns for correlation")
            return

        corr_matrix = self.df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))

        # 创建上三角掩码
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=1,
                    cbar_kws={"shrink": 0.8})

        plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        # 打印高相关性特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))

        if high_corr_pairs:
            print("   High correlation pairs (>0.8):")
            for feat1, feat2, corr in high_corr_pairs[:5]:  # 只显示前5个
                print(f"     {feat1} - {feat2}: {corr:.3f}")

    def _plot_boxplots(self, save_path):
        """绘制箱线图"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            print("   No numeric columns for box plots")
            return

        # 限制显示数量
        plot_cols = numeric_cols[:12]

        # 准备数据
        plot_data = pd.melt(self.df[plot_cols])

        plt.figure(figsize=(14, 6))
        sns.boxplot(x='variable', y='value', data=plot_data, palette='Set2')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.title('Box Plots of Numeric Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

    def generate_report(self):
        """生成分析报告"""
        print("\n" + "=" * 60)
        print("GENERATING ANALYSIS REPORT")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"analysis_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")

            # 基本信息
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"File analyzed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Rows: {len(self.df)}\n")
            f.write(f"Columns: {len(self.df.columns)}\n")
            f.write(f"Shape: {self.df.shape}\n\n")

            # 缺失值
            missing_total = self.df.isnull().sum().sum()
            missing_percent = (missing_total / (self.df.shape[0] * self.df.shape[1])) * 100
            f.write("2. MISSING VALUES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total missing values: {missing_total}\n")
            f.write(f"Missing percentage: {missing_percent:.2f}%\n\n")

            # 数据类型
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

            f.write("3. DATA TYPES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Numeric columns: {len(numeric_cols)}\n")
            f.write(f"Categorical columns: {len(categorical_cols)}\n\n")

            # 数值列统计
            if len(numeric_cols) > 0:
                f.write("4. NUMERIC FEATURES SUMMARY\n")
                f.write("-" * 40 + "\n")
                for col in numeric_cols[:10]:  # 只显示前10个
                    f.write(f"{col}:\n")
                    f.write(f"  Min: {self.df[col].min():.2f}\n")
                    f.write(f"  Mean: {self.df[col].mean():.2f}\n")
                    f.write(f"  Median: {self.df[col].median():.2f}\n")
                    f.write(f"  Max: {self.df[col].max():.2f}\n")
                    f.write(f"  Std: {self.df[col].std():.2f}\n\n")

            # 建议
            f.write("5. RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            issues = []

            # 检查高缺失值
            for col in self.df.columns:
                missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
                if missing_pct > 50:
                    issues.append(f"Consider dropping column '{col}' ({missing_pct:.1f}% missing)")
                elif missing_pct > 0:
                    issues.append(f"Consider imputing missing values in '{col}' ({missing_pct:.1f}% missing)")

            # 检查分类列的唯一值数量
            for col in categorical_cols:
                unique_count = self.df[col].nunique()
                if unique_count > 50:
                    issues.append(f"Column '{col}' has {unique_count} unique values - might be text data")

            if issues:
                for i, issue in enumerate(issues, 1):
                    f.write(f"{i}. {issue}\n")
            else:
                f.write("Data quality looks good. No major issues detected.\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")

        print(f"✓ Report saved to: {report_path}")

    def run_full_analysis(self, file_path):
        """运行完整分析"""
        print("\n" + "=" * 60)
        print("DATA ANALYSIS AI - SIMPLE VERSION")
        print("=" * 60)

        # 1. 加载数据
        if not self.load_data(file_path):
            return

        # 2. 基础分析
        self.basic_analysis()

        # 3. 可视化
        self.create_visualizations()

        # 4. 生成报告
        self.generate_report()

        print("\n" + "=" * 60)
        print("✓ ANALYSIS COMPLETED!")
        print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Simple Data Analysis Tool')
    parser.add_argument('file', type=str, help='Path to data file')

    args = parser.parse_args()

    analyzer = SimpleDataAnalyzer()
    analyzer.run_full_analysis(args.file)


if __name__ == "__main__":
    main()