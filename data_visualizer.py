# data_visualizer.py
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

from config import Config


class DataVisualizer:
    """数据可视化器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.config = Config()
        self.setup_style()

    def setup_style(self):
        """设置绘图样式"""
        plt.style.use(self.config.VISUALIZATION_CONFIG['style'])
        plt.rcParams['figure.figsize'] = self.config.VISUALIZATION_CONFIG['figure_size']
        plt.rcParams['figure.dpi'] = self.config.VISUALIZATION_CONFIG['dpi']

    def plot_missing_values(self, save_path: Optional[str] = None):
        """可视化缺失值"""
        if self.df.isnull().sum().sum() == 0:
            print("No missing values in data")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 缺失值热力图
        missing_data = self.df.isnull()
        axes[0].imshow(missing_data, aspect='auto', cmap='binary', interpolation='nearest')
        axes[0].set_xlabel('Column Index')
        axes[0].set_ylabel('Row Index')
        axes[0].set_title('Missing Values Distribution Heatmap')

        # 每列缺失值数量
        missing_by_col = self.df.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0]

        if len(missing_by_col) > 0:
            # 只显示前20列，避免图表过于拥挤
            if len(missing_by_col) > 20:
                missing_by_col = missing_by_col[:20]
                axes[1].set_title('Missing Values per Column (Top 20)')
            else:
                axes[1].set_title('Missing Values per Column')

            axes[1].barh(range(len(missing_by_col)), missing_by_col.values, color='salmon')
            axes[1].set_yticks(range(len(missing_by_col)))
            axes[1].set_yticklabels(missing_by_col.index)
            axes[1].set_xlabel('Missing Count')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No Missing Values',
                         ha='center', va='center', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Chart saved: {save_path}")

        plt.close()

    def plot_data_types(self, save_path: Optional[str] = None):
        """可视化数据类型分布"""
        # 分类数据类型
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = self.df.select_dtypes(include=['datetime64', 'timedelta64']).columns
        boolean_cols = self.df.select_dtypes(include=['bool']).columns

        # 计数
        type_counts = {
            'Numeric': len(numeric_cols),
            'Categorical': len(categorical_cols),
            'Datetime': len(datetime_cols),
            'Boolean': len(boolean_cols)
        }

        # 过滤掉值为0的类型
        type_counts = {k: v for k, v in type_counts.items() if v > 0}

        if not type_counts:
            print("No data type information available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 饼图
        axes[0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
                    startangle=90, colors=sns.color_palette('Set2'))
        axes[0].set_title('Data Type Distribution')

        # 柱状图
        axes[1].bar(type_counts.keys(), type_counts.values(), color='lightblue', edgecolor='black')
        axes[1].set_xlabel('Data Type')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Number of Columns by Data Type')
        axes[1].grid(True, alpha=0.3)

        # 在柱子上添加数值
        for i, v in enumerate(type_counts.values()):
            axes[1].text(i, v + 0.1, str(v), ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Chart saved: {save_path}")

        plt.close()

    def plot_numeric_distributions(self, columns: Optional[List[str]] = None,
                                   save_path: Optional[str] = None):
        """可视化数值特征的分布"""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = [col for col in columns if col in self.df.columns and
                            np.issubdtype(self.df[col].dtype, np.number)]

        if len(numeric_cols) == 0:
            print("No numeric features to visualize")
            return

        # 限制显示的特征数量
        max_plots = min(len(numeric_cols), 9)
        plot_cols = numeric_cols[:max_plots]

        # 计算子图布局
        n_cols = 3
        n_rows = (max_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        # 如果只有一个子图，确保axes是数组
        if n_rows * n_cols == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i, col in enumerate(plot_cols):
            if i < len(axes):
                # 跳过有太多缺失值的列
                if self.df[col].isnull().sum() / len(self.df) > 0.5:
                    axes[i].text(0.5, 0.5, f"Too many\nmissing values",
                                 ha='center', va='center', fontsize=10)
                    axes[i].set_title(f'{col}')
                    axes[i].axis('off')
                    continue

                # 直方图
                data = self.df[col].dropna()
                if len(data) > 0:
                    axes[i].hist(data, bins=30, alpha=0.7,
                                 color='skyblue', edgecolor='black')
                    axes[i].axvline(data.mean(), color='red', linestyle='dashed',
                                    linewidth=2, label='Mean')
                    axes[i].axvline(data.median(), color='green', linestyle='dashed',
                                    linewidth=2, label='Median')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].legend(fontsize=8)
                    axes[i].grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(len(plot_cols), len(axes)):
            axes[i].axis('off')

        plt.suptitle('Numeric Features Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Chart saved: {save_path}")

        plt.close()

    def plot_categorical_distributions(self, columns: Optional[List[str]] = None,
                                       max_categories: int = 10,
                                       save_path: Optional[str] = None):
        """可视化分类特征的分布"""
        if columns is None:
            categorical_cols = []
            for col in self.df.columns:
                if self.df[col].dtype == 'object' or self.df[col].nunique() <= 20:
                    categorical_cols.append(col)
        else:
            categorical_cols = [col for col in columns if col in self.df.columns]

        if len(categorical_cols) == 0:
            print("No categorical features to visualize")
            return

        # 限制显示的特征数量
        max_plots = min(len(categorical_cols), 9)
        plot_cols = categorical_cols[:max_plots]

        # 计算子图布局
        n_cols = 3
        n_rows = (max_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        # 如果只有一个子图，确保axes是数组
        if n_rows * n_cols == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i, col in enumerate(plot_cols):
            if i < len(axes):
                # 跳过有太多缺失值的列
                if self.df[col].isnull().sum() / len(self.df) > 0.5:
                    axes[i].text(0.5, 0.5, f"Too many\nmissing values",
                                 ha='center', va='center', fontsize=10)
                    axes[i].set_title(f'{col}')
                    axes[i].axis('off')
                    continue

                # 获取前N个最常见的类别
                value_counts = self.df[col].value_counts().head(max_categories)

                if len(value_counts) > 0:
                    bars = axes[i].bar(range(len(value_counts)), value_counts.values,
                                       color=sns.color_palette('husl', len(value_counts)))
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'{col} Distribution (Top {len(value_counts)})')
                    axes[i].set_xticks(range(len(value_counts)))

                    # 处理长的类别标签
                    labels = [str(label) for label in value_counts.index]
                    if max(len(label) for label in labels) > 10:
                        axes[i].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                    else:
                        axes[i].set_xticklabels(labels, rotation=0, ha='center', fontsize=8)

                    axes[i].grid(True, alpha=0.3)

                    # 在柱子上添加数值
                    for bar in bars:
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                                     f'{int(height)}', ha='center', va='bottom', fontsize=8)

        # 隐藏多余的子图
        for i in range(len(plot_cols), len(axes)):
            axes[i].axis('off')

        plt.suptitle('Categorical Features Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Chart saved: {save_path}")

        plt.close()

    def plot_correlation_heatmap(self, method: str = 'pearson',
                                 save_path: Optional[str] = None):
        """绘制相关性热力图"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            print("Not enough numeric columns for correlation analysis")
            return

        # 限制特征数量，避免图表过于复杂
        if len(numeric_cols) > 15:
            numeric_cols = numeric_cols[:15]
            print(f"Using top 15 numeric columns for correlation heatmap")

        # 计算相关系数矩阵
        corr_matrix = self.df[numeric_cols].corr(method=method)

        # 创建热力图
        plt.figure(figsize=(12, 10))

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

        plt.title(f'Feature Correlation Heatmap ({method} method)', fontsize=16, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Chart saved: {save_path}")

        plt.close()

    def plot_boxplots(self, columns: Optional[List[str]] = None,
                      save_path: Optional[str] = None):
        """绘制箱线图"""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = [col for col in columns if col in self.df.columns and
                            np.issubdtype(self.df[col].dtype, np.number)]

        if len(numeric_cols) == 0:
            print("No numeric columns for box plots")
            return

        # 限制显示的特征数量
        max_cols = min(len(numeric_cols), 12)
        plot_cols = numeric_cols[:max_cols]

        # 准备数据
        plot_data = pd.melt(self.df[plot_cols])

        # 创建箱线图
        plt.figure(figsize=(14, 6))
        sns.boxplot(x='variable', y='value', data=plot_data, palette='Set2')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.title('Box Plots of Numeric Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Chart saved: {save_path}")

        plt.close()