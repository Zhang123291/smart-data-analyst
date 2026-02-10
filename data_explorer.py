# data_explorer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class DataExplorer:
    """数据探索分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.original_shape = df.shape
        self.report = {}

    def get_basic_info(self) -> Dict[str, Any]:
        """获取基本信息"""
        info = {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 ** 2,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'total_cells': self.df.size,
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_rate': self.df.duplicated().mean()
        }

        self.report['basic_info'] = info
        return info

    def analyze_missing_values(self) -> Dict[str, Any]:
        """分析缺失值"""
        total_cells = np.prod(self.df.shape)
        total_missing = self.df.isnull().sum().sum()
        missing_percentage = (total_missing / total_cells) * 100

        missing_by_column = self.df.isnull().sum()
        missing_by_column_pct = (missing_by_column / len(self.df)) * 100

        # 识别有缺失值的列
        columns_with_missing = missing_by_column[missing_by_column > 0].index.tolist()

        missing_info = {
            'total_missing': total_missing,
            'total_missing_percentage': missing_percentage,
            'missing_by_column': missing_by_column[missing_by_column > 0].to_dict(),
            'missing_by_column_percentage': missing_by_column_pct[missing_by_column > 0].to_dict(),
            'columns_with_missing': columns_with_missing,
            'complete_columns': len(self.df.columns) - len(columns_with_missing)
        }

        self.report['missing_values'] = missing_info
        return missing_info

    def analyze_data_types(self) -> Dict[str, Any]:
        """分析数据类型"""
        # 分类数据类型
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist()
        boolean_cols = self.df.select_dtypes(include=['bool']).columns.tolist()

        # 尝试自动检测其他分类列
        for col in self.df.columns:
            if col not in numeric_cols + categorical_cols + datetime_cols + boolean_cols:
                # 检查是否有有限数量的唯一值
                if self.df[col].nunique() <= 20:
                    categorical_cols.append(col)
                else:
                    # 可能是文本列
                    categorical_cols.append(col)

        type_info = {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'boolean_columns': boolean_cols,
            'numeric_count': len(numeric_cols),
            'categorical_count': len(categorical_cols),
            'datetime_count': len(datetime_cols),
            'boolean_count': len(boolean_cols)
        }

        self.report['data_types'] = type_info
        return type_info

    def analyze_numeric_features(self) -> Dict[str, Any]:
        """分析数值特征"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return {}

        stats = {}
        for col in numeric_cols:
            col_stats = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'q1': self.df[col].quantile(0.25),
                'q3': self.df[col].quantile(0.75),
                'iqr': self.df[col].quantile(0.75) - self.df[col].quantile(0.25),
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
                'zeros': (self.df[col] == 0).sum(),
                'zeros_pct': (self.df[col] == 0).mean() * 100,
                'unique_values': self.df[col].nunique()
            }
            stats[col] = col_stats

        # 整体数值特征统计
        numeric_info = {
            'column_stats': stats,
            'total_numeric_columns': len(numeric_cols),
            'numeric_columns_list': numeric_cols.tolist()
        }

        self.report['numeric_features'] = numeric_info
        return numeric_info

    def analyze_categorical_features(self) -> Dict[str, Any]:
        """分析分类特征"""
        categorical_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or self.df[col].nunique() <= 20:
                categorical_cols.append(col)

        if len(categorical_cols) == 0:
            return {}

        stats = {}
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            top_values = value_counts.head(5).to_dict()

            col_stats = {
                'unique_values': self.df[col].nunique(),
                'most_common': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None,
                'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'most_common_percentage': (value_counts.iloc[0] / len(self.df) * 100) if len(value_counts) > 0 else 0,
                'top_5_values': top_values,
                'missing_values': self.df[col].isnull().sum(),
                'missing_percentage': self.df[col].isnull().mean() * 100
            }
            stats[col] = col_stats

        categorical_info = {
            'column_stats': stats,
            'total_categorical_columns': len(categorical_cols),
            'categorical_columns_list': categorical_cols
        }

        self.report['categorical_features'] = categorical_info
        return categorical_info

    def analyze_correlations(self, method: str = 'pearson') -> Dict[str, Any]:
        """分析相关性"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {}

        corr_matrix = self.df[numeric_cols].corr(method=method)

        # 找出高相关性对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.8:  # 高相关性阈值
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })

        # 找出与每个特征相关性最高的特征
        top_correlations = {}
        for col in corr_matrix.columns:
            # 排除自身
            corr_series = corr_matrix[col].drop(col)
            if not corr_series.empty:
                max_corr_feature = corr_series.abs().idxmax()
                max_corr_value = corr_series[max_corr_feature]
                top_correlations[col] = {
                    'most_correlated_with': max_corr_feature,
                    'correlation': max_corr_value
                }

        correlation_info = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'top_correlations': top_correlations,
            'method': method
        }

        self.report['correlations'] = correlation_info
        return correlation_info

    def detect_outliers(self, method: str = 'iqr') -> Dict[str, Any]:
        """检测异常值"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return {}

        outliers_info = {}
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(self.df)) * 100

                outliers_info[col] = {
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }

        self.report['outliers'] = outliers_info
        return outliers_info

    def generate_summary(self) -> Dict[str, Any]:
        """生成数据探索总结"""
        # 执行所有分析
        self.get_basic_info()
        self.analyze_missing_values()
        self.analyze_data_types()
        self.analyze_numeric_features()
        self.analyze_categorical_features()
        self.analyze_correlations()
        self.detect_outliers()

        # 生成总结
        summary = {
            'dataset_name': '当前数据集',
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'total_columns': len(self.df.columns),
            'total_rows': len(self.df),
            'missing_values_summary': self.report['missing_values'],
            'data_types_summary': self.report['data_types'],
            'issues': self._identify_issues(),
            'recommendations': self._generate_recommendations()
        }

        self.report['summary'] = summary
        return summary

    def _identify_issues(self) -> List[str]:
        """识别数据问题"""
        issues = []

        # 检查缺失值
        missing_info = self.report.get('missing_values', {})
        if missing_info.get('total_missing_percentage', 0) > 30:
            issues.append(f"高缺失值比例: {missing_info['total_missing_percentage']:.1f}%")

        # 检查重复行
        basic_info = self.report.get('basic_info', {})
        if basic_info.get('duplicate_rate', 0) > 0.1:
            issues.append(f"高重复率: {basic_info['duplicate_rate'] * 100:.1f}%")

        # 检查异常值
        outliers_info = self.report.get('outliers', {})
        for col, info in outliers_info.items():
            if info['outlier_percentage'] > 10:
                issues.append(f"列 '{col}' 有大量异常值: {info['outlier_percentage']:.1f}%")

        return issues

    def _generate_recommendations(self) -> List[str]:
        """生成数据清洗建议"""
        recommendations = []

        missing_info = self.report.get('missing_values', {})
        missing_by_col = missing_info.get('missing_by_column_percentage', {})

        for col, pct in missing_by_col.items():
            if pct > 50:
                recommendations.append(f"考虑删除列 '{col}'（缺失值 {pct:.1f}%）")
            elif pct > 0:
                recommendations.append(f"考虑填充列 '{col}' 的缺失值（{pct:.1f}%）")

        # 检查分类列的唯一值数量
        categorical_info = self.report.get('categorical_features', {})
        col_stats = categorical_info.get('column_stats', {})

        for col, stats in col_stats.items():
            unique_values = stats.get('unique_values', 0)
            if unique_values > 50:
                recommendations.append(f"列 '{col}' 有 {unique_values} 个唯一值，可能是文本列")

        return recommendations

    def get_report(self) -> Dict[str, Any]:
        """获取完整的分析报告"""
        if not self.report:
            self.generate_summary()
        return self.report