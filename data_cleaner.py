# data_cleaner.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


class DataCleaner:
    """数据清洗器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cleaning_log = []

    def handle_missing_values(self, strategy: str = 'auto',
                              threshold: float = 0.5) -> pd.DataFrame:
        """
        处理缺失值

        Args:
            strategy: 'auto', 'delete', 'fill', 'impute'
            threshold: 缺失值超过此比例时删除列
        """
        df_clean = self.df.copy()

        for col in df_clean.columns:
            missing_pct = df_clean[col].isnull().mean()

            # 1. 高缺失值列：删除
            if missing_pct > threshold:
                df_clean = df_clean.drop(columns=[col])
                self.cleaning_log.append(f"Dropped column '{col}' ({missing_pct:.1%} missing)")

            # 2. 低缺失值列：填充
            elif missing_pct > 0:
                if df_clean[col].dtype in ['float64', 'int64']:
                    # 数值列：用中位数填充
                    fill_value = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    self.cleaning_log.append(f"Filled numeric '{col}' with median: {fill_value}")
                else:
                    # 分类列：用众数填充
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    self.cleaning_log.append(f"Filled categorical '{col}' with mode: {fill_value}")

        return df_clean

    def handle_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """处理异常值"""
        df_clean = self.df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # 标识异常值
                outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outliers_count = outliers_mask.sum()

                if outliers_count > 0:
                    # 方法1：用边界值替换
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound

                    self.cleaning_log.append(
                        f"Capped outliers in '{col}': {outliers_count} values "
                        f"({outliers_count / len(df_clean):.1%})"
                    )

        return df_clean

    def encode_categorical(self, method: str = 'label') -> pd.DataFrame:
        """编码分类变量"""
        df_encoded = self.df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

        if method == 'label':
            from sklearn.preprocessing import LabelEncoder
            label_encoders = {}

            for col in categorical_cols:
                if df_encoded[col].nunique() < 20:  # 只编码少量类别的列
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    label_encoders[col] = le
                    self.cleaning_log.append(f"Label encoded '{col}'")

        elif method == 'onehot':
            # 对少量类别的列进行独热编码
            for col in categorical_cols:
                if df_encoded[col].nunique() <= 10:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                    self.cleaning_log.append(f"One-hot encoded '{col}'")

        return df_encoded

    def get_cleaning_summary(self) -> Dict[str, Any]:
        """获取清洗总结"""
        return {
            'original_shape': self.df.shape,
            'cleaning_log': self.cleaning_log,
            'total_operations': len(self.cleaning_log)
        }