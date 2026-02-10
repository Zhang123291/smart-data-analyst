# ml_analyzer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class MLAnalyzer:
    """机器学习分析器"""

    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.models = {}
        self.results = {}

    def prepare_data(self, test_size: float = 0.2):
        """准备数据"""
        # 分离特征和目标
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # 确保所有特征都是数值型
        X = self._ensure_numeric(X)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }

    def _ensure_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """确保所有特征都是数值型"""
        X_clean = X.copy()

        # 转换分类变量
        for col in X_clean.select_dtypes(include=['object', 'category']).columns:
            # 如果是数值型的字符串，转换为数值
            try:
                X_clean[col] = pd.to_numeric(X_clean[col])
            except:
                # 如果是分类变量，使用标签编码
                X_clean[col] = X_clean[col].astype('category').cat.codes

        return X_clean

    def train_models(self, data: Dict[str, Any]):
        """训练多个模型"""
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }

        for name, model in models_to_train.items():
            print(f"Training {name}...")

            try:
                # 训练模型
                model.fit(data['X_train'], data['y_train'])

                # 预测
                y_pred = model.predict(data['X_test'])

                # 评估
                mse = mean_squared_error(data['y_test'], y_pred)
                mae = mean_absolute_error(data['y_test'], y_pred)
                r2 = r2_score(data['y_test'], y_pred)

                # 交叉验证
                cv_scores = cross_val_score(
                    model, data['X_train'], data['y_train'],
                    cv=5, scoring='r2'
                )

                self.models[name] = model
                self.results[name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }

                print(f"  ✓ R² Score: {r2:.3f}, CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

            except Exception as e:
                print(f"  ✗ Failed to train {name}: {str(e)}")

    def get_best_model(self) -> tuple:
        """获取最佳模型"""
        if not self.results:
            return None, None

        best_model_name = max(self.results, key=lambda x: self.results[x]['r2'])
        return best_model_name, self.results[best_model_name]

    def generate_ml_report(self) -> str:
        """生成机器学习报告"""
        if not self.results:
            return "No models trained."

        report = "=" * 60 + "\n"
        report += "MACHINE LEARNING ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"

        # 模型比较
        report += "MODEL PERFORMANCE COMPARISON:\n"
        report += "-" * 60 + "\n"

        for name, metrics in self.results.items():
            report += f"{name}:\n"
            report += f"  R² Score: {metrics['r2']:.4f}\n"
            report += f"  MSE: {metrics['mse']:.2f}\n"
            report += f"  MAE: {metrics['mae']:.2f}\n"
            report += f"  CV R²: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})\n"
            report += "\n"

        # 最佳模型
        best_name, best_metrics = self.get_best_model()
        report += "BEST MODEL:\n"
        report += "-" * 60 + "\n"
        report += f"Model: {best_name}\n"
        report += f"R² Score: {best_metrics['r2']:.4f}\n"
        report += f"Cross-Validation R²: {best_metrics['cv_mean']:.4f}\n\n"

        # 建议
        report += "RECOMMENDATIONS:\n"
        report += "-" * 60 + "\n"

        if best_metrics['r2'] > 0.8:
            report += "✓ Model performance is excellent (>0.8 R²)\n"
        elif best_metrics['r2'] > 0.6:
            report += "✓ Model performance is good (>0.6 R²)\n"
            report += "  Consider feature engineering to improve further\n"
        else:
            report += "⚠ Model performance needs improvement\n"
            report += "  Suggestions:\n"
            report += "  1. Add more features or feature engineering\n"
            report += "  2. Try more complex models\n"
            report += "  3. Check for data leakage or target leakage\n"

        return report