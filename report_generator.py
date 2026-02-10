# report_generator.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from config import Config
from data_explorer import DataExplorer


class ReportGenerator:
    """报告生成器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.config = Config()
        self.explorer = DataExplorer(df)
        self.analysis_time = datetime.now()

    def _matplotlib_to_base64(self, fig) -> str:
        """将matplotlib图形转换为base64字符串"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def generate_html_report(self, output_path: str) -> str:
        """生成HTML报告"""
        print("Generating HTML report...")

        # 执行数据分析
        report_data = self.explorer.get_report()

        # 创建图表
        charts = self._create_charts()

        # 生成HTML内容
        html_content = self._create_html_content(report_data, charts)

        # 保存HTML文件
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(output_path)

    def _create_charts(self) -> Dict[str, str]:
        """创建报告中的图表"""
        charts = {}

        try:
            # 1. 缺失值图表
            fig, ax = plt.subplots(figsize=(8, 6))
            missing_by_col = self.df.isnull().sum()
            missing_by_col = missing_by_col[missing_by_col > 0]
            if len(missing_by_col) > 0:
                # 只显示前15个
                if len(missing_by_col) > 15:
                    missing_by_col = missing_by_col[:15]
                ax.barh(range(len(missing_by_col)), missing_by_col.values, color='salmon')
                ax.set_yticks(range(len(missing_by_col)))
                ax.set_yticklabels(missing_by_col.index)
                ax.set_xlabel('Missing Count')
                ax.set_title('Missing Values per Column')
                ax.grid(True, alpha=0.3)
            charts['missing_chart'] = self._matplotlib_to_base64(fig)

            # 2. 数据类型图表
            fig, ax = plt.subplots(figsize=(8, 6))
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            datetime_cols = self.df.select_dtypes(include=['datetime64', 'timedelta64']).columns

            labels = []
            sizes = []

            if len(numeric_cols) > 0:
                labels.append('Numeric')
                sizes.append(len(numeric_cols))
            if len(categorical_cols) > 0:
                labels.append('Categorical')
                sizes.append(len(categorical_cols))
            if len(datetime_cols) > 0:
                labels.append('Datetime')
                sizes.append(len(datetime_cols))

            if sizes:
                colors = ['lightblue', 'lightgreen', 'lightcoral'][:len(sizes)]
                ax.pie(sizes, labels=labels, colors=colors,
                       autopct='%1.1f%%', startangle=90)
                ax.set_title('Data Type Distribution')
            charts['dtype_chart'] = self._matplotlib_to_base64(fig)

        except Exception as e:
            print(f"Error creating charts: {str(e)}")
            charts['missing_chart'] = ''
            charts['dtype_chart'] = ''

        return charts

    def _create_html_content(self, report_data: Dict[str, Any], charts: Dict[str, str]) -> str:
        """创建HTML内容"""
        basic_info = report_data.get('basic_info', {})
        missing_info = report_data.get('missing_values', {})
        type_info = report_data.get('data_types', {})
        summary = report_data.get('summary', {})

        # 获取建议和问题
        issues = summary.get('issues', [])
        recommendations = summary.get('recommendations', [])

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Analysis Report - {self.analysis_time.strftime('%Y-%m-%d %H:%M')}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #4CAF50;
                }}
                .header h1 {{
                    color: #2c3e50;
                    margin-bottom: 10px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .section h2 {{
                    color: #2c3e50;
                    margin-top: 0;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                .stat-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .stat-card {{
                    background-color: white;
                    padding: 15px;
                    border-radius: 6px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .stat-card h3 {{
                    color: #3498db;
                    margin-top: 0;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 10px;
                    background-color: white;
                }}
                .issues {{
                    background-color: #fff3cd;
                    border-color: #ffeaa7;
                }}
                .recommendations {{
                    background-color: #d4edda;
                    border-color: #c3e6cb;
                }}
                .summary {{
                    background-color: #d1ecf1;
                    border-color: #bee5eb;
                }}
                .timestamp {{
                    text-align: center;
                    color: #7f8c8d;
                    font-style: italic;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                }}
                ul, ol {{
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Data Analysis Report</h1>
                    <p>Generated: {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="section summary">
                    <h2>📋 Dataset Overview</h2>
                    <div class="stat-grid">
                        <div class="stat-card">
                            <h3>Dataset Shape</h3>
                            <div class="stat-value">{basic_info.get('shape', (0, 0))[0]} rows × {basic_info.get('shape', (0, 0))[1]} columns</div>
                        </div>
                        <div class="stat-card">
                            <h3>Memory Usage</h3>
                            <div class="stat-value">{basic_info.get('memory_usage_mb', 0):.2f} MB</div>
                        </div>
                        <div class="stat-card">
                            <h3>Duplicate Rows</h3>
                            <div class="stat-value">{basic_info.get('duplicate_rows', 0)} ({basic_info.get('duplicate_rate', 0) * 100:.1f}%)</div>
                        </div>
                        <div class="stat-card">
                            <h3>Total Cells</h3>
                            <div class="stat-value">{basic_info.get('total_cells', 0):,}</div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>🔍 Data Quality Analysis</h2>
                    <div class="stat-grid">
                        <div class="stat-card">
                            <h3>Total Missing Values</h3>
                            <div class="stat-value">{missing_info.get('total_missing', 0)}</div>
                            <p>{missing_info.get('total_missing_percentage', 0):.1f}%</p>
                        </div>
                        <div class="stat-card">
                            <h3>Columns with Missing Values</h3>
                            <div class="stat-value">{len(missing_info.get('columns_with_missing', []))}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Complete Columns</h3>
                            <div class="stat-value">{missing_info.get('complete_columns', 0)}</div>
                        </div>
                    </div>

                    {f'<div class="chart-container"><h3>Missing Values Distribution</h3><img src="data:image/png;base64,{charts.get("missing_chart", "")}" alt="Missing Values Chart"></div>' if charts.get('missing_chart') else ''}
                </div>

                <div class="section">
                    <h2>📊 Data Type Analysis</h2>
                    <div class="stat-grid">
                        <div class="stat-card">
                            <h3>Numeric Columns</h3>
                            <div class="stat-value">{type_info.get('numeric_count', 0)}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Categorical Columns</h3>
                            <div class="stat-value">{type_info.get('categorical_count', 0)}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Datetime Columns</h3>
                            <div class="stat-value">{type_info.get('datetime_count', 0)}</div>
                        </div>
                    </div>

                    {f'<div class="chart-container"><h3>Data Type Distribution</h3><img src="data:image/png;base64,{charts.get("dtype_chart", "")}" alt="Data Type Chart"></div>' if charts.get('dtype_chart') else ''}
                </div>

                <div class="section issues">
                    <h2>⚠️ Issues Found</h2>
                    <ul>
        """

        # 添加问题
        if issues:
            for issue in issues[:15]:  # 只显示前15个
                html += f'<li>{issue}</li>\n'
            if len(issues) > 15:
                html += f'<li>... and {len(issues) - 15} more issues</li>\n'
        else:
            html += '<li>✓ No significant issues found</li>\n'

        html += """
                    </ul>
                </div>

                <div class="section recommendations">
                    <h2>💡 Recommendations</h2>
                    <ol>
        """

        # 添加建议
        if recommendations:
            for rec in recommendations[:15]:  # 只显示前15个
                html += f'<li>{rec}</li>\n'
            if len(recommendations) > 15:
                html += f'<li>... and {len(recommendations) - 15} more recommendations</li>\n'
        else:
            html += '<li>Data quality looks good. No special treatment needed.</li>\n'

        html += f"""
                    </ol>
                </div>

                <div class="timestamp">
                    Report generated: {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}<br>
                    Rows: {len(self.df)} | Columns: {len(self.df.columns)}
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def generate_text_summary(self, output_path: str):
        """生成文本总结"""
        report_data = self.explorer.get_report()
        summary = report_data.get('summary', {})

        text = f"""
Data Analysis Report
{'=' * 50}
Generated: {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {summary.get('dataset_name', 'Current Dataset')}

1. DATASET OVERVIEW
{'-' * 30}
Original Shape: {summary.get('original_shape', 'N/A')}
Current Shape: {summary.get('current_shape', 'N/A')}
Total Columns: {summary.get('total_columns', 'N/A')}
Total Rows: {summary.get('total_rows', 'N/A')}

2. DATA QUALITY
{'-' * 30}
Total Missing Percentage: {summary.get('missing_values_summary', {}).get('total_missing_percentage', 0):.1f}%
Columns with Missing Values: {len(summary.get('missing_values_summary', {}).get('columns_with_missing', []))}

3. DATA TYPE DISTRIBUTION
{'-' * 30}
Numeric Columns: {summary.get('data_types_summary', {}).get('numeric_count', 0)}
Categorical Columns: {summary.get('data_types_summary', {}).get('categorical_count', 0)}
Datetime Columns: {summary.get('data_types_summary', {}).get('datetime_count', 0)}
Boolean Columns: {summary.get('data_types_summary', {}).get('boolean_count', 0)}

4. ISSUES FOUND
{'-' * 30}
"""

        issues = summary.get('issues', [])
        if issues:
            for i, issue in enumerate(issues[:20], 1):  # 只显示前20个
                text += f"{i}. {issue}\n"
            if len(issues) > 20:
                text += f"... and {len(issues) - 20} more issues\n"
        else:
            text += "No significant issues found\n"

        text += """
5. RECOMMENDATIONS
{'-'*30}
"""

        recommendations = summary.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations[:20], 1):  # 只显示前20个
                text += f"{i}. {rec}\n"
            if len(recommendations) > 20:
                text += f"... and {len(recommendations) - 20} more recommendations\n"
        else:
            text += "Data quality looks good. No special treatment needed.\n"

        text += f"""
{'=' * 50}
Analysis completed.
"""

        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return str(output_path)