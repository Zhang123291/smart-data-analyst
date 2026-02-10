# run_analysis.py - 运行分析的脚本
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入修复后的main模块
from main import main

if __name__ == "__main__":
    main()