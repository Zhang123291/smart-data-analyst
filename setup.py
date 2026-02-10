# setup.py
from setuptools import setup, find_packages

setup(
    name="data-analysis-ai",
    version="1.0.0",
    description="通用数据分析AI系统",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.2.0',
        'openpyxl>=3.0.0',
        'plotly>=5.13.0',
        'missingno>=0.5.2',
        'statsmodels>=0.14.0',
    ],
    entry_points={
        'console_scripts': [
            'data-ai=main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)