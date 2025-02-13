from setuptools import setup, find_packages

setup(
    name="docling",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.24.0',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'Pillow>=9.0.0',
        'pdf2image>=1.16.0',
        'pytesseract>=0.3.8',
        'opencv-python>=4.6.0',
        'requests>=2.28.0',
        'plotly>=5.13.0',
        'python-dotenv>=0.19.0',
        'tqdm>=4.65.0',
        'openpyxl>=3.0.0',
        'pdfplumber>=0.9.0',
        'camelot-py>=0.11.0',
        'pdfminer.six>=20221105',
        'pypdf>=3.9.0'
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.9.0',
            'pytest-cov>=4.0.0'
        ],
        'viz': [
            'streamlit>=1.24.0',
            'plotly>=5.13.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'docling-dashboard=app.tools.visualization.contribution_dashboard:main',
            'docling-analyze=app.tools.analysis.pdf_analyzer:main',
        ],
    },
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="A document analysis tool for wage and contribution comparison",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/docling",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
) 