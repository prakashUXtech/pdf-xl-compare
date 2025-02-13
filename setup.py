from setuptools import setup, find_packages

setup(
    name="pdf-xl-compare",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.24.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "pdf2image>=1.16.0",
        "pytesseract>=0.3.8",
        "opencv-python-headless>=4.6.0",
        "openpyxl>=3.0.0",
        "python-dotenv>=0.19.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
        "python-dateutil>=2.8.2",
    ],
    python_requires=">=3.8",
) 