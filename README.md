# Document Analysis and Wage Comparison Tool

A sophisticated document analysis tool that extracts and compares wage and contribution information from PDF documents and Excel files.

## Features

- Multiple OCR implementations (Google Vision, Nanonets)
- PDF document analysis
- Excel data processing
- Wage comparison and analysis
- Interactive visualization
- Data validation and cleaning

## Project Structure

```
docling/
├── app/                    # Main application code
│   ├── components/         # UI components
│   ├── processors/         # Document processors
│   ├── services/          # Service implementations
│   │   └── ocr/           # OCR services
│   └── utils/             # Utility functions
├── data/                  # Data files
│   ├── raw/               # Raw input files
│   └── processed/         # Processed data files
├── docs/                  # Documentation
├── outputs/               # Generated outputs
│   ├── reports/           # Generated reports
│   └── visualizations/    # Generated visualizations
└── tests/                 # Test files
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Configure API keys:
   - Set up Google Vision API key
   - Set up Nanonets API key and model ID

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app/main.py
   ```

2. Upload documents:
   - Support for PDF files
   - Support for Excel files

3. View analysis:
   - Wage comparisons
   - Contribution analysis
   - Data visualization

## Development

- Use Python 3.8+
- Follow PEP 8 style guide
- Write tests for new features
- Document code changes

## License

[Your License Here] 