import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_RESPONSES_DIR = DATA_DIR / "raw_responses"
EXPORT_DIR = DATA_DIR / "exports"

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOAD_DIR, PROCESSED_DIR, RAW_RESPONSES_DIR, EXPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
NANONETS_API_KEY = "c6e05194-e5eb-11ef-aea2-62ed866fc8a9"  # Move to environment variable in production
NANONETS_MODEL_ID = "24ed5d9e-fe88-4da8-a22c-1ec06ab783d2"

# Column mappings for standardization
PDF_COLUMNS = {
    'CREDIT_MONTH': 'CREDIT_MONTH',
    'WAGES': 'WAGES',
    'REF_NO': 'REF_NO',
    'PROCESSED_DATE': 'PROCESSED_DATE',
    'STATUS': 'Status'
}

EXCEL_COLUMNS = {
    'MONTH': 'MONTH',  # Column C
    'DUE_MONTH': 'DUE MONTH',  # Column D
    'WAGES': 'WAGES',  # Column E
    'YEARLY_WAGE': 'yearly wage',  # Column F
    'DUE': 'Due',  # Column G
    'PAID': 'Paid',  # Column H
    'PAYABLE': 'Payable'  # Column I
}

# Data validation settings
VALIDATION_RULES = {
    'WAGES': {
        'min_value': 0,
        'max_value': 100000
    },
    'CREDIT_MONTH': {
        'format': '%b-%y',  # Example: Nov-95
        'min_year': 1990,
        'max_year': 2030
    }
}

# Export settings
EXPORT_SETTINGS = {
    'timestamp_format': '%Y%m%d_%H%M%S',
    'default_format': 'csv',
    'available_formats': ['csv', 'xlsx'],
    'filename_template': 'comparison_results_{timestamp}.{format}'
} 