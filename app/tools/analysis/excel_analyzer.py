"""
Excel Analysis Tool for examining and validating Excel files.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelAnalyzer:
    """Excel file analysis tool."""
    
    def __init__(self, file_path: Path):
        """Initialize with Excel file path."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        self.excel = pd.ExcelFile(file_path)
    
    def get_sheet_names(self) -> List[str]:
        """Get list of sheet names."""
        return self.excel.sheet_names
    
    def analyze_sheet(self, sheet_name: str) -> Dict[str, Any]:
        """Analyze a specific sheet in detail."""
        logger.info(f"Analyzing sheet: {sheet_name}")
        
        results = {}
        
        # Read sheet
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        
        # Basic info
        results['columns'] = df.columns.tolist()
        results['row_count'] = len(df)
        results['column_count'] = len(df.columns)
        
        # Data types
        results['column_types'] = {col: str(df[col].dtype) for col in df.columns}
        
        # Find data start point
        for idx, row in df.iterrows():
            if any(isinstance(val, (int, float)) and not pd.isna(val) for val in row):
                results['data_start_row'] = idx
                results['data_start_sample'] = row.to_dict()
                break
        
        # Identify key columns
        key_columns = []
        for col in df.columns:
            if any(key in str(col).upper() for key in ['MONTH', 'WAGE', 'DUE']):
                sample_values = df[col].dropna().head().tolist()
                key_columns.append({
                    'name': col,
                    'sample_values': sample_values
                })
        results['key_columns'] = key_columns
        
        return results
    
    def analyze_all_sheets(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all sheets in the workbook."""
        return {sheet: self.analyze_sheet(sheet) 
                for sheet in self.get_sheet_names()}
    
    def find_wage_data(self) -> Optional[pd.DataFrame]:
        """Attempt to find and extract wage-related data."""
        logger.info("Searching for wage data")
        
        for sheet in self.get_sheet_names():
            df = pd.read_excel(self.file_path, sheet_name=sheet)
            
            # Check if this sheet likely contains wage data
            wage_indicators = ['WAGE', 'SALARY', 'CONTRIBUTION', 'MONTH']
            if any(any(indicator in str(col).upper() for indicator in wage_indicators)
                   for col in df.columns):
                logger.info(f"Found potential wage data in sheet: {sheet}")
                return df
        
        return None 