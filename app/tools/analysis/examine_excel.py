import pandas as pd
import sys
from pathlib import Path

def examine_excel(file_path):
    """Examine Excel file structure in detail."""
    print(f"\nExamining Excel file: {file_path}")
    print("-" * 80)
    
    # Get Excel info
    xl = pd.ExcelFile(file_path)
    print(f"Available sheets: {xl.sheet_names}")
    
    for sheet_name in xl.sheet_names:
        print(f"\nSheet: {sheet_name}")
        print("-" * 40)
        
        # Read without skipping rows first
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print("\nFirst 5 rows with headers:")
        print(df.head())
        
        print("\nColumn names:")
        print(df.columns.tolist())
        
        print("\nData Info:")
        print(df.info())
        
        # Try to identify where actual data starts
        print("\nAnalyzing data start point...")
        for idx, row in df.iterrows():
            if any(isinstance(val, (int, float)) and not pd.isna(val) for val in row):
                print(f"Potential data start at row {idx + 1}:")
                print(row.tolist())
                break
        
        # Check for specific columns we need
        print("\nLooking for key columns...")
        for col in df.columns:
            if any(key in str(col).upper() for key in ['MONTH', 'WAGE', 'DUE']):
                print(f"Found relevant column: {col}")
                # Show sample of non-null values
                sample = df[col].dropna().head()
                print(f"Sample values: {sample.tolist()}")

if __name__ == "__main__":
    # Examine both Excel files
    base_dir = Path("2101-312")
    excel_files = [
        base_dir / "MAIN_FILE_APR-2024.xlsx",
        base_dir / "DEBENDRA NATH AGASTI.xlsx"
    ]
    
    for file_path in excel_files:
        if file_path.exists():
            examine_excel(file_path)
        else:
            print(f"\nFile not found: {file_path}") 