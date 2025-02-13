import pandas as pd
import numpy as np
from pathlib import Path

def analyze_contribution_data(file_path):
    """Analyze contribution details data."""
    print(f"\nAnalyzing contribution data from: {file_path}")
    print("-" * 80)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Basic information
    print("\nBasic Information:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Analyze CREDIT_MONTH
    if 'CREDIT_MONTH' in df.columns:
        print("\nCredit Month Analysis:")
        valid_months = df['CREDIT_MONTH'].dropna()
        print("Unique months:", len(valid_months.unique()))
        print("\nFirst few credit months:")
        try:
            # Filter out non-month entries and sort
            month_entries = valid_months[valid_months.str.contains(r'\d+/\d+', na=False)]
            print(sorted(month_entries.unique())[:10])
        except Exception as e:
            print(f"Error sorting months: {str(e)}")
    
    # Analyze WAGES
    if 'WAGES' in df.columns:
        print("\nWages Analysis:")
        try:
            # Clean wages data - remove non-numeric characters and convert to numeric
            wages = pd.to_numeric(df['WAGES'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
            print("\nWages Statistics:")
            print(wages.describe())
            
            # Check for zero or negative wages
            print(f"\nZero wages entries: {len(wages[wages == 0])}")
            print(f"Negative wages entries: {len(wages[wages < 0])}")
        except Exception as e:
            print(f"Error analyzing wages: {str(e)}")
    
    # Analyze Status
    if 'Status' in df.columns:
        print("\nStatus Analysis:")
        print(df['Status'].value_counts().dropna())
    
    # Check for duplicates
    print("\nDuplicate Analysis:")
    if 'CREDIT_MONTH' in df.columns:
        try:
            # Only check duplicates for valid month entries
            valid_entries = df[df['CREDIT_MONTH'].str.contains(r'\d+/\d+', na=False)]
            duplicates = valid_entries[valid_entries.duplicated(['CREDIT_MONTH'], keep=False)]
            print(f"Duplicate credit months: {len(duplicates)}")
            if len(duplicates) > 0:
                print("\nDuplicate entries:")
                print(duplicates[['CREDIT_MONTH', 'WAGES', 'Status']].sort_values('CREDIT_MONTH'))
        except Exception as e:
            print(f"Error checking duplicates: {str(e)}")
    
    # Save cleaned data
    try:
        # Clean the data
        cleaned_df = df.copy()
        
        # Remove rows that don't contain actual contribution data
        cleaned_df = cleaned_df[cleaned_df['CREDIT_MONTH'].str.contains(r'\d+/\d+', na=False)]
        
        # Convert wages to numeric, removing any non-numeric characters
        if 'WAGES' in cleaned_df.columns:
            cleaned_df['WAGES'] = pd.to_numeric(cleaned_df['WAGES'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
        
        # Sort by credit month
        if 'CREDIT_MONTH' in cleaned_df.columns:
            cleaned_df['sort_date'] = pd.to_datetime(cleaned_df['CREDIT_MONTH'].str.extract(r'(\d+/\d+)')[0], format='%m/%Y')
            cleaned_df = cleaned_df.sort_values('sort_date')
            cleaned_df = cleaned_df.drop('sort_date', axis=1)
        
        # Save cleaned data
        output_file = 'cleaned_contribution_details.csv'
        cleaned_df.to_csv(output_file, index=False)
        print(f"\nSaved cleaned data to: {output_file}")
        print(f"Cleaned data contains {len(cleaned_df)} rows")
        
    except Exception as e:
        print(f"\nError cleaning data: {str(e)}")

if __name__ == "__main__":
    # Analyze the contribution details
    file_path = "contribution_details_20250210_202555.csv"
    if Path(file_path).exists():
        analyze_contribution_data(file_path)
    else:
        print(f"File not found: {file_path}") 