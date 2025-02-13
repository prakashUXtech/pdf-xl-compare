import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import numpy as np
from app.utils.config import EXCEL_COLUMNS, VALIDATION_RULES

class ExcelProcessor:
    def __init__(self, upload_dir="data/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_file(self, uploaded_file):
        """Save uploaded file to temporary location and return path."""
        temp_path = self.upload_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return temp_path

    def read_excel_data(self, excel_path, progress_callback=None):
        """Read data from the appropriate sheet in the Excel file."""
        if progress_callback:
            progress_callback("Reading Excel file...")
        
        # Try to read from "Wages for Refund Calculator" sheet first
        try:
            df = pd.read_excel(excel_path, sheet_name="Wages for Refund Calculator")
            if progress_callback:
                progress_callback("Found 'Wages for Refund Calculator' sheet")
            return df
        except:
            pass
        
        # Try "Calculation Sheet(8.33%)" as fallback
        try:
            df = pd.read_excel(excel_path, sheet_name="Calculation Sheet(8.33%)")
            if progress_callback:
                progress_callback("Found 'Calculation Sheet(8.33%)' sheet")
            return df
        except:
            pass
        
        # If neither sheet is found, try the first sheet
        try:
            df = pd.read_excel(excel_path)
            if progress_callback:
                progress_callback("Using first available sheet")
            return df
        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")
    
    def clean_and_standardize_df(self, df, sheet_type, progress_callback=None):
        """Clean and standardize the dataframe based on sheet type."""
        try:
            if progress_callback:
                progress_callback("Cleaning and standardizing data...")
            
            # Remove any completely empty rows
            df = df.dropna(how='all')
            
            # Create standardized DataFrame
            standardized = pd.DataFrame()
            
            if sheet_type == "main":
                # For "Wages for Refund Calculator" sheet
                month_col = "Wage Month"
                wages_col = "Wages"
                
                if month_col not in df.columns or wages_col not in df.columns:
                    raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")
                
                standardized['CREDIT_MONTH'] = df[month_col].apply(
                    lambda x: f"{x.month}/{x.year}" if pd.notnull(x) else None
                )
                standardized['WAGES'] = pd.to_numeric(df[wages_col], errors='coerce')
                
            elif sheet_type == "calc":
                # For "Calculation Sheet(8.33%)" sheet
                month_col = "MONTH "  # Note the space after MONTH
                wages_col = "WAGES"
                
                if month_col not in df.columns or wages_col not in df.columns:
                    raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")
                
                standardized['CREDIT_MONTH'] = df[month_col].apply(
                    lambda x: f"{x.month}/{x.year}" if pd.notnull(x) else None
                )
                standardized['WAGES'] = pd.to_numeric(df[wages_col], errors='coerce')
                
            else:
                # Try to find columns by common names
                month_col = next((col for col in df.columns if 'MONTH' in str(col).upper()), None)
                wages_col = next((col for col in df.columns if 'WAGE' in str(col).upper()), None)
                
                if not month_col or not wages_col:
                    raise ValueError(f"Could not find month and wages columns. Available columns: {df.columns.tolist()}")
                
                standardized['CREDIT_MONTH'] = df[month_col].apply(
                    lambda x: f"{pd.to_datetime(x).month}/{pd.to_datetime(x).year}" if pd.notnull(x) else None
                )
                standardized['WAGES'] = pd.to_numeric(df[wages_col], errors='coerce')
            
            # Remove invalid rows
            initial_rows = len(standardized)
            standardized = standardized.dropna()
            
            if progress_callback:
                progress_callback(f"Found {len(standardized)} valid entries out of {initial_rows} total rows")
                progress_callback("\nSample of processed data:")
                progress_callback(str(standardized.head()))
            
            # Sort by date
            standardized['sort_date'] = pd.to_datetime(
                standardized['CREDIT_MONTH'].apply(lambda x: f"01/{x}"), 
                format='%d/%m/%Y'
            )
            standardized = standardized.sort_values('sort_date')
            standardized = standardized.drop('sort_date', axis=1)
            
            return standardized
            
        except Exception as e:
            raise Exception(f"Error cleaning data: {str(e)}")
    
    def process(self, uploaded_file, progress_callback=None):
        """Process uploaded Excel file."""
        try:
            # Save uploaded file
            excel_path = self.save_uploaded_file(uploaded_file)
            
            # Read the Excel data without any modifications
            df = self.read_excel_data(excel_path, progress_callback)
            
            if progress_callback:
                progress_callback("Excel file loaded successfully!")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")
        finally:
            # Cleanup uploaded file
            if 'excel_path' in locals():
                try:
                    os.remove(excel_path)
                except:
                    pass
    
    def validate_data(self, df):
        """Validate Excel data."""
        if df is None or df.empty:
            raise ValueError("No valid data found in Excel file")
        
        # Check for required columns
        required_columns = ['CREDIT_MONTH', 'WAGES']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in Excel: {missing_columns}")
        
        # Validate date range
        dates = pd.to_datetime(df['CREDIT_MONTH'].apply(lambda x: f"01/{x}"), format='%d/%m/%Y')
        min_year = dates.dt.year.min()
        max_year = dates.dt.year.max()
        
        if (min_year < VALIDATION_RULES['CREDIT_MONTH']['min_year'] or 
            max_year > VALIDATION_RULES['CREDIT_MONTH']['max_year']):
            raise ValueError(f"Date range must be between "
                           f"{VALIDATION_RULES['CREDIT_MONTH']['min_year']} and "
                           f"{VALIDATION_RULES['CREDIT_MONTH']['max_year']}")
        
        return True 