"""
PDF processing and cleaning service.
"""
import os
import pandas as pd
from pathlib import Path
import tempfile
import json
import glob
import re
from datetime import datetime
from typing import Dict, Optional
import streamlit as st
from app.services.ocr.nanonets import PDFResponseCleaner

class PDFProcessor:
    def __init__(self, upload_dir="data/uploads", processed_dir="data/processed"):
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path(processed_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def get_latest_raw_response(self):
        """Get the most recent raw response JSON file."""
        raw_responses_dir = Path('raw_responses')
        if not raw_responses_dir.exists():
            return None
            
        # Get all raw response files
        response_files = list(raw_responses_dir.glob('raw_response_*.json'))
        if not response_files:
            return None
            
        # Sort by modification time and get the latest
        latest_file = max(response_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading raw response: {str(e)}")
            return None
    
    def save_uploaded_file(self, uploaded_file):
        """Save uploaded file to temporary location and return path."""
        temp_path = self.upload_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return temp_path
    
    def clean_data(self, df):
        """Clean and standardize the extracted data."""
        try:
            if df is None or df.empty:
                return None
                
            # Convert wages to numeric, removing any currency symbols and commas
            if 'WAGES' in df.columns:
                df['WAGES'] = df['WAGES'].replace('[\₹,]', '', regex=True)
                df['WAGES'] = pd.to_numeric(df['WAGES'], errors='coerce')
            
            # Clean credit month format
            if 'CREDIT_MONTH' in df.columns:
                df['CREDIT_MONTH'] = pd.to_datetime(df['CREDIT_MONTH'], format='%m/%Y', errors='coerce')
                df['CREDIT_MONTH'] = df['CREDIT_MONTH'].dt.strftime('%m/%Y')
            
            # Drop rows where essential data is missing
            df = df.dropna(subset=['CREDIT_MONTH', 'WAGES'])
            
            # Sort by credit month
            df['sort_date'] = pd.to_datetime(df['CREDIT_MONTH'], format='%m/%Y')
            df = df.sort_values('sort_date')
            df = df.drop('sort_date', axis=1)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error cleaning data: {str(e)}")
    
    def process(self, uploaded_file, progress_callback=None):
        """Process uploaded PDF file."""
        try:
            # First try to use existing raw response
            if progress_callback:
                progress_callback("Checking for existing processed data...")
            
            raw_response = self.get_latest_raw_response()
            if raw_response:
                if progress_callback:
                    progress_callback("Found existing processed data, extracting information...")
                
                # Extract data from raw response
                result_df = PDFResponseCleaner.clean_raw_response(raw_response)
                
                if result_df is not None and not result_df.empty:
                    if progress_callback:
                        progress_callback("Successfully loaded existing data!")
                else:
                    if progress_callback:
                        progress_callback("No valid data found in existing response, processing PDF...")
                    # Fall back to processing the PDF
                    pdf_path = self.save_uploaded_file(uploaded_file)
                    result_df = self._process_new_pdf(pdf_path)
            else:
                # No existing response, process the PDF
                if progress_callback:
                    progress_callback("No existing data found, processing PDF...")
                pdf_path = self.save_uploaded_file(uploaded_file)
                result_df = self._process_new_pdf(pdf_path)
            
            if result_df is None or result_df.empty:
                raise ValueError("No data was extracted from the PDF")
            
            # Clean and validate the data
            if progress_callback:
                progress_callback("Cleaning and validating extracted data...")
            
            result_df = self.clean_data(result_df)
            
            # Validate the cleaned data
            self.validate_data(result_df)
            
            if progress_callback:
                progress_callback(f"Processing complete! Found {len(result_df)} valid entries")
                progress_callback("\nExtracted Data Preview:")
                progress_callback(str(result_df.head()))
            
            return result_df
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
        finally:
            # Cleanup uploaded file
            if 'pdf_path' in locals():
                try:
                    os.remove(pdf_path)
                except:
                    pass
    
    def _process_new_pdf(self, pdf_path: Path) -> pd.DataFrame:
        """Process a new PDF file."""
        try:
            from app.services.ocr.nanonets_extract import process_pdf
            from app.main import get_nanonets_credentials
            
            # Get API credentials
            api_key, model_id = get_nanonets_credentials()
            
            # Process PDF
            return process_pdf(str(pdf_path), api_key, model_id)
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return pd.DataFrame()
    
    def validate_data(self, df):
        """Validate extracted data."""
        if df is None or df.empty:
            raise ValueError("No data extracted from PDF")
        
        required_columns = ['CREDIT_MONTH', 'WAGES']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types
        if not pd.api.types.is_numeric_dtype(df['WAGES']):
            raise ValueError("WAGES column could not be converted to numeric type")
        
        # Validate there is actual data
        if len(df) == 0:
            raise ValueError("No valid rows found after cleaning")
        
        return True 