"""
Nanonets OCR service integration.
"""
import os
import requests
import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from typing import Optional, Dict
from app.services.ocr.nanonets_extract import process_pdf, check_cached_response, extract_table_data

class PDFResponseCleaner:
    """Handles cleaning and standardization of PDF OCR responses."""
    
    @staticmethod
    def clean_raw_response(raw_response: Dict) -> pd.DataFrame:
        """
        Clean and standardize raw OCR response data.
        
        Args:
            raw_response: Raw JSON response from Nanonets OCR
            
        Returns:
            pd.DataFrame: Cleaned and standardized data with CREDIT_MONTH and WAGES columns
        """
        try:
            # Extract tables from response
            if 'result' not in raw_response or not raw_response['result']:
                st.error("Invalid response format: No results found")
                return pd.DataFrame()
            
            # Extract the relevant table (assuming wage data is in the first table)
            tables = raw_response['result'][0]['tables']
            if not tables:
                st.error("No tables found in the response")
                return pd.DataFrame()
            
            # Convert to DataFrame
            table_data = tables[0]['data']
            df = pd.DataFrame(table_data)
            
            # Clean up column names
            df.columns = [str(col).strip().upper() for col in df.columns]
            
            # Find month and wages columns
            month_col = next((col for col in df.columns if 'MONTH' in col), None)
            wages_col = next((col for col in df.columns if 'WAGE' in col), None)
            
            if not month_col or not wages_col:
                st.error("Required columns not found in PDF data")
                return pd.DataFrame()
            
            # Select and rename columns
            result_df = df[[month_col, wages_col]].copy()
            result_df.columns = ['CREDIT_MONTH', 'WAGES']
            
            # Clean up the data
            result_df['CREDIT_MONTH'] = result_df['CREDIT_MONTH'].apply(PDFResponseCleaner._standardize_date)
            result_df['WAGES'] = result_df['WAGES'].apply(PDFResponseCleaner._clean_wage_value)
            
            # Drop invalid rows
            result_df = result_df.dropna()
            
            return result_df
            
        except Exception as e:
            st.error(f"Error cleaning PDF response: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _standardize_date(date_str: str) -> Optional[str]:
        """
        Standardize date string to MM/YYYY format.
        
        Args:
            date_str: Input date string in various formats
            
        Returns:
            str: Standardized date in MM/YYYY format or None if invalid
        """
        try:
            # Remove any extra whitespace
            date_str = str(date_str).strip()
            
            # Try different date formats
            date_formats = [
                '%b-%y',      # Nov-95
                '%b-%Y',      # Nov-1995
                '%m/%Y',      # 11/1995
                '%m-%Y',      # 11-1995
                '%B %Y',      # November 1995
                '%Y-%m',      # 1995-11
            ]
            
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime('%m/%Y')
                except:
                    continue
            
            return None
            
        except:
            return None
    
    @staticmethod
    def _clean_wage_value(wage_str: str) -> Optional[float]:
        """
        Clean and standardize wage values.
        
        Args:
            wage_str: Input wage string with possible currency symbols and formatting
            
        Returns:
            float: Cleaned wage value or None if invalid
        """
        try:
            # Convert to string and clean up
            wage_str = str(wage_str).strip()
            
            # Remove currency symbols and other non-numeric characters
            # Keep only digits, dots, and commas
            wage_str = re.sub(r'[^\d.,]', '', wage_str)
            
            # Handle thousand separators
            # If number has multiple commas, treat them as thousand separators
            if wage_str.count(',') > 1:
                wage_str = wage_str.replace(',', '')
            # If single comma, check position to determine if it's decimal or thousand separator
            elif wage_str.count(',') == 1:
                if len(wage_str.split(',')[1]) == 2:  # Likely decimal
                    wage_str = wage_str.replace(',', '.')
                else:  # Likely thousand separator
                    wage_str = wage_str.replace(',', '')
            
            # Convert to float
            return float(wage_str)
            
        except:
            return None

class NanonetsOCR:
    """Handles OCR operations using Nanonets API."""
    
    def __init__(self, api_key: str, model_id: str):
        """
        Initialize with API credentials.
        
        Args:
            api_key: Nanonets API key (NANONETS_API_KEY)
            model_id: Nanonets model ID (NANONETS_MODEL_ID)
        """
        if not api_key or len(api_key) != 36:  # UUID format
            st.error("Invalid NANONETS_API_KEY format. Should be a 36-character UUID.")
            st.stop()
            
        if not model_id or len(model_id) != 36:  # UUID format
            st.error("Invalid NANONETS_MODEL_ID format. Should be a 36-character UUID.")
            st.stop()
            
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://app.nanonets.com/api/v2"
        self.raw_responses_dir = Path("raw_responses")
        self.raw_responses_dir.mkdir(exist_ok=True)
        
    def extract_tables(self, pdf_path: Path) -> Optional[pd.DataFrame]:
        """
        Extract tables from PDF using Nanonets OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            pd.DataFrame: Extracted and cleaned data or None if extraction fails
        """
        try:
            # Use original filename for caching
            pdf_name = pdf_path.stem
            cached_response = self._load_cached_response(pdf_name)
            
            if cached_response:
                st.info("Using cached OCR response...")
                return PDFResponseCleaner.clean_raw_response(cached_response)
            
            # If no cache, call Nanonets API
            st.info("Extracting data from PDF using Nanonets API...")
            raw_response = process_pdf(str(pdf_path), self.api_key, self.model_id)
            
            if raw_response is None:
                st.error("Failed to extract data from PDF")
                return None
            
            # Save raw response with original filename
            self._save_raw_response(pdf_name, raw_response)
            st.success(f"Raw response saved as {pdf_name}_raw_response.json")
            
            # Clean and return the response
            cleaned_data = PDFResponseCleaner.clean_raw_response(raw_response)
            if cleaned_data is not None and not cleaned_data.empty:
                st.success("Successfully extracted and cleaned data from PDF")
            return cleaned_data
            
        except Exception as e:
            st.error(f"Error extracting tables: {str(e)}")
            return None
    
    def _load_cached_response(self, pdf_name: str) -> Optional[Dict]:
        """Load cached response if it exists."""
        response_path = self.raw_responses_dir / f"{pdf_name}_raw_response.json"
        if response_path.exists():
            try:
                with open(response_path, 'r') as f:
                    response_data = json.load(f)
                    st.success(f"Found cached response: {response_path}")
                    return response_data
            except Exception as e:
                st.warning(f"Could not load cached response: {str(e)}")
                return None
        return None
    
    def _save_raw_response(self, pdf_name: str, response: Dict) -> None:
        """Save raw response to cache with original filename."""
        try:
            response_path = self.raw_responses_dir / f"{pdf_name}_raw_response.json"
            with open(response_path, 'w') as f:
                json.dump(response, f, indent=2)
            st.info(f"Saved raw response to: {response_path}")
        except Exception as e:
            st.warning(f"Could not cache response: {str(e)}")
    
    def check_duplicate_document(self, pdf_path: Path) -> Optional[Dict]:
        """
        Check if a document has been processed before.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict: Information about duplicate if found, None otherwise
        """
        is_cached, cached_data = check_cached_response(str(pdf_path))
        return cached_data if is_cached else None 