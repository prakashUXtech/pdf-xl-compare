"""
Nanonets API integration for extracting data from PDFs.
"""
import os
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
import streamlit as st
import time
import hashlib
import json
from datetime import datetime

def get_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash of file for duplicate checking."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_cached_response(pdf_path: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check if we have a cached response for this PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple[bool, Optional[Dict]]: (is_cached, cached_response if found)
    """
    try:
        # Generate file hash
        file_hash = get_file_hash(pdf_path)
        
        # Check if we have cached results for this hash
        cache_dir = Path("raw_responses")
        cache_dir.mkdir(exist_ok=True)
        
        for cache_file in cache_dir.glob("*_raw_response.json"):
            try:
                with open(cache_file) as f:
                    cached_data = json.load(f)
                    if cached_data.get('file_hash') == file_hash:
                        st.success(f"Found cached response: {cache_file.name}")
                        return True, cached_data
            except Exception as e:
                st.warning(f"Error reading cache file {cache_file}: {str(e)}")
                continue
        
        return False, None
        
    except Exception as e:
        st.warning(f"Error checking cache: {str(e)}")
        return False, None

def process_pdf(pdf_path: str, api_key: str, model_id: str) -> Optional[Dict]:
    """
    Process PDF using Nanonets API and return raw response.
    
    Args:
        pdf_path: Path to PDF file
        api_key: Nanonets API key
        model_id: Nanonets model ID
        
    Returns:
        Dict: Raw API response or None if request fails
    """
    try:
        # Ensure raw_responses directory exists
        cache_dir = Path("raw_responses")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # First check local cache
        is_cached, cached_response = check_cached_response(pdf_path)
        if is_cached and cached_response:
            return cached_response
        
        # If not in cache, process with API
        url = f"https://app.nanonets.com/api/v2/OCR/Model/{model_id}/LabelFile/"
        
        st.info(f"Uploading file to Nanonets: {Path(pdf_path).name}")
        
        # Make synchronous request
        with open(pdf_path, 'rb') as pdf_file:
            files = {'file': pdf_file}  # Simple file upload
            response = requests.post(
                url,
                auth=requests.auth.HTTPBasicAuth(api_key, ''),
                files=files
            )
        
        # Save response
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            response_json = response.json()
            st.write("API Response:", response_json)  # Show the response for debugging
            
            # Save response with timestamp
            response_file = cache_dir / f"raw_response_{timestamp}.json"
            with open(response_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'status_code': response.status_code,
                    'response': response_json,
                    'file_hash': get_file_hash(pdf_path)  # Add hash for caching
                }, f, indent=2)
            st.success(f"Saved response to: {response_file}")
            
            if response.status_code == 200:
                # Add file hash for caching
                response_json['file_hash'] = get_file_hash(pdf_path)
                return response_json
            else:
                st.error(f"API Error (Status {response.status_code}): {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error processing response: {str(e)}")
            st.write("Raw Response Text:", response.text)
            
            # Save raw text response for debugging
            error_file = cache_dir / f"error_response_{timestamp}.txt"
            with open(error_file, 'w') as f:
                f.write(response.text)
            st.info(f"Saved error response to: {error_file}")
            return None
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def clean_saved_response(response_file: str) -> Optional[pd.DataFrame]:
    """
    Clean and convert a saved raw response file to DataFrame.
    
    Args:
        response_file: Path to saved response JSON file
        
    Returns:
        pd.DataFrame: Cleaned data or None if cleaning fails
    """
    try:
        st.info(f"Loading saved response: {response_file}")
        with open(response_file, 'r') as f:
            saved_data = json.load(f)
        
        # Handle different response formats
        if 'response' in saved_data:
            raw_response = saved_data['response']
        else:
            raw_response = saved_data
            
        return extract_table_data(raw_response)
        
    except Exception as e:
        st.error(f"Error cleaning saved response: {str(e)}")
        return None

def extract_table_data(raw_response: Dict) -> Optional[pd.DataFrame]:
    """
    Extract table data from Nanonets raw response.
    
    Args:
        raw_response: Raw JSON response from Nanonets API
        
    Returns:
        pd.DataFrame: Extracted data or None if extraction fails
    """
    try:
        st.write("Extracting data from response...")
        
        if not raw_response or not isinstance(raw_response, dict):
            st.error("Invalid response format")
            return None
            
        # Get result array
        result = raw_response.get('result', [])
        if not result or not isinstance(result, list):
            st.error("No results found in response")
            return None
            
        data = []
        
        # Process each prediction
        for page in result:
            predictions = page.get('prediction', [])
            if not predictions:
                continue
                
            # Look for table data in predictions
            for pred in predictions:
                if isinstance(pred, dict):
                    # Try different possible locations for table data
                    table_data = None
                    
                    # Check direct table data
                    if 'tables' in pred:
                        table_data = pred['tables']
                    # Check cells data
                    elif 'cells' in pred:
                        table_data = [{'data': pred['cells']}]
                    # Check tabular data
                    elif 'tabular' in pred:
                        table_data = pred['tabular']
                        
                    if table_data and isinstance(table_data, list):
                        for table in table_data:
                            if not isinstance(table, dict):
                                continue
                                
                            cells = table.get('data', [])
                            if not cells:
                                continue
                                
                            # Group cells by row
                            rows = {}
                            for cell in cells:
                                row = cell.get('row', 0)
                                if row not in rows:
                                    rows[row] = []
                                rows[row].append(cell)
                            
                            # Process rows
                            header_mapping = {}
                            for row_num in sorted(rows.keys()):
                                row_cells = sorted(rows[row_num], key=lambda x: x.get('col', 0))
                                row_text = ' '.join(c.get('text', '').upper() for c in row_cells)
                                
                                # Find header row
                                if any(keyword in row_text for keyword in ['SL.NO', 'MONTH', 'YEAR', 'WAGES', 'PEN']):
                                    # Map column positions
                                    for cell in row_cells:
                                        text = cell.get('text', '').upper().strip()
                                        col = cell.get('col', 0)
                                        
                                        if 'SL' in text or 'NO' in text:
                                            header_mapping['SL.NO.'] = col
                                        elif 'MONTH' in text or 'CREDIT' in text:
                                            header_mapping['CREDIT_MONTH'] = col
                                        elif 'PEN' in text and 'WAGE' in text:
                                            header_mapping['PEN.WAGES'] = col
                                        elif 'WAGE' in text and 'PEN' not in text:
                                            header_mapping['WAGES'] = col
                                        elif 'REF' in text:
                                            header_mapping['REF_NO'] = col
                                        elif 'DATE' in text and 'PROCESS' in text:
                                            header_mapping['PROCESSED_DATE'] = col
                                        elif 'EE' in text:
                                            header_mapping['EE'] = col
                                    continue
                                
                                # Extract data using header mapping
                                if header_mapping:
                                    row_data = {}
                                    for header, col in header_mapping.items():
                                        matching_cells = [c for c in row_cells if c.get('col') == col]
                                        if matching_cells:
                                            cell_text = matching_cells[0].get('text', '').strip()
                                            # Clean up numeric values
                                            if header in ['WAGES', 'PEN.WAGES']:
                                                try:
                                                    # Remove any currency symbols and commas
                                                    cell_text = ''.join(c for c in cell_text if c.isdigit() or c in '.-')
                                                    cell_text = float(cell_text) if cell_text else None
                                                except ValueError:
                                                    cell_text = None
                                            row_data[header] = cell_text
                                    
                                    if row_data:
                                        data.append(row_data)
        
        if not data:
            st.error("No data extracted from tables")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_cols = ['CREDIT_MONTH']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns. Found: {', '.join(df.columns)}")
            return None
        
        # Clean up numeric columns
        for col in ['WAGES', 'PEN.WAGES']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by credit month
        if 'CREDIT_MONTH' in df.columns:
            df['sort_date'] = pd.to_datetime(df['CREDIT_MONTH'], format='%m/%Y', errors='coerce')
            df = df.sort_values('sort_date').drop('sort_date', axis=1)
        
        return df
        
    except Exception as e:
        st.error(f"Error extracting table data: {str(e)}")
        import traceback
        st.write("Error details:", traceback.format_exc())
        return None 