import os
import requests
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

# Nanonets API configuration
API_KEY = "c6e05194-e5eb-11ef-aea2-62ed866fc8a9"
MODEL_ID = "24ed5d9e-fe88-4da8-a22c-1ec06ab783d2"

def save_raw_response(response_data, timestamp):
    """Save raw API response to JSON file."""
    # Create raw_responses directory if it doesn't exist
    raw_dir = Path('raw_responses')
    raw_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp
    raw_file = raw_dir / f'raw_response_{timestamp}.json'
    
    try:
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        print(f"\nRaw API response saved to: {raw_file}")
        return True
    except Exception as e:
        print(f"Warning: Failed to save raw response: {str(e)}")
        return False

def upload_file(file_path):
    """Upload a file to Nanonets for processing."""
    url = f"https://app.nanonets.com/api/v2/OCR/Model/{MODEL_ID}/LabelFile/"
    
    data = {'file': open(file_path, 'rb')}
    response = requests.post(
        url,
        auth=requests.auth.HTTPBasicAuth(API_KEY, ''),
        files=data
    )
    
    if response.status_code != 200:
        raise Exception(f"Error uploading file: {response.text}")
    
    return response.json()

def extract_table_data(result):
    """Extract table data from API response."""
    data = []
    
    for prediction in result.get('result', []):
        # Find tables in predictions
        for item in prediction.get('prediction', []):
            if isinstance(item, dict) and item.get('cells'):
                cells = item.get('cells', [])
                
                # Group cells by row
                rows = {}
                for cell in cells:
                    row = cell.get('row', 0)
                    if row not in rows:
                        rows[row] = []
                    rows[row].append(cell)
                
                # Process each row
                for row_num in sorted(rows.keys()):
                    row_cells = sorted(rows[row_num], key=lambda x: x.get('col', 0))
                    
                    # Skip header rows
                    row_text = ' '.join(c.get('text', '').upper() for c in row_cells)
                    if any(keyword in row_text for keyword in ['SL.NO', 'MONTH', 'YEAR', 'WAGES']):
                        continue
                    
                    # Extract row data
                    row_data = {
                        'SL.NO.': '',
                        'PROCESSED_DATE': '',
                        'REF_NO': '',
                        'CREDIT_MONTH': '',
                        'WAGES': '',
                        'EE': '',
                        'ER': '',
                        'Pension': '',
                        'NCP': '',
                        'Status': ''
                    }
                    
                    for cell in row_cells:
                        col = cell.get('col', 0)
                        text = cell.get('text', '').strip()
                        
                        if col == 1:  # SL.NO.
                            row_data['SL.NO.'] = text
                        elif col == 2:  # PROCESSED_DATE
                            row_data['PROCESSED_DATE'] = text
                        elif col == 3:  # REF_NO
                            row_data['REF_NO'] = text
                        elif col == 4:  # CREDIT_MONTH
                            row_data['CREDIT_MONTH'] = text
                        elif col == 5:  # WAGES
                            row_data['WAGES'] = text
                        elif col == 8:  # EE
                            row_data['EE'] = text
                        elif col == 9:  # ER
                            row_data['ER'] = text
                        elif col == 10:  # Pension
                            row_data['Pension'] = text
                        elif col == 11:  # NCP
                            row_data['NCP'] = text
                        elif col == 12:  # Status
                            row_data['Status'] = text
                    
                    # Only add rows with valid data
                    if row_data['REF_NO'] or row_data['CREDIT_MONTH']:
                        data.append(row_data)
    
    return data

def process_pdf(pdf_path):
    """Process PDF and save extracted data."""
    print(f"Processing file: {pdf_path}")
    
    try:
        # Generate timestamp once to use for both raw and processed files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get response from API
        result = upload_file(pdf_path)
        
        # Save raw response first
        save_raw_response(result, timestamp)
        
        # Extract table data
        data = extract_table_data(result)
        
        if data:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV using same timestamp
            output_file = f'contribution_details_{timestamp}.csv'
            df.to_csv(output_file, index=False)
            
            print("\nExtracted Data Preview:")
            print(df)
            print(f"\nProcessed data saved to: {output_file}")
        else:
            print("No valid data found in the PDF")
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise

if __name__ == '__main__':
    import sys
    
    # Get PDF path from command line argument or use default
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else 'test.pdf'
    
    process_pdf(pdf_path) 