import os
import sys
from pdf2image import convert_from_path
import tempfile
from PIL import Image
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import base64
import requests
import cv2
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')  # API key should be set in .env file

def convert_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_table_from_image(image_path, api_key):
    """Extract table data from image using Claude Vision."""
    base64_image = convert_to_base64(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please extract ALL contribution details from this image, focusing on the table data. For each row, extract:\n1. Credit Month/Year (in format M/YYYY or MM/YYYY)\n2. Wages amount\n\nReturn the data in CSV format with headers 'CREDIT_MONTH,WAGES'. Important:\n- Extract EVERY row from the table\n- Include all entries even if they look like duplicates\n- Do not skip any rows\n- Preserve the exact month format (don't add leading zeros)\n- Only exclude rows where the month/year or wages are clearly invalid\n- Verify each row has both month/year and wages before including"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image
                    }
                }
            ]
        }
    ]
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json={
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": messages
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['content'][0]['text']
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

def preprocess_image(image):
    """Enhance image quality for better extraction."""
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary)
    
    return Image.fromarray(denoised)

def crop_table_region(image):
    """Crop the contribution table region from the page."""
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour that could be a table
    max_area = 0
    table_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > (img.shape[0] * img.shape[1] * 0.2):  # At least 20% of page
            max_area = area
            table_contour = contour
    
    if table_contour is not None:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(table_contour)
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        # Crop the image
        cropped = img[y:y+h, x:x+w]
        return Image.fromarray(cropped)
    
    return image

def standardize_date(date_str):
    """Convert date string to M/YYYY format (no leading zeros)."""
    try:
        # Try M/YYYY or MM/YYYY format first
        if len(date_str.split('/')) == 2:
            month, year = date_str.split('/')
            # Remove leading zeros from month
            month = str(int(month))
            return f"{month}/{year}"
        
        # Try DD/MM/YYYY format
        parts = date_str.split('/')
        if len(parts) == 3:
            month = str(int(parts[1]))  # Remove leading zeros
            return f"{month}/{parts[2]}"
        
        return None
    except:
        return None

def get_cache_path(pdf_path):
    """Get cache directory and ensure it exists."""
    cache_dir = Path(pdf_path).parent / '.pdf_cache'
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def get_cache_key(pdf_path):
    """Generate cache key based on PDF path and modification time."""
    pdf_path = Path(pdf_path)
    mtime = pdf_path.stat().st_mtime
    return f"{pdf_path.stem}_{int(mtime)}"

def save_to_cache(cache_path, data):
    """Save data to cache file."""
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"Warning: Failed to save to cache: {str(e)}")
        return False

def load_from_cache(cache_path):
    """Load data from cache file."""
    try:
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load from cache: {str(e)}")
    return None

def save_debug_images(original_image, processed_image, page_num, output_dir='debug_images'):
    """Save original and preprocessed images for debugging."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save images
    original_image.save(output_dir / f'page_{page_num}_original.png')
    processed_image.save(output_dir / f'page_{page_num}_processed.png')

def process_pdf(pdf_path, api_key, start_page=4, save_images=True):
    """Process PDF and extract contribution details using Claude Vision."""
    print(f"Processing PDF: {pdf_path}")
    
    # Setup caching
    cache_dir = get_cache_path(pdf_path)
    cache_key = get_cache_key(pdf_path)
    data_cache = cache_dir / f"{cache_key}_data.json"
    
    # Try to load from cache first
    cached_data = load_from_cache(data_cache)
    if cached_data:
        print("\nLoaded data from cache")
        df = pd.DataFrame(cached_data, columns=['CREDIT_MONTH', 'WAGES'])
        df = deduplicate_entries(df)  # Apply deduplication
        print("\nContribution Details Preview:")
        print(df)
        return df
    
    # Create a temporary directory for page images
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Get total pages
            pages = convert_from_path(pdf_path)
            total_pages = len(pages)
            print(f"Total pages detected: {total_pages}")
            
            if start_page > total_pages:
                print(f"Start page {start_page} is greater than total pages {total_pages}")
                return
            
            # Process each page
            all_data = []
            
            for page_num in tqdm(range(start_page, total_pages + 1), desc="Processing pages"):
                try:
                    # Check if page data is cached
                    page_cache = cache_dir / f"{cache_key}_page_{page_num}.json"
                    page_data = load_from_cache(page_cache)
                    
                    if page_data:
                        print(f"\nLoaded page {page_num} from cache")
                        all_data.extend(page_data)
                        continue
                    
                    # Convert page to image
                    page = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)[0]
                    
                    # Preprocess the image
                    processed_image = preprocess_image(page)
                    
                    # Save debug images if requested
                    if save_images:
                        save_debug_images(page, processed_image, page_num)
                    
                    # Save processed image for OCR
                    temp_img_path = os.path.join(temp_dir, f'page_{page_num}.png')
                    processed_image.save(temp_img_path, 'PNG')
                    
                    # Extract data using Claude Vision
                    print(f"\nExtracting data from page {page_num}...")
                    csv_data = extract_table_from_image(temp_img_path, api_key)
                    
                    # Parse CSV data
                    if csv_data:
                        # Skip header row if present
                        lines = csv_data.strip().split('\n')
                        if len(lines) > 1:
                            data = []
                            for line in lines[1:]:  # Skip header
                                parts = line.strip().split(',')
                                if len(parts) == 2:
                                    month_year, wages = parts
                                    try:
                                        # Standardize date format
                                        std_date = standardize_date(month_year.strip())
                                        if std_date:
                                            wages = float(wages.strip())
                                            # No validation - capture all entries
                                            data.append([std_date, wages])
                                    except ValueError:
                                        continue
                            
                            # Cache the page data
                            save_to_cache(page_cache, data)
                            
                            all_data.extend(data)
                            print(f"Found {len(data)} valid entries")
                
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    continue
            
            if all_data:
                # Convert to DataFrame
                df = pd.DataFrame(all_data, columns=['CREDIT_MONTH', 'WAGES'])
                
                # Deduplicate entries
                df = deduplicate_entries(df)
                
                # Cache the complete data
                save_to_cache(data_cache, df.values.tolist())
                
                # Save to CSV
                output_csv = f'contribution_details_vision_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                df.to_csv(output_csv, index=False)
                
                print("\nContribution Details Preview:")
                print(df)
                print(f"\nContribution details saved to: {output_csv}")
                
                return df
            else:
                print("No valid contribution data was extracted.")
                return None
                
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise

def deduplicate_entries(df):
    """Remove duplicate entries and handle conflicts."""
    if df.empty:
        return df
        
    # Convert to datetime for sorting
    df['sort_date'] = pd.to_datetime(df['CREDIT_MONTH'], format='%m/%Y')
    
    # Group by month and handle duplicates
    def resolve_duplicates(group):
        if len(group) == 1:
            return group.iloc[0]
        
        # If multiple entries exist for same month, use the most frequent wage
        wage_counts = group['WAGES'].value_counts()
        most_common_wage = wage_counts.index[0]
        
        # If there's a tie, use the highest wage
        if len(wage_counts) > 1 and wage_counts.iloc[0] == wage_counts.iloc[1]:
            return group.loc[group['WAGES'].idxmax()]
        
        return group.loc[group['WAGES'] == most_common_wage].iloc[0]
    
    # Apply deduplication
    df = df.groupby('CREDIT_MONTH', as_index=False).apply(resolve_duplicates)
    
    # Sort by date and cleanup
    df = df.sort_values('sort_date')
    df = df.drop('sort_date', axis=1)
    
    # Reset index after groupby operations
    df = df.reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else 'ML2101-323.pdf'
    start_page = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    process_pdf(pdf_path, CLAUDE_API_KEY, start_page, save_images=True) 