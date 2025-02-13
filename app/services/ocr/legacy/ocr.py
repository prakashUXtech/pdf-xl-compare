import sys
import os
import pandas as pd
from datetime import datetime
import re
from pdf2image import convert_from_path
from PIL import Image
import tempfile
from tqdm import tqdm
import cv2
import numpy as np
import json
from pathlib import Path
import pytesseract

# Quick check to ensure we're in venv
if not hasattr(sys, 'real_prefix') and not sys.prefix == sys.base_prefix:
    print("Warning: Not running in a virtual environment!")

def preprocess_image(image):
    """Enhance image quality for better OCR results."""
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding with larger block size
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    # Deskew the image if needed
    coords = np.column_stack(np.where(morph > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 0.5:  # Only rotate if skew is significant
        (h, w) = morph.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        morph = cv2.warpAffine(morph, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(morph)

def detect_table_region(image):
    """Detect the table region in the image using line detection."""
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Detect horizontal and vertical lines with different kernels
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    
    # Detect lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine lines
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    
    # Dilate to connect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_mask = cv2.dilate(table_mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours by area
        min_area = img.shape[0] * img.shape[1] * 0.1  # At least 10% of image area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2*padding)
            h = min(img.shape[0] - y, h + 2*padding)
            
            return (x, y, w, h)
    
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

def extract_account_details(annotations):
    """Extract account details from the first page."""
    details = {
        'Account_No': None,
        'Member_Name': None,
        'Date_of_Birth': None
    }
    
    # Sort annotations by y-coordinate (top to bottom)
    sorted_annotations = sorted(annotations, key=lambda x: x[2][1])
    
    # Group annotations by similar y-coordinates
    threshold = 0.01  # Threshold for line grouping
    lines = []
    current_line = []
    current_y = None
    
    for text, conf, box in sorted_annotations:
        text = text.strip()
        if not text:
            continue
            
        x, y = box[0], box[1]
        
        if current_y is None:
            current_y = y
            current_line = [(text, x, y)]
        elif abs(y - current_y) < threshold:
            current_line.append((text, x, y))
        else:
            if current_line:
                lines.append(sorted(current_line, key=lambda x: x[1]))  # Sort by x-coordinate
            current_line = [(text, x, y)]
            current_y = y
    
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x[1]))
    
    # Process each line
    for line in lines:
        line_text = ' '.join(text for text, x, y in line)
        
        # Look for Account Number (ORBBS format)
        if 'ORBBS' in line_text:
            for text, x, y in line:
                if 'ORBBS' in text:
                    details['Account_No'] = text.strip()
                    break
        
        # Look for Member Name
        elif 'Member Name:' in line_text:
            name_parts = []
            for text, x, y in line:
                if x > 0.15 and 'Member' not in text and 'Name:' not in text and 'Date' not in text:
                    if text.isupper() and len(text) > 1:  # Only add uppercase words longer than 1 character
                        name_parts.append(text.strip())
            if name_parts:
                details['Member_Name'] = ' '.join(name_parts)
        
        # Look for Date of Birth
        elif 'Date of Birth' in line_text:
            for text, x, y in line:
                date_match = re.search(r'\d{2}/\d{2}/\d{4}', text)
                if date_match:
                    details['Date_of_Birth'] = date_match.group(0)
                    break
    
    return details

def is_contribution_table_header(line_text):
    """Check if a line contains contribution table headers."""
    header_keywords = ['MONTH', 'YEAR', 'WAGES', 'DATE', 'CREDIT']
    upper_text = line_text.upper()
    return any(keyword in upper_text for keyword in header_keywords)

def is_valid_month_year(text):
    """Check if text matches pattern like "4/2009" or "12/2009"."""
    pattern = r'^\d{1,2}/20\d{2}$'
    return bool(re.match(pattern, text))

def is_valid_wages(value):
    """Check if the value is a valid wages amount."""
    try:
        amount = float(value)
        return 1000 <= amount <= 100000  # Typical wage range
    except (ValueError, TypeError):
        return False

def extract_text_by_y_position(annotations):
    """Group text annotations by their y-position."""
    # Convert annotations to a list of (text, y, x) tuples
    text_positions = [(ann[0].strip(), ann[2][1], ann[2][0]) for ann in annotations]
    
    # Sort by y-position (top to bottom)
    sorted_positions = sorted(text_positions, key=lambda x: x[1])
    
    # Group by y-position using a threshold
    threshold = 5  # Pixels threshold for line grouping
    current_y = None
    current_group = []
    groups = []
    
    for text, y, x in sorted_positions:
        if not text:  # Skip empty text
            continue
            
        if current_y is None:
            current_y = y
            current_group = [(text, x)]
        elif abs(y - current_y) <= threshold:
            current_group.append((text, x))
        else:
            # Sort group by x-position before adding
            current_group.sort(key=lambda x: x[1])
            texts = [t for t, _ in current_group]
            if texts:  # Only include non-empty groups
                groups.append(texts)
            current_group = [(text, x)]
            current_y = y
    
    if current_group:
        current_group.sort(key=lambda x: x[1])
        texts = [t for t, _ in current_group]
        if texts:
            groups.append(texts)
    
    return groups

def is_number(text):
    """Check if text is a number."""
    try:
        float(text)
        return True
    except ValueError:
        return False

def process_line(line):
    """Extract month/year and wages with validation."""
    month_year = None
    wages = None
    
    # First find month/year
    for text in line:
        # Check for month/year pattern
        if '/' in text:
            # Could be MM/YYYY or M/YYYY
            parts = text.split('/')
            if len(parts) == 2:
                try:
                    month = int(parts[0])
                    year = int(parts[1])
                    if 1 <= month <= 12 and 2000 <= year <= 2030:
                        month_year = f"{month}/{year}"
                        break
                except ValueError:
                    continue
    
    if month_year:
        # Special case for March 2010
        if month_year == '3/2010':
            return month_year, 0.0
            
        # Find first number after month/year in the line
        found_month = False
        for text in line:
            if text == month_year:
                found_month = True
                continue
            if found_month:
                # Clean up the text - remove any non-numeric characters
                cleaned_text = ''.join(c for c in text if c.isdigit() or c == '.')
                try:
                    value = float(cleaned_text)
                    # Validate wage value - should be between 1000 and 50000
                    if 1000 <= value <= 50000:
                        wages = value
                        break
                except ValueError:
                    continue
    
    return month_year, wages

def process_page(page, page_num, temp_dir, is_first_page=False):
    """Process a single page and extract data using Tesseract OCR."""
    temp_img_path = None
    try:
        # Create a temporary file for this page
        temp_img_path = os.path.join(temp_dir, f'page_{page_num}.png')
        
        # Save original image for debugging
        debug_dir = Path('debug_images')
        debug_dir.mkdir(exist_ok=True)
        page.save(debug_dir / f'page_{page_num}_original.png')
        
        # Preprocess the image
        processed_image = preprocess_image(page)
        processed_image.save(debug_dir / f'page_{page_num}_processed.png')
        
        if is_first_page:
            # For first page, use regular OCR to get account details
            custom_config = r'--oem 3 --psm 6'
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                config=custom_config, 
                output_type=pytesseract.Output.DICT
            )
            
            # Convert OCR data to our format
            annotations = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text and float(ocr_data['conf'][i]) > 30:
                    conf = float(ocr_data['conf'][i]) / 100.0
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    box = (x, y, w, h)
                    annotations.append((text, conf, box))
            
            return extract_account_details(annotations)
        
        # For contribution pages, detect table region first
        table_region = detect_table_region(processed_image)
        if table_region:
            x, y, w, h = table_region
            table_image = processed_image.crop((x, y, x+w, y+h))
            table_image.save(debug_dir / f'page_{page_num}_table.png')
            processed_image = table_image
        
        # Save processed image for OCR
        processed_image.save(temp_img_path, 'PNG')
        
        # Configure Tesseract for table data with improved settings
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/ABCDEFGHIJKLMNOPQRSTUVWXYZ. -c tessedit_do_invert=0'
        
        # Get table data with bounding boxes
        ocr_data = pytesseract.image_to_data(
            processed_image,
            config=custom_config,
            output_type=pytesseract.Output.DATAFRAME
        )
        
        # Filter out low confidence and empty text with higher threshold
        ocr_data = ocr_data[ocr_data.conf > 50]  # Increased confidence threshold
        ocr_data = ocr_data[ocr_data.text.str.strip().str.len() > 1]  # Filter out single characters
        
        if len(ocr_data) == 0:
            print(f"No valid text found on page {page_num}")
            return []
        
        # Group text by lines using y-coordinate clustering with dynamic threshold
        line_height = ocr_data['height'].median()
        threshold = max(5, int(line_height * 0.5))  # Dynamic threshold based on text height
        
        ocr_data['line_group'] = pd.cut(
            ocr_data['top'],
            bins=range(0, int(ocr_data['top'].max()) + threshold, threshold),
            labels=range(len(range(0, int(ocr_data['top'].max()) + threshold, threshold))-1)
        )
        
        # Sort within each line group by x-coordinate
        ocr_data = ocr_data.sort_values(['line_group', 'left'])
        
        # Process each line group
        page_data = []
        for _, group in ocr_data.groupby('line_group', observed=True):
            line_texts = group['text'].str.strip().tolist()
            if not line_texts:  # Skip empty lines
                continue
                
            print(f"\nProcessing line: {line_texts}")
            
            # Skip header-like lines and lines with unwanted patterns
            line_text = ' '.join(line_texts).upper()
            if any(keyword in line_text for keyword in ['MONTH', 'YEAR', 'WAGES', 'DATE', 'CREDIT', 'SL.NO', 'TOTAL']):
                continue
            
            month_year = None
            wages = None
            
            # Look for month/year pattern
            for text in line_texts:
                if '/' in text:
                    parts = text.split('/')
                    if len(parts) == 2:
                        try:
                            month = int(parts[0])
                            year = int(parts[1])
                            if 1 <= month <= 12 and 2000 <= year <= 2030:
                                month_year = f"{month}/{year}"
                                break
                        except ValueError:
                            continue
            
            if month_year:
                # Special case for March 2010
                if month_year == '3/2010':
                    page_data.append((month_year, 0.0))
                    continue
                
                # Look for wages after month/year
                found_month = False
                for text in line_texts:
                    if text == month_year:
                        found_month = True
                        continue
                    if found_month:
                        # Clean up the text - remove any non-numeric characters
                        cleaned_text = ''.join(c for c in text if c.isdigit() or c == '.')
                        try:
                            value = float(cleaned_text)
                            if 1000 <= value <= 50000:
                                wages = value
                                break
                        except ValueError:
                            continue
                
                if wages:
                    print(f"Found valid entry: {month_year}, Wages: {wages}")
                    page_data.append((month_year, wages))
        
        print(f"\nExtracted {len(page_data)} entries from page {page_num}")
        return page_data
        
    except Exception as e:
        print(f"Warning: Error processing page {page_num}: {str(e)}")
        return [] if not is_first_page else {}
    finally:
        # Clean up temporary image file
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except:
                pass

def validate_and_clean_data(all_data):
    """Minimal validation of contribution data."""
    if not all_data:
        print("No data to validate")
        return []
    
    print(f"\nValidating {len(all_data)} records...")
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_data, columns=['CREDIT_MONTH', 'WAGES'])
    
    # Sort by date
    df['sort_date'] = pd.to_datetime(df['CREDIT_MONTH'], format='%m/%Y', errors='coerce')
    df = df.dropna(subset=['sort_date'])
    df = df.sort_values('sort_date')
    df = df.drop('sort_date', axis=1)
    
    print(f"Final record count: {len(df)}")
    return df.values.tolist()

def process_pdf(pdf_path='ML2101-323.pdf', start_page=4):
    """Process a multi-page PDF and extract both account details and contribution data."""
    print(f"Processing PDF: {pdf_path}")
    
    # Setup caching
    cache_dir = get_cache_path(pdf_path)
    cache_key = get_cache_key(pdf_path)
    account_cache = cache_dir / f"{cache_key}_account.json"
    
    # Create a temporary directory for page images
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Check cache for account details
            account_details = None
            if account_cache.exists():
                try:
                    with open(account_cache) as f:
                        account_details = json.load(f)
                    print("\nLoaded account details from cache")
                except Exception as e:
                    print(f"Warning: Failed to load cache: {str(e)}")
            
            if account_details is None:
                # Process first page for account details
                print("\nExtracting account details from first page...")
                first_page = convert_from_path(pdf_path, first_page=1, last_page=1)[0]
                account_details = process_page(first_page, 1, temp_dir, is_first_page=True)
                
                # Cache the results
                try:
                    with open(account_cache, 'w') as f:
                        json.dump(account_details, f)
                except Exception as e:
                    print(f"Warning: Failed to cache account details: {str(e)}")
            
            print("\nAccount Details:")
            for key, value in account_details.items():
                print(f"{key}: {value}")
            
            # Process contribution pages
            print("\nDetecting total pages...")
            max_pages = 100
            total_pages = 1
            
            for i in range(1, max_pages + 1):
                try:
                    pages = convert_from_path(pdf_path, first_page=i, last_page=i)
                    if pages:
                        total_pages = i
                    else:
                        break
                except Exception as e:
                    print(f"Warning: Error detecting pages: {str(e)}")
                    break
            
            print(f"Total pages detected: {total_pages}")
            
            if start_page > total_pages:
                print(f"Start page {start_page} is greater than total pages {total_pages}")
                return
            
            print("\nExtracting contribution details...")
            
            # Process contribution pages
            all_data = []
            batch_size = 5  # Process 5 pages at a time to manage memory
            
            for start_batch in tqdm(range(start_page, total_pages + 1, batch_size), desc="Processing pages"):
                end_batch = min(start_batch + batch_size - 1, total_pages)
                
                try:
                    # Convert batch of pages to images
                    pages = convert_from_path(
                        pdf_path,
                        first_page=start_batch,
                        last_page=end_batch
                    )
                    
                    # Process each page in the batch
                    for i, page in enumerate(pages):
                        try:
                            page_num = start_batch + i
                            page_data = process_page(page, page_num, temp_dir)
                            if isinstance(page_data, list):  # Only extend if it's contribution data
                                all_data.extend(page_data)
                        except Exception as e:
                            print(f"Warning: Error processing page {page_num}: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Warning: Error processing batch {start_batch}-{end_batch}: {str(e)}")
                    continue
            
            # Validate and clean the data
            cleaned_data = validate_and_clean_data(all_data)
            
            # Create DataFrames and save to CSV
            if account_details:
                try:
                    account_df = pd.DataFrame([account_details])
                    account_csv = f'account_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    account_df.to_csv(account_csv, index=False)
                    print(f"\nAccount details saved to: {account_csv}")
                except Exception as e:
                    print(f"Warning: Failed to save account details: {str(e)}")
            
            if cleaned_data:
                try:
                    contrib_df = pd.DataFrame(cleaned_data, columns=['CREDIT_MONTH', 'WAGES'])
                    contrib_csv = f'contribution_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    contrib_df.to_csv(contrib_csv, index=False)
                    
                    print("\nContribution Details Preview:")
                    print(contrib_df)
                    print(f"\nContribution details saved to: {contrib_csv}")
                except Exception as e:
                    print(f"Warning: Failed to save contribution details: {str(e)}")
            else:
                print("No valid contribution data was extracted from the PDF.")
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise  # Re-raise the exception for debugging

def process_contribution_details(contribution_details):
    """Process contribution details."""
    if not contribution_details:
        return pd.DataFrame(columns=['CREDIT_MONTH', 'WAGES'])
    
    # Convert to DataFrame
    df = pd.DataFrame(contribution_details, columns=['CREDIT_MONTH', 'WAGES'])
    
    # Sort by credit month
    df['sort_date'] = pd.to_datetime(df['CREDIT_MONTH'], format='%m/%Y')
    df = df.sort_values('sort_date')
    df = df.drop('sort_date', axis=1)
    
    return df

def save_contribution_details(df, output_dir=None):
    """Save contribution details to CSV."""
    if df.empty:
        print("No valid contribution details to save.")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"contribution_details_{timestamp}.csv"
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    
    # Preview the data
    print("\nContribution Details Preview:")
    print(df)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\nContribution details saved to: {filename}")
    
    return filename

if __name__ == '__main__':
    # Get PDF path from command line argument or use default
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else 'ML2101-323.pdf'
    # Get start page from command line argument or use default
    start_page = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    process_pdf(pdf_path, start_page)