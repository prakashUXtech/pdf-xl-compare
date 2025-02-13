import pdfplumber
import camelot
from pdfminer.high_level import extract_text
import sys
from pypdf import PdfReader

def analyze_pdf_structure(pdf_path):
    print(f"\nAnalyzing PDF structure: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        # First page analysis
        print("\n=== First Page Analysis ===")
        first_page = pdf.pages[0]
        
        # Get all text with positions
        words = first_page.extract_words()
        print("\nFirst 10 words with positions:")
        for word in words[:10]:
            print(f"Text: '{word['text']}', x0: {word['x0']:.2f}, top: {word['top']:.2f}")
        
        # Try to find account details
        print("\nLooking for account details...")
        for word in words:
            if 'ORBBS' in word['text']:
                print(f"Found Account No: {word['text']}")
            elif 'Member Name' in word['text']:
                # Look for nearby words
                member_y = word['top']
                name_parts = [w['text'] for w in words if abs(w['top'] - member_y) < 5 and w['x0'] > word['x0']]
                print(f"Possible Member Name parts: {name_parts}")
        
        # Analyze contribution pages
        print("\n=== Contribution Page Analysis (Page 4) ===")
        contrib_page = pdf.pages[3]
        words = contrib_page.extract_words()
        
        # Look for table structure
        print("\nAnalyzing possible table structure...")
        y_positions = {}
        for word in words:
            y = round(word['top'], 1)  # Round to nearest 0.1 to group similar y-positions
            if y not in y_positions:
                y_positions[y] = []
            y_positions[y].append((word['text'], word['x0']))
        
        # Print first few rows
        print("\nPossible table rows:")
        for y in sorted(list(y_positions.keys())[:5]):
            row = sorted(y_positions[y], key=lambda x: x[1])  # Sort by x position
            print(f"Y: {y:.1f} -> {[text for text, _ in row]}")

def test_pdfplumber(pdf_path):
    print("\n=== Testing pdfplumber ===")
    with pdfplumber.open(pdf_path) as pdf:
        # First page for account details
        first_page = pdf.pages[0]
        print("\nFirst page text:")
        print(first_page.extract_text()[:500] + "...")
        
        # Try table extraction
        print("\nTrying table extraction from page 4:")
        table_page = pdf.pages[3]  # 0-based index
        tables = table_page.extract_tables()
        for table in tables:
            print("\nFound table:")
            for row in table[:5]:  # First 5 rows
                print(row)

def test_camelot(pdf_path):
    print("\n=== Testing camelot ===")
    # Try table extraction from page 4
    tables = camelot.read_pdf(pdf_path, pages='4')
    print(f"\nFound {len(tables)} tables")
    for idx, table in enumerate(tables):
        print(f"\nTable {idx + 1} (first 5 rows):")
        print(table.df.head())

def test_pdfminer(pdf_path):
    print("\n=== Testing pdfminer ===")
    # Extract text from first page
    text = extract_text(pdf_path, page_numbers=[0])
    print("\nFirst page text:")
    print(text[:500] + "...")

def analyze_pdf_properties(pdf_path):
    print(f"\nAnalyzing PDF properties: {pdf_path}")
    
    # Using pypdf to check PDF properties
    reader = PdfReader(pdf_path)
    
    print("\n=== PDF Properties ===")
    print(f"Number of pages: {len(reader.pages)}")
    print(f"Is Encrypted: {reader.is_encrypted}")
    
    # Check metadata
    print("\nMetadata:")
    for key, value in reader.metadata.items():
        print(f"{key}: {value}")
    
    # Check first page properties
    page = reader.pages[0]
    print("\nFirst Page Properties:")
    print(f"Page Size: {page.mediabox}")
    print(f"Rotation: {page.rotation}")
    
    # Try to extract raw text from first page
    print("\nFirst page raw text sample:")
    text = page.extract_text()
    print(text[:500] + "..." if text else "No text extracted")
    
    # Check if PDF has form fields
    print("\nForm Fields:")
    if hasattr(reader, 'get_fields'):
        fields = reader.get_fields()
        if fields:
            for field_name, field_value in fields.items():
                print(f"{field_name}: {field_value}")
        else:
            print("No form fields found")
    
    # Try to get page layout
    print("\nPage Layout:")
    if hasattr(page, '/Type'):
        print(f"Page Type: {page['/Type']}")
    if hasattr(page, '/Resources'):
        print("Resources found:")
        for resource_type in page['/Resources']:
            print(f"- {resource_type}")

def analyze_text_extraction(pdf_path):
    print("\n=== Text Extraction Analysis ===")
    
    with pdfplumber.open(pdf_path) as pdf:
        # First page analysis
        first_page = pdf.pages[0]
        
        # Get text in different ways
        print("\n1. Raw text extraction:")
        text = first_page.extract_text()
        print(text[:200] + "..." if text else "No text extracted")
        
        print("\n2. Character extraction (first 10):")
        chars = first_page.chars
        for char in chars[:10]:
            print(f"Char: '{char['text']}', Font: {char.get('fontname', 'N/A')}, "
                  f"Size: {char.get('size', 'N/A')}, "
                  f"Position: ({char['x0']:.1f}, {char['top']:.1f})")
        
        # Try to find specific content
        print("\n3. Looking for specific content:")
        for char in chars:
            if 'ORBBS' in char['text']:
                print(f"Found Account Number at position: ({char['x0']:.1f}, {char['top']:.1f})")
                print(f"Using font: {char.get('fontname', 'N/A')}")
            elif 'Member' in char['text']:
                print(f"Found 'Member' at position: ({char['x0']:.1f}, {char['top']:.1f})")
                print(f"Using font: {char.get('fontname', 'N/A')}")

if __name__ == '__main__':
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else 'ML2101-323.pdf'
    analyze_pdf_structure(pdf_path)
    analyze_pdf_properties(pdf_path)
    analyze_text_extraction(pdf_path)
    
    print(f"Testing extraction methods on: {pdf_path}")
    
    try:
        test_pdfplumber(pdf_path)
    except Exception as e:
        print(f"pdfplumber error: {str(e)}")
    
    try:
        test_camelot(pdf_path)
    except Exception as e:
        print(f"camelot error: {str(e)}")
    
    try:
        test_pdfminer(pdf_path)
    except Exception as e:
        print(f"pdfminer error: {str(e)}") 