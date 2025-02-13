"""
PDF Analysis Tool for extracting and analyzing PDF documents.
"""
import pdfplumber
import camelot
from pdfminer.high_level import extract_text
from pypdf import PdfReader
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFAnalyzer:
    """PDF Analysis tool with multiple extraction methods."""
    
    def __init__(self, pdf_path: Path):
        """Initialize with PDF path."""
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze PDF structure and return detailed information."""
        logger.info(f"Analyzing PDF structure: {self.pdf_path}")
        
        results = {}
        with pdfplumber.open(self.pdf_path) as pdf:
            # Basic properties
            results['num_pages'] = len(pdf.pages)
            
            # First page analysis
            first_page = pdf.pages[0]
            results['first_page'] = {
                'words': first_page.extract_words(),
                'dimensions': {
                    'width': first_page.width,
                    'height': first_page.height
                }
            }
            
            # Table analysis (page 4)
            if len(pdf.pages) >= 4:
                contrib_page = pdf.pages[3]
                results['contribution_page'] = {
                    'words': contrib_page.extract_words(),
                    'tables': contrib_page.extract_tables()
                }
        
        return results
    
    def extract_tables(self, page_number: int = 4) -> List[Dict[str, Any]]:
        """Extract tables using multiple methods."""
        logger.info(f"Extracting tables from page {page_number}")
        
        tables = []
        
        # Try pdfplumber
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                if page_number <= len(pdf.pages):
                    page = pdf.pages[page_number - 1]
                    tables.extend({'method': 'pdfplumber', 'data': table} 
                                for table in page.extract_tables())
        except Exception as e:
            logger.error(f"pdfplumber table extraction failed: {e}")
        
        # Try camelot
        try:
            camelot_tables = camelot.read_pdf(str(self.pdf_path), pages=str(page_number))
            tables.extend({'method': 'camelot', 'data': table.df.to_dict('records')}
                         for table in camelot_tables)
        except Exception as e:
            logger.error(f"camelot table extraction failed: {e}")
        
        return tables
    
    def extract_text(self, page_number: Optional[int] = None) -> str:
        """Extract text using multiple methods and return the best result."""
        logger.info(f"Extracting text from {'all pages' if page_number is None else f'page {page_number}'}")
        
        texts = []
        
        # Try pdfminer
        try:
            text = extract_text(self.pdf_path, page_numbers=[page_number-1] if page_number else None)
            texts.append(text)
        except Exception as e:
            logger.error(f"pdfminer text extraction failed: {e}")
        
        # Try pdfplumber
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                if page_number:
                    if page_number <= len(pdf.pages):
                        text = pdf.pages[page_number-1].extract_text()
                        texts.append(text)
                else:
                    text = '\n'.join(page.extract_text() for page in pdf.pages)
                    texts.append(text)
        except Exception as e:
            logger.error(f"pdfplumber text extraction failed: {e}")
        
        # Return the longest text (assuming it's the most complete)
        return max(texts, key=len, default="") 