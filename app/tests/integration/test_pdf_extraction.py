"""
Integration tests for PDF extraction functionality.
"""
import pytest
from pathlib import Path
from app.tools.analysis.pdf_analyzer import PDFAnalyzer

@pytest.fixture
def sample_pdf_path():
    """Fixture for sample PDF path."""
    return Path('2101-312/ML2101-312.pdf')

@pytest.fixture
def pdf_analyzer(sample_pdf_path):
    """Fixture for PDFAnalyzer instance."""
    return PDFAnalyzer(sample_pdf_path)

def test_pdf_structure_analysis(pdf_analyzer):
    """Test PDF structure analysis."""
    structure = pdf_analyzer.analyze_structure()
    
    assert 'num_pages' in structure
    assert structure['num_pages'] > 0
    
    assert 'first_page' in structure
    assert 'words' in structure['first_page']
    assert 'dimensions' in structure['first_page']

def test_table_extraction(pdf_analyzer):
    """Test table extraction from contribution page."""
    tables = pdf_analyzer.extract_tables(page_number=4)
    
    assert len(tables) > 0
    for table in tables:
        assert 'method' in table
        assert 'data' in table
        assert isinstance(table['data'], (list, dict))

def test_text_extraction(pdf_analyzer):
    """Test text extraction."""
    # Test specific page
    text = pdf_analyzer.extract_text(page_number=1)
    assert isinstance(text, str)
    assert len(text) > 0
    
    # Test full document
    full_text = pdf_analyzer.extract_text()
    assert isinstance(full_text, str)
    assert len(full_text) > len(text) 