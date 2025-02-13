"""
Main application entry point.
"""
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from services.ocr.nanonets import NanonetsOCR
from components.comparison_view import ComparisonView

# Load environment variables
load_dotenv()

# Set page config first before any other Streamlit commands
st.set_page_config(
    page_title="Wage Comparison Tool",
    page_icon="📊",
    layout="wide"
)

def main():
    """Main application entry point."""
    st.title("Wage Comparison Tool")
    
    # Initialize OCR service
    ocr_service = NanonetsOCR()
    
    # Initialize comparison view
    comparison_view = ComparisonView(ocr_service)
    
    # Create tabs
    tab1, tab2 = st.tabs([
        "Process PDF & Raw Response",
        "Compare with Processed Files"
    ])
    
    # Show appropriate view based on selected tab
    with tab1:
        comparison_view.show_raw_response_tab()
    
    with tab2:
        comparison_view.show_processed_csv_tab()

if __name__ == "__main__":
    main() 