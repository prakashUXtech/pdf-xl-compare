"""
Main application entry point.
"""
import streamlit as st
from app.services.ocr.nanonets import NanonetsOCR
from app.components.comparison_view import ComparisonView
from app.config import Config

# Set page config first before any other Streamlit commands
st.set_page_config(
    page_title="PDF vs Excel Comparison",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    """Main application entry point."""
    st.title("PDF vs Excel Data Comparison")
    
    # Initialize configuration and services
    config = Config()  # This handles environment setup and credentials
    
    # Initialize OCR service
    if 'ocr' not in st.session_state:
        st.session_state.ocr = NanonetsOCR(config.api_key, config.model_id)
    
    # Initialize comparison view
    comparison_view = ComparisonView(st.session_state.ocr)
    
    # Create tabs for different workflows
    tab1, tab2 = st.tabs([
        "Process PDF & Raw Response",
        "Compare with Processed Files"
    ])
    
    with tab1:
        comparison_view.show_raw_response_tab()
    
    with tab2:
        comparison_view.show_processed_csv_tab()

if __name__ == "__main__":
    main() 