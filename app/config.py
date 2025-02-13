"""
Application configuration management.
"""
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """Handles application configuration."""
    
    def __init__(self):
        """Initialize configuration."""
        self._load_environment()
        self._setup_directories()
        self._load_credentials()
    
    def _load_environment(self):
        """Load environment variables."""
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
    
    def _setup_directories(self):
        """Ensure required directories exist."""
        directories = [
            "data/processed",
            "data/raw",
            "data/uploads",
            "raw_responses",
            "outputs/reports",
            "outputs/visualizations"
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _load_credentials(self):
        """Load API credentials."""
        # Try streamlit secrets first
        try:
            self.api_key = st.secrets["nanonets"]["NANONETS_API_KEY"]
            self.model_id = st.secrets["nanonets"]["NANONETS_MODEL_ID"]
            return
        except:
            pass
        
        # Try environment variables
        self.api_key = os.getenv("NANONETS_API_KEY")
        self.model_id = os.getenv("NANONETS_MODEL_ID")
        
        if not self.api_key or not self.model_id:
            st.error("""
            Nanonets API credentials not found! Please configure them in one of these ways:
            
            1. Create .streamlit/secrets.toml with:
               [nanonets]
               NANONETS_API_KEY = "your_api_key_here"
               NANONETS_MODEL_ID = "your_model_id_here"
               
            2. Or set environment variables in .env:
               NANONETS_API_KEY=your_api_key_here
               NANONETS_MODEL_ID=your_model_id_here
            """)
            st.stop()
    
    @property
    def debug(self) -> bool:
        """Get debug mode setting."""
        return os.getenv("DEBUG", "False").lower() == "true"
    
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return os.getenv("LOG_LEVEL", "INFO").upper()
    
    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        return Path(os.getenv("OUTPUT_DIR", "outputs"))
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(os.getenv("DATA_DIR", "data")) 