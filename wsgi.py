import os
import sys

# Add the app directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import and run the Streamlit app
from app.main import main

if __name__ == "__main__":
    main() 