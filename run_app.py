#!/usr/bin/env python3
"""
Desktop launcher for the Wage Comparison Tool.
This script creates necessary directories and launches the Streamlit app.
"""
import os
import sys
import subprocess
from pathlib import Path
import webbrowser
from time import sleep

def setup_environment():
    """Create necessary directories if they don't exist."""
    # Create required directories
    directories = [
        'data/uploads',
        'data/processed',
        'raw_responses'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create .env file if it doesn't exist
    env_file = Path('.env')
    if not env_file.exists():
        env_example = Path('.env.example')
        if env_example.exists():
            env_file.write_text(env_example.read_text())

def get_streamlit_cmd():
    """Get the correct streamlit command based on the platform."""
    if sys.platform == 'win32':
        return ['streamlit.exe']
    return ['streamlit']

def main():
    """Main entry point for the desktop app."""
    print("Starting Wage Comparison Tool...")
    
    # Setup directories and environment
    setup_environment()
    
    # Construct the streamlit command
    streamlit_cmd = get_streamlit_cmd()
    cmd = [*streamlit_cmd, 'run', 'app/main.py']
    
    try:
        # Start the Streamlit process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait a bit for the server to start
        sleep(2)
        
        # Open the browser (Streamlit will typically do this automatically)
        # but we include this as a fallback
        webbrowser.open('http://localhost:8501')
        
        print("Application started! You can close this window when you're done.")
        process.wait()
        
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == '__main__':
    main() 