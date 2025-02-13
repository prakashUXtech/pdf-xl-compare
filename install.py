#!/usr/bin/env python3
"""
Installation script for the Wage Comparison Tool.
Sets up the Python environment and installs required packages.
"""
import os
import sys
import venv
import subprocess
from pathlib import Path

def create_virtual_environment():
    """Create a virtual environment for the application."""
    print("Creating virtual environment...")
    venv.create('venv', with_pip=True)

def get_pip_cmd():
    """Get the pip command based on the platform."""
    if sys.platform == 'win32':
        return str(Path('venv/Scripts/pip.exe').absolute())
    return str(Path('venv/bin/pip').absolute())

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    pip_cmd = get_pip_cmd()
    
    # Install required packages
    subprocess.check_call([pip_cmd, 'install', '-r', 'requirements.txt'])

def create_desktop_shortcut():
    """Create a desktop shortcut to launch the application."""
    if sys.platform == 'win32':
        # Windows shortcut
        import winshell
        from win32com.client import Dispatch
        
        desktop = Path(winshell.desktop())
        path = desktop / "Wage Comparison Tool.lnk"
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(str(path))
        shortcut.Targetpath = str(Path('venv/Scripts/python.exe').absolute())
        shortcut.Arguments = str(Path('run_app.py').absolute())
        shortcut.WorkingDirectory = str(Path.cwd())
        shortcut.save()
    else:
        # Linux/Mac desktop entry
        desktop = Path.home() / 'Desktop'
        path = desktop / "wage-comparison-tool.desktop"
        
        content = f"""[Desktop Entry]
Name=Wage Comparison Tool
Exec={sys.executable} {Path('run_app.py').absolute()}
Type=Application
Terminal=false
Categories=Utility;
"""
        path.write_text(content)
        path.chmod(0o755)

def main():
    """Main installation function."""
    try:
        print("Starting installation...")
        
        # Create virtual environment
        create_virtual_environment()
        
        # Install requirements
        install_requirements()
        
        # Create desktop shortcut
        create_desktop_shortcut()
        
        print("""
Installation completed successfully!

You can now:
1. Double-click the desktop shortcut to launch the application
2. Run 'python run_app.py' from the command line

Note: Make sure to configure your API keys in the .env file before running the application.
""")
        
    except Exception as e:
        print(f"Error during installation: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 