"""
Base OCR service interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pathlib

class OCRService(ABC):
    """Base class for OCR services."""
    
    @abstractmethod
    def extract_text(self, file_path: pathlib.Path) -> str:
        """Extract text from a document."""
        pass
    
    @abstractmethod
    def extract_tables(self, file_path: pathlib.Path) -> List[Dict[str, Any]]:
        """Extract tables from a document."""
        pass
    
    @abstractmethod
    def extract_form_fields(self, file_path: pathlib.Path) -> Dict[str, str]:
        """Extract form fields from a document."""
        pass 