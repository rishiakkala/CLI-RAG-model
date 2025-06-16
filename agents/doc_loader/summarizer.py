import os
import sys
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.mistral_runner import MistralRunner

class DocumentSummarizer:
    """Handles document text extraction and summarization using Mistral."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = MistralRunner()
        
    def extract_text(self, file_path, doc_type):
        """Extract text from various document formats."""
        if doc_type == 'pdf':
            return self._extract_from_pdf(file_path)
        elif doc_type in ['docx', 'doc']:
            return self._extract_from_word(file_path)
        elif doc_type in ['md', 'txt']:
            return self._extract_from_text(file_path)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF files."""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing. Install with 'pip install PyPDF2'")
    
    def _extract_from_word(self, file_path):
        """Extract text from Word documents."""
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ImportError("python-docx is required for Word processing. Install with 'pip install python-docx'")
    
    def _extract_from_text(self, file_path):
        """Extract text from plain text or markdown files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def generate_summary(self, text, length='medium'):
        """Generate a summary of the document using Mistral."""
        # Define summary length parameters
        length_params = {
            'short': 500,
            'medium': 900,
            'long': 1400
        }
        max_tokens = length_params.get(length, 900)
        
        # Create prompt for the model
        prompt = f"""Please summarize the following document in a clear and concise way. 
        Focus on the main points and key information. 
        
        Document text:
        {text[:10000]}  # Limit text to prevent token overflow
        
        Summary:"""
        
        # Generate summary using Mistral
        response = self.model.generate(prompt, max_tokens=max_tokens)
        return response.strip()