import os
import sys
from pathlib import Path

# Add parent directory to path to import base_agent
sys.path.append(str(Path(__file__).parent.parent.parent))
from agents.base_agent import BaseAgent
from .summarizer import DocumentSummarizer
from .utils import get_document_type, is_supported_document

class DocumentLoaderAgent(BaseAgent):  # Changed from DocLoaderAgent
    """Agent for loading and summarizing documents using Mistral."""
    
    SUPPORTED_FORMATS = ['.pdf', '.docx', '.doc', '.md', '.txt']
    
    def __init__(self, config=None):
        super().__init__("doc_loader", config)
        self.summarizer = DocumentSummarizer(config)
        
    def process(self, file_path, **kwargs):
        """Process a document file and return a summary."""
        if not self.validate_input(file_path):
            return {"error": f"Unsupported document format: {file_path}"}
            
        self.log_activity(f"Processing document: {file_path}")
        
        try:
            # Extract text from document
            doc_type = get_document_type(file_path)
            doc_text = self.summarizer.extract_text(file_path, doc_type)
            
            # Generate summary
            summary_length = kwargs.get('summary_length', 'medium')
            summary = self.summarizer.generate_summary(doc_text, summary_length)
            
            return {
                "success": True,
                "file_path": file_path,
                "document_type": doc_type,
                "summary": summary,
                "text_length": len(doc_text)
            }
            
        except Exception as e:
            self.log_activity(f"Error processing document: {str(e)}", "error")
            return {"error": str(e)}
    
    def validate_input(self, file_path):
        """Validate that the input file exists and is a supported format."""
        if not os.path.exists(file_path):
            self.log_activity(f"File not found: {file_path}", "error")
            return False
            
        if not is_supported_document(file_path, self.SUPPORTED_FORMATS):
            self.log_activity(f"Unsupported format: {file_path}", "error")
            return False
            
        return True

    def summarize_document(self, file_path, **kwargs):
        """Alias for process method to maintain compatibility."""
        return self.process(file_path, **kwargs)