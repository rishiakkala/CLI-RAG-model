import os
from pathlib import Path

def get_document_type(file_path):
    """Get the document type from file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.docx', '.doc']:
        return 'docx'
    elif ext == '.md':
        return 'md'
    elif ext == '.txt':
        return 'txt'
    else:
        raise ValueError(f"Unsupported document extension: {ext}")

def is_supported_document(file_path, supported_formats):
    """Check if the document format is supported."""
    ext = Path(file_path).suffix.lower()
    return ext in supported_formats

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks for processing."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start = end - overlap
        
        if start >= text_length:
            break
            
    return chunks