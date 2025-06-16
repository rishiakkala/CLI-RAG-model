import sys
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.embedding_runner import EmbeddingRunner

class TextEmbedder:
    """Generates vector embeddings from text using MiniLM."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = EmbeddingRunner()
        self.chunk_size = self.config.get('chunk_size', 512)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        
    def generate_embedding(self, text):
        """Generate embedding vector for a text."""
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Generate embedding
        embedding = self.model.embed_text(processed_text)
        
        return embedding
    
    def generate_embeddings_for_chunks(self, text):
        """Split text into chunks and generate embeddings for each chunk."""
        # Split text into chunks
        chunks = self._chunk_text(text)
        
        # Generate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            embedding = self.generate_embedding(chunk)
            embeddings.append({
                'text': chunk,
                'embedding': embedding
            })
            
        return embeddings
    
    def _preprocess_text(self, text):
        """Preprocess text before embedding."""
        # Simple preprocessing - in a real implementation, this would be more sophisticated
        processed = text.strip()
        
        # Truncate if too long
        if len(processed) > self.chunk_size * 10:  # Arbitrary limit
            processed = processed[:self.chunk_size * 10]
            
        return processed
    
    def _chunk_text(self, text):
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
            if start >= text_length:
                break
                
        return chunks