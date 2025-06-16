import os
import sys  # Add this import
import logging
from typing import List, Dict, Any
import numpy as np

# Import the embedding model runner
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.embedding_runner import EmbeddingRunner

class SearchRetriever:
    """Handles retrieval of relevant documents for search queries."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        self.embedding_model = EmbeddingRunner(config)
        
    def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the search query."""
        try:
            # Generate embedding - fix method name
            embedding = self.embedding_model.embed_text(query)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optionally rerank results based on additional criteria."""
        # This is a simple implementation that could be enhanced with more
        # sophisticated reranking algorithms
        
        # For now, we'll just return the results sorted by similarity
        # which is already done by the VectorDBHandler
        return results
    
    def filter_results(self, results: List[Dict[str, Any]], min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Filter results based on similarity threshold."""
        return [r for r in results if r['similarity'] >= min_similarity]