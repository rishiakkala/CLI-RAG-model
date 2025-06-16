import os
import logging
import numpy as np
import requests
from typing import Dict, Any, Optional, List, Union

class EmbeddingRunner:
    """Interface for running embedding model via API."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # API configuration for Gemini
        self.api_key = self.config.get('embedding_api_key', os.environ.get('GEMINI_API_KEY', '')) 
        self.api_base = self.config.get('embedding_api_base', 'https://generativelanguage.googleapis.com/v1beta')
        self.model_name = self.config.get('embedding_model', 'gemini-embedding-exp-03-07')
        self.embedding_dimension = 3072  # Gemini embedding dimension is 3072
        
        # Check if API key is available
        self._check_api_access()
        
        # Fallback to local model if API is not available
        self.use_local_fallback = self.config.get('use_local_fallback', True)
        self.local_model = None
        if self.use_local_fallback:
            self._initialize_local_model()
    
    def _check_api_access(self) -> None:
        """Check if API key is available and valid."""
        if not self.api_key:
            self.logger.warning("Gemini API key not found. Set it in config or GEMINI_API_KEY environment variable.")
            return
        
        try:
            # Make a simple API call to check access - for Gemini we'll just check models endpoint
            response = requests.get(
                f"{self.api_base}/models?key={self.api_key}"
            )
            
            if response.status_code != 200:
                self.logger.warning(f"Gemini API access check failed: {response.text}")
            else:
                self.logger.info("Gemini API access confirmed.")
                
        except Exception as e:
            self.logger.warning(f"Error checking Gemini API access: {str(e)}")
    
    def _initialize_local_model(self) -> None:
        """Initialize a local embedding model as fallback."""
        # This part remains unchanged
        try:
            # Try to import sentence_transformers
            from sentence_transformers import SentenceTransformer
            
            local_model_name = self.config.get('local_embedding_model', 'all-MiniLM-L6-v2')
            device = self.config.get('device', 'cpu')  # 'cpu' or 'cuda'
            
            self.logger.info(f"Loading local embedding model as fallback: {local_model_name}")
            self.local_model = SentenceTransformer(local_model_name, device=device)
            self.local_embedding_dimension = self.local_model.get_sentence_embedding_dimension()
            self.logger.info(f"Local embedding dimension: {self.local_embedding_dimension}")
            
        except ImportError:
            self.logger.warning("sentence_transformers not installed. Local fallback not available.")
            self.local_model = None
        except Exception as e:
            self.logger.warning(f"Error initializing local embedding model: {str(e)}")
            self.local_model = None
    
    def embed_text(self, text: str) -> Union[np.ndarray, List[float]]:
        """Generate embedding vector for a text using Gemini API or local fallback."""
        # Try API first if key is available
        if self.api_key:
            try:
                # Prepare the API request for Gemini
                headers = {
                    "Content-Type": "application/json"
                }
                
                # Gemini API payload structure
                payload = {
                    "model": f"models/{self.model_name}",
                    "content": {
                        "parts": [{
                            "text": text
                        }]
                    },
                    "taskType": "RETRIEVAL_QUERY"  # Appropriate for search queries
                }
                
                # Make the API request to Gemini embedContent endpoint
                response = requests.post(
                    f"{self.api_base}/{self.model_name}:embedContent?key={self.api_key}",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract embeddings from Gemini response
                    embedding = result.get('embedding', {}).get('values', [])
                    return np.array(embedding)
                else:
                    self.logger.warning(f"Gemini API embedding failed: {response.text}. Trying local fallback.")
            except Exception as e:
                self.logger.warning(f"Error with Gemini API embedding: {str(e)}. Trying local fallback.")
        
        # Use local model as fallback
        if self.local_model is not None:
            try:
                embedding = self.local_model.encode(text)
                return embedding
            except Exception as e:
                self.logger.error(f"Error generating local embedding: {str(e)}")
        
        # If all methods fail, return a zero vector
        self.logger.error("All embedding methods failed. Returning zero vector.")
        return np.zeros(self.embedding_dimension).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[Union[np.ndarray, List[float]]]:
        """Generate embeddings for a batch of texts using Gemini API."""
        # For Gemini, we'll process one at a time since the batch endpoint may be different
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings
    
    # The calculate_similarity method remains unchanged
    def calculate_similarity(self, embedding1: Union[np.ndarray, List[float]], 
                            embedding2: Union[np.ndarray, List[float]]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays if they aren't already
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
            
            # Calculate cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0