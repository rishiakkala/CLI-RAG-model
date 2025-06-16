import os
import sys  # Move this import up here
import logging
from typing import List, Dict, Any

# Import the Mistral model runner
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.mistral_runner import MistralRunner

class SearchResponder:
    """Generates responses to search queries using retrieved documents."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        self.llm = MistralRunner(config)
        
        # Maximum context length for the LLM
        self.max_context_length = self.config.get('max_context_length', 4000)
        
    def generate_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate a response to the query using retrieved documents."""
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(results)
            
            # Create prompt for the LLM
            prompt = self._create_prompt(query, context)
            
            # Generate response using the LLM
            response = self.llm.generate(prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _prepare_context(self, results: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results):
            # Extract content and add to context
            content = result['content']
            content_length = len(content)
            
            # Check if adding this content would exceed max context length
            if total_length + content_length > self.max_context_length:
                # If we already have some context, stop adding more
                if context_parts:
                    break
                # If this is the first item and it's too long, truncate it
                content = content[:self.max_context_length]
                content_length = self.max_context_length
            
            # Add content to context with a header
            context_parts.append(f"Document {i+1} (Relevance: {result['similarity']:.2f}):\n{content}")
            total_length += content_length
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM using the query and context."""
        return f"""You are a helpful AI assistant that answers questions based on the provided information.

Context information:
{context}

User question: {query}

Please answer the question based only on the provided context. If the context doesn't contain relevant information to answer the question, say so clearly. Your answer should be comprehensive, accurate, and helpful.

Answer:"""