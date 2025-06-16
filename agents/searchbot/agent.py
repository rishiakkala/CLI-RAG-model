import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..embedder.db_handler import VectorDBHandler
from .retriever import SearchRetriever
from .responder import SearchResponder

class SearchbotAgent:
    """Agent that performs retrieval-augmented generation using Mistral."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.db_handler = VectorDBHandler(config)
        self.retriever = SearchRetriever(config)
        self.responder = SearchResponder(config)
        
        # Default collection for searches
        self.default_collection = self.config.get('default_collection', 'default')
        
    def search(self, query: str, collection: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        """Perform a search and generate a response using RAG."""
        try:
            # Use specified collection or default
            collection = collection or self.default_collection
            
            # Get query embedding
            self.logger.info(f"Generating embedding for query: {query}")
            query_embedding = self.retriever.get_query_embedding(query)
            
            # Retrieve relevant documents
            self.logger.info(f"Retrieving documents from collection: {collection}")
            results = self.db_handler.search_similar(query_embedding, collection=collection, limit=limit)
            
            if not results:
                self.logger.warning(f"No results found in collection: {collection}")
                return {
                    "success": False,
                    "message": f"No relevant information found in collection: {collection}",
                    "response": "I couldn't find any relevant information to answer your question."
                }
            
            # Generate response using retrieved documents
            self.logger.info(f"Generating response using {len(results)} retrieved documents")
            response = self.responder.generate_response(query, results)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "response": response
            }
            
        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "response": "Sorry, I encountered an error while trying to answer your question."
            }
    
    def search_by_file(self, query: str, file_path: str, limit: int = 5) -> Dict[str, Any]:
        """Search within a specific file's embeddings."""
        try:
            # Normalize path
            file_path = os.path.abspath(file_path)
            file_name = os.path.basename(file_path)
            
            # Create collection name based on file
            collection = f"file_{file_name.replace('.', '_')}"
            
            # Check if collection exists
            if collection not in self.db_handler.collections:
                self.logger.warning(f"No embeddings found for file: {file_name}")
                return {
                    "success": False,
                    "message": f"No embeddings found for file: {file_name}",
                    "response": f"I don't have any information about {file_name}. Please embed this file first."
                }
            
            # Perform search in file-specific collection
            return self.search(query, collection=collection, limit=limit)
            
        except Exception as e:
            self.logger.error(f"Error in file search: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "response": "Sorry, I encountered an error while trying to search that file."
            }