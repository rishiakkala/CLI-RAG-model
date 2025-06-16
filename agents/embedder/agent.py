import os
import sys
from pathlib import Path

# Add parent directory to path to import base_agent
sys.path.append(str(Path(__file__).parent.parent.parent))
from agents.base_agent import BaseAgent
from .embedder import TextEmbedder
from .db_handler import VectorDBHandler

class EmbedderAgent(BaseAgent):
    """Agent for generating and storing vector embeddings using MiniLM."""
    
    def __init__(self, config=None):
        super().__init__("embedder", config)
        self.embedder = TextEmbedder(config)
        self.db_handler = VectorDBHandler(config)
        
    def process(self, input_data, **kwargs):
        """Process input data to generate and store embeddings."""
        operation = kwargs.get('operation', 'embed')
        
        if operation == 'embed':
            return self._embed_content(input_data, **kwargs)
        elif operation == 'batch_embed':
            return self._batch_embed(input_data, **kwargs)
        elif operation == 'delete':
            return self._delete_embeddings(input_data, **kwargs)
        else:
            return {"error": f"Unsupported operation: {operation}"}
    
    def embed_file(self, file_path, collection="default", chunk_size=512):
        """Generate embeddings for a file.
        
        Args:
            file_path (str): Path to the file to embed
            collection (str): Collection name for storing embeddings
            chunk_size (int): Size of text chunks for embedding
            
        Returns:
            dict: Result of the embedding operation
        """
        self.log_activity(f"Embedding file: {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"success": False, "message": f"File not found: {file_path}"}
                
            # Determine file type and extract text
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Extract text based on file type
            if file_ext in [".pdf"]:
                try:
                    import PyPDF2
                    text = ""
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                except ImportError:
                    return {"success": False, "message": "PyPDF2 is required for PDF processing. Install with 'pip install PyPDF2'"}
            elif file_ext in [".docx", ".doc"]:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    return {"success": False, "message": "python-docx is required for Word processing. Install with 'pip install python-docx'"}
            elif file_ext in [".txt", ".md"]:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            else:
                return {"success": False, "message": f"Unsupported file format: {file_ext}"}
            
            # Split text into chunks
            chunks = self._split_text(text, chunk_size)
            
            # Create metadata
            metadata = {
                "source": file_path,
                "file_type": file_ext,
                "total_chunks": len(chunks)
            }
            
            # Embed each chunk
            results = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                
                result = self._embed_content(
                    chunk,
                    metadata=chunk_metadata,
                    collection=collection
                )
                
                results.append(result)
            
            return {
                "success": True,
                "file_path": file_path,
                "collection": collection,
                "chunks": len(chunks),
                "results": results
            }
            
        except Exception as e:
            self.log_activity(f"Error embedding file: {str(e)}", "error")
            return {"success": False, "message": str(e)}
    
    def _split_text(self, text, chunk_size):
        """Split text into chunks of specified size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += 1
            
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
                
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _embed_content(self, content, **kwargs):
        """Generate and store embeddings for a single content item."""
        self.log_activity(f"Generating embeddings for content")
        
        try:
            # Generate metadata
            metadata = kwargs.get('metadata', {})
            collection = kwargs.get('collection', 'default')
            content_id = kwargs.get('id', None)
            
            # Generate embedding
            embedding = self.embedder.generate_embedding(content)
            
            # Store in vector database
            doc_id = self.db_handler.store_embedding(
                content, 
                embedding, 
                metadata, 
                collection=collection,
                doc_id=content_id
            )
            
            return {
                "success": True,
                "id": doc_id,
                "collection": collection,
                "vector_dimensions": len(embedding)
            }
            
        except Exception as e:
            self.log_activity(f"Error generating embeddings: {str(e)}", "error")
            return {"error": str(e)}
    
    def _batch_embed(self, content_list, **kwargs):
        """Generate and store embeddings for multiple content items."""
        self.log_activity(f"Batch embedding {len(content_list)} items")
        
        results = []
        for content_item in content_list:
            # Each item should be a dict with 'content' and optional 'metadata'
            content = content_item.get('content', '')
            metadata = content_item.get('metadata', {})
            content_id = content_item.get('id', None)
            
            result = self._embed_content(
                content, 
                metadata=metadata, 
                collection=kwargs.get('collection', 'default'),
                id=content_id
            )
            
            results.append(result)
            
        return {
            "success": True,
            "results": results,
            "total": len(results),
            "successful": sum(1 for r in results if r.get('success', False))
        }
    
    def _delete_embeddings(self, ids, **kwargs):
        """Delete embeddings from the vector database."""
        collection = kwargs.get('collection', 'default')
        
        if isinstance(ids, str):
            ids = [ids]  # Convert single ID to list
            
        self.log_activity(f"Deleting {len(ids)} embeddings from {collection}")
        
        try:
            deleted = self.db_handler.delete_embeddings(ids, collection=collection)
            
            return {
                "success": True,
                "deleted_count": deleted,
                "collection": collection
            }
            
        except Exception as e:
            self.log_activity(f"Error deleting embeddings: {str(e)}", "error")
            return {"error": str(e)}
