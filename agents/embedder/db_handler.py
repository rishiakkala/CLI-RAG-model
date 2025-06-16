import os
import json
import uuid
import numpy as np
from pathlib import Path

class VectorDBHandler:
    """Handles storage and retrieval of vector embeddings."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.db_path = self.config.get('db_path', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'index'))
        
        # Create DB directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize collections
        self.collections = self._load_collections()
        
    def store_embedding(self, content, embedding, metadata=None, collection='default', doc_id=None):
        """Store content with its embedding vector in the specified collection."""
        # Ensure collection exists
        if collection not in self.collections:
            self._create_collection(collection)
            
        # Generate ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
            
        # Create document
        document = {
            'id': doc_id,
            'content': content,
            'embedding': embedding,
            'metadata': metadata or {}
        }
        
        # Save document
        collection_path = os.path.join(self.db_path, collection)
        doc_path = os.path.join(collection_path, f"{doc_id}.json")
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(document['embedding'], np.ndarray):
                document['embedding'] = document['embedding'].tolist()
            json.dump(document, f)
            
        # Update collection index
        self.collections[collection].append(doc_id)
        self._save_collection_index(collection)
        
        return doc_id
    
    def delete_embeddings(self, doc_ids, collection='default'):
        """Delete embeddings from the specified collection."""
        if collection not in self.collections:
            return 0
            
        deleted_count = 0
        collection_path = os.path.join(self.db_path, collection)
        
        for doc_id in doc_ids:
            doc_path = os.path.join(collection_path, f"{doc_id}.json")
            
            if os.path.exists(doc_path):
                os.remove(doc_path)
                self.collections[collection].remove(doc_id)
                deleted_count += 1
                
        # Update collection index
        self._save_collection_index(collection)
        
        return deleted_count
    
    def search_similar(self, query_embedding, collection='default', limit=5):
        """Search for similar embeddings in the specified collection."""
        if collection not in self.collections:
            return []
            
        collection_path = os.path.join(self.db_path, collection)
        results = []
        
        # Convert query to numpy array if it's a list
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
            
        # Load all documents and calculate similarity
        for doc_id in self.collections[collection]:
            doc_path = os.path.join(collection_path, f"{doc_id}.json")
            
            if os.path.exists(doc_path):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    
                # Calculate cosine similarity
                doc_embedding = np.array(doc['embedding'])
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                results.append({
                    'id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'similarity': float(similarity)
                })
                
        # Sort by similarity (highest first) and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _load_collections(self):
        """Load existing collections from disk."""
        collections = {}
        
        if os.path.exists(self.db_path):
            for item in os.listdir(self.db_path):
                collection_path = os.path.join(self.db_path, item)
                
                if os.path.isdir(collection_path):
                    index_path = os.path.join(collection_path, 'index.json')
                    
                    if os.path.exists(index_path):
                        with open(index_path, 'r', encoding='utf-8') as f:
                            collections[item] = json.load(f)
                    else:
                        # Create index from existing files
                        doc_ids = [f.split('.')[0] for f in os.listdir(collection_path) 
                                  if f.endswith('.json') and f != 'index.json']
                        collections[item] = doc_ids
                        self._save_collection_index(item, doc_ids)
        
        # Ensure default collection exists
        if 'default' not in collections:
            collections['default'] = []
            self._create_collection('default')
            
        return collections
    
    def _create_collection(self, collection_name):
        """Create a new collection."""
        collection_path = os.path.join(self.db_path, collection_name)
        os.makedirs(collection_path, exist_ok=True)
        
        # Initialize empty collection if it doesn't exist
        if collection_name not in self.collections:
            self.collections[collection_name] = []
            self._save_collection_index(collection_name)
            
    def _save_collection_index(self, collection_name, doc_ids=None):
        """Save collection index to disk."""
        if doc_ids is None:
            doc_ids = self.collections.get(collection_name, [])
            
        collection_path = os.path.join(self.db_path, collection_name)
        index_path = os.path.join(collection_path, 'index.json')
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(doc_ids, f)
    
    def get_document(self, doc_id, collection='default'):
        """Retrieve a specific document by ID."""
        if collection not in self.collections or doc_id not in self.collections[collection]:
            return None
            
        collection_path = os.path.join(self.db_path, collection)
        doc_path = os.path.join(collection_path, f"{doc_id}.json")
        
        if not os.path.exists(doc_path):
            return None
            
        with open(doc_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_document(self, doc_id, updates, collection='default'):
        """Update an existing document."""
        document = self.get_document(doc_id, collection)
        
        if document is None:
            return False
            
        # Update document fields
        for key, value in updates.items():
            if key in document:
                document[key] = value
                
        # Save updated document
        collection_path = os.path.join(self.db_path, collection)
        doc_path = os.path.join(collection_path, f"{doc_id}.json")
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            if 'embedding' in document and isinstance(document['embedding'], np.ndarray):
                document['embedding'] = document['embedding'].tolist()
            json.dump(document, f)
            
        return True
    
    def list_collections(self):
        """List all available collections."""
        return list(self.collections.keys())
    
    def collection_stats(self, collection='default'):
        """Get statistics for a collection."""
        if collection not in self.collections:
            return None
            
        return {
            'name': collection,
            'document_count': len(self.collections[collection]),
            'documents': self.collections[collection]
        }
    
    def delete_collection(self, collection):
        """Delete an entire collection."""
        if collection == 'default':
            return False  # Prevent deletion of default collection
            
        if collection not in self.collections:
            return False
            
        # Delete collection directory
        collection_path = os.path.join(self.db_path, collection)
        if os.path.exists(collection_path):
            for file in os.listdir(collection_path):
                file_path = os.path.join(collection_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(collection_path)
            
        # Remove from collections dict
        del self.collections[collection]
        
        return True
    
    def bulk_store_embeddings(self, documents, collection='default'):
        """Store multiple embeddings at once."""
        doc_ids = []
        
        for doc in documents:
            content = doc.get('content')
            embedding = doc.get('embedding')
            metadata = doc.get('metadata', {})
            doc_id = doc.get('id')
            
            if content is not None and embedding is not None:
                stored_id = self.store_embedding(
                    content=content,
                    embedding=embedding,
                    metadata=metadata,
                    collection=collection,
                    doc_id=doc_id
                )
                doc_ids.append(stored_id)
                
        return doc_ids