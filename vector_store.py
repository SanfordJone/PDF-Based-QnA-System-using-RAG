import os
import uuid

class VectorStore:
    """
    Class for creating and managing vector embeddings and search functionality
    with a fallback to simple text search if ChromaDB is not available
    """
    
    def __init__(self, collection_name="pdf_documents", persist_directory="./chroma_db"):
        """Initialize the vector store"""
        # Dictionary to track documents by ID
        self.documents = {}
        
        # Flag to determine if we're using ChromaDB or fallback
        self.using_chromadb = False
        
        # Try to initialize ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Ensure the persist directory exists
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.using_chromadb = True
            print("Using ChromaDB for vector search")
        except ImportError:
            print("ChromaDB not available, using simple text search fallback")
    
    def add_document(self, text, metadata=None):
        """
        Add a document to the vector store
        
        Args:
            text (str): Document text
            metadata (dict, optional): Document metadata
        
        Returns:
            str: Document ID
        """
        if not text:
            return None
            
        # Generate a unique document ID
        doc_id = str(uuid.uuid4())
        
        # Add metadata if provided, or create empty dict
        doc_metadata = metadata or {}
        doc_metadata['doc_id'] = doc_id
        
        # Store document in our tracking dictionary
        self.documents[doc_id] = {
            'text': text,
            'metadata': doc_metadata
        }
        
        # Add document to ChromaDB if available
        if self.using_chromadb:
            try:
                self.collection.add(
                    documents=[text],
                    metadatas=[doc_metadata],
                    ids=[doc_id]
                )
            except Exception as e:
                print(f"Error adding document to ChromaDB: {e}")
        
        return doc_id
    
    def search(self, query, k=5):
        """
        Search for documents similar to the query
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            list: List of dicts containing matched documents and metadata
        """
        if self.using_chromadb:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k
                )
                
                matches = []
                if results and results['documents']:
                    for i, doc_id in enumerate(results['ids'][0]):
                        matches.append({
                            'document': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'id': doc_id
                        })
                        
                return matches
            except Exception as e:
                print(f"Error searching with ChromaDB: {e}")
                # Fall back to simple search if ChromaDB search fails
        
        # Simple search fallback - search for the query in the document text
        matches = []
        query = query.lower()
        
        for doc_id, doc_data in self.documents.items():
            if query in doc_data['text'].lower():
                matches.append({
                    'document': doc_data['text'],
                    'metadata': doc_data['metadata'],
                    'id': doc_id
                })
                
                if len(matches) >= k:
                    break
        
        return matches
    
    def get_document(self, doc_id):
        """
        Get a document by ID
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            dict: Document data or None if not found
        """
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id):
        """
        Delete a document from the vector store
        
        Args:
            doc_id (str): Document ID
        """
        if doc_id in self.documents:
            if self.using_chromadb:
                try:
                    self.collection.delete(ids=[doc_id])
                except Exception as e:
                    print(f"Error deleting document from ChromaDB: {e}")
            
            del self.documents[doc_id]
            return True
        return False
    
    def get_all_documents(self):
        """
        Get all documents in the vector store
        
        Returns:
            list: List of document data
        """
        return list(self.documents.values())