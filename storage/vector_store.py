"""
ChromaDB Vector Store with Default Embeddings
Handles document storage and retrieval for RAG
Uses ChromaDB's built-in embedding function (no heavy dependencies)
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import sys
sys.path.append('..')
from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


class VectorStore:
    """
    ChromaDB-based vector store with default embeddings
    """
    
    def __init__(self):
        self.persist_dir = CHROMA_PERSIST_DIR
        self.collection_name = CHROMA_COLLECTION_NAME
        
        # Initialize ChromaDB client
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use default embedding function (lightweight, no external dependencies)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Vector store initialized. Collection: {self.collection_name}")
        
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            
            # Try to break at a sentence boundary
            if end < text_length:
                # Look for sentence endings
                for sep in ['. ', '! ', '? ', '\n']:
                    last_sep = chunk.rfind(sep)
                    if last_sep > CHUNK_SIZE // 2:
                        chunk = chunk[:last_sep + 1]
                        end = start + last_sep + 1
                        break
                        
            if chunk.strip():
                chunks.append(chunk.strip())
                
            start = end - CHUNK_OVERLAP
            
        return chunks
    
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict]] = None,
        chunk: bool = True
    ) -> int:
        """
        Add documents to the vector store
        
        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document
            chunk: Whether to chunk the texts
            
        Returns:
            Number of chunks added
        """
        all_chunks = []
        all_metadatas = []
        
        for i, text in enumerate(texts):
            if chunk:
                chunks = self._chunk_text(text)
            else:
                chunks = [text] if text.strip() else []
                
            for j, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                
                # Create metadata
                meta = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
                meta["chunk_index"] = j
                meta["source_index"] = i
                all_metadatas.append(meta)
                
        if not all_chunks:
            print("No content to add to vector store")
            return 0
            
        # Generate unique IDs
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(all_chunks))]
        
        # Add to collection (embeddings generated automatically)
        print(f"Adding {len(all_chunks)} chunks to vector store...")
        self.collection.add(
            ids=ids,
            documents=all_chunks,
            metadatas=all_metadatas
        )
        
        print(f"Added {len(all_chunks)} chunks to vector store")
        return len(all_chunks)
    
    def search(
        self, 
        query: str, 
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        # Search (embedding generated automatically)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })
                
        return formatted
    
    def get_relevant_context(self, query: str, n_results: int = 5) -> str:
        """
        Get relevant context as a formatted string for LLM
        
        Args:
            query: Search query
            n_results: Number of results to include
            
        Returns:
            Formatted context string
        """
        results = self.search(query, n_results)
        
        if not results:
            return "No relevant context found."
            
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Context {i}]:\n{result['content']}")
            
        return "\n\n".join(context_parts)
    
    def clear(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared")
        
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.persist_dir
        }


def main():
    """Test the vector store"""
    vs = VectorStore()
    
    # Test with sample content
    sample_texts = [
        "NihongoNext offers Japanese language courses for beginners and advanced learners.",
        "JLPT preparation courses are available for N5 through N1 levels.",
        "Online classes are conducted via Zoom with native Japanese instructors.",
    ]
    
    vs.add_documents(sample_texts, chunk=False)
    
    # Test search
    results = vs.search("JLPT exam preparation")
    print("\nSearch results for 'JLPT exam preparation':")
    for r in results:
        print(f"  - {r['content'][:100]}... (distance: {r['distance']:.4f})")
        
    print(f"\nStats: {vs.get_stats()}")


if __name__ == "__main__":
    main()
