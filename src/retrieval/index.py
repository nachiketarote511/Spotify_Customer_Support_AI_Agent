"""
FAISS Index Module
Creates and manages FAISS vector indices for historical conversation retrieval.
"""

import numpy as np
import os
import json
import pickle
from typing import List, Tuple, Optional


class FAISSIndex:
    """FAISS-based vector index for similarity search."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []  # Store document metadata alongside index
        self._faiss = None
    
    def _import_faiss(self):
        """Lazy import of faiss."""
        if self._faiss is None:
            try:
                import faiss
                self._faiss = faiss
            except ImportError:
                raise ImportError(
                    "FAISS required. Install: pip install faiss-cpu"
                )
    
    def build_index(self, embeddings: np.ndarray, documents: List[dict],
                    index_type: str = 'flat'):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of shape (n_docs, embedding_dim)
            documents: list of document metadata dicts
            index_type: 'flat' for exact search, 'ivf' for approximate
        """
        self._import_faiss()
        
        self.embedding_dim = embeddings.shape[1]
        n_docs = embeddings.shape[0]
        
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        if index_type == 'flat':
            # Exact search (good for < 100K documents)
            self.index = self._faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine for normalized vectors)
        elif index_type == 'ivf':
            # Approximate search (good for > 100K documents)
            n_clusters = min(int(np.sqrt(n_docs)), 100)
            quantizer = self._faiss.IndexFlatIP(self.embedding_dim)
            self.index = self._faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            self.index.train(embeddings)
            self.index.nprobe = min(10, n_clusters)
        
        self.index.add(embeddings)
        self.documents = documents
        
        print(f"FAISS index built: {n_docs:,} documents, dim={self.embedding_dim}, type={index_type}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: numpy array of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of results to return
        
        Returns:
            List of dicts with keys: document, score, rank
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            result = {
                'document': self.documents[idx],
                'score': float(score),
                'rank': rank + 1,
                'index': int(idx),
            }
            results.append(result)
        
        return results
    
    def save(self, output_dir: str):
        """Save index and documents to disk."""
        self._import_faiss()
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, 'faiss_index.bin')
        self._faiss.write_index(self.index, index_path)
        
        # Save documents
        docs_path = os.path.join(output_dir, 'documents.pkl')
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        meta = {
            'embedding_dim': self.embedding_dim,
            'n_documents': len(self.documents),
            'index_type': type(self.index).__name__,
        }
        with open(os.path.join(output_dir, 'index_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Index saved to: {output_dir}")
    
    def load(self, input_dir: str):
        """Load index and documents from disk."""
        self._import_faiss()
        
        index_path = os.path.join(input_dir, 'faiss_index.bin')
        self.index = self._faiss.read_index(index_path)
        
        docs_path = os.path.join(input_dir, 'documents.pkl')
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        with open(os.path.join(input_dir, 'index_metadata.json')) as f:
            meta = json.load(f)
        
        self.embedding_dim = meta['embedding_dim']
        
        print(f"Index loaded: {len(self.documents):,} documents, dim={self.embedding_dim}")
