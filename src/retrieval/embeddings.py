"""
Embeddings Module
Generates sentence embeddings for customer messages and support responses.
"""

import numpy as np
import os
import pickle
import json
from typing import List, Optional


class EmbeddingGenerator:
    """Generate embeddings using Sentence Transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
    
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension
            test_emb = self.model.encode(["test"])
            self.embedding_dim = test_emb.shape[1]
            print(f"  Embedding dimension: {self.embedding_dim}")
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )
    
    def encode(self, texts: List[str], batch_size: int = 64, 
               show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        self.load_model()
        
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: dict, 
                        output_dir: str):
        """Save embeddings and metadata to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Embeddings saved to: {output_dir}")
        print(f"  Shape: {embeddings.shape}")
    
    def load_embeddings(self, input_dir: str):
        """Load saved embeddings."""
        embeddings = np.load(os.path.join(input_dir, 'embeddings.npy'))
        
        with open(os.path.join(input_dir, 'metadata.json')) as f:
            metadata = json.load(f)
        
        return embeddings, metadata
