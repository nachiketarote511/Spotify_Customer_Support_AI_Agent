"""
Retrieval Module
Retrieves similar historical conversations for grounding RAG responses.
Includes similarity thresholding, context-awareness, and lightweight candidate reranking.
"""

import numpy as np
import pandas as pd
import os
import re
from typing import List, Optional

from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.index import FAISSIndex


class HistoricalRetriever:
    """Retrieves similar historical support conversations."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2',
                 index_dir: str = None):
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.index = FAISSIndex()
        
        if index_dir is None:
            index_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'models', 'embedding_index'
            )
        self.index_dir = index_dir
    
    def build_from_conversations(self, conversations_df: pd.DataFrame,
                                 text_column: str = 'customer_message'):
        """
        Build retrieval index from conversation data.
        
        Args:
            conversations_df: DataFrame with conversation records
            text_column: Column to embed for retrieval
        """
        print(f"Building retrieval index from {len(conversations_df):,} conversations...")
        
        # Prepare texts and documents
        texts = conversations_df[text_column].fillna('').tolist()
        documents = []
        for _, row in conversations_df.iterrows():
            doc = {
                'conversation_id': row.get('conversation_id', ''),
                'customer_message': row.get('customer_message', ''),
                'historical_support_response': row.get('historical_support_response', ''),
                'conversation_context': row.get('conversation_context', ''),
                'brand': row.get('brand', ''),
            }
            documents.append(doc)
        
        # Generate embeddings
        print("  Generating embeddings...")
        embeddings = self.embedding_generator.encode(texts)
        
        # Build FAISS index
        index_type = 'flat' if len(texts) < 100000 else 'ivf'
        self.index.build_index(embeddings, documents, index_type)
        
        # Save
        self.index.save(self.index_dir)
        self.embedding_generator.save_embeddings(
            embeddings,
            {'n_documents': len(texts), 'text_column': text_column},
            self.index_dir
        )
        
        print(f"  Index built and saved: {len(texts):,} documents")
    
    def retrieve(self, query: str, top_k: int = 5,
                 min_similarity: float = 0.35,
                 conversation_context: str = "",
                 predicted_intent: str = None) -> List[dict]:
        """
        Retrieve similar historical conversations with similarity filtering and candidate reranking.
        
        Args:
            query: Customer message to find similar cases for
            top_k: Number of final results
            min_similarity: Minimum cosine similarity score threshold
            conversation_context: Optional prior conversation history
            predicted_intent: Optional predicted intent to aid reranking
        
        Returns:
            List of dicts with keys: document, score, rank
        """
        # Ensure model is loaded
        self.embedding_generator.load_model()
        
        # Load index if not loaded
        if self.index.index is None:
            self.index.load(self.index_dir)
        
        # Construct search query text incorporating context if present
        if conversation_context and conversation_context.strip():
            context_clean = conversation_context.strip()
            # Take recent context
            search_text = f"{context_clean[-200:]} {query}"
        else:
            search_text = query
        
        # Encode query
        query_embedding = self.embedding_generator.encode([search_text], show_progress=False)
        
        # Retrieve candidate pool (3x top_k)
        candidate_k = max(top_k * 3, 15)
        raw_results = self.index.search(query_embedding[0], top_k=candidate_k)
        
        # Candidate filtering & reranking
        reranked = []
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        for item in raw_results:
            base_score = item.get('score', 0.0)
            doc = item.get('document', {})
            cust_msg = doc.get('customer_message', '').lower()
            supp_resp = doc.get('historical_support_response', '').lower()
            
            # Apply minimum similarity threshold
            if base_score < min_similarity:
                continue
            
            # Reranking boost based on intent & lexical terms
            boost = 0.0
            
            # Term overlap boost
            cust_terms = set(re.findall(r'\w+', cust_msg))
            overlap = len(query_terms & cust_terms)
            if len(query_terms) > 0:
                boost += 0.05 * (overlap / len(query_terms))
            
            # Intent keyword alignment boost
            if predicted_intent == 'playback_issues' and any(kw in cust_msg or kw in supp_resp for kw in ['play', 'crash', 'song', 'music', 'audio', 'app']):
                boost += 0.05
            elif predicted_intent == 'account_access' and any(kw in cust_msg or kw in supp_resp for kw in ['login', 'password', 'facebook', 'account', 'sign in']):
                boost += 0.05
            elif predicted_intent == 'billing_payment' and any(kw in cust_msg or kw in supp_resp for kw in ['charge', 'pay', 'billing', 'subscription', 'refund', 'family']):
                boost += 0.05
            
            final_score = base_score + boost
            reranked.append({
                'document': doc,
                'score': float(final_score),
                'base_score': float(base_score),
                'rank': 0
            })
        
        # Sort by final score descending
        reranked.sort(key=lambda x: x['score'], reverse=True)
        
        # Assign final ranks and slice top_k
        final_results = reranked[:top_k]
        for rank, res in enumerate(final_results, 1):
            res['rank'] = rank
            
        return final_results
    
    def evaluate_retrieval(self, queries: List[str], 
                           relevant_docs: List[set],
                           k_values: List[int] = [1, 3, 5]) -> dict:
        """
        Evaluate retrieval quality using Recall@K and Precision@K.
        """
        metrics = {}
        
        for k in k_values:
            recalls = []
            precisions = []
            
            for query, relevant in zip(queries, relevant_docs):
                results = self.retrieve(query, top_k=k, min_similarity=0.0)
                retrieved_ids = {r['document']['conversation_id'] for r in results}
                
                if len(relevant) > 0:
                    recall = len(retrieved_ids & relevant) / len(relevant)
                    recalls.append(recall)
                
                precision = len(retrieved_ids & relevant) / max(len(retrieved_ids), 1)
                precisions.append(precision)
            
            metrics[f'recall@{k}'] = float(np.mean(recalls)) if recalls else 0.0
            metrics[f'precision@{k}'] = float(np.mean(precisions))
        
        return metrics
