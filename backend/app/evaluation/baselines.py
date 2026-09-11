"""
Baselines Module
Implements two baseline approaches for comparison.
Baseline 1: Trivial (generic response)
Baseline 2: Simple (TF-IDF retrieval + nearest historical response)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


class TrivialBaseline:
    """
    Baseline 1: Always returns a generic support response.
    This is the floor against which any system should improve.
    """
    
    GENERIC_RESPONSES = {
        'default': "Thank you for reaching out! We're sorry to hear you're experiencing issues. "
                   "Please send us a DM with your account details so we can look into this for you.",
        'account': "We understand your concern about your account. Please DM us with your "
                   "username so we can investigate further.",
        'technical': "We're sorry for the trouble! Please try restarting the app and "
                     "clearing the cache. If the issue persists, DM us with details.",
    }
    
    def predict(self, message: str, intent: str = None) -> dict:
        """Generate a trivial response."""
        response = self.GENERIC_RESPONSES.get(intent, self.GENERIC_RESPONSES['default'])
        return {
            'reply': response,
            'confidence': 0.1,
            'grounding_summary': 'Trivial baseline - generic response',
            'should_escalate': False,
            'escalation_reason': None,
            'baseline': 'trivial',
        }
    
    def predict_batch(self, messages: List[str], intents: List[str] = None) -> List[dict]:
        """Generate trivial responses for a batch."""
        if intents is None:
            intents = [None] * len(messages)
        return [self.predict(msg, intent) for msg, intent in zip(messages, intents)]


class SimpleBaseline:
    """
    Baseline 2: TF-IDF retrieval + nearest historical response.
    Retrieves the most similar historical customer message and returns
    its corresponding support response verbatim.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.responses = None
        self.messages = None
    
    def fit(self, conversations_df: pd.DataFrame):
        """
        Fit the baseline on historical conversations.
        
        Args:
            conversations_df: DataFrame with customer_message and 
                            historical_support_response columns
        """
        self.messages = conversations_df['customer_message'].fillna('').tolist()
        self.responses = conversations_df['historical_support_response'].fillna('').tolist()
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words='english',
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.messages)
        print(f"Simple baseline fitted on {len(self.messages):,} conversations")
    
    def predict(self, message: str, top_k: int = 1) -> dict:
        """
        Find the most similar historical case and return its response.
        """
        if self.vectorizer is None:
            raise RuntimeError("Baseline not fitted. Call fit() first.")
        
        query_vec = self.vectorizer.transform([message])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_idx = similarities.argsort()[-top_k:][::-1]
        best_idx = top_idx[0]
        best_score = float(similarities[best_idx])
        
        return {
            'reply': self.responses[best_idx],
            'confidence': best_score,
            'grounding_summary': f'TF-IDF nearest match (similarity: {best_score:.3f})',
            'should_escalate': best_score < 0.1,
            'escalation_reason': 'Low similarity match' if best_score < 0.1 else None,
            'baseline': 'simple_tfidf',
            'similar_message': self.messages[best_idx],
            'similarity_score': best_score,
        }
    
    def predict_batch(self, messages: List[str]) -> List[dict]:
        """Predict for a batch of messages."""
        return [self.predict(msg) for msg in messages]


def compare_baselines(golden_set: pd.DataFrame, 
                      trivial: TrivialBaseline,
                      simple: SimpleBaseline,
                      system_results: List[dict] = None) -> dict:
    """
    Compare all approaches on the same evaluation set.
    
    Args:
        golden_set: DataFrame with customer_message and ground truth
        trivial: Fitted trivial baseline
        simple: Fitted simple baseline
        system_results: Results from the full system
    
    Returns:
        Comparison metrics dict
    """
    messages = golden_set['customer_message'].tolist()
    
    comparison = {
        'trivial': {
            'responses': trivial.predict_batch(messages),
        },
        'simple': {
            'responses': simple.predict_batch(messages),
        },
    }
    
    if system_results:
        comparison['system'] = {
            'responses': system_results,
        }
    
    # Compute basic statistics for each approach
    for approach, data in comparison.items():
        responses = data['responses']
        data['stats'] = {
            'mean_confidence': float(np.mean([r.get('confidence', 0) for r in responses])),
            'escalation_rate': float(np.mean([r.get('should_escalate', False) for r in responses])),
            'mean_reply_length': float(np.mean([len(r.get('reply', '')) for r in responses])),
        }
    
    return comparison
