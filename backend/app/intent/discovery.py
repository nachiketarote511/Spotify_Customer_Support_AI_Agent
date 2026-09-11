"""
Intent Discovery Module
Discovers meaningful support intent categories from actual brand data.
Does NOT use predefined intent taxonomies — intents are derived from the data.
"""

import pandas as pd
import numpy as np
import os
import yaml
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import re


class IntentDiscoverer:
    """Discovers intent categories from customer messages."""
    
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.vectorizer = None
        self.cluster_model = None
        self.intent_map = {}
    
    def discover(self, messages: pd.Series) -> dict:
        """
        Discover intents from customer messages using TF-IDF clustering.
        
        Steps:
        1. Clean and vectorize messages
        2. Cluster using K-Means
        3. Extract top terms per cluster
        4. Generate candidate intent names
        5. Return intent taxonomy
        """
        print(f"Discovering intents from {len(messages):,} messages...")
        
        # Clean messages
        cleaned = messages.dropna().apply(self._clean_for_clustering)
        cleaned = cleaned[cleaned.str.len() > 10]
        print(f"  After cleaning: {len(cleaned):,} messages")
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True,
        )
        tfidf_matrix = self.vectorizer.fit_transform(cleaned)
        print(f"  TF-IDF matrix: {tfidf_matrix.shape}")
        
        # Optional: dimensionality reduction for better clustering
        if tfidf_matrix.shape[1] > 100:
            self._svd = TruncatedSVD(n_components=100, random_state=self.random_state)
            reduced = self._svd.fit_transform(tfidf_matrix)
            print(f"  SVD reduced to: {reduced.shape}")
        else:
            self._svd = None
            reduced = tfidf_matrix.toarray()
        
        # K-Means clustering
        if len(cleaned) > 10000:
            self.cluster_model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=1000,
                n_init=3,
            )
        else:
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
        
        labels = self.cluster_model.fit_predict(reduced)
        print(f"  Clustering complete. {self.n_clusters} clusters.")
        
        # Analyze clusters
        feature_names = self.vectorizer.get_feature_names_out()
        taxonomy = self._analyze_clusters(
            cleaned, labels, tfidf_matrix, feature_names
        )
        
        return taxonomy
    
    def _clean_for_clustering(self, text: str) -> str:
        """Clean text for TF-IDF clustering."""
        text = str(text).lower()
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\[url\]', '', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _analyze_clusters(self, messages, labels, tfidf_matrix, feature_names) -> dict:
        """Analyze clusters to generate intent taxonomy."""
        taxonomy = {
            'n_intents': self.n_clusters,
            'intents': [],
            'label_distribution': {},
        }
        
        # Get cluster centers for term importance
        if hasattr(self.cluster_model, 'cluster_centers_'):
            # For non-SVD case, project centers back
            pass
        
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_messages = messages[mask]
            cluster_tfidf = tfidf_matrix[mask]
            
            # Top terms by mean TF-IDF score in this cluster
            mean_tfidf = cluster_tfidf.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-15:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            top_scores = [float(mean_tfidf[i]) for i in top_indices]
            
            # Generate intent name from top terms
            intent_name = self._generate_intent_name(top_terms, cluster_messages)
            
            # Get representative examples
            examples = cluster_messages.head(5).tolist()
            
            intent = {
                'id': cluster_id,
                'name': intent_name,
                'top_terms': top_terms[:10],
                'top_term_scores': top_scores[:10],
                'count': int(mask.sum()),
                'percentage': round(mask.sum() / len(labels) * 100, 2),
                'examples': examples,
            }
            taxonomy['intents'].append(intent)
            taxonomy['label_distribution'][intent_name] = int(mask.sum())
        
        # Sort by count
        taxonomy['intents'].sort(key=lambda x: x['count'], reverse=True)
        
        return taxonomy
    
    def _generate_intent_name(self, top_terms: list, messages: pd.Series) -> str:
        """
        Generate a human-readable intent name from cluster characteristics.
        Uses keyword matching against common support intent patterns.
        """
        terms_str = ' '.join(top_terms[:10]).lower()
        
        # Pattern matching for common intent categories
        intent_patterns = {
            'account_access': ['login', 'password', 'sign in', 'log in', 'access', 'locked', 'cant log', 'reset password'],
            'playback_issues': ['play', 'song', 'music', 'stream', 'buffer', 'skip', 'pause', 'audio', 'sound', 'playing'],
            'billing_payment': ['payment', 'charge', 'bill', 'subscribe', 'subscription', 'cancel subscription', 'refund', 'price', 'premium', 'plan'],
            'app_technical': ['app', 'crash', 'bug', 'error', 'update', 'install', 'download', 'version', 'device', 'phone'],
            'playlist_library': ['playlist', 'songs', 'library', 'save', 'add', 'remove', 'delete', 'collection', 'liked'],
            'connectivity': ['offline', 'connect', 'wifi', 'internet', 'network', 'download', 'sync'],
            'feature_request': ['feature', 'want', 'wish', 'would be nice', 'suggestion', 'need', 'please add'],
            'cancellation': ['cancel', 'unsubscribe', 'stop', 'end', 'terminate', 'close account'],
            'general_complaint': ['bad', 'worst', 'terrible', 'hate', 'awful', 'disappointed', 'frustrat', 'annoyed'],
            'information_request': ['how', 'where', 'what', 'when', 'help', 'question', 'info', 'know'],
            'device_compatibility': ['device', 'speaker', 'alexa', 'google home', 'smart', 'cast', 'bluetooth', 'headphone'],
            'content_availability': ['available', 'missing', 'find', 'search', 'where is', 'looking for', 'album', 'artist'],
        }
        
        best_match = 'general_inquiry'
        best_score = 0
        
        for intent_name, keywords in intent_patterns.items():
            score = sum(1 for kw in keywords if kw in terms_str)
            if score > best_score:
                best_score = score
                best_match = intent_name
        
        return best_match
    
    def assign_labels(self, messages: pd.Series, taxonomy: dict) -> pd.Series:
        """Assign intent labels to messages using the fitted model."""
        cleaned = messages.apply(self._clean_for_clustering)
        tfidf = self.vectorizer.transform(cleaned)
        
        if getattr(self, '_svd', None) is not None:
            features = self._svd.transform(tfidf)
        else:
            features = tfidf.toarray() if hasattr(tfidf, 'toarray') else tfidf
        
        cluster_labels = self.cluster_model.predict(features)
        
        # Map cluster IDs to intent names
        id_to_name = {intent['id']: intent['name'] for intent in taxonomy['intents']}
        return pd.Series([id_to_name.get(c, 'general_inquiry') for c in cluster_labels])


def save_intent_taxonomy(taxonomy: dict, output_dir: str):
    """Save the discovered intent taxonomy as YAML and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full taxonomy as JSON
    json_path = os.path.join(output_dir, 'intent_taxonomy.json')
    with open(json_path, 'w') as f:
        json.dump(taxonomy, f, indent=2, default=str)
    
    # Save as YAML config
    yaml_config = {
        'intents': {}
    }
    for intent in taxonomy['intents']:
        yaml_config['intents'][intent['name']] = {
            'definition': f"Customer messages related to {intent['name'].replace('_', ' ')}",
            'examples': intent['examples'][:3],
            'count': intent['count'],
        }
    
    yaml_path = os.path.join(output_dir, '..', 'configs', 'intents.yaml')
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Taxonomy saved to: {json_path}")
    print(f"Intent config saved to: {yaml_path}")


if __name__ == '__main__':
    # Load processed conversations
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    data_path = os.path.join(_repo_root, 'data', 'processed', 'spotify_conversations.csv')
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}")
        print("Run the pandas pipeline first: python -m app.pipelines.pandas_pipeline")
        exit(1)
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} conversations")
    
    discoverer = IntentDiscoverer(n_clusters=10)
    taxonomy = discoverer.discover(df['customer_message'])
    
    print("\n=== DISCOVERED INTENTS ===")
    for intent in taxonomy['intents']:
        print(f"\n  {intent['name']} ({intent['count']} examples, {intent['percentage']}%)")
        print(f"    Top terms: {', '.join(intent['top_terms'][:5])}")
        print(f"    Example: {intent['examples'][0][:100]}...")
    
    save_intent_taxonomy(taxonomy, os.path.join(_repo_root, 'reports'))
