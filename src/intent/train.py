"""
Intent Classification Training Module
Trains both baseline (TF-IDF + LogReg) and improved (Sentence Transformer) classifiers.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
import yaml
from typing import Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder


class IntentClassifier:
    """Intent classification with baseline and improved models."""
    
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'intent_classifier')
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.model_type = None
        
        # Sentence Transformer model (loaded lazily)
        self._st_model = None
        self._st_classifier = None
    
    def train_baseline(self, X: pd.Series, y: pd.Series, 
                       test_size: float = 0.2,
                       random_state: int = 42) -> dict:
        """
        Train baseline: TF-IDF + Logistic Regression.
        
        Args:
            X: Customer messages
            y: Intent labels
            test_size: Fraction for test split
            random_state: Random seed
        
        Returns:
            dict with metrics
        """
        print("Training BASELINE: TF-IDF + Logistic Regression")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Conversation-level split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state,
            stratify=y_encoded
        )
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True,
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Logistic Regression
        self.classifier = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=random_state,
            solver='lbfgs',
            multi_class='multinomial',
        )
        self.classifier.fit(X_train_tfidf, y_train)
        self.model_type = 'tfidf_logreg'
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_tfidf)
        y_proba = self.classifier.predict_proba(X_test_tfidf)
        
        metrics = self._compute_metrics(y_test, y_pred, y_proba, "baseline")
        
        # Save model
        self._save_model('baseline')
        
        return metrics
    
    def train_improved(self, X: pd.Series, y: pd.Series,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       embedding_model: str = 'all-MiniLM-L6-v2') -> dict:
        """
        Train improved: Sentence Transformer embeddings + Logistic Regression.
        """
        print("Training IMPROVED: Sentence Transformer + Logistic Regression")
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("WARNING: sentence-transformers not installed.")
            print("Install: pip install sentence-transformers")
            print("Falling back to baseline.")
            return self.train_baseline(X, y, test_size, random_state)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded, test_size=test_size, random_state=random_state,
            stratify=y_encoded
        )
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Compute embeddings
        print(f"  Loading embedding model: {embedding_model}")
        self._st_model = SentenceTransformer(embedding_model)
        
        print("  Computing train embeddings...")
        X_train_emb = self._st_model.encode(X_train.tolist(), show_progress_bar=True, batch_size=64)
        print("  Computing test embeddings...")
        X_test_emb = self._st_model.encode(X_test.tolist(), show_progress_bar=True, batch_size=64)
        
        # Logistic Regression on embeddings
        self._st_classifier = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=random_state,
            solver='lbfgs',
            multi_class='multinomial',
        )
        self._st_classifier.fit(X_train_emb, y_train)
        self.model_type = 'sentence_transformer'
        
        # Evaluate
        y_pred = self._st_classifier.predict(X_test_emb)
        y_proba = self._st_classifier.predict_proba(X_test_emb)
        
        metrics = self._compute_metrics(y_test, y_pred, y_proba, "improved")
        
        # Save
        self._save_model('improved')
        
        return metrics
    
    def predict(self, messages: list, return_proba: bool = True, conversation_context: Optional[str] = None) -> list:
        """
        Predict intents for a list of messages, optionally incorporating conversation context.
        
        Returns:
            list of dicts with keys: intent, confidence, all_probabilities
        """
        processed_messages = []
        for msg in messages:
            if conversation_context and conversation_context.strip():
                # Prepend recent context for classification
                ctx = conversation_context.strip()[-200:]
                processed_messages.append(f"{ctx} {msg}")
            else:
                processed_messages.append(msg)

        if self.model_type == 'sentence_transformer' and self._st_model is not None:
            embeddings = self._st_model.encode(processed_messages, batch_size=32)
            predictions = self._st_classifier.predict(embeddings)
            probabilities = self._st_classifier.predict_proba(embeddings)
        elif self.vectorizer is not None and self.classifier is not None:
            tfidf = self.vectorizer.transform(processed_messages)
            predictions = self.classifier.predict(tfidf)
            probabilities = self.classifier.predict_proba(tfidf)
        else:
            raise RuntimeError("No model loaded. Train or load a model first.")
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            intent_name = self.label_encoder.inverse_transform([pred])[0]
            confidence = float(proba.max())
            
            # Post-processing heuristics for edge-case messages with context
            msg_orig = messages[i].lower().strip()
            if conversation_context:
                ctx_lower = conversation_context.lower()
                # Case: user replying with account info following an account inquiry
                if ('account' in ctx_lower or 'login' in ctx_lower or 'password' in ctx_lower or '@' in ctx_lower) and ('@' in msg_orig or 'account' in msg_orig or 'detail' in msg_orig):
                    if intent_name == 'other' or confidence < 0.6:
                        acc_idx = list(self.label_encoder.classes_).index('account_access') if 'account_access' in self.label_encoder.classes_ else None
                        if acc_idx is not None:
                            intent_name = 'account_access'
                            confidence = max(confidence, 0.75)
            
            result = {
                'intent': intent_name,
                'confidence': confidence,
            }
            if return_proba:
                result['all_probabilities'] = {
                    self.label_encoder.inverse_transform([j])[0]: float(p)
                    for j, p in enumerate(proba)
                }
            results.append(result)
        
        return results
    
    def _compute_metrics(self, y_true, y_pred, y_proba, model_name: str) -> dict:
        """Compute comprehensive evaluation metrics."""
        labels = self.label_encoder.classes_
        
        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, target_names=labels, 
                                       output_dict=True, zero_division=0)
        metrics['per_class'] = {}
        for label in labels:
            if label in report:
                metrics['per_class'][label] = {
                    'precision': report[label]['precision'],
                    'recall': report[label]['recall'],
                    'f1': report[label]['f1-score'],
                    'support': report[label]['support'],
                }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['class_names'] = labels.tolist()
        
        # Print summary
        print(f"\n  === {model_name.upper()} METRICS ===")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
        
        # Save metrics
        metrics_path = os.path.join(self.model_dir, f'{model_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Metrics saved to: {metrics_path}")
        
        return metrics
    
    def _save_model(self, name: str):
        """Save trained model artifacts."""
        save_dir = os.path.join(self.model_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        if self.vectorizer:
            with open(os.path.join(save_dir, 'vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        if self.classifier:
            with open(os.path.join(save_dir, 'classifier.pkl'), 'wb') as f:
                pickle.dump(self.classifier, f)
        
        if self.label_encoder:
            with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        if self._st_classifier:
            with open(os.path.join(save_dir, 'st_classifier.pkl'), 'wb') as f:
                pickle.dump(self._st_classifier, f)
        
        # Save model metadata
        meta = {
            'model_type': self.model_type,
            'name': name,
            'n_classes': len(self.label_encoder.classes_) if self.label_encoder else 0,
            'classes': self.label_encoder.classes_.tolist() if self.label_encoder else [],
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"  Model saved to: {save_dir}")
    
    def load_model(self, name: str = 'baseline'):
        """Load a previously saved model."""
        save_dir = os.path.join(self.model_dir, name)
        
        with open(os.path.join(save_dir, 'metadata.json')) as f:
            meta = json.load(f)
        
        self.model_type = meta['model_type']
        
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        if self.model_type == 'tfidf_logreg':
            with open(os.path.join(save_dir, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(save_dir, 'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
        elif self.model_type == 'sentence_transformer':
            with open(os.path.join(save_dir, 'st_classifier.pkl'), 'rb') as f:
                self._st_classifier = pickle.load(f)
            # Load sentence transformer model
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("WARNING: sentence-transformers not installed for improved model")
        
        print(f"Model loaded: {name} ({self.model_type})")


class IntentPredictor:
    """Convenience wrapper for inference."""
    
    def __init__(self, model_dir: str = None, model_name: str = 'baseline'):
        self.classifier = IntentClassifier(model_dir)
        self.classifier.load_model(model_name)
    
    def predict(self, message: str) -> dict:
        """Predict intent for a single message."""
        results = self.classifier.predict([message], return_proba=True)
        return results[0]
    
    def predict_batch(self, messages: list) -> list:
        """Predict intents for multiple messages."""
        return self.classifier.predict(messages, return_proba=True)
