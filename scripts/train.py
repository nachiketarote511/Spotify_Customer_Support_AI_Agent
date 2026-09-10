"""
Main Training Script
Runs Phases 7-10: Intent Discovery → Classifier Training → Embeddings → FAISS Index
"""

import os
import sys
import json
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.intent.discovery import IntentDiscoverer, save_intent_taxonomy
from src.intent.train import IntentClassifier
from src.retrieval.retrieve import HistoricalRetriever
from src.evaluation.baselines import SimpleBaseline
from scripts.build_golden_set import build_golden_set


def main():
    # Load config
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    brand = config['brand']
    
    # Load processed conversations
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'spotify_conversations.csv')
    if not os.path.exists(data_path):
        print("ERROR: Processed data not found. Run the pandas pipeline first.")
        print("  python -m src.pipelines.pandas_pipeline")
        sys.exit(1)
    
    print("Loading processed conversations...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} conversations")
    
    # ============================================================
    # PHASE 7: Intent Discovery
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 7: INTENT DISCOVERY")
    print("=" * 70)
    
    discoverer = IntentDiscoverer(n_clusters=10, random_state=42)
    taxonomy = discoverer.discover(df['customer_message'])
    
    # Assign intent labels to all conversations
    df['intent'] = discoverer.assign_labels(df['customer_message'], taxonomy)
    
    # Save taxonomy
    save_intent_taxonomy(taxonomy, os.path.join(PROJECT_ROOT, 'reports'))
    
    # Print intent distribution
    print("\nIntent Distribution:")
    intent_dist = df['intent'].value_counts()
    for intent, count in intent_dist.items():
        pct = count / len(df) * 100
        print(f"  {intent:<30} {count:>6,} ({pct:.1f}%)")
    
    # Save labeled data
    labeled_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'spotify_labeled.csv')
    df.to_csv(labeled_path, index=False)
    print(f"\nLabeled data saved: {labeled_path}")
    
    # ============================================================
    # PHASE 8: Golden Evaluation Set
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 8: GOLDEN EVALUATION SET")
    print("=" * 70)
    
    golden_path = os.path.join(PROJECT_ROOT, 'data', 'golden', 'golden_eval_set.csv')
    golden_df = build_golden_set(
        df, 
        intent_labels=df['intent'],
        target_size=200,
        output_path=golden_path,
        random_state=42
    )
    print(f"Golden set: {len(golden_df)} examples")
    print(f"Intent distribution in golden set:")
    print(golden_df['true_intent'].value_counts().to_string())
    
    # ============================================================
    # PHASE 9: Intent Classifier Training  
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 9: INTENT CLASSIFIER TRAINING")
    print("=" * 70)
    
    # Exclude golden set from training (conversation-level split)
    golden_conv_ids = set(golden_df['conversation_id'].values)
    train_df = df[~df['conversation_id'].isin(golden_conv_ids)]
    print(f"Training set: {len(train_df):,} (excluded {len(golden_conv_ids)} golden set conversations)")
    
    classifier = IntentClassifier()
    
    # Baseline: TF-IDF + Logistic Regression
    print("\n--- Training Baseline ---")
    baseline_metrics = classifier.train_baseline(
        train_df['customer_message'], 
        train_df['intent'],
        test_size=0.2,
        random_state=42
    )
    
    # Improved: Sentence Transformer (if available)
    print("\n--- Training Improved ---")
    try:
        improved_classifier = IntentClassifier()
        improved_metrics = improved_classifier.train_improved(
            train_df['customer_message'],
            train_df['intent'],
            test_size=0.2,
            random_state=42,
        )
    except Exception as e:
        print(f"Improved model training failed: {e}")
        print("Continuing with baseline only.")
        improved_metrics = None
    
    # ============================================================
    # PHASE 10: Embeddings + FAISS Index
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 10: EMBEDDINGS + FAISS INDEX")
    print("=" * 70)
    
    try:
        retriever = HistoricalRetriever(
            embedding_model='all-MiniLM-L6-v2',
            index_dir=os.path.join(PROJECT_ROOT, 'models', 'embedding_index')
        )
        retriever.build_from_conversations(train_df)
        print("FAISS index built successfully.")
    except ImportError as e:
        print(f"Embedding/FAISS build failed: {e}")
        print("Install required packages: pip install sentence-transformers faiss-cpu")
    
    # ============================================================
    # PHASE 14: Fit Baselines
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 14: BASELINES")
    print("=" * 70)
    
    simple = SimpleBaseline()
    simple.fit(train_df)
    
    # Test baseline on a sample
    test_msg = "my spotify keeps crashing when i try to play music"
    trivial_result = {"reply": "Thank you for reaching out! Please DM us with your details."}
    simple_result = simple.predict(test_msg)
    print(f"\nTest message: {test_msg}")
    print(f"Trivial baseline: {trivial_result['reply'][:80]}...")
    print(f"Simple baseline: {simple_result['reply'][:80]}...")
    print(f"Simple similarity: {simple_result['similarity_score']:.3f}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Conversations: {len(df):,}")
    print(f"  Intents discovered: {len(taxonomy['intents'])}")
    print(f"  Golden set size: {len(golden_df)}")
    print(f"  Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  Baseline macro F1: {baseline_metrics['macro_f1']:.4f}")
    if improved_metrics:
        print(f"  Improved accuracy: {improved_metrics['accuracy']:.4f}")
        print(f"  Improved macro F1: {improved_metrics['macro_f1']:.4f}")
    print("=" * 70)
    
    # Save training summary
    summary = {
        'brand': brand,
        'total_conversations': len(df),
        'training_set_size': len(train_df),
        'golden_set_size': len(golden_df),
        'n_intents': len(taxonomy['intents']),
        'baseline_metrics': baseline_metrics,
        'improved_metrics': improved_metrics,
    }
    summary_path = os.path.join(PROJECT_ROOT, 'reports', 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nTraining summary saved: {summary_path}")


if __name__ == '__main__':
    main()
