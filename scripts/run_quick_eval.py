"""
Quick Evaluation Script
Reproduces headline results in under 15 minutes.
Loads prepared sample, runs lightweight classifier, retrieval, and evaluation.

NOTE: This script does NOT require GEMINI_API_KEY for the offline evaluation
(intent classification, baselines, retrieval, escalation). Gemini is only
used if you run the separate integration test:
  python scripts/test_gemini_integration.py
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load .env if present (makes GEMINI_API_KEY available to RAG/Judge if needed)
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


def main():
    start_time = time.time()
    
    print("=" * 70)
    print("  QUICK EVALUATION — Hiver AI Customer Support Agent")
    print("  Target: Complete in < 15 minutes")
    print("=" * 70)
    
    results = {}
    
    # ============================================================
    # Step 1: Load prepared data
    # ============================================================
    print("\n[1/7] Loading prepared data...")
    
    sample_path = os.path.join(PROJECT_ROOT, 'data', 'samples', 'spotify_sample.csv')
    golden_path = os.path.join(PROJECT_ROOT, 'data', 'golden', 'golden_eval_set.csv')
    labeled_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'spotify_labeled.csv')
    
    if not os.path.exists(sample_path):
        print("  Sample data not found. Running pandas pipeline...")
        from src.pipelines.pandas_pipeline import PandasPipeline
        pipeline = PandasPipeline()
        pipeline.run()
    
    sample_df = pd.read_csv(sample_path)
    print(f"  Sample: {len(sample_df):,} conversations")
    
    if os.path.exists(labeled_path):
        labeled_df = pd.read_csv(labeled_path)
        print(f"  Labeled data: {len(labeled_df):,} conversations")
    else:
        labeled_df = None
    
    has_golden = os.path.exists(golden_path)
    if has_golden:
        golden_df = pd.read_csv(golden_path)
        print(f"  Golden set: {len(golden_df)} examples")
    else:
        golden_df = None
    
    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f}s")
    
    # ============================================================
    # Step 2: Intent Discovery (if not already done)
    # ============================================================
    print("\n[2/7] Intent classification...")
    
    if labeled_df is not None and 'intent' in labeled_df.columns:
        print("  Using pre-computed intent labels")
        sample_df = sample_df.merge(
            labeled_df[['conversation_id', 'intent']],
            on='conversation_id', how='left'
        )
    else:
        from src.intent.discovery import IntentDiscoverer
        discoverer = IntentDiscoverer(n_clusters=10, random_state=42)
        taxonomy = discoverer.discover(sample_df['customer_message'])
        sample_df['intent'] = discoverer.assign_labels(
            sample_df['customer_message'], taxonomy
        )
    
    # ============================================================
    # Step 3: Train/Load classifier
    # ============================================================
    print("\n[3/7] Training/loading classifier...")
    
    from src.intent.train import IntentClassifier
    
    classifier = IntentClassifier()
    model_dir = os.path.join(PROJECT_ROOT, 'models', 'intent_classifier', 'baseline')
    
    if os.path.exists(model_dir):
        classifier.load_model('baseline')
        print("  Loaded pre-trained baseline classifier")
    else:
        print("  Training baseline classifier...")
        data = labeled_df if labeled_df is not None else sample_df
        if 'intent' not in data.columns:
            from src.intent.discovery import IntentDiscoverer
            discoverer = IntentDiscoverer(n_clusters=10, random_state=42)
            taxonomy = discoverer.discover(data['customer_message'])
            data['intent'] = discoverer.assign_labels(data['customer_message'], taxonomy)
        
        baseline_metrics = classifier.train_baseline(
            data['customer_message'], data['intent'],
            test_size=0.2, random_state=42
        )
        results['baseline_metrics'] = baseline_metrics
    
    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f}s")
    
    # ============================================================
    # Step 4: Evaluate intent classifier on golden set
    # ============================================================
    print("\n[4/7] Evaluating intent classifier...")
    
    if golden_df is not None and 'true_intent' in golden_df.columns:
        golden_with_intents = golden_df[golden_df['true_intent'].notna() & (golden_df['true_intent'] != '')]
        if len(golden_with_intents) > 0:
            predictions = classifier.predict(golden_with_intents['customer_message'].tolist())
            pred_intents = [p['intent'] for p in predictions]
            pred_confidences = [p['confidence'] for p in predictions]
            
            from src.evaluation.metrics import compute_classification_metrics
            intent_metrics = compute_classification_metrics(
                golden_with_intents['true_intent'].tolist(),
                pred_intents
            )
            results['intent_metrics'] = intent_metrics
            
            print(f"  Accuracy:    {intent_metrics['accuracy']:.4f}")
            print(f"  Macro F1:    {intent_metrics['macro_f1']:.4f}")
            print(f"  Weighted F1: {intent_metrics['weighted_f1']:.4f}")
            print(f"  Mean confidence: {np.mean(pred_confidences):.4f}")
        else:
            print("  No labeled golden set examples - skipping intent eval")
    else:
        print("  Golden set not available - evaluating on test split")
        # Use classifier's own test split metrics
        if 'baseline_metrics' in results:
            m = results['baseline_metrics']
            print(f"  Accuracy:    {m['accuracy']:.4f}")
            print(f"  Macro F1:    {m['macro_f1']:.4f}")
    
    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f}s")
    
    # ============================================================
    # Step 5: Baselines comparison
    # ============================================================
    print("\n[5/7] Running baselines...")
    
    from src.evaluation.baselines import TrivialBaseline, SimpleBaseline
    
    eval_messages = sample_df['customer_message'].head(100).tolist()
    
    trivial = TrivialBaseline()
    trivial_results = trivial.predict_batch(eval_messages)
    
    simple = SimpleBaseline()
    data_for_baseline = labeled_df if labeled_df is not None else sample_df
    simple.fit(data_for_baseline)
    simple_results = simple.predict_batch(eval_messages)
    
    system_results = classifier.predict(eval_messages)
    
    results['baselines'] = {
        'trivial': {
            'mean_confidence': float(np.mean([r['confidence'] for r in trivial_results])),
            'escalation_rate': float(np.mean([r['should_escalate'] for r in trivial_results])),
        },
        'simple': {
            'mean_confidence': float(np.mean([r['confidence'] for r in simple_results])),
            'mean_similarity': float(np.mean([r.get('similarity_score', 0) for r in simple_results])),
            'escalation_rate': float(np.mean([r['should_escalate'] for r in simple_results])),
        },
        'system': {
            'mean_confidence': float(np.mean([r['confidence'] for r in system_results])),
        }
    }
    
    print(f"  Trivial baseline confidence: {results['baselines']['trivial']['mean_confidence']:.4f}")
    print(f"  Simple baseline similarity:  {results['baselines']['simple']['mean_similarity']:.4f}")
    print(f"  System confidence:           {results['baselines']['system']['mean_confidence']:.4f}")
    
    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f}s")
    
    # ============================================================
    # Step 6: Retrieval evaluation (if FAISS available)
    # ============================================================
    print("\n[6/7] Retrieval evaluation...")
    
    index_dir = os.path.join(PROJECT_ROOT, 'models', 'embedding_index')
    if os.path.exists(os.path.join(index_dir, 'faiss_index.bin')):
        try:
            from src.retrieval.retrieve import HistoricalRetriever
            retriever = HistoricalRetriever(index_dir=index_dir)
            
            # Test retrieval on a sample
            test_queries = eval_messages[:20]
            retrieval_scores = []
            for q in test_queries:
                rets = retriever.retrieve(q, top_k=5)
                if rets:
                    retrieval_scores.append(rets[0]['score'])
            
            results['retrieval'] = {
                'mean_top1_similarity': float(np.mean(retrieval_scores)) if retrieval_scores else 0,
                'queries_evaluated': len(test_queries),
            }
            print(f"  Mean top-1 similarity: {results['retrieval']['mean_top1_similarity']:.4f}")
        except Exception as e:
            print(f"  Retrieval eval failed: {e}")
    else:
        print("  FAISS index not built - skipping retrieval eval")
    
    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f}s")
    
    # ============================================================
    # Step 7: Escalation evaluation
    # ============================================================
    print("\n[7/7] Escalation evaluation...")
    
    from src.escalation.decision import EscalationEngine
    engine = EscalationEngine()
    
    esc_results = []
    for pred in system_results[:50]:
        esc = engine.decide(
            intent_confidence=pred['confidence'],
            intent_probabilities=pred.get('all_probabilities', {}),
            retrieval_results=[],
            customer_message=eval_messages[len(esc_results)],
        )
        esc_results.append(esc)
    
    escalation_rate = np.mean([r['should_escalate'] for r in esc_results])
    results['escalation'] = {
        'escalation_rate': float(escalation_rate),
        'mean_score': float(np.mean([r['escalation_score'] for r in esc_results])),
    }
    print(f"  Escalation rate: {escalation_rate:.1%}")
    
    # ============================================================
    # Final Summary
    # ============================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("  QUICK EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Within 15-minute target: {'YES' if total_time < 900 else 'NO'}")
    
    print(f"\n  === HEADLINE METRICS ===")
    if 'intent_metrics' in results:
        print(f"  Intent Accuracy:       {results['intent_metrics']['accuracy']:.4f}")
        print(f"  Intent Macro F1:       {results['intent_metrics']['macro_f1']:.4f}")
    elif 'baseline_metrics' in results:
        print(f"  Intent Accuracy:       {results['baseline_metrics']['accuracy']:.4f}")
        print(f"  Intent Macro F1:       {results['baseline_metrics']['macro_f1']:.4f}")
    
    if 'retrieval' in results:
        print(f"  Retrieval Top-1 Sim:   {results['retrieval']['mean_top1_similarity']:.4f}")
    
    print(f"  Escalation Rate:       {results['escalation']['escalation_rate']:.1%}")
    
    # Save results
    results['total_time_seconds'] = total_time
    results_path = os.path.join(PROJECT_ROOT, 'reports', 'quick_eval_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
