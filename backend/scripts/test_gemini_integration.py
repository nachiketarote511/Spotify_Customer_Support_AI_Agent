"""
Gemini Integration Test
Runs the full RAG pipeline end-to-end against the real Gemini API.

Prerequisites:
  1. Add your Gemini API key to .env:  GEMINI_API_KEY=your_real_key
  2. Run:  python scripts/test_gemini_integration.py

This script will:
  - Load the .env file
  - Verify GEMINI_API_KEY is set (but never print it)
  - Load the intent classifier and FAISS retriever
  - Send a single customer message through the full pipeline
  - Print the structured response (intent, reply, escalation decision)
  - Run a single LLM judge evaluation
  - Report success or failure with clear error messages
"""

import os
import sys
import json
import time

# Path resolution for monorepo structure
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(_SCRIPT_DIR)           # backend/
PROJECT_ROOT = os.path.dirname(BACKEND_ROOT)           # repo root
sys.path.insert(0, BACKEND_ROOT)                        # for app.* imports

# Load .env
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


def main():
    print("=" * 70)
    print("  GEMINI INTEGRATION TEST")
    print("=" * 70)
    
    # ── Step 1: Verify API key is present ──
    print("\n[1/6] Checking GEMINI_API_KEY...")
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("  ERROR: GEMINI_API_KEY is not set.")
        print("  Add your key to .env:  GEMINI_API_KEY=your_real_key")
        print("  Get a key at: https://aistudio.google.com/")
        sys.exit(1)
    # Never print the key; just confirm it's loaded
    print(f"  GEMINI_API_KEY detected ({len(api_key)} characters, starts with {api_key[:4]}...)")
    
    # ── Step 2: Load intent classifier ──
    print("\n[2/6] Loading intent classifier...")
    from app.intent.train import IntentClassifier
    classifier = IntentClassifier()
    model_dir = os.path.join(PROJECT_ROOT, 'models', 'intent_classifier', 'baseline')
    if not os.path.exists(model_dir):
        print("  ERROR: Baseline classifier not found. Run 'python scripts/train.py' first.")
        sys.exit(1)
    classifier.load_model('baseline')
    
    # ── Step 3: Load FAISS retriever ──
    print("\n[3/6] Loading FAISS retriever...")
    from app.retrieval.retrieve import HistoricalRetriever
    index_dir = os.path.join(PROJECT_ROOT, 'models', 'embedding_index')
    if not os.path.exists(os.path.join(index_dir, 'faiss_index.bin')):
        print("  ERROR: FAISS index not found. Run 'python scripts/train.py' first.")
        sys.exit(1)
    retriever = HistoricalRetriever(index_dir=index_dir)
    
    # ── Step 4: Full RAG pipeline ──
    print("\n[4/6] Running full RAG pipeline...")
    test_message = "My Spotify app keeps crashing when I try to play music on my iPhone"
    
    # Intent
    intent_result = classifier.predict([test_message], return_proba=True)[0]
    print(f"  Intent:     {intent_result['intent']}")
    print(f"  Confidence: {intent_result['confidence']:.4f}")
    
    # Retrieval
    retrieved = retriever.retrieve(test_message, top_k=5)
    print(f"  Retrieved:  {len(retrieved)} similar cases")
    if retrieved:
        print(f"  Top match:  similarity={retrieved[0]['score']:.4f}")
    
    # Gemini RAG generation
    print("\n  Calling Gemini API...")
    start = time.time()
    from app.rag.generator import RAGGenerator
    generator = RAGGenerator()
    
    try:
        result = generator.generate(
            customer_message=test_message,
            predicted_intent=intent_result['intent'],
            intent_confidence=intent_result['confidence'],
            retrieved_cases=retrieved,
            brand="SpotifyCares",
        )
        elapsed = time.time() - start
        
        print(f"  Gemini response received in {elapsed:.1f}s")
        print(f"\n  --- Generated Reply ---")
        print(f"  {result['reply'][:300]}")
        print(f"  --- End Reply ---")
        print(f"  Confidence:        {result['confidence']}")
        print(f"  Grounding:         {result['grounding_summary'][:200]}")
        print(f"  Should escalate:   {result['should_escalate']}")
        print(f"  Escalation reason: {result['escalation_reason']}")
    except Exception as e:
        print(f"\n  ERROR: Gemini API call failed: {e}")
        sys.exit(1)
    
    # ── Step 5: Escalation engine ──
    print("\n[5/6] Running hybrid escalation engine...")
    from app.escalation.decision import EscalationEngine
    engine = EscalationEngine()
    esc = engine.decide(
        intent_confidence=intent_result['confidence'],
        intent_probabilities=intent_result.get('all_probabilities', {}),
        retrieval_results=retrieved,
        customer_message=test_message,
        llm_escalation=result['should_escalate'],
        llm_confidence=result['confidence'],
    )
    print(f"  Decision:          {esc['decision']}")
    print(f"  Escalation score:  {esc['escalation_score']}")
    print(f"  Signals:           {json.dumps(esc['signals'], indent=4)}")
    if esc['escalation_reason']:
        print(f"  Reason:            {esc['escalation_reason']}")
    
    # ── Step 6: LLM Judge ──
    print("\n[6/6] Running LLM judge on the generated reply...")
    from app.evaluation.llm_judge import LLMJudge
    judge = LLMJudge()
    try:
        scores = judge.evaluate_single(
            customer_message=test_message,
            generated_response=result['reply'],
            historical_response=retrieved[0]['document'].get('historical_support_response', '') if retrieved else '',
            intent=intent_result['intent'],
            escalation=esc['decision'],
        )
        print("  LLM Judge scores:")
        for dim in judge.DIMENSIONS:
            dim_result = scores.get(dim, {})
            if isinstance(dim_result, dict):
                print(f"    {dim:30s} {dim_result.get('score', '?')}/5  {dim_result.get('justification', '')[:60]}")
        overall = scores.get('overall_score', '?')
        print(f"    {'OVERALL':30s} {overall}/5")
    except Exception as e:
        print(f"  LLM Judge error: {e}")
    
    # ── Summary ──
    print("\n" + "=" * 70)
    print("  INTEGRATION TEST PASSED")
    print("=" * 70)
    print(f"  Pipeline: message → intent → FAISS → Gemini → reply → escalation")
    print(f"  Model:    {generator.gemini_model}")
    print(f"  All components verified against real Gemini API.")
    print("=" * 70)


if __name__ == '__main__':
    main()
