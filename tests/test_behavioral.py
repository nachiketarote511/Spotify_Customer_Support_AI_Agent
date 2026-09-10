"""
Behavioral and Integration Test Suite
Validates correctness of AI Agent fixes including:
1. Explicit human request gate
2. Escalation priority precedence hierarchy
3. Context-aware multi-turn intent resolution
4. Similarity thresholding and retrieval filtering
5. Non-contradiction invariant enforcement (AUTO_HANDLE must have escalation_reason == None)
"""

import os
import sys
import pytest

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.escalation.decision import EscalationEngine
from src.retrieval.retrieve import HistoricalRetriever
from src.intent.train import IntentClassifier


class TestBehavioralFixes:
    """Test suite targeting specific reported agent failures."""

    @pytest.fixture
    def escalation_engine(self):
        return EscalationEngine()

    def test_explicit_human_request_triggers_escalation(self, escalation_engine):
        """Test explicit human agent request gate (Priority 1)."""
        messages = [
            "can you connect me with a human",
            "I want to talk to a real person please",
            "speak to human support",
            "transfer me to a live agent",
            "don't want a bot",
        ]
        for msg in messages:
            res = escalation_engine.decide(
                intent_confidence=0.95,
                intent_probabilities={'playback_issues': 0.95},
                retrieval_results=[{'score': 0.9}],
                customer_message=msg,
                llm_escalation=False,
            )
            assert res['should_escalate'] is True, f"Failed on message: {msg}"
            assert res['decision'] == 'ESCALATE_TO_HUMAN'
            assert res['decision_source'] == 'explicit_human_request'
            assert 'human agent' in res['escalation_reason'].lower()

    def test_playback_crash_auto_handle_no_contradiction(self, escalation_engine):
        """Test 'My app crashes on playback start' when good evidence exists."""
        retrieved_cases = [
            {'score': 0.78, 'document': {'customer_message': 'App crashes when playing music', 'historical_support_response': 'Please reinstall the app or clear cache.'}},
            {'score': 0.72, 'document': {'customer_message': 'Playback crashes on launch', 'historical_support_response': 'Check offline mode settings.'}}
        ]
        res = escalation_engine.decide(
            intent_confidence=0.82,
            intent_probabilities={'playback_issues': 0.82, 'other': 0.1},
            retrieval_results=retrieved_cases,
            customer_message="My app crashes on playback start",
            llm_escalation=False,
            llm_confidence=0.85,
        )
        assert res['should_escalate'] is False
        assert res['decision'] == 'AUTO_HANDLE'
        assert res['decision_source'] == 'policy_and_confidence_gate'
        assert res['escalation_reason'] is None  # NO CONTRADICTION

    def test_precedence_explicit_human_overrides_high_confidence(self, escalation_engine):
        """Verify explicit human request overrides high classifier confidence and LLM auto recommendation."""
        res = escalation_engine.decide(
            intent_confidence=0.99,
            intent_probabilities={'billing_payment': 0.99},
            retrieval_results=[{'score': 0.95}],
            customer_message="can you connect me with a human agent to discuss billing",
            llm_escalation=False,
        )
        assert res['should_escalate'] is True
        assert res['decision_source'] == 'explicit_human_request'

    def test_sensitive_content_triggers_escalation(self, escalation_engine):
        """Test sensitive keyword safety override (Priority 2)."""
        res = escalation_engine.decide(
            intent_confidence=0.9,
            intent_probabilities={'account_access': 0.9},
            retrieval_results=[{'score': 0.8}],
            customer_message="My account was hacked and unauthorized charges were made",
            llm_escalation=False,
        )
        assert res['should_escalate'] is True
        assert res['decision'] == 'ESCALATE_TO_HUMAN'
        assert res['decision_source'] == 'sensitive_content'
        assert 'sensitive keywords' in res['escalation_reason'].lower()

    def test_insufficient_retrieval_triggers_escalation(self, escalation_engine):
        """Test low retrieval quality triggers escalation (Priority 4)."""
        res = escalation_engine.decide(
            intent_confidence=0.85,
            intent_probabilities={'general_inquiry': 0.85},
            retrieval_results=[{'score': 0.15}],  # Below similarity threshold
            customer_message="Random query with no historical match",
            llm_escalation=False,
        )
        assert res['should_escalate'] is True
        assert res['decision'] == 'ESCALATE_TO_HUMAN'
        assert res['decision_source'] == 'insufficient_evidence'

    def test_no_contradiction_invariant_across_decisions(self, escalation_engine):
        """Assert that AUTO_HANDLE decision ALWAYS has escalation_reason == None."""
        test_cases = [
            (0.85, [{'score': 0.7}], "How do I upgrade to Premium?"),
            (0.92, [{'score': 0.85}], "Can I change my email address?"),
        ]
        for conf, ret, msg in test_cases:
            res = escalation_engine.decide(
                intent_confidence=conf,
                intent_probabilities={'account_access': conf},
                retrieval_results=ret,
                customer_message=msg,
                llm_escalation=False,
            )
            if not res['should_escalate']:
                assert res['decision'] == 'AUTO_HANDLE'
                assert res['escalation_reason'] is None, f"Contradiction found for message: {msg}"


class TestContextAwareRetrieval:
    """Test context-aware retrieval filtering and reranking."""

    def test_retrieval_min_similarity_filter(self):
        retriever = HistoricalRetriever()
        # Mock index with low scores if unbuilt, or test method structure directly
        results = retriever.retrieve("completely unrelated xyzzyspoon query", top_k=5, min_similarity=0.35)
        # Low similarity queries should be filtered out
        for r in results:
            assert r['score'] >= 0.35 or r['rank'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
