"""
Grounding validation tests for the RAG response generator.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.rag.generator import RAGGenerator

@pytest.fixture
def generator():
    # Load default config
    return RAGGenerator()

def test_grounding_evidence_gate_fails_no_evidence(generator):
    """Test that if no valid evidence is provided, it returns a safe fallback and escalates."""
    result = generator.generate(
        customer_message="My account was hacked and they bought premium.",
        predicted_intent="account_access",
        intent_confidence=0.8,
        retrieved_cases=[], # No evidence
        conversation_context="",
        brand="SpotifyCares"
    )
    assert result['should_escalate'] is True
    assert "escalate" in result['reply'].lower() or "specialist" in result['reply'].lower()
    assert result['escalation_reason'] == "insufficient retrieval evidence"

def test_grounding_evidence_gate_fails_low_score_evidence(generator):
    """Test that evidence with a score below threshold is rejected."""
    result = generator.generate(
        customer_message="My app is crashing.",
        predicted_intent="playback_issues",
        intent_confidence=0.8,
        retrieved_cases=[{'score': 0.1, 'document': {'customer_message': 'x', 'historical_support_response': 'y'}}], # Score below 0.3
        conversation_context="",
        brand="SpotifyCares"
    )
    assert result['should_escalate'] is True
    assert result['escalation_reason'] == "insufficient retrieval evidence"

def test_grounding_validator_blocks_forbidden_phrases(generator):
    """Test that the internal deterministic validator blocks hallucinatory claims."""
    assert generator._validate_grounding("I checked your account and everything looks fine.") is False
    assert generator._validate_grounding("I have fixed this issue for you.") is False
    assert generator._validate_grounding("I escalated this to our technical team.") is False
    assert generator._validate_grounding("Here is a link to reset: http://example.com/reset") is True # URLs are allowed if they don't contain other forbidden phrases, although we might want to block them later if needed
    assert generator._validate_grounding("I can see your account status is premium.") is False
    assert generator._validate_grounding("Try logging out and logging back in.") is True # Troubleshooting is allowed if grounded
