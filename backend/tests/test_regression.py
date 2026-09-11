import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.api.main import app, state, lifespan

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_models():
    """Ensure models are loaded for testing"""
    import asyncio
    
    async def load():
        async with lifespan(app):
            pass
            
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(load())
    yield


def test_model_artifacts_load_successfully():
    """Test that models and configurations successfully load."""
    assert state.initialized == True
    assert state.config is not None
    assert state.intent_classifier is not None
    assert state.retriever is not None


def test_chat_message_reaches_classifier():
    """Test that a message correctly passes through the classifier and returns a known intent."""
    response = client.post("/support-agent", json={"message": "My app is crashing", "conversation_context": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] != "unknown"
    assert data["intent_confidence"] > 0.0


def test_classifier_does_not_return_unknown_for_all_inputs():
    """Test various distinct inputs do not all return 'unknown'."""
    messages = [
        "My app is crashing",
        "I was charged twice",
        "How do I reset my password?",
        "I can't play any songs"
    ]
    for msg in messages:
        response = client.post("/support-agent", json={"message": msg, "conversation_context": ""})
        data = response.json()
        assert data["intent"] != "unknown"
        assert data["intent_confidence"] > 0.1


def test_retriever_returns_results_when_index_has_matches():
    """Test that FAISS retrieves valid historical cases."""
    response = client.post("/support-agent", json={"message": "My app is crashing"})
    data = response.json()
    assert len(data["retrieved_cases"]) > 0
    assert "score" in data["retrieved_cases"][0]


def test_human_request_escalates():
    """Test that explicit requests for a human properly escalate."""
    response = client.post("/support-agent", json={"message": "I want to speak to my lawyer right now."})
    data = response.json()
    assert data["should_escalate"] == True


def test_backend_does_not_silently_hide_pipeline_errors(monkeypatch):
    """Test that if the classifier fails, it logs and falls back securely instead of swallowing silently."""
    
    def mock_predict(*args, **kwargs):
        raise RuntimeError("Simulated classifier crash")
    
    monkeypatch.setattr(state.intent_classifier, "predict", mock_predict)
    
    response = client.post("/support-agent", json={"message": "Crash test"})
    assert response.status_code == 200
    data = response.json()
    
    # Due to our new robust error handling, it shouldn't just be 'unknown' and 0 confidence without escalation.
    # We set should_escalate=True when pipeline components fail (or the policy gate catches it).
    assert data["should_escalate"] == True
    assert data["decision"] == "ESCALATE_TO_HUMAN"
    reason = data.get("escalation_reason", "").lower()
    assert "system error" in reason or "generation error" in reason or "error" in reason or "low intent confidence" in reason
