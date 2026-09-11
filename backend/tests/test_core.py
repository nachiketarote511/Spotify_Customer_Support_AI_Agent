"""
Tests for core components
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add backend root (for app.* imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataValidation:
    """Test data validation and schema enforcement."""
    
    def test_required_columns(self):
        from app.data.validation import validate_schema, REQUIRED_COLUMNS
        
        # Valid dataframe
        df = pd.DataFrame({
            'conversation_id': ['1', '2'],
            'brand': ['spotify', 'spotify'],
            'customer_message': ['Help me', 'Fix this'],
            'historical_support_response': ['Sure!', 'OK!'],
            'timestamp': ['2023-01-01', '2023-01-02'],
            'conversation_context': ['', ''],
            'customer_author': ['user1', 'user2'],
            'support_author': ['brand', 'brand'],
            'conversation_length': [2, 3],
        })
        result = validate_schema(df)
        assert result['valid'] == True
    
    def test_missing_columns(self):
        from app.data.validation import validate_schema
        
        df = pd.DataFrame({
            'conversation_id': ['1'],
            'brand': ['spotify'],
        })
        result = validate_schema(df)
        assert result['valid'] == False
        assert len(result['errors']) > 0
    
    def test_empty_dataframe(self):
        from app.data.validation import validate_schema
        
        df = pd.DataFrame()
        result = validate_schema(df)
        assert result['valid'] == False
    
    def test_no_leakage(self):
        from app.data.validation import validate_no_leakage
        
        train = pd.DataFrame({
            'conversation_id': ['1', '2', '3'],
            'customer_message': ['hello', 'help', 'fix'],
        })
        test = pd.DataFrame({
            'conversation_id': ['4', '5'],
            'customer_message': ['world', 'test'],
        })
        result = validate_no_leakage(train, test)
        assert result['leakage_detected'] == False
    
    def test_leakage_detected(self):
        from app.data.validation import validate_no_leakage
        
        train = pd.DataFrame({
            'conversation_id': ['1', '2', '3'],
            'customer_message': ['hello', 'help', 'fix'],
        })
        test = pd.DataFrame({
            'conversation_id': ['3', '4'],  # ID 3 overlaps
            'customer_message': ['fix', 'test'],
        })
        result = validate_no_leakage(train, test)
        assert result['leakage_detected'] == True


class TestConversationBuilder:
    """Test conversation reconstruction."""
    
    def test_clean_text(self):
        from app.data.conversation_builder import clean_text
        
        assert clean_text('@brand hello world') == 'hello world'
        assert clean_text('@brand check https://example.com') == 'check [URL]'
        assert clean_text(None) == ''
    
    def test_extract_mention(self):
        from app.data.conversation_builder import extract_mention
        
        assert extract_mention('@spotify help me') == 'spotify'
        assert extract_mention('hello world') is None
        assert extract_mention(None) is None


class TestPipelineSelector:
    """Test pipeline selection logic."""
    
    def test_select_pandas(self):
        from app.pipelines.pipeline_selector import select_pipeline
        
        analysis = {'classification': 'SMALL'}
        assert select_pipeline(analysis) == 'pandas'
    
    def test_select_spark(self):
        from app.pipelines.pipeline_selector import select_pipeline
        
        analysis = {'classification': 'LARGE'}
        assert select_pipeline(analysis) == 'spark'
    
    def test_user_override(self):
        from app.pipelines.pipeline_selector import select_pipeline
        
        analysis = {'classification': 'SMALL'}
        assert select_pipeline(analysis, 'spark') == 'spark'


class TestEscalationEngine:
    """Test escalation decision logic."""
    
    def test_auto_handle(self):
        from app.escalation.decision import EscalationEngine
        
        engine = EscalationEngine()
        result = engine.decide(
            intent_confidence=0.9,
            intent_probabilities={'billing': 0.9, 'technical': 0.1},
            retrieval_results=[{'score': 0.8}],
            customer_message='How do I change my plan?',
        )
        assert result['should_escalate'] == False
    
    def test_escalate_low_confidence(self):
        from app.escalation.decision import EscalationEngine
        
        engine = EscalationEngine()
        result = engine.decide(
            intent_confidence=0.1,
            intent_probabilities={'billing': 0.15, 'technical': 0.14},
            retrieval_results=[],
            customer_message='I want my money back or I will sue',
        )
        assert result['should_escalate'] == True
    
    def test_escalate_sensitive(self):
        from app.escalation.decision import EscalationEngine
        
        engine = EscalationEngine()
        result = engine.decide(
            intent_confidence=0.8,
            intent_probabilities={'billing': 0.8},
            retrieval_results=[{'score': 0.7}],
            customer_message='Someone hacked my account and stole my data',
        )
        assert result['signals']['sensitive_content'] == True


class TestMetrics:
    """Test evaluation metrics computation."""
    
    def test_classification_metrics(self):
        from app.evaluation.metrics import compute_classification_metrics
        
        y_true = ['a', 'b', 'a', 'c', 'b']
        y_pred = ['a', 'b', 'a', 'b', 'b']
        
        metrics = compute_classification_metrics(y_true, y_pred)
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['macro_f1'] <= 1
    
    def test_escalation_metrics(self):
        from app.evaluation.metrics import compute_escalation_metrics
        
        y_true = [True, False, True, False]
        y_pred = [True, False, False, False]
        
        metrics = compute_escalation_metrics(y_true, y_pred)
        assert 'accuracy' in metrics
        assert 'escalation_f1' in metrics


class TestBaselines:
    """Test baseline implementations."""
    
    def test_trivial_baseline(self):
        from app.evaluation.baselines import TrivialBaseline
        
        baseline = TrivialBaseline()
        result = baseline.predict('My music won\'t play')
        assert 'reply' in result
        assert result['confidence'] == 0.1
    
    def test_simple_baseline(self):
        from app.evaluation.baselines import SimpleBaseline
        
        baseline = SimpleBaseline()
        df = pd.DataFrame({
            'customer_message': [
                'music not playing issue', 
                'cant login issue', 
                'billing problem here',
                'another issue here',
                'and another one'
            ],
            'historical_support_response': [
                'try restarting', 
                'reset password', 
                'check payment',
                'fixing',
                'fixed'
            ],
        })
        baseline.fit(df)
        
        result = baseline.predict('my music stopped issue')
        assert 'reply' in result
        assert 'similarity_score' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
