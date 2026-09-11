"""
Escalation Decision Module
Hybrid escalation engine using classifier confidence + retrieval quality + rules + LLM assessment.
Implements a strict priority decision hierarchy:
1. Explicit human request
2. Safety / sensitive issue
3. Low intent confidence
4. Insufficient retrieval evidence
5. LLM recommendation
6. Safe auto-handle
"""

import os
import re
import yaml
from typing import Dict, Any, List, Optional


class EscalationEngine:
    """
    Hybrid escalation decision engine.
    Combines multiple signals with strict priority precedence to decide AUTO_HANDLE vs ESCALATE_TO_HUMAN.
    """
    
    EXPLICIT_HUMAN_PATTERNS = [
        r'\bconnect\s+(?:me\s+)?(?:with\s+)?(?:a\s+)?human\b',
        r'\btalk\s+to\s+(?:a\s+)?(?:human|person|agent|representative)\b',
        r'\bspeak\s+to\s+(?:a\s+)?(?:human|person|agent|representative|someone|support)\b',
        r'\bhuman\s+(?:agent|person|support|please|rep)\b',
        r'\breal\s+person\b',
        r'\btransfer\s+me\b',
        r'\b(?:don\'t|dont|do\s+not)\s+want\s+(?:a\s+)?bot\b',
        r'\blive\s+agent\b',
        r'\bcustomer\s+service\s+(?:rep|agent|person)\b',
    ]

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'configs', 'thresholds.yaml'
            )
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.escalation_threshold = self.config.get('escalation_threshold', 0.5)
        self.signals_config = self.config.get('signals', {})

    def is_explicit_human_request(self, text: str) -> bool:
        """Check if message explicitly requests human agent connection."""
        if not text:
            return False
        text_lower = text.lower().strip()
        for pattern in self.EXPLICIT_HUMAN_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def decide(self, 
               intent_confidence: float,
               intent_probabilities: Dict[str, float],
               retrieval_results: List[dict],
               customer_message: str,
               llm_escalation: bool = False,
               llm_confidence: float = 0.5,
               llm_escalation_reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Make an authoritative escalation decision using a strict precedence hierarchy.
        
        Precedence Hierarchy:
        1. Explicit human request -> ESCALATE (decision_source: explicit_human_request)
        2. Safety / sensitive issue -> ESCALATE (decision_source: sensitive_content)
        3. Low intent confidence -> ESCALATE (decision_source: low_intent_confidence)
        4. Insufficient retrieval evidence -> ESCALATE (decision_source: insufficient_evidence)
        5. LLM recommendation -> ESCALATE (decision_source: llm_recommendation)
        6. Safe auto-handle -> AUTO_HANDLE (decision_source: policy_and_confidence_gate)
        
        Returns:
            dict with: should_escalate, decision, decision_source, escalation_reason, signals, escalation_score
        """
        msg_lower = customer_message.lower() if customer_message else ""
        signals = {}

        # Signal checks for analytics
        is_human_req = self.is_explicit_human_request(customer_message)
        signals['explicit_human_request'] = is_human_req

        sensitive_keywords = [
            'legal', 'lawsuit', 'lawyer', 'fraud', 'unauthorized',
            'hack', 'stolen', 'death', 'threat', 'sue', 'police',
            'discrimination', 'harassment',
        ]
        found_sensitive = [kw for kw in sensitive_keywords if kw in msg_lower]
        signals['sensitive_content'] = len(found_sensitive) > 0

        low_intent_thresh = self.signals_config.get('low_intent_confidence', {}).get('threshold', 0.45)
        signals['low_intent_confidence'] = intent_confidence < low_intent_thresh

        retrieval_thresh = self.signals_config.get('low_retrieval_quality', {}).get('threshold', 0.35)
        top_retrieval_score = max([r.get('score', 0) for r in retrieval_results]) if retrieval_results else 0.0
        signals['low_retrieval_quality'] = (not retrieval_results) or (top_retrieval_score < retrieval_thresh)

        signals['llm_recommendation'] = bool(llm_escalation)

        # STRICT PRECEDENCE EVALUATION (Prevents Contradiction)
        
        # Priority 1: Explicit Human Request
        if is_human_req:
            return {
                'should_escalate': True,
                'decision': 'ESCALATE_TO_HUMAN',
                'decision_source': 'explicit_human_request',
                'escalation_reason': 'Customer explicitly requested to speak with a human agent.',
                'escalation_score': 1.0,
                'escalation_threshold': self.escalation_threshold,
                'signals': signals,
            }

        # Priority 2: Safety / Sensitive Content
        if found_sensitive:
            return {
                'should_escalate': True,
                'decision': 'ESCALATE_TO_HUMAN',
                'decision_source': 'sensitive_content',
                'escalation_reason': f"Sensitive keywords detected: {', '.join(found_sensitive)}",
                'escalation_score': 0.9,
                'escalation_threshold': self.escalation_threshold,
                'signals': signals,
            }

        # Priority 3: Low Intent Confidence
        if signals['low_intent_confidence']:
            return {
                'should_escalate': True,
                'decision': 'ESCALATE_TO_HUMAN',
                'decision_source': 'low_intent_confidence',
                'escalation_reason': f"Low intent confidence ({intent_confidence:.2f} < threshold {low_intent_thresh})",
                'escalation_score': 0.85,
                'escalation_threshold': self.escalation_threshold,
                'signals': signals,
            }

        # Priority 4: Insufficient Retrieval Evidence
        if signals['low_retrieval_quality']:
            reason = "No historical cases retrieved" if not retrieval_results else f"Low retrieval quality (best similarity: {top_retrieval_score:.2f} < {retrieval_thresh})"
            return {
                'should_escalate': True,
                'decision': 'ESCALATE_TO_HUMAN',
                'decision_source': 'insufficient_evidence',
                'escalation_reason': reason,
                'escalation_score': 0.8,
                'escalation_threshold': self.escalation_threshold,
                'signals': signals,
            }

        # Priority 5: LLM Recommendation
        if llm_escalation:
            reason = llm_escalation_reason or "LLM recommended escalation due to issue complexity"
            return {
                'should_escalate': True,
                'decision': 'ESCALATE_TO_HUMAN',
                'decision_source': 'llm_recommendation',
                'escalation_reason': reason,
                'escalation_score': 0.75,
                'escalation_threshold': self.escalation_threshold,
                'signals': signals,
            }

        # Priority 6: Safe Auto-Handle (NO CONTRADICTION)
        return {
            'should_escalate': False,
            'decision': 'AUTO_HANDLE',
            'decision_source': 'policy_and_confidence_gate',
            'escalation_reason': None,
            'escalation_score': 0.1,
            'escalation_threshold': self.escalation_threshold,
            'signals': signals,
        }
