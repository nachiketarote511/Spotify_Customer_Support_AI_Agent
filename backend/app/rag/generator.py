"""
RAG Generator Module
Orchestrates the full RAG pipeline: Intent → Retrieval → Gemini → Response.
Enforces Strict Evidence Grounding policies.
"""

import os
import json
import hashlib
import time
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from app.rag.prompt import build_system_prompt, build_grounded_prompt

# Pydantic schema for Gemini structured output
class GroundedResponse(BaseModel):
    reply: str
    grounded_claims: list[str] = Field(default_factory=list)
    evidence_ids: list[int] = Field(default_factory=list)
    needs_escalation: bool
    grounding_confidence: float

class RAGGenerator:
    """Full RAG pipeline for generating grounded customer support responses."""
    
    def __init__(self, config: dict = None):
        if config is None:
            import yaml
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'configs', 'config.yaml'
            )
            with open(config_path) as f:
                config = yaml.safe_load(f)
        
        self.config = config
        self.gemini_model = config.get('gemini', {}).get('model', 'gemini-2.0-flash')
        self.temperature = config.get('gemini', {}).get('temperature', 0.1) # Strictly low for grounding
        self.max_tokens = config.get('gemini', {}).get('max_output_tokens', 1024)
        
        # Retrieval quality settings
        retrieval_conf = config.get('retrieval', {})
        self.similarity_threshold = retrieval_conf.get('similarity_threshold', 0.3)
        
        # Cache for responses
        self.cache_enabled = config.get('cache', {}).get('enabled', True)
        self.cache_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 
            config.get('cache', {}).get('directory', '.cache')
        )
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Gemini client (lazy init)
        self._client = None
    
    def _init_gemini(self):
        """Initialize Gemini API client using google-genai SDK."""
        if self._client is not None:
            return
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add your API key to the .env file in the project root: "
                "GEMINI_API_KEY=your_key_here  "
                "Get a key at https://aistudio.google.com/"
            )
        
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "google-genai package required. Install: pip install google-genai"
            )
    
    def generate(self, customer_message: str, predicted_intent: str,
                 intent_confidence: float, retrieved_cases: list,
                 conversation_context: str = "",
                 brand: str = "SpotifyCares") -> Dict[str, Any]:
        """
        Generate a grounded response using the RAG pipeline.
        """
        # --- EVIDENCE GATE ---
        # 1. Filter out poor evidence below threshold
        valid_cases = [c for c in retrieved_cases if c.get('score', 0) >= self.similarity_threshold]
        
        if not valid_cases:
            print("  Evidence Gate Failed: No cases passed similarity threshold.")
            return {
                'reply': "I'm sorry, but I don't have enough information from the available support history to give you a reliable answer. I can escalate this to a support specialist.",
                'confidence': 0.0,
                'grounding_summary': "Insufficient retrieval evidence.",
                'should_escalate': True,
                'escalation_reason': "insufficient retrieval evidence"
            }

        # Check cache
        cache_key = self._cache_key(customer_message, predicted_intent, valid_cases)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # --- GENERATION ---
        result = self._attempt_generation(
            customer_message, predicted_intent, intent_confidence,
            valid_cases, conversation_context, brand, is_retry=False
        )

        # --- GROUNDING VALIDATOR ---
        if not self._validate_grounding(result['reply']):
            print("  Grounding Validator Failed: Detected forbidden claims. Retrying once...")
            # Retry once with stricter prompt (appending a strong warning)
            result = self._attempt_generation(
                customer_message, predicted_intent, intent_confidence,
                valid_cases, conversation_context, brand, is_retry=True
            )
            # If it fails again, fallback safely
            if not self._validate_grounding(result['reply']):
                print("  Grounding Validator Failed again. Escalating.")
                result = {
                    'reply': "I'm sorry, I cannot perform that action or verify those details at this time. Let me escalate this to a human specialist.",
                    'confidence': 0.0,
                    'grounding_summary': "Grounding validation failed for LLM response.",
                    'should_escalate': True,
                    'escalation_reason': "grounding failure"
                }

        # Cache result
        if self.cache_enabled and not result.get('should_escalate'):
            self._set_cached(cache_key, result)
        
        return result
    
    def _attempt_generation(self, customer_message: str, predicted_intent: str,
                 intent_confidence: float, valid_cases: list,
                 conversation_context: str, brand: str, is_retry: bool) -> Dict[str, Any]:
        
        system_prompt = build_system_prompt(brand)
        if is_retry:
            system_prompt += "\n\nWARNING: Your previous response was rejected for inventing actions (e.g. 'I checked your account') or hallucinating policies. DO NOT do this. Adhere strictly to the evidence."
            
        user_prompt = build_grounded_prompt(
            customer_message, predicted_intent, intent_confidence,
            valid_cases, conversation_context, brand
        )
        
        try:
            response_text = self._call_gemini(system_prompt, user_prompt)
            return self._parse_response(response_text)
        except Exception as e:
            print(f"  Gemini API call fallback triggered: {e}")
            return {
                'reply': "I'm experiencing a system error and cannot process this right now. Please hold while I escalate your request.",
                'confidence': 0.0,
                'grounding_summary': f"API Error: {e}",
                'should_escalate': True,
                'escalation_reason': "System error"
            }

    def _validate_grounding(self, reply_text: str) -> bool:
        """
        Deterministic post-generation grounding check.
        Returns False if the reply contains forbidden hallucinated phrases.
        """
        reply_lower = reply_text.lower()
        forbidden_phrases = [
            "i checked your account",
            "i've checked your account",
            "i have checked your account",
            "i have fixed this",
            "i fixed this",
            "i escalated this", # We shouldn't claim we did it unless we actually return an escalation signal
            "i can see your account",
            "i have contacted the team",
            "i've updated your",
            "i have updated your"
        ]
        for phrase in forbidden_phrases:
            if phrase in reply_lower:
                return False
                
        # Simple URL hallucination check (if it starts inventing random bit.ly or internal links)
        # We allow spotify.com, but if it hallucinates weird links, we might want to block. 
        # For now, deterministic keyword blocks are safest.
        return True

    def _call_gemini(self, system_prompt: str, user_prompt: str, 
                     max_retries: int = 2) -> str:
        """Call Gemini API with fast retry logic using google-genai SDK."""
        self._init_gemini()
        from google.genai import types
        
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.gemini_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        response_mime_type="application/json",
                        response_schema=GroundedResponse,
                    ),
                )
                return response.text
            except Exception as e:
                error_str = str(e).lower()
                if 'rate' in error_str or 'quota' in error_str or '429' in error_str:
                    print(f"  Gemini API rate limited (429). Fast fallback.")
                    raise RuntimeError("Gemini API rate limit reached (429)") from e
                elif 'timeout' in error_str or 'deadline' in error_str:
                    time.sleep(1)
                else:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Gemini API failed after {max_retries} attempts: {e}") from e
                    time.sleep(1)
        
        raise RuntimeError(f"Gemini API failed after {max_retries} attempts.")
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini's structured response.
        """
        if not response_text:
            return {
                'reply': "I'm sorry, I couldn't generate a response at this moment.",
                'confidence': 0.0,
                'grounding_summary': 'Empty response from Gemini API',
                'should_escalate': True,
                'escalation_reason': 'Empty LLM response',
            }

        text = response_text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        try:
            result = json.loads(text)
            
            needs_escalation = bool(result.get('needs_escalation', False))
            
            return {
                'reply': str(result.get('reply', '')).strip(),
                'confidence': float(result.get('grounding_confidence', 0.5)),
                'grounding_summary': "Claims: " + " | ".join(result.get('grounded_claims', [])),
                'should_escalate': needs_escalation,
                'escalation_reason': 'conflicting evidence or complex request' if needs_escalation else None,
            }
        except Exception as e:
            return {
                'reply': "I'm sorry, I am having trouble processing the evidence to answer your query.",
                'confidence': 0.0,
                'grounding_summary': f'Parse error: {e}',
                'should_escalate': True,
                'escalation_reason': 'Generation parsing error',
            }
    
    def _cache_key(self, message: str, intent: str, cases: list) -> str:
        """Generate cache key from message, intent, and number of retrieved cases."""
        content = f"{message}|{intent}|{len(cases)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[dict]:
        """Get cached response."""
        if not self.cache_enabled:
            return None
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def _set_cached(self, key: str, result: dict):
        """Cache a response."""
        if not self.cache_enabled:
            return
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_path, 'w') as f:
                json.dump(result, f)
        except Exception:
            pass
