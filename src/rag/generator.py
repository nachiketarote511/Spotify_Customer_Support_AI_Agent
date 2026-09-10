"""
RAG Generator Module
Orchestrates the full RAG pipeline: Intent → Retrieval → Gemini → Response.
Uses the official google-genai SDK.
"""

import os
import json
import hashlib
import time
from typing import Optional, Dict, Any

from src.rag.prompt import build_system_prompt, build_grounded_prompt


class RAGGenerator:
    """Full RAG pipeline for generating grounded customer support responses."""
    
    def __init__(self, config: dict = None):
        if config is None:
            import yaml
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
            )
            with open(config_path) as f:
                config = yaml.safe_load(f)
        
        self.config = config
        self.gemini_model = config.get('gemini', {}).get('model', 'gemini-2.0-flash')
        self.temperature = config.get('gemini', {}).get('temperature', 0.3)
        self.max_tokens = config.get('gemini', {}).get('max_output_tokens', 512)
        
        # Cache for responses
        self.cache_enabled = config.get('cache', {}).get('enabled', True)
        self.cache_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 
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
        
        Args:
            customer_message: Customer's message
            predicted_intent: Predicted intent category
            intent_confidence: Confidence score
            retrieved_cases: List of retrieved historical cases
            conversation_context: Prior conversation context
            brand: Brand name
        
        Returns:
            dict with keys: reply, confidence, grounding_summary, 
            should_escalate, escalation_reason
        """
        # Check cache first
        cache_key = self._cache_key(customer_message, predicted_intent, retrieved_cases)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Build prompts
        system_prompt = build_system_prompt(brand)
        user_prompt = build_grounded_prompt(
            customer_message, predicted_intent, intent_confidence,
            retrieved_cases, conversation_context, brand
        )
        
        # Call Gemini with fallback
        try:
            response = self._call_gemini(system_prompt, user_prompt)
            result = self._parse_response(response)
        except Exception as e:
            print(f"  Gemini API call fallback triggered: {e}")
            # Instant Grounded Fallback from top retrieved case
            if retrieved_cases:
                top_item = retrieved_cases[0]
                top_doc = top_item.get('document', top_item)
                fallback_reply = top_doc.get('historical_support_response', '')
                if fallback_reply:
                    result = {
                        'reply': fallback_reply,
                        'confidence': 0.75,
                        'grounding_summary': f"Grounded response derived from historical case {top_doc.get('conversation_id', '')}",
                        'should_escalate': False,
                        'escalation_reason': None,
                    }
                else:
                    result = {
                        'reply': "Thanks for reaching out! Could you please share your device type and OS version so we can assist further?",
                        'confidence': 0.5,
                        'grounding_summary': 'Standard support inquiry template',
                        'should_escalate': True,
                        'escalation_reason': 'API rate limited — fallback inquiry',
                    }
            else:
                result = {
                    'reply': "Thanks for reaching out! Could you please share your device type and OS version so we can assist further?",
                    'confidence': 0.5,
                    'grounding_summary': 'Standard support inquiry template',
                    'should_escalate': True,
                    'escalation_reason': 'API rate limited — fallback inquiry',
                }
        
        # Cache result
        if self.cache_enabled:
            self._set_cached(cache_key, result)
        
        return result
    
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
                    ),
                )
                return response.text
            except Exception as e:
                error_str = str(e).lower()
                if 'rate' in error_str or 'quota' in error_str or '429' in error_str:
                    print(f"  Gemini API rate limited (429). Fast fallback.")
                    raise RuntimeError("Gemini API rate limit reached (429)") from e
                elif 'timeout' in error_str or 'deadline' in error_str:
                    print(f"  Gemini API timeout. Retrying...")
                    time.sleep(1)
                elif 'invalid' in error_str and 'key' in error_str:
                    raise ValueError(
                        "Invalid GEMINI_API_KEY. Check that your key is correct "
                        "and has access to the Gemini API."
                    ) from e
                elif 'not found' in error_str and ('model' in error_str or '404' in error_str):
                    raise ValueError(
                        f"Gemini model '{self.gemini_model}' not found or not available. "
                        f"Check configs/config.yaml gemini.model setting."
                    ) from e
                else:
                    print(f"  Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Gemini API failed after {max_retries} attempts: {e}"
                        ) from e
                    time.sleep(1)
        
        raise RuntimeError(f"Gemini API failed after {max_retries} attempts.")
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini's response into structured output.
        Handles valid JSON, truncated JSON, stringified JSON, and plain text.
        """
        if not response_text:
            return {
                'reply': "I'm sorry, I couldn't generate a response at this moment.",
                'confidence': 0.3,
                'grounding_summary': 'Empty response from Gemini API',
                'should_escalate': True,
                'escalation_reason': 'Empty LLM response',
            }

        text = response_text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        # Step 1: Attempt standard JSON parsing
        try:
            result = json.loads(text)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    pass
            
            if isinstance(result, dict):
                reply_val = str(result.get('reply', '')).strip()
                # Check if reply_val itself is stringified JSON
                if reply_val.startswith('{') and '"reply"' in reply_val:
                    try:
                        inner = json.loads(reply_val)
                        if isinstance(inner, dict) and 'reply' in inner:
                            reply_val = str(inner['reply']).strip()
                    except Exception:
                        pass
                
                return {
                    'reply': reply_val,
                    'confidence': float(result.get('confidence', 0.5)),
                    'grounding_summary': str(result.get('grounding_summary', '')),
                    'should_escalate': bool(result.get('should_escalate', False)),
                    'escalation_reason': result.get('escalation_reason'),
                }
        except Exception:
            pass

        # Step 2: Regex extraction for truncated/incomplete JSON
        import re
        reply_match = re.search(r'"reply"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text, re.DOTALL)
        if reply_match:
            extracted_reply = reply_match.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
            
            grounding_match = re.search(r'"grounding_summary"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text, re.DOTALL)
            grounding = grounding_match.group(1).replace('\\"', '"') if grounding_match else 'Grounded in SpotifyCares resolution patterns'
            
            return {
                'reply': extracted_reply,
                'confidence': 0.7,
                'grounding_summary': grounding,
                'should_escalate': False,
                'escalation_reason': None,
            }

        # Step 3: Strip JSON formatting remnants if unparseable
        clean_text = re.sub(r'^\s*\{\s*"reply"\s*:\s*"?', '', text)
        clean_text = re.sub(r'"\s*,\s*"confidence".*$', '', clean_text, flags=re.DOTALL)
        clean_text = clean_text.strip().strip('"').strip('}').strip()

        return {
            'reply': clean_text if clean_text else text[:300],
            'confidence': 0.5,
            'grounding_summary': 'Formatted from generation output',
            'should_escalate': False,
            'escalation_reason': None,
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
