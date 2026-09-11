"""
LLM-as-Judge Evaluation Module
Uses Gemini to evaluate generated replies on multiple quality dimensions.
Uses the official google-genai SDK.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional


JUDGE_RUBRIC = """
You are an expert evaluator for customer support quality. 
Rate the following customer support response on a scale of 1-5 for each dimension.

RATING SCALE:
1 = Very Poor
2 = Poor  
3 = Acceptable
4 = Good
5 = Excellent

DIMENSIONS:
1. Relevance: Does the response address the customer's specific issue?
2. Groundedness: Is the response based on factual information rather than fabricated claims?
3. Correctness: Is the information in the response accurate?
4. Helpfulness: Does the response actually help solve the customer's problem?
5. Brand Consistency: Does the response match the brand's typical communication style?
6. Safety: Is the response free from harmful, misleading, or inappropriate content?
7. Escalation Appropriateness: If escalation was recommended, was it justified? If not, should it have been?

CUSTOMER MESSAGE:
{customer_message}

GENERATED RESPONSE:
{generated_response}

HISTORICAL BRAND RESPONSE (for reference):
{historical_response}

PREDICTED INTENT: {intent}
ESCALATION DECISION: {escalation}

Rate each dimension as a JSON object:
{{
    "relevance": {{"score": 1-5, "justification": "brief reason"}},
    "groundedness": {{"score": 1-5, "justification": "brief reason"}},
    "correctness": {{"score": 1-5, "justification": "brief reason"}},
    "helpfulness": {{"score": 1-5, "justification": "brief reason"}},
    "brand_consistency": {{"score": 1-5, "justification": "brief reason"}},
    "safety": {{"score": 1-5, "justification": "brief reason"}},
    "escalation_appropriateness": {{"score": 1-5, "justification": "brief reason"}},
    "overall_score": 1-5,
    "overall_justification": "brief overall assessment"
}}

Return ONLY the JSON object.
"""


class LLMJudge:
    """Automated evaluation using LLM-as-judge with Gemini."""
    
    DIMENSIONS = [
        'relevance', 'groundedness', 'correctness', 'helpfulness',
        'brand_consistency', 'safety', 'escalation_appropriateness'
    ]
    
    def __init__(self, model: str = 'gemini-2.0-flash'):
        self.model_name = model
        self._client = None
    
    def _init_model(self):
        """Initialize Gemini client for judging."""
        if self._client is not None:
            return
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "The LLM judge requires a valid Gemini API key. "
                "Add your key to the .env file: GEMINI_API_KEY=your_key_here"
            )
        
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "google-genai package required. Install: pip install google-genai"
            )
    
    def evaluate_single(self, customer_message: str, generated_response: str,
                        historical_response: str = "", intent: str = "",
                        escalation: str = "") -> dict:
        """Evaluate a single response using the LLM judge."""
        self._init_model()
        
        from google.genai import types
        
        prompt = JUDGE_RUBRIC.format(
            customer_message=customer_message,
            generated_response=generated_response,
            historical_response=historical_response or "Not available",
            intent=intent or "Unknown",
            escalation=escalation or "Not specified",
        )
        
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            return self._parse_judge_response(response.text)
        except Exception as e:
            print(f"  LLM Judge error: {e}")
            return self._default_scores()
    
    def evaluate_batch(self, evaluations: List[dict], 
                       delay_seconds: float = 1.0) -> List[dict]:
        """
        Evaluate a batch of responses.
        
        Args:
            evaluations: List of dicts with keys: customer_message, 
                        generated_response, historical_response, intent, escalation
            delay_seconds: Delay between API calls for rate limiting
        """
        results = []
        for i, eval_item in enumerate(evaluations):
            print(f"  Judging {i+1}/{len(evaluations)}...")
            result = self.evaluate_single(
                customer_message=eval_item.get('customer_message', ''),
                generated_response=eval_item.get('generated_response', ''),
                historical_response=eval_item.get('historical_response', ''),
                intent=eval_item.get('intent', ''),
                escalation=eval_item.get('escalation', ''),
            )
            results.append(result)
            
            if delay_seconds > 0 and i < len(evaluations) - 1:
                time.sleep(delay_seconds)
        
        return results
    
    def aggregate_scores(self, results: List[dict]) -> dict:
        """Compute aggregate scores across all evaluations."""
        import numpy as np
        
        aggregated = {}
        for dim in self.DIMENSIONS:
            scores = [r.get(dim, {}).get('score', 3) for r in results if isinstance(r.get(dim), dict)]
            if scores:
                aggregated[dim] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'n': len(scores),
                }
        
        overall = [r.get('overall_score', 3) for r in results if 'overall_score' in r]
        if overall:
            aggregated['overall'] = {
                'mean': float(np.mean(overall)),
                'std': float(np.std(overall)),
            }
        
        return aggregated
    
    def _parse_judge_response(self, text: str) -> dict:
        """Parse the judge's JSON response."""
        try:
            clean = text.strip()
            if '```json' in clean:
                clean = clean.split('```json')[1].split('```')[0].strip()
            elif '```' in clean:
                clean = clean.split('```')[1].split('```')[0].strip()
            
            return json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            return self._default_scores()
    
    def _default_scores(self) -> dict:
        """Default scores when parsing fails."""
        return {dim: {'score': 3, 'justification': 'Parse failed'} 
                for dim in self.DIMENSIONS}
