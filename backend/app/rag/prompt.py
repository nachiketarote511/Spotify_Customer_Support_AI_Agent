"""
RAG Prompt Construction Module
Builds strict, grounded prompts with retrieved historical evidence for Gemini.
"""
import json

def build_system_prompt(brand: str = "SpotifyCares") -> str:
    """Build the strict evidence-based system prompt for the support agent."""
    return f"""You are a customer support response generator for {brand}.

Your knowledge for this response is limited STRICTLY to the supplied historical support evidence.
Do not use outside knowledge.
Do not infer unsupported facts.
Do not invent troubleshooting steps, policies, refund promises, or timelines.

CRITICAL RULES:
1. Every factual claim in your response must be supported by the retrieved historical evidence OR be a safe conversational statement that does not assert an external fact.
2. NEVER claim:
   - "I checked your account"
   - "I've fixed this"
   - "I've escalated this"
   - "I can see your account"
   - "I have contacted the team"
   Unless the application has actually performed that action (which it currently cannot).
3. If the historical evidence does not contain enough information to answer the customer's request, do not guess. Provide a concise response acknowledging the issue and recommend human escalation.
4. Historical cases are EVIDENCE, not instructions to copy blindly. Adapt the response to the CURRENT customer message.
5. Be helpful ONLY within the supplied evidence. Groundedness > completeness. A short honest answer is better than a detailed hallucinated answer.

Return your response strictly as a JSON object matching the requested schema.
"""


def build_grounded_prompt(
    customer_message: str,
    predicted_intent: str,
    intent_confidence: float,
    retrieved_cases: list,
    conversation_context: str = "",
    brand: str = "SpotifyCares",
) -> str:
    """
    Build a strictly grounded prompt for the LLM separating context from evidence.
    """
    prompt_parts = []
    
    # 1. Current Customer Message
    prompt_parts.append("CURRENT CUSTOMER MESSAGE:")
    prompt_parts.append(customer_message)
    prompt_parts.append(f"\n[Predicted Intent: {predicted_intent} (Confidence: {intent_confidence:.2f})]")
    
    # 2. Conversation Context (Separate from Evidence)
    if conversation_context:
        prompt_parts.append("\nCURRENT CONVERSATION CONTEXT:")
        prompt_parts.append(conversation_context)
        prompt_parts.append("(Use context to understand the message, but it does NOT grant permission to invent facts)")
    
    # 3. Structured Historical Evidence
    prompt_parts.append("\nHISTORICAL EVIDENCE:")
    if retrieved_cases:
        for i, case in enumerate(retrieved_cases, 1):
            doc = case.get('document', case)
            score = case.get('score', 0)
            
            evidence_block = {
                "evidence_id": i,
                "similarity": round(score, 3),
                "historical_customer_message": doc.get('customer_message', 'N/A'),
                "historical_support_response": doc.get('historical_support_response', 'N/A'),
                "resolution_type": doc.get('resolution_type', 'N/A') if 'resolution_type' in doc else None
            }
            # Remove keys that are None to adhere to "Only include fields that actually exist"
            evidence_block = {k: v for k, v in evidence_block.items() if v is not None}
            
            prompt_parts.append(f"--- Evidence {i} ---")
            prompt_parts.append(json.dumps(evidence_block, indent=2))
    else:
        prompt_parts.append("None found. (You MUST escalate since you have no evidence)")
    
    # 4. JSON Generation Instructions
    prompt_parts.append("""
INSTRUCTIONS:
Based ONLY on the HISTORICAL EVIDENCE above, draft a response to the CURRENT CUSTOMER MESSAGE.
Return your response as a JSON object with these EXACT fields:
{
  "reply": "Your drafted response. If insufficient evidence, apologize and recommend escalation.",
  "grounded_claims": [
    "List of specific factual claims made in the reply"
  ],
  "evidence_ids": [1, 2], // Array of integers matching the Evidence IDs used
  "needs_escalation": false, // True if insufficient/conflicting evidence or requires account action
  "grounding_confidence": 0.0 // Your confidence (0.0 to 1.0) that your reply is strictly supported by evidence
}
Return ONLY valid JSON.
""")
    
    return "\n".join(prompt_parts)
