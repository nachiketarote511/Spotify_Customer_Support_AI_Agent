# AI Customer Support Agent

## Problem
Modern customer support teams handle massive volumes of repetitive queries. Relying solely on human agents for generic issues is slow and expensive, while relying on rigid rule-based chatbots leads to poor customer experiences.

## Solution
This project implements an AI-powered customer support agent that classifies incoming customer messages by intent, retrieves historically similar resolved cases, and generates a contextually grounded draft reply using a Large Language Model. A hybrid escalation engine acts as a safety gate, deciding whether the draft should be sent automatically or escalated to a human agent.

## Selected Brand
**SpotifyCares** was selected as the brand for this agent. From the Kaggle Twitter Customer Support Dataset, SpotifyCares provided the optimal balance of data volume (43K outbound, 27K inbound tweets) and conversation completeness (88.7%). This ensures the agent is trained on realistic, coherent conversational data.

## Architecture

```text
Customer
 ↓
FastAPI
 ↓
Conversation Context
 ↓
Intent Classifier
 ↓
FAISS Retrieval
 ↓
Evidence Gate
 ↓
Gemini RAG
 ↓
Grounding Validation
 ↓
Escalation Engine
 ↓
Final Response
```

## Data Pipeline

### Pandas + NumPy
The primary small-data pipeline is implemented using Pandas and NumPy. It processes ~2.8M raw tweets down to 21K reconstructed SpotifyCares conversations in seconds. It handles data cleaning, thread reconstruction, and train/test splitting locally.

### PySpark
An architectural specification of a large-data pipeline using PySpark (and logically equivalent to the Pandas pipeline). While the dataset size doesn't strictly necessitate distributed processing, this pipeline demonstrates how the data transformations scale to a cluster environment. Both pipelines produce the exact same logical data contract (`ConversationRecord`).

## Intent Classification
Intent classification uses a TF-IDF vectorizer (5000 features) combined with Logistic Regression. The 6 supported intents (e.g., `playback_issues`, `billing_payment`) were not predefined but were data-derived via unsupervised clustering (TF-IDF + SVD + KMeans) on actual SpotifyCares messages. The model outputs a predicted intent and a confidence score used for escalation.

## Historical Retrieval
The retrieval system uses Sentence Transformers (`all-MiniLM-L6-v2`) and a FAISS `IndexFlatIP` index to search the historical dataset. Given a customer's message, it retrieves the top-5 most semantically similar historical conversations. An evidence filtering step ensures that only cases matching the predicted intent are heavily weighted.

## RAG Response Generation
Draft responses are generated using Gemini (`gemini-2.0-flash`). To prevent hallucination, the generation is strictly grounded. The prompt includes the customer's message, conversation history, the predicted intent, and the 5 retrieved historical support responses. Gemini is instructed to adopt the SpotifyCares tone and extract actionable solutions from the retrieved cases.

## Escalation
The `EscalationEngine` enforces deterministic precedence over the LLM. It computes an escalation score based on 6 signals:
1. Low intent confidence
2. Ambiguous multi-intent probabilities
3. Poor retrieval quality
4. Sensitive or safety-critical keywords
5. High customer emotion
6. LLM-suggested escalation
If any critical threshold is breached, the engine overrides with `ESCALATE_TO_HUMAN` and provides a transparent escalation reason.

## Evaluation
The system is evaluated on multiple dimensions:
- **Golden Set:** A holdout set for rigorous evaluation.
- **Automated Metrics:** Accuracy, Macro F1 for Intent Classification; Recall@K for Retrieval.
- **Baselines:** Comparison against a trivial baseline and a simple TF-IDF retrieval baseline.
- **LLM Judge:** A fixed Gemini-based rubric evaluating Relevance, Groundedness, Correctness, Helpfulness, Brand Consistency, Safety, and Escalation Appropriateness.
- **Human Evaluation:** An annotation UI exists for human verification.

## Golden Evaluation Set
200 real examples were sampled and AI-assisted draft labels were generated. Human review is tracked separately. The sampling is stratified at the conversation level to prevent data leakage and ensure coverage across rare intents.

## Results
- **Intent Classification:** 96.4% Accuracy, 93.6% Macro F1. (See "Misleading Headline Number").
- **Retrieval:** Average FAISS retrieval latency <50ms.
- **Generation:** Successfully replicates SpotifyCares tone and avoids hallucinating company policies, constrained by the top-5 retrieved contexts.

## Baselines
1. **Trivial Baseline:** Always replies "Thank you for reaching out! Please DM us with your details and we'll look into this." (0% specific problem resolution).
2. **Simple Baseline (TF-IDF Retrieval):** Directly returns the historical response of the closest TF-IDF match without LLM synthesis. Often fails on context shifts.
3. **Final RAG System:** Outperforms both by synthesizing a tailored response grounded in multiple past cases while evaluating when to escalate.

## Failure Analysis
Top 5 actual failure modes with real examples and hypotheses:
1. **Majority Class Dominance:** "My playlist disappeared" → Predicted `playback_issues`. Hypothesis: The classifier over-indexes on music vocabulary and defaults to the 68% majority class.
2. **Ambiguous Multi-Intent:** "I can't log in and my premium subscription was charged twice" → Predicted `billing_payment`. Hypothesis: Single-label classifier loses the `account_access` component.
3. **Thank-You Misclassification:** "Thanks, that worked! But now..." → Predicted `information_request`. Hypothesis: Lexical bias towards the word "thanks".
4. **Device-Specific Context Loss:** Customer replies "Spotify 8.4.25 and iOS 11". Hypothesis: Single-turn classification misses the fact this is an answer to a previous support question.
5. **Retrieval Brand Tone Mismatch:** Embedding matches on generic vocabulary rather than problem-specific intent, retrieving irrelevant historical solutions.

## What Is Misleading About My Headline Number?
The 96.4% intent classification accuracy is misleading. The intents were discovered via TF-IDF clustering, and the baseline classifier uses TF-IDF features. The classifier is essentially just learning to reproduce the same patterns that created the labels, leading to circular agreement. Additionally, the heavy class imbalance (68% `playback_issues`) inflates accuracy. Real-world performance on novel queries would be significantly lower.

## What I Would Do Next Week
1. Complete true human annotation of the 200-example golden set with multiple annotators to measure inter-annotator agreement.
2. Implement Intent-Filtered Retrieval (restrict FAISS to search only within the predicted intent).
3. Shift from single-turn to multi-turn contextual intent classification.
4. Apply class-weighted loss or SMOTE to handle the heavy label imbalance.

## Decision Log
Reference `reports/decision_log.md` for a detailed log of 14 engineering decisions, including alternatives considered and reasons for rejection.

## Project Structure
```text
.
├── backend/
│   ├── app/           # FastAPI application, Pipelines, RAG, Intent, Escalation
│   ├── data/          # Processed data, Golden set, Annotations
│   └── scripts/       # Evaluation, Setup and Utilities
├── configs/           # YAML configuration files
├── dataset/           # Raw Kaggle CSVs
├── docs/              # Architecture diagrams and documentation
├── frontend/          # Fast Human Annotation UI (HTML/JS)
├── models/            # Serialized Intent classifiers and FAISS indices
├── reports/           # Decision log, Audit, Evaluation reports
├── README.md          # Project overview
├── docker-compose.yml # Container definitions (if applicable)
└── test_*.py          # Pytest suite
```

## Quick Start
1. **Clone & Setup:**
   ```bash
   git clone <repo_url>
   cd Customer_Support_AI_Agent
   python -m venv venv
   source venv/bin/activate  # Or venv\Scripts\activate on Windows
   pip install -r backend/requirements.txt
   ```
2. **Environment Setup:**
   Copy `.env.example` to `.env` and insert your Gemini API Key.
   ```bash
   cp .env.example .env
   ```
3. **Start the API & Annotation UI:**
   ```bash
   python backend/app/api/main.py
   ```
   Navigate to `http://localhost:8000/ui/annotation.html`

## Quick Evaluation
To reproduce the headline evaluation (under 15 minutes, largely due to LLM caching):
```bash
python backend/scripts/run_eval.py
```
*(If run_eval.py does not exist, use `pytest test_pipeline.py` or the provided FastAPI `/evaluate` endpoint).*

## Environment Variables
The following environment variables are required in `.env`:
- `GEMINI_API_KEY`: Required for Gemini 2.0 Flash text generation.
- `HF_MODEL_REPO_ID`: (Optional) Hugging Face repo ID for downloading model artifacts if not generated locally.

## Model Artifacts
Model artifacts (TF-IDF vectorizer, Logistic Regression weights, FAISS index) are tracked and can be loaded from Hugging Face or built locally from the processed Pandas pipeline output. The embeddings use `sentence-transformers/all-MiniLM-L6-v2`.

## Limitations
- **Dataset Limitations:** Based on static 2017 Twitter data. Policies and product features have evolved.
- **Golden Set Limitations:** Currently uses machine-suggested labels. True human inter-annotator agreement is pending.
- **LLM Judge Limitations:** Evaluated using Gemini, which introduces potential self-reinforcing bias since the generator is also Gemini.
- **Grounding Limitations:** FAISS relies on exact word/semantic matches. Conflicting historical agent responses can confuse the RAG context.
- **Action Execution:** The system drafts text responses but lacks real customer backend/account access to actually process refunds or account changes.
