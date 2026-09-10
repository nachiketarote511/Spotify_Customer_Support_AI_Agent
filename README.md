# Hiver AI Customer Support Agent

An AI-powered customer support system that classifies incoming messages, retrieves historically similar cases, generates grounded draft replies using RAG + Gemini, and decides whether to auto-handle or escalate to a human agent.

Built for the **Hiver SDE Intern Take-Home Assignment**.

---

## Project Overview

This system processes the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset (~2.8M tweets) and builds a complete AI customer support pipeline for **SpotifyCares**.

### Key Capabilities
- **Intent Classification:** Classifies customer messages into 6 data-derived intents
- **Historical Retrieval:** FAISS-based similarity search over 21K conversation embeddings
- **RAG Generation:** Grounded reply generation using Gemini with retrieved evidence
- **Escalation Engine:** Hybrid 6-signal decision system for auto-handle vs escalate
- **Evaluation Framework:** Three-tier baseline comparison with LLM-as-judge

---

## Architecture

```
Customer Message
       ↓
┌──────────────────┐
│ Intent Classifier │ ← TF-IDF + Logistic Regression (96.4% accuracy)
│ (6 intents)       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Embedding Layer   │ ← all-MiniLM-L6-v2 (384-dim)
│                   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ FAISS Retrieval   │ ← Top-5 similar historical conversations
│ (21,253 docs)     │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ RAG Generator     │ ← Gemini 2.0 Flash with grounded context
│                   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Escalation Engine │ ← Hybrid: classifier + retrieval + rules + LLM
│ (6 signals)       │
└────────┬─────────┘
         ↓
    Final Response
    {intent, confidence, reply, should_escalate, reason}
```

---

## Dataset

| Metric | Value |
|--------|-------|
| Source | Customer Support on Twitter (Kaggle) |
| Total tweets | ~2.8M |
| Selected brand | **SpotifyCares** |
| Reconstructed conversations | 21,453 |
| Training set | 21,253 |
| Golden evaluation set | 200 |

---

## Brand Selection

Brand selected automatically using a scoring algorithm that evaluates:
- Conversation volume (inbound + outbound)
- Conversation completeness (% reconstructable)
- Intent diversity
- Historical resolution quality

**SpotifyCares** scored **0.856** — highest among 101 brands.

Full analysis: `reports/brand_selection.json`

---

## Data Processing

### Pipeline A — Lightweight (Pandas)
```bash
python -m src.pipelines.pandas_pipeline
```
Processes locally in seconds. Used for quick evaluation.

### Pipeline B — Large Scale (PySpark + dbt)
```bash
# Requires Spark cluster
spark-submit src/pipelines/spark_pipeline.py
```
Architectural specification demonstrating scalable processing. Produces the same data contract as Pipeline A.

### Dataset Size Detection
```bash
python -c "from src.pipelines.pipeline_selector import DatasetAnalyzer; DatasetAnalyzer().analyze('dataset/twcs/twcs.csv')"
```
Classifies datasets as SMALL or LARGE and recommends the appropriate pipeline.

---

## Intent Taxonomy

6 intents discovered from actual SpotifyCares data:

| Intent | Count | % | Description |
|--------|-------|---|-------------|
| playback_issues | 14,576 | 68.4% | Playback, streaming, audio |
| billing_payment | 3,301 | 15.5% | Subscription, payment, family plans |
| app_technical | 1,601 | 7.5% | App crashes, versions, devices |
| information_request | 807 | 3.8% | Confirmations, thank-you |
| account_access | 586 | 2.7% | Login, password, Facebook auth |
| playlist_library | 437 | 2.0% | Playlist management |

Full taxonomy: `reports/intent_taxonomy.json`

---

## RAG Pipeline

The generation pipeline explicitly grounds responses in historical evidence:

1. Customer message → Intent classification
2. Message embedding → FAISS retrieval (top-5 similar cases)
3. Retrieved cases + intent + message → Grounded prompt
4. Gemini generates structured JSON response
5. Escalation engine evaluates confidence signals

**Critical constraints enforced:**
- No invented policies, refunds, or guarantees
- Responses grounded in actual SpotifyCares historical behavior
- Automatic escalation when evidence is insufficient

---

## Escalation Logic

The escalation engine uses 6 weighted signals:

| Signal | Weight | Threshold |
|--------|--------|-----------|
| Low intent confidence | 0.25 | <0.4 |
| Low retrieval quality | 0.20 | <0.3 |
| Sensitive content | 0.20 | keyword match |
| Ambiguous intent | 0.15 | margin <0.1 |
| Conflicting evidence | 0.10 | response length variance |
| High emotion | 0.10 | keyword match |

Escalation triggered when weighted score ≥ 0.5.

---

## Gemini API Setup

### 1. Create the environment file

```bash
cp .env.example .env
```

### 2. Add your API key

Open `.env` and replace the placeholder:

```
GEMINI_API_KEY=your_real_gemini_api_key
```

Get a free key from [Google AI Studio](https://aistudio.google.com/).

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs the official `google-genai` SDK (the current Gemini Python client).

### 4. Run the application

```bash
python api/main.py
```

The API loads `.env` automatically at startup. Gemini is called lazily — only when a customer message reaches the RAG generation endpoint.

### 5. Verify Gemini integration

```bash
python scripts/test_gemini_integration.py
```

This sends a single test message through the complete pipeline:

```
Customer message → Intent classification → FAISS retrieval → Gemini generation → Escalation decision → LLM Judge
```

### Components that use Gemini

| Component | File | Purpose |
|-----------|------|---------|
| RAG Generator | `src/rag/generator.py` | Generates grounded customer support replies |
| LLM Judge | `src/evaluation/llm_judge.py` | Evaluates reply quality on 7 dimensions |
| Escalation Engine | `src/escalation/decision.py` | Incorporates Gemini's escalation recommendation as one of 6 signals |

### Security

- The API key is loaded exclusively from the `GEMINI_API_KEY` environment variable
- `.env` is in `.gitignore` and is **never** committed to version control
- The key is never printed, logged, serialized to reports, or included in cached responses
- If `GEMINI_API_KEY` is missing, the system raises a clear error — it does **not** silently fall back to fake responses

---

## Quick Start

### Prerequisites
- Python 3.10+
- ~2GB disk space (for models)

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd Customer_Support_AI_Agent

# Install dependencies
pip install -r requirements.txt

# Set up Gemini API key
cp .env.example .env
# Edit .env with your API key
```

### Training (if models not present)
```bash
python scripts/train.py
```
Runs Phases 7-10: intent discovery → classifier training → embeddings → FAISS index.

### Start the API
```bash
python api/main.py
```
Opens at `http://localhost:8000`. UI at `http://localhost:8000/`.

---

## Quick Evaluation

**Reproduces headline results in under 15 minutes.**

```bash
python scripts/run_quick_eval.py
```

This script:
1. Loads prepared sample data and trained models
2. Runs intent classification on golden evaluation set
3. Evaluates against baselines
4. Reports metrics

**No Gemini API calls needed for basic evaluation.** RAG evaluation requires a GEMINI_API_KEY.

---

## Full Pipeline

```bash
# Phase 1-2: Dataset audit and brand selection
python scripts/audit_dataset.py

# Phase 3-6: Conversation reconstruction + pipeline processing
python -m src.pipelines.pandas_pipeline

# Phase 7-10: Intent discovery + classifier + embeddings + FAISS
python scripts/train.py

# Phase 11-16: RAG + evaluation (requires GEMINI_API_KEY)
python scripts/run_quick_eval.py

# Start API
python api/main.py
```

---

## Results

### Intent Classification

| System | Accuracy | Macro F1 |
|--------|----------|----------|
| Baseline (TF-IDF + LR) | **96.42%** | **93.62%** |
| Improved (SentenceTransformer + LR) | 75.96% | 63.48% |

### Key Insight
The TF-IDF baseline outperforms the Sentence Transformer model because intent labels were derived from TF-IDF clustering — the classifier and labels share the same information source. This is an honest limitation documented in the evaluation report.

---

## Baselines

| Baseline | Description |
|----------|-------------|
| Trivial | Always replies: "Thank you for reaching out! Please DM us..." |
| Simple | TF-IDF cosine similarity → nearest historical response |
| Final System | Intent → FAISS retrieval → Gemini RAG generation |

All three systems evaluated on the same 200-example golden set with the same metrics.

---

## Failure Analysis

Top 5 failure modes identified from real evaluation examples:

1. **Majority class dominance** — playlist issues misclassified as playback_issues
2. **Multi-intent messages** — single-label classifier drops secondary intents
3. **Thank-you misclassification** — "thanks" pattern captures messages with new issues
4. **Context loss** — follow-up messages classified without conversation history
5. **Cross-intent retrieval** — embedding similarity doesn't guarantee intent alignment

Full analysis: `reports/evaluation_report.md`

---

## Golden Evaluation Set

- **200 examples** with stratified sampling across intents
- **No train/evaluation leakage** — conversation-level splitting
- Machine-generated intent suggestions clearly distinguished from human labels
- Annotation template supports human labeling workflow

Location: `data/golden/golden_eval_set.csv`

---

## LLM Judge

Automated evaluation using Gemini on 7 dimensions:
1. Relevance
2. Groundedness
3. Correctness
4. Helpfulness
5. Brand Consistency
6. Safety
7. Escalation Appropriateness

Fixed 1-5 rubric. Results aggregated with mean, std, min, max.

---

## Human Agreement

Framework implemented in `src/evaluation/human_agreement.py`:
- Exact agreement rate
- Near agreement (within 1 point)
- Mean Absolute Error
- Pearson & Spearman correlation

Annotation template created; human labels to be added by annotators.

---

## Decision Log

14 non-obvious engineering decisions documented in `reports/decision_log.md`:
- Why SpotifyCares? (highest completeness score)
- Why data-derived intents? (assignment requirement)
- Why TF-IDF baseline? (interpretable, reproducible)
- Why FAISS flat index? (exact search at 21K scale)
- Why hybrid escalation? (LLM alone is unreliable)
- And 9 more...

---

## Limitations

1. Single-label classification cannot handle multi-intent messages
2. No conversation state tracking — each message treated independently
3. Gemini dependency — degrades without API access
4. 2017 data — product and policies have evolved
5. LLM judge uses same model family — potential self-reinforcing bias
6. 96.4% headline is inflated by circular label-classifier relationship

---

## Project Structure

```
Customer_Support_AI_Agent/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
│
├── configs/
│   ├── config.yaml          # Main configuration
│   ├── intents.yaml          # Intent taxonomy
│   └── thresholds.yaml       # Escalation thresholds
│
├── data/
│   ├── raw/                  # Raw data samples
│   ├── processed/            # Processed conversations
│   ├── golden/               # Golden evaluation set (200 examples)
│   └── schemas/              # Data contract
│
├── src/
│   ├── data/                 # Conversation builder, validation
│   ├── pipelines/            # Pandas + Spark + selector
│   ├── intent/               # Discovery, training, prediction
│   ├── retrieval/            # Embeddings, FAISS index, retriever
│   ├── rag/                  # Prompt construction, generator
│   ├── escalation/           # Hybrid decision engine
│   └── evaluation/           # Metrics, baselines, LLM judge, human agreement
│
├── dbt/                      # dbt models for Pipeline B
├── api/                      # FastAPI application
├── ui/                       # Web interface
├── scripts/                  # Training, evaluation, audit scripts
├── tests/                    # Unit tests
├── models/                   # Trained models and indices
└── reports/                  # Evaluation report, decision log, figures
```

---

## Future Work

1. Multi-label intent classification
2. Conversation state tracking across turns
3. Intent-filtered FAISS retrieval
4. Active learning loop for golden set annotation
5. A/B evaluation framework
6. Docker containerization
7. Monitoring and drift detection
