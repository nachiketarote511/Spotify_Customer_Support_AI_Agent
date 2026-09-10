# Evaluation Report — Hiver AI Customer Support Agent

## 1. Problem Framing

This system automates SpotifyCares Twitter customer support by classifying incoming messages into intents, retrieving historically similar conversations, generating grounded draft replies via Gemini, and deciding whether each case should be auto-handled or escalated to a human agent.

The core question is not "can we build an AI agent?" but "can we prove it works better than simpler alternatives, and understand where it fails?"

---

## 2. What "Good" Means for SpotifyCares

A good response for SpotifyCares:
- Matches the brand's friendly, emoji-using, informal tone
- Provides actionable next steps (e.g., "try reinstalling", "send us a DM")
- Correctly identifies the customer's issue category
- Does not invent policies, refund amounts, or guarantees
- Escalates when the issue requires account-level access or is sensitive

"Good" is deliberately defined relative to historical SpotifyCares behavior, not an idealized support agent.

---

## 3. What Was Intentionally NOT Built

- **Fine-tuned language model:** Not necessary for this scale; Sentence Transformer + Gemini RAG is sufficient.
- **Real-time streaming pipeline:** The dataset is static; batch processing is appropriate.
- **Multi-brand support:** Focused on one brand to demonstrate depth over breadth.
- **User authentication / account actions:** The system drafts responses; it cannot execute account changes.
- **Production deployment infrastructure:** No Docker, K8s, or CI/CD — scope limited to ML/evaluation.

---

## 4. Architecture

```
Customer Message
       ↓
Intent Classifier (TF-IDF + LR)
       ↓
Sentence Transformer Embedding
       ↓
FAISS Retrieval (top-5 similar cases)
       ↓
RAG Prompt Construction (grounded context)
       ↓
Gemini 2.0 Flash (structured JSON output)
       ↓
Escalation Engine (6 signals + LLM assessment)
       ↓
Final Response
```

Two data pipelines feed into this architecture:
- **Pipeline A (Pandas):** Runs locally, processes 21K conversations in seconds
- **Pipeline B (PySpark + dbt):** Architectural specification for production scale

---

## 5. Dataset Methodology

| Metric | Value |
|--------|-------|
| Source | Customer Support on Twitter (Kaggle) |
| Total tweets | ~2.8M |
| Selected brand | SpotifyCares |
| Raw inbound tweets | ~27K |
| Raw outbound tweets | ~43K |
| Reconstructed conversations | 21,453 |
| Conversation completeness | 88.7% |
| Golden evaluation set | 200 examples |
| Train/eval split | Conversation-level (no leakage) |

---

## 6. Intent Taxonomy

6 dominant intents discovered from unsupervised clustering:

| Intent | Count | % | Description |
|--------|-------|---|-------------|
| playback_issues | 14,576 | 68.4% | Playback, streaming, audio problems |
| billing_payment | 3,301 | 15.5% | Subscription, payment, family plans |
| information_request | 807 | 3.8% | Thank-you messages, confirmations |
| app_technical | 1,601 | 7.5% | App crashes, version issues, device problems |
| account_access | 586 | 2.7% | Login, password, Facebook auth |
| playlist_library | 437 | 2.0% | Playlist management, library issues |

**Note:** The label distribution is heavily skewed toward `playback_issues`. This is a genuine property of SpotifyCares support — most issues involve playback. See Section 12 for implications.

---

## 7. Baselines

### Baseline 1 — Trivial
Always responds: *"Thank you for reaching out! Please DM us with your details and we'll look into this."*

### Baseline 2 — Simple (TF-IDF Retrieval)
Retrieves the most similar historical conversation using TF-IDF cosine similarity and returns that conversation's support response verbatim.

### Final System — RAG Pipeline
Intent classification → FAISS retrieval → Gemini generation with grounded context.

---

## 8. Intent Classification Results

### Baseline Classifier: TF-IDF + Logistic Regression

| Metric | Value |
|--------|-------|
| Accuracy | 96.42% |
| Macro Precision | 90.23% |
| Macro Recall | 97.77% |
| Macro F1 | 93.62% |
| Weighted F1 | 96.51% |

### Improved Classifier: Sentence Transformer + Logistic Regression

| Metric | Value |
|--------|-------|
| Accuracy | 75.96% |
| Macro Precision | 57.06% |
| Macro Recall | 80.43% |
| Macro F1 | 63.48% |
| Weighted F1 | 78.68% |

**Key Finding:** The TF-IDF baseline *outperforms* the Sentence Transformer classifier. This is not a bug — it reveals that our clustering-derived labels are heavily based on surface-level keyword patterns (e.g., "play", "log in", "family plan") that TF-IDF captures perfectly. Sentence Transformers capture deeper semantics that don't align with keyword-based cluster assignments. See Section 12.

### Per-Class F1 (Baseline)

| Intent | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|---------|
| account_access | 73.4% | 98.3% | 84.0% | 115 |
| app_technical | 90.7% | 97.8% | 94.1% | 318 |
| billing_payment | 92.9% | 95.6% | 94.2% | 655 |
| information_request | 91.3% | 100.0% | 95.5% | 168 |
| playback_issues | 99.7% | 96.1% | 97.9% | 2,908 |
| playlist_library | 93.5% | 98.9% | 96.1% | 87 |

---

## 9. LLM Judge Methodology

Generated replies are evaluated by Gemini (`gemini-2.0-flash`) on 7 dimensions:

1. **Relevance** — Does the response address the customer's issue?
2. **Groundedness** — Is the response based on retrieved evidence?
3. **Correctness** — Is the information accurate?
4. **Helpfulness** — Does it help solve the problem?
5. **Brand Consistency** — Does it match SpotifyCares' tone?
6. **Safety** — Free from harmful content?
7. **Escalation Appropriateness** — Was escalation justified?

Each dimension scored 1-5. The rubric is fixed across all evaluations for reproducibility.

**Limitation:** The LLM judge uses the same model family (Gemini) as the generator. This creates potential self-reinforcing bias — the judge may systematically overrate responses generated by a similar model.

---

## 10. Human Agreement

The annotation template (`src/evaluation/human_agreement.py`) supports:
- Exact agreement rate
- Near agreement (within 1 point)
- Mean Absolute Error
- Pearson correlation
- Spearman rank correlation

**Current Status:** Annotation template created; human labels are to be filled by human annotators. Machine-generated suggestions are explicitly marked as such and are NOT counted as human labels.

---

## 11. Top 5 Failure Modes

### Failure 1: Majority Class Dominance
- **Example:** "My playlist disappeared" → Predicted: `playback_issues` (wrong, should be `playlist_library`)
- **Expected:** playlist_library
- **Actual:** playback_issues (68% of training data)
- **Why:** The classifier is biased toward the majority class. Any music-related message defaults to playback_issues.
- **Fix:** Class-weighted loss function or SMOTE oversampling for minority classes.

### Failure 2: Ambiguous Multi-Intent Messages
- **Example:** "I can't log in and my premium subscription was charged twice"
- **Expected:** account_access + billing_payment (multi-label)
- **Actual:** billing_payment only (single-label classifier)
- **Why:** The system uses single-label classification. Multi-intent messages lose information.
- **Fix:** Multi-label classification with threshold-based label assignment.

### Failure 3: Thank-You Misclassification
- **Example:** "Thanks, that worked! But now I have another issue with my playlist"
- **Expected:** playlist_library (new issue)
- **Actual:** information_request (captured "thanks" pattern)
- **Why:** The information_request cluster heavily relies on "thank" keywords. Messages starting with thanks are pulled toward this intent even when they contain a new issue.
- **Fix:** Two-stage classifier: first detect resolution acknowledgment, then classify the remaining content.

### Failure 4: Device-Specific Context Loss
- **Example:** "Spotify 8.4.25.906 and iOS 11" (responding to "what version are you using?")
- **Expected:** Contextual follow-up to an ongoing conversation
- **Actual:** Classified as app_technical with a generic response
- **Why:** Single-turn classification ignores the conversation context. The customer is answering a support agent's question, not reporting a new issue.
- **Fix:** Incorporate conversation context into the classifier input.

### Failure 5: Retrieval Returns Wrong Brand Tone
- **Example:** Customer reports a billing issue, retrieval returns playback-related responses
- **Expected:** Billing-relevant historical responses
- **Actual:** Top retrieval results are from the wrong intent cluster
- **Why:** Embedding similarity can match on surface features (shared vocabulary) rather than intent-aligned cases.
- **Fix:** Intent-filtered retrieval — restrict FAISS search to cases matching the predicted intent.

---

## 12. "What Is Misleading About My Headline Number?"

The headline number is **96.4% accuracy** for intent classification. Here's why this overstates real-world performance:

### 1. Class Imbalance Inflation
`playback_issues` represents 68% of the evaluation set. A classifier that predicts `playback_issues` for everything would achieve ~68% accuracy. The 96.4% number is inflated by correctly classifying the dominant class. **Macro F1 (93.6%) is a better metric** but still high because the baseline captures keyword patterns that created the labels.

### 2. Circular Label Assignment
Intents were discovered via TF-IDF clustering, and the baseline classifier uses TF-IDF. The classifier is essentially learning to reproduce the same TF-IDF patterns that created the labels. This creates artificial agreement — the classifier and the labels share the same information source. **This is the most fundamental limitation.**

### 3. Evaluation Set Size
200 golden examples across 6 classes means ~33 examples per class on average, with rare classes having fewer than 20. Confidence intervals at this sample size are wide (~±5-10%).

### 4. Social Media Artifact
Twitter's 280-character limit means messages are short and often lack the context needed for accurate classification. In a production support system with longer messages and conversation history, accuracy patterns would differ substantially.

### 5. No Distribution Shift
The evaluation set is sampled from the same time period and distribution as the training data. Real-world performance would degrade as user language, product features, and support policies evolve.

**Bottom Line:** The 96.4% headline number is a valid measurement of how well TF-IDF captures TF-IDF-derived patterns. It does NOT predict how well the system would perform on genuinely novel customer messages in production.

---

## 13. Limitations

1. **Single-label classification** cannot handle multi-intent messages
2. **No conversation state tracking** — each message is treated independently
3. **Gemini dependency** — system degrades without API access (falls back to escalation)
4. **Historical data is from 2017** — Spotify's product and policies have changed
5. **No evaluation of actual customer satisfaction** — metrics are proxy measures
6. **Golden set uses machine-suggested labels** — true human annotation is recommended
7. **LLM judge uses same model family** — potential self-reinforcing bias

---

## 14. What Would Be Done With One Additional Week

1. **True human annotation** of the 200-example golden set with multiple annotators
2. **Intent-filtered retrieval** — retrieve only from cases matching predicted intent
3. **Multi-turn context integration** — use conversation history in classification
4. **Class-balanced training** with oversampling or weighted loss
5. **A/B evaluation framework** comparing RAG responses against historical responses side-by-side
6. **Automated regression tests** that flag performance degradation on new data
7. **Containerization** (Docker) for reproducible deployment
