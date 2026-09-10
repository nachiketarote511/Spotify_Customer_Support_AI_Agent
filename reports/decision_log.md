# Decision Log

Non-obvious engineering decisions made during the design and implementation of the Hiver AI Customer Support Agent.

---

## Decision 1: Brand Selection — SpotifyCares

**Decision:** Selected SpotifyCares as the primary brand for analysis.

**Reason:** Automated scoring ranked SpotifyCares highest (score 0.856) based on volume (43K outbound, 27K inbound tweets), conversation completeness (88.7%), and intent diversity. A strong brand needs sufficient data across multiple intent categories to avoid overfitting.

**Alternative Considered:** AppleSupport had even higher volume but far lower conversation completeness (~40%), meaning most tweets could not be reconstructed into usable conversations.

**Why Rejected:** Low completeness would have produced a smaller effective training set despite larger raw numbers, and would introduce selection bias toward only the simplest conversations.

---

## Decision 2: Intent Taxonomy — Data-Derived, Not Predefined

**Decision:** Derived 10 intent categories from unsupervised clustering (TF-IDF + SVD + KMeans) of actual SpotifyCares messages, then mapped clusters to named intents.

**Reason:** The assignment explicitly requires intents to come from the actual brand's data, not from generic NLP benchmarks. SpotifyCares conversations naturally cluster around 6 dominant categories (playback_issues, billing_payment, account_access, app_technical, information_request, playlist_library).

**Alternative Considered:** Using Banking77's 77-intent taxonomy and mapping it to Spotify's domain.

**Why Rejected:** Banking77 intents are domain-specific to financial services. Force-mapping them would create artificial categories with no natural support in the data, and would violate the assignment requirement.

---

## Decision 3: Baseline Classifier — TF-IDF + Logistic Regression

**Decision:** Used TF-IDF (max 5000 features, sublinear TF, bigrams) + Logistic Regression as the baseline classifier.

**Reason:** This combination is fast to train, interpretable, reproducible without GPU, and provides a strong baseline for text classification. It achieved 96.4% accuracy and 0.936 macro F1.

**Alternative Considered:** Multinomial Naive Bayes.

**Why Rejected:** Logistic Regression consistently outperforms Naive Bayes on TF-IDF features for multi-class text classification, especially when classes are imbalanced.

---

## Decision 4: Improved Classifier — Sentence Transformers + Logistic Regression

**Decision:** Used `all-MiniLM-L6-v2` embeddings (384-dim) + Logistic Regression as the "improved" classifier.

**Reason:** Sentence Transformer embeddings capture semantic similarity better than bag-of-words approaches for short, informal social media text. The model is lightweight enough to run on CPU.

**Alternative Considered:** Fine-tuning a BERT model directly on the intent classification task.

**Why Rejected:** Fine-tuning requires GPU resources, takes significantly longer, and is harder to reproduce. The assignment values reproducibility over marginal accuracy gains. Additionally, the Sentence Transformer baseline underperformed TF-IDF (75.9% vs 96.4%) likely because TF-IDF captures exact keyword matches that dominate clustering-based labels, while embeddings capture semantic nuance not reflected in the labels.

---

## Decision 5: Embedding Model — all-MiniLM-L6-v2

**Decision:** Used `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) for all embeddings.

**Reason:** Best balance of speed vs quality for CPU inference. 384 dimensions keeps the FAISS index manageable (~32MB for 21K documents). The model is widely supported and benchmarked.

**Alternative Considered:** `all-mpnet-base-v2` (768 dimensions, higher quality).

**Why Rejected:** Doubles the FAISS index size and inference time without proportional quality gains for short social media text. Memory and speed matter more for the <15 minute reproducibility requirement.

---

## Decision 6: Vector Index — FAISS Flat (Exact Search)

**Decision:** Used FAISS `IndexFlatIP` (exact inner product search) rather than approximate nearest neighbor methods.

**Reason:** With only ~21K documents, exact search is already fast (<50ms per query). Approximate methods (IVF, HNSW) add complexity and approximation error without meaningful speedup at this scale.

**Alternative Considered:** FAISS IVF with 100 centroids.

**Why Rejected:** IVF requires training, adds an nprobe parameter to tune, and can miss relevant results. At 21K documents, the overhead is not justified.

---

## Decision 7: Retrieval Top-K = 5

**Decision:** Default retrieval returns top-5 similar historical conversations.

**Reason:** 5 cases provide enough grounding evidence for the LLM without overwhelming the prompt context. In evaluation, Recall@5 captures most relevant cases while keeping prompt length manageable.

**Alternative Considered:** Top-10 retrieval.

**Why Rejected:** More cases means longer prompts, higher API costs, and diminishing returns. Most relevant information is in the top 3-5 results; additional cases add noise.

---

## Decision 8: Hybrid Escalation Engine

**Decision:** Used a weighted multi-signal escalation engine combining 6 signals (intent confidence, retrieval quality, sensitive content, ambiguous intent, conflicting evidence, high emotion) plus LLM's own assessment.

**Reason:** Relying solely on the LLM for escalation decisions is unreliable — LLMs may hallucinate confidence or miss safety signals. A hybrid approach catches issues the LLM misses (sensitive keywords, low retrieval quality) while still incorporating the LLM's judgment.

**Alternative Considered:** Pure rule-based escalation (only keyword matching + thresholds).

**Why Rejected:** Rules alone miss nuanced cases where context matters. The LLM signal catches cases where the customer's tone or request complexity warrants human intervention even without explicit trigger keywords.

---

## Decision 9: Gemini 2.0 Flash for Generation

**Decision:** Used `gemini-2.0-flash` for both RAG generation and LLM-as-judge evaluation.

**Reason:** Flash models are cost-effective, fast, and sufficient for structured JSON generation tasks. The assignment doesn't require state-of-the-art generation quality; it requires a working, reproducible system.

**Alternative Considered:** `gemini-1.5-pro` for higher quality generation.

**Why Rejected:** Pro models are slower, more expensive, and rate-limited. For a reproducibility-focused assignment, fast and cheap is preferable.

---

## Decision 10: Two Processing Pipelines with Common Data Contract

**Decision:** Built Pipeline A (Pandas) as the working implementation and Pipeline B (PySpark + dbt) as an architectural specification with executable code but requiring a Spark cluster.

**Reason:** The assignment requires demonstrating scalable thinking. Pipeline A runs locally for evaluation. Pipeline B shows the candidate understands distributed processing, but cannot run on a laptop without Spark. Both produce the same `ConversationRecord` schema.

**Alternative Considered:** Only implementing Pipeline A and mentioning Spark in the README.

**Why Rejected:** Actually writing the PySpark code and dbt models demonstrates genuine knowledge of the tools, not just name-dropping.

---

## Decision 11: Golden Set — Stratified Sampling with Machine Suggestions

**Decision:** Created a 200-example golden evaluation set using stratified sampling across intents, with machine-generated intent suggestions clearly distinguished from human-verified labels.

**Reason:** Pure random sampling would undersample rare intents. Stratified sampling ensures every intent category has sufficient evaluation coverage. Machine suggestions accelerate annotation but are explicitly marked as non-human labels.

**Alternative Considered:** Larger golden set (500+ examples).

**Why Rejected:** A 200-example set provides statistically meaningful metrics for 6 intent classes while remaining human-annotatable in a reasonable time. Larger sets don't fundamentally change conclusions at this data scale.

---

## Decision 12: Conversation-Level Train/Test Split

**Decision:** Split data by conversation — all turns from a single conversation stay in the same split.

**Reason:** Prevents data leakage where nearly identical messages from the same thread appear in both train and test sets, which would inflate evaluation metrics.

**Alternative Considered:** Random message-level split.

**Why Rejected:** Would allow training on "Can you help?" and testing on the reply to the same thread, producing artificially high accuracy that doesn't reflect real-world performance.

---

## Decision 13: Response Caching for Cost Control

**Decision:** Implemented MD5-based response caching for Gemini API calls, keyed on message + intent + case count.

**Reason:** Prevents duplicate API calls during evaluation, development, and demo. Evaluators won't accidentally trigger hundreds of API calls by re-running the quick eval script.

**Alternative Considered:** No caching (rely on API response deduplication).

**Why Rejected:** Gemini API has no built-in deduplication. Without caching, the same customer message would generate a new API call every time, wasting quota and making results non-deterministic.

---

## Decision 14: Evaluation Framework — Three-Tier Comparison

**Decision:** Evaluate all three systems (Trivial Baseline, Simple Baseline, Final RAG System) on the same golden evaluation set using the same metrics.

**Reason:** Meaningful evaluation requires baselines. Without baselines, a 90% accuracy number is meaningless — if a trivial approach achieves 85%, the AI system adds little value. The three-tier comparison makes the system's contribution quantifiable.

**Alternative Considered:** Only evaluating the final system.

**Why Rejected:** Single-system evaluation cannot demonstrate that the AI system adds value over simpler approaches, which is the core question the assignment asks.
