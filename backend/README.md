# Backend — Hiver AI Customer Support Agent

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Run API Server

```bash
python app/api/main.py
```

The API will start on `http://localhost:8000`.

## Run Tests

```bash
python -m pytest tests/ -v
```

## Training

```bash
python scripts/train.py
```

## Quick Evaluation

```bash
python scripts/run_quick_eval.py
```

## Project Structure

```
backend/
├── app/                  # Application source code
│   ├── api/             # FastAPI endpoints
│   ├── data/            # Data processing & validation
│   ├── escalation/      # Escalation decision engine
│   ├── evaluation/      # Metrics, baselines, LLM judge
│   ├── intent/          # Intent discovery & classification
│   ├── pipelines/       # Pandas & Spark data pipelines
│   ├── rag/             # RAG prompt & generator (Gemini)
│   ├── retrieval/       # Embeddings, FAISS index, retriever
│   └── utils/           # Shared utilities
├── tests/               # Unit & behavioral test suites
├── scripts/             # Training, evaluation, audit scripts
├── requirements.txt     # Python dependencies
└── Dockerfile           # Container build
```
