# Architecture Overview

## System Architecture

```mermaid
graph TB
    subgraph "Frontend"
        UI["Chatbot UI<br/>index.html"]
    end

    subgraph "Backend API"
        API["FastAPI Server<br/>app/api/main.py"]
        IC["Intent Classifier<br/>app/intent/"]
        RET["FAISS Retriever<br/>app/retrieval/"]
        RAG["RAG Generator<br/>app/rag/ + Gemini"]
        ESC["Escalation Engine<br/>app/escalation/"]
    end

    subgraph "Data & Models"
        CFG["configs/"]
        DATA["data/"]
        MDL["models/"]
    end

    subgraph "Data Pipeline"
        PD["Pandas Pipeline"]
        SP["Spark Pipeline"]
        DBT["dbt Models"]
    end

    UI -->|HTTP POST| API
    API --> IC
    API --> RET
    API --> RAG
    API --> ESC
    IC --> MDL
    RET --> MDL
    RAG -->|Gemini API| GEMINI["Google Gemini"]
    ESC --> CFG
    PD --> DATA
    SP --> DATA
    DBT --> DATA
```

## Request Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Frontend
    participant API as FastAPI
    participant IC as Intent Classifier
    participant RET as FAISS Retriever
    participant RAG as RAG Generator
    participant GEM as Gemini API
    participant ESC as Escalation Engine

    User->>UI: Type message
    UI->>API: POST /generate-reply
    API->>IC: predict(message)
    IC-->>API: intent + confidence
    API->>RET: retrieve(message, intent)
    RET-->>API: top-k similar cases
    API->>RAG: generate(message, intent, cases)
    RAG->>GEM: Grounded prompt
    GEM-->>RAG: JSON response
    RAG-->>API: reply + confidence
    API->>ESC: decide(confidence, cases, message)
    ESC-->>API: AUTO_HANDLE or ESCALATE
    API-->>UI: Full response
    UI-->>User: Display chat bubble + details
```

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Intent Classifier | `backend/app/intent/` | TF-IDF + LogReg / Sentence Transformers |
| FAISS Retriever | `backend/app/retrieval/` | Semantic search over historical cases |
| RAG Generator | `backend/app/rag/` | Gemini-powered grounded response generation |
| Escalation Engine | `backend/app/escalation/` | Strict priority-based escalation decisions |
| Data Pipelines | `backend/app/pipelines/` | Pandas (small) and Spark (large) processing |
| Evaluation | `backend/app/evaluation/` | Metrics, baselines, LLM judge |
